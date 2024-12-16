import os
import time
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from argparse import ArgumentParser
# user
from builders.model_builder import build_model
from builders.dataset_builder import build_dataset_test
from utils.utils import save_predict
from utils.metric.metric import get_iou
from utils.convert_state import convert_state_dict

from dataset.event.base_trainer import BaseTrainer
from spikingjelly.activation_based import layer, neuron, functional
from tqdm import tqdm
from torch.nn import functional as F
from PIL import Image
from torchvision.utils import save_image



def parse_args():
    parser = ArgumentParser(description='Efficient semantic segmentation')
    parser.add_argument('--model', default="LETNet", help="model name: (default ENet)")
    parser.add_argument('--dataset', default="DDD17_events", help="dataset: DDD17_events or DSEC_events")
    parser.add_argument('--input_size', type=str, default="200,346", help="DDD17_events:200,346,DSEC_events:480,640")
    parser.add_argument('--dataset_path', type=str, default="/home/ubuntu/share_container/Datasets/zhuxx/DDD17_events")
    parser.add_argument('--workers', type=int, default=4, help="the number of parallel threads")
    parser.add_argument('--batch_size', type=int, default=2,
                        help=" the batch_size is set to 1 when evaluating or testing")
    parser.add_argument('--checkpoint', type=str,default="/home/ubuntu/code/LETNet_snn/Network/checkpoint/DDD17_events/SDSA1_2/model_best_miou.pth",
                        help="use the file to load the checkpoint for evaluating or testing ")
    parser.add_argument('--save_seg_dir', type=str, default="./result/",
                        help="saving path of prediction result")
    parser.add_argument('--save', type=bool, default=True, help="Save the predicted image")
    parser.add_argument('--cuda', default=True, help="run on CPU or GPU")
    parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")

    parser.add_argument('--split', type=str, default="train", help="spilt in ['train', 'test', 'valid']")
    parser.add_argument('--nr_events_data', type=int, default=1)
    parser.add_argument('--delta_t_per_data', type=int, default=50)
    parser.add_argument('--nr_events_window', type=int, default=32000, help='DDD17:32000,DSEC:100000')
    parser.add_argument('--data_augmentation_train', type=bool, default=True)
    parser.add_argument('--event_representation', type=str, default="voxel_grid")
    parser.add_argument('--nr_temporal_bins', type=int, default=5)
    parser.add_argument('--require_paired_data_train', type=bool, default=False)
    parser.add_argument('--require_paired_data_val', type=bool, default=True)
    parser.add_argument('--separate_pol', type=bool, default=False)
    parser.add_argument('--normalize_event', type=bool, default=True)
    parser.add_argument('--fixed_duration', type=bool, default=True)

    # event datasets
    parser.add_argument('--use_ohem', type=bool, default=True, help='OhemCrossEntropy2d Loss for event dataset')
    parser.add_argument("--use_earlyloss", type=bool, default=True, help='Use early-surpervised training for event dataset')
    parser.add_argument("--balance_weights", type=list, default=[1.0, 0.4], help='balance between out and early_out')

    args = parser.parse_args()

    return args




def test(args, test_loader, model, save=True):
    """
    args:
      test_loader: loaded for test dataset
      model: model
    return: class IoU and mean IoU
    """
    # evaluation or test mode
    model.eval()                          
    total_batches = len(test_loader)

    data_list = []
    for i, (input, image, label, _) in enumerate(test_loader):
        with torch.no_grad():
            input_var = input.cuda()
            
        start_time = time.time()
        output = model(input_var)
        torch.cuda.synchronize()
        time_taken = time.time() - start_time
        print('[%d/%d]  time: %.2f' % (i + 1, total_batches, time_taken))

        # save the predicted image
        if save:
            save_predict(output, label, image, i, args.dataset, args.save_seg_dir,
                         output_grey=False, output_color=True, gt_color=True)

        output = output.cpu().data[0].numpy()
        gt = np.asarray(label[0].numpy(), dtype=np.uint8)
        output = output.transpose(1, 2, 0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
        data_list.append([gt.flatten(), output.flatten()])

    meanIoU, per_class_iu, acc = get_iou(data_list, args.classes)
    return meanIoU, per_class_iu


def test_no_val(args, test_loader, model, save_path, save=True):
    if args.dataset == 'DDD17_events':
        # DDD17:['flat','background','object','vegetation','human','vehicle']
        color_map = [(128, 64,128), #紫
                    (70 , 70, 70), #灰
                    (220,220,  0), #黄
                    (107,142, 35), #绿
                    (220, 20, 60), #红
                    (0  ,  0,142)] #蓝
    elif args.dataset == 'DSEC_events':
        # DSEC:['background','building','fence','person','pole','road','sidewalk','vegetation','car','wall','traffic sign']
        color_map = [(0,  0,  0),
                    (70 ,70, 70),
                    (190,153,153),
                    (220, 20,60),
                    (153,153,153),
                    (128, 64,128),
                    (244, 35,232),
                    (107,142, 35),
                    (0,  0,  142),
                    (102,102,156),
                    (220,220,  0)]
    
    for index, batch in enumerate(tqdm(test_loader)):
        event, image, label, _, _ = batch
        size = label.size()
        event = event.cuda()
        
        pred = model(event)

        if pred.size()[-2] != size[0] or pred.size()[-1] != size[1]:
            pred = F.interpolate(
                pred, size[-2:],
                mode='bilinear', align_corners=True
            )
        pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
        label = label.squeeze(0).cpu().numpy()
        
        if save:
            sv_predict = np.zeros((size[-2], size[-1], 3), dtype=np.uint8)
            sv_label = np.zeros((size[-2], size[-1], 3), dtype=np.uint8)
            
            sv_path_pred = os.path.join(save_path,'pred')
            sv_path_label = os.path.join(save_path,'label')
            sv_path_image = os.path.join(save_path,'image')
            
            if not os.path.exists(sv_path_pred):
                os.mkdir(sv_path_pred)
            if not os.path.exists(sv_path_label):
                os.mkdir(sv_path_label)
            if not os.path.exists(sv_path_image):
                os.mkdir(sv_path_image)
            
            # 验证全部valid数据集
            if len(pred.shape) != 3:
                pred = pred[np.newaxis, :, :]
                label = label[np.newaxis, :, :]
            
            for idx in range(pred.shape[0]):
                for i, color in enumerate(color_map):
                    for j in range(3):
                        sv_predict[:,:,j][pred[idx]==i] = color_map[i][j]
                        sv_label[:,:,j][label[idx]==i] = color_map[i][j]
                
                sv_predict_event = Image.fromarray(sv_predict)
                sv_label_event = Image.fromarray(sv_label)
                sv_predict_event.save(sv_path_pred+"/predict{}.png".format(index * pred.shape[0] + idx))
                sv_label_event.save(sv_path_label+"/label{}.png".format(index * pred.shape[0] + idx))
                save_image(image[idx], sv_path_image+'/image{}.png'.format(index * pred.shape[0] + idx))



def test_model(args):
    """
     main function for testing
     param args: global arguments
     return: None
    """
    print(args)

    if args.cuda:
        print("=====> use gpu id: '{}'".format(args.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        if not torch.cuda.is_available():
            raise Exception("no GPU found or wrong gpu id, please run without --cuda")

    # build the model
    model = build_model(args.model, num_classes=args.classes, ohem=args.use_ohem, early_loss=args.use_earlyloss)
    functional.set_step_mode(model, step_mode='m')

    if args.cuda:
        model = model.cuda()  # using GPU for inference
        cudnn.benchmark = True

    if args.save:
        if not os.path.exists(args.save_seg_dir):
            os.makedirs(args.save_seg_dir)

    # load the test set
    # datas, testLoader = build_dataset_test(args.dataset, args.num_workers)

    # DDD17/DSEC datasets
    base_trainer_instance = BaseTrainer()
    trainLoader, testLoader = base_trainer_instance.createDataLoaders(args)

    
    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            print("=====> loading checkpoint '{}'".format(args.checkpoint))
            checkpoint = torch.load(args.checkpoint)
            model.load_state_dict(checkpoint['model'])
            # model.load_state_dict(convert_state_dict(checkpoint['model']))
        else:
            print("=====> no checkpoint found at '{}'".format(args.checkpoint))
            raise FileNotFoundError("no checkpoint found at '{}'".format(args.checkpoint))

    print("=====> beginning validation")
    print("validation set length: ", len(testLoader))
    # test_no_val(args, testLoader, model, args.save_seg_dir, save=False)
    test(args, testLoader, model, save=False)
    
    # # 需要计算miou值
    # mIOU_val, per_class_iu = test(args, testLoader, model, args.save_seg_dir, save=True)
    # print("mIOU_val:",mIOU_val)
    # print("per_class_iu:",per_class_iu)

    # # Save the result
    # args.logFile = 'test.txt'
    # logFileLoc = os.path.join(os.path.dirname(args.checkpoint), args.logFile)

    # # Save the result
    # if os.path.isfile(logFileLoc):
    #     logger = open(logFileLoc, 'a')
    # else:
    #     logger = open(logFileLoc, 'w')
    #     logger.write("Mean IoU: %.4f" % mIOU_val)
    #     logger.write("\nPer class IoU: ")
    #     for i in range(len(per_class_iu)):
    #         logger.write("%.4f\t" % per_class_iu[i])
    # logger.flush()
    # logger.close()


if __name__ == '__main__':

    args = parse_args()

    args.save_seg_dir = os.path.join(args.save_seg_dir, args.dataset, args.model)

    if args.dataset == 'cityscapes':
        args.classes = 19
    elif args.dataset == 'camvid':
        args.classes = 11
    elif args.dataset == 'DDD17_events':
        args.classes = 6
    elif args.dataset == 'DSEC_events':
        args.classes = 11
    else:
        raise NotImplementedError(
            "This repository now supports two datasets: cityscapes and camvid, %s is not included" % args.dataset)

    test_model(args)
