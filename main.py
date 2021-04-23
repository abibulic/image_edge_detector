import models
import predict
import argparse
import model_to_onnx


def get_args():

    parser = argparse.ArgumentParser(description='Edge detection')

    disc = ['D', 'E']
    #disc = ['Z', 'Y']
    parser.add_argument('--disc', type=list, default=disc,
                        help='D,E-local, Z,Y-remote')

    # model parameters
    parser.add_argument('--backbone_model', type=str, default='resnet34',
                        help='What type of backbone model to use')
    parser.add_argument('--pretrained', type=bool, default=True,
                        help='Use pretrain weights (resnet18 and resnet34 are supported)')
    
    # data parameters
    parser.add_argument('--n_channels', type=int, default=3, 
                        help='Number of channels')
    parser.add_argument('--input_img_size_x', type=int, default=512,
                        help='Input image size')
    parser.add_argument('--input_img_size_y', type=int, default=512,
                        help='Input image size')

    # training parameters
    parser.add_argument('--epochs', type=int, default=200, 
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=1, 
                        help='Batch size')
    parser.add_argument('--n_gpus', type=int, default=1,
                        help='Numbers of gpus used for training')
    parser.add_argument('--num_workers', type=int, default=4, 
                        help='Pre-fetching threads.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, 
                        help='Learning rate', dest='lr')  
    parser.add_argument('--weight_decay',  type=float, default=2e-4,
                        help='default weight decay')  
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--resume', type=str,
                        default='',
                        #default=f'{disc[1]}:/Weights/EdgeDetection/last.ckpt', 
                        help='Path to a checkpoint from witch to resume training')
    parser.add_argument('--valid_split', type=float, default=0.1, 
                        help='Percent of the data that is used as validation (0-1)')

    # paths
    parser.add_argument('--data_path', type=str, default=f'{disc[1]}:/LocalDataSources/NN_Datasets/EdgeDetection/exit/json/',
                        help='Dataset dir path')
    parser.add_argument('--weights_path', type=str, default=f'{disc[1]}:/Weights/EdgeDetection/', 
                        help='Weights dir path')

    # testing parameters
    parser.add_argument('--img_dir', type=str, 
                        #default=f'{disc[1]}:/OpenDataSources/EdgeDetection/BIPED/edges/imgs/test/rgbr/', 
                        #default=f'{disc[1]}:/LocalDataSources/EdgeDetection/tin_bath/data/',
                        #default=f'{disc[1]}:/LocalDataSources/EdgeDetection/exit/data1/',
                        #default=f'{disc[1]}:/LocalDataSources/EdgeDetection/4-10-2020/',
                        default=f'{disc[1]}:/LocalDataSources/TinBath/AGC_Richmond/Exit/JAN_2021/right/',
                        #default=f'{disc[1]}:/LocalDataSources/TinBath/AGC_Richmond/Spread/September_2020/right/',
                        help='Path to img dir for prediction')
    parser.add_argument('--out_dir', type=str, default='', 
                        help='Path where to save out images')
    parser.add_argument('--model', type=str, default=f'{disc[1]}:/Weights/EdgeDetection/trained_exit_resnet34_epoch=153-step=9085.ckpt',
                        help='Path to trained model')
    parser.add_argument('--create_vid', type=bool, default=False,
                        help='Create video')

    # onnx
    parser.add_argument('--create_onnx', type=bool, default=False,
                        help='Parse weights to onnx model')   
    parser.add_argument('--onnx_model', type=str, default=f'{disc[1]}:/Weights/EdgeDetection/trained_exit_resnet34_epoch=153-step=9085.ckpt',
                        help='Path to model that is gonna be parsed to onnx model')                

    return parser.parse_args(args=[])


def train(args):
    print(f'Backbone model: {args.backbone_model}')
    model = models.HED(args)

    profiler = models.pl.profiler.AdvancedProfiler()
    checkpoint_callback = models.ModelCheckpoint(filepath=args.weights_path, 
                                                save_last=True, 
                                                save_top_k=1, 
                                                monitor='val_loss', 
                                                mode='min', 
                                                save_weights_only=False)

    if args.resume:
        try:
            trainer = models.pl.Trainer(gpus=args.n_gpus, 
                                        resume_from_checkpoint=args.resume, 
                                        checkpoint_callback=checkpoint_callback,
                                        #accumulate_grad_batches=4,
                                        profiler=False)
        except:
            print('Checkpoint doesnt exist')
    else:
        trainer = models.pl.Trainer(gpus=args.n_gpus, 
                                    max_epochs=args.epochs,
                                    auto_lr_find=False,
                                    checkpoint_callback=checkpoint_callback,
                                    #accumulate_grad_batches=4,
                                    profiler=False)
  
    print('Start training...')
    trainer.fit(model)


def test(args):
    predict.predict(args)

def onnx(args):
    model_to_onnx.model_to_onnx(args)

def compare_F1_results(atgs):
    import cv2
    import numpy as np
    import matplotlib.pyplot  as plt
    from sklearn.metrics import f1_score
    
    threshold = 0.5

    files_BIPED_res1 = models.list_files(f'{args.disc[0]}:/aleks/PROJECTS/edge_detector/temp_res/', ext='.png')
    files_BIPED_res2 = models.list_files(f'{args.disc[0]}:/aleks/PROJECTS/DexiNed/results/edges/DexiNed_BIPED2CLASSIC/pred-a/', ext='.png')
    files_BIPED_gt = models.list_files(f'{args.disc[1]}:/OpenDataSources/EdgeDetection/BIPED/edges/edge_maps/test/rgbr/', ext='.png')
    sum_res1 = 0
    sum_res2 = 0
    for i in range(len(files_BIPED_gt)):
        gt = cv2.imread(files_BIPED_gt[i])
        gt = gt.astype(np.float32)
        gt /= 255

        res1 = cv2.imread(files_BIPED_res1[i])
        res1 = res1.astype(np.float32)
        res1 /= 255

        res2 = cv2.imread(files_BIPED_res2[i])
        res2 = res2.astype(np.float32)
        res2 /= 255
        res2_temp = np.ones(gt.shape)
        res2_temp -= res2
        res2 = res2_temp
        
        res1 = np.transpose(res1, (2, 0, 1))
        res2 = np.transpose(res2, (2, 0, 1))
        gt = np.transpose(gt, (2, 0, 1))

        res1 = res1[0]
        res2 = res2[0]
        gt = gt[0]

        res1[res1 < threshold] = 0
        res1[res1 >= threshold] = 1

        res2[res2 < threshold] = 0
        res2[res2 >= threshold] = 1

        # res2 += gt
        # res2[res2 < 0.5] = 0
        # res2[res2 == 1] = 255
        # res2[res2 == 2] = 128
        # res2 = res2.astype(np.uint8)
        # cv2.imwrite('test.png', res2)

        # plt.imshow(np.concatenate((gt, res1, res2), axis=1), cmap='gray')
        # plt.show()

        res1 = res1.flatten()
        res2 = res2.flatten()
        gt = gt.flatten()

        # res2 = res2[np.newaxis, :]
        # gt = gt[np.newaxis, :]

        # res2 = torch.from_numpy(res2)
        # gt = torch.from_numpy(gt)

        sum_res1 += f1_score(gt, res1, average='micro', zero_division='warn')
        sum_res2 += f1_score(gt, res2, average='micro', zero_division='warn')
        # sum_res += self.calc_F1(res2,gt,0.5)

    print(sum_res1/(i+1))
    print(sum_res2/(i+1))

if __name__ == "__main__":
    args = get_args()

    #compare_F1_results(args)

    #train(args)

    test(args)

    if args.create_onnx:
        onnx(args)