import os
import cv2
import time
import torch
import argparse
import numpy as np
from matplotlib import pyplot as plt

import models
from dataset import EdgeDataset
#from helpers import masks_to_colorimg

def show_image(winName, img, scale):
    img_res = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    cv2.imshow(winName, img_res)
    
def list_files(root_dir, ext='.jpg'):
    file_list = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(ext):
                file_list.append(os.path.join(root, file).replace("\\","/"))
    return file_list

def calc_F1(pred, target, threshold):
    pred[pred < threshold] = 0
    pred[pred >= threshold] = 1

    tp = (target * pred).sum().to(torch.float32)
    tn = ((1 - target) * (1 - pred)).sum().to(torch.float32)
    fp = ((1 - target) * pred).sum().to(torch.float32)
    fn = (target * (1 - pred)).sum().to(torch.float32)

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    f1 = 2* (precision*recall) / (precision + recall + epsilon)

    return f1

def find_mean_a_b(linesP):
    line_a = 0
    line_b = 0

    for i in range(len(linesP)):
        l = linesP[i][0]
        temp = (l[3] - l[1])/(l[2] - l[0])

        if abs(temp) < 1e-7:
            line_a += 1e-7
        else:
            line_a += temp
        
        temp = temp*(-l[0]) + l[1]
        line_b += temp

    line_a /= len(linesP)
    line_b /= len(linesP)

    return line_a, line_b


def parse_results(res):
    m = []
    n = []

    line1_a = 0
    line1_b = 0
    line2_a = 0
    line2_b = 0

    glass_line = None
    trim_line = None

    count = 1

    temp_res = res.copy()
    mask = np.zeros((res.shape[0]+2, res.shape[1]+2), np.uint8)

    for row in range(res.shape[0]):
        for col in range(res.shape[1]):
            if temp_res[row][col] == 255:
                m.append(row)
                n.append(col)
                cv2.floodFill(temp_res, None, (col, row), count)
                count += 1

    if len(m) == 0:
        return None, None

    histSize = 256
    hist = np.histogram(temp_res, 256, [1,256])[0]

    max1 = 0
    max2 = 0
    max1_idx = -1
    max2_idx = -1

    for i in range(histSize):
        if hist[i-1] >= max1:
            max2 = max1
            max2_idx = max1_idx
            max1 = hist[i-1]
            max1_idx = i - 1

        if hist[i-1] < max1 and hist[i-1] > max2:
            max2 = hist[i-1]
            max2_idx = i - 1

    temp_res2 = temp_res.copy()

    for i in range(count-1):
        if i != max1_idx and i != max2_idx:
            cv2.floodFill(temp_res, None, (n[i], m[i]), 0)
            cv2.floodFill(temp_res2, None, (n[i], m[i]), 0)

    if max2/max1 > 0.5:
        cv2.floodFill(temp_res, None, (n[max1_idx], m[max1_idx]), 255)
        cv2.floodFill(temp_res, None, (n[max2_idx], m[max2_idx]), 0)
        linesP1 = cv2.HoughLinesP(temp_res, 1, np.pi/180, 50, 50, 10)
        
        cv2.floodFill(temp_res2, None, (n[max1_idx], m[max1_idx]), 0)
        cv2.floodFill(temp_res2, None, (n[max2_idx], m[max2_idx]), 255)
        linesP2 = cv2.HoughLinesP(temp_res2, 1, np.pi/180, 50, 50, 10) 

        if linesP1 is None or linesP2 is None:
            return None, None
        else:
            line1_a, line1_b = find_mean_a_b(linesP1)
            line2_a, line2_b = find_mean_a_b(linesP2)

            if line1_b > line2_b:
                glass_line = [line1_a, line1_b]
                trim_line = [line2_a, line2_b]
            else:
                glass_line = [line2_a, line2_b]
                trim_line = [line1_a, line1_b]
    else:
        cv2.floodFill(temp_res, None, (n[max1_idx], m[max1_idx]), 255)
        cv2.floodFill(temp_res, None, (n[max2_idx], m[max2_idx]), 0)
        linesP1 = cv2.HoughLinesP(temp_res, 1, np.pi/180, 50, 50, 10)
        
        if linesP1 is None:
            return None, None
        else:
            line1_a, line1_b = find_mean_a_b(linesP1)
            glass_line = [line1_a, line1_b]      
    
    return glass_line, trim_line

    


def predict(args):

    if args.create_vid:
        video = cv2.VideoWriter('demo.mp4', 0, 15, (512, 3584))  

    print(f'Backbone model: {args.backbone_model}')
    model = models.HED(args)
    checkpoint = torch.load(args.model)

    if 'vgg' in args.backbone_model and args.model.split('/')[-1][0] == 'c':
        temp_dict = checkpoint['state_dict'].copy()
        for i, s in enumerate(temp_dict):
            temp = 'model.'+s
            checkpoint['state_dict'][temp] = checkpoint['state_dict'].pop(s)

    model.load_state_dict(checkpoint['state_dict'])
    model.cuda()
    model.eval()
     
    imgPaths =list_files(args.img_dir)

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    data_test = EdgeDataset(imgPaths, args, transforms=False)
    test_loader = torch.utils.data.DataLoader(data_test,
                                               batch_size=1, shuffle=False,
                                               num_workers=0, pin_memory=False)

    score = []

    #f = open("images_for_label.txt", 'a')

    for inputs, targets, _, name in test_loader:
        imp = name[0]
        
        # img_orig = cv2.imread(imp)

        _, _, H, W = inputs.shape

        start_time = time.time()
        pred = model(inputs.cuda())

        score.append(calc_F1(pred[-1].detach().cpu(), targets, 0.5))
        orig = cv2.imread(name[0])
        orig = cv2.resize(orig, (args.input_img_size_x, args.input_img_size_y))
        orig = orig[:,:,0]

        results_all = []
        for i in range(len(pred)):
            results_all.append(torch.squeeze(pred[i].detach()).cpu().numpy())
        #results_all.append(torch.squeeze(pred[-1].detach()).cpu().numpy())
        results_all.append(torch.squeeze(targets.detach()).cpu().numpy())
    
        results_all = np.array(results_all).transpose(1,0,2).reshape(H,W*len(results_all))
        
        results_all = torch.squeeze(pred[i].detach()).cpu().numpy()

        results_all[results_all < 0.5] = 0
        results_all[results_all >= 0.5] = 255
        #results_all *= 255
        results_all = results_all.astype(np.uint8) 

        # glass_line, trim_line = parse_results(results_all)
        # if glass_line == None or trim_line == None:
        #     f.write(imp+'\n')

        #results_all = np.concatenate((orig, results_all), axis=1)
        results_all = cv2.addWeighted(orig, 0.8, results_all, 0.2, 0.0)
        results_all = np.concatenate((results_all, orig), axis=1)
        # results_all = cv2.resize(results_all, (0, 0), fx=0.5, fy=0.5)
        w_name = f'{args.disc[0]}:/aleks/PROJECTS/edge_detector/temp_res/' + name[0].split('/')[-1]
        w_name = w_name[:-4] + '.png'
        #cv2.imwrite(w_name, results_all)

        show_image('Test', results_all, 1)
        cv2.waitKey(1)
        if args.create_vid:
            video.write(results_all) 

        # plt.imshow(results_all, cmap='gray', vmin=0, vmax=255)
        # plt.show()


    #f.close()
    score = sum(score)/len(test_loader)
    print(f'Mean F1 score: {score}')

    if args.create_vid:
        video.release()