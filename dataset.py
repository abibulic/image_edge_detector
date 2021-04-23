import sys

import torch
from torch.utils.data.dataset import Dataset 
from torchvision import transforms

import cv2
import json
import base64
import os.path
import PIL.Image
import numpy as np
from skimage.util import img_as_float
from skimage.color import rgb2gray, gray2rgb
from skimage import feature, color, transform, io

import matplotlib.pyplot  as plt
from albumentate import augment

class EdgeDataset(Dataset):
    def __init__(self, imgPaths, args, transforms=False):
        self.imgPaths = imgPaths
        self.args = args
        self.transform = transforms

        self.mean = 3*[0.4151235818862915]
        self.std = 3*[0.14493609964847565]

        # calculate mean and stddev
        # mean = []
        # stddev = []
        # for path in  self.imgPaths:
        #     im = cv2.imread(path)
        #     mean.append(np.mean(im[:,:,0]))
        #     stddev.append(np.std(im[:,:,0]))
        # self.mean = np.mean(mean)
        # self.std =np.mean(stddev)

    def normalize_image(self, im):
        im -= np.array((104.00698793,116.66876762,122.67891434))
        return im

    def __getitem__(self, index):
        img_path = self.imgPaths[index]

        if img_path[-4:] == 'json' and '/tin_bath/' in img_path:
            data = json.load(open(img_path))
            height = data.get('imageHeight')
            width = data.get('imageWidth')
            img_data = data.get('imageData')
            points = data.get('points')
            jpg_original = base64.b64decode(img_data[2:-1])
            jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
            im = cv2.imdecode(jpg_as_np, flags=1)

            height = im.shape[0]
            width = im.shape[1]

            im = cv2.resize(im, (self.args.input_img_size_y, self.args.input_img_size_x))
            im = im.astype(np.float32)
            im = self.normalize_image(im)

            mask = np.zeros((self.args.input_img_size_y, self.args.input_img_size_x), dtype=np.float32)
            for i in range(len(points[:-1])):
                p1 = points[i].copy()
                p2 = points[i+1].copy()
                p1[0] = int(p1[0]/width*self.args.input_img_size_x)
                p1[1] = int(p1[1]/height*self.args.input_img_size_y)

                p2[0] = int(p2[0]/width*self.args.input_img_size_x)
                p2[1] = int(p2[1]/height*self.args.input_img_size_y)

                cv2.line(mask, tuple(p1), tuple(p2), 1., 2, -1)

        elif img_path[-4:] == 'json' and '/exit' in img_path:
            data = json.load(open(img_path))
            height = data.get('imageHeight')
            width = data.get('imageWidth')
            img_data = data.get('imageData')
            points = data.get('points')
            jpg_original = base64.b64decode(img_data[2:-1])
            jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
            im = cv2.imdecode(jpg_as_np, flags=1)

            height = im.shape[0]
            width = im.shape[1]

            im = cv2.resize(im, (self.args.input_img_size_y, self.args.input_img_size_x))
            im = im.astype(np.float32)
            im = self.normalize_image(im)

            mask = np.zeros((self.args.input_img_size_y, self.args.input_img_size_x), dtype=np.float32)
            for i in range(len(points)):
                p1 = points[i][0].copy()
                p2 = points[i][1].copy()
                p1[0] = int(p1[0]/width*self.args.input_img_size_x)
                p1[1] = int(p1[1]/height*self.args.input_img_size_y)

                p2[0] = int(p2[0]/width*self.args.input_img_size_x)
                p2[1] = int(p2[1]/height*self.args.input_img_size_y)

                cv2.line(mask, tuple(p1), tuple(p2), 1., 2, -1)

        else:
            img_path = self.imgPaths[index].split(' ')
            mask_path = []
            if len(img_path) > 1:
                mask_path = img_path[1][:-1]
            img_path = img_path[0]
            
            im =  np.array(cv2.imread(img_path), dtype=np.float32)
            im = self.normalize_image(im)

            # if self.args.n_channels == 1:
            #     im = rgb2gray(im)
            # else:
            #     im = gray2rgb(im)
            #     self.mean = 3*[0.4151235818862915]
            #     self.std = 3*[0.14493609964847565]

            # im = img_as_float(im)
            # im = im.astype(np.float32)

            # plt.imshow(im, cmap='gray')
            # plt.show()
            if 'tin_bath' in img_path:
                mask_path = img_path[:-4]+'_maks.jpg'
                mask_path = mask_path.replace('/tin_bath/data/', '/tin_bath/masks/')

            elif 'data1' in img_path:
                mask_path = img_path[:-4]+'_maks.jpg'
                mask_path = mask_path.replace('/exit/data1/', '/exit/masks/')
            
            elif 'rgbr' in img_path:
                mask_path = img_path[:-4]+'.png'
                mask_path = mask_path.replace('/imgs/t', '/edge_maps/t')

            elif not(mask_path):
                mask_path = img_path[:-4]+'.png'
                mask_path = mask_path.replace('/nn_edges/data/', '/nn_edges/masks/')

            if os.path.isfile(mask_path):
                mask = np.array(cv2.imread(mask_path), dtype=np.float32)
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            else:
                print('Failed to load mask')
                mask = np.zeros((im.shape[0], im.shape[1]), dtype=np.float32)

            # mask[mask <= 0.0] = 0.0
            # mask[mask > 0.0] = 1.0
            
            # max_val = mask.max()
            # if max_val > 128 or max_val == 0:
            #     max_val = 128
            # mask[mask < max_val] = 0.0
            # mask[mask >= max_val] = 1.0
            
            mask[mask <= 10] = 0.0
            mask[mask > 10] = 1.0

            # mask[mask <= 127] = 0.0
            # mask[mask > 127] = 1.0

        data = {"image": im, "mask": mask}
        data = augment(self.transform, self.args, data)

        im = data['image']
        mask = data['mask']

        if np.isnan(im).any() or np.isnan(mask).any():
            sys.exit('Encountered Nan in image')

        gaussian_blure_mask = cv2.GaussianBlur(mask, (int(self.args.input_img_size_x/4-1), int(self.args.input_img_size_y/4-1)), 10)
        gaussian_blure_mask = (gaussian_blure_mask-gaussian_blure_mask.min())/(gaussian_blure_mask.max()-gaussian_blure_mask.min())

        # plt.imshow(np.concatenate((im[:,:,0], mask, gaussian_blure_mask), axis=1))
        # plt.imshow(np.concatenate((mask, gaussian_blure_mask), axis=1))
        # plt.show()

        mask = torch.from_numpy(mask[np.newaxis, :, :])
        gaussian_blure_mask = torch.from_numpy(gaussian_blure_mask[np.newaxis, :, :])

        im = np.transpose(im, (2, 0, 1))
        im = torch.from_numpy(im)
        #im = transforms.functional.normalize(im, mean=self.mean, std=self.std)

        return im, mask.to(torch.float32), gaussian_blure_mask.to(torch.float32), img_path

    def __len__(self):
        return len(self.imgPaths)