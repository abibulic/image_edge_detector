import os
import cv2
import numpy as np

def list_files(root_dir, ext='.jpg'):
    file_list = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(ext):
                file_list.append(os.path.join(root, file).replace("\\","/"))
    return file_list

def main():
    video = cv2.VideoWriter('D:/WORK/PROJECTS/exit/neural_network/temp_vide/demo.mp4', 0, 15, (1904, 2*192))  

    left_list = list_files('D:/WORK/PROJECTS/exit/neural_network/temp_vide/LEFT/')
    right_list = list_files('D:/WORK/PROJECTS/exit/neural_network/temp_vide/RIGHT/')

    for i in range(len(left_list)):
        img_left = cv2.imread(left_list[i])
        img_right = cv2.imread(right_list[i])
        res = np.concatenate((img_left, img_right), axis=0)

        video.write(res) 

    video.release()
           

    return


if __name__ == '__main__':
    main()