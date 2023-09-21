import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import random
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
#from Comput_heatmap import read_box,comput_heatmap
import cv2
import random
dir_path="opra_hand.txt"

def ReadTxtName(rootdir):
    lines = []
    with open(rootdir, 'r') as file_to_read:
        while True:
            line = file_to_read.readline()
            if not line:
                break
            line = line.strip('\n')
            lines.append(line)
    return lines
def pad_to_square(img, pad_value):
    h, w, _ = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = ((pad1, pad2), (0,0), (0,0)) if h <= w else ((0,0), (pad1, pad2), (0,0))  # 分别对应h,w,c的padding
    # Add padding
    img = np.pad(img, pad, 'constant', constant_values=pad_value)

    return img, (*pad[1], *pad[0])  # 
def comput_heatmap(root_dir):
    lines=ReadTxtName(root_dir)
    dict_list={}
    for line in lines:
        data=line.split(" ")
        img_path=data[0]
        ll=len(data[1:])
        if ll==0:
            dict_list[img_path]=[0]
        else:
            hand=[]
            hand_data=data[1:]
            num_hand=ll//4
            for i in range(num_hand):
                x1=float(hand_data[i*4])
                y1=float(hand_data[i*4+1])
                w=float(hand_data[i*4+2])
                h=float(hand_data[i*4+3])
                hand.append([x1,y1,w,h])
            dict_list[img_path]=hand
    return dict_list
def read_box(img_path,dict_list,normalized_labels=True,
             ker_size=3,ratio=0.2,i_ratio=0.2,w_h_max=448,root_path=None):

    img_path_1=root_path+img_path
    
    img=cv2.imread(img_path_1)
    img = np.array(img)
    
    if len(img.shape) != 3:

        img = img[None, :, :]
        img = img.repeat(3, 0)
    heatmap = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)

    hand_data=dict_list[img_path]

    if hand_data!=[0]:
        for hand in hand_data:
            h, w, _ = img.shape  

            h_factor, w_factor = (h, w) if normalized_labels else (1, 1)
            img, pad = pad_to_square(img, 0)
            x1 = w_factor * ((hand[0]*2 - hand[2])/2)
            y1 = h_factor * ((hand[1]*2 - hand[3])/2)
            x2 = w_factor * ((hand[2] + hand[0]*2)/2)
            y2 = h_factor * ((hand[3] + hand[1]*2)/2)
            # Adjust for added padding
            x1 += pad[0]  # 
            y1 += pad[2]
            x2 += pad[0]
            y2 +=pad [2]
            m=int((x2-x1)*ratio)
            n=int((y2-y1)*ratio)
            m_2=int((x2-x1)*i_ratio)
            n_2=int((y2-y1)*i_ratio)
            for rr in range(max(int(x1)-m,0),min(int(x2)+m,h_factor-1)):
                for cc in range(max(int(y1)-n,0),min(int(y2)+n,w_factor-1)):
                    try:
                        heatmap[cc,rr]= 1.0
                    except:
                        col=max(min(0,cc),w_factor-1)
                        row=max(min(0,rr),h_factor-1)
                        heatmap[col,row]=1.0
            for rr in range(max(int(x1)+m_2,0),min(int(x2)-m_2,h_factor-1)):
                for cc in range(max(int(y1)+n_2,0),min(int(y2)-n_2,w_factor-1)):
                    try:
                        heatmap[cc,rr]= 0.0
                    except:
                        col=max(min(0,cc),w_factor-1)
                        row=max(min(0,rr),h_factor-1)
                        heatmap[col,row]=0.0
        heatmap=cv2.GaussianBlur(heatmap,(55,55),0)
    if np.max(heatmap)>0:
        heatmap/=np.max(heatmap)
    return heatmap*255.0
