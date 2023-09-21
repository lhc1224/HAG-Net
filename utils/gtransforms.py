# Adapted from: https://github.com/yjxiong/tsn-pytorch/blob/master/transforms.py

import torchvision
import random
from PIL import Image
import numbers
import torch
import torchvision.transforms.functional as F

class GroupResize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = torchvision.transforms.Resize(size, interpolation)
        
    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]

class GroupRandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_group):
       # print(img_group)
        w, h = img_group[0].size
        th, tw = self.size

        out_images = list()
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        for img in img_group:
            assert(img.size[0] == w and img.size[1] == h)
            if w == tw and h == th:
                out_images.append(img)

            else:
                out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))
        return out_images

class GroupCenterCrop(object):
    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]

class GroupRandomHorizontalFlip(object):
    def __call__(self, img_group):
        if random.random() < 0.5:
            img_group = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
        return img_group

class ToTensor(object):
    def __init__(self):
        self.worker = torchvision.transforms.ToTensor()

    def __call__(self, img_group):
        img_group = [self.worker(img) for img in img_group]
        out_img=[]
        for img in img_group:
            if img.size(0)!=3:
                img=img.repeat(3,1,1)
            out_img.append(img)

        return torch.stack(out_img, 0)

class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor): # (T, 3, 224, 224)

        for b in range(tensor.size(0)//2):
            for t, m, s in zip(tensor[b], self.mean, self.std):
                t.sub_(m).div_(s)

        return tensor

class ZeroPad(object):
    def __init__(self, max_len):
        self.max_len = max_len
    
    def __call__(self,tensor):
        #print(tensor.size())
        if (tensor.size(0)/2)>=self.max_len:
            return tensor
        num_f=tensor.size(0)//2
       # print(tensor.size(0))

        masks=tensor[num_f:]
        tensor=tensor[:num_f]
        n_pad = self.max_len - tensor.size(0)

        pad = torch.zeros(n_pad, tensor.size(1), tensor.size(2), tensor.size(3))
        tensor = torch.cat([tensor, pad], 0) # (T, 3, 224, 224)

        pad_mask=torch.zeros(n_pad,masks.size(1),masks.size(2),masks.size(3))
        masks=torch.cat([masks,pad_mask],0)

        out_tensor=torch.cat([tensor,masks],0)

        return out_tensor
