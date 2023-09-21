import collections
import os
from PIL import Image
import tqdm
import numpy as np
import json
import torch

from utils import util
from data.hotspot_dataset import VideoInteractions, HeatmapDataset
from data.hotspot_dataset import generate_heatmaps
from data.comput_heatmap import read_box, comput_heatmap


class OPRAInteractions(VideoInteractions):

    def __init__(self, root, split, max_len, sample_rate=1, label_path=None,ratio=0.3,i_ratio=0.2,w_h_max=448):
        super().__init__(root, split, max_len, sample_rate) 
      
        self.ratio=ratio
        self.w_h_max=w_h_max
        self.i_ratio=i_ratio
        annots = json.load(open('data/opra/annotations.json'))
        self.verbs, self.nouns = annots['verbs'], annots['nouns']
        self.train_data, self.val_data = annots['train_clips'], annots['test_clips']
        self.data = self.train_data if self.split == 'train' else self.val_data
        #self.data=self.train_data+self.val_data
        label_path = "opra_hand.txt"
        self.dict_list = comput_heatmap(label_path)
        # Remove instances that have not been downloaded
        data = []
        for entry in tqdm.tqdm(self.data, total=len(self.data)):
            if not os.path.exists(self.root + '/data/frames_16/%s/%s/%s/%s_%s.mp4' % tuple(entry['clip'])):
               
                continue
            data.append(entry)
        print('Removing %s missing instances' % (len(self.data) - len(data)))
        self.data = data

        # Use every frame. For OPRA sample_rate = 1 --> 5fps
        for entry in self.data:
            entry['frames']=[]
            clip_path = self.root + '/data/frames_16/%s/%s/%s/%s_%s.mp4' % tuple(entry['clip'])
            for f_id in range(entry['nframes']):
                frame_path = clip_path + '/image-%08d.jpg' % f_id
                if os.path.exists(frame_path):
                    entry['frames'].append((entry['clip'], f_id))
            entry['nframes']=len(entry['frames'])
        data=[]
        for entry in tqdm.tqdm(self.data, total=len(self.data)):
            if len(entry['frames'])>0:
                data.append(entry)
        print('Removing %s missing instances' % (len(self.data) - len(data)))
        self.data = data

        #print('Train data: %d | Val data: %d' % (len(self.train_data), len(self.val_data)))

        verbs = [entry['verb'] for entry in self.data]
        
        print('# actions: %d' % (len(set(verbs))))
        print('action distribution:',
              sorted(collections.Counter([self.verbs[v] for v in verbs]).items(), key=lambda x: -x[1]))

    def load_frame(self, frame):
        v_id, f_id = frame
        clip_path = self.root + '/data/frames_16/%s/%s/%s/%s_%s.mp4' % tuple(v_id)
        frame_path = clip_path + '/image-%08d.jpg' % f_id
        frame = util.load_img(frame_path)
        return frame

    def load_path(self,frames):
        v_id,f_id=frames
        clip_path = self.root + '/data/frames_16/%s/%s/%s/%s_%s.mp4' % tuple(v_id)
        frame_path = clip_path + '/image-%08d.jpg' % f_id
        return frame_path


    def load_box_mask(self, frame):

        v_id, f_id = frame
        img_path='/%s/%s/%s/%s_%s.mp4' % tuple(v_id)
        
        img_path = img_path + '/image-%08d.jpg' % f_id
       
        frame_mask = read_box(img_path, self.dict_list, 
                              normalized_labels=True,ratio=self.ratio,
                              i_ratio=self.i_ratio,w_h_max=self.w_h_max,root_path=self.root+"data/frames_16")  ### [w,h,1]
        frame_mask = Image.fromarray(frame_mask).convert('L')

        return frame_mask

    def load_static_image(self, entry):
        path = os.path.join(*entry['image'])
        path = '%s/data/images/%s' % (self.root, path)
        
        img = util.load_img(path)
        
        img = self.img_transform(img)
        return img

    def select_inactive_instances(self, entry):
        positive = entry['image'][0:3]
        negative = positive
        while negative == positive:
            neg_entry = self.data[np.random.randint(0, len(self.data))]
            negative = neg_entry['image'][0:3]
        positive = self.load_static_image(entry)
        negative = self.load_static_image(neg_entry)
        return positive, negative

# ----------------------------------------------------------------------------------------------------------#

class OPRAHeatmaps(HeatmapDataset):
    def __init__(self, root, split, std_norm=True):
        hm_file = 'data/opra/heatmaps.h5'
        super().__init__(root, split, hm_file=hm_file, std_norm=std_norm)

        annots = json.load(open('data/opra/annotations.json'))
        if not os.path.exists(hm_file):
            generate_heatmaps(annots, kernel_size=3.0, out_file=hm_file, transpose=False)

        self.verbs = annots['verbs']
        self.train_data, self.val_data = annots['train_images'], annots['test_images']
        self.data = self.train_data if self.split == 'train' else self.val_data
        print('%d train images, %d test images' % (len(self.train_data), len(self.val_data)))

    def load_image(self, entry):
        path = os.path.join(*entry['image'])
        path = '%s/data/images/%s' % (self.root, path)
        img = util.load_img(path)
        return img

    def load_image_heatmap(self, entry):
        img = self.load_image(entry)
        hm_key = tuple(entry['image']) + (str(entry['verb']),)
        heatmap = self.heatmaps(hm_key)
        img, heatmap = self.pair_transform(img, heatmap)
        return img, heatmap
