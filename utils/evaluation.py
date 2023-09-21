import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
import cv2

#--------------------------------------- Baselines ---------------------------------------------#

def makeGaussian(size, fwhm=3, center=None):
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]
    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

class Baselines:

    def __init__(self, N):
        self.N = N

    def gaussian(self, **kwargs):
        sz = 225
        hm = torch.zeros(sz, sz)
        hm[sz//2, sz//2] = 1.0

        K = sz//2
        g = makeGaussian(K, fwhm=sz//4)
        kernel = torch.from_numpy(g)

        hm = hm.unsqueeze(0).unsqueeze(0).float()
        kernel = kernel.unsqueeze(0).unsqueeze(0).float()
        hm = F.conv2d(hm, weight=kernel, padding=K//2-1)[0] # (1, 224, 224)

        hm = hm.expand(self.N, hm.shape[1], hm.shape[2])
        
        return {'heatmaps':hm, 'keys':None}

    def constant(self, fill=0, sz=7, **kwargs):
        hm = torch.zeros(self.N, sz, sz).fill_(fill)
        return {'heatmaps':hm, 'keys':None}

    def checkpoint(self, fl):
        return torch.load(fl) # (N, C, 28, 28) + keys

#--------------------------------------- Metrics ---------------------------------------------#
# adapted from: https://github.com/herrlich10/saliency/blob/master/benchmark/metrics.py

# KL Divergence
# kld(map2||map1) -- map2 is gt
def KLD(map1, map2, eps = 1e-12):
    map1, map2 = map1/(map1.sum()+eps), map2/(map2.sum() + eps)
    kld = np.sum(map2*np.log( map2/(map1+eps) + eps))
    return kld

# historgram intersection
def SIM(map1, map2, eps=1e-12):
    map1, map2 = map1/(map1.sum()+eps), map2/(map2.sum() + eps)
    intersection = np.minimum(map1, map2)
    return np.sum(intersection)

# AUC-J
def AUC_Judd(saliency_map, fixation_map, jitter=True):
    saliency_map = np.array(saliency_map, copy=False)
    fixation_map = np.array(fixation_map, copy=False) > 0.5
    # If there are no fixation to predict, return NaN
    if not np.any(fixation_map):
        return np.nan
    # Make the saliency_map the size of the fixation_map
    if saliency_map.shape != fixation_map.shape:
        saliency_map = cv2.resize(saliency_map, fixation_map.shape)
    # Jitter the saliency map slightly to disrupt ties of the same saliency value
    if jitter:
        saliency_map += np.random.rand(*saliency_map.shape) * 1e-7
    # Normalize saliency map to have values between [0,1]
    saliency_map =  (saliency_map - np.min(saliency_map)) / (np.max(saliency_map) - np.min(saliency_map) + 1e-12)

    S = saliency_map.ravel()
    F = fixation_map.ravel()
    S_fix = S[F] # Saliency map values at fixation locations
    n_fix = len(S_fix)
    n_pixels = len(S)
    # Calculate AUC
    thresholds = sorted(S_fix, reverse=True)
    tp = np.zeros(len(thresholds)+2)
    fp = np.zeros(len(thresholds)+2)
    tp[0] = 0; tp[-1] = 1
    fp[0] = 0; fp[-1] = 1
    for k, thresh in enumerate(thresholds):
        above_th = np.sum(S >= thresh) # Total number of saliency map values above threshold
        tp[k+1] = (k + 1) / float(n_fix) # Ratio saliency map values at fixation locations above threshold
        fp[k+1] = (above_th - k - 1) / float(n_pixels - n_fix) # Ratio other saliency map values above threshold
    return np.trapz(tp, fp) # y, x

def MAE(saliency_map, fixation_map):

    # If there are no fixation to predict, return NaN
    if not np.any(fixation_map):
        return np.nan
    # Make the saliency_map the size of the fixation_map
    if saliency_map.shape != fixation_map.shape:
        saliency_map = cv2.resize(saliency_map, fixation_map.shape)
    saliency_map = (saliency_map - np.min(saliency_map)) / (np.max(saliency_map) - np.min(saliency_map) + 1e-12)

    dis = abs(fixation_map - saliency_map)

    output=dis.mean()
   # print(dis)
    return output

def F_measure(saliency_map, fixation_map):
    saliency_map = np.array(saliency_map, copy=False)
    fixation_map = np.array(fixation_map, copy=False)>0.5
    # If there are no fixation to predict, return NaN
    if not np.any(fixation_map):
        return np.nan
    # Make the saliency_map the size of the fixation_map
    if saliency_map.shape != fixation_map.shape:
        saliency_map = cv2.resize(saliency_map, fixation_map.shape)
    saliency_map = (saliency_map - np.min(saliency_map)) / (np.max(saliency_map) - np.min(saliency_map) + 1e-12)
    thresh = 2 * np.mean(saliency_map)  #### ???????????????

    saliency_map=np.where(saliency_map>thresh, 1, 0)

    TP=np.where((saliency_map==fixation_map),1,0)

    tmp_1=np.where(saliency_map==1,1,0)
    tmp_2=np.where(fixation_map==0,1,0)

    FP=np.minimum(tmp_1,tmp_2)


    tmp_1=np.where(saliency_map==0,1,0)
    tmp_2=np.where(fixation_map==1,1,0)
    FN=np.minimum(tmp_1,tmp_2)


    TP=np.sum(TP)
    FP=np.sum(FP)

    FN=np.sum(FN)
    #print(FN)

    reca=TP/(TP+FN)

    prec=TP/(TP+FP)


    return reca,prec

def CC(saliency_map, fixation_map):
    saliency_map=(saliency_map - np.min(saliency_map)) / (np.max(saliency_map) - np.min(saliency_map) + 1e-12)
    #fixation_map=(fixation_map - np.min(fixation_map)) / (np.max(fixation_map) - np.min(fixation_map) + 1e-12)
    cor_array = np.corrcoef(saliency_map, fixation_map)
    return cor_array[0,1]


def image_binary(image,threshold):
    output=np.zeros(image.size).reshape(image.shape)
    for xx in range(image.shape[0]):
        for yy in range(image.shape[1]):
            if(image[xx][yy]>threshold):
                output[xx][yy]=1
    return output

def NSS_gt(saliency_map,fixation_map):

    if saliency_map.shape != fixation_map.shape:
        saliency_map = cv2.resize(saliency_map, fixation_map.shape)
    std=np.std(saliency_map)
    u=np.mean(saliency_map)

    smap=(saliency_map-u)/std
    fixation_map = (fixation_map - np.min(fixation_map)) / (np.max(fixation_map) - np.min(fixation_map) + 1e-12)
    fixation_map=image_binary(fixation_map,0.1)

    nss=smap*fixation_map

    nss=np.sum(nss)/np.sum(fixation_map)


    return nss


#--------------------------------------- Evaluator ---------------------------------------------#

class Evaluator:

    def __init__(self, gt, res=28, log=None):
        self.res = res
        self.keys = gt['keys']
        self.gt = self.make_dict(gt)

    def make_dict(self, preds):
        heatmaps, keys = preds['heatmaps'], preds['keys']
       # print(keys)
        if keys is None:
            keys = self.keys

        if heatmaps.shape[-1]!=self.res:
            heatmaps = self.batch_interp(heatmaps.unsqueeze(1), self.res).squeeze(1)
        pred_dict = dict(zip(keys, heatmaps))
        return pred_dict

    def score(self, pred, gt):

        pred, gt = pred.numpy(), gt.numpy()
        pred = pred/(pred.max()+1e-12)
        pred = cv2.GaussianBlur(pred, (7, 7), 0)

        scores = {}

        # KLD
        gt_real = np.array(gt)
        if gt_real.sum() == 0:
            gt_real = np.ones(gt_real.shape) / np.product(gt_real.shape)
        score = KLD(pred, gt_real)
        scores['KLD'] = score if not np.isnan(score) else None

        # SIM
        score = SIM(pred, gt_real)
        scores['SIM'] = score if not np.isnan(score) else None

        # AUC-J
        gt_binary = np.array(gt)
        gt_binary = (gt_binary/gt_binary.max()+1e-12) if gt_binary.max()>0 else gt_binary
        gt_binary = np.where(gt_binary>0.5, 1, 0)
        score = AUC_Judd(pred, gt_binary)
        scores['AUC-J'] = score if not np.isnan(score) else None

        #### MAE
        gt_real=np.array(gt)
        if gt_real.sum()==0:
            gt_real=np.ones(gt_real.shape)/np.product(gt_real.shape)
        gt_real=(gt_real/(gt_real.max()+1e-12))
        score=MAE(pred,gt_real)
        scores["MAE"]=score if not np.isnan(score) else None



        ###### CC
        gt_real = np.array(gt)
        if gt_real.sum() == 0:
            gt_real = np.ones(gt_real.shape) / np.product(gt_real.shape)
        score = CC(pred, gt_real)
        scores['CC'] = score if not np.isnan(score) else None

        ### NSS
        gt_real = np.array(gt)
        if gt_real.sum() == 0:
            gt_real = np.ones(gt_real.shape) / np.product(gt_real.shape)
        score = NSS_gt(pred, gt_real)
        scores['NSS'] = score if not np.isnan(score) else None


        return dict(scores)



    def batch_interp(self, inp, sz):
        out = []
        for i in range(0, inp.shape[0], 512):
            chunk = inp[i:i+512]
            out.append(F.interpolate(chunk, (sz, sz), mode='bilinear', align_corners=False))
        out = torch.cat(out, 0)
        return out

    def evaluate(self, preds):

        preds = self.make_dict(preds)

        scores = []
      #  recas=[]
      #  precs=[]
        for key in tqdm.tqdm(self.gt.keys()):
            if key not in preds:
                continue

            score= self.score(preds[key], self.gt[key])

            scores.append(score)
            

        
        #save to csv
        import csv
        headers=['KLD','SIM', 'AUC-J',"MAE","CC","NSS"]
        with open('result.csv','w') as f:
            f_csv=csv.DictWriter(f,headers)
            f_csv.writeheader()
            f_csv.writerows(scores)
        

       # write_out = []
        for key in ['KLD', 'SIM', 'AUC-J',"MAE","CC","NSS"]:
            key_score = [s[key] for s in scores if s[key] is not None]
            mean, stderr = np.mean(key_score), np.std(key_score)/(np.sqrt(len(key_score)))
            log_str = '%s: %.3f ± %.3f (%d/%d)'%(key, mean, stderr, len(key_score), len(self.gt.keys()))

            print(log_str)
        
        
        return scores, None


#----------------------------------------------------------------------------------------------#
