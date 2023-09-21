import torch
import argparse
import tqdm
import os
import glob

from utils import evaluation

parser = argparse.ArgumentParser()
parser.add_argument('--dset', default='opra')
parser.add_argument('--load', default=None)
parser.add_argument('--res', type=int, default=28)
parser.add_argument('--batch_size', type=int, default=1)
args = parser.parse_args()
# ------------------------------------------------------------#

import data
from data import opra
import torch.nn.functional as F


def generate_gt():

    dataset = opra.OPRAHeatmaps(root=data._DATA_ROOTS[args.dset], split='val')

    dataset.heatmaps = dataset.init_hm_loader()

    heatmaps, keys = [], []
    for index in tqdm.tqdm(range(len(dataset))):
        entry = dataset.data[index]
        hm_key = entry['image_key'] + (str(entry['verb']),)
        heatmap = dataset.heatmaps(hm_key)
        heatmap = torch.from_numpy(heatmap)
        heatmap = F.interpolate(heatmap.unsqueeze(0).unsqueeze(0), size=(224, 224),
                                mode='bilinear', align_corners=False)[0][0]
        heatmap = heatmap / (heatmap.sum() + 1e-12)
        print(heatmap.size())

        heatmaps.append(heatmap)
        keys.append(hm_key)

    heatmaps = torch.stack(heatmaps, 0)
    print(heatmaps.shape)
    torch.save({'heatmaps': heatmaps, 'keys': keys}, 'cv/%s/%s_gt.t7' % (args.dset, args.dset))

# ------------------------------------------------------------#

from models import intcam

def generate_heatmaps(model_path):

    testset = opra.OPRAHeatmaps(root=data._DATA_ROOTS[args.dset], split='val')


    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    print("testloader")

    from models import backbones, model

    torch.backends.cudnn.enabled = False
    
    net =model.hand_net(len(testset.verbs), max_len=-1, backbone=backbones.dr50_n28,groups=2)

    checkpoint = torch.load(model_path)

    weights, start_iter = checkpoint['net'], checkpoint['iter']
    net.load_state_dict(weights)
    net.eval().cuda()
    print('Loaded checkpoint from %s' % os.path.basename(args.load))

    gcam = intcam.IntCAM(net)
    heatmaps = []
    for batch in tqdm.tqdm(testloader, total=len(testloader)):
        img, verb = batch['img'], batch['verb']
        masks = gcam.generate_cams(img.cuda(), [verb])  # (B, T, C, 7, 7)
        # print(masks)
        mask = masks.mean(1)  # (B, C, 7, 7) <-- average across hallucinated time dim
        mask = mask.squeeze(1)  # get rid of single class dim
        heatmaps.append(mask.cpu())

    heatmaps = torch.cat(heatmaps, 0)  # (N, C, 7, 7)
    print(heatmaps.shape)

    keys = [testset.key(entry) for entry in testset.data]
    torch.save({'heatmaps': heatmaps, 'keys': keys}, '%s.%s.heatmaps' % (args.load, args.dset))


# generate heatmap predictions if they do not already exist
       
path="" ### model_path
path_1=os.listdir(path)
for pth_path in path_1:
    model_path = path + pth_path

    if model_path[-3:] == "pth":
        if os.path.exists('%s.%s.heatmaps' % (args.load, args.dset)):
            os.unlink('%s.%s.heatmaps' % (args.load, args.dset))

        if args.load is not None and not os.path.exists('%s.%s.heatmaps' % (args.load, args.dset)):
            generate_heatmaps(model_path=model_path)

        # ------------------------------------------------------------#

        # generate gt heatmaps if they do not already exist
        if not os.path.exists('cv/%s/%s_gt.t7' % (args.dset, args.dset)):
            generate_gt()
        gt = torch.load('cv/%s/%s_gt.t7' % (args.dset, args.dset))

        baselines = evaluation.Baselines(gt['heatmaps'].shape[0])
        heval = evaluation.Evaluator(gt, res=args.res, log=args.load)

        predictions = {
            'hand_net': baselines.checkpoint('%s.%s.heatmaps' % (args.load, args.dset)),
            
        }

        for method in predictions:
            print(method)
            print(pth_path)

            heatmaps = predictions[method]
            scores = heval.evaluate(heatmaps)






