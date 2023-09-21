import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models import backbones
from models.SEM import Semantic_Enhancement_Module

class Hand_Net(nn.Module):

    def __init__(self, num_classes, max_len, hidden_size, pool_fn='L2', ant_loss=None,groups=16):
        super().__init__()
        self.num_classes = num_classes
        self.max_len = max_len
        self.hidden_size = hidden_size

        self.pool_fn = pool_fn
        self.ant_loss = ant_loss
        
        self.groups=groups

    def init_backbone(self, backbone):
        self.backbone = backbone()
        self.spatial_dim = self.backbone.spatial_dim
        self.rnn = nn.LSTM(self.backbone.feat_dim, self.hidden_size, batch_first=True)  # (B, T, num_maps)
        self.fc = nn.Linear(self.hidden_size, self.num_classes)

        feat_dim = self.backbone.feat_dim
        self.project = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(True),
            nn.Conv2d(feat_dim, feat_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(True),
        )
          
        self.sem = Semantic_Enhancement_Module(num_class=self.num_classes,groups=self.groups)
        
        self.backbone_fn = backbone.__name__

        # LSTM hidden state
        n_layers = 1
        h0 = torch.zeros(n_layers, 1, self.hidden_size)
        c0 = torch.zeros(n_layers, 1, self.hidden_size)
        nn.init.xavier_normal_(h0, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(c0, gain=nn.init.calculate_gain('relu'))
        self.h0 = nn.Parameter(h0, requires_grad=True)
        self.c0 = nn.Parameter(c0, requires_grad=True)

        print('FrameLSTM created with out_ch: %d and spatial dim: %d' % (self.hidden_size, self.spatial_dim))

    def get_hidden_state(self, B, device):
        h = self.h0.expand(self.h0.shape[0], B, self.h0.shape[2]).contiguous().to(device)
        c = self.c0.expand(self.c0.shape[0], B, self.c0.shape[2]).contiguous().to(device)
        return (h, c)

    def flatten_apply(self, tensor, module):
        shape = tensor.shape
        flat = (shape[0] * shape[1],) + shape[2:]
        tensor = tensor.view(flat)
        out = module(tensor)
        uflat = (shape[0], shape[1],) + out.shape[1:]
        out = out.view(uflat)
        return out

    def embed_clip(self, frame_feats, **kwargs):

        # sort clip features by length
        B = frame_feats.shape[0]
        packed_input = pack_padded_sequence(frame_feats, kwargs['length'], batch_first=True, enforce_sorted=False)
        self.rnn.flatten_parameters()  # Otherwise it throws a warning?
        packed_output, (hn, cn) = self.rnn(packed_input, self.get_hidden_state(B, frame_feats.device))
        clip_feats = hn[-1]
        output, _ = pad_packed_sequence(packed_output)
        output = output.transpose(0, 1)  # (B, T, 2048)
        return clip_feats, output

    def pool(self, frame_feats):
        if self.pool_fn == 'L2':
            pool_feats = F.lp_pool2d(frame_feats, 2, self.spatial_dim)
        elif self.pool_fn == 'avg':
            pool_feats = F.avg_pool2d(frame_feats, self.spatial_dim)
        return pool_feats

    def anticipation_loss(self, frame_feats, lstm_feats, batch):
        B = frame_feats.shape[0]
        T = lstm_feats.shape[1]

        positive, negative = batch['positive'], batch['negative']
        target, length = batch['verb'], batch['length']

        # select the active frame from the clip
        lstm_preds = self.fc(lstm_feats)  # (B, max_L, #classes)
        lstm_preds = lstm_preds.view(B * T, -1)

        target_flat = target.unsqueeze(1).expand(target.shape[0], T).contiguous().view(-1)
        pred_scores = -F.cross_entropy(lstm_preds, target_flat, reduction='none').view(B, T)

        _, frame_idx = pred_scores.max(1)
        frame_idx = torch.min(frame_idx, length - 1)  # don't select a padding frame!
        active_feats = frame_feats[torch.arange(B), frame_idx]  # (B, 256, 28, 28)
        active_pooled = self.pool(active_feats).view(B, -1)

        mean_feats = frame_feats.mean(1)
        mean_pooled=self.pool(mean_feats).view(B,-1)

        label = torch.zeros(B, self.num_classes).cuda()
        for i in range(B):
            label[i][target[i]] = 1

        def embed(x,label):
            pred_frame = self.backbone(x)
            
            pred_frame=self.sem(pred_frame,label)
            pred_frame=self.project(pred_frame)
            pooled = self.pool(pred_frame).view(B, -1)
            return pooled

        positive_pooled = embed(positive,label)

        _, (hn, cn) = self.rnn(positive_pooled.unsqueeze(1), self.get_hidden_state(B, positive_pooled.device))
        preds = self.fc(hn[-1])
        aux_loss = F.cross_entropy(preds, target, reduction='none')

        if self.ant_loss == 'mse':
            ant_loss = 0.1 * ((positive_pooled - active_pooled) ** 2).mean(1)
            ant_loss+=0.02*((positive_pooled - mean_pooled) ** 2).mean(1)
        elif self.ant_loss == 'triplet':
            negative_pooled = self.backbone(negative)
            negative_pooled = self.pool(negative_pooled).view(B, -1)
            anc, pos, neg,mean_p = F.normalize(positive_pooled, 2), F.normalize(active_pooled, 2), F.normalize(negative_pooled,2),F.normalize(mean_pooled,2)
            ant_loss = F.triplet_margin_loss(anc, pos, neg, margin=0.5, reduction='none')
            ant_loss+=0.2*F.triplet_margin_loss(mean_p,pos,neg,margin=0.5, reduction='none')

        return {'ant_loss': ant_loss, 'aux_loss': aux_loss}

    def forward(self, batch, **kwargs):
        frames, length = batch['frames'], batch['length']
        
        frames_masks = batch['frames_mask']

        B, T = frames.shape[:2]
        S = self.spatial_dim

        frame_feats = self.flatten_apply(frames, lambda t: self.backbone(t))  # (B, T, 256, 28, 28)
        frames_masks = frames_masks.view(B * T, 1, frames_masks.size(2), frames_masks.size(3))
        frames_masks = F.interpolate(frames_masks, (28, 28), mode="bilinear")

        frames_masks = frames_masks.view(B, T, 1, 28, 28).repeat(1, 1, 2048, 1, 1)
        frames_masks = frames_masks * frame_feats

        pool_feats = self.flatten_apply(frame_feats, lambda t: self.pool(t)).view(B, T, -1)  # (B, T, 256)
        pool_mask = self.flatten_apply(frames_masks, lambda t: self.pool(t)).view(B, T, -1)
        pool_feats = pool_feats +0.5* pool_mask
        

        clip_feats, lstm_feats = self.embed_clip(pool_feats, length=length)  # (B, 2048)
        preds = self.fc(clip_feats)

        loss_dict = {}
        target = batch['verb']
        cls_loss = F.cross_entropy(preds, target, reduction='none')

        loss_dict.update({'cls_loss': cls_loss})

        if not self.training or self.ant_loss is None:
            return preds, loss_dict

        loss_dict.update(self.anticipation_loss(frame_feats, lstm_feats, batch))

        return preds, loss_dict

    # forward/backward passes for class activation mapping
    def ic_features(self, image, **kwargs):
        feat = self.backbone(image)
        return feat.unsqueeze(1)

    def ic_classifier(self, frame_feats,label, **kwargs):
        B = frame_feats.shape[0]
        frame_feats = frame_feats.squeeze(1)

        if self.ant_loss is not None:
            
            frame_feats=self.sem(frame_feats,label)
            frame_feats = self.project(frame_feats)

        pool_feats = self.pool(frame_feats).view(B, -1)

        _, (hn, cn) = self.rnn(pool_feats.unsqueeze(1), self.get_hidden_state(B, pool_feats.device))
        pred = self.fc(hn[-1])
        return pred


def hand_net(num_classes, max_len, backbone, hidden_size=2048, ant_loss='mse',groups=16):
    net = Hand_Net(num_classes, max_len, hidden_size, ant_loss=ant_loss, groups=groups)
    net.init_backbone(backbone)
    
    return net
