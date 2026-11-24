import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .utils import make_coord
from . import rdn
from . import mlp


class LIIF(nn.Module):

    def __init__(self, n_colors, encoder_spec, imnet_spec=None, inp_size = None, 
                 local_ensemble=False, feat_unfold=False, cell_decode=False,
                 in_dim=None, out_dim=1, hidden_list=[128, 128, 64]):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode
        self.inp_size = inp_size

        self.encoder = rdn.RDN()
        self.out_dim = out_dim
        self.hidden_list = hidden_list

        if imnet_spec is not None:
            imnet_in_dim = self.encoder.G0
            if self.feat_unfold:
                imnet_in_dim *= 9
            imnet_in_dim += 3 # attach coord
            if self.cell_decode:
                imnet_in_dim += 3
            self.imnet = mlp.MLP(imnet_in_dim, self.out_dim, self.hidden_list)
        else:
            self.imnet = None

    def gen_feat(self, inp):
        self.feat = self.encoder(inp)
        return self.feat

    def query_rgb(self, coord, cell=None, height=None, weight=None):
        #coord:[B,HWC,3]
        #cell:[B,HWC,3]
        feat = self.feat #[B,N,c,HW]

        if self.imnet is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            return ret

        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])
            #[B,9N,c,HW]

        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            vz_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, vz_lst, eps_shift = [0], [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / self.inp_size / 2
        ry = 2 / self.inp_size / 2
        rz = 2 / feat.shape[-2] / 2
        #print(make_coord([48,48,feat.shape[-2]], flatten=False).shape)
        feat_coord = make_coord([self.inp_size, self.inp_size, feat.shape[-2]], flatten=False).cuda() #[H,W,c,3]
        feat_coord = feat_coord.permute(3, 0, 1, 2).view(3, self.inp_size*self.inp_size, feat.shape[-2]) \
            .unsqueeze(0).expand(feat.shape[0], 3, self.inp_size*self.inp_size, feat.shape[-2]) #[3,H,W,c]>[3,HW,c]>[B,3,HW,c]

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                for vz in vz_lst:
                    coord_ = coord.clone() #[B,HWC,3]
                    coord_[:, :, 0] += vx * rx + eps_shift
                    coord_[:, :, 1] += vy * ry + eps_shift
                    coord_[:, :, 2] += vz * rz + eps_shift
                    coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                    q_feat = F.grid_sample(
                        feat.permute(0,1,3,2).view(feat.shape[0],feat.shape[1],self.inp_size,self.inp_size,feat.shape[2]), coord_.flip(-1).view(feat.shape[0],height,weight,-1,3),
                        mode='nearest', align_corners=False) #[B,N,H,W,c], [B,H,W,C,3]>[B,N,H,W,C]
                    q_feat = q_feat.view(feat.shape[0],feat.shape[1],coord.shape[1]).permute(0, 2, 1)#[B,N,C,H,W]>[B,N,HWC]>[B,HWC,N]
                    q_coord = F.grid_sample(
                        feat_coord.view(feat.shape[0],-1,self.inp_size,self.inp_size,feat.shape[2]), coord_.flip(-1).view(feat.shape[0],height,weight,-1,3),
                        mode='nearest', align_corners=False) #[B,3,H,W,c], [B,H,W,C,3]>[B,3,H,W,C]
                    q_coord = q_coord.view(feat.shape[0],-1,coord.shape[1]).permute(0, 2, 1) #[B,HWC,3]
                    
                    rel_coord = coord - q_coord #[B,HWC,3]
                    rel_coord[:, :, 0] *= self.inp_size
                    rel_coord[:, :, 1] *= self.inp_size
                    rel_coord[:, :, 2] *= feat.shape[2]
                    inp = torch.cat([q_feat, rel_coord], dim=-1)

                    if self.cell_decode:
                        rel_cell = cell.clone()
                        rel_cell[:, :, 0] *= self.inp_size
                        rel_cell[:, :, 1] *= self.inp_size
                        rel_cell[:, :, 2] *= feat.shape[2]
                        inp = torch.cat([inp, rel_cell], dim=-1)
                    
                    bs, q = coord.shape[:2]
                    pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                    preds.append(pred)

                    area = torch.abs(rel_coord[:, :, 0]) + torch.abs(rel_coord[:, :, 1]) + torch.abs(rel_coord[:, :, 2])
                    areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]; areas[0] = areas[7]; areas[7] = t
            t = areas[1]; areas[1] = areas[6]; areas[6] = t
            t = areas[2]; areas[2] = areas[5]; areas[5] = t
            t = areas[3]; areas[3] = areas[4]; areas[4] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret

    def forward(self, inp, coord, cell, height, weight):
        #inp:[B,c,H,W]
        #coord:[B,HWC,3]
        #cell:[B,HWC,3]
        H = height
        W = weight
        inp = inp.view(inp.shape[0],inp.shape[1],-1) #[B,c,HW]
        inp = inp.unsqueeze(1) #[B,1,c,HW]
        self.gen_feat(inp) #[B,N,c,HW]
        return self.query_rgb(coord, cell, height=H, weight=W) #[B,HWC,1]
