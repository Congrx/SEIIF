import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import scipy.io as scio
from math import sqrt
from numpy import clip
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
from einops import rearrange


class MLPRefiner(nn.Module):
    """Multilayer perceptrons (MLPs), refiner used in LIIF.

    Args:
        in_dim (int): Input dimension.
        out_dim (int): Output dimension.
        hidden_list (list[int]): List of hidden dimensions.
    """

    def __init__(self, in_dim, out_dim, hidden_list=None, act=None):
        super().__init__()
        layers = []
        lastv = in_dim
        if hidden_list:
            for hidden in hidden_list:
                layers.append(nn.Linear(lastv, hidden))
                if act == 'cos':
                    layers.append(Cos())
                elif act == 'sin':
                    layers.append(Sin())
                else:
                    layers.append(nn.ReLU())
                lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): The input of MLP.

        Returns:
            Tensor: The output of MLP.
        """
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is None:
            pass
        else:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')


class LF_Decoding(nn.Module):
    def __init__(self, angRes, channel):
        super(LF_Decoding, self).__init__()
        self.angRes = angRes
        self.channel = 32
        self.shrink = nn.Conv2d(channel,self.channel,1,1,0)
        self.combine = MLPRefiner((self.channel)*3, self.channel)
        self.epi1_decoding = MLPRefiner((self.channel+2)*3+2, self.channel, hidden_list=[self.channel*4, self.channel*4])
        self.epi2_decoding = MLPRefiner((self.channel+2)*3+2, 1, hidden_list=[self.channel*4,self.channel*4])
        self.spatial_decoding_1 = MLPRefiner((self.channel+2)+2, self.channel, hidden_list=[self.channel*4,self.channel*4])
        self.spatial_decoding_2 = MLPRefiner((self.channel+2)+2, 1, hidden_list=[self.channel*4,self.channel*4])
        
        
    def forward(self, fea, HH, WW, scale):     
        fea = self.shrink(fea)
        B, C, h , w = fea.shape
        B = B // self.angRes // self.angRes
        b = B
        
        # Set slope candidates for oriented line sampling used in EIIF based on scaling factor
        if scale < 2.5:
            line_slope = [-4,-3,-2,-1,0,1,2,3,4]      
        elif scale < 3.5:
            line_slope = [-3,-2,-1,0,1,2,3]       
        else:
            line_slope = [-2,-1,0,1,2]       
        
        # SIIF branch 1: The first upsampling step in SIIF branch
        fea_1 = fea
        coord_highres1 = make_coord((h, WW)).repeat(B*self.angRes*self.angRes, 1, 1).clamp(-1 + 1e-6, 1 - 1e-6).cuda()     
        feat_coord = make_coord((h,w), flatten=False).cuda().permute(2, 0, 1).unsqueeze(0).expand(B*self.angRes*self.angRes, 2, h, w)       
        high_fea1 = F.grid_sample(fea_1, coord_highres1.flip(-1).unsqueeze(1), mode='nearest',align_corners=False)
        high_fea1 = high_fea1[:, :, 0, :].permute(0,2,1)     
        high_coord1 = F.grid_sample(feat_coord, coord_highres1.flip(-1).unsqueeze(1),mode='nearest', align_corners=False)
        high_coord1 = high_coord1[:, :, 0, :].permute(0, 2, 1)   
        spatial_coord1 = coord_highres1 - high_coord1
        spatial_coord1[:,:,0] *= h
        spatial_coord1[:,:,1] *= w
        cell_spatial_1 = torch.ones_like(spatial_coord1)
        cell_spatial_1[:,:,0] *= 2 / h * h
        cell_spatial_1[:,:,1] *= 2 / WW * w
        inp1 = torch.cat([high_fea1, spatial_coord1, cell_spatial_1], dim=-1)      
        inp1 = rearrange(inp1,'(B a1 a2) (H W) C -> (B a1 a2 H W) C',a1=self.angRes,a2=self.angRes,H=h,W=WW)
        spatial_fea1 = self.spatial_decoding_1(inp1)   
        spatial_fea1 = rearrange(spatial_fea1, '(B a1 a2 H W) C -> B (a1 a2) C H W',a1=self.angRes,a2=self.angRes,H=h,W=WW)
        spatial_fea1 = rearrange(spatial_fea1, 'B (a1 a2) C H W -> (B a2 W) C a1 H',a1=self.angRes,a2=self.angRes,H=h,W=WW)
        
        
        # EIIF branch 1: The first upsampling step in EIIF branch
        fea = rearrange(fea,'(B a1 a2) C H W -> (B a1 H) C a2 W',a1=self.angRes,a2=self.angRes,H=h,W=w)        
        rv = 2 / self.angRes      
        rw = 2 / w
        delta_v_total = [0,1,-1]    
        coord_highres = make_coord((self.angRes, WW)).repeat(B*self.angRes*h, 1, 1).clamp(-1 + 1e-6, 1 - 1e-6).cuda()    
        bs, q = coord_highres.shape[:2] 
        feat_coord = make_coord((self.angRes,w), flatten=False).cuda().permute(2, 0, 1).unsqueeze(0).expand(B*self.angRes*h, 2, self.angRes, w)      
        # residual feature
        high_fea_base = F.grid_sample(fea, coord_highres.flip(-1).unsqueeze(1), mode='nearest',align_corners=False)
        high_fea_base = high_fea_base[:, :, 0, :]  
        high_fea_base = rearrange(high_fea_base, '(B a1 H) C (a2 W) -> (B a2 W) C a1 H', a1=self.angRes,a2=self.angRes,H=h,W=WW)
        # Calculate feature variance under all slope candidates for oriented line sampling 
        total_sample_feature_var = []
        for slope_item in line_slope:      
            sample_feature_no_coord = []
            for delta_v in delta_v_total:   
                delta_w = slope_item * delta_v
                coord_ = coord_highres.clone()
                v_coord = coord_[:,:,0].clone() + delta_v * rv
                w_coord = coord_[:,:,1].clone() + delta_w * rw
                if delta_v == 1:              
                    overarea_coord = v_coord > 1    
                    v_coord[overarea_coord] -= (delta_v * rv + 2 * rv)   
                    w_coord[overarea_coord] -= (delta_w * rw + slope_item * 2 * rw)
                if delta_v == -1: 
                    overarea_coord = v_coord < -1   
                    v_coord[overarea_coord] -= (delta_v * rv - 2 * rv) 
                    w_coord[overarea_coord] -= (delta_w * rw - slope_item * 2 * rw)
                coord_[:,:,0] = v_coord
                coord_[:,:,1] = w_coord
                coord_ = coord_.clamp(-1+1e-6,1-1e-6)
                high_fea_spatial = F.grid_sample(fea, coord_.flip(-1).unsqueeze(1), mode='nearest',align_corners=False)
                high_fea_spatial = high_fea_spatial[:, :, 0, :].permute(0,2,1)    
                sample_feature_no_coord.append(high_fea_spatial)
            sample_feature_no_coord = torch.stack(sample_feature_no_coord,dim=2)   
            var_sample = sample_feature_no_coord.var(dim=2)        
            var_sample = var_sample.sum(dim=-1)                  
            total_sample_feature_var.append(var_sample)
        total_sample_feature_var = torch.stack(total_sample_feature_var,dim=-1)  
        _,slope_index = torch.min(total_sample_feature_var,dim=-1)     
        point_slope = line_slope[0] + slope_index * 1
        # Execute oriented line sampling based on minimum feature variance
        total_sample_feature = []
        for delta_v in delta_v_total:  
            delta_w = point_slope * delta_v
            coord_ = coord_highres.clone()
            v_coord = coord_[:,:,0].clone() + delta_v * rv
            w_coord = coord_[:,:,1].clone() + delta_w * rw
            if delta_v == 1:            
                overarea_coord = v_coord > 1    
                v_coord = v_coord - overarea_coord * (delta_v * rv + 2 * rv)
                w_coord = w_coord - overarea_coord * (delta_w * rw + point_slope * 2 * rw)
            if delta_v == -1:  
                overarea_coord = v_coord < -1   
                v_coord = v_coord - overarea_coord * (delta_v * rv - 2 * rv)
                w_coord = w_coord - overarea_coord * (delta_w * rw - point_slope * 2 * rw)
            coord_[:,:,0] = v_coord
            coord_[:,:,1] = w_coord
            coord_ = coord_.clamp(-1+1e-6,1-1e-6)
            high_fea_spatial = F.grid_sample(fea, coord_.flip(-1).unsqueeze(1), mode='nearest',align_corners=False)
            high_fea_spatial = high_fea_spatial[:, :, 0, :].permute(0,2,1)     
            high_coord = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1),mode='nearest', align_corners=False)
            high_coord = high_coord[:, :, 0, :].permute(0, 2, 1)   
            spatial_coord = coord_highres - high_coord
            spatial_coord[:,:,0] *= self.angRes
            spatial_coord[:,:,1] *= w      
            inp = torch.cat([high_fea_spatial, spatial_coord], dim=-1)
            total_sample_feature.append(inp)
        total_sample_feature = torch.cat(total_sample_feature,dim=-1) 
        cell_epi_1 = torch.ones_like(spatial_coord)      
        cell_epi_1[:,:,0] *= 2 / self.angRes * self.angRes
        cell_epi_1[:,:,1] *= 2 / WW * w
        total_sample_feature = torch.cat([total_sample_feature,cell_epi_1],dim=-1)
        total_sample_feature = total_sample_feature.view(bs*q,-1)     
        HR_LF_feat = self.epi1_decoding(total_sample_feature)   
        
        # Cross-branch feature interaction between SIIF branch and EIIF branch after the first upsampling step
        HR_LF_feat = rearrange(HR_LF_feat, '(B a2 W) C-> B C a2 W', a2=self.angRes, W=WW)
        HR_LF_feat = rearrange(HR_LF_feat, '(B a1 H) C a2 W -> (B a2 W) C a1 H', a1=self.angRes, a2=self.angRes, H=h, W=WW)
        HR_LF_feat = torch.cat([high_fea_base,HR_LF_feat,spatial_fea1],dim=1)
        HR_LF_feat = rearrange(HR_LF_feat, '(B a2 W) C a1 H -> (B a2 W a1 H) C', a1=self.angRes, a2=self.angRes, H=h, W=WW)
        HR_LF_feat = self.combine(HR_LF_feat)
        HR_LF_feat = rearrange(HR_LF_feat, '(B a2 W a1 H) C -> (B a2 W) C a1 H', a1=self.angRes, a2=self.angRes, H=h, W=WW)
        spatial_fea1 = rearrange(HR_LF_feat, '(B a2 W) C a1 H -> (B a1 a2) C H W', a1=self.angRes, a2=self.angRes, H=h, W=WW)

        
        # SIIF branch 2: The second upsampling step in SIIF branch
        coord_highres2 = make_coord((HH, WW)).repeat(B*self.angRes*self.angRes, 1, 1).clamp(-1 + 1e-6, 1 - 1e-6).cuda()  
        feat_coord = make_coord((h,WW), flatten=False).cuda().permute(2, 0, 1).unsqueeze(0).expand(B*self.angRes*self.angRes, 2, h, WW)     
        high_fea2 = F.grid_sample(spatial_fea1, coord_highres2.flip(-1).unsqueeze(1), mode='nearest',align_corners=False)
        high_fea2 = high_fea2[:, :, 0, :].permute(0,2,1)     
        high_coord2 = F.grid_sample(feat_coord, coord_highres2.flip(-1).unsqueeze(1),mode='nearest', align_corners=False)
        high_coord2 = high_coord2[:, :, 0, :].permute(0, 2, 1)  
        spatial_coord2 = coord_highres2 - high_coord2
        spatial_coord2[:,:,0] *= h
        spatial_coord2[:,:,1] *= WW
        cell_spatial_2 = torch.ones_like(spatial_coord2)
        cell_spatial_2[:,:,0] *= 2 / HH * h
        cell_spatial_2[:,:,1] *= 2 / WW * WW
        inp1 = torch.cat([high_fea2, spatial_coord2,cell_spatial_2], dim=-1)     
        inp1 = rearrange(inp1,'(B a1 a2) (H W) C -> (B a1 a2 H W) C',a1=self.angRes,a2=self.angRes,H=HH,W=WW)
        spatial_fea2 = self.spatial_decoding_2(inp1) 
        out_1 = rearrange(spatial_fea2, '(B a1 a2 H W) C -> B (a1 a2) C H W',a1=5,a2=5,H=HH,W=WW)

        # EIIF branch 2: The second upsampling step in EIIF branch
        ru = 2 / 5    
        rh = 2 / h
        delta_u_total = [0,1,-1]    
        coord_highres = make_coord((self.angRes, HH)).repeat(B*self.angRes*WW, 1, 1).clamp(-1 + 1e-6, 1 - 1e-6).cuda()  
        bs, q = coord_highres.shape[:2] 
        feat_coord = make_coord((self.angRes,h), flatten=False).cuda().permute(2, 0, 1).unsqueeze(0).expand(B*self.angRes*WW, 2, self.angRes, h)  
        # Calculate feature variance under all slope candidates for oriented line sampling 
        total_sample_feature_var = []
        for slope_item in line_slope:       
            sample_feature_no_coord = []
            for delta_u in delta_u_total:  
                delta_h = slope_item * delta_u
                coord_ = coord_highres.clone()
                u_coord = coord_[:,:,0].clone() + delta_u * ru
                h_coord = coord_[:,:,1].clone() + delta_h * rh
                if delta_u == 1:
                    overarea_coord = u_coord > 1    
                    u_coord[overarea_coord] -= (delta_u * ru + 2 * ru) 
                    h_coord[overarea_coord] -= (delta_h * rh + slope_item * 2* rh)
                if delta_u == -1:
                    overarea_coord = u_coord < -1  
                    u_coord[overarea_coord] -= (delta_u * ru - 2 * ru)  
                    h_coord[overarea_coord] -= (delta_h * rh - slope_item * 2* rh)
                coord_[:,:,0] = u_coord
                coord_[:,:,1] = h_coord
                coord_ = coord_.clamp(-1+1e-6,1-1e-6)
                high_fea_spatial = F.grid_sample(HR_LF_feat, coord_.flip(-1).unsqueeze(1), mode='nearest',align_corners=False)
                high_fea_spatial = high_fea_spatial[:, :, 0, :].permute(0,2,1)
                sample_feature_no_coord.append(high_fea_spatial)
            sample_feature_no_coord = torch.stack(sample_feature_no_coord,dim=2) 
            var_sample = sample_feature_no_coord.var(dim=2)        
            var_sample = var_sample.sum(dim=-1)              
            total_sample_feature_var.append(var_sample)    
        total_sample_feature_var = torch.stack(total_sample_feature_var,dim=-1)  
        _,slope_index = torch.min(total_sample_feature_var,dim=-1)      
        point_slope = line_slope[0] + slope_index * 1
        # Execute oriented line sampling based on minimum feature variance
        total_sample_feature = []
        for delta_u in delta_u_total:  
            delta_h = point_slope * delta_u
            coord_ = coord_highres.clone()
            u_coord = coord_[:,:,0].clone() + delta_u * ru
            h_coord = coord_[:,:,1].clone() + delta_h * rh
            if delta_u == 1:
                overarea_coord = u_coord > 1   
                u_coord = u_coord - overarea_coord * (delta_u * ru + 2 * ru)
                h_coord = h_coord - overarea_coord * (delta_h * rh + point_slope * 2* rh)
            if delta_u == -1:
                overarea_coord = u_coord < -1   
                u_coord = u_coord - overarea_coord * (delta_u * ru - 2 * ru)
                h_coord = h_coord - overarea_coord * (delta_h * rh - point_slope * 2* rh)
            coord_[:,:,0] = u_coord
            coord_[:,:,1] = h_coord
            coord_ = coord_.clamp(-1+1e-6,1-1e-6)
            high_fea_spatial = F.grid_sample(HR_LF_feat, coord_.flip(-1).unsqueeze(1), mode='nearest',align_corners=False)
            high_fea_spatial = high_fea_spatial[:, :, 0, :].permute(0,2,1)   
            high_coord = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1),mode='nearest', align_corners=False)
            high_coord = high_coord[:, :, 0, :].permute(0, 2, 1) 
            spatial_coord = coord_highres - high_coord
            spatial_coord[:,:,0] *= self.angRes
            spatial_coord[:,:,1] *= h      
            inp = torch.cat([high_fea_spatial, spatial_coord], dim=-1)
            total_sample_feature.append(inp)
        total_sample_feature = torch.cat(total_sample_feature,dim=-1) 
        cell_epi_2 = torch.ones_like(spatial_coord)
        cell_epi_2[:,:,0] *= 2 / self.angRes * self.angRes
        cell_epi_2[:,:,1] *= 2 / HH * h
        total_sample_feature = torch.cat([total_sample_feature,cell_epi_2],dim=-1)
        total_sample_feature = total_sample_feature.view(bs*q,-1)     
        HR_LF_result = self.epi2_decoding(total_sample_feature)    
        HR_LF_result = rearrange(HR_LF_result, '(B a1 H) C-> B C a1 H', a1=self.angRes, H=HH)
        out_sv = rearrange(HR_LF_result, '(B a2 W) C a1 H -> B (a1 a2) C H W', a1=self.angRes, a2=self.angRes, H=HH, W=WW)
        
        # Final step: combine upsampling results of EIIF branch and SIIF branch
        out = (out_sv+out_1) / 2
        out = rearrange(out, 'B (a1 a2) C H W -> B C (a1 H) (a2 W)', a1=self.angRes, a2=self.angRes, H=HH, W=WW)
        return out

def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):      
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()        
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)    
    if flatten:   
        ret = ret.view(-1, ret.shape[-1])
    return ret


