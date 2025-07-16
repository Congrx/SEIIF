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

def horizontal_sampling(fea,WW):     
    b, c, h, w = fea.shape    
    coord_highres = make_coord((h, WW)).repeat(b, 1, 1).clamp(-1 + 1e-6, 1 - 1e-6).cuda()     
    feat_coord = make_coord((h,w), flatten=False).cuda().permute(2, 0, 1).unsqueeze(0).expand(b, 2, h, w)     
    high_fea = F.grid_sample(fea, coord_highres.flip(-1).unsqueeze(1), mode='nearest',align_corners=False)
    high_fea = high_fea[:, :, 0, :].permute(0,2,1)     
    high_coord = F.grid_sample(feat_coord, coord_highres.flip(-1).unsqueeze(1),mode='nearest', align_corners=False)
    high_coord = high_coord[:, :, 0, :].permute(0, 2, 1)    
    relative_coord = coord_highres - high_coord   
    relative_coord[:,:,0] *= h
    relative_coord[:,:,1] *= w
    cell = torch.ones_like(relative_coord)
    cell[:,:,0] *= 2 / h * h
    cell[:,:,1] *= 2 / WW * w
    inp = torch.cat([high_fea, relative_coord, cell], dim=-1)
    return inp     

def vertical_sampling(fea,HH):      
    b, c, h, w = fea.shape    
    coord_highres = make_coord((HH, w)).repeat(b, 1, 1).clamp(-1 + 1e-6, 1 - 1e-6).cuda()     
    feat_coord = make_coord((h,w), flatten=False).cuda().permute(2, 0, 1).unsqueeze(0).expand(b, 2, h, w)     
    high_fea = F.grid_sample(fea, coord_highres.flip(-1).unsqueeze(1), mode='nearest',align_corners=False)
    high_fea = high_fea[:, :, 0, :].permute(0,2,1)      
    high_coord = F.grid_sample(feat_coord, coord_highres.flip(-1).unsqueeze(1),mode='nearest', align_corners=False)
    high_coord = high_coord[:, :, 0, :].permute(0, 2, 1)   
    relative_coord = coord_highres - high_coord  
    relative_coord[:,:,0] *= h
    relative_coord[:,:,1] *= w
    cell = torch.ones_like(relative_coord)
    cell[:,:,0] *= 2 / HH * h
    cell[:,:,1] *= 2 / w * w
    inp = torch.cat([high_fea, relative_coord, cell], dim=-1)
    return inp      

def generate_epi_coord(coord_highres, delta_ang, delta_spa, r_ang, r_spa, slope):
    coord_ = coord_highres.clone()
    ang_coord = coord_[:,:,0].clone() + delta_ang * r_ang
    spa_coord = coord_[:,:,1].clone() + delta_spa * r_spa
    if delta_ang == 1:                  
        overarea_coord = ang_coord > 1    
        ang_coord = ang_coord - overarea_coord * (delta_ang * r_ang + 2 * r_ang)   
        spa_coord = spa_coord - overarea_coord * (delta_spa * r_spa + slope * 2 * r_spa)
    if delta_ang == -1: 
        overarea_coord = ang_coord < -1   
        ang_coord = ang_coord - overarea_coord * (delta_ang * r_ang - 2 * r_ang) 
        spa_coord = spa_coord - overarea_coord * (delta_spa * r_spa - slope * 2 * r_spa)
    coord_[:,:,0] = ang_coord
    coord_[:,:,1] = spa_coord
    coord_ = coord_.clamp(-1+1e-6,1-1e-6)
    return coord_

def oriented_line_sampling(fea, Ang, Spa, line_slope):      
    b, c, ang, spa = fea.shape      
    r_ang = 2 / ang     
    r_spa = 2 / spa 
    delta_ang_list = [0,1,-1]
    
    eps_shift = 1e-6
    coord_highres = make_coord((Ang, Spa)).repeat(b, 1, 1).clamp(-1 + 1e-6, 1 - 1e-6).cuda()     
    feat_coord = make_coord((ang, spa), flatten=False).cuda().permute(2, 0, 1).unsqueeze(0).expand(b, 2, ang, spa)    
    
    # Calculate feature variance under all slope candidates for oriented line sampling 
    total_sample_feature_var = []
    for slope_item in line_slope:      
        sample_feature_no_coord = []
        for delta_ang in delta_ang_list:   
            delta_spa = slope_item * delta_ang
            coord_ = generate_epi_coord(coord_highres, delta_ang, delta_spa, r_ang, r_spa, slope_item)   
            high_fea = F.grid_sample(fea, coord_.flip(-1).unsqueeze(1), mode='nearest',align_corners=False)
            high_fea = high_fea[:, :, 0, :].permute(0,2,1)    
            sample_feature_no_coord.append(high_fea)
        sample_feature_no_coord = torch.stack(sample_feature_no_coord,dim=2)   
        var_sample = sample_feature_no_coord.var(dim=2)        
        var_sample = var_sample.sum(dim=-1)                  
        total_sample_feature_var.append(var_sample)
    total_sample_feature_var = torch.stack(total_sample_feature_var,dim=-1)  
    _,slope_index = torch.min(total_sample_feature_var,dim=-1)     
    point_slope = line_slope[0] + slope_index * 1
    # Execute oriented line sampling based on minimum feature variance
    total_sample_feature = []
    for delta_ang in delta_ang_list:  
        delta_spa = point_slope * delta_ang
        coord_ = generate_epi_coord(coord_highres, delta_ang, delta_spa, r_ang, r_spa, point_slope)
        high_fea = F.grid_sample(fea, coord_.flip(-1).unsqueeze(1), mode='nearest',align_corners=False)
        high_fea = high_fea[:, :, 0, :].permute(0,2,1)     
        high_coord = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1),mode='nearest', align_corners=False)
        high_coord = high_coord[:, :, 0, :].permute(0, 2, 1)   
        epi_coord = coord_highres - high_coord
        epi_coord[:,:,0] *= ang
        epi_coord[:,:,1] *= spa      
        inp = torch.cat([high_fea, epi_coord], dim=-1)
        total_sample_feature.append(inp)
    total_sample_feature = torch.cat(total_sample_feature,dim=-1) 
    cell_epi = torch.ones_like(epi_coord)      
    cell_epi[:,:,0] *= 2 / Ang * ang
    cell_epi[:,:,1] *= 2 / Spa * spa
    total_sample_feature = torch.cat([total_sample_feature,cell_epi],dim=-1)        
    return total_sample_feature     

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
        spatial_fea_hor = rearrange(horizontal_sampling(fea,WW),'(b a1 a2) (h w) c -> (b a1 a2 h w) c', a1=self.angRes,a2=self.angRes,h=h,w=WW)
        spatial_fea_hor = self.spatial_decoding_1(spatial_fea_hor) 
        spatial_fea_hor = rearrange(spatial_fea_hor, '(b a1 a2 h w) c -> (b a2 w) c a1 h',a1=self.angRes,a2=self.angRes,h=h,w=WW)

        
        # EIIF branch 1: The first upsampling step in EIIF branch
        fea = rearrange(fea,'(b a1 a2) c h w -> (b a1 h) c a2 w',a1=self.angRes,a2=self.angRes,h=h,w=w)
        epipolar_fea1 = oriented_line_sampling(fea, self.angRes, WW, line_slope)
        epipolar_fea1 = rearrange(epipolar_fea1,'(b a1 h) (a2 w) c -> (b a1 h a2 w) c', a1=self.angRes,a2=self.angRes,h=h,w=WW)  
        epipolar_fea1 = self.epi1_decoding(epipolar_fea1)   
        epipolar_fea1 = rearrange(epipolar_fea1, '(b a1 h a2 w) c-> (b a2 w) c a1 h', a1=self.angRes,a2=self.angRes,h=h,w=WW)
        
        # Cross-branch feature interaction between SIIF branch and EIIF branch after the first upsampling step
        # residual feature
        coord_highres = make_coord((self.angRes, WW)).repeat(B*self.angRes*h, 1, 1).clamp(-1 + 1e-6, 1 - 1e-6).cuda()
        high_fea_base = F.grid_sample(fea, coord_highres.flip(-1).unsqueeze(1), mode='nearest',align_corners=False)
        high_fea_base = high_fea_base[:, :, 0, :]
        high_fea_base = rearrange(high_fea_base, '(b a1 h) c (a2 w) -> (b a2 w) c a1 h', a1=self.angRes,a2=self.angRes,h=h,w=WW)
        HR_LF_feat = torch.cat([high_fea_base, epipolar_fea1, spatial_fea_hor],dim=1)
        HR_LF_feat = rearrange(HR_LF_feat, '(b a2 w) c a1 h -> (b a2 w a1 h) c', a1=self.angRes, a2=self.angRes, h=h, w=WW)
        HR_LF_feat = self.combine(HR_LF_feat)
        HR_LF_feat = rearrange(HR_LF_feat, '(b a2 w a1 h) c -> (b a2 w) c a1 h', a1=self.angRes, a2=self.angRes, h=h, w=WW)
        spatial_fea = rearrange(HR_LF_feat, '(b a2 w) c a1 h -> (b a1 a2) c h w', a1=self.angRes, a2=self.angRes, h=h, w=WW)

        
        # SIIF branch 2: The second upsampling step in SIIF branch
        spatial_fea_ver = rearrange(vertical_sampling(spatial_fea,HH),'(b a1 a2) (h w) c -> (b a1 a2 h w) c', a1=self.angRes,a2=self.angRes,h=HH,w=WW)
        spatial_fea_ver = self.spatial_decoding_2(spatial_fea_ver) 
        out_s = rearrange(spatial_fea_ver, '(b a1 a2 h w) c -> b (a1 a2) c h w',a1=self.angRes,a2=self.angRes,h=HH,w=WW)

        # EIIF branch 2: The second upsampling step in EIIF branch
        epipolar_fea2 = oriented_line_sampling(HR_LF_feat, self.angRes, HH, line_slope)
        epipolar_fea2 = rearrange(epipolar_fea2,'(b a2 w) (a1 h) c -> (b a2 w a1 h) c', a1=self.angRes,a2=self.angRes,h=HH,w=WW)    
        epipolar_fea2 = self.epi2_decoding(epipolar_fea2)    
        out_epi = rearrange(epipolar_fea2, '(b a2 w a1 h) c -> b (a1 a2) c h w', a1=self.angRes, a2=self.angRes, h=HH, w=WW)
        
        # Final step: combine upsampling results of EIIF branch and SIIF branch
        out = (out_epi+out_s) / 2
        out = rearrange(out, 'b (a1 a2) c h w -> b c (a1 h) (a2 w)', a1=self.angRes, a2=self.angRes, h=HH, w=WW)
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


