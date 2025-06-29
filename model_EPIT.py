import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from SEIIF_Decoder import LF_Decoding
from math import sqrt
class Net(nn.Module):
    def __init__(self, angRes):
        super(Net, self).__init__()
        channels = 64
        self.angRes = angRes

        #################### Initial Feature Extraction #####################
        self.conv_init0 = nn.Sequential(nn.Conv3d(1, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False))
        self.conv_init = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        ############# Deep Spatial-Angular Correlation Learning #############
        self.altblock = nn.Sequential(
            AltFilter(self.angRes, channels),
            AltFilter(self.angRes, channels),
            AltFilter(self.angRes, channels),
            AltFilter(self.angRes, channels),
            AltFilter(self.angRes, channels),
        )

        ########################### UP-Sampling #############################
        self.upsample = LF_Decoding(self.angRes,channels)
        
    def forward(self, lr, h_ori, w_ori, scale):
        u = 5
        v = 5
        x_multi = LFsplit(lr, self.angRes)
        b,n,c,h,w = x_multi.shape
        x_multi = x_multi.contiguous().view(b*n,-1,h,w)
        x_upscale = F.interpolate(x_multi,size=(h_ori,w_ori),mode='bicubic',align_corners=False)
        x_upscale = x_upscale.unsqueeze(1).contiguous().view(b,-1,c,int(h*scale),int(w*scale))
        
        
        # Initial Feature Extraction
        x = rearrange(x_multi, '(b n) c h w -> b c n h w',b=b,n=n)
        buffer = self.conv_init0(x)
        buffer = self.conv_init(buffer) + buffer

        # Deep Spatial-Angular Correlation Learning
        buffer = self.altblock(buffer) + buffer

        # UP-Sampling
        
        buffer = rearrange(buffer, 'b c n h w -> (b n) c h w')
        out = self.upsample(buffer,h_ori,w_ori,scale) + FormOutput(x_upscale)
        return out


class BasicTrans(nn.Module):
    def __init__(self, channels, spa_dim, num_heads=8, dropout=0.):
        super(BasicTrans, self).__init__()
        self.linear_in = nn.Linear(channels, spa_dim, bias=False)
        self.norm = nn.LayerNorm(spa_dim)
        self.attention = nn.MultiheadAttention(spa_dim, num_heads, dropout, bias=False)
        nn.init.kaiming_uniform_(self.attention.in_proj_weight, a=math.sqrt(5))
        self.attention.out_proj.bias = None
        self.attention.in_proj_bias = None
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(spa_dim),
            nn.Linear(spa_dim, spa_dim*2, bias=False),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(spa_dim*2, spa_dim, bias=False),
            nn.Dropout(dropout)
        )
        self.linear_out = nn.Linear(spa_dim, channels, bias=False)

    def gen_mask(self, h: int, w: int, k_h: int, k_w: int):
        attn_mask = torch.zeros([h, w, h, w])
        k_h_left = k_h // 2
        k_h_right = k_h - k_h_left
        k_w_left = k_w // 2
        k_w_right = k_w - k_w_left
        for i in range(h):
            for j in range(w):
                temp = torch.zeros(h, w)
                temp[max(0, i - k_h_left):min(h, i + k_h_right), max(0, j - k_w_left):min(w, j + k_w_right)] = 1
                attn_mask[i, j, :, :] = temp

        attn_mask = rearrange(attn_mask, 'a b c d -> (a b) (c d)')
        attn_mask = attn_mask.float().masked_fill(attn_mask == 0, float('-inf')).masked_fill(attn_mask == 1, float(0.0))

        return attn_mask

    def forward(self, buffer):
        [_, _, n, v, w] = buffer.size()
        attn_mask = self.gen_mask(v, w, self.mask_field[0], self.mask_field[1]).to(buffer.device)

        epi_token = rearrange(buffer, 'b c n v w -> (v w) (b n) c')
        epi_token = self.linear_in(epi_token)

        epi_token_norm = self.norm(epi_token)
        epi_token = self.attention(query=epi_token_norm,
                                   key=epi_token_norm,
                                   value=epi_token,
                                   attn_mask=attn_mask,
                                   need_weights=False)[0] + epi_token

        epi_token = self.feed_forward(epi_token) + epi_token
        epi_token = self.linear_out(epi_token)
        buffer = rearrange(epi_token, '(v w) (b n) c -> b c n v w', v=v, w=w, n=n)

        return buffer


class AltFilter(nn.Module):
    def __init__(self, angRes, channels):
        super(AltFilter, self).__init__()
        self.angRes = angRes
        self.epi_trans = BasicTrans(channels, channels*2)
        self.conv = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
        )

    def forward(self, buffer):
        shortcut = buffer
        [_, _, _, h, w] = buffer.size()
        self.epi_trans.mask_field = [self.angRes * 2, 11]

        # Horizontal
        buffer = rearrange(buffer, 'b c (u v) h w -> b c (v w) u h', u=self.angRes, v=self.angRes)
        buffer = self.epi_trans(buffer)
        buffer = rearrange(buffer, 'b c (v w) u h -> b c (u v) h w', u=self.angRes, v=self.angRes, h=h, w=w)
        buffer = self.conv(buffer) + shortcut

        # Vertical
        buffer = rearrange(buffer, 'b c (u v) h w -> b c (u h) v w', u=self.angRes, v=self.angRes)
        buffer = self.epi_trans(buffer)
        buffer = rearrange(buffer, 'b c (u h) v w -> b c (u v) h w', u=self.angRes, v=self.angRes, h=h, w=w)
        buffer = self.conv(buffer) + shortcut

        return buffer

def LFsplit(data, angRes):
    b, _, H, W = data.shape
    h = int(H/angRes)
    w = int(W/angRes)
    data_sv = []
    for u in range(angRes):
        for v in range(angRes):
            data_sv.append(data[:, :, u*h:(u+1)*h, v*w:(v+1)*w])

    data_st = torch.stack(data_sv, dim=1)
    return data_st


def FormOutput(intra_fea):
    b, n, c, h, w = intra_fea.shape
    angRes = int(sqrt(n+1))
    out = []
    kk = 0
    for u in range(angRes):
        buffer = []
        for v in range(angRes):
            buffer.append(intra_fea[:, kk, :, :, :])
            kk = kk+1
        buffer = torch.cat(buffer, 3)
        out.append(buffer)
    out = torch.cat(out, 2)

    return out

class get_loss(nn.Module):
    def __init__(self, args):
        super(get_loss, self).__init__()
        self.criterion_Loss = torch.nn.L1Loss()

    def forward(self, out, HR, degrade_info=None):
        loss = self.criterion_Loss(out['SR'], HR)

        return loss


def weights_init(m):
    pass


if __name__ == "__main__":
    from fvcore.nn import FlopCountAnalysis
    net = Net(5,4).cuda()
    input = torch.randn(1,1,160,160).cuda()
    flops = FlopCountAnalysis(net,input)
    print("Flops: ",flops.total())
    total = sum([param.nelement() for param in net.parameters()])
    print("Number of parameters: %.4fM" % (total/1e6))

