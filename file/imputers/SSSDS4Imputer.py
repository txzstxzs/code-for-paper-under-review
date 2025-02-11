import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.util import calc_diffusion_step_embedding
from imputers.S4Model import S4Layer


def swish(x):
    return x * torch.sigmoid(x)


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):       # 14 ->256  
        super(Conv, self).__init__()
        self.padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=self.padding)
        self.conv = nn.utils.weight_norm(self.conv)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        out = self.conv(x)
        return out
    
    
class ZeroConv1d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ZeroConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=1, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

    def forward(self, x):
        out = self.conv(x)
        return out


class Residual_block(nn.Module):
    def __init__(self, res_channels, skip_channels, 
                 diffusion_step_embed_dim_out, in_channels,
                s4_lmax,
                s4_d_state,
                s4_dropout,
                s4_bidirectional,
                s4_layernorm):
        super(Residual_block, self).__init__()
        
        self.res_channels = res_channels

        self.fc_t = nn.Linear(diffusion_step_embed_dim_out, self.res_channels)
        
        self.S41 = S4Layer(features=2*self.res_channels, 
                          lmax=s4_lmax,
                          N=s4_d_state,
                          dropout=s4_dropout,
                          bidirectional=s4_bidirectional,
                          layer_norm=s4_layernorm)

        self.conv_layer = Conv(self.res_channels, 2 * self.res_channels, kernel_size=3)

        self.S42 = S4Layer(features=2*self.res_channels,          # 若像csdi一样用作特征提取的话这里改为时间步长度 如sp500应为30      
                          lmax=s4_lmax,
                          N=s4_d_state,
                          dropout=s4_dropout,
                          bidirectional=s4_bidirectional,
                          layer_norm=s4_layernorm)
        
        self.cond_conv = Conv(2*in_channels, 2*self.res_channels, kernel_size=1)  

        self.res_conv = nn.Conv1d(res_channels, res_channels, kernel_size=1)
        self.res_conv = nn.utils.weight_norm(self.res_conv)
        nn.init.kaiming_normal_(self.res_conv.weight)

        
        self.skip_conv = nn.Conv1d(res_channels, skip_channels, kernel_size=1)
        self.skip_conv = nn.utils.weight_norm(self.skip_conv)
        nn.init.kaiming_normal_(self.skip_conv.weight)

    def forward(self, input_data):      
        x, cond, diffusion_step_embed = input_data
        h = x                             # [50, 256, 100]
        B, C, L = x.shape
        assert C == self.res_channels                      
                 
        part_t = self.fc_t(diffusion_step_embed)      
        part_t = part_t.view([B, self.res_channels, 1])  
        
        h = h + part_t                     
        
        
        h = self.conv_layer(h)                 # [50, 256, 100] ->[50, 512, 100]
        h = self.S41(h.permute(2,0,1)).permute(1,2,0)  # [50, 512, 100] -> [100, 50, 512] -> [50, 512, 100]
        
        

        out = torch.tanh( h[:,:self.res_channels,:] ) * torch.sigmoid( h[:,self.res_channels:,:] ) 

        res = self.res_conv(out)           
        
        assert x.shape == res.shape
        skip = self.skip_conv(out)         

#         return x + res, skip
        return (x + res) * math.sqrt(0.5), skip  

    

class Residual_group(nn.Module):
    def __init__(self, res_channels, skip_channels, num_res_layers,      # 256 256 36
                 diffusion_step_embed_dim_in, 
                 diffusion_step_embed_dim_mid,
                 diffusion_step_embed_dim_out,
                 in_channels,                            # 14
                 s4_lmax,
                 s4_d_state,
                 s4_dropout,
                 s4_bidirectional,
                 s4_layernorm):
        super(Residual_group, self).__init__()
        self.num_res_layers = num_res_layers
        self.diffusion_step_embed_dim_in = diffusion_step_embed_dim_in

        self.fc_t1 = nn.Linear(diffusion_step_embed_dim_in, diffusion_step_embed_dim_mid)    # 128 512
        self.fc_t2 = nn.Linear(diffusion_step_embed_dim_mid, diffusion_step_embed_dim_out)    # 256 512
        
        self.residual_blocks = nn.ModuleList()
        
        for n in range(self.num_res_layers):                              #  36个残差模块
            self.residual_blocks.append(Residual_block(res_channels, skip_channels, 
                                                       diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
                                                       in_channels=in_channels,
                                                       s4_lmax=s4_lmax,
                                                       s4_d_state=s4_d_state,
                                                       s4_dropout=s4_dropout,
                                                       s4_bidirectional=s4_bidirectional,
                                                       s4_layernorm=s4_layernorm))

            
    def forward(self, input_data):
        noise, conditional, diffusion_steps = input_data    # 扩维后的加噪数据  已有值和掩码的拼接  扩散步
                                                 
                                             # 扩散步嵌入  嵌入 -> fc+swish -> fc+swish
        diffusion_step_embed = calc_diffusion_step_embedding(diffusion_steps, self.diffusion_step_embed_dim_in)   # 128
        diffusion_step_embed = swish(self.fc_t1(diffusion_step_embed))                              # 512
        diffusion_step_embed = swish(self.fc_t2(diffusion_step_embed))                              # 512

        h = noise
        skip = 0
        
        
        for n in range(self.num_res_layers):
            h, skip_n = self.residual_blocks[n]((h, conditional, diffusion_step_embed))  
            skip += skip_n                                           

        return skip * math.sqrt(1.0 / self.num_res_layers)     


class SSSDS4Imputer(nn.Module):
    def __init__(self, in_channels, res_channels, skip_channels, out_channels,    # 14 256 256 14
                 num_res_layers,                               # 36
                 diffusion_step_embed_dim_in,                       # 128
                 diffusion_step_embed_dim_mid,                      # 512
                 diffusion_step_embed_dim_out,                      # 512
                 s4_lmax,                                    # 100
                 s4_d_state,                                  # 64
                 s4_dropout,                                  # 0
                 s4_bidirectional,                              # 1
                 s4_layernorm):                                # 1
        super(SSSDS4Imputer, self).__init__()

        self.init_conv = nn.Sequential( Conv(in_channels, res_channels, kernel_size=1), nn.ReLU())   # 14->256
        
        self.residual_layer = Residual_group( res_channels=res_channels, 
                                             skip_channels=skip_channels, 
                                             num_res_layers=num_res_layers, 
                                             diffusion_step_embed_dim_in=diffusion_step_embed_dim_in,
                                             diffusion_step_embed_dim_mid=diffusion_step_embed_dim_mid,
                                             diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
                                             in_channels=in_channels,
                                             s4_lmax=s4_lmax,
                                             s4_d_state=s4_d_state,
                                             s4_dropout=s4_dropout,
                                             s4_bidirectional=s4_bidirectional,
                                             s4_layernorm=s4_layernorm)
        
        self.final_conv = nn.Sequential(Conv(skip_channels, skip_channels, kernel_size=1),
                                        nn.ReLU(),
                                        ZeroConv1d(skip_channels, out_channels))    # 256->14

    def forward(self, input_data):
                                                 
        noise, conditional, mask, diffusion_steps = input_data     
            
        x = noise
        x = self.init_conv(x)                             
        x = self.residual_layer((x, conditional, diffusion_steps))    
        y = self.final_conv(x)                            

        return y

    
    
    
    