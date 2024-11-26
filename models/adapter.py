# --------------------------------------------------------
# References:
# https://github.com/jxhe/unify-parameter-efficient-tuning
# --------------------------------------------------------

import math
import torch
import torch.nn as nn
from torch import nn, Tensor
from typing import Callable, List, Any, Tuple, Dict
import torch.nn.functional as F

class CLIP_Adapter(nn.Module):
    def __init__(
        self,
        skip_connect=False,
    ) -> None:
        super().__init__()
        self.skip_connect=skip_connect

        self.down = nn.Linear(512, 32) 
        self.up = nn.Linear(32, 512)


        with torch.no_grad():
                nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
               
                nn.init.zeros_(self.up.weight)
              
                nn.init.zeros_(self.down.bias)
           
                nn.init.zeros_(self.up.bias)
             
                
    def forward(self, x):
        x0 = self.down(x)
    
        x = F.relu(x0, inplace=True)
        x_up = self.up(x)
       
        return x_up

class Bert_Adapter(nn.Module):
    def __init__(
        self,
        skip_connect=False,
    ) -> None:
        super().__init__()
        self.skip_connect=skip_connect

        self.down = nn.Linear(768, 32) 
        self.up = nn.Linear(32, 768)


        with torch.no_grad():
                nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
               
                nn.init.zeros_(self.up.weight)
              
                nn.init.zeros_(self.down.bias)
           
                nn.init.zeros_(self.up.bias)
             
                
    def forward(self, x):
        x0 = self.down(x)
    
        x = F.relu(x0, inplace=True)
        x_up = self.up(x)
       
        return x_up
    
class Text_Adapter(nn.Module):
    def __init__(
        self,
        skip_connect=False,
    ) -> None:
        super().__init__()
        self.skip_connect=skip_connect

        self.down = nn.Linear(1280, 768) 
        # self.up = nn.Linear(32, 768)


        with torch.no_grad():
                nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
               
                # nn.init.zeros_(self.up.weight)
              
                nn.init.zeros_(self.down.bias)
           
                # nn.init.zeros_(self.up.bias)
             
                
    def forward(self, x):
        x0 = self.down(x)
    
        # x = F.relu(x0, inplace=True)
        # x_up = self.up(x)
       
        return x0

class MOE_Adapter(nn.Module):
    def __init__(
        self,
        skip_connect=False,
    ) -> None:
        super().__init__()
        self.skip_connect=skip_connect
     
        # self.down1 = nn.Linear(768, 256) 
        # # self.non_linear_func = nn.ReLU()
        # self.down2 = nn.Linear(256, 128)
        # self.down3 = nn.Linear(384, 256)

        # self.up1 = nn.Linear(128, 256)
        # self.up2 = nn.Linear(256, 768)

        self.down1 = nn.Linear(768, 32) 
        # self.non_linear_func = nn.ReLU()
        self.down2 = nn.Linear(768, 32)
        self.down3 = nn.Linear(768, 32)

        self.up1 = nn.Linear(32, 768)
        self.up2 = nn.Linear(32, 768)
        self.up3 = nn.Linear(32, 768)
       


        with torch.no_grad():
                nn.init.kaiming_uniform_(self.down1.weight, a=math.sqrt(5))
                nn.init.kaiming_uniform_(self.down2.weight, a=math.sqrt(5))
                nn.init.kaiming_uniform_(self.down3.weight, a=math.sqrt(5))

                # nn.init.zeros_(self.up1.weight)
                nn.init.zeros_(self.up1.weight)
                nn.init.zeros_(self.up2.weight)
                nn.init.zeros_(self.up3.weight)
               
             
                nn.init.zeros_(self.down1.bias)
                nn.init.zeros_(self.down2.bias)
                nn.init.zeros_(self.down3.bias)
           
                nn.init.zeros_(self.up1.bias)
                nn.init.zeros_(self.up2.bias)
                nn.init.zeros_(self.up3.bias)
                

    def forward(self, x):
        x0 = self.down1(x)
        x_1 = self.down2(x)
        x_2 = self.down3(x)

        # x_add = 0.6*x0+0.3*x_1+0.1*x_2
        x_add = 0.5*x0+0.4*x_1+0.1*x_2
        x_add = F.relu(x_add, inplace=True)

        # x_0 = F.relu(x0, inplace=True)
        # x_1 = F.relu(x_1, inplace=True)
        # x_2 = F.relu(x_2, inplace=True)
       
        x_up_1 = self.up1(x_add)
        x_up_2 = self.up2(x_add)
        x_up_3 = self.up3(x_add)

        x_up = 0.5*x_up_1 + 0.4*x_up_2 + 0.1*x_up_3
       
        return x_up

class MulConvAdapter(nn.Module):
    def __init__(
        self,
        fc_in_channels: int,
        in_channels: int,
        ch1x1: int,
        ch3x3red: int,
        ch3x3: int,
        skip_connect=False,
    ) -> None:
        super().__init__()
        self.skip_connect=skip_connect
        conv_block = BasicConv2d
        self.branch1 = conv_block(288, ch1x1, kernel_size=1)  # 384,192

        self.branch2 = nn.Sequential(
            conv_block(288, ch3x3red, kernel_size=1),  # 384,24
            conv_block(ch3x3red, ch3x3, kernel_size=3, padding=1) # 24,96
        )

        # self.branch3 = nn.Sequential(
        #     conv_block(in_channels, ch5x5red, kernel_size=1),   # 384,24
        #     conv_block(ch5x5red, ch5x5, kernel_size=5, padding=2),  # 24,96
        # )

        self.D_fc1 = nn.Linear(fc_in_channels, 288)  # 768,384
        self.D_fc2 = nn.Linear(288, fc_in_channels)  # 384,768

    def forward(self, x: Tensor) -> List[Tensor]:
        x0 = self.D_fc1(x)
        B,P,D = x0.shape
        W = H = int(math.sqrt(P-1))

        x0 = F.relu(x0, inplace=True)
        
        xs = x0[:,1:,:]
        xs = xs.reshape(B,W,H,D).permute(0,3,1,2)
        branch1 = self.branch1(xs)
        branch2 = self.branch2(xs)
        # branch3 = self.branch3(xs)
        # outputs = [branch1, branch2, branch3]
        outputs = [branch1, branch2]
        outputs = torch.cat(outputs,dim=1)
        outputs = outputs.reshape(B,D,W*H).permute(0,2,1)
        clstoken =  x0[:,0:1,:]
        outputs = torch.cat([clstoken,outputs],dim=1)

        outputs += x0

        outputs = self.D_fc2(outputs)
        if self.skip_connect:
            outputs+=x
        return outputs


# class MulConvAdapter(nn.Module):
#     def __init__(
#         self,
#         fc_in_channels: int,
#         in_channels: int,
#         ch1x1: int,
#         ch3x3red: int,
#         ch3x3: int,
#         ch5x5red: int,
#         ch5x5: int,
#         skip_connect=False,
#     ) -> None:
#         super().__init__()
#         self.skip_connect=skip_connect
#         conv_block = BasicConv2d
#         self.branch1 = conv_block(in_channels, ch1x1, kernel_size=1)  # 384,192

#         self.branch2 = nn.Sequential(
#             conv_block(in_channels, ch3x3red, kernel_size=1),  # 384,24
#             conv_block(ch3x3red, ch3x3, kernel_size=3, padding=1) # 24,96
#         )

#         self.branch3 = nn.Sequential(
#             conv_block(in_channels, ch5x5red, kernel_size=1),   # 384,24
#             conv_block(ch5x5red, ch5x5, kernel_size=5, padding=2),  # 24,96
#         )

#         self.D_fc1 = nn.Linear(fc_in_channels, in_channels)  # 768,384
#         self.D_fc2 = nn.Linear(in_channels, fc_in_channels)  # 384,768

#     def forward(self, x: Tensor) -> List[Tensor]:
#         x0 = self.D_fc1(x)
#         B,P,D = x0.shape
#         W = H = int(math.sqrt(P-1))

#         x0 = F.relu(x0, inplace=True)
        
#         xs = x0[:,1:,:]
#         xs = xs.reshape(B,W,H,D).permute(0,3,1,2)
#         branch1 = self.branch1(xs)
#         branch2 = self.branch2(xs)
#         branch3 = self.branch3(xs)
#         outputs = [branch1, branch2, branch3]
#         outputs = torch.cat(outputs,dim=1)
#         outputs = outputs.reshape(B,D,W*H).permute(0,2,1)
#         clstoken =  x0[:,0:1,:]
#         outputs = torch.cat([clstoken,outputs],dim=1)

#         outputs += x0

#         outputs = self.D_fc2(outputs)
#         if self.skip_connect:
#             outputs+=x
#         return outputs

class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=True, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
