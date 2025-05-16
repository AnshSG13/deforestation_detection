import torch
import torch.nn.functional as F

import torch
import torch.nn as nn
from MambaCD.changedetection.models.Mamba_backbone import Backbone_VSSM
from MambaCD.classification.models.vmamba import VSSM, LayerNorm2d, VSSBlock, Permute
import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict
from MambaCD.changedetection.models.ChangeDecoder import ChangeDecoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count

#fourier block code
# class FourierBlock(nn.Module):
#     def __init__(self, in_channels, kernel_size=3):
#         super().__init__()
#         padding = kernel_size // 2
#         self.conv_real = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding)
#         self.conv_imag = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding)

#     def forward(self, x):
#         ffted = torch.fft.fft2(x, norm='ortho')
#         real = ffted.real
#         imag = ffted.imag

#         real_out = self.conv_real(real)
#         imag_out = self.conv_imag(imag)
#         ffted_out = torch.complex(real_out, imag_out)
#         iffted = torch.fft.ifft2(ffted_out, norm='ortho')
#         return iffted.real

# Modified STMambaBCD using FourierBlock
class STMambaBCD(nn.Module):
    def __init__(self, pretrained, **kwargs):
        super(STMambaBCD, self).__init__()
        # Instantiate the encoder (assumed to return a list of features)
        self.encoder = Backbone_VSSM(out_indices=(0, 1, 2, 3), pretrained=pretrained, cm_type="mlp", **kwargs)
        
    
        # Setup norm and activation layers as before
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        _ACTLAYERS = dict(
            silu=nn.SiLU, 
            gelu=nn.GELU, 
            relu=nn.ReLU, 
            sigmoid=nn.Sigmoid,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(kwargs['norm_layer'].lower(), None)        
        ssm_act_layer: nn.Module = _ACTLAYERS.get(kwargs['ssm_act_layer'].lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(kwargs['mlp_act_layer'].lower(), None)
        
        # Remove these keys from kwargs for decoder initialization
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['norm_layer', 'ssm_act_layer', 'mlp_act_layer']}
        self.decoder = ChangeDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            cm_type="mlp",
            **clean_kwargs
        )
        
        self.main_clf = nn.Conv2d(in_channels=128, out_channels=2, kernel_size=1)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, pre_data, post_data, inference=False):
        # Encoder processing: obtain feature maps from both inputs
        pre_features = self.encoder(pre_data)
        post_features = self.encoder(post_data)

        # Apply the FourierBlock to each feature map to enhance frequency domain features
        # pre_features_enhanced = [
        #     fourier_block(feat) for fourier_block, feat in zip(self.fourier_blocks, pre_features)
        # ]
        # post_features_enhanced = [
        #     fourier_block(feat) for fourier_block, feat in zip(self.fourier_blocks, post_features)
        # ]
        
        # Pass the enhanced features into the decoder
        output = self.decoder(pre_features, post_features)    
        output = self.main_clf(output)
        output = F.interpolate(output, size=pre_data.size()[-2:], mode='bilinear')
        if inference:
            output = F.softmax(output, dim=1)
        return output
