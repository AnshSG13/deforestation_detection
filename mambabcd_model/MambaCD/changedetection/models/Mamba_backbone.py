from MambaCD.classification.models.vmamba import VSSM, LayerNorm2d

import torch
import torch.nn as nn


class Backbone_VSSM(VSSM):
    def __init__(self, out_indices=(0, 1, 2, 3), pretrained=None, norm_layer='ln2d', cm_type="mlp", **kwargs):
        # norm_layer='ln'
        kwargs.update(norm_layer=norm_layer, cm_type=cm_type.lower())
        super().__init__(**kwargs)
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        for layer in self.layers:
            for block in layer.blocks:
                if hasattr(block, 'cm_type'):
                    block.cm_type = cm_type.lower()
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)        
        
        self.out_indices = out_indices
        for i in out_indices:
            layer = norm_layer(self.dims[i])
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, layer)

        del self.classifier
        self.load_pretrained(pretrained)

    def load_pretrained(self, ckpt=None, key="model"):
        if ckpt is None:
            return
        
        try:
            _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"), weights_only=True)
            print(f"Successfully load ckpt {ckpt}")
            
            # Handle the mismatch in the patch_embed.0.weight
            if 'patch_embed.0.weight' in _ckpt[key] and self.patch_embed[0].weight.shape[1] != _ckpt[key]['patch_embed.0.weight'].shape[1]:
                # Get the original weights
                original_weights = _ckpt[key]['patch_embed.0.weight']
                
                # Create a new tensor with the right shape
                new_weights = torch.zeros_like(self.patch_embed[0].weight)
                
                # Copy the original channels
                new_weights[:, :original_weights.shape[1], :, :] = original_weights
                
                # Replace the original weights in the checkpoint
                _ckpt[key]['patch_embed.0.weight'] = new_weights
            
            # Now load the modified state dict
            incompatibleKeys = self.load_state_dict(_ckpt[key], strict=False)
            print(incompatibleKeys)
        except Exception as e:
            print(f"Failed loading checkpoint from {ckpt}: {e}")

    def forward(self, x):
        def layer_forward(l, x):
            x = l.blocks(x)
            y = l.downsample(x)
            return x, y

        x = self.patch_embed(x)
        outs = []
        for i, layer in enumerate(self.layers):
            o, x = layer_forward(layer, x) # (B, H, W, C)
            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                out = norm_layer(o)
                if not self.channel_first:
                    out = out.permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        if len(self.out_indices) == 0:
            return x
        
        return outs

