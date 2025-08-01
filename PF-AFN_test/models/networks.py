import torch
import torch.nn as nn
import torch.nn.parallel
import os

class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        use_bias = norm_layer == nn.InstanceNorm2d

        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        uprelu = nn.ReLU(True)
        if norm_layer != None:
            downnorm = norm_layer(inner_nc)
            upnorm = norm_layer(outer_nc)

        if outermost:
            upsample = nn.Upsample(scale_factor=2, mode='bilinear')
            upconv = nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            down = [downconv]
            up = [uprelu, upsample, upconv]
            model = down + [submodule] + up
        elif innermost:
            upsample = nn.Upsample(scale_factor=2, mode='bilinear')
            upconv = nn.Conv2d(inner_nc, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            if norm_layer == None:
                up = [uprelu, upsample, upconv]
            else:
                up = [uprelu, upsample, upconv, upnorm]
            model = down + up
        else:
            upsample = nn.Upsample(scale_factor=2, mode='bilinear')
            upconv = nn.Conv2d(inner_nc*2, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            if norm_layer == None:
                down = [downrelu, downconv]
                up = [uprelu, upsample, upconv]
            else:
                down = [downrelu, downconv, downnorm]
                up = [uprelu, upsample, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)

class ResidualBlock(nn.Module):
    def __init__(self, in_features=64, norm_layer=nn.BatchNorm2d):
        super(ResidualBlock, self).__init__()
        self.relu = nn.ReLU(True)
        if norm_layer == None:
            self.block = nn.Sequential(
                nn.Conv2d(in_features, in_features, 3, 1, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_features, in_features, 3, 1, 1, bias=False),
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_features, in_features, 3, 1, 1, bias=False),
                norm_layer(in_features),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_features, in_features, 3, 1, 1, bias=False),
                norm_layer(in_features)
            )

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out

class ResUnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(ResUnetGenerator, self).__init__()
        unet_block = ResUnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)

        for i in range(num_downs - 5):
            unet_block = ResUnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = ResUnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = ResUnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = ResUnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = ResUnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)


class ResUnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(ResUnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        use_bias = norm_layer == nn.InstanceNorm2d

        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=3,
                             stride=2, padding=1, bias=use_bias)

        res_downconv = [ResidualBlock(inner_nc, norm_layer), ResidualBlock(inner_nc, norm_layer)]
        res_upconv = [ResidualBlock(outer_nc, norm_layer), ResidualBlock(outer_nc, norm_layer)]
        
        downrelu = nn.ReLU(True)
        uprelu = nn.ReLU(True)
        if norm_layer != None:
            downnorm = norm_layer(inner_nc)
            upnorm = norm_layer(outer_nc)

        if outermost:
            upsample = nn.Upsample(scale_factor=2, mode='nearest')
            upconv = nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            down = [downconv, downrelu] + res_downconv
            up = [upsample, upconv]
            model = down + [submodule] + up
        elif innermost:
            upsample = nn.Upsample(scale_factor=2, mode='nearest')
            upconv = nn.Conv2d(inner_nc, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            down = [downconv, downrelu] + res_downconv
            if norm_layer == None:
                up = [upsample, upconv, uprelu] + res_upconv
            else:
                up = [upsample, upconv, upnorm, uprelu] + res_upconv
            model = down + up
        else:
            upsample = nn.Upsample(scale_factor=2, mode='nearest')
            upconv = nn.Conv2d(inner_nc*2, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            if norm_layer == None:
                down = [downconv, downrelu] + res_downconv
                up = [upsample, upconv, uprelu] + res_upconv
            else:
                down = [downconv, downnorm, downrelu] + res_downconv
                up = [upsample, upconv, upnorm, uprelu] + res_upconv

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


# def save_checkpoint(model, save_path):
#     if not os.path.exists(os.path.dirname(save_path)):
#         os.makedirs(os.path.dirname(save_path))
#     torch.save(model.state_dict(), save_path)
def save_checkpoint(model, optimizer, scheduler, epoch, step, save_path):
    """
    Saves a comprehensive checkpoint including the scheduler.
    """
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    # The model state_dict is already handled by the calling script
    # because of the DDP wrapper (.module attribute).
    # So, 'model' here will actually be the state_dict.
    state = {
        'epoch': epoch,
        'step': step,
        'state_dict': model,  # 'model' is now the state_dict dictionary
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() # Add the scheduler state
    }
    torch.save(state, save_path)

# def load_checkpoint(model, checkpoint_path):

#     if not os.path.exists(checkpoint_path):
#         print('No checkpoint!')
#         return

#     checkpoint = torch.load(checkpoint_path)
#     checkpoint_new = model.state_dict()
#     for param in checkpoint_new:
#         checkpoint_new[param] = checkpoint[param]

#     model.load_state_dict(checkpoint_new)
def load_checkpoint(model, checkpoint_path):
    """
    Loads a checkpoint into the model.

    This function is designed to be robust and can handle checkpoints saved
    with or without a nested 'state_dict' key. It is suitable for inference.
    """
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at '{checkpoint_path}'")
        return

    # Load the checkpoint onto the CPU first to avoid GPU memory issues
    # and to examine its structure. We'll move the model to the GPU later.
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # --- Identify the actual state dictionary ---
    # Check for common keys where the state_dict might be stored
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        # If no common key is found, assume the checkpoint IS the state_dict
        state_dict = checkpoint
        print("Warning: Checkpoint does not contain a 'state_dict' key. "
              "Assuming the file is a raw state_dict.")

    # --- Clean the keys (e.g., remove 'module.' prefix from DDP) ---
    # This makes the function work for models saved with or without DataParallel/DDP
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            # Remove the 'module.' prefix
            cleaned_state_dict[k[7:]] = v
        else:
            cleaned_state_dict[k] = v
    
    # --- Load the state_dict into the model ---
    # `strict=False` is useful for transfer learning or if some layers are intentionally different.
    # For evaluation, `strict=True` is better to ensure the architectures match perfectly.
    # If you still get errors, `strict=False` can help you debug by printing missing/unexpected keys.
    incompatible_keys = model.load_state_dict(cleaned_state_dict, strict=False)
    
    if incompatible_keys.missing_keys:
        print(f"Warning: Missing keys in state_dict: {incompatible_keys.missing_keys}")
    if incompatible_keys.unexpected_keys:
        print(f"Warning: Unexpected keys in state_dict: {incompatible_keys.unexpected_keys}")

    print(f"Successfully loaded checkpoint from {checkpoint_path}")



