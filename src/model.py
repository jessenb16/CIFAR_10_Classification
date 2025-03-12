'''
ResNet Implementation in PyTorch
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, skip_kernel_size=1, stride=1):
        """
        Basic Block for ResNet.
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of convolution kernel.
            skip_kernel_size (int): Kernel size for skip connection.
            stride (int): Stride for the first convolution.
        """
        super(BasicBlock, self).__init__()

        padding = kernel_size // 2

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection: Adjust dimensions if needed
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=skip_kernel_size, stride=stride, 
                          padding=(skip_kernel_size - 1) // 2, bias=False), 
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Residual connection
        return F.relu(out)
    

class ResNet(nn.Module):
    """
    ResNet Model with customizable parameters.
    Args:
        block (nn.Module): Block type (e.g., BasicBlock).
        blocks_per_layer (list): Number of blocks in each layer.
        channels_per_layer (list): Number of channels in each layer.
        kernel_size (int): Size of convolution kernel.
        skip_kernel_size (int): Kernel size for skip connection.
        pool_size (int): Pool size.
        num_classes (int): Number of output classes.
    """
    def __init__(self, block, blocks_per_layer, channels_per_layer, kernel_size=3, skip_kernel_size=1, pool_size=4, num_classes=10):
        super(ResNet, self).__init__()

        assert len(blocks_per_layer) == len(channels_per_layer), "blocks_per_layer and channels_per_layer must have the same length"

        self.in_channels = channels_per_layer[0]
        self.pool_size = pool_size
        
        # Initial Convolutional Layer
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)

        # Create Residual Layers
        self.residual_layers = self._make_layers(block, blocks_per_layer, channels_per_layer, kernel_size, skip_kernel_size)

        # Output Layer
        self.fc = nn.Linear(channels_per_layer[-1], num_classes)

    # Creates a single residual layer
    def _make_layer(self, block, out_channels, num_blocks, kernel_size, skip_kernel_size, stride=1):
        layers = []
        for i in range(num_blocks):
            layers.append(block(self.in_channels, out_channels, kernel_size, skip_kernel_size, stride if i == 0 else 1))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    # Creates all residual layers
    def _make_layers(self, block, blocks_per_layer, channels_per_layer, kernel_size, skip_kernel_size):
        layers = []
        for i in range(len(blocks_per_layer)):
            layers.append(self._make_layer(block, channels_per_layer[i], blocks_per_layer[i], kernel_size, skip_kernel_size, stride=1 if i == 0 else 2))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # Initial Conv Layer
        x = self.residual_layers(x)  # Residual layers
        x = F.avg_pool2d(x, self.pool_size)  # Adaptive Pooling
        x = x.view(x.size(0), -1)  # Flatten
        return self.fc(x)  # Fully Connected Layer
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Corrected model creator function (OUTSIDE the class)
def create_model(blocks_per_layer, channels_per_layer, kernel_size=3, skip_kernel_size=1, pool_size=4, num_classes=10):
    return ResNet(BasicBlock, blocks_per_layer, channels_per_layer, kernel_size, skip_kernel_size, pool_size, num_classes)
    
    

# Example Usage
if __name__ == "__main__":
    try:
        model = ResNet(
            block=BasicBlock,
            blocks_per_layer=[2, 2, 2, 2],       # ResNet-18 style
            channels_per_layer=[64, 128, 256, 512],  # Standard ResNet filters
            kernel_size=3,                       # Standard 3x3 convs
            skip_kernel_size=1,                  # 1x1 convs for skip connections
            pool_size=1,                         # Global average pooling
        )

        # Check output shape
        x = torch.randn(1, 3, 32, 32)
        y = model(x)
        print(f"Output Shape: {y.shape}")  # Should be (1, num_classes)

        # Check number of parameters
        num_params = model.count_parameters()
        print(f"Number of Trainable Parameters: {num_params}")
    except Exception as e:
        print(f"Error: {e}")

    



    
    

       
