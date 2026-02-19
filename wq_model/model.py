"""Model architecture for water quality regression"""

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

from .config import Config


class MultiModalResNetRegressor(nn.Module):
    """Multi-modal model combining image and tabular data for water quality regression"""
    
    def __init__(
        self,
        tab_dim: int,
        out_dim: int,
        index_bands_count: int,
        use_images: bool = True,
        use_tabular: bool = True,
        tab_dropout: float = 0.3,
        head_dropout: float = 0.4,
        include_landcoverage_channel: bool = Config.INCLUDE_LANDCOVERAGE_CHANNEL,
    ):
        """
        Args:
            tab_dim: Dimension of tabular features
            out_dim: Number of output targets
            index_bands_count: Number of input image bands
            use_images: Whether to use image branch
            use_tabular: Whether to use tabular branch
            tab_dropout: Dropout rate for tabular branch
            head_dropout: Dropout rate for head layers
            include_landcoverage_channel: Whether to append land coverage as one extra channel
        """
        super().__init__()
        self.use_images = use_images
        self.use_tabular = use_tabular
        self.include_landcoverage_channel = include_landcoverage_channel

        if use_images:
            # Use smaller pretrained MobileNetV3-Small
            self.cnn = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
            # Modify first conv to accept index bands instead of RGB
            in_channels = index_bands_count + int(include_landcoverage_channel)
            self.cnn.features[0][0] = nn.Conv2d(
                in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False
            )
            # Remove classifier, keep only features
            self.cnn.classifier = nn.Identity()
            # MobileNetV3-Small outputs 576 features
            self.img_proj = nn.Sequential(
                nn.Linear(576, 192),
                nn.ReLU(inplace=True),
                nn.Dropout(head_dropout),
            )

        if use_tabular:
            self.tab_branch = nn.Sequential(
                nn.Linear(tab_dim, 64),
                nn.LayerNorm(64),
                nn.ReLU(inplace=True),
                nn.Dropout(tab_dropout),
                nn.Linear(64, 32),
                nn.ReLU(inplace=True),
                nn.Dropout(tab_dropout),
            )

        head_in = (192 if use_images else 0) + (32 if use_tabular else 0)
        
        # Shared trunk
        self.shared_trunk = nn.Sequential(
            nn.Linear(head_in, 96),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
        )
        
        # Separate head for each target
        self.heads = nn.ModuleList([
            nn.Linear(96, 1) for _ in range(out_dim)
        ])

    def forward(self, x_img, x_tab):
        """
        Forward pass
        
        Args:
            x_img: Image tensor (B, C, H, W)
            x_tab: Tabular features (B, D)
            
        Returns:
            Predictions for all targets (B, num_targets)
        """
        parts = []
        if self.use_images:
            parts.append(self.img_proj(self.cnn(x_img)))
        if self.use_tabular:
            parts.append(self.tab_branch(x_tab))
        z = torch.cat(parts, dim=1)
        
        # Pass through shared trunk
        trunk_out = self.shared_trunk(z)
        
        # Apply each head and concatenate
        outputs = [head(trunk_out) for head in self.heads]
        return torch.cat(outputs, dim=1)
