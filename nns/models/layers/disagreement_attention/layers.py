# -*- coding: utf-8 -*-
""" nns/models/layers/disagreement_attention/layers """

from typing import Optional

import torch
from torch.nn.modules.batchnorm import _BatchNorm
from gtorch_utils.nns.models.segmentation.unet.unet_parts import XConv


__all__ = ['AttentionConvBlock']


class AttentionConvBlock(torch.nn.Module):
    """
    Attention convolutional block for XAttentionUNet

    Usage:
         class UNet_3Plus_DA(UNet_3Plus):
            def __init__(self, ...):
                super().__init__(in_channels, out_channels)
                # intra-class DA skip-con down3 & gating signal down4 -> up1
                self.up1_with_da = AttentionConvBlock(
                    # attention to skip_connection
                    self.da_block_cls(self.filters[3], self.filters[4] // 2,
                                      # resample=torch.nn.Upsample(scale_factor=2, mode='bilinear'),
                                      **self.da_block_config),
                    2*self.filters[3],
                    self.filters[3] // factor,
                    batchnorm_cls=self.batchnorm_cls,
                )
    """

    def __init__(
            self, dablock_obj: torch.nn.Module, conv_in_channels: int, conv_out_channels: int, /, *,
            only_attention: bool = False, batchnorm_cls: Optional[_BatchNorm] = None,
            data_dimensions: int = 2, conv_layers: int = 2
    ):
        """
        Kwargs:
            dablock <torch.nn.Module>: Disagreement attention block instance.
                                       e.g ThresholdedDisagreementAttentionBlock(96, 96), 192, 96)
            conv_in_channels    <int>: conv_block in channels
            conv_out_channels   <int>: conv_block out channels
            only_attention     <bool>: If true returns only the attention; otherwise, returns the
                                       activation maps with attention. Default False
            batchnom_cls <_BatchNorm>: Batch normalization class. Default torch.nn.BatchNorm2d or
                                       torch.nn.BatchNorm3d
            data_dimensions <int>: Number of dimensions of the data. 2 for 2D [bacth, channel, height, width],
                                       3 for 3D [batch, channel, depth, height, width]. This argument will
                                       determine to use conv2d or conv3d.
                                       Default 2
            conv_layers         <int>: Number of convolutional layers to stack. Default 2
        """
        super().__init__()
        self.dattentionblock = dablock_obj
        self.identity = torch.nn.Identity()
        self.only_attention = only_attention
        self.batchnorm_cls = batchnorm_cls
        self.data_dimensions = data_dimensions
        self.conv_layers = conv_layers

        if self.batchnorm_cls is None:
            self.batchnorm_cls = torch.nn.BatchNorm2d if self.data_dimensions == 2 else torch.nn.BatchNorm3d

        assert isinstance(dablock_obj, torch.nn.Module), \
            'The provided dablock_obj is not an instance of torch.nn.Module'
        assert isinstance(conv_in_channels, int), type(conv_in_channels)
        assert isinstance(conv_out_channels, int), type(conv_out_channels)
        assert isinstance(only_attention, bool), type(only_attention)
        assert issubclass(batchnorm_cls, _BatchNorm), type(batchnorm_cls)
        assert self.data_dimensions in (2, 3), 'only 2d and 3d data is supported'
        assert isinstance(self.conv_layers, int), type(self.conv_layers)
        assert self.conv_layers > 0, self.conv_layers

        mode = 'bilinear' if self.data_dimensions == 2 else 'trilinear'

        self.up = torch.nn.Upsample(scale_factor=2, mode=mode, align_corners=False)
        self.conv_block = XConv(
            conv_in_channels, conv_out_channels, batchnorm_cls=self.batchnorm_cls,
            data_dimensions=self.data_dimensions, conv_layers=self.conv_layers
        )

    def forward(self, x: torch.Tensor, skip_connection: torch.Tensor, /, *, disable_attention: bool = False,
                central_gating: torch.Tensor = None):
        """
        Kwargs:
            x               <torch.Tensor>: activation/feature maps
            skip_connection <torch.Tensor>: skip connection containing activation/feature maps
            disable_attention       <bool>: When set to True, identity(x) will be used instead of
                                        dattentionblock(x, skip_connection). Default False
            central_gating  <torch.Tensor>: Gating calculated from the last Down layer (central part of UNet).
                                            Default None

        Returns:
            Union[torch.Tensor, None]
        """
        assert isinstance(x, torch.Tensor), type(x)
        assert isinstance(skip_connection, torch.Tensor), type(skip_connection)
        assert isinstance(disable_attention, bool), type(disable_attention)

        if central_gating is not None:
            assert isinstance(central_gating, torch.Tensor), type(central_gating)

        if disable_attention:
            if central_gating is not None:
                da, att = self.identity(central_gating), None
            else:
                da, att = self.identity(x), None
        else:
            if central_gating is not None:
                da, att = self.dattentionblock(skip_connection, central_gating)
            else:
                da, att = self.dattentionblock(skip_connection, x)

        if self.only_attention:
            return att

        decoder_x = self.up(x) if self.dattentionblock.upsample else x
        decoder_x = torch.cat((da, decoder_x), dim=1)
        x = self.conv_block(decoder_x)

        return x
