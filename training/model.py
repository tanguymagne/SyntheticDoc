import torch
import torch.nn as nn
import torch.nn.functional as F

BM_HEIGHT = 121
BM_WIDTH = 177


def conv3x3(in_channels, out_channels, kernel_size, stride=1):
    """Plain convolution with 'same' padding (name kept from the original UVDoc code)."""
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2
    )


def dilated_conv_bn_act(in_channels, out_channels, act_fn, BatchNorm, dilation):
    """3x3 dilated conv + norm + activation, used as the building block of the bridge."""
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            bias=False,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
        ),
        BatchNorm(out_channels),
        act_fn,
    )


def dilated_conv(in_channels, out_channels, kernel_size, dilation, stride=1):
    """Dilated convolution with 'same' padding, so the spatial size is preserved."""
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation * (kernel_size // 2),
            dilation=dilation,
        )
    )


class ResidualBlockWithDilation(nn.Module):
    """Residual block using plain convs when downsampling, dilated ones otherwise."""

    def __init__(
        self,
        in_channels,
        out_channels,
        BatchNorm,
        kernel_size,
        stride=1,
        downsample=None,
        is_top=False,
    ):
        super().__init__()
        self.downsample = downsample
        if stride != 1 or is_top:
            self.conv1 = conv3x3(in_channels, out_channels, kernel_size, stride)
            self.conv2 = conv3x3(out_channels, out_channels, kernel_size)
        else:
            self.conv1 = dilated_conv(in_channels, out_channels, kernel_size, dilation=3)
            self.conv2 = dilated_conv(out_channels, out_channels, kernel_size, dilation=3)
        self.bn1 = BatchNorm(out_channels)
        self.bn2 = BatchNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.downsample(x) if self.downsample is not None else x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


def _make_resnet_layer(
    in_channels,
    out_channels,
    block_nums,
    BatchNorm,
    kernel_size,
    stride=1,
    block=ResidualBlockWithDilation,
):
    """Stack block_nums residual blocks, the first one handling the stride / channel change."""
    downsample = None
    if stride != 1 or in_channels != out_channels:
        downsample = nn.Sequential(
            conv3x3(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
            BatchNorm(out_channels),
        )
    layers = [
        block(in_channels, out_channels, BatchNorm, kernel_size, stride, downsample, is_top=True)
    ]
    for _ in range(1, block_nums):
        layers.append(block(out_channels, out_channels, BatchNorm, kernel_size, is_top=False))
    return nn.Sequential(*layers)


def _shadow_dec_block(in_ch, skip_ch, out_ch, kernel_size, BatchNorm):
    """Upsample + fuse skip + refine conv block for the shadow decoder."""
    return nn.Sequential(
        nn.Conv2d(
            in_ch + skip_ch,
            out_ch,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            padding_mode="reflect",
            bias=False,
        ),
        BatchNorm(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        BatchNorm(out_ch),
        nn.ReLU(inplace=True),
    )


class UVDocNet(nn.Module):
    """
    UVDocnet backbone adapted for the training pipeline.

    Outputs:
      bm_map     : (B, 2, bm_out_size[0], bm_out_size[1])  normalised [-1, 1]
      shadow_map : (B, 3, output_size[0], output_size[1])   full-resolution shadow map

    The shadow head is a 4-stage U-Net decoder that fuses encoder skip connections
    to eliminate the sawtooth / blockiness artefacts from naive single-step upsampling.
    The BM head is unchanged from the original UVDoc paper.
    """

    def __init__(self, num_filter=32, kernel_size=5, bm_out_size=(121, 177), output_size=None):
        super().__init__()
        self.bm_out_size = bm_out_size
        self.output_size = output_size

        map_num = [1, 2, 4, 8, 16]
        BatchNorm = nn.BatchNorm2d
        act_fn = nn.ReLU(inplace=True)
        nf = num_filter

        # ── Encoder ──────────────────────────────────────────────────────────
        # Split resnet_head into two separate convs so we can capture skip s1.
        # s1: (B, nf, H/2, W/2)
        self.enc_s1 = nn.Sequential(
            nn.Conv2d(
                3,
                nf * map_num[0],
                bias=False,
                kernel_size=kernel_size,
                stride=2,
                padding=kernel_size // 2,
            ),
            BatchNorm(nf * map_num[0]),
            act_fn,
        )
        # s2: (B, nf, H/4, W/4)
        self.enc_s2 = nn.Sequential(
            nn.Conv2d(
                nf * map_num[0],
                nf * map_num[0],
                bias=False,
                kernel_size=kernel_size,
                stride=2,
                padding=kernel_size // 2,
            ),
            BatchNorm(nf * map_num[0]),
            act_fn,
        )

        # ResnetStraight layers exposed individually so we can grab skips.
        # layer1: stride=1 → stays at H/4,  ch: nf→nf   (s3)
        # layer2: stride=2 → H/8,           ch: nf→2nf  (s4)
        # layer3: stride=2 → H/16,          ch: 2nf→4nf (fed into bridge)
        self.enc_s3 = _make_resnet_layer(
            nf * map_num[0], nf * map_num[0], 3, BatchNorm, kernel_size, stride=1
        )
        self.enc_s4 = _make_resnet_layer(
            nf * map_num[0], nf * map_num[1], 4, BatchNorm, kernel_size, stride=2
        )
        self.enc_s5 = _make_resnet_layer(
            nf * map_num[1], nf * map_num[2], 6, BatchNorm, kernel_size, stride=2
        )

        # ── Bridge (unchanged from original) ─────────────────────────────────
        bridge_ch = nf * map_num[2]  # 4*nf
        self.bridge_1 = dilated_conv_bn_act(bridge_ch, bridge_ch, act_fn, BatchNorm, dilation=1)
        self.bridge_2 = dilated_conv_bn_act(bridge_ch, bridge_ch, act_fn, BatchNorm, dilation=2)
        self.bridge_3 = dilated_conv_bn_act(bridge_ch, bridge_ch, act_fn, BatchNorm, dilation=5)
        self.bridge_4 = nn.Sequential(
            *[dilated_conv_bn_act(bridge_ch, bridge_ch, act_fn, BatchNorm, d) for d in [8, 3, 2]]
        )
        self.bridge_5 = nn.Sequential(
            *[dilated_conv_bn_act(bridge_ch, bridge_ch, act_fn, BatchNorm, d) for d in [12, 7, 4]]
        )
        self.bridge_6 = nn.Sequential(
            *[dilated_conv_bn_act(bridge_ch, bridge_ch, act_fn, BatchNorm, d) for d in [18, 12, 6]]
        )
        self.bridge_concat = nn.Sequential(
            nn.Conv2d(bridge_ch * 6, bridge_ch, bias=False, kernel_size=1),
            BatchNorm(bridge_ch),
            act_fn,
        )

        # ── BM head (unchanged from original) ────────────────────────────────
        self.bm_head = nn.Sequential(
            nn.Conv2d(
                bridge_ch,
                nf * map_num[0],
                bias=False,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                padding_mode="reflect",
            ),
            BatchNorm(nf * map_num[0]),
            nn.PReLU(),
            nn.Conv2d(
                nf * map_num[0],
                2,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                padding_mode="reflect",
            ),
            nn.Tanh(),
        )

        # ── Shadow U-Net decoder ──────────────────────────────────────────────
        # Stage 1: bridge (H/16, 4nf) + s4 skip (H/8,  2nf) → H/8,  2nf
        # Stage 2: H/8,  2nf          + s3 skip (H/4,  nf)  → H/4,  nf
        # Stage 3: H/4,  nf           + s2 skip (H/4,  nf)  → H/4,  nf
        # Stage 4: H/4,  nf           + s1 skip (H/2,  nf)  → H/2,  nf
        # Final:   H/2,  nf           → H,    3ch
        self.shd_dec1 = _shadow_dec_block(
            bridge_ch, nf * map_num[1], nf * map_num[1], kernel_size, BatchNorm
        )
        self.shd_dec2 = _shadow_dec_block(
            nf * map_num[1], nf * map_num[0], nf * map_num[0], kernel_size, BatchNorm
        )
        self.shd_dec3 = _shadow_dec_block(
            nf * map_num[0], nf * map_num[0], nf * map_num[0], kernel_size, BatchNorm
        )
        self.shd_dec4 = _shadow_dec_block(
            nf * map_num[0], nf * map_num[0], nf * map_num[0], kernel_size, BatchNorm
        )
        self.shd_out = nn.Conv2d(nf * map_num[0], 3, kernel_size=1)

        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier init with a small gain, as in the original UVDoc implementation."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_normal_(m.weight, gain=0.2)

    def forward(self, x):
        """Run the shared encoder/bridge, then the BM and shadow heads."""
        H, W = x.shape[2], x.shape[3]

        # Encoder — capture skips at each resolution
        s1 = self.enc_s1(x)  # (B, nf,  H/2,  W/2)
        s2 = self.enc_s2(s1)  # (B, nf,  H/4,  W/4)
        s3 = self.enc_s3(s2)  # (B, nf,  H/4,  W/4)  stride=1
        s4 = self.enc_s4(s3)  # (B, 2nf, H/8,  W/8)
        feat = self.enc_s5(s4)  # (B, 4nf, H/16, W/16)

        # Bridge
        bridge = self.bridge_concat(
            torch.cat(
                [
                    self.bridge_1(feat),
                    self.bridge_2(feat),
                    self.bridge_3(feat),
                    self.bridge_4(feat),
                    self.bridge_5(feat),
                    self.bridge_6(feat),
                ],
                dim=1,
            )
        )  # (B, 4nf, H/16, W/16)

        # BM head (unchanged)
        bh, bw = self.bm_out_size
        if (bh, bw) == (31, 45):
            bm_raw = F.interpolate(
                self.bm_head(bridge),
                size=(BM_HEIGHT, BM_WIDTH),
                mode="bilinear",
                align_corners=False,
            )
            bm_map = bm_raw[:, :, ::4, ::4]
        else:
            bm_map = F.interpolate(
                self.bm_head(bridge), size=(bh, bw), mode="bilinear", align_corners=False
            )

        # Shadow U-Net decoder
        # Stage 1: H/16 → H/8, fuse s4
        d = F.interpolate(bridge, size=s4.shape[2:], mode="bilinear", align_corners=False)
        d = self.shd_dec1(torch.cat([d, s4], dim=1))

        # Stage 2: H/8 → H/4, fuse s3
        d = F.interpolate(d, size=s3.shape[2:], mode="bilinear", align_corners=False)
        d = self.shd_dec2(torch.cat([d, s3], dim=1))

        # Stage 3: stay H/4, fuse s2 (same spatial size as s3)
        d = self.shd_dec3(torch.cat([d, s2], dim=1))

        # Stage 4: H/4 → H/2, fuse s1
        d = F.interpolate(d, size=s1.shape[2:], mode="bilinear", align_corners=False)
        d = self.shd_dec4(torch.cat([d, s1], dim=1))

        # Final upsample to output resolution
        shadow_size = self.output_size if self.output_size is not None else (H, W)
        shadow_map = F.interpolate(
            self.shd_out(d), size=shadow_size, mode="bilinear", align_corners=False
        )

        return {"bm_map": bm_map, "shadow_map": shadow_map}
