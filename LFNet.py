import torch
import torch.nn as nn

from thop import profile

from einops import rearrange
from einops.layers.torch import Rearrange


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SE(nn.Module):
    def __init__(self, inp, oup, expansion=0.25):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_4x4_bn(inp, oup, image_size, downsample=False):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, 2, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.GELU(),
        nn.Conv2d(oup, oup, 3, 1, 1, groups=oup, bias=False),
        nn.BatchNorm2d(oup),
        nn.GELU(),

    )
def max_bn(inp, oup, image_size, downsample=False):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, 2, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.GELU(),

        nn.Conv2d(oup, oup*4, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup*4),
        nn.GELU(),
        # dw
        nn.Conv2d(oup*4, oup*4, 3, 1, 1,
                  groups=oup*4, bias=False),
        nn.BatchNorm2d(oup*4),
        nn.GELU(),
        SE(oup, oup*4),
        # pw-linear
        nn.Conv2d(oup*4, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),

        
    )



class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """

    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x


class ACBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 padding_mode='zeros', use_affine=True, reduce_gamma=False, gamma_init=None):
        super(ACBlock, self).__init__()

        self.square_conv = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                   kernel_size=(kernel_size, kernel_size), stride=stride,
                                                   padding=padding, dilation=dilation, groups=groups, bias=False,
                                                   padding_mode=padding_mode),
                                         nn.GELU(),
                                         nn.BatchNorm2d(num_features=out_channels, affine=use_affine))

        if padding - kernel_size // 2 >= 0:

            self.crop = 0

            hor_padding = [padding - kernel_size // 2, padding]
            ver_padding = [padding, padding - kernel_size // 2]
        else:

            self.crop = kernel_size // 2 - padding
            hor_padding = [0, padding]
            ver_padding = [padding, 0]

        self.ver_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, 1),
                      stride=stride,
                      padding=ver_padding, dilation=dilation, groups=groups, bias=False,
                      padding_mode=padding_mode),
            nn.GELU(),
            nn.BatchNorm2d(num_features=out_channels, affine=use_affine)
        )

        self.hor_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, kernel_size),
                      stride=stride,
                      padding=hor_padding, dilation=dilation, groups=groups, bias=False,
                      padding_mode=padding_mode),
            nn.GELU(),
            nn.BatchNorm2d(num_features=out_channels, affine=use_affine)
        )
        self.ver_bn = nn.BatchNorm2d(num_features=out_channels, affine=use_affine)
        self.hor_bn = nn.BatchNorm2d(num_features=out_channels, affine=use_affine)

        if reduce_gamma:
            self.init_gamma(1.0 / 3)

        if gamma_init is not None:
            assert not reduce_gamma
            self.init_gamma(gamma_init)

    def forward(self, input):

        square_outputs = self.square_conv(input)

        if self.crop > 0:
            ver_input = input[:, :, :, self.crop:-self.crop]
            hor_input = input[:, :, self.crop:-self.crop, :]
        else:
            ver_input = input
            hor_input = input
        vertical_outputs = self.ver_conv(ver_input)

        horizontal_outputs = self.hor_conv(hor_input)

        result = square_outputs + vertical_outputs + horizontal_outputs

        return result



class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm):
        super().__init__()
        self.norm = norm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(hidden_dim, eps=1e-5),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(dim, eps=1e-5),
            nn.Dropout(dropout),

        )

    def forward(self, x):
        return self.net(x)


class linerAttention(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, dropout=0.):
        super().__init__()
        dim_head = inp // heads
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == inp)

        self.ih, self.iw = image_size

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Sequential(
            Rearrange('b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw),
            nn.Conv2d(inp, inner_dim * 3, 1),
            nn.Conv2d(inner_dim * 3, inner_dim * 3, kernel_size=3, stride=1, padding=1, groups=inner_dim * 3),
            Rearrange('b c ih iw -> b (ih iw) c'),
        )
        self.reatten_matrix = nn.Conv2d(self.heads, self.heads, 1, 1)
        self.var_norm = nn.BatchNorm2d(self.heads)
        self.to_out = nn.Sequential(
            # nn.Conv2d(inner_dim, oup, 1, 1),
            nn.Linear(inner_dim, oup),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)


        dots = torch.matmul(k.transpose(-1, -2), q) * self.scale
        dots = self.attend(dots)

        out = torch.matmul(v, dots)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out




# wu


class Transformer(nn.Module):

    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, dropout=0.):
        super().__init__()
        hidden_dim = int(inp * 4)

        self.ih, self.iw = image_size



        self.attn = nn.Sequential(
            Rearrange('b c ih iw -> b (ih iw) c'),
            linerAttention(inp, oup, image_size, heads, dim_head, dropout),
            Rearrange('b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
        )

    def forward(self, x):
        x = self.attn(x)

        return x




class stage1(nn.Module):
    def __init__(self, in_channel, image_size, expansion=4, **kwargs, ):
        super(stage1, self).__init__()
        ih, iw = image_size
        inp = in_channel

        in_channel = _make_divisible(in_channel * 0.7, 32)
        out_channel = inp - in_channel

        self.conv_1 = nn.Conv2d(in_channels=inp, out_channels=in_channel, kernel_size=1, stride=1)
        self.norm1 = nn.BatchNorm2d(in_channel, eps=1e-5)

        self.conv = nn.Sequential(
            ACBlock(in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1, groups=in_channel)
        )

        self.conv_2 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1)
        self.norm2 = nn.BatchNorm2d(out_channel, eps=1e-5)

        self.T = nn.Sequential(  # Pooling(),
            Transformer(inp=out_channel, oup=out_channel, image_size=image_size)
        )
        self.conv_3 = nn.Conv2d(in_channels=inp, out_channels=inp, kernel_size=1, stride=1)
        self.norm3 = nn.BatchNorm2d(inp, eps=1e-5)

    def forward(self, x):
        B, C, H, W = x.shape
        out = self.conv_1(x)
        out = self.norm1(out)
        out = self.conv(out)
        x = out
        out = self.conv_2(out)
        out = self.norm2(out)
        out = out + self.T(out)
        x = torch.cat((out, x), dim=1)
        x = self.conv_3(x)


        return x






class no_down_block1(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, downsample=False, dropout=0.):
        super(no_down_block1, self).__init__()
        self.downsample = downsample
        stride = 1 if self.downsample == False else 2
        self.ih, self.iw = image_size
        hidden_dim = int(inp * 4)
        self.conv = nn.Sequential(ACBlock(in_channels=inp, out_channels=inp, kernel_size=3, padding=1, groups=inp),

                                  )

        self.ff = FeedForward(oup, hidden_dim, dropout)
        self.norm = nn.Sequential(
            Rearrange('b c ih iw -> b (ih iw) c'),
            nn.LayerNorm(inp),
            Rearrange('b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
        )


        self.s = stage1(in_channel=inp, image_size=image_size)

    def forward(self, x):
        x = x + self.s(x)

        return x






class po(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, downsample=False, dropout=0.):
        super(po, self).__init__()
        self.downsample = downsample
        stride = 1 if self.downsample == False else 2
        self.ih, self.iw = image_size
        hidden_dim = int(inp * 4)
        if self.downsample:

            self.pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)


        self.conv = ACBlock(in_channels=inp, out_channels=inp, kernel_size=3, padding=1, groups=inp)
        self.attn = linerAttention(inp, oup, image_size, heads, dim_head, dropout)
        self.attn = nn.Sequential(
            Rearrange('b c ih iw -> b (ih iw) c'),
            PreNorm(inp, self.attn, nn.LayerNorm),
            Rearrange('b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
        )
        self.ff = FeedForward(oup, hidden_dim, dropout)
        self.norm = nn.Sequential(
            Rearrange('b c ih iw -> b (ih iw) c'),
            nn.LayerNorm(inp),
            Rearrange('b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
        )


    def forward(self, x):
        if self.downsample:
            # x = self.proj(self.pool1(x)) + self.attn(self.pool2(x))
            x = self.proj(self.pool(x))
            # x = self.down(x)
            # x = self.patch(x)
        else:
            x = x + self.conv(x)
            x = x + self.ff(x)
            # x = self.biattn(x)
            # x = self.s(x)
        return x


class Meatf(nn.Module):
    def __init__(self, image_size, in_channels, num_blocks, channels, num_classes=1000,
                 block_types=['p', 'p', 'p', 'p', 'no_down1']):
        super().__init__()
        ih, iw = image_size
        block = {'p': po, 'no_down1': no_down_block1}
        self.s0 = self._make_layer(
            conv_4x4_bn, in_channels, channels[0], num_blocks[0], (ih // 2, iw // 2))
        self.s1 = self._make_layer(
            max_bn, channels[0], channels[1], num_blocks[1], (ih // 4, iw // 4))
        self.s2 = self._make_layer(
            block[block_types[2]], channels[1],channels[2], num_blocks[1], (ih // 8, iw // 8))

        self.s2_1 = self._make_layer_no(block[block_types[4]], channels[2], channels[2], num_blocks[5],
                                        (ih // 8, iw // 8))

        self.s3 = self._make_layer(
            block[block_types[2]], channels[2], channels[3], num_blocks[3], (ih // 16, iw // 16))

        self.s3_1 = self._make_layer_no(block[block_types[4]], channels[3], channels[3], num_blocks[6],
                                        (ih // 16, iw // 16))

        self.s4 = self._make_layer(
            block[block_types[3]], channels[3], channels[4], num_blocks[4], (ih // 32, iw // 32))

        self.s4_1 = self._make_layer_no(block[block_types[4]], channels[4], channels[4], num_blocks[5],
                                        (ih // 32, iw // 32))

        self.pool = nn.AvgPool2d(ih // 32, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[-1], num_classes, bias=False)

    def forward(self, x):
        x = self.s0(x)
        x = self.s1(x)
        x = self.s2(x)
        x = self.s2_1(x)
        x = self.s3(x)
        x = self.s3_1(x)
        x = self.s4(x)
        x = self.s4_1(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def _make_layer(self, block, inp, oup, depth, image_size):
        layers = nn.ModuleList([])
        for i in range(depth):
            if i == 0:
                layers.append(block(inp, oup, image_size, downsample=True))
            else:
                layers.append(block(oup, oup, image_size))
        return nn.Sequential(*layers)

    def _make_layer_no(self, block, inp, oup, depth, image_size):
        layers = nn.ModuleList([])
        for i in range(depth):
            layers.append(block(oup, oup, image_size))
        return nn.Sequential(*layers)


def LFNet(num_classes=1000):
    num_blocks = [1, 1, 1, 1, 1, 1, 2]  # L
    channels = [64, 96, 128, 160, 192]  # D
    return Meatf((224, 224), 3, num_blocks, channels, num_classes=1000)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    img = torch.randn(1, 3, 224, 224)

    net = LFNet()
    out = net(img)

    input_profile = torch.randn(1, 3, 224, 224)
    flops, params = profile(net, inputs=(input_profile,))
    print(flops / (10 ** 9), params / (10 ** 6))


