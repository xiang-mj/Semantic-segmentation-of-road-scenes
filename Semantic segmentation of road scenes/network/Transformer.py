import torch
import torch.nn as nn
from functools import partial
import math
import warnings

def Transformer(seg_edge):
    # print(seg_edge.size())#torch.Size([2, 256, 360, 360])
    dim = 256
    # size = 36
    # conv_layer = nn.Conv2d(256, dim, kernel_size=size, stride=size)
    # seg_edge = conv_layer(seg_edge)
    B, C, H, W = seg_edge.shape
    # # seg_edge = np.reshape(seg_edge, [B, C, H * W])
    seg_edge = torch.reshape(seg_edge, [B, C, H * W])
    # print(seg_edge.shape)##torch.Size([2, 256, 100])
    seg_edge = seg_edge.flatten(2).transpose(1, 2)
    # print(seg_edge.shape)  # torch.Size([2, 100, 256])
    B, N, C = seg_edge.shape
    cls_token = nn.Parameter(torch.zeros(1, 1, C))
    pos_embed = nn.Parameter(torch.zeros(1, N + 1, C))
    pos_drop = nn.Dropout(p=0.1)

    def _no_grad_trunc_normal_(tensor, mean, std, a, b):
        def norm_cdf(x):
            # Computes standard normal cumulative distribution function
            return (1. + math.erf(x / math.sqrt(2.))) / 2.

        if (mean < a - 2 * std) or (mean > b + 2 * std):
            warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                          "The distribution of values may be incorrect.",
                          stacklevel=2)

        with torch.no_grad():
            l = norm_cdf((a - mean) / std)
            u = norm_cdf((b - mean) / std)

            tensor.uniform_(2 * l - 1, 2 * u - 1)

            tensor.erfinv_()

            # Transform to proper mean, std
            tensor.mul_(std * math.sqrt(2.))
            tensor.add_(mean)

            # Clamp to ensure it's in the proper range
            tensor.clamp_(min=a, max=b)
            return tensor

    def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
        return _no_grad_trunc_normal_(tensor, mean, std, a, b)

    trunc_normal_(pos_embed, std=.02)
    trunc_normal_(cls_token, std=.02)

    cls_tokens = cls_token.expand(B, -1, -1)
    seg_edge = torch.cat((cls_tokens, seg_edge), dim=1)
    seg_edge = seg_edge + pos_embed
    # print(seg_edge.shape)#torch.Size([2, 101, 256])
    seg_edge = pos_drop(seg_edge)
    # print(seg_edge.shape)  # torch.Size([2, 101, 256])######


    norm_layer = partial(nn.LayerNorm, eps=1e-6)
    #dim = 256#####################################################
    #drop_path = 0.
    mlp_ratio = 8.
    norm1 = norm_layer(dim)

    """首先通过线性变换将输入映射为查询、键和值，然后计算注意力权重，并将注意力权重与值相乘得到最终的输出。
    最后，对输出进行线性变换和丢弃操作，得到最终的结果。这个自注意力机制可以用于处理序列数据，并在其中学习到序列中不同位置的重要性和关联性。"""
    def Attention(x):
        B, N, C = x.shape
        num_heads = 8
        # dim = 4096
        head_dim = dim // num_heads
        """如果 scale 为 None，则将其设置为 head_dim 的倒数的平方根。缩放因子的引入是为了确保注意力权重的数值范围适当，有助于提高模型的学习能力。"""
        scale = None or head_dim ** -0.5
        qkv = nn.Linear(dim, dim * 3, bias=True)
        attn_drop = nn.Dropout(0.)
        proj = nn.Linear(dim, dim)
        proj_drop = nn.Dropout(0.)
        q, k, v = qkv(x).reshape(B, N, 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
        """将查询与键的转置相乘，并乘以缩放因子 scale，得到了注意力权重。这一步骤实质上是计算了查询与键之间的相似度。"""
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        attn = attn_drop(attn)
        """将注意力权重与值相乘，并进行维度的转置和重塑操作，得到最终的输出 x。注意力权重与值的相乘操作实际上是对值按照注意力权重进行加权求和。"""
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = proj(x)
        x = proj_drop(x)
        return x

    def DropPath(x):
        drop_prob = drop_path ###########
        if drop_prob == 0.:
            return x
        keep_prob = 1 - drop_prob
        # work with diff dim tensors, not just 2D ConvNets
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

    drop_path = 0.
    drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    norm2 = norm_layer(dim)
    mlp_hidden_dim = int(dim * mlp_ratio)

    """实现了一个多层感知机（MLP）模块。它包含两个线性层和一个激活函数层，通过线性变换和非线性激活函数的组合实现特征的映射和转换。
    丢弃层用于在训练过程中减少过拟合。这个多层感知机模块可以用于对输入进行非线性变换和特征提取。"""
    def Mlp(x):
        mlp_ratio = 8.
        # dim = 4096
        act_layer = nn.GELU
        mlp_hidden_dim = int(dim * mlp_ratio)
        out_features = None or dim
        """定义隐藏层特征维度 hidden_features。如果 mlp_hidden_dim 不为零，则将其设置为 mlp_hidden_dim，否则将其设置为输入特征维度 dim。"""
        hidden_features = mlp_hidden_dim or dim
        """将输入特征维度 dim 映射到隐藏层特征维度 hidden_features。"""
        fc1 = nn.Linear(dim, hidden_features)
        act = act_layer()
        fc2 = nn.Linear(hidden_features, out_features)
        drop = nn.Dropout(0.1)

        x = fc1(x)
        x = act(x)
        x = drop(x)
        x = fc2(x)
        x = drop(x)
        return x

    seg_edge = seg_edge + drop_path(Attention(norm1(seg_edge)))
    seg_edge = seg_edge + drop_path(Mlp(norm2(seg_edge)))


    seg_edge = seg_edge.flatten(2).transpose(1, 2)
    seg_edge = seg_edge[:, :, 1:]
    # n, hidden_size, num_tokens = seg_edge.shape
    # h = int(num_tokens ** 0.5)  # 假设特征图的高度和宽度相同
    # w = h
    # c = hidden_size
    # seg_edge = seg_edge.reshape(n, c, h, w)

    # upsample = nn.Upsample(scale_factor=size, mode='nearest')
    # seg_edge = upsample(seg_edge)
    # conv = nn.Conv2d(dim, 256, kernel_size=1, stride=1)
    # seg_edge = conv(seg_edge)
    return seg_edge
