# This code is built from the Stable Diffusion repository: https://github.com/CompVis/stable-diffusion, and
# Paint-by-Example repo https://github.com/Fantasy-Studio/Paint-by-Example
# Copyright (c) 2022 Robin Rombach and Patrick Esser and contributors.
# CreativeML Open RAIL-M
#
# ==========================================================================================
#
# Adobe’s modifications are Copyright 2024 Adobe Research. All rights reserved.
# Adobe’s modifications are licensed under the Adobe Research License. To view a copy of the license, visit
# LICENSE.md.
#
# ==========================================================================================

from inspect import isfunction
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
import glob

from ldm.modules.diffusionmodules.util import checkpoint


def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0., only_crossref=False):
        super().__init__()
        inner_dim = dim_head * heads
        # forcing attention to only attend on vectors of same size
        # breaking the image2text attention
        context_dim = default(context_dim, query_dim)
                
        # print('creating cross attention. Query dim', query_dim, ' context dim', context_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        
        self.only_crossref = only_crossref
        if only_crossref:
            self.merge_attentions = zero_module(nn.Conv2d(self.heads * 2,
                                                self.heads,
                                                kernel_size=1,
                                                stride=1,
                                                padding=0))
        else:
            self.merge_attentions = zero_module(nn.Conv2d(self.heads * 3,
                                                self.heads,
                                                kernel_size=1,
                                                stride=1,
                                                padding=0))
            
        
        self.merge_attentions_missing = zero_module(nn.Conv2d(self.heads * 2,
                                              self.heads,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))
        

    def forward(self, x, context=None, mask=None, passed_qkv=None, masks=None, corresp=None, missing_region=None):
        is_self_attention = context is None
        
        # if masks is not None:
        #     print(is_self_attention, masks.keys())
        
        h = self.heads
        
        # if passed_qkv is not None:
        #     assert context is None
            
        #     _,_,_,_, x_features = passed_qkv
        #     assert x_features is not None
            
        #     # print('x shape', x.shape, 'x features', x_features.shape)
        #     # breakpoint()
        #     x = torch.concat([x, x_features], dim=1)
        
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            assert False
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        inter_out = rearrange(out, '(b h) n d -> b h n d', h=h)
        
        combined_attention = inter_out
        out = rearrange(combined_attention, 'b h n d -> b n (h d)', h=h)
        
        final_out = self.to_out(out)
        
        if is_self_attention:
            return final_out, q, k, v, inter_out #TODO add attn out
        else:
            return final_out


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.attn3 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    # TODO add attn in
    def forward(self, x, context=None, passed_qkv=None, masks=None, corresp=None):
        if passed_qkv is None:
            return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)
        else:
            q, k, v, attn, x_features = passed_qkv
            d = int(np.sqrt(q.shape[1]))
            current_mask = masks[d]
            if corresp:
                current_corresp, missing_region = corresp[d]
                current_corresp = current_corresp.float()
                missing_region = missing_region.float()
            else:
                raise ValueError('cannot have empty corresp')
                current_corresp = None
                missing_region = current_mask.float()
            # breakpoint()
            stuff = [q, k, v, attn, x_features, current_mask, current_corresp, missing_region]
            for element in stuff:
                assert element is not None
            return checkpoint(self._forward, (x, context, q, k, v, attn, x_features, current_mask, current_corresp, missing_region), self.parameters(), self.checkpoint)

    # TODO add attn in
    def _forward(self, x, context=None, q=None, k=None, v=None, attn=None, passed_x=None, masks=None, corresp=None, missing_region=None):
        if q is not None:
            passed_qkv = (q, k, v, attn, passed_x)
        else:
            passed_qkv = None
        x_features = self.norm1(x)
        attended_x, q, k, v, attn = self.attn1(x_features, passed_qkv=passed_qkv, masks=masks, corresp=corresp, missing_region=missing_region)
        x = attended_x + x
        # killing CLIP features        
        
        if passed_x is not None:
            normed_x = self.norm2(x)
            attn_out  = self.attn3(normed_x, context=passed_x)
            x = attn_out + x
            # then use y + x
            # print('y shape', y.shape, ' x shape', x.shape)
        
        x = self.ff(self.norm3(x)) + x
        return x, q, k, v, attn, x_features


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        
        # print('creating spatial transformer')
        # print('in channels', in_channels, 'inner dim', inner_dim)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    # TODO add attn in and corresp
    def forward(self, x, context=None, passed_qkv=None, masks=None, corresp=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        # print('spatial transformer x shape given', x.shape)
        # if context is not None:
        #     print('also context was provided with shape ', context.shape)
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        
        qkvs = []
        for block in self.transformer_blocks:
            x, q, k, v, attn, x_features = block(x, context=context, passed_qkv=passed_qkv, masks=masks, corresp=corresp)
            qkv = (q,k,v,attn, x_features)
            qkvs.append(qkv)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in, qkvs