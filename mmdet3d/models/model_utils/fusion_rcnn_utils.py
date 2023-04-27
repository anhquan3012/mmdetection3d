import torch.nn as nn
import pdb
import torch
import numpy as np
from numpy import *
import torch.nn.functional as F
from typing import Optional, List
from torch import Tensor
import copy
from copy import deepcopy

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class Transformer(nn.Module):
    def __init__(self, d_model=512,
                 nhead=8, 
                 num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu",
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = EncoderLayer(pts_self_attention=dict(d_model=d_model,
                                                             nhead=nhead,
                                                             dim_feedforward=dim_feedforward,
                                                             dropout=dropout,
                                                             activation=activation),
                                    img_self_attention=dict(d_model=d_model,
                                                             nhead=nhead,
                                                             dim_feedforward=dim_feedforward,
                                                             dropout=dropout,
                                                             activation=activation),
                                    cross_attention=dict(d_model=d_model,
                                                        nhead=nhead,
                                                        dim_feedforward=dim_feedforward,
                                                        dropout=dropout,
                                                        activation=activation))
        self.encoder = Encoder(encoder_layer, num_encoder_layers, None)

        decoder_layer = DecoderLayer(d_model = d_model, 
                                     nhead = nhead, 
                                     dim_feedforward=dim_feedforward,
                                     dropout=dropout, 
                                     activation=activation)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = Decoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        self.return_intermediate_dec = return_intermediate_dec

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, pts_src, img_src, query_embed):
        bs, n, c = pts_src.shape
        pts_src = pts_src.permute(1, 0, 2) # (n, bs, c)
        img_src = pts_src.permute(1, 0, 2)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(pts_src, img_src) # (n, bs, c)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=None,
                          pos=None, query_pos=query_embed) # (num_decoder, 1, bs, c)
        if self.return_intermediate_dec:
            return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, n)
        else:
            return hs.transpose(0, 1) # (bs, 1, c)

class Encoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, pts_src,
                img_src,
                pts_src_mask: Optional[Tensor] = None,
                pts_src_key_padding_mask: Optional[Tensor] = None,
                img_src_mask: Optional[Tensor] = None,
                img_src_key_padding_mask: Optional[Tensor] = None,
                cross_src_mask: Optional[Tensor] = None,
                cross_src_key_padding_mask: Optional[Tensor] = None,
                pts_pos: Optional[Tensor] = None,
                img_pos: Optional[Tensor] = None):
        
        for layer in self.layers:
            pts_src, img_src = layer(pts_src,
                        img_src,
                        pts_src_mask,
                        pts_src_key_padding_mask,
                        img_src_mask,
                        img_src_key_padding_mask,
                        cross_src_mask,
                        cross_src_key_padding_mask,
                        pts_pos,
                        img_pos)

        if self.norm is not None:
            pts_src = self.norm(pts_src)

        return pts_src
    
class EncoderLayer(nn.Module):
    def __init__(self,
                 pts_self_attention=None,
                 img_self_attention=None,
                 cross_attention=None):
        super().__init__()
        self.pts_attention = SelfAttention(**pts_self_attention)
        self.img_attention = SelfAttention(**img_self_attention)
        self.cross_attention = CrossAttention(**cross_attention)

    def forward(self, pts_src,
                img_src,
                pts_src_mask: Optional[Tensor] = None,
                pts_src_key_padding_mask: Optional[Tensor] = None,
                img_src_mask: Optional[Tensor] = None,
                img_src_key_padding_mask: Optional[Tensor] = None,
                cross_src_mask: Optional[Tensor] = None,
                cross_src_key_padding_mask: Optional[Tensor] = None,
                pts_pos: Optional[Tensor] = None,
                img_pos: Optional[Tensor] = None):
        pts_src = self.pts_attention(pts_src, pts_src_mask, pts_src_key_padding_mask, pts_pos)
        img_src = self.img_attention(img_src, img_src_mask, img_src_key_padding_mask, img_pos)
        pts_src = self.cross_attention(pts_src, img_src, cross_src_mask, cross_src_key_padding_mask, pts_pos, img_pos)

        return pts_src, img_src

class CrossAttention(nn.Module):
    def __init__(self, d_model, 
                 nhead, 
                 dim_feedforward=2048, 
                 dropout=0.1,
                 activation="relu"):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, pts_src,
                img_src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos_q: Optional[Tensor] = None,
                pos_k: Optional[Tensor] = None,):
        q = self.with_pos_embed(pts_src, pos_q)
        k = self.with_pos_embed(img_src, pos_k)
        src2 = self.cross_attn(q, k, value=img_src, attn_mask=src_mask,
                            key_padding_mask=src_key_padding_mask)[0]
        pts_src = pts_src + self.dropout1(src2)
        pts_src = self.norm1(pts_src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(pts_src))))
        pts_src = pts_src + self.dropout2(src2)
        pts_src = self.norm2(pts_src)
        return pts_src
    
class SelfAttention(nn.Module):
    def __init__(self, d_model, 
                 nhead, 
                 dim_feedforward=2048, 
                 dropout=0.1,
                 activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos        

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
    
class Decoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)

def attention(query, key,  value):
    dim = query.shape[1]
    scores_1 = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    scores_2 = torch.einsum('abcd, aced->abcd', key, scores_1)
    prob = torch.nn.functional.softmax(scores_2, dim=-1)
    output = torch.einsum('bnhm,bdhm->bdhn', prob, value)
    return output, prob

class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(merge) for _ in range(3)])
        self.down_mlp = MLP(input_dim = self.dim, hidden_dim = 32, output_dim = 1, num_layers = 1)

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        # pdb.set_trace()
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, prob = attention(query, key, value)
        x = self.down_mlp(x)
        return x.contiguous().view(batch_dim, self.dim*self.num_heads, -1)


class DecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiHeadedAttention(nhead, d_model)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    
    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                            key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos).permute(1,2,0),
                                key=self.with_pos_embed(memory, pos).permute(1,2,0),
                                value=memory.permute(1,2,0))
        tgt2 = tgt2.permute(2,0,1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")