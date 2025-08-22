import os
import copy
import math
import torch
import torch.nn as nn
import sys
import numpy as np

from models.embedding.position import PositionalEmbedding
import torch.nn.functional as F

from mamba.mamba_ssm import BiMamba
from models.GAT import GAT


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0., max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        self.pe = nn.Parameter(torch.zeros(max_len, d_model),requires_grad=True)

        # pe = torch.zeros(max_len, d_model)        
        # position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # pe[:, 0::2] = torch.sin(position * div_term)
        # pe[:, 1::2] = torch.cos(position * div_term)
        # self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0., reduce=False):
        super().__init__()
        out_dim = dim // 2 if reduce else dim
        self.reduce = reduce
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class MixerBlock(nn.Module):

    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout=0.1, channel_first=False,
                 spatial_temporal_enhance=False, num_objects=6, group=2):
        super().__init__()
        self.channel_first = channel_first
        self.num_objects = num_objects
        self.group = group
        self.time = num_patch
        self.dim = dim
        # self.spatial_temporal_enhance = spatial_temporal_enhance
        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patch, token_dim, dropout),
            Rearrange('b d n -> b n d')
        )

        # self.token_mix_equal = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     Rearrange('b n d -> b d n'),
        #     FeedForward(num_patch, num_patch, dropout),
        #     Rearrange('b d n -> b n d')
        # )

        # self.token_mix_reduce = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     Rearrange('b n d -> b d n'),
        #     FeedForward(num_patch, num_patch // 2, dropout),
        #     Rearrange('b d n -> b n d')
        # )

        # self.token_mix_equal_o = nn.Sequential(
        #     nn.LayerNorm(dim // num_objects * (num_objects -1)),
        #     Rearrange('b n d -> b d n'),
        #     FeedForward(num_patch, num_patch, dropout),
        #     Rearrange('b d n -> b n d')
        # )
        #
        # self.token_mix_reduce_o = nn.Sequential(
        #     nn.LayerNorm(dim // num_objects * (num_objects -1)),
        #     Rearrange('b n d -> b d n'),
        #     FeedForward(num_patch, num_patch // 2, dropout),
        #     Rearrange('b d n -> b n d')
        # )

        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout),
        )

        # self.local_mix = nn.Sequential(
        #     FeedForward(self.group, self.group, dropout)
        # )
        # self.local_mix_ln = nn.LayerNorm(dim)

    def forward(self, x):
        b = x.shape[0]
        if self.channel_first:
            x = x + self.channel_mix(x)
            # temporal information
            x = x + self.token_mix(x)
            # x = x + self.token_mix(x) + self.token_mix_reduce(x) + self.token_mix_equal(x)

            # # local
            # x = x.reshape(b, self.time, self.num_objects, self.dim // self.num_objects)
            # h, o  = x[..., 0, :], x[..., 1:, :].reshape(b, self.time, -1)
            # h = h + self.token_mix_reduce(h) + self.token_mix_equal(h)
            # o = o + self.token_mix_reduce_o(o) + self.token_mix_equal_o(o)
            # x = torch.cat([h, o], dim=-1)
            # x = x + self.token_mix_reduce(x) + self.token_mix_equal(x)

            # local relation
            # x = self.local_mix_ln(x)
            # x = x.reshape(b, self.time // self.group, self.group, self.dim).permute(0, 1, 3, 2).reshape(b, -1,
            #                                                                                             self.group)
            # x = x + self.local_mix(x)
            # x = x.reshape(b, self.time // self.group, self.dim, self.group).permute(0, 1, 3, 2).reshape(b, self.time,
            #                                                                                             self.dim)
        else:
            x = x + self.token_mix(x)
            # x = x + self.token_mix(x) + self.token_mix_reduce(x) + self.token_mix_equal(x)
            # x = x + self.token_mix_reduce(x) + self.token_mix_equal(x)

            # # local
            # x = x.reshape(b, self.time, self.num_objects, self.dim // self.num_objects)
            # h, o = x[..., 0, :], x[..., 1:, :].reshape(b, self.time, -1)
            # h = h + self.token_mix_reduce(h) + self.token_mix_equal(h)
            # o = o + self.token_mix_reduce_o(o) + self.token_mix_equal_o(o)
            # x = torch.cat([h, o], dim=-1)

            # local relation
            # x = self.local_mix_ln(x)
            # x = x.reshape(b, self.time // self.group, self.group, self.dim).permute(0, 1, 3, 2).reshape(b, -1,
            #                                                                                             self.group)
            # x = x + self.local_mix(x)
            # x = x.reshape(b, self.time // self.group, self.dim, self.group).permute(0, 1, 3, 2).reshape(b, self.time,
            #                                                                                             self.dim)

            x = x + self.channel_mix(x)

        # if self.spatial_temporal_enhance:
        #     b, t, c = x.shape

        return x

class MultiMixerBlock(nn.Module):

    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout=0.1, channel_first=False,
                 spatial_temporal_enhance=False, num_objects=6, group=2, last_layer=False):
        super().__init__()
        self.channel_first = channel_first
        self.num_objects = num_objects
        self.group = group
        self.time = num_patch
        self.dim = dim
        self.last_layer = last_layer
        # self.spatial_temporal_enhance = spatial_temporal_enhance
        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patch, token_dim, dropout),
            Rearrange('b d n -> b n d')
        )

        self.token_mix_equal = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patch, num_patch, dropout),
            Rearrange('b d n -> b n d')
        )

        self.token_mix_reduce = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patch, num_patch // 2, dropout),
            Rearrange('b d n -> b n d')
        )

        self.token_mix_equal_modal = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patch, num_patch, dropout),
            Rearrange('b d n -> b n d')
        )

        self.token_mix_reduce_modal = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patch, num_patch // 2, dropout),
            Rearrange('b d n -> b n d')
        )

        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout)
        )

        self.channel_mix_modal = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout)
        )

        if self.last_layer:
            dim = dim // self.num_objects
            self.channel_mix_ = nn.Sequential(
                nn.LayerNorm(dim * 2),
                FeedForward(dim * 2, channel_dim, dropout, reduce=self.last_layer),
            )
        else:
            pass

        # self.local_mix = nn.Sequential(
        #     FeedForward(self.group, self.group, dropout)
        # )
        # self.local_mix_ln = nn.LayerNorm(dim)

    def forward(self, x, y=None):
        b = x.shape[0]
        if self.channel_first:

            x = self.channel_mix(x) + x
            y = self.channel_mix_modal(y) + y

            if self.last_layer:
                x, y = x.reshape(b, self.time, self.num_objects, -1), y.reshape(b, self.time, self.num_objects, -1)
                x1 = torch.cat([x, y], dim=-1)
                x = x + self.channel_mix_(x1)
                if x.shape[-1] == (self.dim // self.num_objects * 2):
                    x, y = x[..., :self.dim // self.num_objects], x[..., self.dim // self.num_objects:]
                    # recover
                    x, y = x.reshape(b, self.time, -1), y.reshape(b, self.time, -1)
                else:
                    y = None
                    x= x.reshape(b, self.time, -1)

            # temporal information
            # x = x + self.token_mix(x)
            # x = x + self.token_mix(x) + self.token_mix_reduce(x) + self.token_mix_equal(x)

            # # local
            # x = x.reshape(b, self.time, self.num_objects, self.dim // self.num_objects)
            # h, o  = x[..., 0, :], x[..., 1:, :].reshape(b, self.time, -1)
            # h = h + self.token_mix_reduce(h) + self.token_mix_equal(h)
            # o = o + self.token_mix_reduce_o(o) + self.token_mix_equal_o(o)
            # x = torch.cat([h, o], dim=-1)
            x = x + self.token_mix_reduce(x)  # + self.token_mix_equal(x)
            if y is not None:
                # y = y + self.token_mix_reduce_modal(y) + self.token_mix_equal_modal(y)
                y = y + self.token_mix_equal_modal(y)

            # local relation
            # x = self.local_mix_ln(x)
            # x = x.reshape(b, self.time // self.group, self.group, self.dim).permute(0, 1, 3, 2).reshape(b, -1,
            #                                                                                             self.group)
            # x = x + self.local_mix(x)
            # x = x.reshape(b, self.time // self.group, self.dim, self.group).permute(0, 1, 3, 2).reshape(b, self.time,
            #                                                                                             self.dim)
        else:
            # x = x + self.token_mix(x)
            # x = x + self.token_mix(x) + self.token_mix_reduce(x) + self.token_mix_equal(x)
            # x = x + self.token_mix_reduce(x) + self.token_mix_equal(x)
            # y = y + self.token_mix_reduce_modal(y) + self.token_mix_equal_modal(y)

            x = x + self.token_mix_equal(x)
            y = y + self.token_mix_equal_modal(y)

            # # local
            # x = x.reshape(b, self.time, self.num_objects, self.dim // self.num_objects)
            # h, o = x[..., 0, :], x[..., 1:, :].reshape(b, self.time, -1)
            # h = h + self.token_mix_reduce(h) + self.token_mix_equal(h)
            # o = o + self.token_mix_reduce_o(o) + self.token_mix_equal_o(o)
            # x = torch.cat([h, o], dim=-1)

            # local relation
            # x = self.local_mix_ln(x)
            # x = x.reshape(b, self.time // self.group, self.group, self.dim).permute(0, 1, 3, 2).reshape(b, -1,
            #                                                                                             self.group)
            # x = x + self.local_mix(x)
            # x = x.reshape(b, self.time // self.group, self.dim, self.group).permute(0, 1, 3, 2).reshape(b, self.time,
            #                                                                                             self.dim)
            x = self.channel_mix(x) + x
            y = self.channel_mix_modal(y) + y

            if self.last_layer:
                x, y = x.reshape(b, self.time, self.num_objects, -1), y.reshape(b, self.time, self.num_objects, -1)
                x1 = torch.cat([x, y], dim=-1)
                x = x + self.channel_mix_(x1)
                if x.shape[-1] == (self.dim // self.num_objects * 2):
                    x, y = x[..., :self.dim // self.num_objects], x[..., self.dim // self.num_objects:]
                    # recover
                    x, y = x.reshape(b, self.time, -1), y.reshape(b, self.time, -1)
                else:
                    y = None
                    x = x.reshape(b, self.time, -1)

        # if self.spatial_temporal_enhance:
        #     b, t, c = x.shape

        return x, y

class TimeReduceBlock(nn.Module):

    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout=0.1, channel_first=False,
                 spatial_temporal_enhance=False):
        super().__init__()
        self.channel_first = channel_first
        # self.spatial_temporal_enhance = spatial_temporal_enhance
        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patch, token_dim, dropout, reduce=True),
            Rearrange('b d n -> b n d')
        )

        # self.channel_mix = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     FeedForward(dim, channel_dim, dropout),
        # )

    def forward(self, x):
        x = self.token_mix(x)

        # if self.spatial_temporal_enhance:
        #     b, t, c = x.shape

        return x

class MLPMixer(nn.Module):

    def __init__(self, in_channels, dim, patch_size, image_size, depth, token_dim, channel_dim, num_classes=None,
                 channel_first=False, dropout=0.1, half=True, num_objects=6):
        super().__init__()

        # assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        # self.num_patch = (image_size // patch_size) ** 2
        self.num_patch = image_size
        self.half = half
        self.channel_first = channel_first
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, dim, (patch_size, patch_size), patch_size),
            Rearrange('b c h w -> b w  (h c)'),
            # Rearrange('b c h w -> b w  (c h)'),
            # Rearrange('b c h w -> b (h w) c'),
        )

        self.mixer_blocks = nn.ModuleList([])
        dim = dim * num_objects

        half_depth = depth // 2
        # token_dim = self.num_patch * 2

        for _ in range(depth):
            # if _ == half_depth:
            #     # token_dim = self.num_patch // 2
            #     token_dim = self.num_patch
            self.mixer_blocks.append(
                MixerBlock(dim, self.num_patch, token_dim, channel_dim, channel_first=self.channel_first,
                           dropout=dropout, spatial_temporal_enhance=False, num_objects=num_objects))
            

        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):

        x = self.to_patch_embedding(x)

        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)

        x = self.layer_norm(x)

        # x = x.mean(dim=1)

        # return self.mlp_head(x)
        return x

class MultiMLPMixer(nn.Module):

    def __init__(self, in_channels1, in_channels2, dim, patch_size, image_size, depth, token_dim, channel_dim,
                 num_classes=None,
                 channel_first=False, dropout=0.1, half=True, num_objects=6):
        super().__init__()

        # assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        # self.num_patch = (image_size // patch_size) ** 2
        self.num_patch = image_size
        self.half = half
        self.channel_first = channel_first
        self.depth = depth
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels1, dim, (patch_size, patch_size), patch_size),
            Rearrange('b c h w -> b w  (h c)'),
            # Rearrange('b c h w -> b w  (c h)'),
            # Rearrange('b c h w -> b (h w) c'),
        )

        self.to_patch_embedding_modal2 = nn.Sequential(
            nn.Conv2d(in_channels2, dim, (patch_size, patch_size), patch_size),
            Rearrange('b c h w -> b w  (h c)'),
            # Rearrange('b c h w -> b w  (c h)'),
            # Rearrange('b c h w -> b (h w) c'),
        )

        self.mixer_blocks = nn.ModuleList([])
        dim = dim * num_objects

        # self.to_confusion = nn.Sequential(
        #     nn.Linear(dim * 2, dim)
        # )

        half_depth = depth // 2
        # token_dim = self.num_patch * 2
        last_layer = False

        for _ in range(depth):
            if _ == depth - 1:
                last_layer = True
            # if _ == half_depth:
            #     # token_dim = self.num_patch // 2
            #     token_dim = self.num_patch

            self.mixer_blocks.append(
                MultiMixerBlock(dim, self.num_patch, token_dim, channel_dim, channel_first=self.channel_first,
                                dropout=dropout, spatial_temporal_enhance=False, num_objects=num_objects,
                                last_layer=last_layer))
            self.channel_first = not self.channel_first

        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x, y=None):

        x = self.to_patch_embedding(x)
        y = self.to_patch_embedding_modal2(y)

        for mixer_block in self.mixer_blocks:
            x, y = mixer_block(x, y)

        if y is not None:
            x = x + y
        # x = self.to_confusion(torch.cat([x, y], dim=-1))
        # x = self.layer_norm(x)

        # x = x.mean(dim=1)

        # return self.mlp_head(x)
        return x



def get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class VisualModelV(nn.Module):
    def __init__(self, args, out_type=None):
        super(VisualModelV, self).__init__()
        self.nr_boxes = args.nr_boxes
        self.nr_frames = args.nr_frames
        self.subact_classes = args.subact_classes
        self.afford_classes = args.afford_classes
        self.feat_drop = args.feat_drop
        self.attn_drop = args.attn_drop
        self.cls_dropout = args.cls_dropout

        self.embedding_feature_dim = 256
        self.res_feat_dim = 2048
        self.preprocess_dim = 1024
        self.out_dim = 512
        self.appearence_in_dim = 2 * (self.embedding_feature_dim + self.preprocess_dim)

        # pre process
        self.appearence_preprocess = nn.Linear(self.res_feat_dim, self.preprocess_dim)
        # self.category_embed_layer = nn.Embedding(12, self.embedding_feature_dim // 2, padding_idx=0, scale_grad_by_freq=True)
        self.coord_to_feature = nn.Sequential(
            nn.Linear(4, self.embedding_feature_dim // 2, bias=False),
            nn.BatchNorm1d(self.embedding_feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embedding_feature_dim // 2, self.embedding_feature_dim, bias=False),
            nn.BatchNorm1d(self.embedding_feature_dim),
            nn.ReLU()
        )

        # edge_list = [(0,1),(0,2),(0,3),(0,4),(0,5)]
        edge_list = [(0, n) for n in range(1, self.nr_boxes)]
        src, dst = tuple(zip(*edge_list))
        self.spatial_graph = dgl.graph((src, dst))
        self.spatial_graph = dgl.to_bidirected(self.spatial_graph)
        self.spatial_graph = self.spatial_graph.to('cuda')

        node_list = [x for x in range(self.nr_boxes)]
        node_frame_list = []
        for f_idx in range(self.nr_frames):
            temp = []
            for n_idx in node_list:
                temp.append(f_idx * self.nr_boxes + n_idx)
            node_frame_list.append(temp)
        edge_list = []
        for i in range(self.nr_frames):
            for j in range(self.nr_frames):
                if i == j:
                    continue
                src_nodes = node_frame_list[i]
                dst_nodes = node_frame_list[j]
                for src in src_nodes:
                    for idx, dst in enumerate(dst_nodes):
                        # if idx == 0:
                        #     continue
                        edge_list.append((src, dst))
        src, dst = tuple(zip(*edge_list))
        temp = []
        for frame_idx in range(10):
            temp_ = []
            for idx, dst_idx in enumerate(dst):
                if dst_idx == frame_idx * 6:
                    temp_.append(idx)
            temp.append(temp_)

        self.temporal_graph = dgl.graph((src, dst))
        self.temporal_graph = dgl.to_bidirected(self.temporal_graph)
        self.temporal_graph = self.temporal_graph.to('cuda')

        self.appearence_RNN = nn.RNN(input_size=self.appearence_in_dim, hidden_size=self.appearence_in_dim // 2,
                                     num_layers=1, batch_first=True, bidirectional=True)
        self.appearence_RNN.flatten_parameters()

        self.appearence_spatial_subnet = GAT(self.appearence_in_dim, -1, self.out_dim, feat_drop=self.feat_drop,
                                             attn_drop=self.attn_drop, activation=nn.ReLU())
        # self.gat = GATConv(in_feats=self.appearence_in_dim,out_feats=self.out_dim,num_heads=1,feat_drop=self.feat_drop,attn_drop=self.attn_drop,residual=True,activation=nn.ReLU())
        self.spatial_temporal_subnet = GAT(self.appearence_in_dim, -1, self.out_dim, feat_drop=self.feat_drop,
                                           attn_drop=self.attn_drop, activation=nn.ReLU())

        # RNN blocks for frame-level temporal subnet
        self.subact_frame_RNN = nn.RNN(input_size=2 * self.out_dim, hidden_size=2 * self.out_dim, num_layers=1,
                                       batch_first=True, bidirectional=True)
        self.afford_frame_RNN = nn.RNN(input_size=2 * self.out_dim, hidden_size=2 * self.out_dim, num_layers=1,
                                       batch_first=True, bidirectional=True)
        self.subact_frame_RNN.flatten_parameters()
        self.afford_frame_RNN.flatten_parameters()

        self.classifier_human = nn.Sequential(
            nn.Linear(4 * self.out_dim, 2 * self.out_dim),
            # nn.BatchNorm1d(self.embedding_feature_dim),
            nn.Dropout(self.cls_dropout),
            nn.ReLU(inplace=True),
            nn.Linear(2 * self.out_dim, 512),  # self.embedding_feature_dim),
            nn.Dropout(self.cls_dropout),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.subact_classes)
        )

        self.classifier_object = nn.Sequential(
            nn.Linear(4 * self.out_dim, 2 * self.out_dim),
            # nn.BatchNorm1d(self.embedding_feature_dim),
            nn.Dropout(self.cls_dropout),
            nn.ReLU(inplace=True),
            nn.Linear(2 * self.out_dim, 512),  # self.embedding_feature_dim),
            nn.Dropout(self.cls_dropout),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.afford_classes)
        )

    def forward(self, num_objs, node_features, box_input, box_categories, out_type='scores'):
        batch_size = box_input.size(0)
        batch_spatial_graph = [self.spatial_graph for x in range(batch_size * self.nr_frames)]
        batch_spatial_graph = dgl.batch(batch_spatial_graph)

        batch_temporal_graph = [self.temporal_graph for x in range(batch_size)]
        batch_temporal_graph = dgl.batch(batch_temporal_graph)

        # spatial
        box_input = box_input.transpose(2, 1).contiguous()
        box_input = box_input.view(batch_size * self.nr_boxes * self.nr_frames, 4)
        spatial_feats = self.coord_to_feature(box_input)

        # appearence
        appearence_feats = self.appearence_preprocess(
            node_features.reshape(batch_size * self.nr_boxes * self.nr_frames, self.res_feat_dim))

        # appearence_spatial
        appearence_spatial_feats = torch.cat([spatial_feats, appearence_feats], dim=1)
        appearence_spatial_feats = appearence_spatial_feats.reshape(batch_size, self.nr_boxes, self.nr_frames, -1)
        appearence_spatial_node_feats = torch.zeros(
            (batch_size, self.nr_boxes, self.nr_frames, self.appearence_in_dim)).float().cuda()
        appearence_spatial_node_feats[:, 0, :, :self.appearence_in_dim // 2] = appearence_spatial_feats[:, 0, :, :]
        appearence_spatial_node_feats[:, 1:, :, self.appearence_in_dim // 2:] = appearence_spatial_feats[:, 1:, :, :]

        appearence_spatial_node_feats = self.appearence_RNN(
            appearence_spatial_node_feats.reshape(batch_size * self.nr_boxes, self.nr_frames, self.appearence_in_dim))[
            0].reshape(batch_size, self.nr_boxes, self.nr_frames, self.appearence_in_dim).permute(0, 2, 1, 3)

        appearence_spatial_node_feats = appearence_spatial_node_feats.reshape(
            batch_size * self.nr_frames * self.nr_boxes, self.appearence_in_dim)

        appearence_spatial_subnet_node_feats = self.appearence_spatial_subnet(batch_spatial_graph,
                                                                              appearence_spatial_node_feats)
        appearence_spatial_subnet_node_feats = appearence_spatial_subnet_node_feats.reshape(batch_size, self.nr_frames,
                                                                                            self.nr_boxes, self.out_dim)

        appearence_spatial_temporal_node_feats = self.spatial_temporal_subnet(batch_temporal_graph,
                                                                              appearence_spatial_node_feats)
        appearence_spatial_temporal_node_feats = appearence_spatial_temporal_node_feats.reshape(batch_size,
                                                                                                self.nr_frames,
                                                                                                self.nr_boxes,
                                                                                                self.out_dim)

        spatial_temproal_feats = torch.cat(
            [appearence_spatial_subnet_node_feats, appearence_spatial_temporal_node_feats], dim=3)

        human_node_feats = spatial_temproal_feats[:, :, 0, :]

        obj_node_feats = []
        for b in range(batch_size):
            obj_feats = spatial_temproal_feats[b, :, 1: 1 + num_objs[b], :]

            obj_node_feats.append(obj_feats)

        # obj_node_feats = []
        # for b in range(batch_size):
        #     obj_feats = spatial_graph[b, :, 1: 1+num_objs[b], :]

        #     concat_feats = torch.zeros((self.nr_frames, num_objs[b], 2*self.out_dim)).float().cuda()
        #     for o in range(num_objs[b]):
        #         concat_feats[:, o, :] = torch.cat((human_node_feats[b, :, :], obj_feats[:, o, :]), 1)

        #     obj_node_feats.append(concat_feats)

        obj_node_feats = torch.cat(obj_node_feats, dim=1)
        obj_node_feats = obj_node_feats.permute(1, 0, 2)

        ## Frame-level Temporal subnet
        human_rnn_feats = self.subact_frame_RNN(human_node_feats, None)[0]
        obj_rnn_feats = self.afford_frame_RNN(obj_node_feats, None)[0]

        subact_cls_scores = torch.sum(self.classifier_human(human_rnn_feats), dim=1)
        afford_cls_scores = torch.sum(self.classifier_object(obj_rnn_feats), dim=1)

        return subact_cls_scores, afford_cls_scores

class SemanticModelV(nn.Module):
    def __init__(self, args, out_type=None):
        super(SemanticModelV, self).__init__()
        self.nr_boxes = args.nr_boxes
        self.nr_frames = args.nr_frames
        self.subact_classes = args.subact_classes
        self.afford_classes = args.afford_classes
        self.feat_drop = args.feat_drop
        self.attn_drop = args.attn_drop
        self.cls_dropout = args.cls_dropout

        self.embedding_feature_dim = 256
        self.spatial_dim = 256
        self.semantic_in_dim = self.embedding_feature_dim + self.spatial_dim
        self.out_dim = 512

        self.category_embed_layer = nn.Embedding(12, self.embedding_feature_dim, padding_idx=0, scale_grad_by_freq=True)
        self.coord_to_feature = nn.Sequential(
            nn.Linear(4, self.spatial_dim // 2, bias=False),
            nn.BatchNorm1d(self.spatial_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.spatial_dim // 2, self.spatial_dim, bias=False),
            nn.BatchNorm1d(self.spatial_dim),
            nn.ReLU()
        )

        # edge_list = [(0,1),(0,2),(0,3),(0,4),(0,5)]
        edge_list = [(0, n) for n in range(1, self.nr_boxes)]
        src, dst = tuple(zip(*edge_list))
        self.spatial_graph = dgl.graph((src, dst))
        self.spatial_graph = dgl.to_bidirected(self.spatial_graph)
        self.spatial_graph = self.spatial_graph.to('cuda')

        node_list = [x for x in range(self.nr_boxes)]
        node_frame_list = []
        for f_idx in range(self.nr_frames):
            temp = []
            for n_idx in node_list:
                temp.append(f_idx * self.nr_boxes + n_idx)
            node_frame_list.append(temp)
        edge_list = []
        for i in range(self.nr_frames):
            for j in range(self.nr_frames):
                if i == j:
                    continue
                src_nodes = node_frame_list[i]
                dst_nodes = node_frame_list[j]
                for src in src_nodes:
                    for idx, dst in enumerate(dst_nodes):
                        # if idx == 0:
                        #     continue
                        edge_list.append((src, dst))
        src, dst = tuple(zip(*edge_list))
        self.temporal_graph = dgl.graph((src, dst))
        self.temporal_graph = dgl.to_bidirected(self.temporal_graph)
        self.temporal_graph = self.temporal_graph.to('cuda')

        self.semantic_RNN = nn.RNN(input_size=self.semantic_in_dim, hidden_size=self.semantic_in_dim // 2, num_layers=1,
                                   batch_first=True, bidirectional=True)
        self.semantic_RNN.flatten_parameters()

        self.semantic_spatial_subnet = GAT(self.semantic_in_dim, -1, self.out_dim, feat_drop=self.feat_drop,
                                           attn_drop=self.attn_drop, activation=nn.ReLU())
        # self.gat = GATConv(in_feats=self.semantic_in_dim,out_feats=self.out_dim,num_heads=1,feat_drop=self.feat_drop,attn_drop=self.attn_drop,residual=True,activation=nn.ReLU())
        self.spatial_temporal_subnet = GAT(self.semantic_in_dim, -1, self.out_dim, feat_drop=self.feat_drop,
                                           attn_drop=self.attn_drop, activation=nn.ReLU())

        # RNN blocks for frame-level temporal subnet
        self.subact_frame_RNN = nn.RNN(input_size=2 * self.out_dim, hidden_size=2 * self.out_dim, num_layers=1,
                                       batch_first=True, bidirectional=True)
        self.afford_frame_RNN = nn.RNN(input_size=2 * self.out_dim, hidden_size=2 * self.out_dim, num_layers=1,
                                       batch_first=True, bidirectional=True)
        self.subact_frame_RNN.flatten_parameters()
        self.afford_frame_RNN.flatten_parameters()

        self.classifier_human = nn.Sequential(
            nn.Linear(4 * self.out_dim, 2 * self.out_dim),
            # nn.BatchNorm1d(self.embedding_feature_dim),
            nn.Dropout(self.cls_dropout),
            nn.ReLU(inplace=True),
            nn.Linear(2 * self.out_dim, 512),  # self.embedding_feature_dim),
            nn.Dropout(self.cls_dropout),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.subact_classes)
        )

        self.classifier_object = nn.Sequential(
            nn.Linear(4 * self.out_dim, 2 * self.out_dim),
            # nn.BatchNorm1d(self.embedding_feature_dim),
            nn.Dropout(self.cls_dropout),
            nn.ReLU(inplace=True),
            nn.Linear(2 * self.out_dim, 512),  # self.embedding_feature_dim),
            nn.Dropout(self.cls_dropout),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.afford_classes)
        )

    def forward(self, num_objs, node_features, box_input, box_categories, out_type='scores'):
        batch_size = box_input.size(0)
        batch_spatial_graph = [self.spatial_graph for x in range(batch_size * self.nr_frames)]
        batch_spatial_graph = dgl.batch(batch_spatial_graph)

        batch_temporal_graph = [self.temporal_graph for x in range(batch_size)]
        batch_temporal_graph = dgl.batch(batch_temporal_graph)

        # spatial
        box_input = box_input.transpose(2, 1).contiguous()
        box_input = box_input.view(batch_size * self.nr_boxes * self.nr_frames, 4)
        spatial_feats = self.coord_to_feature(box_input)
        spatial_feats = spatial_feats.view(batch_size, self.nr_boxes, self.nr_frames, -1)

        # embedding
        box_categories = box_categories.long()
        box_categories = box_categories.transpose(2, 1).contiguous()
        box_categories = box_categories.view(batch_size * self.nr_boxes * self.nr_frames)
        box_category_embeddings = self.category_embed_layer(box_categories).view(batch_size, self.nr_boxes,
                                                                                 self.nr_frames,
                                                                                 self.embedding_feature_dim)

        # spatial_embedding
        spatial_embedding = torch.cat([spatial_feats, box_category_embeddings], dim=3)

        semantic_spatial_node_feats = \
            self.semantic_RNN(
                spatial_embedding.reshape(batch_size * self.nr_boxes, self.nr_frames, self.semantic_in_dim))[
                0].reshape(batch_size, self.nr_boxes, self.nr_frames, self.semantic_in_dim).permute(0, 2, 1, 3)

        semantic_spatial_node_feats = semantic_spatial_node_feats.reshape(batch_size * self.nr_frames * self.nr_boxes,
                                                                          self.semantic_in_dim)

        semantic_spatial_subnet_node_feats = self.semantic_spatial_subnet(batch_spatial_graph,
                                                                          semantic_spatial_node_feats)
        semantic_spatial_subnet_node_feats = semantic_spatial_subnet_node_feats.reshape(batch_size, self.nr_frames,
                                                                                        self.nr_boxes, self.out_dim)

        semantic_spatial_temporal_node_feats = self.spatial_temporal_subnet(batch_temporal_graph,
                                                                            semantic_spatial_node_feats)
        semantic_spatial_temporal_node_feats = semantic_spatial_temporal_node_feats.reshape(batch_size, self.nr_frames,
                                                                                            self.nr_boxes, self.out_dim)

        spatial_temproal_feats = torch.cat([semantic_spatial_subnet_node_feats, semantic_spatial_temporal_node_feats],
                                           dim=3)

        human_node_feats = spatial_temproal_feats[:, :, 0, :]

        obj_node_feats = []
        for b in range(batch_size):
            obj_feats = spatial_temproal_feats[b, :, 1: 1 + num_objs[b], :]

            obj_node_feats.append(obj_feats)

        # obj_node_feats = []
        # for b in range(batch_size):
        #     obj_feats = spatial_graph[b, :, 1: 1+num_objs[b], :]

        #     concat_feats = torch.zeros((self.nr_frames, num_objs[b], 2*self.out_dim)).float().cuda()
        #     for o in range(num_objs[b]):
        #         concat_feats[:, o, :] = torch.cat((human_node_feats[b, :, :], obj_feats[:, o, :]), 1)

        #     obj_node_feats.append(concat_feats)

        obj_node_feats = torch.cat(obj_node_feats, dim=1)
        obj_node_feats = obj_node_feats.permute(1, 0, 2)

        ## Frame-level Temporal subnet
        human_rnn_feats = self.subact_frame_RNN(human_node_feats, None)[0]
        obj_rnn_feats = self.afford_frame_RNN(obj_node_feats, None)[0]

        subact_cls_scores = torch.sum(self.classifier_human(human_rnn_feats), dim=1)
        afford_cls_scores = torch.sum(self.classifier_object(obj_rnn_feats), dim=1)

        return subact_cls_scores, afford_cls_scores


class HOI_Mamba_block(nn.Module):
    
    def __init__(self, layer, num_frame, d_model, ds, dconv, expands):
        super(HOI_Mamba_block, self).__init__()
        self.layer = layer
        self.ds=ds
        self.dconv=dconv
        self.st1 = BiMamba(d_model=d_model, d_state=self.ds, d_conv=self.dconv, expand=expands)
        # 
        self.st2 = BiMamba(d_model=d_model, d_state=self.ds, d_conv=self.dconv, expand=expands)
        
        
        self.stList1 = get_clones(self.st1, layer)
        
        self.stList2 = get_clones(self.st2, layer)
        
        self.position_encoding = PositionalEncoding(d_model, 0.2, num_frame)
        self.norm = nn.LayerNorm(d_model, eps=1e-5)
        
    def forward(self, x):
        batch, num_object, num_frame, feats = x.size()
                
        x = x.reshape(-1, num_frame, feats)
        x = self.position_encoding(x)
        x = x.reshape(batch, num_object, num_frame, feats)

        for l in range(self.layer):
              # (batch, num_frame, num_object, feats)
            st_feats1 = self.norm(self.stList1[l](x.reshape(batch, -1, feats)))
            y=x.permute(0,2,1,3)
            st_feats2=self.norm(self.stList2[l](y.reshape(batch, -1, feats)))    
            out=(st_feats1+st_feats2)/2        
            out = out.reshape(batch, num_object, num_frame, feats)
            x = out.contiguous()+x
        return x.contiguous()

class HOI_Mamba(nn.Module):
    def __init__(self, args, out_type=None):
        super(HOI_Mamba, self).__init__()
        self.nr_boxes = args.nr_boxes
        self.nr_frames = args.nr_frames
        self.subact_classes = args.subact_classes
        self.afford_classes = args.afford_classes
        self.feat_drop = args.feat_drop
        self.attn_drop = args.attn_drop
        self.cls_dropout = args.cls_dropout

        self.embedding_feature_dim = 128  # for multi-modal

        self.res_feat_dim = 2048+512
        self.preprocess_dim = 256
        self.out_dim = 512
        # pre process
        self.appearence_preprocess = nn.Linear(self.res_feat_dim, self.preprocess_dim)

        self.coord_to_feature = nn.Sequential(
            nn.Linear(4, self.embedding_feature_dim // 2, bias=False),
            nn.BatchNorm1d(self.embedding_feature_dim // 2),
            nn.SiLU(inplace=True),
            nn.Linear(self.embedding_feature_dim // 2, self.embedding_feature_dim, bias=False),
            nn.BatchNorm1d(self.embedding_feature_dim),
            nn.SiLU()
        )

        self.category_embed_layer = nn.Embedding(12, self.embedding_feature_dim, padding_idx=0, scale_grad_by_freq=True)
        # multi-model
        self.modal1 = 2*self.embedding_feature_dim + self.preprocess_dim
       
        self.norm = nn.LayerNorm(self.out_dim, eps=1e-5)
        
        self.hoi_mamba=HOI_Mamba_block(8, self.nr_frames, self.out_dim, 16, 2, 1)

        self.classifier_human = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim),
            nn.Dropout(self.cls_dropout),
            nn.SiLU(inplace=True),
            nn.Linear(self.out_dim, self.out_dim // 2),  # self.embedding_feature_dim),
            nn.Dropout(self.cls_dropout),
            nn.SiLU(inplace=True),
            nn.Linear(self.out_dim // 2, self.subact_classes)
        )

        self.classifier_object = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim),
            nn.Dropout(self.cls_dropout),
            nn.SiLU(inplace=True),
            nn.Linear(self.out_dim, self.out_dim // 2),  # self.embedding_feature_dim),
            nn.Dropout(self.cls_dropout),
            nn.SiLU(inplace=True),
            nn.Linear(self.out_dim // 2, self.afford_classes)
        )

    def forward(self, num_objs, node_features, box_input, box_categories, out_type='scores'):
        batch_size = box_input.size(0)
        # node_features = node_features1[:, :, :, :self.res_feat_dim]
        # spatial
        box_input = box_input.transpose(2, 1).contiguous()
        box_input = box_input.view(batch_size * self.nr_boxes * self.nr_frames, 4)
        spatial_feats = self.coord_to_feature(box_input)

        # embedding
        box_categories = box_categories.long()
        box_categories = box_categories.transpose(2, 1).contiguous()
        box_categories = box_categories.view(batch_size * self.nr_boxes * self.nr_frames)
        box_category_embeddings = self.category_embed_layer(box_categories)

        # appearence
        appearence_feats = self.appearence_preprocess(
            node_features.reshape(batch_size * self.nr_boxes * self.nr_frames, self.res_feat_dim))
        
        appearence_spatial_feats = torch.cat([appearence_feats, spatial_feats, box_category_embeddings], dim=1)

        appearence_spatial_feats = appearence_spatial_feats.reshape(batch_size, self.nr_boxes,self.nr_frames, -1)
        
        spatial_temporal_feats=self.hoi_mamba(appearence_spatial_feats)
        
        human_node_feats = spatial_temporal_feats[:, 0, :, :]

        obj_node_feats = []
        for b in range(batch_size):
            obj_feats = spatial_temporal_feats[b,1: 1 + num_objs[b], :,  :]

            obj_node_feats.append(obj_feats)

        obj_node_feats = torch.cat(obj_node_feats, dim=0)

        # final, useful
        subact_cls_scores = torch.sum(self.classifier_human(human_node_feats), dim=1)
        afford_cls_scores = torch.sum(self.classifier_object(obj_node_feats), dim=1)

        return subact_cls_scores, afford_cls_scores
