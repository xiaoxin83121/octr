import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter
from einops import rearrange, repeat
from torch.functional import einsum
import spconv.pytorch as spconv
from functools import partial
import numpy as np
import math

import time

@torch.no_grad()
def parent_dense(inv_indices, count):
    """
    args:
        inv_indices: [N_i], values in [0, N_i-1); count: count in each grid
    return:
        ret: [N_i-1*8] 8 means 2*2*2 grid upsampling, values in [0, N_i]
    """
    device = inv_indices.device
    _, sort = torch.sort(inv_indices)
    max_n = torch.max(count).item()
    ret = torch.zeros(count.shape[0] * max_n, device=device, dtype=torch.long)
    mask = torch.arange(max_n, device=device)[None, :] < count[:, None]
    ret.masked_scatter_(mask.view(-1), sort)
    return ret.view(-1, max_n), mask.int()


@torch.no_grad()
def coords_generator(data_dict, tree_depth):
    """
    args:
        data_dict: 
        tree_depth: int
    rets:
        data_dict
            eg: depth==4
            coords_2x: H//2, W//2, L//2
            coords_4x: H//4, W//4, L//4
            coords_8x: H//8, W//8, L//8
            parent_2x: 1->2, range //2
            parent_4x: 2->4, range //4
            parent_8x: 4->8, range //8
    """
    voxel_coords = data_dict['voxel_coords']

    # data preparation
    for i in range(1, tree_depth):
        # when it comes to z=1, this also gets correct result
        coord_i = torch.cat((voxel_coords[:, 0].unsqueeze(-1), torch.div(voxel_coords[:, 1:], 2**i, rounding_mode='floor')), dim=-1)
        coord_compress, coord_inv, count = torch.unique(coord_i, return_inverse=True, return_counts=True, dim=0)
        data_dict['info_{}x'.format(2**i)] = {
            'coord_compress': coord_compress, 'coord_inv': coord_inv,
        }
        if i == 1:
            data_dict['parent_{}x'.format(2**i)] = {'parent_inv': coord_inv, 'count': count}
        else:
            last_coords = data_dict['info_{}x'.format(2**(i-1))]['coord_compress']
            last_coords = torch.cat((last_coords[:, 0].unsqueeze(-1), torch.div(last_coords[:, 1:], 2, rounding_mode='floor')), dim=-1)
            _, parent_inv, count = torch.unique(last_coords, return_inverse=True, return_counts=True, dim=0)
            data_dict['parent_{}x'.format(2**i)] = {'parent_inv': parent_inv, 'count': count}
    return data_dict

def batch_dense(coords, features):
    """
    args:
        coords: [N_i] only batch_idx
        features: [N_i d]
    return:
        ret: [batch_size max_tokens d]
        mask: [batch_size max_tokens]
        batch_inv: batch_idx inverse indices
    """
    device = coords.device
    _, batch_inv, batch_counts = torch.unique(coords, return_inverse=True, return_counts=True, dim=0)
    __, sort = torch.sort(batch_inv)
    t, sort_inv = torch.sort(sort)
    max_n = int(torch.max(batch_counts).item())
    batch_size = batch_counts.shape[0]
    ret = torch.zeros((batch_size*max_n, features.shape[-1]), device=device)
    inds = torch.zeros(batch_size*max_n, device=device, dtype=torch.long)
    mask = torch.arange(max_n, device=device)[None, :] < batch_counts[:, None]
    ret[mask.view(-1), :] = features[sort]
    inds.masked_scatter_(mask.view(-1), sort)
    return ret.view(batch_size, max_n, features.shape[-1]), mask.int(), inds.view(-1, max_n), batch_inv, sort_inv

def batch_dense_v2(batch_inv, batch_counts, features):
    """
    args:
        coords: [N_i] only batch_idx
        features: [N_i d]
    return:
        ret: [batch_size max_tokens d]
        mask: [batch_size max_tokens]
        batch_inv: batch_idx inverse indices
    """
    device = features.device
    __, sort = torch.sort(batch_inv)
    t, sort_inv = torch.sort(sort)
    max_n = int(torch.max(batch_counts).item())
    batch_size = batch_counts.shape[0]
    ret = torch.zeros((batch_size*max_n, features.shape[-1]), device=device)
    inds = torch.zeros(batch_size*max_n, device=device, dtype=torch.long)
    mask = torch.arange(max_n, device=device)[None, :] < batch_counts[:, None]
    ret[mask.view(-1), :] = features[sort]
    inds.masked_scatter_(mask.view(-1), sort)
    return ret.view(batch_size, max_n, features.shape[-1]), mask.int(), inds.view(-1, max_n), batch_inv, sort_inv


def index_inds(inds, topk):
    """
    args:
        inds: [N_i n]
        topk: [N_i k]
    rets:
        select_inds: [N_i k] index form inds
    """
    device = inds.device
    B = inds.shape[0]
    view_shape = list(topk.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(topk.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    select_inds = inds[batch_indices, topk.long()]
    return select_inds

def children_sample(inds, mask, max_token_num):
    """
    compaction for inds
    example: inds=[2, 1, 3, 4], mask=[1, 0, 1, 1], 2 => [2, 3], [1, 1]
    args:
        inds: [N_i max_n*k]
        mask: [N_i max_n*k]
    rets:
        compact inds: [N_i max_token_num]
        compact mask: [N_i max_token_num]
    """
    device = inds.device
    n = inds.shape[0]
    count = torch.sum(mask, dim=-1)
    max_num = max(torch.max(count).item(), max_token_num)
    compact_inds = torch.zeros(n * max_num, device=device, dtype=torch.long)
    compact_mask = torch.arange(max_num, device=device)[None, :] < count[:, None]
    select_inds = torch.masked_select(inds, mask.bool())
    compact_inds.masked_scatter_(compact_mask.view(-1), select_inds)
    return compact_inds.view(-1, max_num)[:, :max_token_num], compact_mask[:, :max_token_num].int()


def topk_gumbel_softmax(logits: torch.Tensor, tau: float = 1, hard: bool = False,
        topk: int = 2, eps: float = 1e-10, dim: int = -1):
    gumbels = torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).uniform_()
    gumbels = -torch.log(eps - torch.log(gumbels+eps))
    gumbels = (logits + gumbels) / tau
    y_soft = gumbels.softmax(dim)
    if topk is None:
        topk = (gumbels>gumbels.mean(dim, keepdim=True)).int().sum(dim)
    if hard:
        # Straight through.
        if len(torch.tensor(topk).view(-1)) == 1:
            indices = y_soft.topk(topk)[1]
            y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, indices, 1.0)
        else:
            y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format)
            for i, k in enumerate(topk):
                indices = y_soft[i].topk(k)[1]
                y_hard[i].scatter_(dim, indices, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, inp_q, inp_kv, attn_mask, topk, pos_bias=None):
        """
        args:
            attn_mask = [n_i max_tokens] in set([0, 1])
        return:
            out_feature: [n_i 1 d]
            topk_id: [n_i 1 topk]
        """
        n_q, h = inp_q.shape[1], self.heads
        q = rearrange(self.to_q(inp_q), 'b n (h d) -> b h n d', h=h)
        k = rearrange(self.to_k(inp_kv), 'b n (h d) -> b h n d', h=h)
        v = rearrange(self.to_v(inp_kv), 'b n (h d) -> b h n d', h=h)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        if attn_mask is not None:
            attn_mask = -10000. * (1- attn_mask)
            dots += repeat(attn_mask, 'b n -> b h m n', h=h, m=n_q)
        attn = self.attend(dots)
        if self.training:
            topk_idx = topk_gumbel_softmax(torch.sum(attn, dim=1), hard=True, topk=topk)
        else:
            _, topk_idx = torch.topk(torch.sum(attn, dim=1), dim=-1, k=topk, largest=True)  # [n_i 1 topk]
        attn = self.attn_drop(attn)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), topk_idx


class Mlp(nn.Module):
    def __init__(self, dim, hidden_dim, dropout, norm=nn.LayerNorm):
        super().__init__()
        self.norm_layer = norm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.norm_layer(x)
        return self.net(x)


class OctreeAttentionDense(nn.Module):
    def __init__(self, dim, nhead, head_dim, dropout, tree_depth, topks, layer_idx, norm_func=nn.BatchNorm1d, pos_emb=None, mode='global', wsize=5):
        super().__init__()
        self.dim = dim
        self.nhead = nhead
        self.head_dim = head_dim
        self.tree_depth = tree_depth
        self.topks = topks
        self.mode = mode
        self.wsize = wsize
        assert self.tree_depth == len(self.topks), "mismatch on topks and tree_depth"
        assert mode in ['windows', 'global', 'shift_windows']
        self.reduction = 'max'
        assert self.reduction in ['mean', 'max', 'sum']
        self.attn_layer = CrossAttention(dim, nhead, head_dim, dropout)
        self.top_layer = CrossAttention(dim, nhead, head_dim, dropout)
        self.ffn = Mlp(dim, dim * 2, dropout, norm=norm_func)
        self.norm_layer = nn.ModuleList([norm_func(dim) for _ in range(tree_depth)])
        self.proj = nn.Linear(dim*self.tree_depth, dim, bias=False)
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.local_pe = spconv.SparseSequential(
                post_act_block(dim, dim, 3, norm_fn=norm_fn, padding=1, indice_key=f'lpe_subm{layer_idx}'),
                post_act_block(dim, dim, 3, norm_fn=norm_fn, padding=1, indice_key=f'lpe_subm{layer_idx}'),
            )

    def process_coarse_level(self, feature, info_dict, topk, grid_size):
        """
        args:
            feature: [N d] (finest)
            coords: N * [batch_idx, nx, ny, nz]
        returns:
            feature: [N, d] (coarest)
            topk_area: [N_1 (h) topk]
        """
        kv_coords_compress, kv_inv = info_dict['coord_compress'], info_dict['coord_inv']
        coords = kv_coords_compress
        if self.mode == 'windows':
            coords[:, 1:] = torch.div(kv_coords_compress[:, 1:], self.wsize, rounding_mode='floor')
        elif self.mode == 'shift_windows':
            coords[:, 2:] = coords[:, 2:] + self.wsize // 2
            coords[:, 1:] = torch.div(kv_coords_compress[:, 1:], self.wsize, rounding_mode='floor')

        query_feature = scatter(feature, kv_inv.long(), reduce=self.reduction, dim=0)  # [N_1 d]
        query_feature = self.norm_layer[0](query_feature)
        
        # scatter to dense [B kv_count d]
        dense_batch_features, dense_batch_mask, \
            dense_batch_inds, batch_inv_indices, sort_inv = batch_dense(kv_coords_compress[:, 0] if self.mode=='global' else coords, query_feature)
 
        out_feature, topk_area = self.top_layer(dense_batch_features, dense_batch_features, dense_batch_mask, topk)
        # keep the order of query_feature after sort_inv
        scatter_feature = out_feature[dense_batch_mask > 0, :][sort_inv]
        topk_area = topk_area[dense_batch_mask > 0, :][sort_inv]
        # dense_batch_inds and batch_inv_indices are from torch.unique(coords), the order are just the same as query_feature without sort_inv
        dense_kv_inds = dense_batch_inds[batch_inv_indices, ...]
        dense_kv_mask = dense_batch_mask[batch_inv_indices, ...]
        if self.training:
            ret_inds = dense_kv_inds[topk_area == 1.].view(-1, topk)
            ret_masks = dense_kv_mask[topk_area == 1.].view(-1, topk)
        else:
            ret_inds = index_inds(dense_kv_inds, topk_area)
            ret_masks = index_inds(dense_kv_mask, topk_area)
        return scatter_feature[kv_inv, :], ret_inds, ret_masks

    def process_fine_level(self, feature, coord_dict, topk, prev_topk, kv_max_tokens, topk_area, topk_mask, parent_dict, idx, grid_size):
        """
        args:
            feature: N * d (finest)
            coords: N * [batch_idx, nx, ny, nz]
            topk: int
            kv_max_tokens: int
            topk_area: candidates key/value [ N_i-1, k_i-1 ]
            parent_dict: inverse indices between depth_i and depth_i-1
                'parent_inv'
                'count'
        returns:
            feature: N_i * d (coarest)
            topks index: N_i * topk
        """
        if coord_dict['coord_inv'] is not None:
            kv_inv = coord_dict['coord_inv']  # [N_i 4], [N]
            query_feature = scatter(feature, kv_inv.long(), reduce=self.reduction, dim=0)  # [N_i d]
        else:
            query_feature = feature
        query_feature = self.norm_layer[idx](query_feature)
        
        # scatter topk_area to depth N_i
        parent_inv, parent_count = parent_dict['parent_inv'], parent_dict['count']
        # the order of parent_info and parent_mask is binded to coord_compress from last layer
        parent_info, parent_mask = parent_dense(parent_inv, parent_count)  # [N_i-1, max_n], [N_i-1, max_n]
        this_topk_inds = parent_info[topk_area.view(-1), :]  # [N_i-1*k, max_n]
        this_topk_inds_mask = parent_mask[topk_area.view(-1), :]
        this_topk_inds_mask[topk_mask.view(-1) == 0] = 0
        this_topk_inds, this_topk_inds_mask = children_sample(rearrange(this_topk_inds, '(n k) g -> n (g k)', k=prev_topk),
                rearrange(this_topk_inds_mask, '(n k) g -> n (g k)', k=prev_topk), kv_max_tokens
            )

        # the order of dense_batch_features and dense_batch_mask is binded to coord_compress from last layer
        dense_batch_features, dense_batch_mask, dense_batch_inds, batch_inv_indices, sort_inv = batch_dense_v2(parent_inv, parent_count, query_feature)
        
        # index from query_feature
        kv_features = query_feature[this_topk_inds.reshape(-1), :].reshape(-1, kv_max_tokens, query_feature.shape[-1])  # [N_i-1 max_tokens d]

        # attn and topk
        out_feature, topk_area = self.attn_layer(dense_batch_features, kv_features, this_topk_inds_mask, topk)
        # keep the order of query_feature after sort_inv
        out_feature = out_feature[dense_batch_mask > 0, :][sort_inv]
        topk_area = topk_area[dense_batch_mask > 0, :][sort_inv]

        if self.training:
            ret_inds = this_topk_inds[parent_inv, :][topk_area == 1.].view(-1, topk)
            ret_masks = this_topk_inds_mask[parent_inv, :][topk_area == 1.].view(-1, topk)
        else:
            ret_inds = index_inds(this_topk_inds[parent_inv, :], topk_area)
            ret_masks = index_inds(this_topk_inds_mask[parent_inv, :], topk_area)

        return out_feature[kv_inv, :] if coord_dict['coord_inv'] is not None else out_feature, ret_inds, ret_masks

    def forward(self, data_dict):
        voxel_features = data_dict['voxel_features']
        lpe = spconv.SparseConvTensor(
                features=data_dict['voxel_features'],
                indices=data_dict['voxel_coords'].int(),
                spatial_shape=data_dict['grid_size'],
                batch_size=data_dict['batch_size']
            )

        out_feature, topk_area, topk_mask = self.process_coarse_level(voxel_features, data_dict['info_{}x'.format(2**(self.tree_depth-1))], 
                                                            topk=self.topks[self.tree_depth-1], 
                                                            grid_size=data_dict['grid_size']//2**(self.tree_depth-1))

        out_features = [out_feature.unsqueeze(-1)]
        for i in range(self.tree_depth-1, 0, -1):
            out_feature, topk_area, topk_mask = self.process_fine_level(
                voxel_features,
                data_dict['info_{}x'.format(2**(i-1))] if i != 1 else {'coord_compress': data_dict['voxel_coords'], 'coord_inv': None},
                topk=self.topks[i-1],
                prev_topk=self.topks[i],
                kv_max_tokens=self.topks[i] * 3,
                topk_area=topk_area,
                topk_mask=topk_mask,
                parent_dict=data_dict['parent_{}x'.format(2**i)],
                idx=self.tree_depth-i,
                grid_size=data_dict['grid_size']//2**(i-1),
            )
            out_features.append(out_feature.unsqueeze(-1))
        
        feature = torch.cat(out_features, dim=-1)  # N d depth
        # feature = torch.mean(feature, dim=-1) + voxel_features
        feature = self.proj(rearrange(feature, 'n c d -> n (d c)')) + self.local_pe(lpe).features
        feature = feature + self.ffn(feature)
        data_dict['voxel_features'] = feature
        return data_dict
        

class BasicBlock(nn.Module):
    def __init__(self, dim, nhead, head_dim, dropout, tree_depth, topks):
        super().__init__()
        self.dim = dim
        self.nhead = nhead
        self.head_dim = head_dim
        self.tree_depth = tree_depth
        self.topks = topks
        self.depth = 2

        self.attn_layers = nn.ModuleList([
            # OctreeAttentionDense(dim, nhead, head_dim, dropout, tree_depth-1, topks[1:], layer_idx=0, mode='windows', wsize=5), 
            # OctreeAttentionDense(dim, nhead, head_dim, dropout, tree_depth-1, topks[1:], layer_idx=1, mode='shift_windows', wsize=5),
            OctreeAttentionDense(dim, nhead, head_dim, dropout, tree_depth-1, topks[1:], layer_idx=0), 
            OctreeAttentionDense(dim, nhead, head_dim, dropout, tree_depth-1, topks[1:], layer_idx=1),
            ]
        )
        

    def forward(self, batch_dict):
        batch_dict = coords_generator(batch_dict, self.tree_depth)
        for i in range(self.depth):
            batch_dict = self.attn_layers[i](batch_dict)
        return batch_dict


class OcTrV2(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size,**kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.block_nums = self.model_cfg.BLOCK_NUMS
        self.tree_depths = self.model_cfg.TREE_DEPTHS
        self.ds_block = self.model_cfg.DOWN_SAMPLE_BLOCK
        self.dims = self.model_cfg.DIMS
        self.nheads = self.model_cfg.NHEADS
        self.block_depths = self.model_cfg.BLOCKDEPTHS
        assert self.block_nums == len(self.tree_depths)
        self.spatial_shape = grid_size[::-1]
        self.num_point_features = 128  # build detector_3d
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64
        }
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block
        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 32] <- [800, 704, 16]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.blocks = nn.ModuleList([])
        for i in range(self.block_nums):
            self.blocks.append(BasicBlock(self.dims[i], 2, 32, 0, self.tree_depths[i], topks=[8]*(self.tree_depths[i]-1)+[8])
                )
        self.ds = nn.ModuleList([])
        for i in range(self.block_nums):
            in_channel = 32 if i == 0 else self.dims[i-1]
            self.ds.append(spconv.SparseSequential(
                        spconv.SparseConv3d(in_channel, self.dims[i], 3, stride=2, padding=1, bias=False),
                        partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)(self.dims[i]),
                        nn.ReLU(),
                    )
                )
        self.conv_out = spconv.SparseSequential(
            spconv.SparseConv3d(self.dims[-1], self.num_point_features, (3, 1, 1), stride=(2, 1, 1), padding=0, bias=False),
            partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)(self.num_point_features),
            nn.ReLU(),
        )


    def forward(self, batch_dict):
        dct = {}
        batch_dict['grid_size'] = self.spatial_shape
        batch_size = batch_dict['batch_size']
        
        sp_tensor = spconv.SparseConvTensor(
                features=batch_dict['voxel_features'],
                indices=batch_dict['voxel_coords'].int(),
                spatial_shape=batch_dict['grid_size'],
                batch_size=batch_size
            )
        x = self.conv_input(sp_tensor)
        x_conv1 = self.conv1(x)
        dct['x_conv1'] = x_conv1
        x_conv2 = self.conv2(x_conv1)

        for i in range(self.block_nums):
            sp_tensor = spconv.SparseConvTensor(
                features=batch_dict['voxel_features'],
                indices=batch_dict['voxel_coords'].int(),
                spatial_shape=batch_dict['grid_size'],
                batch_size=batch_size
            ) if i != 0 else x_conv2
            dct['x_conv'+str(2+i)] = sp_tensor
            sp_tensor = self.ds[i](sp_tensor)
            batch_dict['voxel_features'] = sp_tensor.features
            batch_dict['voxel_coords'] = sp_tensor.indices
            batch_dict['grid_size'] = np.array(sp_tensor.spatial_shape)
            batch_dict = self.blocks[i](batch_dict)
        
        sp_out = spconv.SparseConvTensor(
            features=batch_dict['voxel_features'],
            indices=batch_dict['voxel_coords'].int(),
            spatial_shape=batch_dict['grid_size'],
            batch_size=batch_size
        )
        dct['x_conv4'] = sp_out
        batch_dict.update({
            'encoded_spconv_tensor': self.conv_out(sp_out), 
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': dct
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })
        return batch_dict

def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m

class PositionalEncodingFourier(nn.Module):
    """
    Positional encoding relying on a fourier kernel matching the one used in the
    "Attention is all of Need" paper. The implementation builds on DeTR code
    https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py
    """
    def __init__(self, hidden_dim=64, dim=128, temperature=10000):
        super().__init__()
        self.token_projection = nn.Linear(hidden_dim * 3, dim)
        self.scale = 2 * math.pi
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        

    def forward(self, pos_embed, max_len=(1, 1, 1)):
        z_embed, y_embed, x_embed = pos_embed.chunk(3, 1)
        z_max, y_max, x_max = max_len
        
        eps = 1e-6
        z_embed = z_embed / (z_max + eps) * self.scale
        y_embed = y_embed / (y_max + eps) * self.scale
        x_embed = x_embed / (x_max + eps) * self.scale

        dim_t = torch.arange(self.hidden_dim, dtype=torch.float32, device=pos_embed.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.hidden_dim)

        pos_x = x_embed / dim_t
        pos_y = y_embed / dim_t
        pos_z = z_embed / dim_t
        pos_x = torch.stack((pos_x[:, 0::2].sin(),
                             pos_x[:, 1::2].cos()), dim=2).flatten(1)
        pos_y = torch.stack((pos_y[:, 0::2].sin(),
                             pos_y[:, 1::2].cos()), dim=2).flatten(1)
        pos_z = torch.stack((pos_z[:, 0::2].sin(),
                             pos_z[:, 1::2].cos()), dim=2).flatten(1)

        pos = torch.cat((pos_z, pos_y, pos_x), dim=1)

        pos = self.token_projection(pos)
        return pos   