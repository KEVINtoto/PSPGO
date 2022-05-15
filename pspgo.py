
from logzero import logger
from dgl.heterograph import DGLHeteroGraph
import torch.nn as nn
import torch.nn.functional as F
import torch as th
import dgl
from dgl.utils import expand_as_pair
from dgl.nn.pytorch.utils import Identity
from dgl.nn.functional import edge_softmax
import dgl.function as fn
import numpy as np


class PSPGO(nn.Module): 
    def __init__(self, in_dim, h_dim, out_dim, n_hidden_layer, n_prop_step, mlp_drop=0.0,
                 attn_heads=1, feat_drop=0.0, attn_drop=0.0, residual=True, share_weight=True):
        super(PSPGO, self).__init__()
        self.n_prop_step = n_prop_step
        self.dropout = mlp_drop

        self.embed_layer = nn.EmbeddingBag(in_dim, h_dim, mode="sum", include_last_offset=True)
        self.embed_bias = nn.Parameter(th.zeros(h_dim))

        self.mlp = MLP(h_dim, h_dim, h_dim, n_hidden_layer, mlp_drop)

        self.prop_layers = nn.ModuleList([PropagateLayer(h_dim, h_dim, attn_heads, feat_drop, attn_drop,
                                                         residual=residual, share_weight=share_weight) for _ in range(n_prop_step)])

        self.output_layer = nn.Linear(h_dim, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.embed_layer.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.constant_(self.output_layer.bias, 0)

    def forward(self, blocks, x, y=None): 
        h = F.relu(self.embed_layer(*x) + self.embed_bias)
        h = self.mlp(h)
        for block, prop_layer in zip(blocks, self.prop_layers):
            h, y = prop_layer(block, h, y)
        h = self.output_layer(h)
        return h, y

    def inference(self, g, idx, x, y, batch_size, device): 
        self.eval()
        unique_idx = np.unique(idx)
        index_mapping = {idx: i for i, idx in enumerate(unique_idx)}
        idx = np.asarray([ index_mapping[idx] for idx in idx ])

        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.n_prop_step)
        dataloader = dgl.dataloading.NodeDataLoader(g, unique_idx, sampler, device='cpu',
                                     batch_size=int(batch_size/2), shuffle=False, num_workers=0,  drop_last=False)
        x_output_list = []
        y_output_list = []
        for input_nodes, _, blocks in dataloader:
            blocks = [blk.to(device) for blk in blocks]
            batch_x = ( th.from_numpy(x[input_nodes.numpy()].indices).long().to(device), 
                        th.from_numpy(x[input_nodes.numpy()].indptr).long().to(device), 
                        th.from_numpy(x[input_nodes.numpy()].data).float().to(device) )  
            batch_y = th.from_numpy(y[input_nodes.numpy()]).float().to(device)
            batch_x_hat, batch_y_hat = self.forward(blocks, batch_x, batch_y)
            x_output_list.append(th.sigmoid(batch_x_hat).cpu().detach().numpy())
            y_output_list.append(batch_y_hat.cpu().detach().numpy())
        
        x_output = np.vstack(x_output_list)[idx]
        y_output = np.vstack(y_output_list)[idx]

        return x_output, y_output



class PropagateLayer(nn.Module):
    def __init__(self, in_dim, out_dim, attn_heads, 
                 feat_drop=0., attn_drop=0.,
                 residual=True,
                 activation=F.elu,
                 share_weight=True):
        super(PropagateLayer, self).__init__()
        self.attn_heads = attn_heads
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.share_weight = share_weight
        if share_weight:
            self.gat = GraphAttention(in_dim, out_dim, attn_heads, feat_drop=feat_drop, attn_drop=attn_drop, residual=residual)
        else:
            self.gat_p = GraphAttention(in_dim, out_dim, attn_heads, feat_drop=feat_drop, attn_drop=attn_drop, residual=residual)
            self.gat_s = GraphAttention(in_dim, out_dim, attn_heads, feat_drop=feat_drop, attn_drop=attn_drop, residual=residual)
        self.feat_drop = nn.Dropout(feat_drop)
        self.residual = residual
        self.activation = activation

    def forward(self, block: DGLHeteroGraph, x, y):
        with block.local_scope():

            block_p = block['interaction']
            block_s = block['similarity']

            if self.share_weight:
                h_p, a_p = self.gat(block_p, x)
                h_s, a_s = self.gat(block_s, x)
            else:
                h_p, a_p = self.gat_p(block_p, x)
                h_s, a_s = self.gat_s(block_s, x)

            h = self.activation(h_p + h_s)
            h = self.feat_drop(h)

            # label propagate
            if y != None:
                block_p.edata['a'] = a_p
                block_s.edata['a'] = a_s
                dst_flag = block.dstdata['flag']
                y0 = y[:block.number_of_dst_nodes()][dst_flag]
                block_p.srcdata.update({'y': y})
                block_p.update_all(fn.u_mul_e('y', 'a', 'm'),
                                   fn.sum('m', 'y_hat'))
                y_hat_i = block_p.dstdata.pop('y_hat')
                block_s.srcdata.update({'y': y})
                block_s.update_all(fn.u_mul_e('y', 'a', 'm'),
                                   fn.sum('m', 'y_hat'))
                y_hat_s = block_s.dstdata.pop('y_hat')
                y_hat = F.normalize(y_hat_i + y_hat_s)
                y_hat[dst_flag] = y0
            else:
                y_hat = y

            return h, y_hat


class GraphAttention(nn.Module):
    def __init__(self,
                 in_feats, out_feats, num_heads,
                 feat_drop=0., attn_drop=0.,
                 negative_slope=0.2, residual=False, activation=None):
        super(GraphAttention, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self.fc_src = nn.Linear(
            self._in_src_feats, out_feats * num_heads)
        self.fc_dst = nn.Linear(
            self._in_src_feats, out_feats * num_heads)
        self.attn = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
        nn.init.constant_(self.fc_src.bias, 0)
        nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.constant_(self.fc_dst.bias, 0)
        nn.init.xavier_normal_(self.attn, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
            if self.bias:
                nn.init.constant_(self.res_fc.bias, 0)

    def forward(self, graph: DGLHeteroGraph, feat):
        with graph.local_scope():
            h_src = h_dst = self.feat_drop(feat)
            feat_src = self.fc_src(h_src).view(
                -1, self._num_heads, self._out_feats)
            feat_dst = self.fc_dst(h_src).view(
                 -1, self._num_heads, self._out_feats)
            if graph.is_block:
                feat_dst = feat_src[:graph.number_of_dst_nodes()]
                h_dst = h_dst[:graph.number_of_dst_nodes()]

            graph.srcdata.update({'el': feat_src}) # (num_src_edge, num_heads, out_dim)
            graph.dstdata.update({'er': feat_dst})
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e')) # (num_src_edge, num_heads, out_dim)
            e = (e * self.attn).sum(dim=-1).unsqueeze(dim=2) # (num_edge, num_heads, 1)
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e)) # (num_edge, num_heads)
            # feature propagation
            graph.update_all(fn.u_mul_e('el', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval
            # activation
            if self.activation:
                rst = self.activation(rst)

            return rst.mean(1), graph.edata['a'].mean(1)

            
class MLP(nn.Module):
    def __init__(self, input_d, hidden_d, output_d, 
                 num_layers, dropout, norm = 'layer'):
        super(MLP, self).__init__()

        self.norm = norm
        self.dropout = dropout

        self.layers = nn.ModuleList([])

        if num_layers == 1:
            self.layers.append(nn.Linear(input_d, output_d))
        elif num_layers > 1:
            self.layers.append(nn.Linear(input_d, hidden_d))
            for _ in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_d, hidden_d))
            self.layers.append(nn.Linear(hidden_d, output_d))
        
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_d) for _ in range(num_layers)])

        self.reset_parameters()

    def reset_parameters(self):
        """reset mlp parameters using xavier_norm"""
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight.data)
            nn.init.constant_(layer.bias.data, 0)

    def forward(self, x):
        """The forward pass of mlp."""

        for layer, norm in zip(self.layers, self.norms):
            x = layer(x)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)

        return x