
# ref1: https://github.com/DQSSSSS/VectorNet 该版本中MLP实现较为简略
# ref2: https://github.com/xk-huang/yet-another-vectornet 此版本较为完整的实现了VectorNet中的细节，但是耦合了torch_geometric，需要torch_geometric.data格式输入
# ref3: https://github.com/Henry1iu/TNT-Trajectory-Prediction 此版本参考上述实现，部分去除了对torch_geometric的耦合
# 本实现参考上述三个实现，去除了耦合，简化了代码，实现了vectornet

import torch
from torch import nn
import torch.nn.functional as F
from mmdet3d.models.builder import BACKBONES, build_backbone
from mmcv.runner import BaseModule
from mmdet.models.utils.builder import TRANSFORMER
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence, build_positional_encoding
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmcv.cnn import normal_init, xavier_init
# ref3
class MLP(nn.Module):
    def __init__(self, in_channel, out_channel, hidden=32, bias=True):
        super(MLP, self).__init__()
        act_layer = nn.ReLU
        norm_layer = nn.LayerNorm

        # insert the layers
        self.linear1 = nn.Linear(in_channel, hidden, bias=bias)
        self.linear1.apply(self._init_weights)
        self.linear2 = nn.Linear(hidden, out_channel, bias=bias)
        self.linear2.apply(self._init_weights)

        self.norm1 = norm_layer(hidden)
        self.norm2 = norm_layer(out_channel)

        self.act1 = act_layer(inplace=True)
        self.act2 = act_layer(inplace=True)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        out = self.linear1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.linear2(out)
        out = self.norm2(out)
        return self.act2(out)

# ref1
class SubGraph(nn.Module):
    r"""
    Subgraph of VectorNet. This network accept a number of initiated vectors belong to
    the same polyline, flow three layers of network, then output this polyline's feature vector.
    """

    def __init__(self, v_len, layers_number=4):
        r"""
        Given all vectors of this polyline, we should build a 3-layers subgraph network,
        get the output which is the polyline's feature vector.

        Args:
            v_len: the length of vector. node特征长度 node_dim
            layers_number: the number of subgraph network.
        """
        super(SubGraph, self).__init__()
        self.layers = nn.Sequential()
        for i in range(layers_number):
            self.layers.add_module("sub{}".format(i), SubGraphLayer(v_len * (2 ** i)))
        self.v_len = v_len
        self.layers_number = layers_number

    def forward(self, x):
        r"""
        Args:
            x: a number of vectors. x.shape=[vector_num, node_num, node_dim].
        Returns:
            The feature of this polyline. 
            [vector_num, node_feature_dim]
            node_feature_dim = node_dim * (2 ** layers_number)
        """
        assert len(x.shape) == 3
        batch_size = x.shape[0]
        x = self.layers(x) # [vector_num, node_num, node_dim]
        x = x.permute(0, 2, 1)  # [vector_num, node_dim, node_num]
        x = F.max_pool1d(x, kernel_size=x.shape[2])
        x = x.permute(0, 2, 1)  # [vector_num, 1, node_feature_dim]
        x.squeeze_(1)
        assert x.shape == (batch_size, self.v_len * (2 ** self.layers_number))
        # polyline-level feature
        return x


class SubGraphLayer(nn.Module):
    r"""
    One layer of subgraph, include the MLP of g_enc.
    The calculation detail in this paper's 3.2 section.
    Input some vectors with 'len' length, the output's length is '2*len'(because of
    concat operator).
    """

    def __init__(self, len):
        r"""
        Args:
            len: the length of input vector.
        """
        super(SubGraphLayer, self).__init__()
        self.g_enc = MLP(len, len)

    def forward(self, x):
        r"""
        Args:
            x: A number of vectors. x.shape = [batch_size, n, len]
            Notes: len节点特征长度为4较小, 因此需要升维到, 保证最后得到的feature_dim=128
        Returns: 
            All processed vectors with shape [batch_size, n, len*2]
        """
        assert len(x.shape) == 3
        x = self.g_enc(x) # [batch_size,n,len] --> [batch_size,n,16*len]
        batch_size, n, length = x.shape

        x2 = x.permute(0, 2, 1) # [batch_size, len, n]
        x2 = F.max_pool1d(x2, kernel_size=x2.shape[2])  # [batch_size, len, 1]
        x2 = torch.cat([x2] * n, dim=2)  # [batch_size, len, n]

        y = torch.cat((x2.permute(0, 2, 1), x), dim=2)
        assert y.shape == (batch_size, n, length*2)
        return y


class GlobalGraph(nn.Module):
    r"""
    Self-Attention module, corresponding the global graph.
    Given lots of polyline vectors, each length is 'C', we want to get the predicted feature vector.
    """

    def __init__(self, len, layers_number):
        r"""
        self.linear is 3 linear transformers for Q, K, V.

        Args:
            len: the length of input feature vector.
        """
        super(GlobalGraph, self).__init__()
        # self.linears = [nn.Linear(len, len) for _ in range(3)]
        self.linear1 = nn.Linear(len, len)
        self.linear2 = nn.Linear(len, len)
        self.linear3 = nn.Linear(len, len)
        self.layers_number = layers_number

    def last_layer(self, P):
        vector_num, node_feature_dim = P.shape
        P = P.unsqueeze(0)
        Q = self.linear1(P) # [1, vector_num, node_feature_dim]
        K = self.linear2(P)  
        V = self.linear3(P)
        ans = torch.matmul(Q, K.permute(0, 2, 1))
        ans = F.softmax(ans, dim=2)
        ans = torch.matmul(ans, V)
        ans.squeeze_(0)
        assert ans.shape == (vector_num, node_feature_dim)
        return ans

    def not_last_layer(self, P):
        assert False
        return P

    def forward(self, P):
        r"""
        Args:
            P: all polyline-level feature
                P.shape = [vector_num, node_feature_dim]
        
        Returns: 
            The feature after attention. [vector_num, node_feature_dim]
        """
        for i in range(self.layers_number - 1):
            P = self.not_last_layer(P)
        ans = self.last_layer(P)
        return ans
    
    
class VectorNet(nn.Module):
    r""" 
    Vector network.
    """

    def __init__(self, v_len=4, sub_layers=5, global_layers=1):
        r"""
        Construct a VectorNet.
        Args:
            v_len (int): node特征长度
            sub_layers (int): 
            global_layers (int): 
        """
        super(VectorNet, self).__init__()
        self.sub_graph = SubGraph(layers_number=sub_layers, v_len=v_len)
        self.p_len = v_len * (2 ** sub_layers)
        self.global_graph = GlobalGraph(len=self.p_len, layers_number=global_layers)

    def forward(self, polyline_list):
        r"""
        Note: Because different data has different number of agents, different agent has different number of vectors, so we
        choose batch_size=1
        
        Args:
            item_num (Tensor): [batch_size, 1], number of items
            target_id (Tensor, dtype=torch.int64): [batch_size, 1], prediction agent id
            polyline_list (list): list of polyline [N, len, node_feature_dim]
        
        Returns: 
            A tensor represents the embedding of prediction agent,
            shape is [batch_size, self.p_len]
        """
        # batch_size = item_num.shape[0]

        p_list = []
        for polyline in polyline_list:
            if polyline.shape == (1,1):
                feature = None
            else:
                polyline_start = polyline[:,:-1,:]
                polyline_end = polyline[:,1:,:]
                input = torch.cat((polyline_start, polyline_end), dim=-1).cuda() # vector_num, node_num, node_dim
                # polyline = torch.cat([polyline] * batch_size, axis=0) # [batch_size, v_number, v_len]
                p = self.sub_graph(input) # out: [vector_num, node_feature_dim]
                feature = self.global_graph(p) # [vector_num, node_feature_dim]
            p_list.append(feature)
        return p_list


@BACKBONES.register_module()
class VectorMLPEncoder(BaseModule):
    def __init__(self, input=40, output=128):
        super(VectorMLPEncoder, self).__init__()
        self.input = input
        self.output = output
        self.mlp = MLP(input, output, 64)

    def forward(self, x_list):
        p_list = []
        for x in x_list:
            if x.size() == torch.Size([1]):
                feature = None
            else:
                n, len, node_feat_dim = x.shape
                x = x.reshape(-1, len * node_feat_dim)
                feature = self.mlp(x)
            p_list.append(feature)
        return p_list

@BACKBONES.register_module()
class RasterCNNEncoder(BaseModule):
    def __init__(self, input=1, output=128):
        super(RasterCNNEncoder, self).__init__()
        self.input = input
        self.output = output
        self.conv_nn = nn.Sequential(
            nn.Conv2d(input, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, output, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(output, output, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(output),
            nn.ReLU(),
        )
    
    def init_weights(self):
        normal_init(self.conv_nn, mean=0, std=0.01)

    def forward(self, x):
        x = self.conv_nn(x)
        return x

@TRANSFORMER.register_module()
class mmBEVMapAttention(TransformerLayerSequence):
    def __init__(self, 
                 transformerlayers=None, 
                 num_layers=None, 
                 init_cfg=None,
                 encoder=None,
                 positional_encoding=None):
        super().__init__(transformerlayers, num_layers, init_cfg)
        self.encoder = build_backbone(encoder)
        self.positional_encoding = build_positional_encoding(positional_encoding)
        # bev_pos = self.positional_encoding(bev_mask).to(dtype)
        self.init_weights()
        print("init_weights")

    def init_weights(self):
        super().init_weights()
        self.encoder.init_weights()
    
    def forward(self,
                bev_embed,
                sd_map_data,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs):
        """
        Args:
            bev_embed: (bs, bev_h*bev_w, dim)
            sd_map_data: (bs, 1, bev_h, bev_w)
        """
        if sd_map_data is None:
            return bev_embed
        else:
            sd_feature = self.encoder(sd_map_data) # [b,c,h,w]
            b,c,h,w = sd_feature.shape
            dtype = bev_embed.dtype
            sd_pe_mask = torch.zeros((b, h, w), 
                device=sd_feature.device).to(dtype)
            sd_pe = self.positional_encoding(sd_pe_mask).to(dtype)

            query = bev_embed.permute(1, 0, 2) # [bs,hw,dim] -> [num_q, bs, dim]
            value = (sd_feature + sd_pe).reshape(b,c,-1).permute(2, 0, 1) 
            key = value
            for layer in self.layers:
                query = layer(
                    query,
                    key,
                    value,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_masks=attn_masks,
                    query_key_padding_mask=query_key_padding_mask,
                    key_padding_mask=key_padding_mask,
                    **kwargs)
            bev_fused = query.permute(1, 0, 2)
            return bev_fused
        
@TRANSFORMER.register_module()
class simplyConcat(BaseModule):
    def __init__(self, 
                 embed_dim=256,
                 sd_dim=1,):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(embed_dim+sd_dim, out_channels=embed_dim+sd_dim, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(embed_dim+sd_dim, out_channels=embed_dim, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )
        self.embed_dim = embed_dim
        self.init_weights()
        print("init_weights")

    def init_weights(self):
        super().init_weights()
        normal_init(self.encoder, mean=0, std=0.01)
    
    def forward(self,
                bev_embed,
                sd_map_data,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs):
        """
        Args:
            bev_embed: (bs, bev_h*bev_w, dim)
            sd_map_data: (bs, 1, bev_h, bev_w)
        """
        if sd_map_data is None:
            return bev_embed
        else:
            b, c, h, w = sd_map_data.shape
            bev_feature = bev_embed.permute(0,2,1).reshape(b, self.embed_dim, h, w)
            fused_feature = torch.concat((bev_feature, sd_map_data), dim=1)
            bev_fused = self.encoder(fused_feature)
            bev_fused = bev_fused.reshape(b, self.embed_dim, h*w).permute(0,2,1)
            return bev_fused

@TRANSFORMER.register_module()
class BEVMapAttention(BaseModule):
    def __init__(self, dim, encoder=None, bev_h=None, bev_w=None, use_concat=False):
        super(BEVMapAttention, self).__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.use_concat = use_concat
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.linear3 = nn.Linear(dim, dim)
        self.scale = torch.sqrt(torch.tensor(dim))
        # TODO: 增加positional encoding

        self.encoder = build_backbone(encoder)

    def forward(self, bev, sd_map_data):
        """
        Input:
            bev: [B, bev_hw, D]
            sdmap: list of tesnor or None, (number, 20, 2)
        Output:
            cross_attention_feature: [H, W, C]
        """
        # import pdb; pdb.set_trace()
        if sd_map_data is None:
            return bev
        else:
            vector_feature = self.encoder(sd_map_data)
            b, bev_hw, dim = bev.shape
            # bev = bev.reshape(b, self.bev_h, self.bev_w, dim)
            tmp_bev_list = []
            for (idx, sd_map_feat) in enumerate(vector_feature):
                tmp_bev = bev[idx].clone().detach()
                if sd_map_feat is not None:
                    # H, W, C = tmp_bev.shape
                    tmp_bev = tmp_bev.unsqueeze(0) # add batch channel
                    sd_map_feat = sd_map_feat.unsqueeze(0)
                    Q = self.linear1(tmp_bev)
                    K = self.linear2(sd_map_feat)  
                    V = self.linear3(sd_map_feat)
                    ans = torch.matmul(Q, K.permute(0, 2, 1)) / self.scale
                    ans = nn.functional.softmax(ans, dim=2)
                    ans = torch.matmul(ans, V)
                    ans = ans.squeeze(0)
                    if self.use_concat:
                        tmp_bev_list.append(ans)
                    else:
                        bev[idx] = bev[idx] + ans * 0.1
                else:
                    if self.use_concat:
                        tmp_bev_list.append(tmp_bev)
            # bev = bev.reshape(b, bev_hw, dim)
            # import pdb; pdb.set_trace()
            if self.use_concat:
                return torch.concat((bev, torch.stack(tmp_bev_list)), dim=-1)
            return bev

@TRANSFORMER.register_module()
class QueryAttention(TransformerLayerSequence):
    def __init__(self, 
                 transformerlayers=None, 
                 num_layers=None, 
                 init_cfg=None,
                 encoder=None,
                 positional_encoding=None):
        super().__init__(transformerlayers, num_layers, init_cfg)
        self.encoder = build_backbone(encoder)
        self.positional_encoding = None
        if positional_encoding is not None:
            self.positional_encoding = build_positional_encoding(positional_encoding)
        self.init_weights()

    def init_weights(self):
        super().init_weights()
        self.encoder.init_weights()
    
    def forward(self,
                query,
                sd_map_data,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs):
        """
        Args:
            query: (num_q, bs, dim)
            sd_map_data: (bs, 1, bev_h, bev_w)
        """
        if sd_map_data is None:
            return query
        else:
            sd_feature = self.encoder(sd_map_data) # [b,c,h,w]
            b,c,h,w = sd_feature.shape
            dtype = query.dtype
            sd_pe_mask = torch.zeros((b, h, w), 
                device=sd_feature.device).to(dtype)
            sd_pe = self.positional_encoding(sd_pe_mask).to(dtype)

            query = query.permute(1, 0, 2) # [bs,hw,dim] -> [num_q, bs, dim]
            value = (sd_feature + sd_pe).reshape(b,c,-1).permute(2, 0, 1) 
            key = value
            for layer in self.layers:
                query = layer(
                    query,
                    key,
                    value,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_masks=attn_masks,
                    query_key_padding_mask=query_key_padding_mask,
                    key_padding_mask=key_padding_mask,
                    **kwargs)
            bev_fused = query.permute(1, 0, 2)
            return bev_fused