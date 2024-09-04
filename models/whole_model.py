import torch
import torch.nn as nn
import torch.nn.functional as F
from models.transformer_basics import (TranformerConfig, BertLayerNorm, BertSelfAttention, LayerNormChannel)
from models.baseline import build_hrnet
from models.SMPL_predictor import SMPLPredictor
from models.SMPL_handler import SMPLHandler
import config as project_cfg
from utils.geometry import rot6d_to_rotmat
from models.gcn import GraphResBlock







class Graphformer(nn.Module):
    """
    input -> (Norm) -> (Attention)  -> (Linear) -> (Drop) -> (Add) -> (GCN)
              -> (Norm) -> (Linear) -> (Gelu) -> (Linear) -> (Drop) -> (Add) -> output
    """
    def __init__(self, config):
        super().__init__()
        self.LayerNorm_atten = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = BertSelfAttention(config)  # attention layer
        self.linear_attn = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.gelu = nn.GELU()

        # Add GCN Layer
        self.gcn_layer = GraphResBlock(config.hidden_size, config.hidden_size, mesh_type='body')

        LayerNorm_mlp = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        mlp_linear1 = nn.Linear(config.hidden_size, config.hidden_size * config.interm_size_scale)
        mlp_linear2 = nn.Linear(config.hidden_size * config.interm_size_scale, config.hidden_size)
        intermediate_act_fn = nn.GELU()
        self.mlp = nn.Sequential(LayerNorm_mlp, mlp_linear1, intermediate_act_fn, mlp_linear2, nn.Dropout(config.hidden_dropout_prob))   # changed to use a new dropout just for clarity



    def forward(self, hidden_states, key_states=None, value_states=None):
        hidden_states_norm = self.LayerNorm_atten(hidden_states)
        if key_states is None: 
            # self-attention
            key_states = hidden_states_norm
            value_states = hidden_states_norm
        elif value_states is None:
            # cross-attention
            value_states = key_states

        # print("hidden_state: ",hidden_states.shape)
        # print("key_states: ", key_states.shape)
        attention_output = self.attention(hidden_states_norm, key_states, value_states)
        attention_output = self.linear_attn(attention_output)
        attention_output = self.dropout(attention_output)
        attention_output = attention_output + hidden_states

        # attention_output:[b, k+2, h]
        joints = attention_output[:, 0:24, :]
        others = attention_output[:, -2:, :]
        gcn_output = self.gcn_layer(joints)
        # print("gcn_outputs: ", gcn_output.shape)
        # print("others: ", others.shape)
        gcn_output = torch.cat([gcn_output, others], dim=1)
        gcn_output = gcn_output + attention_output
        gcn_output = self.gelu(gcn_output)


        mlp_output = self.mlp(gcn_output)
        #mlp_output = self.mlp(attention_output)

        return mlp_output + gcn_output
        #return mlp_output + attention_output



        




class graphtrans_unit(nn.Module):
    def __init__(self, config):
        super().__init__()
        # feat:[b,c,h,w]
        # q = cross_Attention(F, T)
        # self_Attention(q)
        # self.corss_attention =
        self.trans_layer = Graphformer(config=config)

    def forward(self, query, feat):
        co_query = self.trans_layer(query, feat.flatten(-2).transpose(1,2))
        return self.trans_layer(co_query)




class GraphTransformer(nn.Module):
    def __init__(self, config, pos_embeddings= None):
        super().__init__()
        self.config = config
        # query position encoding
        self.early_embedding = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size) if pos_embeddings is None else pos_embeddings
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        feat_res_list = config.feat_res_list
        self.feat_position_embedding = nn.Parameter(torch.randn(1, config.hidden_size, feat_res_list[0], feat_res_list[0]) * 0.1)
        self.pooling = nn.AvgPool2d(kernel_size=8, stride=8)

        self.trans_units = nn.ModuleList([graphtrans_unit(config) for _ in range(config.num_units_per_block)])

        # output
        self.head = nn.Linear(config.hidden_size, config.hidden_size)
        self.residual = nn.Linear(config.hidden_size, config.hidden_size)

        self.apply(self.init_weights)

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_() 
    

    def forward(self, query, feat):

        b, seq_length, c = query.shape
        query_embed = self.early_embedding(query)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=query.device)
        # [seq_length]——>[1, seq_length]——>[b, seq_length]
        position_embeddings = self.position_embeddings(position_ids)
        query_embed = position_embeddings + query_embed
        query_embed = self.dropout(query_embed)
        # print("query's shape: ", query.shape)
        # print("hidden_size: ", self.config.hidden_size)
        # print("config.feat_res_list: ", self.config.feat_res_list)
        # print("feat in GraphTrans: ", feat.shape)
        # print("feat_position_embd:", self.feat_position_embedding.shape)

        # feature position embedding
        
        new_feat = self.pooling(self.feat_position_embedding) + feat

        # apply units
        for unit in self.trans_units:
            query_embed = unit(query_embed, new_feat)

        # output
        # output
        output = self.head(query_embed) + self.residual(query)
        return output 





class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.GraphTransformer = GraphTransformer(cfg)

        self.theta_predictor = nn.Linear(cfg.hidden_size, 6)
        self.beta_predictor = nn.Linear(cfg.hidden_size, 10)
        self.cam_predictor = nn.Linear(cfg.hidden_size, 3)

        self.raw_feat_projection_layer = nn.Sequential(
            nn.Conv2d(self.cfg.raw_feat_dim[-1], cfg.hidden_size, 1,1),
            LayerNormChannel(cfg.hidden_size, cfg.layer_norm_eps)
        )

        

        self.init_predictors()

    @torch.no_grad()
    def init_predictors(self):
        self.beta_predictor.weight.data.normal_(0, 1e-3)
        self.beta_predictor.bias.data.zero_()
        self.cam_predictor.weight.data.normal_(0, 1e-3)
        self.cam_predictor.bias.data.zero_()

        self.theta_predictor.weight.data.normal_(0, 1e-3)
        self.theta_predictor.bias.data.zero_()
        self.theta_predictor.bias.data[[0,3]] = 0.1


    def forward(self, query, feat):
        """
        input:
        - query: [b, k+2, c]
        - feat: [b,c,h,w]

        output:
        - rotmat: [b, k, 3, 3]
        - beta: [b, 10]
        - cam: [b, 3]
        - query_new: [b, k+2, c]

        """
        new_feat = self.raw_feat_projection_layer(feat)
        #print("new_feat: ", new_feat.shape)
        
        # GraphTransformer
        query_new = self.GraphTransformer(query, new_feat)

        # predictors
        theta = self.theta_predictor(query_new[:, :-2])
        rotmat = rot6d_to_rotmat(theta).reshape(theta.shape[0], -1, 3, 3)
        beta = self.beta_predictor(query_new[:, -2])
        cam = self.cam_predictor(query_new[:, -1])
        return rotmat, beta, cam, query_new


 



class model0(nn.Module):
    def __init__(self, args, trans_cfg=None):
        super().__init__()

        # for tans config 
        if trans_cfg is None:
            trans_cfg = TranformerConfig()
        
        # initial estimation
        self.backbone = build_hrnet(args.hrnet_type) # use w32 or w48
        # predictor inital (theat, beta, cam)
        self.SMPL_predictor = SMPLPredictor(args = args) 
        # handle vertices & 3d joints & 2d joints 
        if args.data_mode == 'h36m':
            self.smpl_handler = SMPLHandler(path_to_regressor=project_cfg.JOINT_REGRESSOR_H36M_correct)
        else:
            self.smpl_handler = SMPLHandler(path_to_regressor=project_cfg.JOINT_REGRESSOR_3DPW)

        self.backbone_feat_dims = project_cfg.hrnet_dict[args.hrnet_type][2]
        # *** Linear layers for query initialization ***
        hidden_size = trans_cfg.hidden_size 

        self.init_query_layer_beta = nn.Linear(10, hidden_size)
        self.init_query_layer_theta = nn.Linear(3*3, hidden_size)
        self.init_query_layer_cam = nn.Linear(3, hidden_size)
        self.global_feat_projection = nn.Linear(self.backbone_feat_dims[-1], hidden_size)

        # *** Transformer Blocks ***
        self.num_transformers = args.num_transformers
        transformer_list = []
        for i in range(self.num_transformers):
            transformer_list.append(TransformerBlock(trans_cfg))
        self.transformer = nn.ModuleList(transformer_list)
        # ***************************

    def compute_init_query(self, pred_rotmat, pred_shape, pred_cam, global_feat):
        # 
        query_beta = self.init_query_layer_beta(pred_shape)
        query_cam = self.init_query_layer_cam(pred_cam)
        query_theta = self.init_query_layer_theta(pred_rotmat.flatten(2))

        query = torch.cat([query_theta, query_beta.unsqueeze(1), query_cam.unsqueeze(1)], dim=1)    # [b, 24+2, c]
        global_feat = self.global_feat_projection(global_feat)
        
        return query + global_feat.unsqueeze(1)



    def forward(self, img):
        """
        img --> (backbone) --> feat_list --> global_feat
        global --> (SMPL_predictor) --> theat, beta, cam
        theta, beta, cam, global_feat --> (compute_init_query) --> query

        query --> (GraphTansformer) --> new_theta, new_beta, new_cam
        """
        feat = self.backbone(img)
        #print(feat.shape)
        feat_mean = feat.flatten(2).mean(2)
        pred_rotmat, pred_shape, pred_cam = self.SMPL_predictor(feat_mean, global_pooling=False) 
        pred_smpl_dict = self.smpl_handler(pred_rotmat, pred_shape, theta_form='rot-matrix', cam=pred_cam)
        # initialize query for Transformer
        #print(feat_mean.shape)
        query = self.compute_init_query(pred_rotmat, pred_shape, pred_cam, feat_mean)

        # iterate  GraphTransformers
        # --- Run all Transformers ---
        pred_smpl_dicts = [pred_smpl_dict]
        for i in range(self.num_transformers):
            delta_theta, delta_beta, delta_cam, query = self.transformer[i](query, feat)

            pred_shape = pred_shape + delta_beta
            pred_cam = pred_cam + delta_cam
            pred_rotmat = torch.matmul(pred_rotmat, delta_theta)
            pred_smpl_dict = self.smpl_handler(pred_rotmat, pred_shape, theta_form='rot-matrix', cam=pred_cam)
            pred_smpl_dicts.append(pred_smpl_dict)
        # --- End of Transformers ---

        return pred_smpl_dicts







        



