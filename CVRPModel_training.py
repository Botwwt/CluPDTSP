# CVRPModel_training.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from CVRProblemDef import training_augment

# GatingModule
class GatingModule(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3 * embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )
    def forward(self, encoded_last_node, encoded_current_cluster, encoded_other_cluster):
        combined_input = torch.cat(
            (encoded_last_node, encoded_current_cluster, encoded_other_cluster), 
            dim=2
        )
        p_stay = self.mlp(combined_input)
        return p_stay

class CVRPModel(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        self.enable_encoder_cluster = model_params.get('enable_encoder_cluster', True)
        self.enable_decoder_cluster = model_params.get('enable_decoder_cluster', True)
        self.use_learned_gate = model_params.get('use_learned_gate', False)
        if self.use_learned_gate and not self.enable_decoder_cluster:
            self.use_learned_gate = False

        self.encoder = CVRP_Encoder(**model_params)
        self.decoder_inter = CVRP_Decoder(**model_params)
        self.decoder_intra = TSP_Decoder(**model_params)
        
        if self.use_learned_gate and self.enable_decoder_cluster:
            self.gate = GatingModule(embedding_dim)
            print(">> CVRPModel: Using STRATEGIC LEARNED GATING mechanism (with other-cluster info).")
        else:
            self.gate = None
            if self.enable_decoder_cluster:
                print(">> CVRPModel: Using PERFECTED RULE-BASED switching.")
            else:
                print(">> CVRPModel: Decoder cluster disabled; using single TSP decoder.")

        self.encoded_nodes = None
        self.cluster_embeddings = None
        self.cluster_list = None

    def pre_forward(self, reset_state):
        depot_xy = reset_state.depot_xy
        node_xy = reset_state.node_xy
        node_demand_encode = reset_state.node_demand_encode
        node_clus = reset_state.node_clus
        depot_clus_val = torch.zeros(reset_state.cluster_list.size(0), 1, device=reset_state.cluster_list.device, dtype=reset_state.cluster_list.dtype)
        cluster_list = torch.cat((depot_clus_val, reset_state.cluster_list), dim=1)
        depot_aug = reset_state.depot_aug
        node_aug = reset_state.node_aug
        mask_clu = (cluster_list.unsqueeze(2) == cluster_list.unsqueeze(1)).float()
        node_xy_demand = torch.cat((node_aug, node_demand_encode[:, :, None], node_clus), dim=2)
        self.encoded_nodes = self.encoder(depot_aug, node_xy_demand, mask_clu)
        if self.enable_decoder_cluster:
            self.cluster_list = cluster_list
            self.cluster_embeddings = cal_cluster_embeddings(self.encoded_nodes, self.cluster_list, num_clusters=3)
            self.decoder_inter.set_kv(self.encoded_nodes)
        else:
            self.cluster_list = None
            self.cluster_embeddings = None
        self.decoder_intra.set_kv(self.encoded_nodes)

    def forward(self, state):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)
        device = state.BATCH_IDX.device

        if state.selected_count == 0:
            selected = torch.zeros(size=(batch_size, pomo_size), dtype=torch.long, device=device)
            prob = torch.ones(size=(batch_size, pomo_size), device=device)
        else:
            first_node_idx = state.first_node_in_cluster
            encoded_first_node = _get_encoding(self.encoded_nodes, first_node_idx)
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            
            probs_intra = self.decoder_intra(encoded_first_node, encoded_last_node, ninf_mask=state.ninf_mask)

            if not self.enable_decoder_cluster:
                probs = probs_intra
            else:
                if state.current_cluster is None:
                    current_cluster_idx = torch.zeros_like(state.current_node)
                else:
                    current_cluster_idx = state.current_cluster
                
                encoded_current_cluster = _get_cluster(self.cluster_embeddings, current_cluster_idx)
                probs_inter = self.decoder_inter(encoded_last_node, encoded_current_cluster, ninf_mask=state.ninf_mask)
                
                if self.use_learned_gate:
                    is_at_depot = (current_cluster_idx == 0)
                    other_cluster_idx = torch.where(is_at_depot, torch.ones_like(current_cluster_idx), 3 - current_cluster_idx)
                    encoded_other_cluster = _get_cluster(self.cluster_embeddings, other_cluster_idx)
                    p_stay = self.gate(encoded_last_node, encoded_current_cluster, encoded_other_cluster)

                    expanded_current_cluster = current_cluster_idx.unsqueeze(-1).expand(batch_size, pomo_size, self.cluster_list.size(1))
                    expanded_cluster_list = self.cluster_list.unsqueeze(1).expand(batch_size, pomo_size, self.cluster_list.size(1))
                    
                    stay_mask = (expanded_cluster_list == expanded_current_cluster)
                    switch_mask = (expanded_cluster_list != expanded_current_cluster) & (expanded_cluster_list != 0)
                    
                    valid_intra_probs = probs_intra * stay_mask.float()
                    valid_inter_probs = probs_inter * switch_mask.float()
                    
                    probs = p_stay * valid_intra_probs + (1 - p_stay) * valid_inter_probs
                    
                    probs[:,:,0] = probs_intra[:, :, 0]

                    # ========================= ### Fallback logic ### =========================
                    probs_sum = probs.sum(dim=2, keepdim=True)
                    is_stuck_mask_L1 = (probs_sum == 0)

                    if is_stuck_mask_L1.any():
                        fallback_probs_L1 = valid_intra_probs + valid_inter_probs
                        fallback_probs_L1[:,:,0] = probs_intra[:, :, 0]
                        probs = torch.where(is_stuck_mask_L1, fallback_probs_L1, probs)
                    
                    probs_sum = probs.sum(dim=2, keepdim=True)
                    is_stuck_mask_L2 = (probs_sum == 0)

                    if is_stuck_mask_L2.any():
                        fallback_probs_L2 = (state.ninf_mask == 0).float()
                        probs = torch.where(is_stuck_mask_L2, fallback_probs_L2, probs)

                    final_probs_sum = probs.sum(dim=2, keepdim=True)
                    final_probs_sum[final_probs_sum == 0] = 1.0
                    probs = probs / final_probs_sum
                    # ==============================================================================

                else: # Rule-based switching
                    in_cluster = state.in_cluster.unsqueeze(-1).expand_as(probs_inter)
                    probs = torch.where(in_cluster, probs_intra, probs_inter)

            eval_type = self.model_params.get('eval_type', 'greedy')

            if self.training:
                probs = probs + 1e-10 
                while True:
                    with torch.no_grad():
                        selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1).squeeze(dim=1).reshape(batch_size, pomo_size)
                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)
                    if (prob > 0).all():
                        break
            else:
                if eval_type == 'greedy':
                    selected = probs.argmax(dim=2)
                    prob = None
                elif eval_type == 'sampling' or eval_type == 'pomo':
                    probs = probs + 1e-10
                    selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1).squeeze(dim=1).reshape(batch_size, pomo_size)
                    prob = None
                else:
                    raise ValueError(f"Unknown eval_type: {eval_type}")
        
        return selected, prob

# ==============================================================================
# All required helper functions and classes are included below.
# ==============================================================================

def cal_cluster_embeddings(node_embeddings, cluster_list, num_clusters):
    one_hot_clusters = torch.nn.functional.one_hot(cluster_list, num_classes=num_clusters)
    one_hot_clusters = one_hot_clusters.unsqueeze(-1).float()
    node_embeddings_exp = node_embeddings.unsqueeze(2)
    cluster_embedding_sum = (one_hot_clusters * node_embeddings_exp).sum(dim=1)
    cluster_count = one_hot_clusters.sum(dim=1).float()
    cluster_count[cluster_count == 0] = 1
    cluster_embedding = cluster_embedding_sum / cluster_count
    return cluster_embedding

def _get_encoding(encoded_nodes, node_index_to_pick):
    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)
    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    return picked_nodes

def _get_cluster(cluster_embeddings, cluster_index_to_pick):
    batch_size = cluster_index_to_pick.size(0)
    pomo_size = cluster_index_to_pick.size(1)
    embedding_dim = cluster_embeddings.size(2)
    gathering_index = cluster_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    picked_clusters = cluster_embeddings.gather(dim=1, index=gathering_index)
    return picked_clusters

class CVRP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']
        self.embedding_depot = nn.Linear(16, embedding_dim)
        self.embedding_node = nn.Linear(19, embedding_dim)
        self.cross_attn = CrossAttnLayer(**model_params)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])
    def forward(self, depot_xy, node_in, mask=None):
        embedded_depot = self.embedding_depot(depot_xy)
        embedded_node = self.embedding_node(node_in)
        crossed_depot = self.cross_attn(embedded_depot, embedded_node)
        out = torch.cat((crossed_depot, embedded_node), dim=1)
        for layer in self.layers:
            out = layer(out, mask)
        return out

class CrossAttnLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        self.enable_encoder_cluster = model_params.get('enable_encoder_cluster', True)
        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.add_n_normalization_1 = AddAndInstanceNormalizationCross(**model_params)
        self.feed_forward = FeedForward(**model_params)
        self.add_n_normalization_2 = AddAndInstanceNormalizationCross(**model_params)
    def forward(self, depot, customer):
        head_num = self.model_params['head_num']
        q = reshape_by_heads(self.Wq(depot), head_num=head_num)
        k = reshape_by_heads(self.Wk(customer), head_num=head_num)
        v = reshape_by_heads(self.Wv(customer), head_num=head_num)
        out_concat = multi_head_attention(q, k, v)
        multi_head_out = self.multi_head_combine(out_concat)
        out1 = self.add_n_normalization_1(depot, multi_head_out)
        out2 = self.feed_forward(out1)
        out3 = self.add_n_normalization_2(out1, out2)
        return out3

class AddAndInstanceNormalizationCross(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
    def forward(self, input1, input2):
        added = input1 + input2
        transposed = added.transpose(1, 2)
        back_trans = transposed.transpose(1, 2)
        return back_trans

class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.enable_encoder_cluster = model_params.get('enable_encoder_cluster', True)
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.Wq2 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk2 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv2 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.add_n_normalization_1 = AddAndInstanceNormalization(**model_params)
        self.feed_forward = FeedForward(**model_params)
        self.add_n_normalization_2 = AddAndInstanceNormalization(**model_params)
    def forward(self, input1, mask):
        head_num = self.model_params['head_num']
        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        out_concat = multi_head_attention(q, k, v)
        if self.enable_encoder_cluster and mask is not None:
            q2 = reshape_by_heads(self.Wq2(input1), head_num=head_num)
            k2 = reshape_by_heads(self.Wk2(input1), head_num=head_num)
            v2 = reshape_by_heads(self.Wv2(input1), head_num=head_num)
            out_concat2 = cross_cluster_attention(q2, k2, v2, mask)
            out_concat = out_concat + out_concat2
        multi_head_out = self.multi_head_combine(out_concat)
        out1 = self.add_n_normalization_1(input1, multi_head_out)
        out2 = self.feed_forward(out1)
        out3 = self.add_n_normalization_2(out1, out2)
        return out3

class CVRP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        self.Wq_last = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_cluster = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.k = None
        self.v = None
        self.single_head_key = None
    def set_kv(self, encoded_nodes):
        head_num = self.model_params['head_num']
        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        self.single_head_key = encoded_nodes.transpose(1, 2)
    def forward(self, encoded_last_node, encoded_last_cluster, ninf_mask):
        head_num = self.model_params['head_num']
        q_last = reshape_by_heads(self.Wq_last(encoded_last_node), head_num=head_num)
        q_cluster = reshape_by_heads(self.Wq_cluster(encoded_last_cluster), head_num=head_num)
        q = q_last + q_cluster
        out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        mh_atten_out = self.multi_head_combine(out_concat)
        score = torch.matmul(mh_atten_out, self.single_head_key)
        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']
        score_scaled = score / sqrt_embedding_dim
        score_clipped = logit_clipping * torch.tanh(score_scaled)
        score_masked = score_clipped + ninf_mask
        probs = F.softmax(score_masked, dim=2)
        return probs

class TSP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        self.Wq_first = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_last = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.k = None
        self.v = None
        self.single_head_key = None
    def set_kv(self, encoded_nodes):
        head_num = self.model_params['head_num']
        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        self.single_head_key = encoded_nodes.transpose(1, 2)
    def forward(self, encoded_first_node, encoded_last_node, ninf_mask):
        head_num = self.model_params['head_num']
        q_last = reshape_by_heads(self.Wq_last(encoded_last_node), head_num=head_num)
        q_first = reshape_by_heads(self.Wq_first(encoded_first_node), head_num=head_num)
        q = q_first + q_last
        out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        mh_atten_out = self.multi_head_combine(out_concat)
        score = torch.matmul(mh_atten_out, self.single_head_key)
        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']
        score_scaled = score / sqrt_embedding_dim
        score_clipped = logit_clipping * torch.tanh(score_scaled)
        score_masked = score_clipped + ninf_mask
        probs = F.softmax(score_masked, dim=2)
        return probs

def reshape_by_heads(qkv, head_num):
    batch_s, n, _ = qkv.size()
    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    q_transposed = q_reshaped.transpose(1, 2)
    return q_transposed

def cross_cluster_attention(q, k, v, mask):
    batch_s, head_num, n, key_dim = q.size()
    score = torch.matmul(q, k.transpose(2, 3))
    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float, device=q.device))
    if mask is not None:
        mask = mask.unsqueeze(1).expand(batch_s, head_num, n, n)
        score_scaled = score_scaled.masked_fill(mask == 0, float('-inf'))
    weights = nn.Softmax(dim=3)(score_scaled)
    out = torch.matmul(weights, v)
    out_transposed = out.transpose(1, 2)
    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    return out_concat

def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    batch_s, head_num, n, key_dim = q.size()
    input_s = k.size(2)
    score = torch.matmul(q, k.transpose(2, 3))
    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float, device=q.device))
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)
    weights = nn.Softmax(dim=3)(score_scaled)
    out = torch.matmul(weights, v)
    out_transposed = out.transpose(1, 2)
    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    return out_concat

class AddAndInstanceNormalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)
    def forward(self, input1, input2):
        added = input1 + input2
        transposed = added.transpose(1, 2)
        normalized = self.norm(transposed)
        back_trans = normalized.transpose(1, 2)
        return back_trans

class FeedForward(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']
        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)
    def forward(self, input1):
        return self.W2(F.relu(self.W1(input1)))
