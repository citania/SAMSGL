import torch as t
import torch.fft
from torch.nn.utils import weight_norm


class GenericBasis(t.nn.Module):
    """
    Generic basis function.
    """
    def __init__(self, backcast_size: int, forecast_size: int):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    def forward(self, theta: t.Tensor):
        return theta[:, :self.backcast_size], theta[:, -self.forecast_size:]


class Align_series(t.nn.Module):
    # Cross-Correlation for spatial modeling of intermedia nodes
    def __init__(self, n_heads, d_model,time_layer, d_keys = None, d_values = None):
        super(Align_series, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.n_heads = n_heads
        self.factor = 1
        self.query_projection = t.nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = t.nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = t.nn.Linear(d_model, d_keys * n_heads)
        self.out_projection = t.nn.Linear(d_values * n_heads, d_model)
        self.time_over = time_layer

    def align_speed(self, values, corr):
        # align (speed up)
        # B N H E L
        batch = values.shape[0]
        node = values.shape[1]
        head = values.shape[2]
        channel = values.shape[3]
        length = values.shape[4]
        # index
        init_index = t.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, node, head,
                                                                                                     channel, 1).cuda()#每个子项都是0：47
        # align
        rank = t.topk(t.mean(corr, dim=3), self.factor, dim=-1)#B，node, head,1
        weight = rank[0]
        delay = rank[1]
        delay = length - delay
        # cal res
        tmp_values = values.repeat(1, 1, 1, 1, 2)
        aligned_values = []
        for i in range(self.factor):
            tmp_delay = init_index + delay[..., i] \
                .unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, channel, 1)
            aligned_values.append(t.gather(tmp_values, dim=-1, index=tmp_delay))
        return aligned_values, delay, weight

    def align_back_speed(self, values, delay):
        # align back (speed up)
        # B N H E L
        batch = values.shape[0]
        node = values.shape[1]
        head = values.shape[2]
        channel = values.shape[3]
        length = values.shape[4]
        # index
        init_index = t.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, node, head,
                                                                                                     channel, 1).cuda()
        # algin back
        delay = length - delay
        # cal res
        agg_values = []
        tmp_values = values.repeat(1, 1, 1, 1, 2)
        tmp_delay = init_index + (delay[..., 0]) \
            .unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, channel, 1)  # B N H
        aligned_values = t.gather(tmp_values, dim=-1, index=tmp_delay)  # B N H E L
        agg_values.append(aligned_values)
        agg_values = t.mean(t.stack(agg_values, dim=0), dim=0)  # mean or sum
        return agg_values


    def forward(self, queries, keys, values, ):
        # x B, T_in, N_nodes, F_in
        #make sure the output shape: batch_size, num_of_timesteps, num_of_vertices, in_channels
        B, L, N, E = queries.shape
        _, S, _, D = values.shape
        H = self.n_heads

        queries = self.query_projection(queries.permute(0,2,1,3)).view(B, N, L, H, -1)
        keys = self.key_projection(keys.permute(0,2,1,3)).view(B, N, S, H, -1)
        values = self.value_projection(values.permute(0,2,1,3)).view(B, N, S, H, -1)

        _, _, S, _, D = values.shape
        if L > S:
            zeros = t.zeros_like(queries[:, :, :(L - S), :, :]).float()
            values = t.cat([values, zeros], dim=2)
            keys = t.cat([keys, zeros], dim=2)
        else:
            values = values[:, :, :L, :, :]
            keys = keys[:, :, :L, :, :]

        # cross-correlation for spatial modeling
        ## average pool for pivot series
        q_fft = torch.fft.rfft(
            t.mean(queries, dim=1).unsqueeze(1).repeat(1, N, 1, 1, 1).permute(0, 1, 3, 4, 2).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 1, 3, 4, 2).contiguous(), dim=-1)
        ## pivot-based cross correlation
        res = q_fft * t.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)  # B N H E L
        # align
        aligned_values, delay, weight = self.align_speed(values.permute(0, 1, 3, 4, 2).contiguous(), corr)#aligned_value 按
        # (weight B N H K, delay B N H K, aligned_values B N H E L Key
        delay = delay.permute(0, 2, 1, 3).contiguous().view(B * H, N, self.factor)#B, H, N, K
        weight = weight.permute(0, 2, 1, 3).contiguous().view(B * H, N, self.factor)#B, H, N, K
        #B N H E L
        EE = E//H
        aligned_values_t = aligned_values[0].view(B, N, H*EE, L).permute(0, 3, 1, 2)
        ####
        _, indices = torch.sort(delay[..., 0])
        sort_indices = indices[:, :, None, None].repeat(1, 1, EE, L)
        # rearrange
        sorted_aligned_values = (
                (aligned_values[0].permute(0, 2, 1, 3, 4).contiguous().view(B * H, N, EE, L))
                * torch.sigmoid(weight[..., 0])[:, :, None, None].repeat(1, 1, EE, L)) \
            .gather(dim=1, index=sort_indices)  # B*H N E L * B*H N E L
        # aggregate
        sorted_aligned_values = sorted_aligned_values.view(B * H, N, EE * L).permute(0, 2, 1).contiguous()  # B*H, E*L, N
        sorted_aligned_values = self.time_over(sorted_aligned_values)
        sorted_aligned_values = sorted_aligned_values.permute(0, 2, 1).contiguous().view(B * H, N, EE, L)
        # sort back
        _, reverse_indices = torch.sort(indices)  # B*H N
        reverse_indices = reverse_indices[:, :, None, None].repeat(1, 1, EE, L)  # B*H N E L
        # rearrange back
        sorted_aligned_values = sorted_aligned_values.gather(dim=1, index=reverse_indices)  # B*H N E L
        sorted_aligned_values = sorted_aligned_values.view(B, H, N, EE, L).permute(0, 2, 1, 3, 4).contiguous()
        # causal_agg_values.append(sorted_aligned_values)  # B, N, H, E, L
        # align back

        return aligned_values_t.contiguous(), delay, weight, sorted_aligned_values.view(B, N, H*EE, L).permute(0, 3, 1, 2)

    def backward(self, value, delay):
        # align back
        B, L, N, E = value.shape
        H = self.n_heads
        value = value.permute(0, 2, 3, 1).contiguous()

        delay = delay.view(B, H, N, self.factor).permute(0, 2, 1, 3).contiguous()
        V = self.align_back_speed(value.view(B, N, H, E//H, L), delay).permute(0, 1, 4, 2, 3).view(B, N, L, -1)  # B N L H E
        return V.permute(0, 2, 1, 3)

class dynamic_graph_learner(t.nn.Module):
    # use all the eigenvectors
    def __init__(self, u, e, in_channel, out_channel, alpha, node_num, mlp_list):
        super(dynamic_graph_learner, self).__init__()
        self.u = u# n,n
        self.e = e# n,
        self.alpha = alpha
        self.gru = t.nn.GRU(input_size=in_channel, hidden_size=out_channel)
        self.mlp = t.nn.ModuleList()
        self.mlp.append(t.nn.Linear(in_features=out_channel, out_features=mlp_list[0]))
        self.mlp.extend(
            [t.nn.Linear(in_features=mlp_list[i], out_features=mlp_list[i + 1]) for i in range(1, len(mlp_list) - 1)])
        self.eig_fuse = t.nn.Linear(node_num*2, node_num)

    def forward(self, x):
        L, B, N, C = x.size()
        x = x.reshape([L, B*N, C])
        h = self.gru(x).reshape([B, N, -1]) # 1, B*N, C -->B,N,C
        #str 1
        eig_value = self.mlp(h).squeeze(-1)#B,N
        eig_value = self.e + self.alpha * eig_value
        #str 2
        eig_value = self.mlp(h).squeeze(-1)  # B,N
        eig_value = self.mlp(torch.cat([self.e.unsqueeze(0).repeat(B,1),eig_value]))#
        eig_value_m = torch.stack([torch.diag(eig_value[i]) for i in range(B)],axis =0)

        g = torch.matmul(torch.matmul(self.u.transpose(), eig_value_m), self.u)#B, N, N
        return g# B,N,N

class dynamic_graph_learner_k(t.nn.Module):
    # use all the eigenvectors
    def __init__(self, u_l, u_s, k, e, in_channel, out_channel, alpha, node_num, mpl_list):
        super(dynamic_graph_learner_k, self).__init__()
        self.u_l = u_l# n,k
        self.u_s = u_s
        self.e = e# k,k
        self.alpha = alpha
        self.gru = t.nn.GRU(input_size=in_channel, hidden_size=out_channel)
        self.mlp = t.nn.ModuleList()
        self.mlp.append(t.nn.Linear(in_features=out_channel,out_features=mpl_list[0]))
        self.mlp.extend([t.nn.Linear(in_features=mpl_list[i],out_features=mpl_list[i+1]) for i in range(1,len(mpl_list)-1)])
        self.generate_e = t.nn.Linear(node_num, 2*k)

    def forward(self, x):
        L, B, N, C = x.size()
        x = x.reshape([L, B*N, C])
        h = self.gru(x).reshape([B, N, -1]) # 1, B*N, C -->B,N,C
        #str 1
        eig_value = self.mlp(h).squeeze(-1)#B,N
        eig_value = self.generate_e(eig_value)
        eig_value = self.e + self.alpha * eig_value
        #str 2
        eig_value = self.mlp(h).squeeze(-1)  # B,N
        eig_value = self.generate_e(torch.cat([self.e.unsqueeze(0).repeat(B,1),eig_value]))#

        eig_value_m = torch.stack([torch.diag(eig_value[i]) for i in range(B)],dim =0)

        g = torch.matmul(torch.matmul(self.u.transpose(), eig_value_m), self.u)#B, N, N
        return g# B,N,N




