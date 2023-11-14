import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..seq2seq import Seq2SeqAttrs
from .gfc import SAMSGL_NET
from .. local_g import local_graph
import warnings
Tensor = torch.Tensor

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def gumbel_softmax(logits: Tensor, tau: float = 1, hard: bool = False, eps: float = 1e-10, dim: int = -1) -> Tensor:
    r"""
    Samples from the Gumbel-Softmax distribution (`Link 1`_  `Link 2`_) and optionally discretizes.

    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd
      dim (int): A dimension along which softmax will be computed. Default: -1.

    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across `dim`.

    .. note::
      This function is here for legacy reasons, may be removed from nn.Functional in the future.

    .. note::
      The main trick for `hard` is to do  `y_hard - y_soft.detach() + y_soft`

      It achieves two things:
      - makes the output value exactly one-hot
      (since we add then subtract y_soft value)
      - makes the gradient equal to y_soft gradient
      (since we strip all other gradients)

    Examples::
        >>> logits = torch.randn(20, 32)
        >>> # Sample soft categorical using reparametrization trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=False)
        >>> # Sample hard categorical using "Straight-through" trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=True)

    .. _Link 1:
        https://arxiv.org/abs/1611.00712
    .. _Link 2:
        https://arxiv.org/abs/1611.01144
    """
    if eps != 1e-10:
        warnings.warn("`eps` parameter is deprecated and has no effect.")

    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    # y_soft = gumbels.softmax(dim)
    y_soft = gumbels

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

class SAMSGL(nn.Module, Seq2SeqAttrs):
    def __init__(self, loc_info, sparse_idx, geodesic, angle_ratio, logger=None, **model_kwargs):
        '''
        graph: physical graph, dynamic graph, static trainable graph

        '''
        super().__init__()
        Seq2SeqAttrs.__init__(self, sparse_idx, geodesic, angle_ratio, **model_kwargs)
        self.register_buffer('node_embeddings', nn.Parameter(torch.randn(self.node_num, self.embed_dim), requires_grad=True))
        self.feature_embedding = nn.Linear(self.input_dim, self.embed_dim)
        self._logger = logger
        # self.conv_ker = local_graph(
        #     self.location_dim,
        #     self.sparse_idx,
        #     self.node_num,
        #     self.lck_structure,
        #     loc_info,
        #     self.angle_ratio,
        #     self.geodesic,
        #     self.max_view,
        #     model_kwargs['knn']
        # )

        self.memory = self.construct_memory()
        # self.node_embedding3 = nn.Parameter(torch.randn(self.node_num, self.mem_num), requires_grad=True)

        self.network = SAMSGL_NET(
            nb_block=self.block_num,
            nb_chev_filter=self.hidden_units,
            nb_time_filter=self.hidden_units,
            time_strides=int(self.seq_len/2),
            conv_ker=self.conv_ker,
            **model_kwargs
        )

    def embedding(self, inputs):
        # inputs: b, l, n, f
        # outputs: b, l, n, e*2
        batch_size, seq_len, node_num, feature_size = inputs.shape
        feature_emb = self.feature_embedding(inputs)
        node_emb = self.node_embeddings[None, None, :, :].expand(batch_size, seq_len, node_num, self.embed_dim)

        return torch.cat([feature_emb, node_emb, inputs], dim=-1)

    def construct_memory(self):
        memory_dict = nn.ParameterDict()
        memory_dict['Memory'] = nn.Parameter(torch.randn(self.mem_num, self.mem_dim), requires_grad=True)     # (M, d)
        # memory_dict['Wq'] = nn.Parameter(torch.randn(self.rnn_units, self.mem_dim), requires_grad=True)    # project to query
        memory_dict['We1'] = nn.Parameter(torch.randn(self.node_num, self.mem_num), requires_grad=True) # project memory to embedding
        memory_dict['We2'] = nn.Parameter(torch.randn(self.node_num, self.mem_num), requires_grad=True) # project memory to embedding
        for param in memory_dict.values():
            nn.init.xavier_normal_(param)
        return memory_dict

    def query_memory(self, h_t:torch.Tensor):
        query = torch.matmul(h_t, self.memory['Wq'])     # (B, N, d)
        att_score = torch.softmax(torch.matmul(query, self.memory['Memory'].t()), dim=-1)         # alpha: (B, N, M)
        value = torch.matmul(att_score, self.memory['Memory'])     # (B, N, d)
        _, ind = torch.topk(att_score, k=2, dim=-1)
        pos = self.memory['Memory'][ind[:, :, 0]] # B, N, d
        neg = self.memory['Memory'][ind[:, :, 1]] # B, N, d
        return value, query, pos, neg

    def scaled_laplacian(self, node_embeddings, is_eval=False):
        # Normalized graph Laplacian function.
        # :param W: np.ndarray, [n_route, n_route], weighted adjacency matrix of G.
        # :return: np.matrix, [n_route, n_route].
        # learned graph
        learned_graph = torch.mm(node_embeddings, node_embeddings.T)
        node_num = self.node_num
        norm = torch.norm(node_embeddings, p=2, dim=1, keepdim=True)  # 求dim=1的2范数
        norm = torch.mm(norm, norm.transpose(0, 1))
        learned_graph = learned_graph / norm
        learned_graph = (learned_graph + 1) / 2.
        # learned_graph = F.sigmoid(learned_graph)
        learned_graph = torch.stack([learned_graph, 1 - learned_graph], dim=-1)

        # make the adj sparse
        if is_eval:
            adj = gumbel_softmax(learned_graph, tau=1, hard=True)
        else:
            adj = gumbel_softmax(learned_graph, tau=1, hard=True)
        adj = adj[:, :, 0].clone().reshape(node_num, -1)
        # mask = torch.eye(self.num_nodes, self.num_nodes).to(device).byte()
        mask = torch.eye(node_num, node_num).bool().cuda()
        adj.masked_fill_(mask, 0)

        # d ->  diagonal degree matrix
        W = adj
        n = W.shape[0]
        d = torch.sum(W, axis=1)
        ## L -> graph Laplacian
        L = -W
        L[range(len(L)), range(len(L))] = d  # 此索引操作用于访问矩阵 L 的对角线元素
        # for i in range(n):
        #    for j in range(n):
        #        if (d[i] > 0) and (d[j] > 0):
        #            L[i, j] = L[i, j] / torch.sqrt(d[i] * d[j])
        ## lambda_max \approx 2.0, the largest eigenvalues of L.
        try:
            # e, _ = torch.eig(L)
            # lambda_max = e[:, 0].max().detach()
            # import pdb; pdb.set_trace()
            # e = torch.linalg.eigvalsh(L)
            # lambda_max = e.max()
            lambda_max = (L.max() - L.min())
        except Exception as e:
            print("eig error!!: {}".format(e))
            lambda_max = 1.0

        # pesudo laplacian matrix, lambda_max = eigs(L.cpu().detach().numpy(), k=1, which='LR')[0][0].real
        tilde = (2 * L / lambda_max - torch.eye(n).cuda())
        # self.adj = adj
        # self.tilde = tilde
        return tilde

    # def scaled_laplacian(self, node_embeddings, is_eval=False):
    #     # Normalized graph Laplacian function.
    #     # :param W: np.ndarray, [n_route, n_route], weighted adjacency matrix of G.
    #     # :return: np.matrix, [n_route, n_route].
    #     # learned graph
    #     learned_graph = torch.mm(node_embeddings, node_embeddings.T)
    #     node_num = self.node_num
    #     norm = torch.norm(node_embeddings, p=2, dim=1, keepdim=True)  # 求dim=1的2范数
    #     norm = torch.mm(norm, norm.transpose(0, 1))
    #     learned_graph = learned_graph / norm
    #     learned_graph = (learned_graph + 1) / 2.
    #     # learned_graph = F.sigmoid(learned_graph)
    #     learned_graph = torch.stack([learned_graph, 1 - learned_graph], dim=-1)
    #
    #     # make the adj sparse
    #     if is_eval:
    #         adj = gumbel_softmax(learned_graph, tau=1, hard=True)
    #     else:
    #         adj = gumbel_softmax(learned_graph, tau=1, hard=True)
    #     adj = adj[:, :, 0].clone().reshape(node_num, -1)
    #     # mask = torch.eye(self.num_nodes, self.num_nodes).to(device).byte()
    #     mask = torch.eye(node_num, node_num).bool()
    #     adj.masked_fill_(mask, 0)
    #
    #     # d ->  diagonal degree matrix
    #     W = adj
    #     n = W.shape[0]
    #     d = torch.sum(W, axis=1)
    #     ## L -> graph Laplacian
    #     L = -W
    #     L[range(len(L)), range(len(L))] = d  # 此索引操作用于访问矩阵 L 的对角线元素
    #     # for i in range(n):
    #     #    for j in range(n):
    #     #        if (d[i] > 0) and (d[j] > 0):
    #     #            L[i, j] = L[i, j] / torch.sqrt(d[i] * d[j])
    #     ## lambda_max \approx 2.0, the largest eigenvalues of L.
    #     try:
    #         # e, _ = torch.eig(L)
    #         # lambda_max = e[:, 0].max().detach()
    #         # import pdb; pdb.set_trace()
    #         # e = torch.linalg.eigvalsh(L)
    #         # lambda_max = e.max()
    #         lambda_max = (L.max() - L.min())
    #     except Exception as e:
    #         print("eig error!!: {}".format(e))
    #         lambda_max = 1.0
    #
    #     # pesudo laplacian matrix, lambda_max = eigs(L.cpu().detach().numpy(), k=1, which='LR')[0][0].real
    #     tilde = (2 * L / lambda_max - torch.eye(n))
    #     self.adj = adj
    #     self.tilde = tilde
    #     return adj, tilde

    def forward(self, inputs, labels=None, batches_seen=None):
        """
        seq2seq forward pass
        :param inputs: shape (seq_len, batch_size, num_sensor, input_dim)
        :param labels: shape (horizon, batch_size, num_sensor, output_dim)
        :param batches_seen: batches seen till now
        :return: output: (self.horizon, batch_size, self.node_num * self.output_dim)
        """
        # kernel = self.conv_ker(0)  # node,node
        node_embeddings1 = torch.matmul(self.memory['We1'], self.memory['Memory'])  # node, ch
        node_embeddings2 = torch.matmul(self.memory['We2'], self.memory['Memory'])  # node, ch
        g1 = self.scaled_laplacian(node_embeddings1)
        g2 = self.scaled_laplacian(node_embeddings2)
        kernel = [g1, g2]
        embedding = self.embedding(inputs)#seq_len, batch_size, node, emb_dim

        outputs, delay, weight = self.network(embedding, kernel)
        if batches_seen == 0:
            self._logger.info(
                "Total trainable parameters {}".format(count_parameters(self))
            )
        outputs = outputs+inputs[-1]
        # print(delay[0])
        return outputs, delay[0], weight[0]
