
import torch
import torch.nn as nn
import torch.nn.functional as F
class local_Attrs:
    def __init__(self, 
        location_dim, 
        sparse_idx,
        node_num,
        lck_structure, 
        local_graph_coor, 
        angle_ratio, 
        geodesic,
        max_view,
        knn):

        self.location_dim = location_dim
        self.node_num = node_num
        self.sparse_idx = sparse_idx
        self.local_graph_coor = local_graph_coor
        self.angle_ratio = angle_ratio
        self.geodesic = geodesic
        self.max_view = max_view
        self.lck_structure = lck_structure
        self.knn = knn

class Attention(nn.Module):
    def __init__(self, location_dim, embed_dim=8):
        super(Attention, self).__init__()
        self.query_linear = nn.Linear(location_dim, embed_dim)
        self.key_linear = nn.Linear(location_dim, embed_dim)
    
    def forward(self, q, k):
        '''
        q: b, l, n, f
        k: b, l, n, f
        '''
        q = self.query_linear(q)
        k = self.key_linear(k)
        att = torch.matmul(q, k.transpose(-2, -1))
        return att.squeeze()


class local_graph(nn.Module, local_Attrs):
    def __init__(
            self,
            location_dim,
            sparse_idx,
            node_num,
            lck_structure,
            local_graph_coor,
            angle_ratio,
            geodesic,
            max_view,
            knn,
    ):
        super(local_graph, self).__init__()
        local_Attrs.__init__(self, location_dim, sparse_idx, node_num, lck_structure,
                             local_graph_coor, angle_ratio, geodesic, max_view, knn=knn)
        if (knn==False):
            location_dim = 1
            self.lcker = LocalConditionalKer(location_dim, lck_structure)
            self.weight_att = Attention(location_dim)
        else:
            # location_dim =location_dim
            self.lcker = LocalConditionalKer(location_dim * 2, lck_structure)
            self.weight_att = Attention(location_dim)


    def kernel(self, coor):
        if len(coor.shape)==1:
            coor =coor.unsqueeze(-1)
        lcker = self.lcker(coor)
        lcker = torch.sparse.FloatTensor(self.sparse_idx, lcker.flatten(), (self.node_num, self.node_num)).to_dense()
        if self.knn:
            sphere_coor = coor[:, :self.location_dim] + coor[:, self.location_dim:]
            sphere_coor = sphere_coor.reshape(self.node_num, -1, self.location_dim)
            center_points = sphere_coor[:, [0], :]
            neighbor_points = sphere_coor
            alpha = self.weight_att(center_points, neighbor_points).abs()
            alpha = torch.sparse.FloatTensor(self.sparse_idx, alpha.flatten(), (self.node_num, self.node_num)).to_dense()
            distance_decay = (- alpha * self.geodesic).exp()
            angle_ratio = self.angle_ratio
            return lcker * distance_decay * angle_ratio
        else:
            distance_decay = (- self.geodesic).exp()
            return lcker * distance_decay

    def forward(self, x):
        kernel = self.kernel(self.local_graph_coor)
        return kernel

    def kernel_prattern(self, centers, vs, angle_ratio):
        '''
        centers: (N, 2)
        vs: (N, M, 2)
        '''
        M = vs.shape[1]
        centers = centers[:, None, :].repeat(1, M, 1)
        coor = torch.cat([centers, vs], dim=-1)
        ker_patterns = []
        for i in range(coor.shape[0]):
            lcker = self.lcker(coor[i]).flatten()
            geodesics = torch.square(vs[i]).sum(dim=-1).sqrt()
            alpha = self.weight_att(centers[i], vs[i]).abs()
            distance_decay = (- alpha[0] * geodesics).exp()
            ker_patterns.append(lcker * distance_decay * angle_ratio)
        return torch.stack(ker_patterns, dim=0)

class LocalConditionalKer(nn.Module):
    def __init__(self, location_dim, structure, activation='tanh'):
        super(LocalConditionalKer, self).__init__()
        '''Initialize the generator.
        '''
        self.structure = structure
        self.network = nn.ModuleList()
        self.network.append(nn.Linear(location_dim, self.structure[0]))

        for i in range(len(self.structure)-1):
            self.network.append(
                nn.Linear(self.structure[i], self.structure[i+1])     
                )  
            self.network.append(
                nn.BatchNorm1d(self.structure[i+1])    
                )  
        self.network.append(nn.Linear(self.structure[-1], 1))

        if activation == 'tanh':
            self.activation = torch.tanh
    
    def forward(self, x):
        for j, layer in enumerate(self.network):
            if j != 1:
                x = self.activation(x)
            x = layer(x)
        return torch.relu(x)
