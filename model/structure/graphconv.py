import torch.nn as nn
import torch

class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, max_view, align):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # self.conv = conv
        self.max_view = max_view
        self.order = 1
        self.align = align
        self.linear = nn.Linear(input_dim * max_view, output_dim)

    def forward(self, x, x_t, delay, over_x, support):
        x_g = []
        for i,kernel in enumerate(support):
            x_g.append(torch.cat([torch.matmul(
                torch.pow(kernel, i + 1), x) for i in range(self.order)], dim=-1))
        # x_g.append(over_x)
        x_g = torch.cat(x_g, dim=-1)
        x_g = self.linear(x_g)
        # x = self.norm(x)
        return x_g
