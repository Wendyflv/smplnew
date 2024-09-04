from __future__ import division
import torch
import torch.nn.functional as F
import numpy as np
import scipy.sparse
import math
import scipy.sparse as sp

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def adj_mx_from_edges(num_pts, edges, sparse=True):
    edges = np.array(edges, dtype=np.int32)
    data, i, j = np.ones(edges.shape[0]), edges[:, 0], edges[:, 1]
    adj_mx = sp.coo_matrix((data, (i, j)), shape=(num_pts, num_pts), dtype=np.float32)

    # build symmetric adjacency matrix
    adj_mx = adj_mx + adj_mx.T.multiply(adj_mx.T > adj_mx) - adj_mx.multiply(adj_mx.T > adj_mx)
    adj_mx = normalize(adj_mx + sp.eye(adj_mx.shape[0]))
    if sparse:
        adj_mx = sparse_mx_to_torch_sparse_tensor(adj_mx)
    else:
        adj_mx = torch.tensor(adj_mx.todense(), dtype=torch.float)
    return adj_mx


class SparseMM(torch.autograd.Function):
    """Redefine sparse @ dense matrix multiplication to enable backpropagation.
    The builtin matrix multiplication operation does not support backpropagation in some cases.
    """
    @staticmethod
    def forward(ctx, sparse, dense):
        ctx.req_grad = dense.requires_grad
        ctx.save_for_backward(sparse)
        return torch.matmul(sparse, dense)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        sparse, = ctx.saved_tensors
        if ctx.req_grad:
            grad_input = torch.matmul(sparse.t(), grad_output)
        return None, grad_input

def spmm(sparse, dense):
    return SparseMM.apply(sparse, dense)


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class BertLayerNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class GraphResBlock(torch.nn.Module):
    """
    Graph Residual Block similar to the Bottleneck Residual Block in ResNet
    """
    def __init__(self, in_channels, out_channels, mesh_type='body'):
        super(GraphResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin1 = GraphLinear(in_channels, out_channels // 2)
        self.conv = GraphConvolution(out_channels // 2, out_channels // 2, mesh_type)
        self.lin2 = GraphLinear(out_channels // 2, out_channels)
        self.skip_conv = GraphLinear(in_channels, out_channels)
        # print('Use BertLayerNorm in GraphResBlock')
        self.pre_norm = BertLayerNorm(in_channels)
        self.norm1 = BertLayerNorm(out_channels // 2)
        self.norm2 = BertLayerNorm(out_channels // 2)

    def forward(self, x):
        trans_y = F.relu(self.pre_norm(x)).transpose(1,2)
        y = self.lin1(trans_y).transpose(1,2)

        y = F.relu(self.norm1(y))
        y = self.conv(y)

        trans_y = F.relu(self.norm2(y)).transpose(1,2)
        y = self.lin2(trans_y).transpose(1,2)

        z = x+y

        return z

# class GraphResBlock(torch.nn.Module):
#     """
#     Graph Residual Block similar to the Bottleneck Residual Block in ResNet
#     """
#     def __init__(self, in_channels, out_channels, mesh_type='body'):
#         super(GraphResBlock, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.conv = GraphConvolution(self.in_channels, self.out_channels, mesh_type)
#         print('Use BertLayerNorm and GeLU in GraphResBlock')
#         self.norm = BertLayerNorm(self.out_channels)
#     def forward(self, x):
#         y = self.conv(x)
#         y = self.norm(y)
#         y = gelu(y)
#         z = x+y
#         return z

class GraphLinear(torch.nn.Module):
    """
    Generalization of 1x1 convolutions on Graphs
    """
    def __init__(self, in_channels, out_channels):
        super(GraphLinear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = torch.nn.Parameter(torch.FloatTensor(out_channels, in_channels))
        self.b = torch.nn.Parameter(torch.FloatTensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        w_stdv = 1 / (self.in_channels * self.out_channels)
        self.W.data.uniform_(-w_stdv, w_stdv)
        self.b.data.uniform_(-w_stdv, w_stdv)

    def forward(self, x):
        return torch.matmul(self.W[None, :], x) + self.b[None, :, None]

class GraphConvolution(torch.nn.Module):
    """Simple GCN layer, similar to https://arxiv.org/abs/1609.02907."""
    def __init__(self, in_features, out_features, mesh='body', bias=True):
        super(GraphConvolution, self).__init__()
        device=torch.device('cpu')
        self.in_features = in_features
        self.out_features = out_features


        
        num_joints = 24
        parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
        edges = list(filter(lambda x: x[1] >= 0, zip(list(range(0, num_joints)), parents)))
        
        self.adjmat = adj_mx_from_edges(num_joints, edges, sparse=True).to(device)

        self.weight = torch.nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # stdv = 1. / math.sqrt(self.weight.size(1))
        stdv = 6. / math.sqrt(self.weight.size(0) + self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        if x.ndimension() == 2:
            support = torch.matmul(x, self.weight)
            output = torch.matmul(self.adjmat, support)
            if self.bias is not None:
                output = output + self.bias
            return output
        else:
            output = []
            for i in range(x.shape[0]):
                support = torch.matmul(x[i], self.weight)
                # output.append(torch.matmul(self.adjmat, support))
                output.append(spmm(self.adjmat, support))
            output = torch.stack(output, dim=0)
            if self.bias is not None:
                output = output + self.bias
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'