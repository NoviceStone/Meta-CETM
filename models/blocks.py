import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class FCResBlock(nn.Module):
    """Residual block with fully-connected layers.

    Attributes:
        dim: number of neurons in each layer.
        num_layers: number of layers in residual block.
        activation: specify activation function.
        batch_norm: use or not use batch norm.
        dropout: use or not use dropout.
    """

    def __init__(
            self,
            dim: int,
            num_layers: int,
            activation: nn,
            batch_norm: bool = False,
            dropout: bool = False,
    ):
        super(FCResBlock, self).__init__()
        self.num_layers = num_layers
        self.activation = activation
        self.batch_norm = batch_norm
        self.dropout = dropout

        self.block = []
        for _ in range(self.num_layers):
            layer = [nn.Linear(dim, dim)]
            if self.batch_norm:
                layer.append(nn.BatchNorm1d(num_features=dim))
            if self.dropout:
                layer.append(nn.Dropout(p=0.1))
            self.block.append(nn.ModuleList(layer))
        self.block = nn.ModuleList(self.block)

    def forward(self, x):
        r = x + 0
        for n, layers in enumerate(self.block):
            for i in range(len(layers)):
                r = layers[i](r)
            if n < (len(self.block) - 1):
                r = self.activation(r)
        # residual
        return self.activation(r + x)


class PostPool(nn.Module):
    """Post process module, map task representation to moments of latent summary.

    Attributes:
        hid_dim: number of neurons in hidden layers.
        c_dim: dimensions of the latent summary variable c.
        activation: specify activation function.
    """

    def __init__(
            self,
            hid_dim: int,
            c_dim: int,
            activation: nn
    ):
        super(PostPool, self).__init__()
        self.c_dim = c_dim
        self.hid_dim = hid_dim
        self.activation = activation
        self.linear_params = nn.Linear(self.hid_dim, 2 * self.c_dim)

    def forward(self, x):
        # affine transformation to parameters
        x = self.linear_params(x)
        mean, logvar = torch.chunk(x, 2, dim=-1)
        return mean, logvar


class TBlock(nn.Module):
    """Transformer block for keys and values over c.

    Attributes:
        hid_dim: number of neurons in hidden layers.
    """

    def __init__(
            self,
            hid_dim: int,
    ):
        super(TBlock, self).__init__()
        self.hid_dim = hid_dim
        self.key = nn.Linear(hid_dim, hid_dim)
        self.query = nn.Linear(hid_dim, hid_dim)
        self.value = nn.Linear(hid_dim, hid_dim)

    def forward(self, qq, kv):
        """
        Compute attention weights alpha(k, q) and update representations.
        We compute a similarity measure between context samples and query sample.

        Args:
            qq: query (1, hid_dim)
            kv: keys (sample_size, hid_dim)
        Returns:
            Updated representations for the context memory.
        """
        N, D = kv.size()

        q = self.query(qq)  # shape: (1, D)
        k = self.key(kv)  # shape: (N, D)
        v = self.value(kv)  # shape: (N, D)

        # sim (1, D) x (D, N) -> (1, N)
        # out (1, N) x (N, D) -> (1, D)
        sim = (q @ k.transpose(0, 1)) * (1.0 / np.sqrt(D))
        sim = F.softmax(sim, dim=-1)
        out = sim @ v
        return out, sim


class StatistiC(nn.Module):
    """Compute the statistic q(c | X).

    Encode X, preprocess, aggregate, postprocess.
    Representation for context.

    Attributes:
        hid_dim: number of neurons in hidden layers.
        c_dim: dimensions of the latent summary variable c.
        activation: specify activation function.
    """

    def __init__(
            self,
            hid_dim: int,
            c_dim: int,
            activation: nn,
    ):
        super(StatistiC, self).__init__()
        self.c_dim = c_dim
        self.hid_dim = hid_dim
        self.activation = activation

        self.postpool = PostPool(self.hid_dim, self.c_dim, self.activation)

    def forward(self, h):
        # aggregate samples: mean
        a = torch.mean(h, dim=0, keepdim=True)
        # map to moments
        mean, logvar = self.postpool(a)
        return mean, logvar, a, None


class AttentiveStatistiC(nn.Module):
    """Compute the statistic q(c | X).

    Encode X, preprocess, aggregate, postprocess.
    Representation for context.

    Attributes:
        hid_dim: number of neurons in hidden layers.
        c_dim: dimensions of the latent summary variable c.
        activation: specify activation function.
    """

    def __init__(
            self,
            hid_dim: int,
            c_dim: int,
            activation: nn,
    ):
        super(AttentiveStatistiC, self).__init__()
        self.c_dim = c_dim
        self.hid_dim = hid_dim
        self.activation = activation

        self.aggregation_module = TBlock(self.hid_dim)
        self.postpool = PostPool(self.hid_dim, self.c_dim, self.activation)

    def forward(self, h):
        # aggregate samples: attention
        r = torch.mean(h, dim=0, keepdim=True)
        a, att = self.aggregation_module(r, h)
        # map to moments
        mean, logvar = self.postpool(a)
        return mean, logvar, a, att


class ThetaPosteriorNet(nn.Module):
    """Inference network q(θ|x, c) gives approximate posterior over latent variables θ.

    Attributes:
        num_layers: number of layers in residual block.
        hid_dim: number of neurons in hidden layers.
        c_dim: number of features for latent variable c and θ.
        activation: specify activation function.
    """

    def __init__(
            self,
            num_layers: int,
            hid_dim: int,
            c_dim: int,
            activation: nn,
    ):
        super(ThetaPosteriorNet, self).__init__()
        self.activation = activation
        self.num_layers = num_layers
        self.hid_dim = hid_dim
        self.c_dim = c_dim

        # modules
        self.linear_h = nn.Linear(self.hid_dim, self.hid_dim)
        self.linear_c = nn.Linear(self.c_dim, self.hid_dim)
        self.linear_params = nn.Linear(self.hid_dim, 2 * self.c_dim)

    def forward(self, h, c):
        """
        Args:
            h: conditional features for hidden layer.
            c: context latent representation.
        Returns:
            moments for the approximate posterior over theta.
        """
        # transform h
        th = self.linear_h(h)

        # transform c and expand for broadcast addition
        if c is not None:
            tc = self.linear_c(c)
            tc = tc.expand_as(th)
        else:
            tc = th.new_zeros(th.size())

        # combine h and c
        t = th + tc
        t = self.activation(t)

        # affine transformation to parameters
        out = self.linear_params(t)
        mean, logvar = torch.chunk(out, 2, dim=-1)
        return mean, logvar


class GraphEncoder(nn.Module):
    """Inference network q(Z|E, A) gives approximate posterior over latent variables Z.

    Attributes:
        in_channels: dimension of initial node features.
        hidden_channels: number of hidden units in graph layers.
        out_channels: dimensions for the latent variable z.
        activation: specify activation function.
    """

    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            out_channels: int,
            activation: nn,
    ):
        super(GraphEncoder, self).__init__()
        self.base_conv = GCNConv(in_channels, hidden_channels, add_self_loops=False)
        self.mean_conv = GCNConv(hidden_channels, out_channels, add_self_loops=False)
        self.logvar_conv = GCNConv(hidden_channels, out_channels, add_self_loops=False)
        self.activation = activation

    def forward(self, x, edge_index, edge_weight=None):
        # x: node feature matrix with shape (num_nodes, in_channels)
        # edge_index: graph connectivity matrix with shape (2, num_edges)
        h = self.activation(self.base_conv(x, edge_index, edge_weight=edge_weight))
        mean = self.mean_conv(h, edge_index, edge_weight=edge_weight)
        logvar = self.logvar_conv(h, edge_index, edge_weight=edge_weight)
        # ToDo:
        #  add relu activation to ensure positive embedding
        #  normalize embedding so that inner-product distance is equivalent to Euclidean distance
        return mean, logvar


class InnerProductDecoder(nn.Module):
    """Symmetric inner product decoder layer"""

    def __init__(
            self,
            droprate: float,
            activation: nn,
    ):
        super(InnerProductDecoder, self).__init__()
        self.activation = activation
        self.dropout = nn.Dropout(droprate)

    def forward(self, inputs):
        inputs = self.dropout(inputs)
        outputs = torch.mm(inputs, inputs.transpose(0, 1))
        # outputs = outputs.view(-1)
        return self.activation(outputs)
