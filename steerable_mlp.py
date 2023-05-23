import lightning.pytorch as lp
import torch
import torch.nn as nn
import torch_geometric as tg  # needed for radialnet class
import torch_geometric.data.batch as tg_batch
import torchmetrics

from e3nn import o3  # needed for convolution class
from e3nn.nn import Gate  # needed for ConvLayer SE3 class
from e3nn.o3 import Irreps  # needed for balanced irreps function

import atom3dutils


def balanced_irreps(hidden_features, lmax):
    """Divide subspaces equally over the feature budget"""
    N = int(hidden_features / (lmax + 1))

    irreps = []
    for l, irrep in enumerate(Irreps.spherical_harmonics(lmax)):
        n = int(N / (2 * l + 1))

        irreps.append(str(n) + "x" + str(irrep[1]))

    irreps = "+".join(irreps)

    irreps = Irreps(irreps)

    # Don't short sell yourself, add some more trivial irreps to fill the gap
    gap = hidden_features - irreps.dim
    if gap > 0:
        irreps = Irreps("{}x0e".format(gap)) + irreps
        irreps = irreps.simplify()

    return irreps

def compute_gate_irreps(irreps_out):
    """Compute irreps_scalars, irreps"""
    irreps_scalars = Irreps([(mul, ir) for mul, ir in irreps_out if ir.l == 0])
    irreps_gated = Irreps([(mul, ir) for mul, ir in irreps_out if ir.l > 0])
    irreps_gates = Irreps([(mul, "0e") for mul, _ in irreps_gated]).simplify()

    return irreps_scalars, irreps_gated, irreps_gates

class Convolution(nn.Module):
    """ SE(3) equivariant convolution, parameterised by a radial network """
    def __init__(self, irreps_in1, irreps_in2, irreps_out):
        super().__init__()
        self.irreps_in1 = irreps_in1
        self.irreps_in2 = irreps_in2
        self.irreps_out = irreps_out
        self.tp =  o3.FullyConnectedTensorProduct(
            irreps_in1,
            irreps_in2,
            irreps_out,
            irrep_normalization="component",
            path_normalization="element",
            internal_weights=False,
            shared_weights=False
        )

        self.radial_net = RadialNet(self.tp.weight_numel)

    def forward(self, x, rel_pos_sh, distance):
        """
        Features of shape [E, irreps_in1.dim]
        rel_pos_sh of shape [E, irreps_in2.dim]
        distance of shape [E, 1]
        """
        weights = self.radial_net(distance)
        return self.tp(x, rel_pos_sh, weights)

class RadialNet(nn.Module):
    def __init__(self, num_weights):
        super().__init__()

        num_basis = 10
        basis = tg.nn.models.dimenet.BesselBasisLayer(num_basis, cutoff=4)

        self.net = nn.Sequential(basis,
                                nn.Linear(num_basis, 16),
                                nn.SiLU(),
                                nn.Linear(16, num_weights))
    def forward(self, dist):
        return self.net(dist.squeeze(-1))

class ConvLayerSE3(tg.nn.MessagePassing):
    def __init__(self, irreps_in1, irreps_in2, irreps_out, activation=True):
        super().__init__(aggr="add")

        self.irreps_in1 = irreps_in1
        self.irreps_in2 = irreps_in2
        self.irreps_out = irreps_out

        irreps_scalars, irreps_gated, irreps_gates = compute_gate_irreps(irreps_out)
        self.conv = Convolution(irreps_in1, irreps_in2, irreps_gates + irreps_out)

        if activation:
            self.gate = Gate(irreps_scalars, [nn.SiLU()], irreps_gates, [nn.Sigmoid()], irreps_gated)
        else:
            self.gate = nn.Identity()

    def forward(self, edge_index, x, rel_pos_sh, dist):
        x = self.propagate(edge_index, x=x, rel_pos_sh=rel_pos_sh, dist=dist)
        x = self.gate(x)
        return x

    def message(self, x_i, x_j, rel_pos_sh, dist):
        return self.conv(x_j, rel_pos_sh, dist)

# Collect the above defined classes in the complete model

class ConvModel(nn.Module):
    def __init__(self, irreps_in, irreps_hidden, irreps_edge, irreps_out, depth, max_z:int=atom3dutils._NUM_ATOM_TYPES):
        super().__init__()

        self.irreps_in = irreps_in
        self.irreps_hidden = irreps_hidden
        self.irreps_edge = irreps_edge
        self.irreps_out = irreps_out

        self.embedder = nn.Embedding(max_z, irreps_in.dim)

        self.layers = nn.ModuleList()
        self.layers.append(ConvLayerSE3(irreps_in, irreps_edge, irreps_hidden))
        for i in range(depth-2):
            self.layers.append(ConvLayerSE3(irreps_hidden, irreps_edge, irreps_hidden))
        self.layers.append(ConvLayerSE3(irreps_hidden, irreps_edge, irreps_out, activation=False))

    def forward(self, graph):
        edge_index = graph.edge_index
        z = graph.z
        pos = graph.pos
        batch = graph.batch

        # Prepare quantities for convolutional layers
        # Index of source and target node
        src, tgt = edge_index[0], edge_index[1]
        # Vector pointing from the source node to the target node
        rel_pos = pos[tgt] - pos[src]
        # That vector in Spherical Harmonics
        rel_pos_sh = o3.spherical_harmonics(self.irreps_edge, rel_pos, normalize=True)
        # The norm of that vector
        dist = torch.linalg.vector_norm(rel_pos, dim=-1, keepdims=True)

        # Embed atom one-hot
        x = self.embedder(z)

        # Convolve nodes
        for layer in self.layers:
            x = layer(edge_index, x, rel_pos_sh, dist)

        # 1-dim output, squeeze it out
        x = x.squeeze(-1)

        # TODO: add dense layers

        # Global pooling of node features
        x = tg.nn.global_mean_pool(x, batch)
        return x


# Then use this model on the atom3d tasks

class Atom3D(lp.LightningModule):
    def __init__(
        self,
        model:nn.Module,
        metrics:dict[str,torchmetrics.Metric],
        lr:float=1e-4,
        *args,
        **kwargs
    ):

        super().__init__(*args, **kwargs)
        self.model = model
        self.lr = lr

        self.metrics = metrics
        self.loss_fn = nn.MSELoss()

    def forward(self, batch:tg_batch):
        return self.model(batch)

    def training_step(self, batch:tg_batch, batch_idx:int):
        out = self(batch)
        loss = self.loss_fn(out, batch.label)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch:tg_batch, batch_idx:int):
        out = self(batch)
        loss = self.loss_fn(out, batch.label)

        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch:tg_batch, batch_idx:int):
        out = self(batch)

        out = out.detach().cpu()
        label = batch.label.detach().cpu()

        results = dict()
        for key, func in self.metrics.items():
            results[f'test/{key}'] = func(out, label)
        self.log_dict(results, on_epoch=True, logger=True)

        return self.loss_fn(out, label)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        return {
            'optimizer': optimizer
        }
