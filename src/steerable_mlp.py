import lightning.pytorch as lp
import torch
import torch.nn as nn
import torch_geometric as tg  # needed for radialnet class
import torch_geometric.data.batch as tg_batch
import torchmetrics

from e3nn import o3  # needed for convolution class
from e3nn.nn import Gate  # needed for ConvLayer SE3 class
from e3nn.o3 import Irreps  # needed for balanced irreps function

# import atom3dutils

def compute_gate_irreps(irreps_out):
    """Compute irreps_scalars, irreps"""
    irreps_scalars = Irreps([(mul, ir) for mul, ir in irreps_out if ir.l == 0])
    irreps_gated = Irreps([(mul, ir) for mul, ir in irreps_out if ir.l > 0])
    irreps_gates = Irreps([(mul, "0e") for mul, _ in irreps_gated]).simplify()

    return irreps_scalars, irreps_gated, irreps_gates


class RadialNet(nn.Module):
    """
    RadialNet module,
    Convert distance to a feature vector based on the Radial basis
    This gives a more expressive representation of the distance between nodes
    """
    def __init__(self, num_weights:int, num_basis:int=10, cutoff:float=4):
        """
        num_weights: int - Number of features to express distance
        num_basis: int - Number of basis to use,
                         higher gives more expressive features
        cutoff: float - Max distance to express
        """
        super().__init__()

        # Convert distance to `num_basis`-shaped feature vector
        basis = tg.nn.models.dimenet.BesselBasisLayer(num_basis, cutoff=cutoff)

        # Now use regular linear layers
        # to express `num_basis` as `num_weights` feature vector
        self.net = nn.Sequential(basis,
                                nn.Linear(num_basis, 16),
                                nn.SiLU(),
                                nn.Linear(16, num_weights))

    def forward(self, dist:torch.Tensor) -> torch.Tensor:
        """
        Input:
        dist: torch.Tensor - Tensor of shape `[E, 1]`,
                             where E is number of edges
        Out:
        torch.Tensor - Tensor of shape `[num_weights]`
        """
        return self.net(dist.squeeze(-1))


class Convolution(nn.Module):
    """
    SE(3) equivariant convolution
    parameterised by a radial network

    Convolves node irrep with edge irrep using
    `e3nn.o3.FullyConnectedTensorProduct` but uses a trainable `RadialNet` to
    give weights to this tensorproduct
    """
    def __init__(self, irreps_node, irreps_edge, irreps_out):
        super().__init__()
        # Save irrep shapes
        self.irreps_node = irreps_node
        self.irreps_edge = irreps_edge
        self.irreps_out = irreps_out

        # Convolve node irrep with edge irrep to give out irrep
        self.tp =  o3.FullyConnectedTensorProduct(
            irreps_node,
            irreps_edge,
            irreps_out,
            irrep_normalization="component",
            path_normalization="element",
            internal_weights=False,
            shared_weights=False
        )

        # Use a trainable RadialNet to give weights to this TensorProduct
        # Is more expressive, since RadialNet is based on Radial Basis, which
        # already encode information for edge/node pair.
        self.radial_net = RadialNet(self.tp.weight_numel)

    def forward(self, x, rel_pos_sh, distance):
        """
        Features of shape [E, irreps_node.dim]
        rel_pos_sh of shape [E, irreps_edge.dim]
        distance of shape [E, 1]

        output of shape [E, irreps.out.dim]
        """
        # Get weights based on RadialNet instance
        weights:torch.Tensor = self.radial_net(distance)

        # Parameterize TensorProduct by those weights
        return self.tp(x, rel_pos_sh, weights)


class ConvLayerSE3(tg.nn.MessagePassing):
    """
    SE(3) equivariant message passing layer

    Messages are composed of convolving a target node with the corresponding edge
    using the above defined Convolution (which uses a FullyConnectTensorProduct
    parameterized by a trainable RadialNet)
    """
    def __init__(self, irreps_node, irreps_edge, irreps_out, activation=True, aggr='add'):
        # Pass up how to aggregate
        super().__init__(aggr=aggr)

        # Each node is of this type
        self.irreps_node = irreps_node
        # Every edge is of this type
        self.irreps_edge = irreps_edge
        # After passing, each node has this type
        self.irreps_out = irreps_out

        # Compute gate effect
        irreps_scalars, irreps_gated, irreps_gates = compute_gate_irreps(irreps_out)

        # Actual convolution, used to define a message
        self.conv = Convolution(irreps_node, irreps_edge, irreps_gates + irreps_out)

        # Add activation method
        if activation:
            # Make sure there are l>0 type vectors to gate
            if irreps_gated:
                self.gate = Gate(irreps_scalars, [nn.SiLU(),], irreps_gates, [nn.Sigmoid(),], irreps_gated)
            # Just gate scalar features
            # else:
            #     self.gate = nn.SiLU()
        else:
            self.gate = nn.Identity()

    def forward(self, edge_index, x, rel_pos_sh, dist):
        """
        edge_index: torch.Tensor of shape [2, E]
        x: torch.Tensor of shape [V, irreps_node.dim]
        rel_pos_sh: torch.Tensor of shape [E, irreps_edge.dim]
        dist: torch.Tensor of shape [E, 1]
        where E is the number of edges and V the number of nodes
        """

        # Kick off message passing
        x = self.propagate(edge_index, x=x, rel_pos_sh=rel_pos_sh, dist=dist)
        # x: torch.Tensor of shape [N, irreps_out.dim]

        # Gate node features
        x = self.gate(x)

        return x

    def message(self, x_j, rel_pos_sh, dist):
        """
        x_j: torch.Tensor of shape [E, irreps_node.dim]
        rel_pos_sh: torch.Tensor of shape [E, irreps_edge.dim]
        dist: torch.Tensor of shape [E, 1]
        where E is the number of edges of the current graph
        """
        return self.conv(x_j, rel_pos_sh, dist)


class ConvModel(nn.Module):
    """
    A Convolutional Model based on SE(3) equivariant ConvLayerSE3 layers

    Converts a graph with node labels to a single label, either per node or
    over the entire graph

    Takes as input a `torch_gemetric.data.batch` instance with
        pos:torch.Tensor        - A [N,3] shaped tensor with locations of the nodes
        z:torch.Tensor          - A [N] shaped tensor with the node label/class to embed
        edge_index:torch.Tensor - [2,E] shaped tensor as edge index
        batch:torch.Tensor      - A [N] shaped tensor with labels which node belongs to which batch
    """
    def __init__(
            self,
            irreps_in:Irreps,
            irreps_hidden:Irreps,
            irreps_edge:Irreps,
            irreps_out:Irreps,
            depth:int,
            max_z:int=9,
            dense:bool=True
        ):
        """
        irreps_in:Irreps - Embed each node using this irrep
        irreps_hidden:Irreps - Intermediate layers use this irrep for nodes
        irreps_edge:Irreps - Edge features
        irreps_out:Irreps - Output irreps of the last ConvLayerSE3 layer
        depth:int - Number of ConvLayerSE3 layers
        max_z:int - Number of labels for the input nodes
        """
        super().__init__()

        # Save irrep shapes
        self.irreps_in = irreps_in
        self.irreps_hidden = irreps_hidden
        self.irreps_edge = irreps_edge
        self.irreps_out = irreps_out

        # Atom type embedder, convert `graph.z` to a vector of shape `irreps_in.dim`
        self.embedder = nn.Embedding(max_z, irreps_in.dim)

        # Convolutional Layers
        self.layers = nn.ModuleList()

        # Take embedded atoms, set them to hidden irrep
        self.layers.append(ConvLayerSE3(irreps_in, irreps_edge, irreps_hidden))

        # Convovle for depth-2 layers
        for i in range(depth-2):
            self.layers.append(ConvLayerSE3(irreps_hidden, irreps_edge, irreps_hidden))

        # Convert hidden irrep to output irrep
        self.layers.append(ConvLayerSE3(irreps_hidden, irreps_edge, irreps_out, activation=False))

        # Finally, add some dense layers, if specified
        # !note! this assumes `irreps_out` only consists of type-0 (scalar) irreps
        # types >0 cannot be put through linear layers
        self.dense=None
        if dense:
            self.dense = nn.Sequential(
                nn.Linear(irreps_out.dim, 2*irreps_out.dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.1),
                nn.Linear(2*irreps_out.dim, 1),
            )

    def forward(self, graph):
        # Edge index, shape `[2, E]`
        edge_index = graph.edge_index

        # Atom type (int), shape `[N]`
        z = graph.z

        # Atom location (XYZ), shape `[N, 3]`
        pos = graph.pos

        # Indicating which atom belongs to which batch, shape `[N,]`
        batch = graph.batch

        # Convert graph structure to features
        # Index of source and target node
        src, tgt = edge_index[0], edge_index[1]
        # Vector pointing from the source node to the target node
        rel_pos = pos[tgt] - pos[src]
        # That vector in Spherical Harmonics (Irrep feature)
        rel_pos_sh = o3.spherical_harmonics(self.irreps_edge, rel_pos, normalize=True)
        # The norm of that vector (Scalar feature)
        dist = torch.linalg.vector_norm(rel_pos, dim=-1, keepdims=True)

        # Embed atom label
        x = self.embedder(z)

        # Convolve nodes
        for layer in self.layers:
            x = layer(edge_index, x, rel_pos_sh, dist)

        if self.dense:
            x = self.dense(x)

        if x.shape[-1] == 1:
            # 1-dim output, squeeze it out
            x = x.squeeze(-1)

            # Global pooling of node features to get single output for the graph
            x = tg.nn.global_mean_pool(x, batch)
        return x


class Atom3D(lp.LightningModule):
    def __init__(
        self,
        model:nn.Module,
        metrics:dict[str,torchmetrics.Metric],
        lr:float=1e-4,
        # Remaining args is passed up to `lp.LightningModule`
        *args,
        **kwargs
    ):

        super().__init__(*args, **kwargs)
        self.model = model
        self.lr = lr

        self.metrics = metrics
        self.loss_fn = nn.MSELoss()

        self.test_step_outputs = []
        self.test_step_labels = []

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

        # Save for later test metric eval
        self.test_step_outputs.append(out)
        self.test_step_labels.append(label)

        return self.loss_fn(out, label)

    def on_test_epoch_end(self):
        # concatenate all batch outputs
        out = torch.cat(self.test_step_outputs, dim=0)
        label = torch.cat(self.test_step_labels, dim=0)

        # Run on metrics
        results = dict()
        for key, func in self.metrics.items():
            results[f'test/{key}'] = func(out, label)

        # Log results
        self.log_dict(results)

        self.test_step_outputs.clear()
        self.test_step_labels.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        return {
            'optimizer': optimizer
        }
