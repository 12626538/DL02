import math
import torch
from e3nn import o3
from e3nn.o3 import Irreps
import plotly.graph_objects as go
from atom3dutils import get_datasets

#@title (This cell contains some code for creating visualizations, feel free to skip over it)
axis = dict(
    showbackground=False,
    showticklabels=False,
    showgrid=False,
    zeroline=False,
    title='',
)

layout = dict(
    showlegend=False,
    scene=dict(
        aspectmode="data",
        xaxis=dict(
            **axis,
        ),
        yaxis=dict(
            **axis,
        ),
        zaxis=dict(
            **axis,
        ),
    ),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=0, r=0, t=0, b=0)
)

def s2_grid(N=100):
    """ Create grid on which we can sample spherical signals """
    betas = torch.linspace(0, math.pi, int(N/2))
    alphas = torch.linspace(0, 2 * math.pi, N)
    beta, alpha = torch.meshgrid(betas, alphas)
    return o3.angles_to_xyz(alpha, beta)

def nodes_to_shs(f, irreps, grid):
    """ Converts node features to irreps. """
    lmax = irreps.lmax
    N = f.size(0)

    assert len(f.shape) == 2, "We expect [N, coefficients]"
    assert irreps == Irreps.spherical_harmonics(lmax), "Irreps aren't SHs"

    shs = o3.spherical_harmonics(irreps.ls, grid, normalize=True).unsqueeze(0).repeat(N, 1, 1, 1)
    shs *= f.view(N, 1, 1, -1)
    shs = shs.sum(-1)
    return shs

def plot_spherical_harmonics(f, positions, irreps, pos_factor=1, offset=0):
    """ Plots feature vectors on molecule as spherical harmonics."""
    # Leave no trace
    f, positions = f.clone(), positions.clone()

    # Convert features to Spherical Harmonics
    grid = s2_grid()
    shs = nodes_to_shs(f, irreps, grid)

    # Let's plot!
    fig = go.Figure(layout=layout)

    # Normalise
    positions -= positions.min()
    positions /= positions.max()
    positions *= pos_factor

    shs -= shs.min()
    shs /= shs.max()
    shs = shs*(1 - offset) + offset

    cmin = offset
    cmax = 1

    # Plot nodes
    for sh, pos in zip(shs, positions):
        x = sh.abs() * grid[..., 0] + pos[0]
        y = sh.abs() * grid[..., 1] + pos[1]
        z = sh.abs() * grid[..., 2] + pos[2]

        fig.add_trace(go.Surface(x=x, y=y, z=z, surfacecolor=sh, colorscale='RdBu_r', cmin=cmin, cmax=cmax))

    # Add edges
    for (pos_i, pos_j) in positions[graph.edge_index].transpose(1, 0):
        d = dict(x=[pos_i[0], pos_j[0]],
                 y=[pos_i[1], pos_j[1]],
                 z=[pos_i[2], pos_j[2]])

        fig.add_trace(go.Scatter3d(**d,
                                   marker=dict(
                                       size=0,
                                   ),
                                   line=dict(
                                       color='black',
                                       width=6,
                                   ), opacity=0.1)
                      )

    fig.show()




if __name__ == '__main__':
    # takes the first protein of lba train dataset
    graph = get_datasets('LBA', lba_split=60)['train'][0] #  <- change this
    graph # Print keys stored in first structure
    if len(graph.z.size) == 1:
        graph.z = graph.z.unsqueeze(1)
    irreps = Irreps("1x0e")
    # print(graph.z)
    plot_spherical_harmonics(graph.z, graph.pos, irreps, pos_factor=15, offset=0.5)
