# Steerable Graph Convolutions for Learning Protein Structures
by *Synthesized Solututions*

## Introduction
<!-- An analysis of the paper and its key components. Think about it as nicely formatted review as you would see on OpenReview.net -->
Machine learning is increasingly being applied to the analysis of molecules for tasks such as protein design, model quality assessment, and ablation studies. These techniques can help us better understand the structure and function of proteins, which is useful for many medical application, such as drug discovery. Convolutional Neural Networks (CNNs) and Graph Neural Networks (GNNs) are two types of machine learning models that are  particularly well-suited for analyzing molecular data. CNNs can operate directly on the geometry of a structure and GNNs are expressive in terms of relational reasoning.

However, proteins are complex biomolecules with a unique three-dimensional structure that is critical to their function and modeling the interactions between non-adjacent amino acids can be challenging. Both CNNs and GNNs might be translation invariant and equivariant (right?), but this is not the case for rotations.  Formally we can define equivariance as follows:

$$f(g\cdot x) = g\cdot f(x)$$

In order to keep more geometric information, Jing, Eismann, Suriana, Townshend, and Dror (2020) propose a method that combines the strengths of CNNs and GNNs to learn from biomolecular structures. Instead of encoding 3D geometry of proteins, i.e. vector features, in terms of rotation-invariant scalars, they propose that vector features be directly represented as geometric vectors in 3D space at all steps of graph propagation. They claim that this approach would improve the GNN's ability to reason geometrically and capture the spatial relationships between atoms and residues in a protein structure.

This modification to the standard GNN consists of changing the multilayer perceptrons (MLPs) with geometric vector perceptrons (GVPs). The GVP approach described in the paper is used to learn the relationship between protein sequences and their structures. GVPs are a type of layer that operates on geometric objects, such as vectors and matrices, rather than on scalar values like most neural networks. This makes GVPs well-suited to tasks that involve analyzing spatial relationships, which is highly important for protein structures.

In GVP-GNNs, node and edge embeddings are represented as tuples of scalar features and geometric vector features. The message and update functions are parameterized by geometric vector perceptrons, which are modules that map between the tuple representations while preserving rotational invariance. In a paper by Jing, Eismann, Soni, and Dror (2021) they extended the GVP-GNN architecture to handle atomic-level structure representations, which allows the architecture to be used for a wider range of tasks. <!-- why, idk rn -->

In the original GVP-GNN architecture, the vector outputs are functions of the vector inputs, but not the scalar inputs, which can be an issue for atomic-level structure graphs where individual atoms may not necessarily have an orientation. <!-- also don't really understand why -->
To address this issue, they propose vector gating as a way to propagate information from the scalar channels into the vector channels. This involves transforming the scalar features and passing them through a sigmoid activation function to "gate" the vector output, replacing the vector nonlinearity. In the paper they note that because the scalar features are invariant and the gating is row-wise, the equivariance of the vector features is not affected. They conclude that vector gating can help improve the GVP-GNN's ability to handle atomic-level structure representations and therefore machine learning on molecules.

<!-- add a better conclusion of this paragraph here -->

<!-- Equivariant message-passing seeks to incorporate the equivariant representations of ENNs within the message-passing framework of GNNs instead of indirectly encoding the 3D geometry in terms of pairwise distances, angles, and other scalar features. <----- this is a sentence from the 2021 paper -->

<p align="center">
    <img src="gvp-pytorch/schematic.png" style="margin:0" alt>
</p>
<p align="center">
    <em>Figure 1.</em> Schematic of the original geometric vector perceptron (GVP) as described in Jing et al. (2020) (top) and the modified GVP presented in Jing et al. 2021 (bottom). The original vector nonlinearity (in red) has been replaced with vector gating (in blue), allowing information to propagate from the scalar channels to the vector channels. Circles denote row- or element-wise operations. The modified GVP is the core module in the equivariant GNN.
</p>


## Strengths and Points of Improvement
<!-- Exposition of its weaknesses/strengths/potential which triggered your group to come with a response. -->

<!-- #BEGIN NOTES#

- Current model is not very expressive, but quite efficient; it's not steerable (slow); can only handle type-1
  - GVPs would kind of be part of the Invariant Message Passing NNs
  - So I consider it as a “incomplete” steerable mlp
  - My point is that steerable MLP can enable the information exchange between all possible pairs of vectors (type 0, 1, …, n), but GVP can only exchange the information from scalar vector to type-1 vector by using gating and from type-1 vector to scalar using norm.
- only invariant to rotation, due to taking norm (scalar value) (i think) -> this is only the case in the 2020 paper, but not necessarily in the 2021 paper, so i think we really need to focus on the expressiveness and not necessarily the equivariance

#END NOTES# -->

The current model of the authors manages to combine the strengths of CNNs and GNNs while maintaining the rotation invariance whilst using a model is not very computationally demanding. The invariance for rotation is essential because the orientation of the molecule does not change the characteristics of the molecule. However, the combination of the molecules into a protein does depend on the orientation of (the linkage between) the molecules, e.g. the shape of the protein does affect the characteristics of the protein. This is a weakness in the otherwise strength of the model. In the follow-up paper, they introduced the vector-gating to retain the rotational equivariance of the vector features, but this version of the GVP can only exchange information between scalar and geometric features using scalar values (either the norm or using gating). A point of improvement we are aiming for is to increase the expresiveness of this model by improving this sharing between scalar and geometric features to incorperate orientation into the scalar features.


### Testing Equivariance
In order to test if the model is really equivariant to rotations, they check if the models behaves the same when the conditions are rotated. We can summarize this behaviour in into two points, namely:

<!-- - n_v = nodes[1] -> vector features of the nodes
- e_v = edges[1] -> vector features of the edges -->

1. We want the output scalar features to be the same if the vector features of the input nodes and edges are rotated;
2. If the output vector features of the original node and edges input are rotated, they need to be close to the output vector features from the rotated input vector features.

This is done with the following function:

```py
def test_equivariance(model, nodes, edges):

    random = torch.as_tensor(Rotation.random().as_matrix(),
                             dtype=torch.float32, device=device)

    with torch.no_grad():

        out_s, out_v = model(nodes, edges)
        n_v_rot, e_v_rot = nodes[1] @ random, edges[1] @ random
        out_v_rot = out_v @ random
        out_s_prime, out_v_prime = model((nodes[0], n_v_rot), (edges[0], e_v_rot))

        assert torch.allclose(out_s, out_s_prime, atol=1e-5, rtol=1e-4)
        assert torch.allclose(out_v_rot, out_v_prime, atol=1e-5, rtol=1e-4)
```

## Our Contribution
<!-- Describe your novel contribution. -->
We aim to improve performance of the GVP layers by no longer requiring the scalar features to be independent of the orientation of the geometric features. In the GVP layers, by taking the norm of these features, important information of the orientation is lost. For the Atom3D tasks, a model is used that only takes the output scalar features of all nodes, and therefore does not take orientation into account explicitely. This limits the expressiveness of the model.

Our hypothesis is that by using a steerable basis, the scalar features used as output of the model will be more expressive of the geometry of the data. These steerable basis will allow for better communication between the scalar and geometric features which includes orientation, rather than just the norm.

### Tasks
The authors of the original paper use eight tasks to test the quality of their model. These tasks are **LBA, SMP, MSP, PSR & RSR**. These tasks mean the following:
#### LBA
*Ligand Binding Affinity* task is a regression task that gets as input structure a
Protein-ligand complex, this is a binding between a protein bound and a ligand (A ligand is a molecule or ion that binds to a central metal atom or ion in a complex, or a small molecule that binds to a larger biomolecule to modify its activity, stability or localization.). From this input structure the goal is to predict the negative log affinity. The negative log is for optimisation and the affinity is the rate at which the molecules bind with each other. We did this with different training/validation splits

#### SMP
*Small Molecule Properties* is a regression task to predict the physiochemical property of a small molecule that are given as input structure. Examples of these properties are melting point, boiling point, density, viscosity, Hydrophobicity/Hydrophilicity.

#### MSP
*Mutation Stability Prediction* is a classification task that takes as input structure a
protein complex and the same protein with a mutation. The aim of the task is to predict whether or not the mutation is stable or not.
####  PSR
*Protein Structure Ranking* is a regression task that gets a protein as input structure and aims to predict the Global distance test total score (GST_TS). This score is a mesure that measures the two protein structures.

#### RSR
*RNA Structure Ranking* is a regression task that gets RNA as input structure and aims to predict the root mean squared deviation (RMSD).

### Reproduction
Since the code of the paper was given to us, it was relatively easy to reproduce the results of the original paper. We reproduced all the tasks that our cluster could handle. Once we had all the results, we could build upon a task that was correctly reproduced. We only want to use tasks that are very close to the original papers since then the task can give a clear and correct indication if our contribution improves the model.

Our resulst were as follows:
| Task                    | Our    | Their |
|-------------------------|--------|-------|
| LBA (Split 30)          | **1.577**  | **1.594** |
| LBA (Split 60)          | **1.596**  | **1.594** |
| SMP $\mu[D]$            | 0.144  | 0.049 |
| SMP $\sigma_{gap} [eV]$ | 0.0058 | 0.065 |
| SMP $U^{at}_0 [eV]$     | 0.0259 | 0.143 |
| MSP                     | **0.672**  | **0.680** |
| PSR (global)            | 0.854  | 0.845 |
| PSR (mean)              | 0.602  | 0.511 |
| RSR (global)            | **0.331**  | **0.330** |
| RSR (mean)              | 0.018  | 0.221 |

From these results we conclude that the LBA task is close enough to the original paper that our reproduction is succesfull. Also MSP is close enough to use this task for the adaption. The reproduction RSR task is globally also very close, however the mean has a big deviation, that we do not aim to use this task for further research. The other tasks were not good enough for us to use futher.

<!-- - changing perhaps change the k in knn for these graph convolution (message passing layers) -->

<!--
ChatGPT stuff on the explanation of steerable graph convolutions
In a steerable graph convolution, the filters are defined in a way that they can be rotated to any direction in the graph, by using a steering matrix. The steering matrix is a set of complex-valued coefficients that are learned during training, and it encodes the rotation of the filters in the spectral domain of the graph Laplacian matrix.

The spectral domain of the graph Laplacian matrix consists of its eigenvalues and eigenvectors. The eigenvectors represent the basis functions of the graph, while the eigenvalues correspond to the frequencies of the functions. By multiplying the filter with the steering matrix in the spectral domain, the filter is rotated to the desired direction in the graph.

The steerable graph convolutional operation can be represented as:

$Y = U g(\Lambda) U^T X$,

where $X$ is the input feature matrix, $Y$ is the output feature matrix, $U$ is the matrix of eigenvectors of the graph Laplacian, $\Lambda$ is the diagonal matrix of eigenvalues, $g$ is a diagonal matrix of learnable filter coefficients, and $T$ denotes matrix transpose.

The steerable graph convolution can be efficiently implemented using the Chebyshev polynomial approximation, which allows for a low-order polynomial approximation of the filter function in the spectral domain. This reduces the computational complexity of the operation and makes it practical for large-scale graphs.

Overall, steerable graph convolutions offer a flexible and efficient way to perform graph convolutional operations in any direction, making them suitable for a wide range of graph-based machine learning tasks.
 -->

## Conclusion
<!-- Conclude -->




## Contributions
<!-- Close the notebook with a description of each student's contribution. -->
