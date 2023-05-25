# Steerable Graph Convolutions for Learning Protein Structures

by *Synthesized Solutions*

## 1. Introduction

<!-- An analysis of the paper and its key components. Think about it as nicely formatted review as you would see on OpenReview.net -->

Machine learning is increasingly being applied to the analysis of molecules for tasks such as protein design, model quality assessment, and ablation studies. These techniques can help us better understand the structure and function of proteins, which is useful for many medical application, such as drug discovery. Convolutional Neural Networks (CNNs) and Graph Neural Networks (GNNs) are two types of machine learning models that are  particularly well-suited for analyzing molecular data. CNNs can operate directly on the geometry of a structure and GNNs are expressive in terms of relational reasoning.

However, proteins are complex biomolecules with a unique three-dimensional structure that is critical to their function and modeling the interactions between non-adjacent amino acids can be challenging. Both CNNs and GNNs can be translation invariant and equivariant, but these properties cannot be guaranteed for rotations in typical implementations.  Formally, we can define invariance and equivariance as follows:

$$\begin{align}
\text{Invariance:}   && f(g\cdot x) &= f(x)         \\
\text{Equivariance:} && f(g\cdot x) &= g\cdot f(x).
\end{align}$$

In order to take more geometric information into account, Jing, Eismann, Suriana, Townshend, and Dror (2020) propose a method that combines the strengths of CNNs and GNNs to learn from biomolecular structures. Instead of encoding 3D geometry of proteins, i.e. vector features, in terms of rotation-invariant scalars, they propose that vector features be directly represented as geometric vectors in 3D space at all steps of graph propagation. They claim this approach improves the GNN's ability to reason geometrically and capture the spatial relationships between atoms and residues in a protein structure.

This modification to the standard GNN consists of changing the multilayer perceptrons (MLPs) with geometric vector perceptrons (GVPs), see also Figure 1 (top). The GVP approach described in the paper is used to learn the relationship between protein sequences and their structures. GVPs are a type of layer that operates on geometric objects, such as vectors and matrices, rather than scalar values like most neural networks. This makes GVPs well-suited to tasks that involve analyzing spatial relationships, which is highly important for protein structures. To show this improvement, the model is evaluated on various tasks from the Atom3D dataset described in [Section 3](#31-tasks).

In GVP-GNNs, node and edge embeddings are represented as tuples of scalar features and geometric vector features. The message and update functions are parameterized by geometric vector perceptrons, which are modules that map between the tuple representations while preserving rotational invariance. In a follow-up paper by Jing, Eismann, Soni, and Dror (2021) they extended the GVP-GNN architecture to handle atomic-level structure representations, which allows the architecture to be used for a wider range of tasks. <!-- why, idk rn -->

In the original GVP-GNN architecture, vector outputs are functions of vector inputs, but these output vectors do not depend on the scalar inputs. This can be an issue for atomic-level structure graphs where individual atoms may not necessarily have an orientation. <!-- also don't really understand why -->
To address this issue, Jing et al. (2021) propose vector gating as a way to propagate information from the scalar channels into the vector channels, see Figure 1 (bottom). This involves transforming the scalar features and passing them through a sigmoid activation function to “gate” the vector output, replacing the vector non-linearity. In their paper they note that the equivariance of the vector features is not affected, because the scalar features are invariant and the gating is row-wise. They conclude that vector gating can help improve the GVP-GNN's ability to handle atomic-level structure representations and therefore machine learning on molecules.

Lastly, as mentioned, the equivariance to rotation of the models is very important and in order to test this property, the original authors have tested the GVP to check if the models behave the same when the conditions are rotated randomly. We can summarize this behaviour into two points, namely:

1. Scalars are invariant to rotations, so the output scalar features should be be the same if the vector features of the input nodes are rotated;
2. The vector features are equivariant to rotations, so if the input is rotated, the output should be close to rotating the original output. 
<!-- Equivariant message-passing seeks to incorporate the equivariant representations of ENNs within the message-passing framework of GNNs instead of indirectly encoding the 3D geometry in terms of pairwise distances, angles, and other scalar features. <----- this is a sentence from the 2021 paper -->

<p align="center">
    <img src="gvp-pytorch/schematic.png" style="margin:0" alt>
</p>
<p align="center">
    <em>Figure 1.</em> The original geometric vector perceptron (GVP) reported by Jing et al. in 2020 is shown in the top diagram, while the updated GVP provided in Jing et al. in 2021 is shown in the bottom diagram. Information can now go from the scalar channels to the vector channels due to the replacement of the original vector non-linearity (in red) with vector gating (in blue). Row- or element-wise operations are indicated by circles. The main module of the equivariant GNN is the modified GVP.
</p>

## 2. Strengths and Points of Improvement
<!-- Exposition of its weaknesses/strengths/potential which triggered your group to come with a response. -->
<!-- #BEGIN NOTES#
- Current model is not very expressive, but quite efficient; it's not steerable (slow); can only handle type-1
  - GVPs would kind of be part of the Invariant Message Passing NNs
  - So I consider it as a “incomplete” steerable mlp
  - My point is that steerable MLP can enable the information exchange between all possible pairs of vectors (type 0, 1, …, n), but GVP can only exchange the information from scalar vector to type-1 vector by using gating and from type-1 vector to scalar using norm.
- only invariant to rotation, due to taking norm (scalar value) (i think) -> this is only the case in the 2020 paper, but not necessarily in the 2021 paper, so i think we really need to focus on the expressiveness and not necessarily the equivariance
#END NOTES# -->

The current model of the authors manages to combine the strengths of CNNs and GNNs while maintaining the rotation invariance whilst using a model is not very computationally demanding. The invariance for rotation is essential because the orientation of the molecule does not change the characteristics of the molecule. However, the combination of the molecules into a protein does depend on the orientation of (the linkage between) the molecules, e.g. the shape of the protein does affect the characteristics of the protein. This is a weakness in the otherwise strength of the model. In the follow-up paper, the authors introduced vector-gating to retain the rotational equivariance of vector features, but this version of the GVP can only exchange information between scalar and geometric features using scalar values (either the norm or using gating). A point of improvement we are aiming for is to increase the expressiveness of this model by improving this sharing between scalar and geometric features to incorporate orientation into the scalar features.

<!-- $\newcommand{\bs}[1]{\boldsymbol{#1}} \newcommand{\norm}[1]{\lVert#1\rVert_2}$ <!-- why does this have to be impossible on GitHub >:( -->

To formalize this, take the transformation of the scalar features $\boldsymbol{s} \in \mathbb{R}^{s_{in}} \mapsto \boldsymbol{s}' \in \mathbb{R}^{s_{out}}$ in the GVP module such that

<!-- Quadruple backslash intended, GitHub does not render it correctly otherwise -->

$$\begin{equation}
\boldsymbol{s}'=\sigma\left( \boldsymbol{W}_m \begin{bmatrix} \lVert{\boldsymbol{W}_h \boldsymbol{V}}\rVert_2 \\\\ \boldsymbol{s} \end{bmatrix} + \boldsymbol{b} \right)
\end{equation}$$

where $\boldsymbol{W}$ is the weight matrix of the linear layers, $\boldsymbol{b}$ is a bias vector, $\sigma$ is some element-wise non-linearity, $\boldsymbol{V} \in \mathbb{R}^{n \times 3}$ are the geometric features and their norm $\lVert{\cdot}\rVert_2$ is taken row-wise. $\boldsymbol{s}'$ is invariant under rotations if, for some unitary $3\times3$ rotation matrix $\boldsymbol{U}$, the rotated geometric features $\boldsymbol{V} \boldsymbol{U}$ give the same $\boldsymbol{s'}$ as defined above. This holds, since

$$\begin{equation} \lVert{\boldsymbol{W}_h \boldsymbol{V}}\rVert_2 = \lVert{\boldsymbol{W}_h \boldsymbol{V} \boldsymbol{U}}\rVert_2.\end{equation}$$

<!--
None of us knew how to maths.  You cannot split vertically, only horizontally (x,y,z separately), and that isn't useful.

Additionally, if the rows of $\boldsymbol{V}$ are rotated *individually* using matrices $\boldsymbol{U}_1, \ldots, \boldsymbol{U}_n$, resulting in some $\boldsymbol{V}'$, they will act on $\boldsymbol{s}$ identically, since their norms are taken row-wise

$$\begin{equation}
\lVert{\boldsymbol{W}_h \boldsymbol{V}'}\rVert_2 = \begin{bmatrix} \lVert{\boldsymbol{W}_h \boldsymbol{v}_1^T \boldsymbol{U}_1}\rVert_2 \\\\ \vdots \\\\ \lVert{\boldsymbol{W}_h \boldsymbol{v}_n^T \boldsymbol{U}_n}\rVert_2 \end{bmatrix} = \begin{bmatrix}\lVert{\boldsymbol{W}_h \boldsymbol{v}_1^T}\rVert_2 \\\\ \vdots \\\\ \lVert{\boldsymbol{W}_h \boldsymbol{v}_n^T}\rVert_2 \end{bmatrix} = \lVert{\boldsymbol{W}_h \boldsymbol{V}}\rVert_2.
\end{equation}$$ 
-->


## 3. Our Contribution  <!-- uitbreiden met uitleg over steerable convolution of graph -->

<!-- Describe your novel contribution. -->
We aim to improve performance of the GVP layers by no longer requiring the scalar features to be independent of the orientation of the geometric features. In the GVP layers, by taking the norm of these features, important information of the orientation is lost. For the Atom3D tasks used for evaluation, a model is used that only takes the output scalar features of all nodes, and therefore does not take orientation into account explicitely. This limits the expressiveness of the model.

Our hypothesis is that the scalar features used as output of the model will be more expressive of the geometry of the data when using steerable basis. These steerable basis will allow for better communication between the scalar and geometric features which includes orientation, rather than just the norm.


### 3.1. Tasks

The authors of the original paper use eight tasks to test the quality of their model. These tasks are [**LBA**](#lba), [**SMP**](#smp), [**MSP**](#msp), [**PSR**](#psr) and [**RSR**](#rsr). These tasks have the following definitions:

#### LBA

“*Ligand Binding Affinity*” is a regression task with the goal to predict the binding affinity of a protein-ligand complex. This complex describes the binding interactions between a small molecule *ligand* and its target *protein*. These interactions can result in gain or loss of functional effects that can be utilized for a beneficial medicinal effect. Predicting and determining the intermolecular forces that affect binding between a protein and potential ligand therefore plays an important role in optimizing the drug discovery process. In this task, the goal of the model is to predict the binding affinity between the biomolecule and its ligand. Here, the affinity is described as the negative logarithm of the equilibrium dissociation constant, $K_D$, which serves as a robust measure for indicating the presence of strong binding interactions. This task was provided with different training/validation splits. ([Wang, R. et al., 2004](https://doi.org/10.1021/jm030580l); [Liu, Z. et al., 2015](https://doi.org/10.1093/bioinformatics/btu626))

#### SMP

“*Small Molecule Properties*” is a regression task that predicts the physiochemical properties of a small molecule structure. This is a common practice in drug discovery where these experiments are often done with quantum chemical calculations and can cover a wide variety of candidate molecules. Predicted properties include melting point, boiling point, density, viscosity, hydrophobicity and hydrophilicity. ([Ramakrishnan, et al., 2014](https://doi.org/10.1038/sdata.2014.22); [Ruddigkeit, L. et al., 2012](https://doi.org/10.1021/ci300415d))

#### MSP

“*Mutation Stability Prediction*” is a classification task with the goal to predict whether a point mutation is stable. <!-- The input mutation gets modeled into the mutated structure with the aim to predict whether or not the mutation is stable. --> The modelled mutated structure is stable if **TODO: Explain shortly what stable means**. Identifying mutations that stabilize a protein’s interactions is a key task in designing new proteins. Experimental methods for investigating point mutations are labor-intensive, which motivates the development of effective computing tools. ([Jankauskaitė, J.et al., 2018](https://doi.org/10.1093/bioinformatics/bty635))

#### PSR

“*Protein Structure Ranking*” is a regression task that predicts the global distance test (GDT_TS) of the true structure and each of the predicted structures submitted in the previous 18 years of the Critical Assessment of protein Structure Prediction (CASP). One of the main workhorses of the cell are proteins; understanding, and designing for, their function <!-- frequently --> depends on understanding their structure. ([Kryshtafovych, A., et al., 2019](https://doi.org/10.1002/prot.25823))


#### RSR
“*RNA Structure Ranking*” is a regression task that aims to predict the root mean squared deviation (RMSD) from the ground truth structure to candidate RNA models. **More explanation?** ([Cruz, J. A., et al., 2012](https://doi.org/10.1261/rna.031054.111); [Watkins, A. M., et al., 2020](https://doi.org/10.1016/j.str.2020.05.011))

### 3.2. Reproduction

Since the code of the paper was given to us, it was relatively easy to reproduce the results of the original paper. We reproduced all the tasks that our cluster could handle. Once we had all the results, we could build upon a task that was correctly reproduced. We only use tasks that are close to the original papers because only these task can give a clear and correct indication if our contribution improves the model.

Our results were as follows:
| Task                    | Metric        | Jing et al. | Ours    |
|-------------------------|---------------|-----------|-----------|
| LBA (Split 30)          | RMSE &#8595;         | **1.594** | **1.598** |
| LBA (Split 60)          | RMSE &#8595;         | **1.594** | **1.641** |
| SMP $\mu[D]$            | MAE   &#8595;        | 0.049     | 0.144     |
| SMP $\sigma_{gap} [eV]$ | MAE   &#8595;        | 0.065     | 0.0058    |
| SMP $U^{at}_0 [eV]$     | MAE   &#8595;        | 0.143     | 0.0259    |
| MSP                     | AUROC       &#8593;  | **0.680** | **0.672** |
| PSR                     | global $R_s$ &#8593; | **0.845** | **0.854** |  <!--Reproduced, right, but @Jip didn't comment on it-->
| PSR                     | mean $R_s$  &#8593;  | 0.511     | 0.602     |
| RSR                     | global $R_s$ &#8593; | **0.330** | **0.331** |
| RSR                     | mean $R_s$  &#8593;  | 0.221     | 0.018     |

<!-- down &#8595; ->
<!-- up &#8593; -->

From these results we conclude that the LBA task is close enough to the original paper that our reproduction is succesfull. Although MSP is close enough to use this task for the adaption, training the model on this task took nearly 10 hours, which is reason to prefer LBA. The reproduction RSR task is globally also very close, however the mean $R_s$ has a big deviation, thus we do not use this task for further research. All other tasks did not have the results reproduced good enough for us to use futher.

### 3.3 Steerable MLP
<!-- Jip schrijft hier nog wat meer met steerable -->

<!--
ChatGPT stuff on the explanation of steerable graph convolutions
In a steerable graph convolution, the filters are defined in a way that they can be rotated to any direction in the graph, by using a steering matrix. The steering matrix is a set of complex-valued coefficients that are learned during training, and it encodes the rotation of the filters in the spectral domain of the graph Laplacian matrix.

The spectral domain of the graph Laplacian matrix consists of its eigenvalues and eigenvectors. The eigenvectors represent the basis functions of the graph, while the eigenvalues correspond to the frequencies of the functions. By multiplying the filter with the steering matrix in the spectral domain, the filter is rotated to the desired direction in the graph.

The steerable graph convolutional operation can be represented as:

Overall, steerable graph convolutions offer a flexible and efficient way to perform graph convolutional operations in any direction, making them suitable for a wide range of graph-based machine learning tasks.
 -->
### 3.4 Testing Equivariance
In order to verify if our implementation of the steerable MLP is equivariant to rotation, we need to perform the same method, used by the original authors, as mentioned before. However, since we work with irreducible representations, the method needs some extra intermediate steps. Since the input of this model is represented using irreducible representations, each individual part needs to be rotated accordingly. So, after sampling a random 3D rotation matrix, it is transformed to do so. The remaining steps of testing equivariance is the same as described in [Section 1](#1-introduction).


## 4. Conclusion
<!-- Conclude -->




## 5. Contributions
<!-- Close the notebook with a description of each student's contribution. -->
Simon: provided taartjes

Noa: cried

