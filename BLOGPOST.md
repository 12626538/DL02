# Steerable Graph Convolutions for Learning Protein Structures

by *Synthesized Solutions*

## 1. Introduction

Machine learning is increasingly applied to molecular analysis for tasks such as protein design, model quality assessment, and ablation studies. These techniques can help us better understand the structure and function of proteins, which is useful for many medical applications, such as drug discovery. Convolutional Neural Networks (CNNs) and Graph Neural Networks (GNNs) are two types of machine learning models that are  particularly well-suited for analyzing molecular data. CNNs can operate directly on the geometry of a structure and GNNs are expressive in terms of relational reasoning.

Proteins are complex biomolecules with a unique three-dimensional structure that is critical to their function and modeling the interactions between non-adjacent amino acids can be challenging. Both CNNs and GNNs are typically translation invariant or equivariant, but these properties cannot be guaranteed for rotations in typical implementations.  Formally, we can define invariance and equivariance as follows:

$$\begin{align}
\text{Invariance:}   && f(g\cdot x) &= f(x)         \\
\text{Equivariance:} && f(g\cdot x) &= g\cdot f(x).
\end{align}$$

In order to take more geometric information into account, [Jing et al. (2020)](https://doi.org/10.48550/arXiv.2009.01411) propose a method that combines the strengths of CNNs and GNNs to learn from biomolecular structures. Instead of encoding 3D geometry of proteins, i.e. vector features, in terms of rotation-invariant scalars, they propose that vector features be directly represented as geometric vectors in 3D space at all steps of graph propagation. They claim this approach improves the GNN's ability to reason geometrically and capture the spatial relationships between atoms and residues in a protein structure.

This modification to the standard GNN consists of changing the multilayer perceptrons (MLPs) with geometric vector perceptrons (GVPs), see also Figure 1 (top). The GVP approach described in the paper is used to learn the relationship between protein sequences and their structures. GVPs are a type of layer that operates on geometric objects, such as vectors and matrices, rather than scalar values like most neural networks. This makes GVPs well-suited to tasks that involve analyzing spatial relationships, which is highly important for protein structures. To show this improvement, the model is evaluated on various tasks from the Atom3D dataset ([Townshend et al. 2020](https://doi.org/10.48550/arXiv.2012.04035)) described in [Section 3](#31-tasks).

In GVP-GNNs, node and edge embeddings are represented as tuples of scalar features and geometric vector features. The message and update functions are parameterized by geometric vector perceptrons, which are modules that map between the tuple representations while preserving rotational invariance. In a follow-up paper [Jing et al. (2021)](https://doi.org/10.48550/arXiv.2106.03843) extend the GVP-GNN architecture to handle atomic-level structure representations, which allows the architecture to be used for a wider range of tasks.

In the original GVP-GNN architecture, vector outputs are functions of vector inputs, but these output vectors do not depend on the scalar inputs. This can be an issue for atomic-level structure graphs where individual atoms may not necessarily have an orientation. To address this issue, [Jing et al. (2021)](https://doi.org/10.48550/arXiv.2106.03843) propose vector gating as a way to propagate information from the scalar channels into the vector channels, see Figure 1 (bottom). This involves transforming the scalar features and passing them through a sigmoid activation function to “gate” the vector output, replacing the vector non-linearity. In their paper they note that the equivariance of the vector features is not affected, because the scalar features are invariant and the gating occurs row-wise. They conclude that vector gating can help improve the GVP-GNN's ability to handle atomic-level structure representations and therefore machine learning on molecules.

Lastly, as mentioned, the equivariance to rotation of the models is very important and in order to test this property, the original authors have tested the GVP to check if the models behave the same when the conditions are rotated randomly. We can summarize this behaviour into two points, namely:

1. Scalars are invariant to rotations, so the output scalar features should be be the same if the vector features of the input nodes are rotated;
2. The vector features are equivariant to rotations, so if the input is rotated, the output should be close to rotating the original output.

<p align="center">
    <img src="src/gvp_schematic.png" style="margin:0" alt>
</p>
<p align="center">
    <em>Figure 1.</em> The original geometric vector perceptron (GVP) reported by Jing et al. fr0m 2020 is shown in the top diagram, while the updated GVP provided in Jing et al. from 2021 is shown in the bottom diagram. Information can now go from the scalar channels to the vector channels due to the replacement of the original vector non-linearity (in red) with vector gating (in blue). Row- or element-wise operations are indicated by circles. The main module of the equivariant GNN is the modified GVP.
</p>

## 2. Strengths and Points of Improvement

The current model of the authors manages to combine the strengths of CNNs and GNNs while maintaining the rotation invariance, which it achieves using a model of low computational burden. The invariance for rotation is essential because the orientation of the molecule does not change the characteristics of the molecule. However, the combination of the molecules into a protein does depend on the orientation of (the linkage between) the molecules, e.g. the shape of the protein does affect the characteristics of the protein. This is a weakness in the otherwise strength of the model. In the follow-up paper, the authors introduced vector-gating to retain the rotational equivariance of vector features, but this version of the GVP can only exchange information between scalar and geometric features using scalar values (either the norm or using gating). We aim to improve the expressiveness of this model by improving the sharing between scalar and geometric features to incorporate orientation into the scalar features.

To formalize the rotation invariance, take the transformation of the scalar features $\boldsymbol{s} \in \mathbb{R}^{s_{in}} \mapsto \boldsymbol{s}' \in \mathbb{R}^{s_{out}}$ in the GVP module such that

<!-- Quadruple backslash intended, GitHub does not render it correctly otherwise -->

$$\begin{equation}
\boldsymbol{s}'=\sigma\left( \boldsymbol{W}_m \begin{bmatrix} \lVert{\boldsymbol{W}_h \boldsymbol{V}}\rVert_2 \\\\ \boldsymbol{s} \end{bmatrix} + \boldsymbol{b} \right)
\end{equation}$$

where $\boldsymbol{W}$ is the weight matrix of the linear layers, $\boldsymbol{b}$ is a bias vector, $\sigma$ is some element-wise non-linearity, $\boldsymbol{V} \in \mathbb{R}^{n \times 3}$ are the geometric features and their norm $\lVert{\cdot}\rVert_2$ is taken row-wise. $\boldsymbol{s}'$ is invariant under rotations if, for some unitary $3\times3$ rotation matrix $\boldsymbol{U}$, the rotated geometric features $\boldsymbol{V} \boldsymbol{U}$ give the same $\boldsymbol{s'}$ as defined above. This trivially holds, since

$$\begin{equation} \lVert{\boldsymbol{W}_h \boldsymbol{V}}\rVert_2 = \lVert{\boldsymbol{W}_h \boldsymbol{V} \boldsymbol{U}}\rVert_2.\end{equation}$$

## 3. Our Contribution

Our objective is to enhance the performance of the GVP layers by removing the dependency of scalar features on the orientation of geometric features. Currently, in the GVP layers, the orientation-related information is lost when the norm of the geometric features is taken. To evaluate the Atom3D tasks, a model is employed that solely considers the scalar features of all nodes as output, neglecting the explicit consideration of orientation. This limits the expressiveness of the model.

We hypothesize that utilizing steerable basis for the scalar features as output in the model will provide a more expressive representation of the data's geometry. By incorporating steerable basis, effective communication between scalar and geometric features, including orientation, can be achieved, going beyond just considering the norm.

### 3.1. Tasks

The authors of the original paper use eight tasks from Atom3D ([Townshend et al. 2020](https://doi.org/10.48550/arXiv.2012.04035)) to test the quality of their model. These tasks are [**LBA**](#lba), [**SMP**](#smp), [**MSP**](#msp), [**PSR**](#psr) and [**RSR**](#rsr). Five of these tasks have the following definitions:

#### LBA

“*Ligand Binding Affinity*” is a regression task with the goal to predict the binding affinity of a protein-ligand complex. This complex describes the binding interactions between a small molecule *ligand* and its target *protein*. These interactions can result in gain or loss of functional effects that can be utilized for a beneficial medicinal effect. Predicting and determining the intermolecular forces that affect binding between a protein and potential ligand therefore plays an important role in optimizing the drug discovery process. In this task, the goal of the model is to predict the binding affinity between the biomolecule and its ligand. Here, the affinity is described as the negative logarithm of the equilibrium dissociation constant $K_D$, which serves as a robust measure for indicating the presence of strong binding interactions. <!-- This task was provided with different training/validation splits. --> ([Liu et al., 2015](https://doi.org/10.1093/bioinformatics/btu626); [Wang et al., 2004](https://doi.org/10.1021/jm030580l))

#### SMP

“*Small Molecule Properties*” is a regression task that predicts the physiochemical properties of a small molecule structure, which can be used to evaluate the suitability of a molecule for drug discovery applications. Predicted properties include electronic properties (orbital energies), thermodynamic properties (enthalpy), and energetic properties (internal energy). ([Ramakrishnan et al., 2014](https://doi.org/10.1038/sdata.2014.22); [Ruddigkeit et al., 2012](https://doi.org/10.1021/ci300415d))

#### MSP

“*Mutation Stability Prediction*” is a classification task with the goal to predict whether a point mutation is stablizing. A mutated protein structure is generated by introducing a single point mutation, meaning that one amino acid residue of the original protein is altered. By evaluating the post-mutation stability of the protein, information can be gathered on whether the replaced residue plays a role in stabilizing the native conformation of the protein or not. Identifying mutations that stabilize a protein’s interactions is a key task in designing new proteins. Experimental methods for investigating point mutations are labor-intensive, which motivates the development of effective computing tools. ([Jankauskaitė et al., 2018](https://doi.org/10.1093/bioinformatics/bty635))

#### PSR

“*Protein Structure Ranking*” is a regression task that predicts the global distance test (GDT_TS) of the true structure and each of the predicted structures submitted in the previous 18 years of the Critical Assessment of protein Structure Prediction (CASP). One of the main workhorses of the cell are proteins; understanding, and designing for, their function <!-- frequently --> depends on understanding their structure. ([Kryshtafovych et al., 2019](https://doi.org/10.1002/prot.25823))

#### RSR

“*RNA Structure Ranking*” is a regression task that aims to predict the root mean squared deviation (RMSD) from the ground truth structure to candidate RNA models. Structural information on the three-dimensional orientation of RNA is relatively low compared to proteins. Developing tools that can help predict their conformation can therefore prove to be essential in future research. The candidate models are obtained through the FARFAR2 and the RNA Puzzle Challenge ([Cruz et al., 2012](https://doi.org/10.1261/rna.031054.111); [Watkins et al., 2020](https://doi.org/10.1016/j.str.2020.05.011))

### 3.2. Reproduction

Since the code of the paper was given to us, it was relatively easy to reproduce the results of the original paper. We reproduced all the tasks that our cluster could handle. Once we had all the results, we could build upon a task that was correctly reproduced. We only use tasks that are close to the original papers because only these task can give a clear and correct indication if our contribution improves the model.

Our results were as follows:
| Task                    | Metric               | Jing et al. | Ours      |
|-------------------------|----------------------|-------------|-----------|
| LBA (Split 30)          | RMSE                 | **1.594**   | **1.598** |
| LBA (Split 60)          | RMSE                 |      -      |   1.641   |
| SMP $\mu[D]$            | MAE                  |   0.049     |   0.144   |
| SMP $\sigma_{gap} [eV]$ | MAE                  |   0.065     |   0.0058  |
| SMP $U^{at}_0 [eV]$     | MAE                  |   0.143     |   0.0259  |
| MSP                     | AUROC                | **0.680**   | **0.672** |
| PSR                     | global $R_s$         | **0.845**   | **0.854** |
| PSR                     | mean $R_s$           |   0.511     |   0.602   |
| RSR                     | global $R_s$         | **0.330**   | **0.331** |
| RSR                     | mean $R_s$           |   0.221     |   0.018   |

From these results we conclude that the LBA task is close enough to the original paper that our reproduction is successful. Although MSP is close enough to use for the adaption as well, training the model for this task took nearly 10 hours, for this reason we decided to focus only on LBA. The reproductions for the PSR and RSR tasks have a similar global $R_s$, however, the mean $R_s$ deviates significantly and we do not use this task for further research. The SMP task does not show reproduced results.

### 3.3. Steerable MLP

### 3.3 Steerable MLP

This section is mostly based on the paper “*Geometric and Physical Quantities improve* $E(3)$ *Equivariant Message Passing*” ([Brandstetter et al., 2022](https://doi.org/10.48550/arXiv.2110.02905)).

Steerable features are vectors that behave equivariant under transformations parameterized by $g$. This work uses $SO(3)$ steerable features, denoted with a tilde ($\boldsymbol{\tilde h}$). The type of this vector indicates the type of information it holds, where the relevant features for this work are type- $0$, scalar features, and type- $1$, 3D euclidean vectors (with $x$, $y$ and $z$ components). More general, a type- $l$ steerable feature is a $2l+1$-dimensional vector.

All type-$l$ vectors form some space denoted by $V_l$. The *direct sum* of independent spaces $V_{l_1}$ and $V_{l_2}$ gives the space $V = V_{l_1} \otimes V_{l_2}$, elements of which are steerable vectors of type $l_1$ and $l_2$. The direct sum of $n$ copies of a type-$l$ vector belongs to $nV_l = \otimes_{i=1}^n V_l$. For example, a $d$-dimensional scalar feature vector is an element of $d V_0$, i.e. $d$ instances of type-$0$ vectors.

Steerable MLPs are a type of Multi-Layer Perceptrons that, just like regular MLPs and GVPs, interleave linear mappings with non-linearities. Unlike traditional MLPs, steerable MLPs make use of conditional weights, parameterized by a steerable vector $\boldsymbol{\tilde a}$. Given a steerable feature vector ${\boldsymbol{\tilde h}}^{(i)}$ at layer $i$, the updated feature vector at layer $i+1$ can be formalized as

$$\begin{equation}
\boldsymbol{\tilde h}^{(i+1)} = \boldsymbol{W}^{(i)}_{\boldsymbol{\tilde a}}\ \boldsymbol{\tilde h}^{(i)}
\end{equation}$$

In geometric graph neural networks, geometric information can be encoded in the edge features between two nodes. Let $\boldsymbol{x}_i,\boldsymbol{x}_j$ be the euclidean coordinates of two nodes in $\mathbb{R}^3$, then a translation invariant edge feature can be defined as $\boldsymbol{e}_{ij} = \boldsymbol{x}_j  - \boldsymbol{x}_i$. The corresponding type- $l$ steerable edge feature $\boldsymbol{\tilde a}$ can now be defined using the *spherical harmonics* $Y^{(l)}_m: S^2 \rightarrow \mathbb{R}$ at $\frac{ \boldsymbol{e}_{ij} }{\lVert \boldsymbol{e}_{ij} \rVert}$

$$\begin{equation}
\boldsymbol{\tilde a}^{(l)} = \left( Y^{(l)}_m\left( \frac{ \boldsymbol{e}_{ij} }{\lVert \boldsymbol{e}_{ij} \rVert} \right) \right)^T_{m=-l, \dots, l}
\end{equation}$$

Using two steerable features ${\boldsymbol{\tilde h}}^{(l_1)}$, ${\boldsymbol{\tilde h}}^{(l_2)}$ of type- $l_1$ and - $l_2$, the Clebsch-Gordan (CG) tensor product $\otimes_{cg}$ can be used to obtain a new type- $l$ steerable vector $\boldsymbol{\tilde h}^{(l)}$ and can furthermore be parameterized by learnable weights $\boldsymbol{W}$:

$$\begin{equation}
\left(
  \boldsymbol{\tilde h}^{(l_1)}
  \otimes^{\boldsymbol{W}}_{cg}
  \boldsymbol{\tilde h}^{(l_2)}
\right)_m^{(l)} =
w_m
\sum_{m_1=-l_1}^{l_1}
\sum_{m_2=-l_2}^{l_2}
  C_{(l_1,m_1),(l_2,m_2)}^{(l,m)}
  h^{(l_1)}_{m_1} h^{(l_2)}_{m_2}
\end{equation}$$

where $C$ are the CG coefficients that assure the resulting vector is type- $l$ steerable.

This can be used to define a linear mapping between steerable features, which can be used in steerable MLPs. Since $\boldsymbol{\tilde a}$ is based on the spherical harmonics of the normalized edge feature $\boldsymbol{e}_{ij} / \lVert \boldsymbol{e}_{ij} \rVert$, this norm $d=\lVert \boldsymbol{e}_{ij} \rVert$ can be re-introduced in the learnable weights $\boldsymbol{W}(d)$, which gives the final linear mapping:

$$\begin{equation}
\boldsymbol{W}_{\!\boldsymbol{\tilde a}}(d)\ \boldsymbol{\tilde h}
\coloneqq
\boldsymbol{\tilde h}
\otimes^{\boldsymbol{W}(d)}_{cg}
\boldsymbol{\tilde a}
\end{equation}$$

The second part of (steerable) MLPs are the activation functions, which introduce the non-linearity. Currently available activation functions include Fourier-based ([Cohen et al., 2018](https://arxiv.org/abs/1801.10130)), norm-altering ([Thomas et al., 2018](https://arxiv.org/abs/1802.08219)), or gated non-linearities ([Weiler et al., 2018](https://proceedings.neurips.cc/paper/2018/hash/488e4104520c6aab692863cc1dba45af-Abstract.html)) ([Brandstetter et al., 2022](https://doi.org/10.48550/arXiv.2110.02905)).

Message passing networks on steerable features at node $i$ with neighbours $\mathcal{N}(i)$ can be summarized as some nonlinearity $\phi$ on the steerable feature ${\boldsymbol{\tilde h}}^{(l)}_i$ and some aggregated message ${\boldsymbol{\tilde m}}^{(l)}_i$. A message ${\boldsymbol{\tilde m}}_{ij}$, in turn, is defined as a nonlinearity $\psi$ between the neighbouring steerable features ${\boldsymbol{\tilde h}}^{(l)}_j$ and the corresponding edge feature $\boldsymbol{e}_{ij}$.

$$\begin{align}
  \boldsymbol{\tilde m}^{(l_m)}_{ij}
  &= \psi\left(
        \boldsymbol{\tilde h}^{(l_1)}_j,
        \boldsymbol{e}_{ij}
      \right) \\[5pt]
  \boldsymbol{\tilde m}^{(l_m)}_i &=
  \frac1{\lvert\mathcal{N}(i)\rvert}
  \sum_{j \in \mathcal{N}(i)}\
      \boldsymbol{\tilde m}^{(l_m)}_{ij} \\[13pt]
\boldsymbol{\tilde h}^{(l_{out})}_i &=
\phi\left(
  \boldsymbol{\tilde h}^{(l_n)}_i,\
  \boldsymbol{\tilde m}^{(l_m)}_i
\right) \\
\end{align}$$

In this work, updated node features only depend on the message passed, not the current node feature. Messages (indicated as type $l_m$) are therefore already of type $l_{out}$.

$$\begin{equation}
\boldsymbol{\tilde h}^{(l_{out})}_i
\coloneqq \boldsymbol{\tilde m}^{(l_m)}_i
\end{equation}$$

A message is defined as a single-layer perceptron making use of the CG tensor product as linear mapping parameterized by the edge feature $\boldsymbol{e}_{ij}$, and a gated nonlinearity $\sigma$

$$\begin{align}
\boldsymbol{\tilde m}^{(l_m)}_{ij}
&\coloneqq
\sigma\left(
  {\boldsymbol{\tilde h}}^{(l_n)}_j
  \otimes^{\boldsymbol{W}(\lVert \boldsymbol{e}_{ij} \rVert)}_{cg}
  \boldsymbol{{\tilde a}^{(l_e)}_{ij}}
\right)
\\
\text{where} \quad
{\boldsymbol{\tilde a}}^{(l_e)}_{ij} &\,= \left( Y^{(l_e)}_m\left( \frac{ \boldsymbol{e}_{ij} }{\lVert \boldsymbol{e}_{ij} \rVert} \right) \right)^T_{m=-l_e, \dots, l_e}
\end{align}$$

### 3.4. Method

#### Data

The Atom3D dataset provided the proteins that are used as input. The information consists of atom types and their position in euclidean space. As such, the model in this work takes these atoms as nodes, with their position and type as label. Different tasks associated with this data are described in [section 3.1](#31-tasks).

#### Model

Edges are drawn between any two nodes less than or equal to $4.5$ Angstroms units apart using the position of each atom. These edges are then encoded into a steerable vector in $V_{edge}=V_0 \otimes V_1$ (one type- $0$ and one type- $1$ steerable feature).

Each node label is embedded into a $32$-dimensional vector $n_{embed}$, with the equivalent steerable vector in $n_{embed}\,V_0$ ($32$ type- $0$ steerable features).

The input is passed through $3$ message passing layers, with hidden node features with $128$ coefficients, balanced across type- $0$ and $1$ vectors. Specifically, each layer has features $(65 V_0) \otimes (21 V_1)$ ($65$ scalar features and $21$ geometric vectors with $3$ coefficients each combine to $128$-dimensional features). The final message passing layer outputs only type- $0$ features. We develop two variants, one which immediately outputs a single scalar (per node) after the final convolutional layer (a $1V_0$ feature), and a second variant for which the convolutional output is in $16 V_0$ and put through a dense 2-layer perceptron with a hidden size of $32$ and a single scalar output.

Each message passing layer conditions the weights of the CG tensor product on the norm of the corresponding edge feature. This is done by first using Radial Basis Functions to obtain a $10$-dimensional encoding of this norm, and using a $2$ layer perceptron with a hidden size of $16$ and output size appropraite for the tensor product. The hidden layer makes use of a SiLU activation function.

The final node embeddings are aggregated using a global mean pooling layer.

All convolutional layers use gated-nonlinearities, with SiLU activation function for the type- $0$ features and sigmoid-gated type-$l>0$ features. The dense layers, if present, use ReLU activation functions and are trained with a dropout of $p=0.1$. The final convolutional and final dense layer do not have any activation functions.

#### Optimization

The ADAM optimizer with learning rate $10^{-4}$ and otherwise default parameters is used for training. LBA is a regression task, and therefore the MSE loss is used. Training is done with a batch size of $8$.

### 3.5. Testing Equivariance

In order to verify if our implementation of the steerable MLP is equivariant to rotation, we need to perform the same method used by the original authors as mentioned before. However, since we work with irreducible representations, the method needs some extra intermediate steps. Since the input of this model is represented using irreducible representations, each individual part needs to be rotated accordingly. So, after sampling a random 3D rotation matrix, it is transformed to do so. The remaining steps of testing equivariance is the same as described in [Section 1](#1-introduction). The implementation for this method/test can be found in this [notebook](./demos/testing_equivariance.ipynb).

## 4. Results

<!-- GVP reproduction RSME different runs on LBA split=30
    - run 1 : 1.577064037322998
    - run 2 : 1.616431474685669
    - run 3 : 1.6020700931549072
    - mean: 1.5985218683879
    - std: 0.019922128535226 -->

<!-- GVP reproduction RSME different runs on LBA split=60
    - run 1 : 1.5962501764297485
    - run 2 : 1.6832526922225952
    - run 3 : 1.6436277627944946
    - mean: 1.6410435438156
    - std: 0.043558788772982-->

<!-- sMLP RSME different runs on LBA split=30
    - run 1 : 1.5121526718139648
    - run 2 : 1.6194381713867188
    - run 3 : 1.491175651550293
    - mean: 1.540922164917
    - std: 0.068801026872796 -->

<!-- sMLP RSME different runs on LBA split=60
    - run 1 : 1.3332748413085938
    - run 2 : 1.3350021839141846
    - run 3 : 1.3006025552749634
    - mean: 1.3229598601659
    - std: 0.019381247111821  -->

<!-- sMLP DENSE RSME different runs on LBA split=30
    - run 1 : 1.5981481075286865
    - run 2 : 1.505611777305603
    - run 3 : 1.462575078010559
    - mean: 1.5221116542816
    -std: 0.069276229966334 -->

<!-- sMLP DENSE RSME different runs on LBA split=60
    - run 1 : 1.3841136693954468
    - run 2 : 1.3234361410140991
    - run 3 : 1.2702085971832275
    - mean: 1.3259194691976
    - std: 0.056993127288015    -->

In this section we compare the results of [Jing et al. (2021)](https://doi.org/10.48550/arXiv.2106.03843) to our reproduction and to the results of our implementation of a steerable MLP (sMLP), without and with dense layers, for both splits of the LBA task. Each task was run 3 times and the obtained metrics were averaged.

| Task                    | Metric               | Jing et al.          |     Reproduction     |  sMLP                | sMLP DENSE            |
|-------------------------|----------------------|----------------------|----------------------|----------------------|-----------------------|
| LBA (Split 30)          | RMSE &#8595;         | 1.594 &#177;  0.073  | 1.599 &#177;  0.020  |  1.541 &#177; 0.069  |  1.522 &#177; 0.070   |
| LBA (Split 60)          | RMSE &#8595;         | -                    | 1.641 &#177;  0.044  |  1.323 &#177; 0.019  |  1.326 &#177; 0.057   |

This table shows that the sMLP, with and without dense layer, outperforms the original and the reproduced results for the LBA task with split 30. The sMLP also significantly outperforms the reproduction of the GVP model for the LBA task with split 60.

For both tasks, the reproducing took on average 1.5 hours with a batch size of 8, whilst the sMLP (with and without dense layers) took on average 3 hours with a batch size of 1. The decrease of batch size was the consequence of the sMLP needing more memory during training than the GVP.

For the LBA task with split 30, the GVP model needed at least 47 epochs to acquire the best model, whilst the sMLP needed at most 22 epochs. For the LBA task with split 60, both models needed more than 40 epochs to find the best model.

<!-- Computational requirements -->
The GVP model has lower memory requirements than our sMLP implementation, both in terms of gradients that should be stored as well as the memory allocated during inference. To show this, we use datasample 10 for which the molecule consists of 551 atoms. The GVP model allocates 250 MB of GPU memory during inference and additionally stores 485 MB of gradients; our sMLP model allocates 630 MB and stores 707 MB of gradients.

Once training is done, the GVP model takes 1.7 seconds for 100 inferences when storing gradients and 1.3 s without.  Our sMLP implementation takes respectively 0.52 seconds and 0.46 seconds.

See the result for computational requirements in the supplementary notebook “[latency and memory](./demos/latency_and_memory.ipynb)”.

## 5. Conclusion

Since the sMLP for LBA task with split 30 reaches the best model significantly faster than the GVP model, we suspect that the sMLP can extract information faster than the GVP. For the other task, similarly to the GVP, the model needed the extra epochs to find the best model. However, it accomplishes this more efficiently and exhibits significantly better performance than the GVP. Although training the sMLP demands more time and memory resources, it compensates for it during inference by operating at a speed approximately 2 to 3 times faster than the GVP. These findings highlight the strengths of the sMLP model and its potential as a more effective and efficient solution in certain tasks compared to the GVP model.

In conclusion, we addressed the limited expressiveness of the GVP by incorporating a steerable basis, resulting in a more comprehensive representation of the data's geometry. This integration enabled effective communication between scalar and geometric features,  including orientation.

Regarding future enhancements, our focus would be on optimizing the memory requirements for training the steerable MLP, aiming to achieve a comparable level with the GVP in this aspect. By improving the memory efficiency, we aim to further refine the performance and competitiveness of the steerable MLP model.

## 6. Contributions

Moet hier de contribution van je personal essay?

<!-- Jip: My contribution to this project has been mostly in writing the core of the steerable model. In the first few weeks, I focussed writing some job files for the LISA cluster to reproduce the original paper. From there, my teammates were able to run all tasks covered in the GVP paper, and combine everything into the first research proposal and draft version of the final mini-project.

In the meantime, I started on the steerable model by taking the third tutorial (on graph neural networks) and slowly rewriting everything to our specific datasets. This consisted of adding comments and making small changes to let our model run on the Atom3D dataset. I was also able to use some code from the GVP paper as utility functions for this. Once the rough set-up of the model was there, I did not have access to a GPU to test the model performance (LISA was down at the time), so I continued finetuning and debugging the model with one teammate who had acces to a personal GPU.

Finally, I wrote the section on steerability in the final report, which is a more elaborate version of the Steerable Message Passing section above. -->

<!-- Zjos: As noted in the plan proposal my main contribution has been executing the experiments and monitoring these. During this I also have debugged a lot of the code and kept the team updated on our results, indicating whether our methods worked and we were making actual progress. At the start of the project there were issues with setting up the environment to reproduce the experiments of the original paper, on which I spend some time to fix it. The README has also been assigned to me to be kept updated throughout the project. In the end I have also taken on the responsibility to restructure our repository to match the preferred structure as shown on Canvas. - Overall I have been very active within my team to make sure I was contributing in a way they agreed with. - I hope y'all agree lol -->

<!-- Noa: At the beginning of the course, I created two collaborative PDFs of the papers for all of us to work and comment on together, allowing us to thoroughly examine and understand the papers and identify potential areas for improvement. Initially, my main focus was on attending the lectures, which often put me ahead in terms of course material, although I found some parts challenging to comprehend. While some members of my group were busy running the original authors' repository, I took on the task of writing the introduction for our blog post and highlighting as many potential areas for improvement in the paper as I could, based on my knowledge at that time. I put my findings in our blogpost as comments, which another team member further worked on. After Jip implemented the sMLP, I started working on the demo to test the equivariance for both the GVP and sMLP. I enjoyed taking the initiative to gather the team together (as other team members did as well) and provide a clear overview of what everyone was working on and who needed assistance with what. Towards the end, I focused on writing wherever it was needed. I contributed to sections on the results and wrote the conclusion. Additionally, I revised certain parts in the section titled "Our Contribution." -->

<!--Simon: In the beginning of this course my main objective was to understand the lectures and the objective of our assignment. Due to circomstances my pace of understanding was not as high as I expected it to be. This resulted that I was behind on knowledge in comparison to the team. While the rest was buzy reproducing the originals authors I used the aquired knowledge to write parts of the introduction, the first versions of the strengths and weaknesses of the GVP from the original authors and our contribution. Later a.o. Jip and Noa extended and improved these parts. By this time the rest of the team had managed to make the code of the original authors work. The reproduction cost a lot of computation so I ran some tasks on the Lisa cluster. I then focussed on figuring out what the tasks entailed. Then I explained this in my own word and as clear for a CS student as possible. By this time the final reproduction results were in and I made a table in the blogpost to display those with a description.
The next step was implementing our improvement of GVP. Altough I understood the concept of our idea, I had a hard time having a thurough enough understanding to implement it. Furthermore, the team was very fast at this stage, for me it felt like blinking two times and it was there. This made me realize even more that I was not performing as good as I expect from myself. The team liked to have a visualization of the input data. I threw myself on it and with help of the tutorial and time to figure understand the data and the code I managed to make it work. It is however very compute intensive and some proteins are too much to visualize. Our supervisor Cong also suggested we looked at performance outside of accuracy for a given task, but also look at computational costs. Also the difference in batchsize also gave different errors of the model. Together with Zjos I ran different experiments to pinpoint the problem. With the help of Zirk we managed to locate the error in our implementation and Jip solved it. In the final days the most important thing to do was the writing. I helpen where I could with this. At the end I mainly focced on making a good poster.
-->
