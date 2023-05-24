## Project Description
[Jing et al., (2020)](https://arxiv.org/abs/2009.01411) proposed a method that combines the strengths of CNNs and GNNs to learn from biomolecular structures. This involves changing the multilayer perceptrons (MLPs) with geometric vector perceptrons (GVPs). The GVP approach is used to learn the relationship between protein sequences and their structures, operating on geometric objects rather than scalar values. In a follow-up paper by [Jing et al, (2021)](https://arxiv.org/abs/2106.03843), the authors extended the GVP-GNN architecture to handle atomic-level structure representations with vector gating, replacing the vector non-linearity. This retains the rotational equivariance of the vector features, but this version of the GVP can only exchange information from scalar vectors to type-1 vectors and vice versa, using the norm. This triggered us to figure out an approach to take away the weak point while maintaining the strength. 

## Setting up the environment
First create virtual environment with the correct python version. We name the env `gvp` here. Then install all the packages needed for the GVP.
Might need to change the building wheel in order to install the files. For this navigate to https://data.pyg.org/whl/ and find the correct link.
```bash
conda create -n gvp python==3.11.3 pip
conda activate gvp
pip install torch==2.0.0 # required install before the packages
pip install -r requirements.txt
```

## Downloading the data
The datasets for each task are found by navigating to <https://www.atom3d.ai/> and selecting the task. The zip-files have to be structured within the data folder as follows, except for the `RES`-task, which has an additional 'raw' data file:
```
.
├── atom3d-data
|    ├── <TASK>
|    |   └── splits
|    |       └── <extracted zip-file(s)>
|    └── RES
|        ├── raw
|        └── splits
|            └── split-by-cath-topology
└── models
```
The files within the split folder for the tasks are as follows:
```
* LBA: split-by-sequence-identity-30
* LBA: split-by-sequence-identity-60
* LEP: split-by-protein 
* MSP: split-by-sequence-identity-30
* PPI: DIPS-split
* PSR: split-by-year
* RES: split-by-cath-topology
* RSR: candidates-by-split-by-time
* SMP: random
```
Note an additional empty folder 'models' is created in the same directory to allow for saving the model checkpoints during training.

## Training the models with the GVP model
The models can be trained by specifying the task and additional arguments as follows:
```
python run_atom3d.py <TASK> <Additional arguments>
```
These additional arguments can be task-specific for the following tasks:
```
* LBA: --lba-split  [30, 60]
* SMP: --smp-idx    [0 .. 19]
```
This also allows for setting hyperparameters as noted by the [original authors](https://github.com/drorlab/gvp-pytorch#training--testing-1)

With the now added option to monitor the data with `--monitor` which creates the files needed to track the training and validation loss with tensorboard.

## Reproducing the original results
To reproduce the original tasks the code was run with default paramaters where the only change occured in increasing or decreasing batch-size. For the `SMP` task only the indexes 3, 7, and 15 were required. The results are then aquired by running the following for each task.

```
python run_atom3d.py --test model/<checkpoint> <Additional arguments>
```

This returns the task-specific result metrics, which will are reported and discussed in the following [blogpost](./BLOGPOST.md). These results can also be obtained from the model checkpoints with the help of the [demo-notebook](./demo.ipynb).

## Extending with Steerable MLPs
TODO explain a bit about the extension and how it differs (compact version of the text in the blogpost.md)

## Training on ATOM3d with the extended models
The training of the extended model is similar to the GVP model. The following file can be used by specifying the task and additional arguments as follows:
```
python run_sMLP.py <TASK> <Additional arguments>
```
These additional arguments again include the task-specifics arguments for `LBA` and `SMP`. It also allows for setting the following properties of the model:
```
* --l-max       >1
* --embed-dim   
* --hidden-dim 
* --depth
```
Then ...

## Steerable MLP results
Below we show short summary of the results obtained by the steerable MLP model, focused on the LBA (split 30 shown) task. 
<!--  report improvement in metrics (as reported in the original paper) arrows indicate whether higher or lower is considered 'beter' (or if they're all better when lower we can inidcate whether there was an improvement next to the result) -->

|                       | RMSE &#8595;  |
| -------------         | ------------- |
| GVP (original paper)  | 1.594 &#177; 0.073    | 
| GVP (reproduced)      | 1.598  &#177; 0.020   |
| sMLP                  | xx &#177;             |
| sMLP (dense)          | ....   &#177;         |

<!-- down &#8595; ->
<!-- up &#8593; -->

<!-- GVP reproduction RSME different runs op LBA split=30
    - run 1 : 1.577064037322998 
    - run 2 : 1.616431474685669
    - run 3 : 1.6020700931549072 -->

<!-- GVP reproduction RSME different runs op LBA split=60
    - run 1 : 1.5962501764297485 
    - run 2 : 1.6832526922225952
    - run 3 : 1.6436277627944946 -->

## Deep Learning 2
This repository contains the code and final delivery for the mini-project assignment by '*Synthesized Solutions*' for the DL02 course, april 2023, University of Amsterdam

As of may 14 2023 the project plan has been completed as follows:
- [x] Study the paper and the original code
- [x] Create set up for reproduction and expansion of the original paper
- [x] Recreate the original papers results
- [x] Report on the reproduced results
- [x] Implement our proposed expansion 
- [ ] Report on the results with expansion
- [ ] Finish final deliverables and present on June 1st 2023