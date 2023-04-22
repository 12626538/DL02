# DL02
Repository for the DL02 course, apr 2023, Universiteit van Amsterdam

# Set up environment
First create virtual environment with the correct python version. We name the env `gvp` here.
Then install all the packages needed for the GVP.
Clone the github repo with all the GVP files.
Go in to folder.
Might need the flag `SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True` in order to install the files.
```bash
conda create -n gvp python==3.6.13
pip install -r requirements.txt
git clone https://github.com/drorlab/gvp-pytorch.git
cd gvp-pytorch
SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True pip install .
```

# Schedule [OUTDATED]

Below you can find the schedule, grouped per module

## [1pt] Module 7 - Self-supervised and Vision-Language Learning

- Lecture, apr 13 9-11, SP G0.05
- Lecutre, apr 25 9-11, G0.10-G0.12

## [2pt] Module 1 - Group Equivarent Deep Learning

- Lecture, apr 11 9-11, G0.05
- Seminar, apr 13 13-15, G0.05
- Lecture, apr 18 9-11, G0.05
- Seminar, apr 20 13-15, G0.05
- Lecture, apr 25 9-11, G0.05
- Seminar, apr 25 11-13, G0.05
- Lecture, may 09 9-11, G0.05
- Office Hour, may 9 11-13, G0.05
- Office Hour, may 16 11-13, G0.05
- Office Hour, may 23 11-13, G0.05

## [3pt] Module 5 - Diffusion and Advanced Generative Models

- Lecture, apr 13 9-11, G0.23-G0.25
- Lecture, apr 20 9-11, G0.05
- Lecture, apr 25 9-11, G3.02
- Lecture, may 09 9-11, G3.02
- Lecture, may 11 9-11, G3.02


# Usage on LISA

**TODO**[This is still work in progress] First make sure to create an enviornment with the requirements in the GVP repo. [This file](./install_env.job) should do the job, but has not been shown to work....

Download the dataset files in some directory. If you are planning to use [`run_atom3d.py`](./gvp-pytorch/run_atom3d.py), keep the file structure found [here](https://github.com/drorlab/gvp-pytorch/blob/82af6b22eaf8311c15733117b0071408d24ed877/run_atom3d.py#L207).

Take a look at a job file like [this one](./run_atom3d.job) to try to understand what needs to happen to run LISA.
