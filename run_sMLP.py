import argparse

parser = argparse.ArgumentParser()
parser.add_argument('task', metavar='TASK', choices=[
        'SMP', 'PSR', 'RSR', 'MSP', 'SMP', 'LBA', 'LEP', 'PPI', 'RES'
    ], help="{PSR, RSR, PPI, RES, MSP, SMP, LBA, LEP}")
parser.add_argument('--num-workers', metavar='N', type=int, default=4,
                   help='number of threads for loading data, default=4')
parser.add_argument('--smp-idx', metavar='IDX', type=int, default=None,
                   choices=list(range(20)),
                   help='label index for SMP, in range 0-19')
parser.add_argument('--lba-split', metavar='SPLIT', type=int, choices=[30, 60],
                    help='identity cutoff for LBA, 30 (default) or 60', default=30)
parser.add_argument('--batch', metavar='SIZE', type=int, default=8,
                    help='batch size, default=8')
parser.add_argument('--epochs', metavar='N', type=int, default=50,
                    help='training epochs, default=50')
parser.add_argument('--test', metavar='PATH', default=None,
                    help='evaluate a trained model')
parser.add_argument('--lr', metavar='RATE', default=1e-3, type=float,
                    help='learning rate')
parser.add_argument('--modeldir', metavar='DIR', default='sMLPmodels/',
                    help='directory to save models to')
parser.add_argument('--data', metavar='DIR', default='atom3d-data/',
                    help='directory to data')

# Added compared to run_atomd3d.py
parser.add_argument('--logdir', metavar='DIR', default='runs/',
                    help='directory to save models to')
parser.add_argument('--embed-dim', metavar='NUM', type=int, default=32,
                   help='Embedding size, default=32')
parser.add_argument('--hidden-dim', metavar='NUM', type=int, default=128,
                   help='Dimensionality of hidden irreps, will be balanced across type-l<=lmax irreps using `balanced_irreps`, default=128')
parser.add_argument('--l-max', metavar='LMAX', type=int, default=1,
                   help='Hidden representations will be of max this type, default=1')
parser.add_argument('--depth', metavar='DEPTH', type=int, default=3,
                   help='Number of convolutional layers, default=3')
parser.add_argument('--dense', action='store_true',
                    help='trigger additional dense layers')

args = parser.parse_args()

# For logging during the training process
import time
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

# For loading the task-specific data and metrics
from atom3dutils import get_datasets, get_metrics

# For the steerable MLP
from e3nn.o3 import Irreps
import torch_geometric as tg
import lightning.pytorch as lp
from steerable_mlp import ConvModel, Atom3D

def balanced_irreps(hidden_features:int, lmax:int) -> Irreps:
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

# Embed nodes as `args.embed_dim` * type-0 irreps
irreps_in = Irreps(f"{args.embed_dim}x0e")
# Hidden Irreps balanced across type-l<args.l_max, such that `irreps_hidden.dim == args.hidden_dim`
irreps_hidden = balanced_irreps(args.hidden_dim, args.l_max)
# Encode edges using spherical harmonics
irreps_edge = Irreps.spherical_harmonics(args.l_max)
# Convolutional layers output
if args.dense:
     irreps_out = Irreps("16x0e")
else:
    irreps_out = Irreps("1x0e")

model = ConvModel(
    irreps_in=irreps_in,
    irreps_hidden=irreps_hidden,
    irreps_edge=irreps_edge,
    irreps_out=irreps_out,
    depth=args.depth,
    dense=args.dense,
    )


# Loading Task-specific dataloaders and metrics

metrics:dict[str,callable]=get_metrics(args.task)
print("Test metrics:", list(metrics.keys()))

datasets:dict[str,any] = get_datasets(
    task=args.task,
    smp_idx=args.smp_idx,
    lba_split=args.lba_split,
    data_dir=args.data)

dataloaders:dict[str,tg.loader.DataLoader] = {
    "train": tg.loader.DataLoader(datasets['train'], batch_size=args.batch, num_workers=args.num_workers, shuffle=True),
    "valid": tg.loader.DataLoader(datasets['valid'], batch_size=args.batch, num_workers=args.num_workers),
    "test":  tg.loader.DataLoader(datasets['test'],  batch_size=args.batch, num_workers=args.num_workers),
}

# Setting up the logger
_name = str(args.task)

# SMP consists of 20 regression metrics, specify which is trained here
if _name == 'SMP':
    _name+=f'-smp_idx={args.smp_idx}'
# LBA has a 30 or 60 split, specify which is trained here
elif _name == 'LBA':
    _name+=f'-lba_split={args.lba_split}'

# Add version to not overwrite previous models
_version = f"{time.strftime('%Y%b%d-%T')}"

logger = TensorBoardLogger(
    save_dir=args.logdir,
    name=_name,
    version=_version,
)

checkpoint_callback = ModelCheckpoint(
    dirpath=args.modeldir,
    save_top_k=2,
    monitor="val/loss", #adjusted
    mode='min',
    save_on_train_epoch_end=True,
    filename=_name+"-"+_version+"-{epoch:02d}", #adjusted
    save_last=True,
)

plmodule = Atom3D(
    model=model,
    metrics=metrics,
    lr=args.lr,
)

# Set-up trainer
trainer = lp.Trainer(
    max_epochs=args.epochs,
    logger=logger,
    default_root_dir=args.modeldir,
    callbacks=[checkpoint_callback,],
)

# if no checkpoint given we train the model
if not args.test:
    trainer.fit(plmodule, dataloaders["train"], dataloaders["valid"])
    print("Best model:", checkpoint_callback.best_model_path)

# Testing best model - based on either the model just trained or the provided checkpoint
if checkpoint_callback.best_model_path:
    results = trainer.test(plmodule, dataloaders['test'], ckpt_path=checkpoint_callback.best_model_path)
elif args.test:
    results = trainer.test(plmodule, dataloaders['test'], ckpt_path=args.test)
