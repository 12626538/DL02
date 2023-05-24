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
parser.add_argument('--num-feat', metavar='NUM', type=int, default=32,
                   help='number of feature ...TODO, default=32')
parser.add_argument('--l-max', metavar='LMAX', type=int, default=1,
                   help='...TODO, default=1')
parser.add_argument('--depth', metavar='DEPTH', type=int, default=3,
                   help='number of layers ...TODO, default=3')
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


irreps_in = (Irreps("1x0e")*args.num_feat).simplify()
irreps_hidden = (Irreps.spherical_harmonics(args.l_max)*args.num_feat).sort()[0].simplify()
irreps_edge = Irreps.spherical_harmonics(args.l_max) #Irreps("1x1o")
irreps_out = Irreps("1x0e")

model = ConvModel(irreps_in, irreps_hidden, irreps_edge, irreps_out, args.depth)


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
# NOTE: perhaps adjust? because using tensorboard directly seems like less lines of code
_name = str(args.task)
if args.task == 'SMP': # kan dit niet gwn if '_name ==' zijn?
    _name+=f'-smp_idx={args.smp_idx}'
elif args.task == 'LBA':
    _name+=f'-lba_split={args.lba_split}'

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
    #dense=args.dense   #TODO
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
