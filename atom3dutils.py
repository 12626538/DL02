import torch, random, scipy, math
import pandas as pd
import numpy as np
from atom3d.datasets import LMDBDataset
from torch.utils.data import IterableDataset
import torch_cluster, torch_geometric
from functools import partial
from collections import defaultdict
import sklearn.metrics as sk_metrics
from atom3d.util import metrics

# Supresses the following warning:
#   UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class.
#   This should only matter to you if you are using storages directly.
#   To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
# Source yet undetermined
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

_NUM_ATOM_TYPES = 9
_element_mapping = lambda x: {
    'H' : 0,
    'C' : 1,
    'N' : 2,
    'O' : 3,
    'F' : 4,
    'S' : 5,
    'Cl': 6, 'CL': 6,
    'P' : 7
}.get(x, 8)
_DEFAULT_V_DIM = (100, 16)
_DEFAULT_E_DIM = (32, 1)


def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))

def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design

    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    rbf = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return D

def _edge_features(coords, edge_index, D_max=4.5, num_rbf=16, device='cpu'):

    E_vectors = coords[edge_index[0]] - coords[edge_index[1]]
    rbf = _rbf(E_vectors.norm(dim=-1),
               D_max=D_max, D_count=num_rbf, device=device)

    edge_s = rbf
    edge_v = _normalize(E_vectors).unsqueeze(-2)

    edge_s, edge_v = map(torch.nan_to_num,
            (edge_s, edge_v))

    return edge_s, edge_v

def get_datasets(task, data_dir='/media/jip/T7/DL02/data/', lba_split=None, smp_idx=None):
    assert task in {'PPI','RSR','PSR','MSP','LEP','LBA','SMP'}, f"Unknown task {task}"
    assert task!='SMP' or smp_idx in range(20), "Set SMP index (range 0-19) when running SMP task"
    assert task!='LBA' or lba_split in {30,60}, "Set LBA split (30,60) when running LBA task"

    data_path = data_dir + {
        'PPI' : 'PPI/splits/DIPS-split/data/',
        'RSR' : 'RSR/splits/candidates-split-by-time/data/',
        'PSR' : 'PSR/splits/split-by-year/data/',
        'MSP' : 'MSP/splits/split-by-sequence-identity-30/data/',
        'LEP' : 'LEP/splits/split-by-protein/data/',
        'LBA' : f'LBA/splits/split-by-sequence-identity-{lba_split}/data/',
        'SMP' : 'SMP/splits/random/data/'
    }[task]

    if task == 'PPI':
        trainset = PPIDataset(data_path+'train')
        valset = PPIDataset(data_path+'val')
        testset = PPIDataset(data_path+'test')

    else:
        transform = {
            'RSR' : RSRTransform,
            'PSR' : PSRTransform,
            'MSP' : MSPTransform,
            'LEP' : LEPTransform,
            'LBA' : LBATransform,
            'SMP' : partial(SMPTransform, smp_idx=smp_idx),
        }[task]()

        trainset = LMDBDataset(data_path+'train', transform=transform)
        valset = LMDBDataset(data_path+'val', transform=transform)
        testset = LMDBDataset(data_path+'test', transform=transform)

    return {
        "train": trainset,
        "valid": valset,
        "test": testset,
    }

def get_metrics(task):
    def _correlation(metric, targets, predict, ids=None, glob=True):
        if glob: return metric(targets, predict)
        _targets, _predict = defaultdict(list), defaultdict(list)
        for _t, _p, _id in zip(targets, predict, ids):
            _targets[_id].append(_t)
            _predict[_id].append(_p)
        return np.mean([metric(_targets[_id], _predict[_id]) for _id in _targets])

    correlations = {
        'pearson': partial(_correlation, metrics.pearson),
        'kendall': partial(_correlation, metrics.kendall),
        'spearman': partial(_correlation, metrics.spearman)
    }
    mean_correlations = {f'mean {k}' : partial(v, glob=False) \
                            for k, v in correlations.items()}

    return {
        'RSR' : {**correlations, **mean_correlations},
        'PSR' : {**correlations, **mean_correlations},
        'PPI' : {'auroc': metrics.auroc},
        'RES' : {'accuracy': metrics.accuracy},
        'MSP' : {'auroc': metrics.auroc, 'auprc': metrics.auprc},
        'LEP' : {'auroc': metrics.auroc, 'auprc': metrics.auprc},
        'LBA' : {'rmse': partial(sk_metrics.mean_squared_error, squared=False)}, #**correlations removed for now
        'SMP' : {'mae': sk_metrics.mean_absolute_error}
    }[task]

class BaseTransform:
    '''
    Implementation of an ATOM3D Transform which featurizes the atomic
    coordinates in an ATOM3D dataframes into `torch_geometric.data.Data`
    graphs. This class should not be used directly; instead, use the
    task-specific transforms, which all extend BaseTransform. Node
    and edge features are as described in the EGNN manuscript.

    Returned graphs have the following attributes:
    -x          atomic coordinates, shape [n_nodes, 3]
    -atoms      numeric encoding of atomic identity, shape [n_nodes]
    -edge_index edge indices, shape [2, n_edges]
    -edge_s     edge scalar features, shape [n_edges, 16]
    -edge_v     edge scalar features, shape [n_edges, 1, 3]

    Subclasses of BaseTransform will produce graphs with additional
    attributes for the tasks-specific training labels, in addition
    to the above.

    All subclasses of BaseTransform directly inherit the BaseTransform
    constructor.

    :param edge_cutoff: distance cutoff to use when drawing edges
    :param num_rbf: number of radial bases to encode the distance on each edge
    :device: if "cuda", will do preprocessing on the GPU
    '''
    def __init__(self, edge_cutoff=4.5, num_rbf=16, device='cpu'):
        self.edge_cutoff = edge_cutoff
        self.num_rbf = num_rbf
        self.device = device

    def __call__(self, df):
        '''
        :param df: `pandas.DataFrame` of atomic coordinates
                    in the ATOM3D format

        :return: `torch_geometric.data.Data` structure graph
        '''
        with torch.no_grad():
            pos = torch.as_tensor(df[['x', 'y', 'z']].to_numpy(),
                                     dtype=torch.float32, device=self.device)
            atoms = torch.as_tensor(list(map(_element_mapping, df.element)),
                                            dtype=torch.long, device=self.device)

            edge_index = torch_cluster.radius_graph(pos, r=self.edge_cutoff)

            return torch_geometric.data.Data(pos=pos, z=atoms, edge_index=edge_index)

########################################################################

class SMPTransform(BaseTransform):
    '''
    Transforms dict-style entries from the ATOM3D SMP dataset
    to featurized graphs. Returns a `torch_geometric.data.Data`
    graph with attribute `label` and all structural attributes
    as described in BaseTransform.

    Includes hydrogen atoms.
    '''
    def __init__(self, smp_idx:int, *args, **kwargs):
        self.smp_idx = smp_idx
        super().__init__(*args, **kwargs)

    def __call__(self, elem):
        data = super().__call__(elem['atoms'])
        with torch.no_grad():
            _label = torch.as_tensor(elem['labels'],
                            device=self.device, dtype=torch.float32)
            data.label = _label[self.smp_idx::20]
        return data


########################################################################

class PPIDataset(IterableDataset):
    '''
    A `torch.utils.data.IterableDataset` wrapper around a
    ATOM3D PPI dataset. Extracts (many) individual amino acid pairs
    from each structure of two interacting proteins. The returned graphs
    are seperate and each represents a 30 angstrom radius from the
    selected residue's alpha carbon.

    On each iteration, returns a pair of `torch_geometric.data.Data`
    graphs with the (same) attribute `label` which is 1 if the two
    amino acids interact and 0 otherwise, `ca_idx` for the node index
    of the alpha carbon, and all structural attributes as
    described in BaseTransform.

    Modified from
    https://github.com/drorlab/atom3d/blob/master/examples/ppi/gnn/data.py

    Excludes hydrogen atoms.

    :param lmdb_dataset: path to ATOM3D dataset
    '''
    def __init__(self, lmdb_dataset):
        self.dataset = LMDBDataset(lmdb_dataset)
        self.transform = BaseTransform()

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            gen = self._dataset_generator(list(range(len(self.dataset))), shuffle=True)
        else:
            per_worker = int(math.ceil(len(self.dataset) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.dataset))
            gen = self._dataset_generator(
                list(range(len(self.dataset)))[iter_start:iter_end],
                shuffle=True)
        return gen

    def _df_to_graph(self, struct_df, chain_res, label):

        struct_df = struct_df[struct_df.element != 'H'].reset_index(drop=True)

        chain, resnum = chain_res
        res_df = struct_df[(struct_df.chain == chain) & (struct_df.residue == resnum)]
        if 'CA' not in res_df.name.tolist():
            return None
        ca_pos = res_df[res_df['name']=='CA'][['x', 'y', 'z']].astype(np.float32).to_numpy()[0]

        kd_tree = scipy.spatial.KDTree(struct_df[['x','y','z']].to_numpy())
        graph_pt_idx = kd_tree.query_ball_point(ca_pos, r=30.0, p=2.0)
        graph_df = struct_df.iloc[graph_pt_idx].reset_index(drop=True)

        ca_idx = np.where((graph_df.chain == chain) & (graph_df.residue == resnum) & (graph_df.name == 'CA'))[0]
        if len(ca_idx) != 1:
            return None

        data = self.transform(graph_df)
        data.label = label

        data.ca_idx = int(ca_idx)
        data.n_nodes = data.num_nodes

        return data

    def _dataset_generator(self, indices, shuffle=True):
        if shuffle: random.shuffle(indices)
        with torch.no_grad():
            for idx in indices:
                data = self.dataset[idx]

                neighbors = data['atoms_neighbors']
                pairs = data['atoms_pairs']

                for i, (ensemble_name, target_df) in enumerate(pairs.groupby(['ensemble'])):
                    sub_names, (bound1, bound2, _, _) = nb.get_subunits(target_df)
                    positives = neighbors[neighbors.ensemble0 == ensemble_name]
                    negatives = nb.get_negatives(positives, bound1, bound2)
                    negatives['label'] = 0
                    labels = self._create_labels(positives, negatives, num_pos=10, neg_pos_ratio=1)

                    for index, row in labels.iterrows():

                        label = float(row['label'])
                        chain_res1 = row[['chain0', 'residue0']].values
                        chain_res2 = row[['chain1', 'residue1']].values
                        graph1 = self._df_to_graph(bound1, chain_res1, label)
                        graph2 = self._df_to_graph(bound2, chain_res2, label)
                        if (graph1 is None) or (graph2 is None):
                            continue
                        yield graph1, graph2

    def _create_labels(self, positives, negatives, num_pos, neg_pos_ratio):
        frac = min(1, num_pos / positives.shape[0])
        positives = positives.sample(frac=frac)
        n = positives.shape[0] * neg_pos_ratio
        n = min(negatives.shape[0], n)
        negatives = negatives.sample(n, random_state=0, axis=0)
        labels = pd.concat([positives, negatives])[['chain0', 'residue0', 'chain1', 'residue1', 'label']]
        return labels

########################################################################

class LBATransform(BaseTransform):
    '''
    Transforms dict-style entries from the ATOM3D LBA dataset
    to featurized graphs. Returns a `torch_geometric.data.Data`
    graph with attribute `label` for the neglog-affinity
    and all structural attributes as described in BaseTransform.

    The transform combines the atomic coordinates of the pocket
    and ligand atoms and treats them as a single structure / graph.

    Includes hydrogen atoms.
    '''
    def __call__(self, elem):
        pocket, ligand = elem['atoms_pocket'], elem['atoms_ligand']
        df = pd.concat([pocket, ligand], ignore_index=True)

        data = super().__call__(df)
        with torch.no_grad():
            data.label = elem['scores']['neglog_aff']
            lig_flag = torch.zeros(df.shape[0], device=self.device, dtype=torch.bool)
            lig_flag[-len(ligand):] = 1
            data.lig_flag = lig_flag
        return data

########################################################################

class LEPTransform(BaseTransform):
    '''
    Transforms dict-style entries from the ATOM3D LEP dataset
    to featurized graphs. Returns a tuple (active, inactive) of
    `torch_geometric.data.Data` graphs with the (same) attribute
    `label` which is equal to 1. if the ligand activates the protein
    and 0. otherwise, and all structural attributes as described
    in BaseTransform.

    The transform combines the atomic coordinates of the pocket
    and ligand atoms and treats them as a single structure / graph.

    Excludes hydrogen atoms.
    '''
    def __call__(self, elem):
        active, inactive = elem['atoms_active'], elem['atoms_inactive']
        with torch.no_grad():
            active, inactive = map(self._to_graph, (active, inactive))
        active.label = inactive.label = 1. if elem['label'] == 'A' else 0.
        return active, inactive

    def _to_graph(self, df):
        df = df[df.element != 'H'].reset_index(drop=True)
        return super().__call__(df)

########################################################################

class MSPTransform(BaseTransform):
    '''
    Transforms dict-style entries from the ATOM3D MSP dataset
    to featurized graphs. Returns a tuple (original, mutated) of
    `torch_geometric.data.Data` graphs with the (same) attribute
    `label` which is equal to 1. if the mutation stabilizes the
    complex and 0. otherwise, and all structural attributes as
    described in BaseTransform.

    The transform combines the atomic coordinates of the two proteis
    in each complex and treats them as a single structure / graph.

    Adapted from
    https://github.com/drorlab/atom3d/blob/master/examples/msp/gnn/data.py

    Excludes hydrogen atoms.
    '''
    def __call__(self, elem):
        mutation = elem['id'].split('_')[-1]
        orig_df = elem['original_atoms'].reset_index(drop=True)
        mut_df = elem['mutated_atoms'].reset_index(drop=True)
        with torch.no_grad():
            original, mutated = self._transform(orig_df, mutation), \
                                self._transform(mut_df, mutation)
        original.label = mutated.label = 1. if elem['label'] == '1' else 0.
        return original, mutated

    def _transform(self, df, mutation):

        df = df[df.element != 'H'].reset_index(drop=True)
        data = super().__call__(df)
        data.node_mask = self._extract_node_mask(df, mutation)
        return data

    def _extract_node_mask(self, df, mutation):
        chain, res = mutation[1], int(mutation[2:-1])
        idx = df.index[(df.chain.values == chain) & (df.residue.values == res)].values
        mask = torch.zeros(len(df), dtype=torch.long, device=self.device)
        mask[idx] = 1
        return mask

########################################################################

class PSRTransform(BaseTransform):
    '''
    Transforms dict-style entries from the ATOM3D PSR dataset
    to featurized graphs. Returns a `torch_geometric.data.Data`
    graph with attribute `label` for the GDT_TS, `id` for the
    name of the target, and all structural attributes as
    described in BaseTransform.

    Includes hydrogen atoms.
    '''
    def __call__(self, elem):
        df = elem['atoms']
        df = df[df.element != 'H'].reset_index(drop=True)
        data = super().__call__(df)
        data.label = elem['scores']['gdt_ts']
        data.id = eval(elem['id'])[0]
        return data

########################################################################

class RSRTransform(BaseTransform):
    '''
    Transforms dict-style entries from the ATOM3D RSR dataset
    to featurized graphs. Returns a `torch_geometric.data.Data`
    graph with attribute `label` for the RMSD, `id` for the
    name of the target, and all structural attributes as
    described in BaseTransform.

    Includes hydrogen atoms.
    '''
    def __call__(self, elem):
        df = elem['atoms']
        df = df[df.element != 'H'].reset_index(drop=True)
        data = super().__call__(df)
        data.label = elem['scores']['rms']
        data.id = eval(elem['id'])[0]
        return data
