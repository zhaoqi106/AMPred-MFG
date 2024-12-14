import torch
from functools import partial
import dgl.backend as F

from .MolGraph_Construction import smiles_to_Molgraph,ATOM_FEATURIZER, BOND_FEATURIZER
import numpy as np
from random import Random
from collections import defaultdict
from rdkit.Chem.Scaffolds import MurckoScaffold
from dgl.data.utils import  Subset
from rdkit.Chem import AllChem, MACCSkeys
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector

from .utils import ames

try:
    from rdkit import Chem
except ImportError:
    pass
def count_and_log(message, i, total, log_every_n):
    if (log_every_n is not None) and ((i + 1) % log_every_n == 0):
        print('{} {:d}/{:d}'.format(message, i + 1, total))
def prepare_mols(dataset, mols, sanitize, log_every_n=1000):
    if mols is not None:
        # Sanity check
        assert len(mols) == len(dataset), \
            'Expect mols to be of the same size as that of the dataset, ' \
            'got {:d} and {:d}'.format(len(mols), len(dataset))
    else:
        if log_every_n is not None:
            print('Start initializing RDKit molecule instances...')
        mols = []
        for i, s in enumerate(dataset.smiles):
            count_and_log('Creating RDKit molecule instance',
                          i, len(dataset.smiles), log_every_n)
            mols.append(Chem.MolFromSmiles(s, sanitize=sanitize))

    return mols
def scaffold_split(dataset, frac=None, balanced=True, include_chirality=False, ramdom_state=0):
    if frac is None:
        frac = [0.8, 0.1, 0.1]
    assert sum(frac) == 1
    mol_list = prepare_mols(dataset, None, True)
    n_total_valid = int(np.floor(frac[1] * len(mol_list)))
    n_total_test = int(np.floor(frac[2] * len(mol_list)))
    n_total_train = len(mol_list) - n_total_valid - n_total_test

    scaffolds_sets = defaultdict(list)
    for idx, mol in enumerate(mol_list):
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
        scaffolds_sets[scaffold].append(idx)

    random = Random(ramdom_state)

    # Put stuff that's bigger than half the val/test size into train, rest just order randomly
    if balanced:
        index_sets = list(scaffolds_sets.values())
        big_index_sets, small_index_sets = list(), list()
        for index_set in index_sets:
            if len(index_set) > n_total_valid / 2 or len(index_set) > n_total_test / 2:
                big_index_sets.append(index_set)
            else:
                small_index_sets.append(index_set)

        random.seed(ramdom_state)
        random.shuffle(big_index_sets)
        random.shuffle(small_index_sets)
        index_sets = big_index_sets + small_index_sets
    else:  # Sort from largest to smallest scaffold sets
        index_sets = sorted(list(scaffolds_sets.values()), key=lambda index_set: len(index_set), reverse=True)

    train_index, valid_index, test_index = list(), list(), list()
    for index_set in index_sets:
        if len(train_index) + len(index_set) <= n_total_train:
            train_index += index_set
        elif len(valid_index) + len(index_set) <= n_total_valid:
            valid_index += index_set
        else:
            test_index += index_set

    return [Subset(dataset, train_index),
                Subset(dataset, valid_index),
                Subset(dataset, test_index)]


import random
from torch.utils.data import Subset

import random
import numpy as np
from torch.utils.data import Subset


def random_split(dataset, frac=None, random_state=0):
    if frac is None:
        frac = [0.8, 0.1, 0.1]
    assert np.isclose(sum(frac), 1), "Fractions must sum to 1"

    random.seed(random_state)
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    n_total = len(dataset)
    n_total_train = int(np.ceil(frac[0] * n_total))
    n_total_valid = int(np.ceil(frac[1] * n_total))
    n_total_test = n_total - n_total_train - n_total_valid

    train_indices = indices[:n_total_train]
    valid_indices = indices[n_total_train:n_total_train + n_total_valid]
    test_indices = indices[n_total_train + n_total_valid:]

    return [Subset(dataset, train_indices),
            Subset(dataset, valid_indices),
            Subset(dataset, test_indices)]


def get_classification_dataset(dataset: str,
                               n_jobs: int,
                               seed: int,
                               split_ratio:list):
    assert dataset in ['Tox21', 'ClinTox',
                      'SIDER', 'BBBP', 'BACE']

    def get_task_pos_weights(labels, masks):
        num_pos = F.sum(labels, dim=0)
        num_indices = F.sum(masks, dim=0)
        task_pos_weights = (num_indices - num_pos) / num_pos
        return task_pos_weights

    def get_data(sub_data):
        gs, ys, ms = [], [], []
        smiles=[]
        for i in range(len(sub_data)):
            smiles.append(sub_data[i][0])
            gs.append(sub_data[i][1])
            ys.append(sub_data[i][2])
            ms.append(sub_data[i][3])
        ys = torch.stack(ys)
        ms = torch.stack(ms)
        task_weights = get_task_pos_weights(ys, ms)

        utils = GraphUtils()
        X, A = utils.preprocess_smile(smiles)

        fs = []
        macc = []
        rdit = []
        for i in range(len(smiles)):

            mol = Chem.MolFromSmiles(smiles[i])
            mogen_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            maccs_fp = torch.tensor(MACCSkeys.GenMACCSKeys(mol))
            rdit_fp = torch.tensor(Chem.RDKFingerprint(mol))

            fs.append(np.array(mogen_fp))
            macc.append(np.array(maccs_fp))
            rdit.append(np.array(rdit_fp))

        mogen_fp = torch.tensor(fs)
        maccs_fp = torch.tensor(macc)
        rdit_fp = torch.tensor(rdit)

        X = torch.FloatTensor(X)
        A = torch.FloatTensor(A)

        f_sum = mogen_fp.sum(axis=-1)
        macc_sum = maccs_fp.sum(axis=-1)
        rdit_fp_sum = rdit_fp.sum(axis=-1)

        mogen_fp = mogen_fp / (np.reshape(f_sum, (-1, 1)))
        maccs_fp = maccs_fp / (np.reshape(macc_sum, (-1, 1)))
        rdit_fp = rdit_fp / (np.reshape(rdit_fp_sum, (-1, 1)))
        return gs, ys, ms, task_weights,mogen_fp,maccs_fp,rdit_fp,X,A
    mol_g = partial(smiles_to_Molgraph)
    # data = getattr(dgldata, dataset)(mol_g,
    #                                  ATOM_FEATURIZER,
    #                                  BOND_FEATURIZER,
    #                                  n_jobs=n_jobs)
    # data= ames.SIDER(mol_g, ATOM_FEATURIZER, BOND_FEATURIZER, n_jobs=n_jobs)
    data= ames.AMES(mol_g, ATOM_FEATURIZER, BOND_FEATURIZER, n_jobs=n_jobs)
    # data = dili.SIDER(mol_g, ATOM_FEATURIZER, BOND_FEATURIZER, n_jobs=n_jobs)
    """
    train, val,test= ScaffoldSplitter.train_val_test_split(dataset=data,sanitize=False,
                                                      frac_train=split_ratio[0], 
                                                      frac_val=split_ratio[1],
                                                      frac_test=split_ratio[2],scaffold_func='smiles')
    """
    train,val,test=random_split(dataset=data)
    train_gs, train_ls, train_masks, train_tw,morgan_fp_list_train,macc_train,ecfp_train,X_tain,A_train= get_data(train)
    val_gs, val_ls, val_masks, val_tw,morgan_fp_list_val,macc_val,ecfp_val ,X_val,A_val= get_data(val)
    test_gs, test_ls, test_masks, test_tw ,morgan_fp_list_test,macc_test,ecfp_test,X_test,A_test= get_data(test)


    return train_gs, train_ls,train_tw, val_gs, val_ls,test_gs,test_ls,morgan_fp_list_train,\
        morgan_fp_list_val,morgan_fp_list_test,macc_train,macc_val,macc_test,ecfp_train,ecfp_val,ecfp_test,X_tain,X_val,X_test,A_train,A_val,A_test

import scipy.sparse as sp
def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)) # D
    d_inv_sqrt = np.power(rowsum, -0.5).flatten() # D^-0.5
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt) # D^-0.5
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo() # D^-0.5AD^0.5

class GraphUtils():
    def _convert_smile_to_graph(self, smiles):
        features = []
        adj = []
        maxNumAtoms = 100
        for smile in smiles:
            iMol = Chem.MolFromSmiles(smile)
            iAdjTmp = Chem.rdmolops.GetAdjacencyMatrix(iMol)

            iFeature = np.zeros((maxNumAtoms, 65))
            iFeatureTmp = []
            for atom in iMol.GetAtoms():
                iFeatureTmp.append(self.atom_feature(atom))
            iFeature[0:len(iFeatureTmp), 0:65] = iFeatureTmp
            # feature normalize
            iFeature = normalize(iFeature)

            # Adj-preprocessing
            iAdj = np.zeros((maxNumAtoms, maxNumAtoms))
            iAdj[0:len(iFeatureTmp), 0:len(iFeatureTmp)] = iAdjTmp + np.eye(len(iFeatureTmp))

            # adj normalize
            iAdj = normalize_adj(iAdj)

            features.append(iFeature)
            adj.append(iAdj.A)
        features = np.asarray(features)
        adj = np.asarray(adj)
        return features, adj

    def atom_feature(self, atom):
        return np.array(self.one_of_k_encoding_unk(atom.GetSymbol(),
                                                   ['C', 'N', 'O', 'S', 'F', 'H', 'Si', 'P', 'Cl', 'Br',
                                                    'Li', 'Na', 'K', 'Mg', 'Ca', 'Fe', 'As', 'Al', 'I', 'B',
                                                    'V', 'Tl', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn',
                                                    'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'Mn', 'Cr', 'Pt', 'Hg', 'Pb']) +
                        self.one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
                        self.one_of_k_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                        self.one_of_k_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) +
                        [atom.GetIsAromatic()] + self.get_ring_info(atom))

    def one_of_k_encoding_unk(self, x, allowable_set):
        """Maps inputs not in the allowable set to the last element."""
        if x not in allowable_set:
            x = allowable_set[-1]
        return list(map(lambda s: x == s, allowable_set))

    def one_of_k_encoding(self, x, allowable_set):
        if x not in allowable_set:
            raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
        return list(map(lambda s: x == s, allowable_set))

    def get_ring_info(self, atom):
        ring_info_feature = []
        for i in range(3, 9):
            if atom.IsInRingSize(i):
                ring_info_feature.append(1)
            else:
                ring_info_feature.append(0)
        return ring_info_feature

    def preprocess_smile(self, smiles):
        X, A = self._convert_smile_to_graph(smiles)
        return [X, A]