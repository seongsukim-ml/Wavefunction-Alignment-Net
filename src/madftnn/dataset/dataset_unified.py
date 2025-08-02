from madftnn.dataset.sqlite_database.hamiltonian_database_qhnet import HamiltonianDatabase_qhnet

import lmdb
import numpy as np
import torch

#!/usr/bin/env python3

import numpy as np

from torch_geometric.transforms.radius_graph import RadiusGraph
import torch

from tqdm import tqdm
# from cal_initH import cal_initH

import bisect
import pickle
from torch.utils.data import Dataset
import lmdb
import glob
from .buildblock import *
import copy

class HamiltonianDataset_qhnet_clean(torch.utils.data.Dataset):
    def __init__(
        self, 
        filepath, 
        enable_hami = True,
        dtype=torch.float32,
        remove_init=False,
        **kwargs
        ):
        super(HamiltonianDataset_qhnet_clean, self).__init__()
        
        self.dtype = dtype
        self.database = HamiltonianDatabase_qhnet(filepath)

        self.enable_hami = enable_hami
        self.remove_init=remove_init
        
    def __len__(self):
        return len(self.database)
    
    def pairwise_distances(self, coords):
        dists = np.zeros((coords.shape[0], coords.shape[0]))
        if len(coords.shape)!=2 or coords.shape[1]!=3:
            raise ValueError(f"sorry, the coords shape is {coords.shape}, please check")
        dists = np.sqrt(
                    np.sum((coords.reshape(-1,1,3)-coords.reshape(1,-1,3))**2,axis = -1)+1e-10
            )
        return dists

    def __getitem__(self, idx):
        R, E, F, diag, non_diag, diag_init, non_diag_init, diag_mask, non_diag_mask, A, G, L = \
            self.database[idx]
        
        local_orbitals = []
        local_orbitals_number = 0
        for z in L:
            local_orbitals.append(
                tuple((int(z), int(l)) for l in self.database.get_orbitals(z))
            )
            local_orbitals_number += sum(2 * l + 1 for _, l in local_orbitals[-1])
        if self.remove_init:
            diag = diag -diag_init
            non_diag = non_diag -non_diag_init
        out= {'pos': R, 
                'energy': E,
                'forces': F,
                # "full_hamiltonian": torch.cat([t.view(1, -1) for t in H], dim=1).squeeze(0),
                # "full_hamiltonian": torch.block_diag(*H),
                # "overlap_matrix": torch.block_diag(*S),
                # "core_hamiltonian": torch.block_diag(*C),
                'edge_index': A, 
                'labels': G,
                'atomic_numbers': L,
                'molecule_size':F.shape[0],
                # "orbitals": orbitals,
                # "mask": masks,#torch.block_diag(*mask),
                # "mask_l1": torch.block_diag(*mask_l1),
                # "init_hamiltonian": torch.cat([t.view(1, -1) for t in initH], dim=1).squeeze(0),
                # "init_hamiltonian": torch.block_diag(*initH),
                # "batch":idx
                }
        if self.enable_hami:
            dis = self.pairwise_distances(R)
            non_diag_mask_dis = np.triu(np.ones_like(dis, dtype=bool), k=1)
            non_diag_dis = dis[non_diag_mask_dis+non_diag_mask_dis.T]
            dis_mask = np.exp(-0.09378197*non_diag_dis*non_diag_dis-0.61498875*non_diag_dis-3.61389775)    #QH9 gap

            out.update({
                    'diag_hamiltonian': diag,
                        'non_diag_hamiltonian': non_diag,
                        'diag_mask': diag_mask,
                        'non_diag_mask': non_diag_mask,
                        "mask_l1": dis_mask})



        return out
    
DATA_SPLIT_RATIO = {'buckyball_catcher':[600./6102,50./6102, 1 - 650./6102],
            'double_walled_nanotube':[800./5032,100./5032, 1 - 900./5032],
            'AT_AT':[ 3000./19990, 200./19990, 1 - 3200./19990],
            'AT_AT_CG_CG':[ 2000./10153, 200./10153, 1 - 2200./10153],
            'stachyose':[ 8000./27138, 800./27138, 1 - 8800./27138],
            'DHA':[ 8000./69388, 800./69388, 1 - 8800./69388],
            'Ac_Ala3_NHMe':[ 6000./85109, 600./85109, 1 - 6600./85109],
        }
SYSTEM_REF = {
        "Ac_Ala3_NHMe":  
        -620662.75,
        "AT_AT": 
        -1154896.6,
        "AT_AT_CG_CG": 
        -2329950.2,
        "DHA":
        -631480.2,
        "stachyose":
        -1578839.0,
        "buckyball_catcher": # buckyball_catcher/radius3_broadcast_kmeans
        -2877475.2,
        "double_walled_nanotube": # double_walled_nanotube/radius3_broadcast_kmeans
        -7799787.0,
}


# when train ratio is -1, we can use this pre-defined split ratio
def get_data_default_config(data_name):
    # train ratio , val ratio,test ratio can be int or float.
    train_ratio,val_ratio,test_ratio = None,None,None
    if data_name.lower() == "qh9":
        train_ratio,val_ratio,test_ratio = 0.8,0.1,0.1
        atom_reference = np.zeros([20])
        system_ref = 0.
    elif data_name.lower() == "pubchem":
        train_ratio,val_ratio,test_ratio = 0.8,0.1,0.1
        atom_reference = np.array([0.0000, -376.3395, 0.0000, 0.0000, 0.0000,
                        0.0000,-23905.9824,-34351.3164,-47201.4062,0.0000,
                        0.0000,0.0000,0.0000,0.0000,0.0000,
                        -214228.1250,-249841.3906])
        system_ref = 0.
    else:
        atom_reference = np.zeros([20])
        system_ref = SYSTEM_REF[data_name]
        train_ratio,val_ratio,test_ratio = DATA_SPLIT_RATIO[data_name]
    return atom_reference,system_ref,train_ratio,val_ratio,test_ratio

def get_full_energy(data_name, energy, atomic_numbers):
    atom_reference, system_ref,_,_,_ = get_data_default_config(data_name)
    unique,counts = np.unique(atomic_numbers, return_counts=True)
    energy += np.sum(atom_reference[unique]*counts)
    energy += system_ref
    return energy
    

class LmdbDataset(Dataset):
    r"""Dataset class to load from LMDB files containing relaxation
    trajectories or single point computations.
    Useful for Structure to Energy & Force (S2EF), Initial State to
    Relaxed State (IS2RS), and Initial State to Relaxed Energy (IS2RE) tasks.
    Args:
            @path: the path to store the data
            @task: the task for the lmdb dataset
            @split: splitting of the data
            @transform: some transformation of the data
    """
    energy = 'energy'
    forces = 'forces'
    def __init__(self, path, 
                 data_name = "pubchem",
                 transforms = [], 
                 enable_hami = True,
                 old_blockbuild = False,
                 remove_init = False,
                 remove_atomref_energy = False,
                 Htoblock_otf = True, ## on save H matrix, H to block is process in collate unifined for memory saving.
                 basis = "def2-tzvp"):
        super(LmdbDataset, self).__init__()
        if data_name.lower() == "pubchem":
            if basis != "def2-tzvp":
                raise ValueError("sorry, when using pubchem the basis should be def2-tzvp")
        self.atom_reference, self.system_ref, _,_,_ = get_data_default_config(data_name)
        db_paths = []
        if isinstance(path,str):
            if path.endswith("lmdb"):
                db_paths.append(path)
            else:
                db_paths.extend(glob.glob(path+"/*.lmdb"))
                
        elif isinstance(path,list):
            for p in path:
                if p.endswith("lmdb"):
                    db_paths.append(p)
                else:
                    db_paths.extend(glob.glob(p+"/*.lmdb"))
        # print(db_paths)
        assert len(db_paths) > 0, f"No LMDBs found in '{self.path}'"
        self.enable_hami = enable_hami
        self._keys, self.envs = [], []
        self.db_paths = sorted(db_paths)
        self.open_db()
        self.transforms = transforms
        self.remove_init = remove_init
        self.remove_atomref_energy = remove_atomref_energy
        self.conv, self.orbitals_ref, self.mask,self.chemical_symbols = None,None,None,None
        self.old_blockbuild = old_blockbuild
        self.Htoblock_otf = Htoblock_otf
        if self.enable_hami:
            if (not self.old_blockbuild):
                self.conv, _, self.mask,_ = get_conv_variable_lin(basis)
            else:
                self.conv, self.orbitals_ref, self.mask,self.chemical_symbols = get_conv_variable(basis)

    def open_db(self):
        for db_path in self.db_paths:
            self.envs.append(self.connect_db(db_path))
            length = pickle.loads(
                self.envs[-1].begin().get("length".encode("ascii"))
            )
            self._keys.append(list(range(length)))
 
        keylens = [len(k) for k in self._keys]
        self._keylen_cumulative = np.cumsum(keylens).tolist()
        self.num_samples = sum(keylens)

           
   
    def __len__(self):
        return self.num_samples
 
    def __getitem__(self, idx):
        # Data(file_name='/data/recalculated_data/adequate-worm/GePar3-yl/111255333_pos.npy.npz',
        # atomic_numbers=[68, 1], energy=[1, 1], forces=[68, 3], 
        # init_fock=[1158, 1158], fock=[1158, 1158], num_nodes=68, pos=[68, 3],
        # edge_index=[2, 1388], labels=[68], num_labels=5, 
        # grouping_graph=[2, 1112], interaction_graph=[2, 214], id=0)

        # Figure out which db this should be indexed from.
        db_idx = bisect.bisect(self._keylen_cumulative, idx)
        # Extract index of element within that db.
        el_idx = idx
        if db_idx != 0:
            el_idx = idx - self._keylen_cumulative[db_idx - 1]
        assert el_idx >= 0

        # Return features.
        datapoint_pickled = (
            self.envs[db_idx]
            .begin()
            .get(f"{self._keys[db_idx][el_idx]}".encode("ascii"))
        )
        data_object = pickle.loads(datapoint_pickled)
        data_object.id = el_idx #f"{db_idx}_{el_idx}"

 
        for transform in self.transforms:
            data_object = transform(data_object)
        out = {'pos': data_object.pos.numpy().astype(np.float32), 
            'forces': data_object.forces.numpy().astype(np.float32),
            # 'edge_index': data_object.edge_index.numpy(), 
            # 'labels': data_object.labels.numpy(),
            'atomic_numbers': data_object.atomic_numbers.numpy(),
            'molecule_size':data_object.pos.shape[0],
            "idx":idx
            }
        
        energy = data_object.energy.numpy()
        out["pyscf_energy"] = copy.deepcopy(energy.astype(np.float32))  # this is pyscf energy ground truth
        if self.remove_atomref_energy:
            unique,counts = np.unique(out["atomic_numbers"],return_counts=True)
            energy = energy - np.sum(self.atom_reference[unique]*counts)
            energy = energy - self.system_ref
            
        out["energy"] = energy.astype(np.float32) # this is used from model training, mean/ref is removed.
        
        if self.enable_hami:
            # out.update({"init_fock":data_object.init_fock.numpy().astype(np.float32)})
            if self.remove_init:
                data_object.fock = data_object.fock - data_object.init_fock
            if self.Htoblock_otf == True:
                out.update({"buildblock_mask":self.mask,
                            "max_block_size":self.conv.max_block_size,
                            "fock":data_object.fock.numpy().astype(np.float32)
                            })
            else:
                diag,non_diag,diag_mask,non_diag_mask = None,None,None,None
                if (not self.old_blockbuild):
                    diag,non_diag,diag_mask,non_diag_mask = matrixtoblock_lin(data_object.fock.numpy().astype(np.float32),
                                                                            data_object.atomic_numbers.numpy(),
                                                                            self.mask,self.conv.max_block_size)
                else:
                    H = data_object.fock
                    initH = data_object.init_fock
                    Z = data_object.atomic_numbers

                    diag, non_diag, diag_init, non_diag_init, diag_mask, non_diag_mask = split2blocks(
                        matrix_transform(H,Z,self.conv).numpy(),
                        matrix_transform(initH,Z,self.conv).numpy(),
                        Z.numpy(), self.orbitals_ref, self.mask, self.conv.max_block_size)
                out.update({'diag_hamiltonian': diag,
                        'non_diag_hamiltonian': non_diag,
                        'diag_mask': diag_mask,
                        'non_diag_mask': non_diag_mask})
            out.update({"init_fock":data_object.init_fock.numpy().astype(np.float32)})
            # out.update({"s1e":data_object.s1e.numpy().astype(np.float32)})

        return out
    
    # def get_raw_data(self, idx):
    #     out = self.__getitem__(idx)
    #     db_idx = bisect.bisect(self._keylen_cumulative, idx)
    #     # Extract index of element within that db.
    #     el_idx = idx
    #     if db_idx != 0:
    #         el_idx = idx - self._keylen_cumulative[db_idx - 1]
    #     assert el_idx >= 0

    #     # Return features.
    #     datapoint_pickled = (
    #         self.envs[db_idx]
    #         .begin()
    #         .get(f"{self._keys[db_idx][el_idx]}".encode("ascii"))
    #     )
    #     data_object = pickle.loads(datapoint_pickled)
        
    #     if self.remove_init:
    #             data_object.fock = data_object.fock - data_object.init_fock
        
    #     data_object.id = el_idx #f"{db_idx}_{el_idx}"

    #     out.update({})
    
    def connect_db(self, lmdb_path=None):
        env = lmdb.open(
            str(lmdb_path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=32,
        )
        return env
 
    def close_db(self):
        if not self.path.is_file():
            for env in self.envs:
                env.close()
            self.envs = []
        else:
            self.env.close()
            self.env = None


