"""
Convert data from the 'schnorb_hamiltonian' format into a format compatible with the 'HamiltonianDatabase_qhnet' class. 
This script is adapted from https://github.com/divelab/AIRS/blob/main/OpenDFT/QHNet/ori_dataset.py

Usage:
export PYTHONPATH=$PYTHONPATH:./src
python src/madftnn/data_prepare/convert_phisnet.py ~/project/PhiSNet/schnorb_hamiltonian_water.db local_files/water.db
python src/madftnn/data_prepare/convert_phisnet.py ~/project/PhiSNet/schnorb_hamiltonian_ethanol_dft.db local_files/ethanol.db
python src/madftnn/data_prepare/convert_phisnet.py ~/project/PhiSNet/schnorb_hamiltonian_malondialdehyde.db local_files/malondialdehyde.db
python src/madftnn/data_prepare/convert_phisnet.py ~/project/PhiSNet/schnorb_hamiltonian_uracil.db local_files/uracil.db

"""
import sys
from ase.db import connect
from argparse import Namespace
import numpy as np
import torch
import apsw
from torch_geometric.data import Data
from madftnn.dataset.sqlite_database.hamiltonian_database_qhnet import HamiltonianDatabase_qhnet
from tqdm import tqdm

chemical_symbols = ["n", "H", "He" ,"Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", 
            "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga",
            "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",
            "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", 
            "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", 
            "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", 
            "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", 
            "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"]
#orbital reference (for def2svp basis)
orbitals_ref = {}
orbitals_ref[1] = np.array([0,0,1])       #H: 2s 1p
orbitals_ref[6] = np.array([0,0,0,1,1,2]) #C: 3s 2p 1d
orbitals_ref[7] = np.array([0,0,0,1,1,2]) #N: 3s 2p 1d
orbitals_ref[8] = np.array([0,0,0,1,1,2]) #O: 3s 2p 1d
orbitals_ref[9] = np.array([0,0,0,1,1,2]) #F: 3s 2p 1d

def cut_matrix(matrix, atoms, orbital_mask):
        all_diagonal_matrix_blocks = []
        all_non_diagonal_matrix_blocks = []
        all_diagonal_matrix_block_masks = []
        all_non_diagonal_matrix_block_masks = []
        col_idx = 0
        for idx_i, atom_i in enumerate(atoms): # (src)
            row_idx = 0
            atom_i = atom_i.item()
            mask_i = orbital_mask[atom_i]
            for idx_j, atom_j in enumerate(atoms): # (dst)
                atom_j = atom_j.item()
                mask_j = orbital_mask[atom_j]
                matrix_block = torch.zeros(14, 14).type(torch.float64)
                matrix_block_mask = torch.zeros(14, 14).type(torch.float64)
                extracted_matrix = \
                    matrix[row_idx: row_idx + len(mask_j), col_idx: col_idx + len(mask_i)]

                # for matrix_block
                tmp = matrix_block[mask_j]
                tmp[:, mask_i] = extracted_matrix
                matrix_block[mask_j] = tmp

                tmp = matrix_block_mask[mask_j]
                tmp[:, mask_i] = 1
                matrix_block_mask[mask_j] = tmp

                if idx_i == idx_j:
                    all_diagonal_matrix_blocks.append(matrix_block)
                    all_diagonal_matrix_block_masks.append(matrix_block_mask)
                else:
                    all_non_diagonal_matrix_blocks.append(matrix_block)
                    all_non_diagonal_matrix_block_masks.append(matrix_block_mask)
                row_idx = row_idx + len(mask_j)
            col_idx = col_idx + len(mask_i)
        return torch.stack(all_diagonal_matrix_blocks, dim=0), \
               torch.stack(all_non_diagonal_matrix_blocks, dim=0),\
               torch.stack(all_diagonal_matrix_block_masks, dim=0), \
               torch.stack(all_non_diagonal_matrix_block_masks, dim=0)

def hamiltonian_transform(hamiltonian, atoms):
    conv = Namespace(
        atom_to_orbitals_map={'H': 'ssp', 'O': 'sssppd', 'C': 'sssppd', 'N': 'sssppd'},
        orbital_idx_map={'s': [0], 'p': [2, 0, 1], 'd': [4, 2, 0, 1, 3]},
        orbital_sign_map={'s': [1], 'p': [1, 1, 1], 'd': [1, 1, 1, 1, 1]},
        orbital_order_map={'H': [0, 1, 2], 'O': [0, 1, 2, 3, 4, 5], 'C': [0, 1, 2, 3, 4, 5], 'N': [0, 1, 2, 3, 4, 5]},
    )

    orbitals = ''
    orbitals_order = []
    for a in atoms:
        offset = len(orbitals_order)
        orbitals += conv.atom_to_orbitals_map[a]
        orbitals_order += [idx + offset for idx in conv.orbital_order_map[a]]

    transform_indices = []
    transform_signs = []
    for orb in orbitals:
        offset = sum(map(len, transform_indices))
        map_idx = conv.orbital_idx_map[orb]
        map_sign = conv.orbital_sign_map[orb]
        transform_indices.append(np.array(map_idx) + offset)
        transform_signs.append(np.array(map_sign))

    transform_indices = [transform_indices[idx] for idx in orbitals_order]
    transform_signs = [transform_signs[idx] for idx in orbitals_order]
    transform_indices = np.concatenate(transform_indices).astype(np.int64)
    transform_signs = np.concatenate(transform_signs)

    hamiltonian_new = hamiltonian[...,transform_indices, :]
    hamiltonian_new = hamiltonian_new[...,:, transform_indices]
    hamiltonian_new = hamiltonian_new * transform_signs[:, None]
    hamiltonian_new = hamiltonian_new * transform_signs[None, :]
    return hamiltonian_new

def get_mol(row):
    pos = torch.tensor(row['positions'] * 1.8897261258369282, dtype=torch.float64)
    atoms = torch.tensor(row['numbers'], dtype=torch.int64).view(-1, 1)
    energy = torch.tensor(row.data['energy'], dtype=torch.float64)
    force = torch.tensor(row.data['forces'], dtype=torch.float64)
    atom_types = ''.join([chemical_symbols[i] for i in atoms])
    hamiltonian = torch.tensor(hamiltonian_transform(
        row.data['hamiltonian'], atom_types), dtype=torch.float64)
    overlap = torch.tensor(hamiltonian_transform(
        row.data['overlap'], atom_types), dtype=torch.float64)

    hamiltonian_diagonal_blocks, hamiltonian_non_diagonal_blocks, \
    hamiltonian_diagonal_block_masks, hamiltonian_non_diagonal_block_masks = \
        cut_matrix(hamiltonian, atoms, orbital_mask)
    overlap_diagonal_blocks, overlap_non_diagonal_blocks, \
    overlap_diagonal_block_masks, overlap_non_diagonal_block_masks  = \
        cut_matrix(overlap, atoms, orbital_mask)
    data = Data(pos=pos,
                atoms=atoms,
                energy=energy,
                force=force,
                hamiltonian_diagonal_blocks=hamiltonian_diagonal_blocks,
                hamiltonian_non_diagonal_blocks=hamiltonian_non_diagonal_blocks,
                hamiltonian_diagonal_block_masks=hamiltonian_diagonal_block_masks,
                hamiltonian_non_diagonal_block_masks=hamiltonian_non_diagonal_block_masks,
                overlap_diagonal_blocks=overlap_diagonal_blocks,
                overlap_non_diagonal_blocks=overlap_non_diagonal_blocks,
                overlap_diagonal_block_masks=overlap_diagonal_block_masks,
                overlap_non_diagonal_block_masks=overlap_non_diagonal_block_masks)
    return data



if __name__ == "__main__":
    orbital_mask = {}
    idx_1s_2s = torch.tensor([0, 1])
    idx_2p = torch.tensor([3, 4, 5])
    orbital_mask_line1 = torch.cat([idx_1s_2s, idx_2p])
    orbital_mask_line2 = torch.arange(14)
    for i in range(1, 11):
        orbital_mask[i] = orbital_mask_line1 if i <= 2 else orbital_mask_line2

    input_db = sys.argv[1]
    output_db = sys.argv[2]
    
    database = HamiltonianDatabase_qhnet(output_db, flags=(apsw.SQLITE_OPEN_READWRITE | apsw.SQLITE_OPEN_CREATE))
    cursor = database._get_connection().cursor()
    if 'water' in input_db:
        Zref = np.array([8, 1, 1], dtype='int32')
    elif 'ethanol' in input_db:
        Zref = np.array([6, 6, 8, 1, 1, 1, 1, 1, 1], dtype='int32')
    elif 'malondialdehyde' in input_db:
        Zref = np.array([6, 6, 6, 8, 8, 1, 1, 1, 1], dtype='int32')
    elif 'uracil' in input_db:
        Zref = np.array([6, 6, 7, 6, 7, 6, 8, 8, 1, 1, 1, 1], dtype='int32')
    elif 'aspirin' in input_db:
        Zref = np.array([6, 6, 6, 6, 6, 6, 6, 8, 8, 8, 6, 6, 8,
                     1, 1, 1, 1, 1, 1, 1, 1], dtype='int32')
    atom_types = ''.join([chemical_symbols[i] for i in Zref])
    database.add_Z(Z=Zref)
    for Z in database.Z:
        database.add_orbitals(Z, orbitals_ref[Z])

    db = connect(input_db)
    cursor.execute('''BEGIN''') #begin transaction
    DEFAULT = np.array(0)
    for row in tqdm(db.select()):
        data = get_mol(row)
        database.add_data(
            R=data['pos'].numpy(), E=data['energy'].numpy(), F=data['force'].numpy(), 
            diag=data['hamiltonian_diagonal_blocks'].numpy(), 
            non_diag=data['hamiltonian_non_diagonal_blocks'].numpy(), 
            diag_init=DEFAULT, 
            non_diag_init=DEFAULT, 
            diag_mask=data['hamiltonian_diagonal_block_masks'].numpy(), 
            non_diag_mask=data['hamiltonian_non_diagonal_block_masks'].numpy(), 
            A=DEFAULT, G=DEFAULT, 
            L=database.Z, 
            transaction=False
        )
    cursor.execute('''COMMIT''') #commit transaction