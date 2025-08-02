import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
from pyscf import gto, dft, scf
# from madft.cuda_factory import CUDAFactory
# import cudft
# from cudft.utils import load_xyz_file
import numpy as np
import torch
import random
from torch_geometric.data import Data  
import sys
sys.path.append('/home/weixinran/erpailuo/se_3_equivariant_prediction/src/madftnn')
from dataset.dataset_unified import HamiltonianDataset_qhnet_clean,HamiltonianDatabase_qhnet
from utility.pyscf import get_pyscf_obj_from_dataset
from dataset.buildblock import get_conv_variable_lin,block2matrix

import lmdb  
import pickle  
from tqdm import tqdm

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def add_parser():
    cur_path = os.path.dirname(__file__)
    parser = argparse.ArgumentParser(prog="data_generation.py",
                                     description="Test CUDA DFT with pyscf interface.")
    parser.add_argument("-c", "--cpu", action="store_true", default=False, 
                        help="Use CPU to run the test.")       
    parser.add_argument("-l", "--gpu-list", nargs="+", type=int, default=[0], 
                        help="List of GPU to run the test.")
    parser.add_argument("-s", "--spheric", action="store_true", default=False, 
                        help="Use spherical coordinate instead of Cartesian coordinate.")   
    parser.add_argument("-e", "--elec-energy", action="store_true", default=False, 
                        help="Only calculate the electron energy instead run whole SCF.")
    # parser.add_argument("-t", "--type", type=str, default="RKS", 
    #                     help="The SCF type of running the test, should be one of [RHF, UHF, RKS, UKS].")
    parser.add_argument("-b", "--basis", type=str, default="def2-svp", 
                        help="The basis function used to run the calculation.")
    parser.add_argument("-x", "--xc", type=str, default="B3LYP", 
                        help="The functional to used with RKS/UKS, should be compatible with pyscf.")
    parser.add_argument("-g", "--gradient", action="store_true", default=False, 
                        help="Whether to calculate the gradient.")
    parser.add_argument("-i", "--incore-max-memory", type=int, default=0, 
                        help="Set incore max memory.")
    parser.add_argument("--cycle", type=int, default=200, 
                        help="The max cycle of iteration.")
    parser.add_argument("--no-drop", action="store_true", default=False, 
                        help="Turn off grids drop in EXC.")
    parser.add_argument("--qqr", action="store_true", default=False, 
                        help="Turn on QQR in ERI.")
    parser.add_argument("--ecp", action="store_true", default=False, 
                        help="Turn on ECP.")
    parser.add_argument("--no-exc", action="store_true", default=False, 
                        help="Turn off CUDA EXC.")
    parser.add_argument("--no-eri", action="store_true", default=False, 
                        help="Turn off CUDA ERI.")
    parser.add_argument("--level", type=int, default=2, 
                        help="Set the grids level.")
    parser.add_argument("--task-per-gpu", type=int, default=4, 
                        help="Set the number of task per GPU.")
    parser.add_argument("--scf-cutoff", type=str, default="1e-8", 
                        help="Set the direct scf cutoff.")
    parser.add_argument("-v", "--verbose", type=int, default=1, 
                        help="Set the verbose level.")
    parser.add_argument("-p", type=int, default=0)
    parser.add_argument("--total_partion", type=int, default=1)
    parser.add_argument("--save_all", type=int, default=0)
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    
    return parser

def build_final_matrix(molecule_size, atomic_numbers, diag_hamiltonian, non_diag_hamiltonian, basis, sym=True):
    atom_start = 0
    atom_pair_start = 0
    rebuildfocks = []
    conv,_,mask_lin,_ = get_conv_variable_lin(basis)
    for idx,n_atom in enumerate([molecule_size]):
        # n_atom = n_atom.item()
        Z = atomic_numbers[atom_start:atom_start+n_atom]
        diag = diag_hamiltonian[atom_start:atom_start+n_atom]
        if sym:
            non_diag = non_diag_hamiltonian[atom_pair_start:atom_pair_start+n_atom*(n_atom-1)//2]
        else:
            non_diag = non_diag_hamiltonian[atom_pair_start:atom_pair_start+n_atom*(n_atom-1)]
        # diag = batch_data["diag_hamiltonian"][atom_start:atom_start+n_atom]
        # non_diag = batch_data["non_diag_hamiltonian"][atom_pair_start:atom_pair_start+n_atom*(n_atom-1)//2]

        atom_start += n_atom
        atom_pair_start += n_atom*(n_atom-1)//2
        
        rebuildfock = block2matrix(Z,diag,non_diag,mask_lin,conv.max_block_size, sym=sym)
        rebuildfocks.append(rebuildfock)
    # batch_data["pred_hamiltonian"] = rebuildfocks
    return rebuildfocks

def fock_to_dm(mf: scf.RHF, fock: np.ndarray, s1e: np.ndarray = None):
    if s1e is None:
        s1e = mf.get_ovlp()
    mo_energy, mo_coeff = mf.eig(fock, s1e)
    mo_occ = mf.get_occ(mo_energy, mo_coeff)
    dm = mf.make_rdm1(mo_coeff, mo_occ)
    return dm

def get_energy_from_h(mf: scf.RHF, h: np.ndarray):
    """
    Calculates the energy for a given mean-field object and Hamiltonian matrix.

    Args:
        mf (scf.RHF): The mean-field object.
        h (np.ndarray): The Hamiltonian matrix.

    Returns:
        float: The energy.
    """
    dm = fock_to_dm(mf, h)
    e_tot = mf.energy_tot(dm=dm)
    return e_tot

def parse_ecp(basis:str):
    ecp_dict = {}
    for atom in gto.ELEMENTS:
        ecp = gto.basis.load_ecp(basis, atom)
        if ecp:
            ecp_dict[atom] = basis
    return ecp_dict
                            
def display_running_info(args: argparse.Namespace):
    print("========================== Config area =========================")
    # print(f"Run with module: {cudft}")
    for k, v in args._get_kwargs():
        print(f"Args: {k:<15}: {str(v):<10}")
    print("=============================== End ============================")


def slice_data_with_maxelement():
    for name in [
                "b3lyp_pm6_chon500nosalt_9244073_18488146",
                "b3lyp_pm6_chon500nosalt_18488146_27732219",
                "b3lyp_pm6_chon500nosalt_27732219_36976292",
                "b3lyp_pm6_chon500nosalt_36976292_46220365"]:
        print(name)
        mols = torch.load(f"/home/hul/teamdrive/dataset/filtered_data/{name}.pth")
        for i in range(len(mols)//200000+1):
            print(i)
            torch.save(mols[i*200000:(i+1)*200000],f"/home/hul/teamdrive/dataset/filtered_data/{name}_split200000_{i}.pth")


if "__main__" == __name__:
    parser = add_parser()
    print(parser)
    args = parser.parse_args()
    print(args)
    display_running_info(args)
    ecp = None if not args.ecp else parse_ecp(args.basis)
    cart = True if not args.spheric else False
    input_dir = args.input_dir
    output_dir = args.output_dir

        
    # mol_dir = os.path.join(input_dir, 'dataset', 'June_1_data')
    # grep all xyz file
    print('Loading data... (This may take a while.)')
    # if not os.path.exists(input_dir):
    #     raise ValueError(f"sorry, we can not find the input directionary :{input_dir}")  
    
    mols = HamiltonianDatabase_qhnet('/home/weixinran/erpailuo/se_3_equivariant_prediction/used_data/QH9_new.db')
    # mols = torch.load(input_dir)
    mol_idxes = list(range(len(mols)))
    # random.shuffle(mol_idxes)
    # print(f"first 1000 element idx of mol_idxes: {mol_idxes[:1000]}")
    
    data_list = []
    for i in tqdm(mol_idxes):
        #get the name of xyz file
        R, E, F, diag, non_diag, diag_init, non_diag_init, diag_mask, non_diag_mask, A, G, L = mols[i]
        atom_pos = R
        atom_num = L
        atom_pos = np.array(atom_pos).reshape(-1, 3)

        atom = [' '.join([str(atom_num[i]), str(atom_pos[i][0]), str(atom_pos[i][1]), str(atom_pos[i][2])]) for i in range(len(atom_num))]
        atom = '\n'.join(atom)
        # print(atom)
        # charge, multiplicity, atom = load_xyz_file(xyz_file)
        ecp = None if not args.ecp else parse_ecp(args.basis)
        mol = gto.M(atom=atom, cart=False, basis=args.basis, ecp=ecp)
        mol.build()
        if args.verbose >= 5:
            print(f"Number of ao:{mol.nao}.")

        mol.verbose = args.verbose

        if mol.spin!=0:
            scf_type = dft.UKS
        else:
            scf_type = dft.RKS
        mf = scf_type(mol,
            xc = args.xc)
    
        mf.grids.level=3
        mf.direct_scf_tol=1e-13
        mf.conv_tol_grad=3.16e-5
        # mf.conv_tol=1e-5

        mf.max_cycle = args.cycle
        # mf.conv_tol = float(args.scf_cutoff)
        # mf.direct_scf_tol = float(mf.conv_tol)*0.001
        if args.incore_max_memory > 0:
            mol.incore_anyway = True

        # if not args.cpu:
        #     factory = CUDAFactory()
        #     params = CUDAFactory.get_default_params(mol)
        #     if args.task_per_gpu > 0:
        #         params.eri_tasks_per_gpu = args.task_per_gpu
        #     # mf.small_rho_cutoff = 1e-30
        #     params.eri_incore_memory = args.incore_max_memory
        #     params.enable_qqr = args.qqr
        #     params.gpu_id_list = args.gpu_list
        #     params.enable_ERI = not args.no_eri
        #     params.enable_EXC = not args.no_exc
        #     if args.gradient:
        #         params.enable_gradient = True
        #     mf = factory.generate_cuda_instance(mf, params)

        init_fock = build_final_matrix(F.shape[0], L, diag_init, non_diag_init, 'def2-svp', sym=True)[0]
        fock = build_final_matrix(F.shape[0], L, diag, non_diag, 'def2-svp', sym=True)[0]
        
        # output = mf.kernel()
        energy = get_energy_from_h(mf, fock)

        # np.save(os.path.join(output_dir, str(molecule['cid']) + '_energy.npy'), output)
            
        if args.gradient:
            grad_instance = mf.nuc_grad_method()
            grad = grad_instance.kernel()
            # print(grad.max(), grad.min(), grad.mean(), grad.std())
            # np.save(os.path.join(output_dir, str(molecule['cid']) + '_grad.npy'), grad)

        
        if args.save_all:
                # Save the SCF results
            dm = mf.make_rdm1()
            h1e = mf.get_hcore()
            vhf = mf.get_veff(mf.mol, dm)
            s1e = mf.get_ovlp(mol)
            scf_result = dotdict()
            scf_result.s1e = s1e
            scf_result.mo_energy = mf.mo_energy
            scf_result.mo_occ = mf.mo_occ
            scf_result.mo_coeff = mf.mo_coeff
            scf_result.e_tot = mf.e_tot
            scf_result.dm = dm
            scf_result.converged = mf.converged
            # fock matrix related
            scf_result.fock = h1e + vhf
            scf_result.h1e = h1e
            scf_result.vhf = vhf
            scf_result.vxc = vhf - vhf.vj
            scf_result.fock_woVxc = h1e + vhf - scf_result["vxc"]
            # Save all energy results
            scf_result.nuc = mf.scf_summary["nuc"]
            scf_result.e1 = mf.scf_summary["e1"]
            scf_result.coul = mf.scf_summary["coul"]
            scf_result.exc = mf.scf_summary["exc"]
            scf_result.non_exc = scf_result.nuc + scf_result.e1 + scf_result.coul
            scf_result.energy = energy
            scf_result.grad = grad if args.gradient else None
            # np.savez(os.path.join(output_dir, str(molecule['cid']) + '_scf.npz'), **scf_result)

        data = Data(file_name='', atomic_numbers=atom_num, pyscf_energy=energy, forces=None, init_fock=init_fock, fock=fock, num_nodes=atom_num.shape[0], \
                    pos=atom_pos, edge_index=A, labels=G, num_labels=np.unique(G).shape[0],)
                            #   grouping_graph=[2, 937], interaction_graph=[2, 243])

        data_list.append(data)

        # print(f"{molecule['cid']}: SCF-converged: {mf.converged}, with energy {mf.e_tot}.")
        # if not mf.converged:
        #     print(f"Warning: not converged!")
        
        # if not args.cpu:
        #     factory.free_resources()

    # 创建LMDB环境  
    env = lmdb.open('/home/weixinran/erpailuo/se_3_equivariant_prediction/used_data/QH9', map_size=1e12)  
    
    # 开始写入数据  
    with env.begin(write=True) as txn:  
        for i, data_object in enumerate(data_list):  
            # 将Data对象序列化为字节串  
            key = f'data_{i}'.encode('utf-8')  # 创建唯一的键  
            value = pickle.dumps(data_object)   # 序列化Data对象  
            txn.put(key, value)                 # 写入键值对  
    
    # 关闭LMDB环境  
    env.close()  