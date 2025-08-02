import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
from pyscf import gto, dft, scf
from cudft.cuda_factory import CUDAFactory
import cudft
from cudft.utils import load_xyz_file
import numpy as np
import torch
import random

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
    parser.add_argument("-b", "--basis", type=str, default="def2-tzvp", 
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
    parser.add_argument("-v", "--verbose", type=int, default=5, 
                        help="Set the verbose level.")
    parser.add_argument("-p", type=int, default=0)
    parser.add_argument("--total_partion", type=int, default=1)
    parser.add_argument("--save_all", type=int, default=1)
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    
    return parser

def parse_ecp(basis:str):
    ecp_dict = {}
    for atom in gto.ELEMENTS:
        ecp = gto.basis.load_ecp(basis, atom)
        if ecp:
            ecp_dict[atom] = basis
    return ecp_dict
                            
def display_running_info(args: argparse.Namespace):
    print("========================== Config area =========================")
    print(f"Run with module: {cudft}")
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
    if not os.path.exists(input_dir):
        raise ValueError(f"sorry, we can not find the input directionary :{input_dir}")  
    
    
    mols = torch.load(input_dir)
    mol_idxes = list(range(len(mols)))
    random.shuffle(mol_idxes)
    print(f"first 1000 element idx of mol_idxes: {mol_idxes[:1000]}")
    
    for i in mol_idxes:
        #get the name of xyz file
        molecule = mols[i]
        mol_name = os.path.join(output_dir,str(molecule['cid']))
        print(f"Running {molecule['cid']}, i {i}...")
        if os.path.exists(os.path.join(output_dir,mol_name + '_grad.npy')):
            print(f"Skip {mol_name}...")
            continue
        charge = molecule['data']['pubchem']['charge']
        multiplicity = molecule['data']['pubchem']['multiplicity']
        print('charge:', charge, 'multiplicity:', multiplicity)
        atom_pos = molecule['data']['pubchem']['B3LYP@PM6']['atoms']['coords']['3d']
        atom_num = molecule['data']['pubchem']['B3LYP@PM6']['atoms']['elements']['number']
        atom_pos = np.array(atom_pos).reshape(-1, 3)
        np.savez(os.path.join(output_dir, str(molecule['cid']) + '_pos.npy'), atom_pos=atom_pos, atom_num=atom_num)
        # convert to xyz string
        # num2atom = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F'}
        atom = [' '.join([str(atom_num[i]), str(atom_pos[i][0]), str(atom_pos[i][1]), str(atom_pos[i][2])]) for i in range(len(atom_num))]
        atom = '\n'.join(atom)
        # print(atom)
        # charge, multiplicity, atom = load_xyz_file(xyz_file)
        ecp = None if not args.ecp else parse_ecp(args.basis)
        mol = gto.M(atom=atom, charge=charge, spin=multiplicity-1, cart=cart, basis=args.basis, ecp=ecp)
        mol.build()
        if args.verbose >= 5:
            print(f"Number of ao:{mol.nao}.")

        mol.verbose = args.verbose

        # run_type = {"RHF": scf.RHF, "UHF": scf.UHF, "RKS": dft.RKS, "UKS": dft.UKS}
        # scf_type = run_type.get(args.type, None)
        # if not scf_type:
        #     raise ValueError(f"Not supported running type {args.type}!")

        # run_type = {"RHF": scf.RHF, "UHF": scf.UHF, "RKS": dft.RKS, "UKS": dft.UKS}
        # scf_type = run_type.get(args.type, None)
        # if not scf_type:
        #     raise ValueError(f"Not supported running type {args.type}!")
        if mol.spin!=0:
            scf_type = dft.UKS
        else:
            scf_type = dft.RKS
        mf = scf_type(mol,
            xc = args.xc)
    

        mf.max_cycle = args.cycle
        mf.conv_tol = float(args.scf_cutoff)
        mf.direct_scf_tol = float(mf.conv_tol)*0.001
        if args.incore_max_memory > 0:
            mol.incore_anyway = True

        if not args.cpu:
            factory = CUDAFactory()
            params = CUDAFactory.get_default_params(mol)
            if args.task_per_gpu > 0:
                params.eri_tasks_per_gpu = args.task_per_gpu
            # mf.small_rho_cutoff = 1e-30
            params.eri_incore_memory = args.incore_max_memory
            params.enable_qqr = args.qqr
            params.gpu_id_list = args.gpu_list
            params.enable_ERI = not args.no_eri
            params.enable_EXC = not args.no_exc
            if args.gradient:
                params.enable_gradient = True
            mf = factory.generate_cuda_instance(mf, params)

        init_dm = mf.get_init_guess()
        init_h1e = mf.get_hcore()
        init_vhf = mf.get_veff(mf.mol, init_dm)
        init_fock = init_h1e + init_vhf
        np.savez(os.path.join(output_dir, str(molecule['cid']) + '_init.npy'), 
                **{"init_dm": init_dm, "init_h1e": init_h1e, "init_vhf": init_vhf, "init_fock": init_fock})

        output = mf.kernel()

        np.save(os.path.join(output_dir, str(molecule['cid']) + '_energy.npy'), output)
            
        if args.gradient:
            grad_instance = mf.nuc_grad_method()
            grad = grad_instance.kernel()
            print(grad.max(), grad.min(), grad.mean(), grad.std())
            np.save(os.path.join(output_dir, str(molecule['cid']) + '_grad.npy'), grad)
        
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
            scf_result.energy = output
            scf_result.grad = grad if args.gradient else None
            np.savez(os.path.join(output_dir, str(molecule['cid']) + '_scf.npz'), **scf_result)


        print(f"{molecule['cid']}: SCF-converged: {mf.converged}, with energy {mf.e_tot}.")
        if not mf.converged:
            print(f"Warning: {molecule['cid']} not converged!")
        
        if not args.cpu:
            factory.free_resources()
            
# import pickle 
# import lmdb
# from madftnn.utility.pyscf import *
# env = lmdb.open(
#     str("/data/used_data/pubchem_20230831_processed/data.0000.lmdb"),
#     map_size=1099511627776 * 2,
#     subdir=False,
#     meminit=False,
#     map_async=True,
#     max_readers=32,
# )
# length = pickle.loads(
#                 env.begin().get("length".encode("ascii"))
#             )

# import tqdm
# for idx in tqdm.tqdm(range(length)):
#     datapoint_pickled = (
#                 env
#                 .begin()
#             .get(f"{idx}".encode("ascii"))
#             )
#     data_object = pickle.loads(datapoint_pickled)
#     pos = data_object.pos.numpy()
#     atomic_numbers = data_object.atomic_numbers.numpy()

#     mol, mf = get_pyscf_obj_from_dataset(pos,atomic_numbers.reshape(-1), basis="def2-tzvp", xc="b3lyp5",gpu=False)
#     s1e = mf.get_ovlp(mol)
#     data_object["s1e"] = torch.from_numpy(s1e)*HATREE_TO_KCAL

#     txn = env.begin(write=True)
#     txn.put(
#         f"{idx}".encode("ascii"),
#         pickle.dumps(data_object, protocol=-1),
#     )
#     txn.commit()