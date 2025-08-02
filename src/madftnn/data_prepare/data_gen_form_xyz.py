import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
from pyscf import gto, dft, scf
from madft.cuda_factory import CUDAFactory
import madft
from torch_geometric.data import Data
# from madft.utils import load_xyz_file
import numpy as np
import torch
import lmdb  
import pickle  

global __ATOM_LIST__
__ATOM_LIST__ = \
    ['h',  'he',
     'li', 'be', 'b',  'c',  'n',  'o',  'f',  'ne',
     'na', 'mg', 'al', 'si', 'p',  's',  'cl', 'ar',
     'k',  'ca', 'sc', 'ti', 'v ', 'cr', 'mn', 'fe', 'co', 'ni', 'cu',
     'zn', 'ga', 'ge', 'as', 'se', 'br', 'kr',
     'rb', 'sr', 'y',  'zr', 'nb', 'mo', 'tc', 'ru', 'rh', 'pd', 'ag',
     'cd', 'in', 'sn', 'sb', 'te', 'i',  'xe',
     'cs', 'ba', 'la', 'ce', 'pr', 'nd', 'pm', 'sm', 'eu', 'gd', 'tb', 'dy',
     'ho', 'er', 'tm', 'yb', 'lu', 'hf', 'ta', 'w',  're', 'os', 'ir', 'pt',
     'au', 'hg', 'tl', 'pb', 'bi', 'po', 'at', 'rn',
     'fr', 'ra', 'ac', 'th', 'pa', 'u',  'np', 'pu']

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
    parser.add_argument("-s", "--spheric", action="store_true", default=True, 
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
    parser.add_argument("--cycle", type=int, default=10, 
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
    parser.add_argument("--input_dir", type=str, default="local_files/pubchem_1_200")
    parser.add_argument("--output_dir", type=str, default="local_files/waters")
    
    return parser

def parse_ecp(basis:str):
    ecp_dict = {}
    for atom in gto.ELEMENTS:
        ecp = gto.basis.load_ecp(basis, atom)
        if ecp:
            ecp_dict[atom] = basis
    return ecp_dict

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

def load_xyz_file(filename: str):
    """load molecule information from the xyz file. Note that the xyz file should be in the format: \n
        n \n
        charge multiplicity \n
        atom_1 x y z \n
        atom_2 x y z \n
        ...          \n
        atom_n x y z \n

    Args:
        filename (str): The path to the xyz file.

    Returns:
        Tuple[int, int, str]: The `charge`, `multiplicity` and `coordinates` of the mol.
    """
    with open(filename, "r") as f:
        atm = int(f.readline().strip())
        info = []
        try:
            for token in f.readline().split():
                try:
                    info.append(int(token))
                except:
                    pass
            charge, multiplicity = info[0], info[1]
        except Exception:
            charge, multiplicity = 0, 1
        atom_loc = "".join(f.readlines())

    return charge, multiplicity, atom_loc

def int_atom(atom):
    """
    convert str atom to integer atom
    """
    global __ATOM_LIST__
    #print(atom)
    atom = atom.lower()
    return __ATOM_LIST__.index(atom) + 1

def atom_to_num_and_pos(atom):
    """
    """

    atomic_symbols = []
    xyz_coordinates = []
    atom_list = atom.split('\n')
    if len(atom.split('\n')[-1])<= 10:
        atom_list = atom_list[:-1]

    for line_number, line in enumerate(atom_list):
        atomic_symbol, x, y, z = line.split()
        atomic_symbols.append(atomic_symbol)
        xyz_coordinates.append([float(x), float(y), float(z)])

    if atomic_symbols[0].isdigit():
        atoms = atomic_symbols
    else:
        atoms = [int_atom(atom) for atom in atomic_symbols]

    return atoms, xyz_coordinates

if "__main__" == __name__:
    parser = add_parser()
    print(parser)
    args = parser.parse_args()
    print(args)
    ecp = None if not args.ecp else parse_ecp(args.basis)
    cart = True if not args.spheric else False
    input_dir = args.input_dir
    output_dir = args.output_dir

        
    # mol_dir = os.path.join(input_dir, 'dataset', 'June_1_data')
    # grep all xyz file
    print('Loading data... (This may take a while.)')
    if not os.path.exists(input_dir):
        raise ValueError(f"sorry, we can not find the input directionary :{input_dir}")  
    
    
    mol_idxes = os.listdir(input_dir)
    mol_idxes.sort()
    # random.shuffle(mol_idxes)
    # print(f"first 1000 element idx of mol_idxes: {mol_idxes[:1000]}")
    ori_path = "/home/weixinran/MADFT-NN/local_files/"
    
    for idx, xyz_file in enumerate(mol_idxes):
        #get the name of xyz file
        charge, multiplicity, atom = load_xyz_file(input_dir + "/" + xyz_file)
        atomic_numbers, pos = atom_to_num_and_pos(atom)
        print(len(atomic_numbers))
        # if len(atomic_numbers) not in [189,195]:
        #     continue
        db_dir = os.path.join(ori_path, "pubchem_tzvp_{}".format(len(atomic_numbers)))
        if not os.path.exists(db_dir):
            os.mkdir(db_dir)
        db_path = os.path.join(ori_path, "pubchem_tzvp_{}/data.lmdb".format(len(atomic_numbers)))
        if os.path.exists(db_path):
            # os.remove(db_path)
            continue
        db = lmdb.open(db_path, 
                    map_size=2**39,
                    subdir=False,
                    meminit=False,
                    map_async=True,)  

        ecp = None if not args.ecp else parse_ecp(args.basis)
        mol = gto.M(atom=atom, charge=charge, spin=multiplicity-1, cart=cart, basis=args.basis, ecp=ecp)
        mol.build()
        if args.verbose >= 5:
            print(f"Number of atom:{len(atomic_numbers)},ao:{mol.nao}.")

        mol.verbose = args.verbose

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
            params.gpu_id_list = [0,1,2,3]
            params.enable_ERI = not args.no_eri
            params.enable_EXC = not args.no_exc
            if args.gradient:
                params.enable_gradient = True
            mf = factory.generate_cuda_instance(mf, params)

        init_dm = mf.get_init_guess()
        init_h1e = mf.get_hcore()
        init_vhf = mf.get_veff(mf.mol, init_dm)
        init_fock = init_h1e + init_vhf
        # np.savez(os.path.join(output_dir, str(molecule['cid']) + '_init.npy'), 
        #         **{"init_dm": init_dm, "init_h1e": init_h1e, "init_vhf": init_vhf, "init_fock": init_fock})

        try:
            output = mf.kernel()
        except:
            continue

        # np.save(os.path.join(output_dir, str(molecule['cid']) + '_energy.npy'), output)
            
        if args.gradient:
            grad_instance = mf.nuc_grad_method()
            grad = grad_instance.kernel()
            print(grad.max(), grad.min(), grad.mean(), grad.std())
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
            scf_result.energy = output
            scf_result.grad = grad if args.gradient else None
            # np.savez(os.path.join(output_dir, str(molecule['cid']) + '_scf.npz'), **scf_result)
        
        data = Data(file_name='', atomic_numbers=torch.tensor(atomic_numbers), energy=torch.tensor(output), forces=torch.zeros((len(atomic_numbers),3)), 
                    init_fock=torch.tensor(init_fock), fock=torch.tensor(vhf), num_nodes=torch.tensor(len(atomic_numbers)), \
                    pos=torch.tensor(pos), edge_index=None, labels=None, num_labels=None,)
                            #   grouping_graph=[2, 937], interaction_graph=[2, 243])

 
        txn = db.begin(write=True)
        txn.put(
            f"{0}".encode("ascii"),
            pickle.dumps(data, protocol=-1),
        )
        txn.commit()

        txn = db.begin(write=True)
        txn.put("length".encode("ascii"), pickle.dumps(1, protocol=-1))
        txn.commit()

        if not args.cpu:
            factory.free_resources()