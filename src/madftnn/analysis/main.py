import argparse
import os

import numpy as np
from ase.io import read

from Simulator import BaseSimulator


def args_register():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', type=str, default='./logs', help='A directory for saving results')
    parser.add_argument('--ckpt-path', type=str, default='/home/hul/v-yunyangli/lightnp_amlt/amlt/flying-skink/HUL_NIPS-2-LSRM-AT_AT/checkpoints/AT_AT_radius3_broadcast/2023_05_12_16_48_56__Visnorm_shared_LSRMNorm2_2branchSerial/best_model/checkpoint-best.pth.tar', help='A directory including well-trained pytorch models')
    parser.add_argument('--atoms-file', type=str, default="./testcases/md22_AT_AT.xyz", help='Atoms file for simulation')
    parser.add_argument('--sim-steps', type=int, default=100, help='Simulation steps for simulation')
    parser.add_argument('--timestep', type=float, default=1, help='TimeStep (fs) for simulation')
    parser.add_argument('--idx', type=int, default=0, help='ith molecule for simulation')
    args = parser.parse_args()
    
    args.log_dir = os.path.abspath(args.log_dir)
    args.ckpt_path = os.path.abspath(args.ckpt_path)
    args.atoms_file = os.path.abspath(args.atoms_file)
    
    return args

if __name__ == "__main__":
    
    args = args_register()
    
    atoms_name = os.path.basename(args.atoms_file)[:-4]

    atoms = read(args.atoms_file)
    
    atoms_info = np.load(f'./sample/{atoms_name}_sample_fs.npz')
    atoms_coords = atoms_info['R'][args.idx]
    # atoms_velocities = atoms_info['V'][args.idx]
    atoms.set_positions(atoms_coords)
    # atoms.set_velocities(atoms_velocities)
    
    os.makedirs(f'{args.log_dir}_{atoms_name}_{args.idx}', exist_ok=True)
    
    info = {'model_path': args.ckpt_path, 'device': 'cuda:0'}
    
    simulator = BaseSimulator(atoms=atoms, log_path=f'{args.log_dir}_{atoms_name}_{args.idx}')
    simulator.set_calculator(info)
    
    simulator.simulate(name=atoms_name, simulation_steps=args.sim_steps, time_step=args.timestep, record_per_steps=1)
    