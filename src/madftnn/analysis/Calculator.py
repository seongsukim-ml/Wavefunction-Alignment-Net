from typing import List

import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.units import kcal, mol
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage
from torch_geometric.data import Data
from torch_geometric.transforms import RadiusGraph

import sys
sys.path.append("..")
from lightnp.LSTM.data import collate_fn
from lightnp.LSTM.models.lsrm_modules import Visnorm_shared_LSRMNorm2_2branchSerial
from lightnp.LSTM.models.torchmdnet.models.model import create_model
from lightnp.LSTM.utils.transforms import convert_to_neighbor, reconstruct_group_with_threshold
from lightnp.LSTM.utils.neighborhood_expansion import build_neighborhood_n_interaction
from lightnp.LSTM.utils.build_group_graph import build_grouping_graph


class LSRMCalculator(Calculator):
    r"""
        Calculate the energy and forces of the system using deep learning model, i.e., ViSNet.
        
        Parameters:
        -----------
            model: 
                Deep learning model.
            device: cpu | cuda
                Device to use for calculation.
            properties: List[str]
                Targets of the calculation.
            label: str
                Label of the calculator to save the median results.
            step_save_path: str
                Path to save the intermediate results.
            dipix: str
                Name and index of the dipeptide to transfer to the monitor.
    """
    implemented_properties = ['energy', 'forces']

    def __init__(self, path, device='cpu', label='lsrmcalc', properties=['energy', 'forces'], **kwargs):
        super().__init__(**kwargs)
        self.model, self.ema = LSRMCalculator.load_from_path(path)
        self.model.to(device)
        self.model.eval()
        self.mean = -1154896.7500
        self.std = 10.8456
        self.device = device
        self.properties = properties
        self.set_label(label)
        
    @staticmethod
    def atoms2loader(ase_atoms):
        '''
        Convert ase object to torch_geometric.data.Data object
        '''
        data = Data()   
        data.atomic_numbers = torch.Tensor(ase_atoms.get_atomic_numbers()).long().unsqueeze(-1)
        data.num_nodes = data.atomic_numbers.shape[0]
        data.pos = torch.Tensor(torch.Tensor(ase_atoms.get_positions()).float())
        neighbor_finder = RadiusGraph(r = 4.0)
        data = neighbor_finder(data)
        data.labels = torch.Tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3,
            2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 4, 4, 5, 5, 6, 6, 6,
            6, 6, 6, 6, 6, 7, 6, 6, 6, 7, 7, 7]).long()
        data.num_labels = 8
        build_grouping_graph(data)
        build_neighborhood_n_interaction(data)
        data = convert_to_neighbor(r = 4.0)(data) 
        data = reconstruct_group_with_threshold()(data)
        return DataLoader([data], batch_size=1, shuffle=False, collate_fn=collate_fn(unit = 1, with_force=False, with_energy = False))
    
    def dl_potential(self, atoms: Atoms):
        
        data_batch = LSRMCalculator.atoms2loader(atoms)
        data = next(iter(data_batch))
        with self.ema.average_parameters():
            data.to(self.device)
            results = self.model(data)
        energy = results['energy'].detach().cpu().numpy() * self.std + self.mean
        forces = results['forces'].detach().cpu().numpy() * self.std    
        
        return energy, forces
    
    def calculate(
        self, 
        atoms=None, 
        properties=['energy', 'forces'],
        system_changes=all_changes
    ):

        if getattr(self, "properties", None) is None:
            self.properties = properties

        Calculator.calculate(self, atoms, properties, system_changes)
        
        energy, forces = self.dl_potential(atoms)

        self.results = {
            'energy': energy * (kcal / mol),
        }

        if 'forces' in self.properties:
            self.results['forces'] = forces * (kcal / mol)
    
    @staticmethod
    def get_model(config, mean, std, regress_forces, atomref):
        if config['model'] in ["TorchMD_Norm", "TorchMD_ET", "PaiNN"]:
            model = create_model(config, mean=mean, std=std, atomref=atomref)
        elif config["model"].startswith("TorchMD_NeurIPS_LSRM") or config["model"].startswith("TorchMD_NeurIPS_PointTransformer_LSRM") or \
            config["model"].startswith("Visnorm"):
            model = eval(config["model"])(regress_forces = regress_forces,
                    hidden_channels=config["hidden_channels"],
                    num_layers=config['num_interactions'],
                    num_rbf=50,
                    rbf_type="expnorm",
                    trainable_rbf=False,
                    activation="silu",
                    attn_activation="silu",
                    neighbor_embedding=True,
                    num_heads=8,
                    distance_influence="both",
                    short_cutoff_lower=config["short_cutoff_lower"],
                    short_cutoff_upper=config["short_cutoff_upper"], ###10
                    long_cutoff_lower=config["long_cutoff_lower"],
                    long_cutoff_upper=config["long_cutoff_upper"],
                    mean = mean,
                    std = std,
                    atom_ref = atomref,
                    max_z=100,
                    max_num_neighbors=32,
                    group_center='center_of_mass',
                    tf_writer = None,
                    config=config
                )
        else:
            raise NotImplementedError
        return model
    
    @staticmethod
    def load_from_path(path):
        lsrm_args = {
            'model': 'Visnorm_shared_LSRMNorm2_2branchSerial', 
            'batch_size': 32, 
            'num_interactions': 6, 
            'long_num_layers': 2, 
            'adaptive_cutoff': False, 
            'short_cutoff_lower': 0.0, 
            'short_cutoff_upper': 4.0, 
            'long_cutoff_lower': 0.0, 
            'long_cutoff_upper': 9.0, 
            'otfcutoff': 4.0, 
            'group_center': 'center_of_mass', 
            'hidden_channels': 128,
            'otf_graph': True, 
            'not_otf_graph': False, 
            'no_broadcast': False, 
            'dropout': 0.0, 
            'ema_decay': 0.999
        }
        model = LSRMCalculator.get_model(lsrm_args, torch.Tensor([0]), torch.Tensor([1]), True, None)
        weights = torch.load(path, map_location='cpu')
        model.load_state_dict(weights['model'])
        ema =  ExponentialMovingAverage(model.parameters(), decay=0.999)
        ema.load_state_dict(weights['ema'])
        return model, ema