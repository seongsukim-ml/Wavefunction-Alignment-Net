import sys
import torch
sys.path.append('/home/yl2428/MADFT-NN/src')
from madftnn.models.equiformer_v2.equiformer_v2_oc20 import TransBlockV2, EquiformerV2_OC20
from madftnn.models.QHNet import QHNetBackBoneSO2, QHNet_backbone
from madftnn.models.QHNet_modify import QHNet_backbone_symmetrySO2
from torch_scatter import scatter


# if __name__ == "__main__":
#     model = EquiformerV2_OC20(0 , 0 , 0, lmax_list = [4])
#     model.to('cuda:2')
#     print(model)
#     data = torch.load('/home/yl2428/MADFT-NN/test/test_cases/ConvNetLayerTestCases/data.pth').to('cuda:2')
#     # return a list of number of atoms from data.batch
#     data.natoms = scatter(torch.ones_like(data.batch), data.batch, dim=0, reduce='sum')
#     energy, forces = model(data)

if __name__ == "__main__":

    data = torch.load('/home/yl2428/MADFT-NN/test/test_cases/ConvNetLayerTestCases/data.pth').to('cuda:2')

    config = {
    "max_neighbors": 20,
    "max_radius": 12.0,
    "max_num_elements": 90,
    "num_layers": 8,
    "sphere_channels": 96,
    "attn_hidden_channels": 64,  # Note: [64, 96] This is the hidden size of message passing. Do not necessarily use 96.
    "num_heads": 8,
    "attn_alpha_channels": 64,  # Note: Not used when `use_s2_act_attn` is True.
    "attn_value_channels": 16,
    "ffn_hidden_channels": 128,
    "norm_type": "layer_norm_sh",  # Note: ['rms_norm_sh', 'layer_norm', 'layer_norm_sh']
    "lmax_list": [4],
    "mmax_list": [2],
    "grid_resolution": 18,  # Note: [18, 16, 14, None] For `None`, simply comment this line.
    "num_sphere_samples": 128,
    "edge_channels": 128,
    "use_atom_edge_embedding": True,
    "share_atom_edge_embedding": False,  # Note: If `True`, `use_atom_edge_embedding` must be `True` and the atom edge embedding will be shared across all blocks.
    "distance_function": "gaussian",
    "num_distance_basis": 512,  # Note: not used
    "attn_activation": "silu",
    "use_s2_act_attn": False,  # Note: [False, True] Switch between attention after S2 activation or the original EquiformerV1 attention.
    "use_attn_renorm": True,  # Attention re-normalization. Used for ablation study.
    "ffn_activation": "silu",  # Note: ['silu', 'swiglu']
    "use_gate_act": False,  # Note: [True, False] Switch between gate activation and S2 activation
    "use_grid_mlp": True,  # Note: [False, True] If `True`, use projecting to grids and performing MLPs for FFNs.
    "use_sep_s2_act": True,  # Separable S2 activation. Used for ablation study.
    "alpha_drop": 0.1,  # Note: [0.0, 0.1]
    "drop_path_rate": 0.1,  # Note: [0.0, 0.05]
    "proj_drop": 0.0,
    "weight_init": "uniform"
    }
    # config = {
    # "use_pbc": False,
    # "regress_forces": False,
    # "otf_graph": True,
    # "max_neighbors": 20,
    # "max_radius": 12.0,
    # "max_num_elements": 90,
    # "num_layers": 20,
    # "sphere_channels": 128,
    # "attn_hidden_channels": 64,  # Note: This determines the hidden size of message passing. Do not necessarily use 96.
    # "num_heads": 8,
    # "attn_alpha_channels": 64,  # Note: Not used when `use_s2_act_attn` is True.
    # "attn_value_channels": 16,
    # "ffn_hidden_channels": 128,
    # "norm_type": "layer_norm_sh",  # Note: Options are ['rms_norm_sh', 'layer_norm', 'layer_norm_sh']
    # "lmax_list": [6],
    # "mmax_list": [3],
    # "grid_resolution": 18,  # Note: Options are [18, 16, 14, None]. For `None`, simply omit this line.
    # "num_sphere_samples": 128,
    # "edge_channels": 128,
    # "use_atom_edge_embedding": True,
    # "share_atom_edge_embedding": False,  # Note: If `True`, `use_atom_edge_embedding` must also be `True` and the atom edge embedding will be shared across all blocks.
    # "distance_function": "gaussian",
    # "num_distance_basis": 512,  # Note: Not used
    # "attn_activation": "silu",
    # "use_s2_act_attn": False,  # Note: Options are [False, True]. Switch between attention after S2 activation or the original EquiformerV1 attention.
    # "use_attn_renorm": True,  # Attention re-normalization. Used for ablation study.
    # "ffn_activation": "silu",  # Note: Options are ['silu', 'swiglu']
    # "use_gate_act": False,  # Note: Options are [True, False]. Switch between gate activation and S2 activation.
    # "use_grid_mlp": True,  # Note: Options are [False, True]. If `True`, use projecting to grids and performing MLPs for FFNs.
    # "use_sep_s2_act": True,  # Separable S2 activation. Used for ablation study.
    # "alpha_drop": 0.1,  # Note: Options are [0.0, 0.1].
    # "drop_path_rate": 0.1,  # Note: Options are [0.0, 0.05].
    # "proj_drop": 0.0,
    # "weight_init": "uniform"  # Note: Options are ['uniform', 'normal'].
    # }
    model = QHNet_backbone_symmetrySO2(order = config['lmax_list'][0], embedding_dimension = config['sphere_channels'],
                                       num_gnn_layers=config['num_layers'], start_layer=config['num_layers'] - 2, load_pretrain= '/home/yl2428/MADFT-NN/eq2_31M_ec4_allmd.pt', **config) 
    # load the state dict when the module is available
    # Assuming your model is already defined and instantiated as `model`
    # Load the state dict from the file
    model.to('cuda:2')
    print(model)
    # # return a list of number of atoms from data.batch
    # data.natoms = scatter(torch.ones_like(data.batch), data.batch, dim=0, reduce='sum')
    # energy, forces = model(data)
    model(data)

