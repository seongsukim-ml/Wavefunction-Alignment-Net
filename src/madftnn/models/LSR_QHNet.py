import time

import torch.nn.functional as F

import math
import torch
from torch import nn
# from torch_cluster import radius_graph

import numpy as np
import warnings
from e3nn import o3

from madftnn.models.QHNet import *
from madftnn.models.QHNet_modify import PairNetLayer_symmetry
from madftnn.models.lsrm.lsrm_modules import Visnorm_shared_LSRMNorm2_2branchSerial

class PairNetLayer_symmetry_exp(torch.nn.Module):
    def __init__(self,
                 irrep_in_node,
                 irrep_bottle_hidden,
                 irrep_out,
                 sh_irrep,
                 edge_attr_dim,
                 node_attr_dim,
                 resnet: bool = True,
                 invariant_layers=1,
                 invariant_neurons=8,
                 nonlinear='ssp'):
        super().__init__()
        self.nonlinear_scalars = {1: "ssp", -1: "tanh"}
        self.nonlinear_gates = {1: "ssp", -1: "abs"}
        self.invariant_layers = invariant_layers
        self.invariant_neurons = invariant_neurons
        self.irrep_in_node = irrep_in_node if isinstance(irrep_in_node, o3.Irreps) else o3.Irreps(irrep_in_node)
        self.irrep_bottle_hidden = irrep_bottle_hidden \
            if isinstance(irrep_bottle_hidden, o3.Irreps) else o3.Irreps(irrep_bottle_hidden)
        self.irrep_out = irrep_out if isinstance(irrep_out, o3.Irreps) else o3.Irreps(irrep_out)
        self.sh_irrep = sh_irrep if isinstance(sh_irrep, o3.Irreps) else o3.Irreps(sh_irrep)

        self.edge_attr_dim = edge_attr_dim
        self.node_attr_dim = node_attr_dim
        self.nonlinear_layer = get_nonlinear(nonlinear)

        self.irrep_tp_in_node, _ = get_feasible_irrep(self.irrep_in_node, o3.Irreps("0e"), self.irrep_bottle_hidden)
        self.irrep_tp_out_node_pair, instruction_node_pair = get_feasible_irrep(
            self.irrep_tp_in_node, self.irrep_tp_in_node, self.irrep_bottle_hidden, tp_mode='uuu')

        self.irrep_tp_out_node_pair_msg, instruction_node_pair_msg = get_feasible_irrep(
            self.irrep_tp_in_node, self.sh_irrep, self.irrep_bottle_hidden, tp_mode='uvu')

        self.linear_node_pair = Linear(
            irreps_in=self.irrep_in_node,
            irreps_out=self.irrep_tp_in_node,
            internal_weights=True,
            shared_weights=True,
            biases=True
        )

        self.linear_node_pair_n = Linear(
            irreps_in=self.irrep_in_node,
            irreps_out=self.irrep_in_node,
            internal_weights=True,
            shared_weights=True,
            biases=True
        )
        self.linear_node_pair_inner = Linear(
            irreps_in=self.irrep_in_node,
            irreps_out=self.irrep_in_node,
            internal_weights=True,
            shared_weights=True,
            biases=True
        )

        self.tp_node_pair = TensorProduct(
            self.irrep_tp_in_node,
            self.irrep_tp_in_node,
            self.irrep_tp_out_node_pair,
            instruction_node_pair,
            shared_weights=False,
            internal_weights=False,
        )

        self.irrep_tp_out_node_pair_2, instruction_node_pair_2 = get_feasible_irrep(
            self.irrep_tp_out_node_pair, self.irrep_tp_out_node_pair, self.irrep_bottle_hidden, tp_mode='uuu')

        self.tp_node_pair_2 = TensorProduct(
            self.irrep_tp_out_node_pair,
            self.irrep_tp_out_node_pair,
            self.irrep_tp_out_node_pair_2,
            instruction_node_pair_2,
            shared_weights=True,
            internal_weights=True
        )


        self.fc_node_pair = FullyConnectedNet(
            [self.edge_attr_dim] + invariant_layers * [invariant_neurons] + [self.irrep_in_node[0][0]],
            self.nonlinear_layer
        )

        self.linear_node_pair_2 = Linear(
            irreps_in=self.irrep_tp_out_node_pair_2,
            irreps_out=self.irrep_out,
            internal_weights=True,
            shared_weights=True,
            biases=True
        )

        if self.irrep_in_node == self.irrep_out and resnet:
            self.resnet = True
        else:
            self.resnet = False

        self.linear_node_pair = Linear(
            irreps_in=self.irrep_tp_out_node_pair,
            irreps_out=self.irrep_out,
            internal_weights=True,
            shared_weights=True,
            biases=True
        )
        self.norm_gate = NormGate(self.irrep_tp_out_node_pair)
        self.inner_product = InnerProduct(self.irrep_in_node)
        self.norm = o3.Norm(self.irrep_in_node)
        num_mul = 0
        for mul, ir in self.irrep_in_node:
            num_mul = num_mul + mul

        self.norm_gate_pre = NormGate(self.irrep_tp_out_node_pair)
        self.fc = nn.Sequential(
            # nn.Linear(self.irrep_in_node[0][0] + num_mul, self.irrep_in_node[0][0]),
            nn.Linear(num_mul, self.irrep_in_node[0][0]),
            nn.SiLU(),
            nn.Linear(self.irrep_in_node[0][0], self.irrep_in_node[0][0]))

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, data, node_attr, node_pair_attr=None):
        dst, src = data.full_edge_index
        node_attr_0 = self.linear_node_pair_inner(node_attr)
        s0 = self.inner_product(node_attr_0[dst], node_attr_0[src])[:, self.irrep_in_node.slices()[0].stop:]

        s0 = torch.cat([0.5*node_attr_0[dst][:, self.irrep_in_node.slices()[0]]+
                        0.5*node_attr_0[src][:, self.irrep_in_node.slices()[0]], s0], dim=-1)

        node_attr = self.norm_gate_pre(node_attr)
        node_attr = self.linear_node_pair_n(node_attr)

        # node_pair = self.tp_node_pair(node_attr[src], node_attr[dst],
        #     self.fc_node_pair(data.full_edge_attr) * self.fc(s0))
        total_orb = int(node_attr.shape[-1]/self.irrep_in_node[0][0])
        node_pair = (node_attr[src] + node_attr[dst]) * (self.fc_node_pair(data.full_edge_attr) * self.fc(s0)).unsqueeze(2).expand(-1, -1, total_orb).reshape(-1, node_attr.shape[-1])

        node_pair = self.norm_gate(node_pair)
        node_pair = self.linear_node_pair(node_pair)

        if self.resnet and node_pair_attr is not None:
            node_pair = node_pair + node_pair_attr
        return node_pair

class LSR_QHNet_backbone(QHNet_backbone):
    def __init__(self,
                 order = 4,
                 embedding_dimension=128,
                 bottle_hidden_size=32,
                 num_gnn_layers=5,
                 max_radius=15,
                 num_nodes=20,
                 radius_embed_dim=32,
                 use_equi_norm=False,
                 short_cutoff_upper=4,
                 long_cutoff_upper=9,
                 num_scale_atom_layers=3,
                 num_long_range_layers=3,
                 lsrm_ckpt_path=None,
                 **kwargs):  # maximum nuclear charge (+1, i.e. 87 for up to Rn) for embeddings, can be kept at default
        """
        Initialize the LSR_QHNet model.

        Args:
            order (int): The order of the spherical harmonics.
            embedding_dimension (int): The size of the hidden layer.
            bottle_hidden_size (int): The size of the bottleneck hidden layer.
            num_gnn_layers (int): The number of GNN layers.
            max_radius (int): The maximum radius cutoff.
            num_nodes (int): The number of nodes.
            radius_embed_dim (int): The dimension of the radius embedding.
        """
        super(LSR_QHNet_backbone, self).__init__(
                 order = order, 
                embedding_dimension=embedding_dimension,
                bottle_hidden_size=bottle_hidden_size,
                num_gnn_layers=num_gnn_layers,
                max_radius=max_radius,
                num_nodes=num_nodes,
                radius_embed_dim=radius_embed_dim,
                use_equi_norm=use_equi_norm,
                start_layer=num_gnn_layers-3,
                **kwargs)
        
        self.long_cutoff_upper=long_cutoff_upper
        if long_cutoff_upper is not None:
            self.LSRM_module = Visnorm_shared_LSRMNorm2_2branchSerial(
                hidden_channels=embedding_dimension,
                num_layers=num_scale_atom_layers,
                long_num_layers=num_long_range_layers,
                short_cutoff_upper=short_cutoff_upper,
                long_cutoff_lower=0,
                long_cutoff_upper=long_cutoff_upper,
            )
            if lsrm_ckpt_path is not None:
                self.LSRM_module.load_state_dict(torch.load(lsrm_ckpt_path))
        
    def forward(self, batch_data):
        batch_data['ptr'] = torch.cat([torch.Tensor([0]).to(batch_data["molecule_size"].device).int(),
                              torch.cumsum(batch_data["molecule_size"],dim = 0)],dim = 0)
        
        edge_index = radius_graph(batch_data.pos, self.max_radius, batch_data.batch)
        # dst, src = edge_index
        edge_vec = batch_data.pos[edge_index[0].long()] - batch_data.pos[edge_index[1].long()]
        rbf_new = self.radial_basis_functions(edge_vec.norm(dim=-1).unsqueeze(-1)).squeeze().type(batch_data.pos.type())

        edge_sh = o3.spherical_harmonics(
            self.sh_irrep, edge_vec[:, [1, 2, 0]],
            normalize=True, normalization='component').type(batch_data.pos.type())

        node_attr = self.LSRM_module(batch_data)            
        batch_data.node_attr, batch_data.edge_index, batch_data.edge_attr, batch_data.edge_sh = \
            node_attr, edge_index, rbf_new, edge_sh
        
        for layer_idx, layer in enumerate(self.e3_gnn_layer):
            node_attr = layer(batch_data, node_attr)

        batch_data["node_embedding"] = batch_data.node_attr
        batch_data["node_vec"] = node_attr
        
        # batch_data["node_embedding"] = None
        # batch_data["node_vec"] = node_attr         
        # batch_data["fii"] = fii
        # batch_data["fij"] = fij
        return batch_data

class LSR_QHNetBackBoneSO2_symmetry(nn.Module):
    def __init__(self,
                 order = 4,
                 embedding_dimension=128,
                 bottle_hidden_size=32,
                 num_gnn_layers=5,
                 max_radius=15,
                 num_nodes=20,
                 radius_embed_dim=32,
                 use_equi_norm=False,
                 start_layer = 3,
                 load_pretrain = '',
                 short_cutoff_upper=5,
                 long_cutoff_upper=15,
                 num_scale_atom_layers=4,
                 num_long_range_layers=2,
                 lsrm_ckpt_path=None,
                 **kwargs):  # maximum nuclear charge (+1, i.e. 87 for up to Rn) for embeddings, can be kept at default
        """
        Initialize the QHNet_backbone model.

        Args:
            order (int): The order of the spherical harmonics.
            embedding_dimension (int): The size of the hidden layer.
            bottle_hidden_size (int): The size of the bottleneck hidden layer.
            num_gnn_layers (int): The number of GNN layers.
            max_radius (int): The maximum radius cutoff.
            num_nodes (int): The number of nodes.
            radius_embed_dim (int): The dimension of the radius embedding.
        """
        
        super().__init__()
        self.order = order
        self.sh_irrep = o3.Irreps.spherical_harmonics(lmax=self.order)
        self.hs = embedding_dimension
        self.hbs = bottle_hidden_size
        self.radius_embed_dim = radius_embed_dim
        self.max_radius = max_radius
        self.num_gnn_layers = num_gnn_layers
        self.node_embedding = nn.Embedding(num_nodes, self.hs)
        
        self.init_sph_irrep = o3.Irreps(construct_o3irrps(1, order=order))
        
        self.irreps_node_embedding = construct_o3irrps_base(self.hs, order=order)
        self.hidden_irrep = o3.Irreps(construct_o3irrps(self.hs, order=order))
        self.hidden_irrep_base = o3.Irreps(self.irreps_node_embedding)
        self.hidden_bottle_irrep = o3.Irreps(construct_o3irrps(self.hbs, order=order))
        self.hidden_bottle_irrep_base = o3.Irreps(construct_o3irrps_base(self.hbs, order=order))
        
        self.input_irrep = o3.Irreps(f'{self.hs}x0e')
        self.radial_basis_functions = ExponentialBernsteinRadialBasisFunctions(self.radius_embed_dim, self.max_radius)
        self.nonlinear_scalars = {1: "ssp", -1: "tanh"}
        self.nonlinear_gates = {1: "ssp", -1: "abs"}
        self.num_fc_layer = 1

        self.e3_gnn_layer = nn.ModuleList()
        self.e3_gnn_node_pair_layer = nn.ModuleList()
        self.e3_gnn_node_layer = nn.ModuleList()
        self.udpate_layer = nn.ModuleList()
        self.start_layer = start_layer
        # prevent double kwargs
        [kwargs.pop(x, None) for x in ["use_pbc", "regress_forces", "max_raius", "otf_graph", "num_layers", "sphere_channels", "lmax_list"]]
        self.node_attr_encoder = EquiformerV2_OC20Backbone(0, 0, 0, max_radius = max_radius, lmax_list=[order], 
                                                           sphere_channels=embedding_dimension, 
                                                           num_layers = num_gnn_layers, use_pbc=False, 
                                                           regress_forces=False, otf_graph=False, **kwargs)
        if load_pretrain != '':
            loaded_state_dict = torch.load(load_pretrain)['state_dict']
            state_dict = {k.replace('module.module.', ''): v for k, v in loaded_state_dict.items()}
            self.node_attr_encoder.load_state_dict(state_dict, strict=False)

        for i in range(self.num_gnn_layers):
            if i > self.start_layer:
                self.e3_gnn_node_layer.append(SelfNetLayer(
                        irrep_in_node=self.hidden_irrep_base,
                        irrep_bottle_hidden=self.hidden_irrep_base,
                        irrep_out=self.hidden_irrep_base,
                        sh_irrep=self.sh_irrep,
                        edge_attr_dim=self.radius_embed_dim,
                        node_attr_dim=self.hs,
                        resnet=True,
                ))

                self.e3_gnn_node_pair_layer.append(PairNetLayer_symmetry(
                        irrep_in_node=self.hidden_irrep_base,
                        irrep_bottle_hidden=self.hidden_irrep_base,
                        irrep_out=self.hidden_irrep_base,
                        sh_irrep=self.sh_irrep,
                        edge_attr_dim=self.radius_embed_dim,
                        node_attr_dim=self.hs,
                        invariant_layers=self.num_fc_layer,
                        invariant_neurons=self.hs,
                        resnet=True,
                ))
        
        self.LSRM_module = Visnorm_shared_LSRMNorm2_2branchSerial(
            hidden_channels=embedding_dimension,
            num_layers=num_scale_atom_layers,
            long_num_layers=num_long_range_layers,
            short_cutoff_upper=short_cutoff_upper,
            long_cutoff_lower=0,
            long_cutoff_upper=long_cutoff_upper,
        )
        if lsrm_ckpt_path is not None:
            self.LSRM_module.load_state_dict(torch.load(lsrm_ckpt_path))

        self.output_ii = Linear(self.hidden_irrep, self.hidden_bottle_irrep)
        self.output_ij = Linear(self.hidden_irrep, self.hidden_bottle_irrep)

    def reset_parameters(self):
        warnings.warn("reset parameter is not init in qhnet backbone model")
    
    def build_graph(self, data, max_radius):
        node_attr = data.atomic_numbers.squeeze()
        radius_edges = radius_graph(data.pos, max_radius, data.batch,max_num_neighbors=200)

        if max_radius < 100:
            radius_edges = self.remove_repeat_edge(radius_edges)

        dst, src = radius_edges
        edge_vec = data.pos[dst.long()] - data.pos[src.long()]
        rbf = self.radial_basis_functions(edge_vec.norm(dim=-1).unsqueeze(-1)).squeeze().type(data.pos.type())

        edge_sh = o3.spherical_harmonics(
            self.sh_irrep, edge_vec[:, [1, 2, 0]],
            normalize=True, normalization='component').type(data.pos.type())

        start_edge_index = 0
        all_transpose_index = []
        for graph_idx in range(data.ptr.shape[0] - 1):
            num_nodes = data.ptr[graph_idx +1] - data.ptr[graph_idx]
            graph_edge_index = radius_edges[:, start_edge_index:start_edge_index+num_nodes*(num_nodes-1)]
            sub_graph_edge_index = graph_edge_index - data.ptr[graph_idx]
            bias = (sub_graph_edge_index[0] < sub_graph_edge_index[1]).type(torch.int)
            transpose_index = sub_graph_edge_index[0] * (num_nodes - 1) + sub_graph_edge_index[1] - bias
            transpose_index = transpose_index + start_edge_index
            all_transpose_index.append(transpose_index)
            start_edge_index = start_edge_index + num_nodes*(num_nodes-1)

        return node_attr, radius_edges, rbf, edge_sh, torch.cat(all_transpose_index, dim=-1)

    def forward(self, batch_data):
        batch_data['ptr'] = torch.cat([torch.Tensor([0]).to(batch_data["molecule_size"].device).int(),
                              torch.cumsum(batch_data["molecule_size"],dim = 0)],dim = 0)
        
        batch_data['natoms'] = scatter(torch.ones_like(batch_data.batch), batch_data.batch, dim=0, reduce='sum')
        if "non_diag_hamiltonian" in batch_data:
            full_edge_index = radius_graph(batch_data.pos, 1000, batch_data.batch,max_num_neighbors=1000)
            batch_data["non_diag_hamiltonian"] = batch_data["non_diag_hamiltonian"][full_edge_index[0]>full_edge_index[1]]
            batch_data['non_diag_mask'] = batch_data["non_diag_mask"][full_edge_index[0]>full_edge_index[1]]
            batch_data['non_diag_ocp_mask'] = batch_data["non_diag_ocp_mask"][full_edge_index[0]>full_edge_index[1]]
            full_edge_index = full_edge_index[:,full_edge_index[0]>full_edge_index[1]]

            full_edge_vec = batch_data.pos[full_edge_index[0].long()] - batch_data.pos[full_edge_index[1].long()]
            full_edge_rbf = self.radial_basis_functions(full_edge_vec.norm(dim=-1).unsqueeze(-1)).squeeze().type(batch_data.pos.type())
            full_edge_sh = o3.spherical_harmonics(
                    self.sh_irrep, full_edge_vec[:, [1, 2, 0]],
                    normalize=True, normalization='component').type(batch_data.pos.type())


            batch_data.full_edge_index = full_edge_index
            batch_data.full_edge_attr = full_edge_rbf
            batch_data.full_edge_sh = full_edge_sh
        
        # batch_data.node_attr = self.node_embedding(batch_data.atomic_numbers.squeeze())
        batch_data.node_attr = self.LSRM_module(batch_data)
        edge_index = radius_graph(batch_data.pos, self.max_radius, batch_data.batch)
        batch_data.edge_index = edge_index
        # dst, src = edge_index
        # edge_vec = batch_data.pos[edge_index[0].long()] - batch_data.pos[edge_index[1].long()]
        # rbf_new = self.radial_basis_functions(edge_vec.norm(dim=-1).unsqueeze(-1)).squeeze().type(batch_data.pos.type())

        # edge_sh = o3.spherical_harmonics(
        #     self.sh_irrep, edge_vec[:, [1, 2, 0]],
        #     normalize=True, normalization='component').type(batch_data.pos.type())

        # node_attr = self.node_embedding(batch_data.atomic_numbers.squeeze())
        
        # batch_data.node_attr, batch_data.edge_index, batch_data.edge_attr, batch_data.edge_sh = \
        #     node_attr, edge_index, rbf_new, edge_sh

        

        node_attr = self.node_attr_encoder(batch_data)
        batch_data["node_embedding"] = None
        batch_data["node_vec"] = node_attr  
        fii = None
        fij = None

        if "non_diag_hamiltonian" in batch_data:
            for layer_idx in range(self.num_gnn_layers):
                if layer_idx > self.start_layer:
                    fii = self.e3_gnn_node_layer[layer_idx-self.start_layer-1](batch_data, node_attr, fii)
                    fij = self.e3_gnn_node_pair_layer[layer_idx-self.start_layer-1](batch_data, node_attr, fij)
            fii = self.output_ii(fii)
            fij = self.output_ij(fij)
        
            batch_data["fii"] = fii
            batch_data["fij"] = fij
        return batch_data



class LSR_QHNetBackBoneSO2(nn.Module):
    def __init__(self,
                 order = 4,
                 embedding_dimension=128,
                 bottle_hidden_size=32,
                 num_gnn_layers=5,
                 max_radius=15,
                 num_nodes=20,
                 use_equi_norm=False,
                 start_layer = 3,
                 load_pretrain = '',
                 radius_embed_dim=32,
                 short_cutoff_upper=5,
                 long_cutoff_upper=15,
                 num_scale_atom_layers=4,
                 num_long_range_layers=2,
                 lsrm_ckpt_path=None,
                 **kwargs):  # maximum nuclear charge (+1, i.e. 87 for up to Rn) for embeddings, can be kept at default
        """
        Initialize the QHNet_backbone model.

        Args:
            order (int): The order of the spherical harmonics.
            embedding_dimension (int): The size of the hidden layer.
            bottle_hidden_size (int): The size of the bottleneck hidden layer.
            num_gnn_layers (int): The number of GNN layers.
            max_radius (int): The maximum radius cutoff.
            num_nodes (int): The number of nodes.
            radius_embed_dim (int): The dimension of the radius embedding.
        """
        
        super(LSR_QHNetBackBoneSO2, self).__init__()
        self.order = order
        self.sh_irrep = o3.Irreps.spherical_harmonics(lmax=self.order)
        self.hs = embedding_dimension
        self.hbs = bottle_hidden_size
        self.radius_embed_dim = radius_embed_dim
        self.max_radius = max_radius
        self.num_gnn_layers = num_gnn_layers
        self.node_embedding = nn.Embedding(num_nodes, self.hs)
        
        self.init_sph_irrep = o3.Irreps(construct_o3irrps(1, order=order))
        
        self.irreps_node_embedding = construct_o3irrps_base(self.hs, order=order)
        self.hidden_irrep = o3.Irreps(construct_o3irrps(self.hs, order=order))
        self.hidden_irrep_base = o3.Irreps(self.irreps_node_embedding)
        self.hidden_bottle_irrep = o3.Irreps(construct_o3irrps(self.hbs, order=order))
        self.hidden_bottle_irrep_base = o3.Irreps(construct_o3irrps_base(self.hbs, order=order))
        
        self.input_irrep = o3.Irreps(f'{self.hs}x0e')
        self.radial_basis_functions = ExponentialBernsteinRadialBasisFunctions(self.radius_embed_dim, self.max_radius)
        self.nonlinear_scalars = {1: "ssp", -1: "tanh"}
        self.nonlinear_gates = {1: "ssp", -1: "abs"}
        self.num_fc_layer = 1

        self.e3_gnn_layer = nn.ModuleList()
        self.e3_gnn_node_pair_layer = nn.ModuleList()
        self.e3_gnn_node_layer = nn.ModuleList()
        self.udpate_layer = nn.ModuleList()
        self.start_layer = start_layer
        # prevent double kwargs
        [kwargs.pop(x, None) for x in ["use_pbc", "regress_forces", "max_raius", "otf_graph", "num_layers", "sphere_channels", "lmax_list"]]
        # self.node_attr_encoder = EquiformerV2_OC20Backbone(0, 0, 0, 
        #                                                    max_radius = max_radius, lmax_list=[order], 
        #                                                    sphere_channels=embedding_dimension, 
        #                                                    num_layers= num_gnn_layers, use_pbc=False) 
        self.node_attr_encoder = EquiformerV2_OC20Backbone(0, 0, 0, max_radius = max_radius, lmax_list=[order], 
                                                    sphere_channels=embedding_dimension, 
                                                    num_layers = num_gnn_layers, use_pbc=False, 
                                                    regress_forces=False, otf_graph=False, **kwargs) # the first three parameters are not used in the backbone model
        # - Build modules -
        for i in range(self.num_gnn_layers):
            if i > self.start_layer:
                self.e3_gnn_node_layer.append(SelfNetLayer(
                        irrep_in_node= self.hidden_irrep_base,
                        irrep_bottle_hidden=self.hidden_irrep_base,
                        irrep_out=self.hidden_irrep_base,
                        sh_irrep=self.sh_irrep,
                        edge_attr_dim=self.radius_embed_dim,
                        node_attr_dim=self.hs,
                        resnet=True,
                ))

                self.e3_gnn_node_pair_layer.append(PairNetLayer(
                        irrep_in_node=self.hidden_irrep_base,
                        irrep_bottle_hidden=self.hidden_irrep_base,
                        irrep_out=self.hidden_irrep_base,
                        sh_irrep=self.sh_irrep,
                        edge_attr_dim=self.radius_embed_dim,
                        node_attr_dim=self.hs,
                        invariant_layers=self.num_fc_layer,
                        invariant_neurons=self.hs,
                        resnet=True,
                ))
        self.LSRM_module = Visnorm_shared_LSRMNorm2_2branchSerial(
            hidden_channels=embedding_dimension,
            num_layers=num_scale_atom_layers,
            long_num_layers=num_long_range_layers,
            short_cutoff_upper=short_cutoff_upper,
            long_cutoff_lower=0,
            long_cutoff_upper=long_cutoff_upper,
        )
        if lsrm_ckpt_path is not None:
            self.LSRM_module.load_state_dict(torch.load(lsrm_ckpt_path))

        self.output_ii = Linear(self.hidden_irrep, self.hidden_bottle_irrep)
        self.output_ij = Linear(self.hidden_irrep, self.hidden_bottle_irrep)

    def reset_parameters(self):
        warnings.warn("reset parameter is not init in qhnet backbone model")
    
    def build_graph(self, data, max_radius):
        node_attr = data.atomic_numbers.squeeze()
        radius_edges = radius_graph(data.pos, max_radius, data.batch,max_num_neighbors=200)
        if max_radius < 100:
            radius_edges = self.remove_repeat_edge(radius_edges)

        dst, src = radius_edges
        edge_vec = data.pos[dst.long()] - data.pos[src.long()]
        rbf = self.radial_basis_functions(edge_vec.norm(dim=-1).unsqueeze(-1)).squeeze().type(data.pos.type())

        edge_sh = o3.spherical_harmonics(
            self.sh_irrep, edge_vec[:, [1, 2, 0]],
            normalize=True, normalization='component').type(data.pos.type())

        start_edge_index = 0
        all_transpose_index = []
        for graph_idx in range(data.ptr.shape[0] - 1):
            num_nodes = data.ptr[graph_idx +1] - data.ptr[graph_idx]
            graph_edge_index = radius_edges[:, start_edge_index:start_edge_index+num_nodes*(num_nodes-1)]
            sub_graph_edge_index = graph_edge_index - data.ptr[graph_idx]
            bias = (sub_graph_edge_index[0] < sub_graph_edge_index[1]).type(torch.int)
            transpose_index = sub_graph_edge_index[0] * (num_nodes - 1) + sub_graph_edge_index[1] - bias
            transpose_index = transpose_index + start_edge_index
            all_transpose_index.append(transpose_index)
            start_edge_index = start_edge_index + num_nodes*(num_nodes-1)

        return node_attr, radius_edges, rbf, edge_sh, torch.cat(all_transpose_index, dim=-1)

    def forward(self, batch_data):
        batch_data['ptr'] = torch.cat([torch.Tensor([0]).to(batch_data["molecule_size"].device).int(),
                              torch.cumsum(batch_data["molecule_size"],dim = 0)],dim = 0)
        batch_data['natoms'] = scatter(torch.ones_like(batch_data.batch), batch_data.batch, dim=0, reduce='sum')
        _, full_edge_index, full_edge_attr, full_edge_sh, transpose_edge_index = self.build_graph(batch_data, 10000)
        batch_data.full_edge_index, batch_data.full_edge_attr, batch_data.full_edge_sh = \
        full_edge_index, full_edge_attr, full_edge_sh

        batch_data.full_edge_index = full_edge_index
        batch_data.full_edge_attr = full_edge_attr
        batch_data.transpose_edge_index = transpose_edge_index
        # batch_data.node_attr = self.node_embedding(batch_data.atomic_numbers.squeeze())
        batch_data.node_attr = self.LSRM_module(batch_data)
        edge_index = radius_graph(batch_data.pos, self.max_radius, batch_data.batch)
        batch_data.edge_index = edge_index
        # dst, src = edge_index
        # edge_vec = batch_data.pos[edge_index[0].long()] - batch_data.pos[edge_index[1].long()]
        # rbf_new = self.radial_basis_functions(edge_vec.norm(dim=-1).unsqueeze(-1)).squeeze().type(batch_data.pos.type())

        # edge_sh = o3.spherical_harmonics(
        #     self.sh_irrep, edge_vec[:, [1, 2, 0]],
        #     normalize=True, normalization='component').type(batch_data.pos.type())
        # batch_data.edge_index, batch_data.edge_attr, batch_data.edge_sh =  edge_index, rbf_new, edge_sh


        fii = None
        fij = None
        node_attr = self.node_attr_encoder(batch_data)
        for layer_idx in range(self.num_gnn_layers):
            if layer_idx > self.start_layer:
                fii = self.e3_gnn_node_layer[layer_idx-self.start_layer-1](batch_data, node_attr, fii)
                fij = self.e3_gnn_node_pair_layer[layer_idx-self.start_layer-1](batch_data, node_attr, fij)
        fii = self.output_ii(fii)
        fij = self.output_ij(fij)
        
        batch_data["node_embedding"] = None
        batch_data["node_vec"] = node_attr         
        batch_data["fii"] = fii
        batch_data["fij"] = fij
        return batch_data

class LSR_QHNetBackBoneSO2_symmetry_exp(nn.Module):
    def __init__(self,
                 order = 4,
                 embedding_dimension=128,
                 bottle_hidden_size=32,
                 num_gnn_layers=5,
                 max_radius=15,
                 num_nodes=20,
                 radius_embed_dim=32,
                 use_equi_norm=False,
                 start_layer = 3,
                 load_pretrain = '',
                 short_cutoff_upper=5,
                 long_cutoff_upper=15,
                 num_scale_atom_layers=4,
                 num_long_range_layers=2,
                 lsrm_ckpt_path=None,
                 **kwargs):  # maximum nuclear charge (+1, i.e. 87 for up to Rn) for embeddings, can be kept at default
        """
        Initialize the QHNet_backbone model.

        Args:
            order (int): The order of the spherical harmonics.
            embedding_dimension (int): The size of the hidden layer.
            bottle_hidden_size (int): The size of the bottleneck hidden layer.
            num_gnn_layers (int): The number of GNN layers.
            max_radius (int): The maximum radius cutoff.
            num_nodes (int): The number of nodes.
            radius_embed_dim (int): The dimension of the radius embedding.
        """
        
        super().__init__()
        self.order = order
        self.sh_irrep = o3.Irreps.spherical_harmonics(lmax=self.order)
        self.hs = embedding_dimension
        self.hbs = bottle_hidden_size
        self.radius_embed_dim = radius_embed_dim
        self.max_radius = max_radius
        self.num_gnn_layers = num_gnn_layers
        self.node_embedding = nn.Embedding(num_nodes, self.hs)
        
        self.init_sph_irrep = o3.Irreps(construct_o3irrps(1, order=order))
        
        self.irreps_node_embedding = construct_o3irrps_base(self.hs, order=order)
        self.hidden_irrep = o3.Irreps(construct_o3irrps(self.hs, order=order))
        self.hidden_irrep_base = o3.Irreps(self.irreps_node_embedding)
        self.hidden_bottle_irrep = o3.Irreps(construct_o3irrps(self.hbs, order=order))
        self.hidden_bottle_irrep_base = o3.Irreps(construct_o3irrps_base(self.hbs, order=order))
        
        self.input_irrep = o3.Irreps(f'{self.hs}x0e')
        self.radial_basis_functions = ExponentialBernsteinRadialBasisFunctions(self.radius_embed_dim, self.max_radius)
        self.nonlinear_scalars = {1: "ssp", -1: "tanh"}
        self.nonlinear_gates = {1: "ssp", -1: "abs"}
        self.num_fc_layer = 1

        self.e3_gnn_layer = nn.ModuleList()
        self.e3_gnn_node_pair_layer = nn.ModuleList()
        self.e3_gnn_node_layer = nn.ModuleList()
        self.udpate_layer = nn.ModuleList()
        self.start_layer = start_layer
        # prevent double kwargs
        [kwargs.pop(x, None) for x in ["use_pbc", "regress_forces", "max_raius", "otf_graph", "num_layers", "sphere_channels", "lmax_list"]]
        self.node_attr_encoder = EquiformerV2_OC20Backbone(0, 0, 0, max_radius = max_radius, lmax_list=[order], 
                                                           sphere_channels=embedding_dimension, 
                                                           num_layers = num_gnn_layers, use_pbc=False, 
                                                           regress_forces=False, otf_graph=False, **kwargs)
        if load_pretrain != '':
            loaded_state_dict = torch.load(load_pretrain)['state_dict']
            state_dict = {k.replace('module.module.', ''): v for k, v in loaded_state_dict.items()}
            self.node_attr_encoder.load_state_dict(state_dict, strict=False)

        for i in range(self.num_gnn_layers):
            if i > self.start_layer:
                self.e3_gnn_node_layer.append(SelfNetLayer(
                        irrep_in_node=self.hidden_irrep_base,
                        irrep_bottle_hidden=self.hidden_irrep_base,
                        irrep_out=self.hidden_irrep_base,
                        sh_irrep=self.sh_irrep,
                        edge_attr_dim=self.radius_embed_dim,
                        node_attr_dim=self.hs,
                        resnet=True,
                ))

                self.e3_gnn_node_pair_layer.append(PairNetLayer_symmetry_exp(
                        irrep_in_node=self.hidden_irrep_base,
                        irrep_bottle_hidden=self.hidden_irrep_base,
                        irrep_out=self.hidden_irrep_base,
                        sh_irrep=self.sh_irrep,
                        edge_attr_dim=self.radius_embed_dim,
                        node_attr_dim=self.hs,
                        invariant_layers=self.num_fc_layer,
                        invariant_neurons=self.hs,
                        resnet=True,
                ))
        
        self.LSRM_module = Visnorm_shared_LSRMNorm2_2branchSerial(
            hidden_channels=embedding_dimension,
            num_layers=num_scale_atom_layers,
            long_num_layers=num_long_range_layers,
            short_cutoff_upper=short_cutoff_upper,
            long_cutoff_lower=0,
            long_cutoff_upper=long_cutoff_upper,
        )
        if lsrm_ckpt_path is not None:
            self.LSRM_module.load_state_dict(torch.load(lsrm_ckpt_path))

        self.output_ii = Linear(self.hidden_irrep, self.hidden_bottle_irrep)
        self.output_ij = Linear(self.hidden_irrep, self.hidden_bottle_irrep)

    def reset_parameters(self):
        warnings.warn("reset parameter is not init in qhnet backbone model")
    
    def build_graph(self, data, max_radius):
        node_attr = data.atomic_numbers.squeeze()
        radius_edges = radius_graph(data.pos, max_radius, data.batch,max_num_neighbors=200)

        if max_radius < 100:
            radius_edges = self.remove_repeat_edge(radius_edges)

        dst, src = radius_edges
        edge_vec = data.pos[dst.long()] - data.pos[src.long()]
        rbf = self.radial_basis_functions(edge_vec.norm(dim=-1).unsqueeze(-1)).squeeze().type(data.pos.type())

        edge_sh = o3.spherical_harmonics(
            self.sh_irrep, edge_vec[:, [1, 2, 0]],
            normalize=True, normalization='component').type(data.pos.type())

        start_edge_index = 0
        all_transpose_index = []
        for graph_idx in range(data.ptr.shape[0] - 1):
            num_nodes = data.ptr[graph_idx +1] - data.ptr[graph_idx]
            graph_edge_index = radius_edges[:, start_edge_index:start_edge_index+num_nodes*(num_nodes-1)]
            sub_graph_edge_index = graph_edge_index - data.ptr[graph_idx]
            bias = (sub_graph_edge_index[0] < sub_graph_edge_index[1]).type(torch.int)
            transpose_index = sub_graph_edge_index[0] * (num_nodes - 1) + sub_graph_edge_index[1] - bias
            transpose_index = transpose_index + start_edge_index
            all_transpose_index.append(transpose_index)
            start_edge_index = start_edge_index + num_nodes*(num_nodes-1)

        return node_attr, radius_edges, rbf, edge_sh, torch.cat(all_transpose_index, dim=-1)

    def forward(self, batch_data):
        batch_data['ptr'] = torch.cat([torch.Tensor([0]).to(batch_data["molecule_size"].device).int(),
                              torch.cumsum(batch_data["molecule_size"],dim = 0)],dim = 0)
        
        batch_data['natoms'] = scatter(torch.ones_like(batch_data.batch), batch_data.batch, dim=0, reduce='sum')
        if "non_diag_hamiltonian" in batch_data:
            full_edge_index = radius_graph(batch_data.pos, 1000, batch_data.batch,max_num_neighbors=1000)
            batch_data["non_diag_hamiltonian"] = batch_data["non_diag_hamiltonian"][full_edge_index[0]>full_edge_index[1]]
            batch_data['non_diag_mask'] = batch_data["non_diag_mask"][full_edge_index[0]>full_edge_index[1]]
            batch_data['non_diag_ocp_mask'] = batch_data["non_diag_ocp_mask"][full_edge_index[0]>full_edge_index[1]]
            full_edge_index = full_edge_index[:,full_edge_index[0]>full_edge_index[1]]

            full_edge_vec = batch_data.pos[full_edge_index[0].long()] - batch_data.pos[full_edge_index[1].long()]
            full_edge_rbf = self.radial_basis_functions(full_edge_vec.norm(dim=-1).unsqueeze(-1)).squeeze().type(batch_data.pos.type())
            full_edge_sh = o3.spherical_harmonics(
                    self.sh_irrep, full_edge_vec[:, [1, 2, 0]],
                    normalize=True, normalization='component').type(batch_data.pos.type())


            batch_data.full_edge_index = full_edge_index
            batch_data.full_edge_attr = full_edge_rbf
            batch_data.full_edge_sh = full_edge_sh
        
        # batch_data.node_attr = self.node_embedding(batch_data.atomic_numbers.squeeze())
        batch_data.node_attr = self.LSRM_module(batch_data)
        edge_index = radius_graph(batch_data.pos, self.max_radius, batch_data.batch)
        batch_data.edge_index = edge_index
        # dst, src = edge_index
        # edge_vec = batch_data.pos[edge_index[0].long()] - batch_data.pos[edge_index[1].long()]
        # rbf_new = self.radial_basis_functions(edge_vec.norm(dim=-1).unsqueeze(-1)).squeeze().type(batch_data.pos.type())

        # edge_sh = o3.spherical_harmonics(
        #     self.sh_irrep, edge_vec[:, [1, 2, 0]],
        #     normalize=True, normalization='component').type(batch_data.pos.type())

        # node_attr = self.node_embedding(batch_data.atomic_numbers.squeeze())
        
        # batch_data.node_attr, batch_data.edge_index, batch_data.edge_attr, batch_data.edge_sh = \
        #     node_attr, edge_index, rbf_new, edge_sh

        

        node_attr = self.node_attr_encoder(batch_data)
        batch_data["node_embedding"] = None
        batch_data["node_vec"] = node_attr  
        fii = None
        fij = None

        if "non_diag_hamiltonian" in batch_data:
            for layer_idx in range(self.num_gnn_layers):
                if layer_idx > self.start_layer:
                    fii = self.e3_gnn_node_layer[layer_idx-self.start_layer-1](batch_data, node_attr, fii)
                    fij = self.e3_gnn_node_pair_layer[layer_idx-self.start_layer-1](batch_data, node_attr, fij)
            fii = self.output_ii(fii)
            fij = self.output_ij(fij)
        
            batch_data["fii"] = fii
            batch_data["fij"] = fij
        return batch_data
