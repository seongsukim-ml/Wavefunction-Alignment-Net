from abc import abstractmethod, ABCMeta
from typing import Optional
import torch
# from torchmdnet.models.utils import act_class_mapping, GatedEquivariantBlock
from torch_scatter import scatter
from torch import nn
from e3nn import o3

from madftnn.models.equiformer.graph_attention_transformer import FeedForwardNetwork


from .equiformer.tensor_product_rescale import (TensorProductRescale, LinearRS,
    FullyConnectedTensorProductRescale, irreps2gate)
from .equiformer.fast_activation import Activation, Gate
from .equiformer.dp_attention_transformer_md17 import _RESCALE
import warnings

##
# Scalar and EquivariantScalar is for visnet kindof model (EGNN)
# EquivariantScalar_viaTP is for equiformer kind of tensor product model
__all__ = ["EquivariantScalar_viaTP"] 


class OutputModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, allow_prior_model):
        super(OutputModel, self).__init__()
        self.allow_prior_model = allow_prior_model

    def reset_parameters(self):
        warnings.warn("sorry, output model not implement reset parameters")

    @abstractmethod
    def pre_reduce(self, data):
        return

    def post_reduce(self, x):
        return x


class EquivariantScalar_viaTP(OutputModel):
    def __init__(self, 
                 irrep_features, 
                 node_attr_dim = None, 
                 activation="silu", 
                 allow_prior_model=True,
                 use_sparse_tp=False,):
        super().__init__(allow_prior_model=allow_prior_model)

        if activation != "silu":
            raise ValueError("This model supports only 'silu' activation.")

        if not isinstance(irrep_features, o3.Irreps):
            irrep_features = o3.Irreps(irrep_features)

        self.irreps_feature = irrep_features
        if node_attr_dim is None:
            self.irreps_node_attr = o3.Irreps(f"{self.irreps_feature[0][0]}x0e")
        else:
            self.irreps_node_attr = o3.Irreps(f"{node_attr_dim}x0e")

        self.equivariant_layer = FeedForwardNetwork(
            irreps_node_input=self.irreps_feature,
            irreps_node_attr=self.irreps_node_attr,
            irreps_node_output=self.irreps_node_attr, 
            irreps_mlp_mid=self.irreps_feature,
            proj_drop=0.0,
            use_sparse_tp=use_sparse_tp,
        )

        self.output_network = nn.Sequential(
            LinearRS(self.irreps_node_attr, self.irreps_node_attr, _RESCALE, use_sparse_tp=use_sparse_tp),
            Activation(self.irreps_node_attr, [torch.nn.SiLU()] * len(self.irreps_node_attr)),
            LinearRS(self.irreps_node_attr, o3.Irreps("1x0e"), _RESCALE, use_sparse_tp=use_sparse_tp),
        )

        self.reset_parameters()

    def pre_reduce(self, batch):
        features = batch["node_vec"] # features after NodeEncoder
        attributes = batch["node_embedding"] # just embedding

        outputs = self.equivariant_layer(features, attributes)
        outputs = self.output_network(outputs)

        batch["pred_energy"] = outputs
        return batch


# class Scalar(OutputModel):
#     def __init__(self, hidden_channels, activation="silu", allow_prior_model=True):
#         super(Scalar, self).__init__(allow_prior_model=allow_prior_model)
#         act_class = act_class_mapping[activation]
#         self.output_network = nn.Sequential(
#             nn.Linear(hidden_channels, hidden_channels // 2),
#             act_class(),
#             nn.Linear(hidden_channels // 2, 1),
#         )

#         self.reset_parameters()

#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.output_network[0].weight)
#         self.output_network[0].bias.data.fill_(0)
#         nn.init.xavier_uniform_(self.output_network[2].weight)
#         self.output_network[2].bias.data.fill_(0)

#     def pre_reduce(self, data):
#         x = data["node_embedding"]
#         # v = data[""]
#         return self.output_network(x)


# class EquivariantScalar(OutputModel):
#     def __init__(self, hidden_channels, activation="silu", allow_prior_model=True):
#         super(EquivariantScalar, self).__init__(allow_prior_model=allow_prior_model)
#         self.output_network = nn.ModuleList(
#             [
#                 GatedEquivariantBlock(
#                     hidden_channels,
#                     hidden_channels // 2,
#                     activation=activation,
#                     scalar_activation=True,
#                 ),
#                 GatedEquivariantBlock(hidden_channels // 2, 1, activation=activation),
#             ]
#         )

#         self.reset_parameters()

#     def reset_parameters(self):
#         for layer in self.output_network:
#             layer.reset_parameters()

#     def pre_reduce(self, data):
#         x = data["node_embedding"]
#         v = data["node_vec"]
#         for layer in self.output_network:
#             x, v = layer(x, v)
#         # include v in output to make sure all parameters have a gradient
#         return x + v.sum() * 0


# class DipoleMoment(Scalar):
#     def __init__(self, hidden_channels, activation="silu"):
#         super(DipoleMoment, self).__init__(
#             hidden_channels, activation, allow_prior_model=False
#         )
#         atomic_mass = torch.from_numpy(ase.data.atomic_masses).float()
#         self.register_buffer("atomic_mass", atomic_mass)

#     def pre_reduce(self, x, v: Optional[torch.Tensor], z, pos, batch):
#         x = self.output_network(x)

#         # Get center of mass.
#         mass = self.atomic_mass[z].view(-1, 1)
#         c = scatter(mass * pos, batch, dim=0) / scatter(mass, batch, dim=0)
#         x = x * (pos - c[batch])
#         return x

#     def post_reduce(self, x):
#         return torch.norm(x, dim=-1, keepdim=True)


# class EquivariantDipoleMoment(EquivariantScalar):
#     def __init__(self, hidden_channels, activation="silu"):
#         super(EquivariantDipoleMoment, self).__init__(
#             hidden_channels, activation, allow_prior_model=False
#         )
#         atomic_mass = torch.from_numpy(ase.data.atomic_masses).float()
#         self.register_buffer("atomic_mass", atomic_mass)

#     def pre_reduce(self, x, v, z, pos, batch):
#         for layer in self.output_network:
#             x, v = layer(x, v)

#         # Get center of mass.
#         mass = self.atomic_mass[z].view(-1, 1)
#         c = scatter(mass * pos, batch, dim=0) / scatter(mass, batch, dim=0)
#         x = x * (pos - c[batch])
#         return x + v.squeeze()

#     def post_reduce(self, x):
#         return torch.norm(x, dim=-1, keepdim=True)


# class ElectronicSpatialExtent(OutputModel):
#     def __init__(self, hidden_channels, activation="silu"):
#         super(ElectronicSpatialExtent, self).__init__(allow_prior_model=False)
#         act_class = act_class_mapping[activation]
#         self.output_network = nn.Sequential(
#             nn.Linear(hidden_channels, hidden_channels // 2),
#             act_class(),
#             nn.Linear(hidden_channels // 2, 1),
#         )
#         atomic_mass = torch.from_numpy(ase.data.atomic_masses).float()
#         self.register_buffer("atomic_mass", atomic_mass)

#         self.reset_parameters()

#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.output_network[0].weight)
#         self.output_network[0].bias.data.fill_(0)
#         nn.init.xavier_uniform_(self.output_network[2].weight)
#         self.output_network[2].bias.data.fill_(0)

#     def pre_reduce(self, x, v: Optional[torch.Tensor], z, pos, batch):
#         x = self.output_network(x)

#         # Get center of mass.
#         mass = self.atomic_mass[z].view(-1, 1)
#         c = scatter(mass * pos, batch, dim=0) / scatter(mass, batch, dim=0)

#         x = torch.norm(pos - c[batch], dim=1, keepdim=True) ** 2 * x
#         return x


# class EquivariantElectronicSpatialExtent(ElectronicSpatialExtent):
#     pass


# class EquivariantVectorOutput(EquivariantScalar):
#     def __init__(self, hidden_channels, activation="silu"):
#         super(EquivariantVectorOutput, self).__init__(
#             hidden_channels, activation, allow_prior_model=False
#         )

#     def pre_reduce(self, x, v, z, pos, batch):
#         for layer in self.output_network:
#             x, v = layer(x, v)
#         return v.squeeze()



# # class EquivariantGradientOutput(EquivariantScalar):
#     def __init__(self, hidden_channels, activation="silu"):
#         super(EquivariantGradientOutput, self).__init__(
#             hidden_channels, activation, allow_prior_model=False
#         )

#     def pre_reduce(self, x, v, z, pos, batch):
#         for layer in self.output_network:
#             x, v = layer(x, v)
        
#         print('pos requires grad', pos.requires_grad)
#         energy = scatter(x, batch, dim = 0)
#         forces = -1 * (
#                 torch.autograd.grad(
#                     energy,
#                     pos,
#                     grad_outputs=torch.ones_like(energy),
#                     create_graph=True,
#                     retain_graph=True
#                 )[0]
#             )
        
#         return forces