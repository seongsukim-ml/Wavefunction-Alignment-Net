import math
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.autograd import grad
from torch.nn import Embedding, LayerNorm, Linear, Parameter

from torch_geometric.nn import MessagePassing, radius_graph
from torch_geometric.utils import scatter
from .utils import construct_o3irrps


class CosineCutoff(torch.nn.Module):
    r"""Appies a cosine cutoff to the input distances.

    .. math::
        \text{cutoffs} =
        \begin{cases}
        0.5 * (\cos(\frac{\text{distances} * \pi}{\text{cutoff}}) + 1.0),
        & \text{if } \text{distances} < \text{cutoff} \\
        0, & \text{otherwise}
        \end{cases}

    Args:
        cutoff (float): A scalar that determines the point at which the cutoff
            is applied.
    """
    def __init__(self, cutoff: float) -> None:
        super().__init__()
        self.cutoff = cutoff

    def forward(self, distances: Tensor) -> Tensor:
        r"""Applies a cosine cutoff to the input distances.

        Args:
            distances (torch.Tensor): A tensor of distances.

        Returns:
            cutoffs (torch.Tensor): A tensor where the cosine function
                has been applied to the distances,
                but any values that exceed the cutoff are set to 0.
        """
        cutoffs = 0.5 * ((distances * math.pi / self.cutoff).cos() + 1.0)
        cutoffs = cutoffs * (distances < self.cutoff).float()
        return cutoffs


class ExpNormalSmearing(torch.nn.Module):
    r"""Applies exponential normal smearing to the input distances.

    .. math::
        \text{smeared\_dist} = \text{CosineCutoff}(\text{dist})
        * e^{-\beta * (e^{\alpha * (-\text{dist})} - \text{means})^2}

    Args:
        cutoff (float, optional): A scalar that determines the point at which
            the cutoff is applied. (default: :obj:`5.0`)
        num_rbf (int, optional): The number of radial basis functions.
            (default: :obj:`128`)
        trainable (bool, optional): If set to :obj:`False`, the means and betas
            of the RBFs will not be trained. (default: :obj:`True`)
    """
    def __init__(
        self,
        cutoff: float = 5.0,
        num_rbf: int = 128,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.num_rbf = num_rbf
        self.trainable = trainable

        self.cutoff_fn = CosineCutoff(cutoff)
        self.alpha = 5.0 / cutoff

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter('means', Parameter(means))
            self.register_parameter('betas', Parameter(betas))
        else:
            self.register_buffer('means', means)
            self.register_buffer('betas', betas)

    def _initial_params(self) -> Tuple[Tensor, Tensor]:
        r"""Initializes the means and betas for the radial basis functions."""
        start_value = torch.exp(torch.tensor(-self.cutoff))
        means = torch.linspace(start_value, 1, self.num_rbf)
        betas = torch.tensor([(2 / self.num_rbf * (1 - start_value))**-2] *
                             self.num_rbf)
        return means, betas

    def reset_parameters(self):
        r"""Resets the means and betas to their initial values."""
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist: Tensor) -> Tensor:
        r"""Applies the exponential normal smearing to the input distance.

        Args:
            dist (torch.Tensor): A tensor of distances.
        """
        dist = dist.unsqueeze(-1)
        smeared_dist = self.cutoff_fn(dist) * (-self.betas * (
            (self.alpha * (-dist)).exp() - self.means)**2).exp()
        return smeared_dist


class Sphere(torch.nn.Module):
    r"""Computes spherical harmonics of the input data.

    This module computes the spherical harmonics up to a given degree
    :obj:`lmax` for the input tensor of 3D vectors.
    The vectors are assumed to be given in Cartesian coordinates.
    See `here <https://en.wikipedia.org/wiki/Table_of_spherical_harmonics>`_
    for mathematical details.

    Args:
        lmax (int, optional): The maximum degree of the spherical harmonics.
            (default: :obj:`2`)
    """
    def __init__(self, lmax: int = 2) -> None:
        super().__init__()
        self.lmax = lmax

    def forward(self, edge_vec: Tensor) -> Tensor:
        r"""Computes the spherical harmonics of the input tensor.

        Args:
            edge_vec (torch.Tensor): A tensor of 3D vectors.
        """
        return self._spherical_harmonics(
            self.lmax,
            edge_vec[..., 0],
            edge_vec[..., 1],
            edge_vec[..., 2],
        )

    @staticmethod
    def _spherical_harmonics(
        lmax: int,
        x: Tensor,
        y: Tensor,
        z: Tensor,
    ) -> Tensor:
        r"""Computes the spherical harmonics up to degree :obj:`lmax` of the
        input vectors.

        Args:
            lmax (int): The maximum degree of the spherical harmonics.
            x (torch.Tensor): The x coordinates of the vectors.
            y (torch.Tensor): The y coordinates of the vectors.
            z (torch.Tensor): The z coordinates of the vectors.
        """

        sh_1_0 = x
        sh_1_1 = y
        sh_1_2 = z
        if lmax == 1:
            return torch.stack([
                sh_1_0, sh_1_1, sh_1_2
            ], dim=-1)

        sh_2_0 = math.sqrt(3.0) * x * z
        sh_2_1 = math.sqrt(3.0) * x * y
        y2 = y.pow(2)
        x2z2 = x.pow(2) + z.pow(2)
        sh_2_2 = y2 - 0.5 * x2z2
        sh_2_3 = math.sqrt(3.0) * y * z
        sh_2_4 = math.sqrt(3.0) / 2.0 * (z.pow(2) - x.pow(2))

        if lmax == 2:
            return torch.stack([
                sh_1_0, sh_1_1, sh_1_2,
                sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4
            ], dim=-1)

        sh_3_0 = math.sqrt(5.0 / 6.0) * (sh_2_0 * z + sh_2_4 * x)
        sh_3_1 = math.sqrt(5.0) * sh_2_0 * y
        sh_3_2 = math.sqrt(3.0 / 8.0) * (4.0 * y2 - x2z2) * x
        sh_3_3 = 0.5 * y * (2.0 * y2 - 3.0 * x2z2)
        sh_3_4 = math.sqrt(3.0 / 8.0) * z * (4.0 * y2 - x2z2)
        sh_3_5 = math.sqrt(5.0) * sh_2_4 * y
        sh_3_6 = math.sqrt(5.0 / 6.0) * (sh_2_4 * z - sh_2_0 * x)

        if lmax == 3:
            return torch.stack([
                sh_1_0, sh_1_1, sh_1_2,
                sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
                sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6
            ], dim=-1)

        sh_4_0 = 0.935414346693485*sh_3_0*z + 0.935414346693485*sh_3_6*x
        sh_4_1 = 0.661437827766148*sh_3_0*y + 0.810092587300982*sh_3_1*z + 0.810092587300983*sh_3_5*x
        sh_4_2 = -0.176776695296637*sh_3_0*z + 0.866025403784439*sh_3_1*y + 0.684653196881458*sh_3_2*z + 0.684653196881457*sh_3_4*x + 0.176776695296637*sh_3_6*x
        sh_4_3 = -0.306186217847897*sh_3_1*z + 0.968245836551855*sh_3_2*y + 0.790569415042095*sh_3_3*x + 0.306186217847897*sh_3_5*x
        sh_4_4 = -0.612372435695795*sh_3_2*x + sh_3_3*y - 0.612372435695795*sh_3_4*z
        sh_4_5 = -0.306186217847897*sh_3_1*x + 0.790569415042096*sh_3_3*z + 0.968245836551854*sh_3_4*y - 0.306186217847897*sh_3_5*z
        sh_4_6 = -0.176776695296637*sh_3_0*x - 0.684653196881457*sh_3_2*x + 0.684653196881457*sh_3_4*z + 0.866025403784439*sh_3_5*y - 0.176776695296637*sh_3_6*z
        sh_4_7 = -0.810092587300982*sh_3_1*x + 0.810092587300982*sh_3_5*z + 0.661437827766148*sh_3_6*y
        sh_4_8 = -0.935414346693485*sh_3_0*x + 0.935414346693486*sh_3_6*z
        if lmax == 4:
            return torch.stack([
                sh_1_0, sh_1_1, sh_1_2,
                sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
                sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
                sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8
            ], dim=-1)

        sh_5_0 = 0.948683298050513*sh_4_0*z + 0.948683298050513*sh_4_8*x
        sh_5_1 = 0.6*sh_4_0*y + 0.848528137423857*sh_4_1*z + 0.848528137423858*sh_4_7*x
        sh_5_2 = -0.14142135623731*sh_4_0*z + 0.8*sh_4_1*y + 0.748331477354788*sh_4_2*z + 0.748331477354788*sh_4_6*x + 0.14142135623731*sh_4_8*x
        sh_5_3 = -0.244948974278318*sh_4_1*z + 0.916515138991168*sh_4_2*y + 0.648074069840786*sh_4_3*z + 0.648074069840787*sh_4_5*x + 0.244948974278318*sh_4_7*x
        sh_5_4 = -0.346410161513776*sh_4_2*z + 0.979795897113272*sh_4_3*y + 0.774596669241484*sh_4_4*x + 0.346410161513776*sh_4_6*x
        sh_5_5 = -0.632455532033676*sh_4_3*x + sh_4_4*y - 0.632455532033676*sh_4_5*z
        sh_5_6 = -0.346410161513776*sh_4_2*x + 0.774596669241483*sh_4_4*z + 0.979795897113273*sh_4_5*y - 0.346410161513776*sh_4_6*z
        sh_5_7 = -0.244948974278318*sh_4_1*x - 0.648074069840787*sh_4_3*x + 0.648074069840786*sh_4_5*z + 0.916515138991169*sh_4_6*y - 0.244948974278318*sh_4_7*z
        sh_5_8 = -0.141421356237309*sh_4_0*x - 0.748331477354788*sh_4_2*x + 0.748331477354788*sh_4_6*z + 0.8*sh_4_7*y - 0.141421356237309*sh_4_8*z
        sh_5_9 = -0.848528137423857*sh_4_1*x + 0.848528137423857*sh_4_7*z + 0.6*sh_4_8*y
        sh_5_10 = -0.948683298050513*sh_4_0*x + 0.948683298050513*sh_4_8*z
        if lmax == 5:
            return torch.stack([
                sh_1_0, sh_1_1, sh_1_2,
                sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
                sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
                sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
                sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10
            ], dim=-1)

        sh_6_0 = 0.957427107756337*sh_5_0*z + 0.957427107756338*sh_5_10*x
        sh_6_1 = 0.552770798392565*sh_5_0*y + 0.874007373475125*sh_5_1*z + 0.874007373475125*sh_5_9*x
        sh_6_2 = -0.117851130197757*sh_5_0*z + 0.745355992499929*sh_5_1*y + 0.117851130197758*sh_5_10*x + 0.790569415042094*sh_5_2*z + 0.790569415042093*sh_5_8*x
        sh_6_3 = -0.204124145231931*sh_5_1*z + 0.866025403784437*sh_5_2*y + 0.707106781186546*sh_5_3*z + 0.707106781186547*sh_5_7*x + 0.204124145231931*sh_5_9*x
        sh_6_4 = -0.288675134594813*sh_5_2*z + 0.942809041582062*sh_5_3*y + 0.623609564462323*sh_5_4*z + 0.623609564462322*sh_5_6*x + 0.288675134594812*sh_5_8*x
        sh_6_5 = -0.372677996249965*sh_5_3*z + 0.986013297183268*sh_5_4*y + 0.763762615825972*sh_5_5*x + 0.372677996249964*sh_5_7*x
        sh_6_6 = -0.645497224367901*sh_5_4*x + sh_5_5*y - 0.645497224367902*sh_5_6*z
        sh_6_7 = -0.372677996249964*sh_5_3*x + 0.763762615825972*sh_5_5*z + 0.986013297183269*sh_5_6*y - 0.372677996249965*sh_5_7*z
        sh_6_8 = -0.288675134594813*sh_5_2*x - 0.623609564462323*sh_5_4*x + 0.623609564462323*sh_5_6*z + 0.942809041582062*sh_5_7*y - 0.288675134594812*sh_5_8*z
        sh_6_9 = -0.20412414523193*sh_5_1*x - 0.707106781186546*sh_5_3*x + 0.707106781186547*sh_5_7*z + 0.866025403784438*sh_5_8*y - 0.204124145231931*sh_5_9*z
        sh_6_10 = -0.117851130197757*sh_5_0*x - 0.117851130197757*sh_5_10*z - 0.790569415042094*sh_5_2*x + 0.790569415042093*sh_5_8*z + 0.745355992499929*sh_5_9*y
        sh_6_11 = -0.874007373475124*sh_5_1*x + 0.552770798392566*sh_5_10*y + 0.874007373475125*sh_5_9*z
        sh_6_12 = -0.957427107756337*sh_5_0*x + 0.957427107756336*sh_5_10*z
        if lmax == 6:
            return torch.stack([
                sh_1_0, sh_1_1, sh_1_2,
                sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
                sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
                sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
                sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10,
                sh_6_0, sh_6_1, sh_6_2, sh_6_3, sh_6_4, sh_6_5, sh_6_6, sh_6_7, sh_6_8, sh_6_9, sh_6_10, sh_6_11, sh_6_12
            ], dim=-1)

        sh_7_0 = 0.963624111659433*sh_6_0*z + 0.963624111659432*sh_6_12*x
        sh_7_1 = 0.515078753637713*sh_6_0*y + 0.892142571199771*sh_6_1*z + 0.892142571199771*sh_6_11*x
        sh_7_2 = -0.101015254455221*sh_6_0*z + 0.699854212223765*sh_6_1*y + 0.82065180664829*sh_6_10*x + 0.101015254455222*sh_6_12*x + 0.82065180664829*sh_6_2*z
        sh_7_3 = -0.174963553055942*sh_6_1*z + 0.174963553055941*sh_6_11*x + 0.82065180664829*sh_6_2*y + 0.749149177264394*sh_6_3*z + 0.749149177264394*sh_6_9*x
        sh_7_4 = 0.247435829652697*sh_6_10*x - 0.247435829652697*sh_6_2*z + 0.903507902905251*sh_6_3*y + 0.677630927178938*sh_6_4*z + 0.677630927178938*sh_6_8*x
        sh_7_5 = -0.31943828249997*sh_6_3*z + 0.95831484749991*sh_6_4*y + 0.606091526731326*sh_6_5*z + 0.606091526731326*sh_6_7*x + 0.31943828249997*sh_6_9*x
        sh_7_6 = -0.391230398217976*sh_6_4*z + 0.989743318610787*sh_6_5*y + 0.755928946018454*sh_6_6*x + 0.391230398217975*sh_6_8*x
        sh_7_7 = -0.654653670707977*sh_6_5*x + sh_6_6*y - 0.654653670707978*sh_6_7*z
        sh_7_8 = -0.391230398217976*sh_6_4*x + 0.755928946018455*sh_6_6*z + 0.989743318610787*sh_6_7*y - 0.391230398217975*sh_6_8*z
        sh_7_9 = -0.31943828249997*sh_6_3*x - 0.606091526731327*sh_6_5*x + 0.606091526731326*sh_6_7*z + 0.95831484749991*sh_6_8*y - 0.31943828249997*sh_6_9*z
        sh_7_10 = -0.247435829652697*sh_6_10*z - 0.247435829652697*sh_6_2*x - 0.677630927178938*sh_6_4*x + 0.677630927178938*sh_6_8*z + 0.903507902905251*sh_6_9*y
        sh_7_11 = -0.174963553055942*sh_6_1*x + 0.820651806648289*sh_6_10*y - 0.174963553055941*sh_6_11*z - 0.749149177264394*sh_6_3*x + 0.749149177264394*sh_6_9*z
        sh_7_12 = -0.101015254455221*sh_6_0*x + 0.82065180664829*sh_6_10*z + 0.699854212223766*sh_6_11*y - 0.101015254455221*sh_6_12*z - 0.82065180664829*sh_6_2*x
        sh_7_13 = -0.892142571199772*sh_6_1*x + 0.892142571199772*sh_6_11*z + 0.515078753637713*sh_6_12*y
        sh_7_14 = -0.963624111659431*sh_6_0*x + 0.963624111659433*sh_6_12*z
        if lmax == 7:
            return torch.stack([
                sh_1_0, sh_1_1, sh_1_2,
                sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
                sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
                sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
                sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10,
                sh_6_0, sh_6_1, sh_6_2, sh_6_3, sh_6_4, sh_6_5, sh_6_6, sh_6_7, sh_6_8, sh_6_9, sh_6_10, sh_6_11, sh_6_12,
                sh_7_0, sh_7_1, sh_7_2, sh_7_3, sh_7_4, sh_7_5, sh_7_6, sh_7_7, sh_7_8, sh_7_9, sh_7_10, sh_7_11, sh_7_12, sh_7_13, sh_7_14
            ], dim=-1)

        sh_8_0 = 0.968245836551854*sh_7_0*z + 0.968245836551853*sh_7_14*x
        sh_8_1 = 0.484122918275928*sh_7_0*y + 0.90571104663684*sh_7_1*z + 0.90571104663684*sh_7_13*x
        sh_8_2 = -0.0883883476483189*sh_7_0*z + 0.661437827766148*sh_7_1*y + 0.843171097702002*sh_7_12*x + 0.088388347648318*sh_7_14*x + 0.843171097702003*sh_7_2*z
        sh_8_3 = -0.153093108923948*sh_7_1*z + 0.7806247497998*sh_7_11*x + 0.153093108923949*sh_7_13*x + 0.7806247497998*sh_7_2*y + 0.780624749799799*sh_7_3*z
        sh_8_4 = 0.718070330817253*sh_7_10*x + 0.21650635094611*sh_7_12*x - 0.21650635094611*sh_7_2*z + 0.866025403784439*sh_7_3*y + 0.718070330817254*sh_7_4*z
        sh_8_5 = 0.279508497187474*sh_7_11*x - 0.279508497187474*sh_7_3*z + 0.927024810886958*sh_7_4*y + 0.655505530106345*sh_7_5*z + 0.655505530106344*sh_7_9*x
        sh_8_6 = 0.342326598440729*sh_7_10*x - 0.342326598440729*sh_7_4*z + 0.968245836551854*sh_7_5*y + 0.592927061281572*sh_7_6*z + 0.592927061281571*sh_7_8*x
        sh_8_7 = -0.405046293650492*sh_7_5*z + 0.992156741649221*sh_7_6*y + 0.75*sh_7_7*x + 0.405046293650492*sh_7_9*x
        sh_8_8 = -0.661437827766148*sh_7_6*x + sh_7_7*y - 0.661437827766148*sh_7_8*z
        sh_8_9 = -0.405046293650492*sh_7_5*x + 0.75*sh_7_7*z + 0.992156741649221*sh_7_8*y - 0.405046293650491*sh_7_9*z
        sh_8_10 = -0.342326598440728*sh_7_10*z - 0.342326598440729*sh_7_4*x - 0.592927061281571*sh_7_6*x + 0.592927061281571*sh_7_8*z + 0.968245836551855*sh_7_9*y
        sh_8_11 = 0.927024810886958*sh_7_10*y - 0.279508497187474*sh_7_11*z - 0.279508497187474*sh_7_3*x - 0.655505530106345*sh_7_5*x + 0.655505530106345*sh_7_9*z
        sh_8_12 = 0.718070330817253*sh_7_10*z + 0.866025403784439*sh_7_11*y - 0.216506350946109*sh_7_12*z - 0.216506350946109*sh_7_2*x - 0.718070330817254*sh_7_4*x
        sh_8_13 = -0.153093108923948*sh_7_1*x + 0.7806247497998*sh_7_11*z + 0.7806247497998*sh_7_12*y - 0.153093108923948*sh_7_13*z - 0.780624749799799*sh_7_3*x
        sh_8_14 = -0.0883883476483179*sh_7_0*x + 0.843171097702002*sh_7_12*z + 0.661437827766147*sh_7_13*y - 0.088388347648319*sh_7_14*z - 0.843171097702002*sh_7_2*x
        sh_8_15 = -0.90571104663684*sh_7_1*x + 0.90571104663684*sh_7_13*z + 0.484122918275927*sh_7_14*y
        sh_8_16 = -0.968245836551853*sh_7_0*x + 0.968245836551855*sh_7_14*z
        if lmax == 8:
            return torch.stack([
                sh_1_0, sh_1_1, sh_1_2,
                sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
                sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
                sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
                sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10,
                sh_6_0, sh_6_1, sh_6_2, sh_6_3, sh_6_4, sh_6_5, sh_6_6, sh_6_7, sh_6_8, sh_6_9, sh_6_10, sh_6_11, sh_6_12,
                sh_7_0, sh_7_1, sh_7_2, sh_7_3, sh_7_4, sh_7_5, sh_7_6, sh_7_7, sh_7_8, sh_7_9, sh_7_10, sh_7_11, sh_7_12, sh_7_13, sh_7_14,
                sh_8_0, sh_8_1, sh_8_2, sh_8_3, sh_8_4, sh_8_5, sh_8_6, sh_8_7, sh_8_8, sh_8_9, sh_8_10, sh_8_11, sh_8_12, sh_8_13, sh_8_14, sh_8_15, sh_8_16
            ], dim=-1)

        sh_9_0 = 0.97182531580755*sh_8_0*z + 0.971825315807551*sh_8_16*x
        sh_9_1 = 0.458122847290851*sh_8_0*y + 0.916245694581702*sh_8_1*z + 0.916245694581702*sh_8_15*x
        sh_9_2 = -0.078567420131839*sh_8_0*z + 0.62853936105471*sh_8_1*y + 0.86066296582387*sh_8_14*x + 0.0785674201318385*sh_8_16*x + 0.860662965823871*sh_8_2*z
        sh_9_3 = -0.136082763487955*sh_8_1*z + 0.805076485899413*sh_8_13*x + 0.136082763487954*sh_8_15*x + 0.74535599249993*sh_8_2*y + 0.805076485899413*sh_8_3*z
        sh_9_4 = 0.749485420179558*sh_8_12*x + 0.192450089729875*sh_8_14*x - 0.192450089729876*sh_8_2*z + 0.831479419283099*sh_8_3*y + 0.749485420179558*sh_8_4*z
        sh_9_5 = 0.693888666488711*sh_8_11*x + 0.248451997499977*sh_8_13*x - 0.248451997499976*sh_8_3*z + 0.895806416477617*sh_8_4*y + 0.69388866648871*sh_8_5*z
        sh_9_6 = 0.638284738504225*sh_8_10*x + 0.304290309725092*sh_8_12*x - 0.304290309725092*sh_8_4*z + 0.942809041582063*sh_8_5*y + 0.638284738504225*sh_8_6*z
        sh_9_7 = 0.360041149911548*sh_8_11*x - 0.360041149911548*sh_8_5*z + 0.974996043043569*sh_8_6*y + 0.582671582316751*sh_8_7*z + 0.582671582316751*sh_8_9*x
        sh_9_8 = 0.415739709641549*sh_8_10*x - 0.415739709641549*sh_8_6*z + 0.993807989999906*sh_8_7*y + 0.74535599249993*sh_8_8*x
        sh_9_9 = -0.66666666666666666667*sh_8_7*x + sh_8_8*y - 0.66666666666666666667*sh_8_9*z
        sh_9_10 = -0.415739709641549*sh_8_10*z - 0.415739709641549*sh_8_6*x + 0.74535599249993*sh_8_8*z + 0.993807989999906*sh_8_9*y
        sh_9_11 = 0.974996043043568*sh_8_10*y - 0.360041149911547*sh_8_11*z - 0.360041149911548*sh_8_5*x - 0.582671582316751*sh_8_7*x + 0.582671582316751*sh_8_9*z
        sh_9_12 = 0.638284738504225*sh_8_10*z + 0.942809041582063*sh_8_11*y - 0.304290309725092*sh_8_12*z - 0.304290309725092*sh_8_4*x - 0.638284738504225*sh_8_6*x
        sh_9_13 = 0.693888666488711*sh_8_11*z + 0.895806416477617*sh_8_12*y - 0.248451997499977*sh_8_13*z - 0.248451997499977*sh_8_3*x - 0.693888666488711*sh_8_5*x
        sh_9_14 = 0.749485420179558*sh_8_12*z + 0.831479419283098*sh_8_13*y - 0.192450089729875*sh_8_14*z - 0.192450089729875*sh_8_2*x - 0.749485420179558*sh_8_4*x
        sh_9_15 = -0.136082763487954*sh_8_1*x + 0.805076485899413*sh_8_13*z + 0.745355992499929*sh_8_14*y - 0.136082763487955*sh_8_15*z - 0.805076485899413*sh_8_3*x
        sh_9_16 = -0.0785674201318389*sh_8_0*x + 0.86066296582387*sh_8_14*z + 0.628539361054709*sh_8_15*y - 0.0785674201318387*sh_8_16*z - 0.860662965823871*sh_8_2*x
        sh_9_17 = -0.9162456945817*sh_8_1*x + 0.916245694581702*sh_8_15*z + 0.458122847290851*sh_8_16*y
        sh_9_18 = -0.97182531580755*sh_8_0*x + 0.97182531580755*sh_8_16*z
        if lmax == 9:
            return torch.stack([
                sh_1_0, sh_1_1, sh_1_2,
                sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
                sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
                sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
                sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10,
                sh_6_0, sh_6_1, sh_6_2, sh_6_3, sh_6_4, sh_6_5, sh_6_6, sh_6_7, sh_6_8, sh_6_9, sh_6_10, sh_6_11, sh_6_12,
                sh_7_0, sh_7_1, sh_7_2, sh_7_3, sh_7_4, sh_7_5, sh_7_6, sh_7_7, sh_7_8, sh_7_9, sh_7_10, sh_7_11, sh_7_12, sh_7_13, sh_7_14,
                sh_8_0, sh_8_1, sh_8_2, sh_8_3, sh_8_4, sh_8_5, sh_8_6, sh_8_7, sh_8_8, sh_8_9, sh_8_10, sh_8_11, sh_8_12, sh_8_13, sh_8_14, sh_8_15, sh_8_16,
                sh_9_0, sh_9_1, sh_9_2, sh_9_3, sh_9_4, sh_9_5, sh_9_6, sh_9_7, sh_9_8, sh_9_9, sh_9_10, sh_9_11, sh_9_12, sh_9_13, sh_9_14, sh_9_15, sh_9_16, sh_9_17, sh_9_18
            ], dim=-1)

        sh_10_0 = 0.974679434480897*sh_9_0*z + 0.974679434480897*sh_9_18*x
        sh_10_1 = 0.435889894354067*sh_9_0*y + 0.924662100445347*sh_9_1*z + 0.924662100445347*sh_9_17*x
        sh_10_2 = -0.0707106781186546*sh_9_0*z + 0.6*sh_9_1*y + 0.874642784226796*sh_9_16*x + 0.070710678118655*sh_9_18*x + 0.874642784226795*sh_9_2*z
        sh_10_3 = -0.122474487139159*sh_9_1*z + 0.824621125123533*sh_9_15*x + 0.122474487139159*sh_9_17*x + 0.714142842854285*sh_9_2*y + 0.824621125123533*sh_9_3*z
        sh_10_4 = 0.774596669241484*sh_9_14*x + 0.173205080756887*sh_9_16*x - 0.173205080756888*sh_9_2*z + 0.8*sh_9_3*y + 0.774596669241483*sh_9_4*z
        sh_10_5 = 0.724568837309472*sh_9_13*x + 0.223606797749979*sh_9_15*x - 0.223606797749979*sh_9_3*z + 0.866025403784438*sh_9_4*y + 0.724568837309472*sh_9_5*z
        sh_10_6 = 0.674536878161602*sh_9_12*x + 0.273861278752583*sh_9_14*x - 0.273861278752583*sh_9_4*z + 0.916515138991168*sh_9_5*y + 0.674536878161602*sh_9_6*z
        sh_10_7 = 0.62449979983984*sh_9_11*x + 0.324037034920393*sh_9_13*x - 0.324037034920393*sh_9_5*z + 0.953939201416946*sh_9_6*y + 0.62449979983984*sh_9_7*z
        sh_10_8 = 0.574456264653803*sh_9_10*x + 0.374165738677394*sh_9_12*x - 0.374165738677394*sh_9_6*z + 0.979795897113272*sh_9_7*y + 0.574456264653803*sh_9_8*z
        sh_10_9 = 0.424264068711928*sh_9_11*x - 0.424264068711929*sh_9_7*z + 0.99498743710662*sh_9_8*y + 0.741619848709567*sh_9_9*x
        sh_10_10 = -0.670820393249937*sh_9_10*z - 0.670820393249937*sh_9_8*x + sh_9_9*y
        sh_10_11 = 0.99498743710662*sh_9_10*y - 0.424264068711929*sh_9_11*z - 0.424264068711929*sh_9_7*x + 0.741619848709567*sh_9_9*z
        sh_10_12 = 0.574456264653803*sh_9_10*z + 0.979795897113272*sh_9_11*y - 0.374165738677395*sh_9_12*z - 0.374165738677394*sh_9_6*x - 0.574456264653803*sh_9_8*x
        sh_10_13 = 0.62449979983984*sh_9_11*z + 0.953939201416946*sh_9_12*y - 0.324037034920393*sh_9_13*z - 0.324037034920393*sh_9_5*x - 0.62449979983984*sh_9_7*x
        sh_10_14 = 0.674536878161602*sh_9_12*z + 0.916515138991168*sh_9_13*y - 0.273861278752583*sh_9_14*z - 0.273861278752583*sh_9_4*x - 0.674536878161603*sh_9_6*x
        sh_10_15 = 0.724568837309472*sh_9_13*z + 0.866025403784439*sh_9_14*y - 0.223606797749979*sh_9_15*z - 0.223606797749979*sh_9_3*x - 0.724568837309472*sh_9_5*x
        sh_10_16 = 0.774596669241484*sh_9_14*z + 0.8*sh_9_15*y - 0.173205080756888*sh_9_16*z - 0.173205080756887*sh_9_2*x - 0.774596669241484*sh_9_4*x
        sh_10_17 = -0.12247448713916*sh_9_1*x + 0.824621125123532*sh_9_15*z + 0.714142842854285*sh_9_16*y - 0.122474487139158*sh_9_17*z - 0.824621125123533*sh_9_3*x
        sh_10_18 = -0.0707106781186548*sh_9_0*x + 0.874642784226796*sh_9_16*z + 0.6*sh_9_17*y - 0.0707106781186546*sh_9_18*z - 0.874642784226796*sh_9_2*x
        sh_10_19 = -0.924662100445348*sh_9_1*x + 0.924662100445347*sh_9_17*z + 0.435889894354068*sh_9_18*y
        sh_10_20 = -0.974679434480898*sh_9_0*x + 0.974679434480896*sh_9_18*z
        if lmax == 10:
            return torch.stack([
                sh_1_0, sh_1_1, sh_1_2,
                sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
                sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
                sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
                sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10,
                sh_6_0, sh_6_1, sh_6_2, sh_6_3, sh_6_4, sh_6_5, sh_6_6, sh_6_7, sh_6_8, sh_6_9, sh_6_10, sh_6_11, sh_6_12,
                sh_7_0, sh_7_1, sh_7_2, sh_7_3, sh_7_4, sh_7_5, sh_7_6, sh_7_7, sh_7_8, sh_7_9, sh_7_10, sh_7_11, sh_7_12, sh_7_13, sh_7_14,
                sh_8_0, sh_8_1, sh_8_2, sh_8_3, sh_8_4, sh_8_5, sh_8_6, sh_8_7, sh_8_8, sh_8_9, sh_8_10, sh_8_11, sh_8_12, sh_8_13, sh_8_14, sh_8_15, sh_8_16,
                sh_9_0, sh_9_1, sh_9_2, sh_9_3, sh_9_4, sh_9_5, sh_9_6, sh_9_7, sh_9_8, sh_9_9, sh_9_10, sh_9_11, sh_9_12, sh_9_13, sh_9_14, sh_9_15, sh_9_16, sh_9_17, sh_9_18,
                sh_10_0, sh_10_1, sh_10_2, sh_10_3, sh_10_4, sh_10_5, sh_10_6, sh_10_7, sh_10_8, sh_10_9, sh_10_10, sh_10_11, sh_10_12, sh_10_13, sh_10_14, sh_10_15, sh_10_16, sh_10_17, sh_10_18, sh_10_19, sh_10_20
            ], dim=-1)

        sh_11_0 = 0.977008420918394*sh_10_0*z + 0.977008420918394*sh_10_20*x
        sh_11_1 = 0.416597790450531*sh_10_0*y + 0.9315409787236*sh_10_1*z + 0.931540978723599*sh_10_19*x
        sh_11_2 = -0.0642824346533223*sh_10_0*z + 0.574959574576069*sh_10_1*y + 0.88607221316445*sh_10_18*x + 0.886072213164452*sh_10_2*z + 0.0642824346533226*sh_10_20*x
        sh_11_3 = -0.111340442853781*sh_10_1*z + 0.84060190949577*sh_10_17*x + 0.111340442853781*sh_10_19*x + 0.686348585024614*sh_10_2*y + 0.840601909495769*sh_10_3*z
        sh_11_4 = 0.795129803842541*sh_10_16*x + 0.157459164324444*sh_10_18*x - 0.157459164324443*sh_10_2*z + 0.771389215839871*sh_10_3*y + 0.795129803842541*sh_10_4*z
        sh_11_5 = 0.74965556829412*sh_10_15*x + 0.203278907045435*sh_10_17*x - 0.203278907045436*sh_10_3*z + 0.838140405208444*sh_10_4*y + 0.74965556829412*sh_10_5*z
        sh_11_6 = 0.70417879021953*sh_10_14*x + 0.248964798865985*sh_10_16*x - 0.248964798865985*sh_10_4*z + 0.890723542830247*sh_10_5*y + 0.704178790219531*sh_10_6*z
        sh_11_7 = 0.658698943008611*sh_10_13*x + 0.294579122654903*sh_10_15*x - 0.294579122654903*sh_10_5*z + 0.9315409787236*sh_10_6*y + 0.658698943008611*sh_10_7*z
        sh_11_8 = 0.613215343783275*sh_10_12*x + 0.340150671524904*sh_10_14*x - 0.340150671524904*sh_10_6*z + 0.962091385841669*sh_10_7*y + 0.613215343783274*sh_10_8*z
        sh_11_9 = 0.567727090763491*sh_10_11*x + 0.385694607919935*sh_10_13*x - 0.385694607919935*sh_10_7*z + 0.983332166035633*sh_10_8*y + 0.56772709076349*sh_10_9*z
        sh_11_10 = 0.738548945875997*sh_10_10*x + 0.431219680932052*sh_10_12*x - 0.431219680932052*sh_10_8*z + 0.995859195463938*sh_10_9*y
        sh_11_11 = sh_10_10*y - 0.674199862463242*sh_10_11*z - 0.674199862463243*sh_10_9*x
        sh_11_12 = 0.738548945875996*sh_10_10*z + 0.995859195463939*sh_10_11*y - 0.431219680932052*sh_10_12*z - 0.431219680932053*sh_10_8*x
        sh_11_13 = 0.567727090763491*sh_10_11*z + 0.983332166035634*sh_10_12*y - 0.385694607919935*sh_10_13*z - 0.385694607919935*sh_10_7*x - 0.567727090763491*sh_10_9*x
        sh_11_14 = 0.613215343783275*sh_10_12*z + 0.96209138584167*sh_10_13*y - 0.340150671524904*sh_10_14*z - 0.340150671524904*sh_10_6*x - 0.613215343783274*sh_10_8*x
        sh_11_15 = 0.658698943008611*sh_10_13*z + 0.9315409787236*sh_10_14*y - 0.294579122654903*sh_10_15*z - 0.294579122654903*sh_10_5*x - 0.65869894300861*sh_10_7*x
        sh_11_16 = 0.70417879021953*sh_10_14*z + 0.890723542830246*sh_10_15*y - 0.248964798865985*sh_10_16*z - 0.248964798865985*sh_10_4*x - 0.70417879021953*sh_10_6*x
        sh_11_17 = 0.749655568294121*sh_10_15*z + 0.838140405208444*sh_10_16*y - 0.203278907045436*sh_10_17*z - 0.203278907045435*sh_10_3*x - 0.749655568294119*sh_10_5*x
        sh_11_18 = 0.79512980384254*sh_10_16*z + 0.77138921583987*sh_10_17*y - 0.157459164324443*sh_10_18*z - 0.157459164324444*sh_10_2*x - 0.795129803842541*sh_10_4*x
        sh_11_19 = -0.111340442853782*sh_10_1*x + 0.84060190949577*sh_10_17*z + 0.686348585024614*sh_10_18*y - 0.111340442853781*sh_10_19*z - 0.840601909495769*sh_10_3*x
        sh_11_20 = -0.0642824346533226*sh_10_0*x + 0.886072213164451*sh_10_18*z + 0.57495957457607*sh_10_19*y - 0.886072213164451*sh_10_2*x - 0.0642824346533228*sh_10_20*z
        sh_11_21 = -0.9315409787236*sh_10_1*x + 0.931540978723599*sh_10_19*z + 0.416597790450531*sh_10_20*y
        sh_11_22 = -0.977008420918393*sh_10_0*x + 0.977008420918393*sh_10_20*z
        return torch.stack([
            sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
            sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
            sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
            sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10,
            sh_6_0, sh_6_1, sh_6_2, sh_6_3, sh_6_4, sh_6_5, sh_6_6, sh_6_7, sh_6_8, sh_6_9, sh_6_10, sh_6_11, sh_6_12,
            sh_7_0, sh_7_1, sh_7_2, sh_7_3, sh_7_4, sh_7_5, sh_7_6, sh_7_7, sh_7_8, sh_7_9, sh_7_10, sh_7_11, sh_7_12, sh_7_13, sh_7_14,
            sh_8_0, sh_8_1, sh_8_2, sh_8_3, sh_8_4, sh_8_5, sh_8_6, sh_8_7, sh_8_8, sh_8_9, sh_8_10, sh_8_11, sh_8_12, sh_8_13, sh_8_14, sh_8_15, sh_8_16,
            sh_9_0, sh_9_1, sh_9_2, sh_9_3, sh_9_4, sh_9_5, sh_9_6, sh_9_7, sh_9_8, sh_9_9, sh_9_10, sh_9_11, sh_9_12, sh_9_13, sh_9_14, sh_9_15, sh_9_16, sh_9_17, sh_9_18,
            sh_10_0, sh_10_1, sh_10_2, sh_10_3, sh_10_4, sh_10_5, sh_10_6, sh_10_7, sh_10_8, sh_10_9, sh_10_10, sh_10_11, sh_10_12, sh_10_13, sh_10_14, sh_10_15, sh_10_16, sh_10_17, sh_10_18, sh_10_19, sh_10_20,
            sh_11_0, sh_11_1, sh_11_2, sh_11_3, sh_11_4, sh_11_5, sh_11_6, sh_11_7, sh_11_8, sh_11_9, sh_11_10, sh_11_11, sh_11_12, sh_11_13, sh_11_14, sh_11_15, sh_11_16, sh_11_17, sh_11_18, sh_11_19, sh_11_20, sh_11_21, sh_11_22
        ], dim=-1)



class VecLayerNorm(torch.nn.Module):
    r"""Applies layer normalization to the input data.

    This module applies a custom layer normalization to a tensor of vectors.
    The normalization can either be :obj:`"max_min"` normalization, or no
    normalization.

    Args:
        hidden_channels (int): The number of hidden channels in the input.
        trainable (bool): If set to :obj:`True`, the normalization weights are
            trainable parameters.
        norm_type (str, optional): The type of normalization to apply, one of
            :obj:`"max_min"` or :obj:`None`. (default: :obj:`"max_min"`)
    """
    def __init__(
        self,
        hidden_channels: int,
        trainable: bool,
        norm_type: Optional[str] = 'max_min',
    ) -> None:
        super().__init__()

        self.hidden_channels = hidden_channels
        self.norm_type = norm_type
        self.eps = 1e-12

        weight = torch.ones(self.hidden_channels)
        if trainable:
            self.register_parameter('weight', Parameter(weight))
        else:
            self.register_buffer('weight', weight)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets the normalization weights to their initial values."""
        torch.nn.init.ones_(self.weight)

    def max_min_norm(self, vec: Tensor) -> Tensor:
        r"""Applies max-min normalization to the input tensor.

        .. math::
            \text{dist} = ||\text{vec}||_2
            \text{direct} = \frac{\text{vec}}{\text{dist}}
            \text{max\_val} = \max(\text{dist})
            \text{min\_val} = \min(\text{dist})
            \text{delta} = \text{max\_val} - \text{min\_val}
            \text{dist} = \frac{\text{dist} - \text{min\_val}}{\text{delta}}
            \text{normed\_vec} = \max(0, \text{dist}) \cdot \text{direct}

        Args:
            vec (torch.Tensor): The input tensor.
        """
        dist = torch.norm(vec, dim=1, keepdim=True)

        if (dist == 0).all():
            return torch.zeros_like(vec)

        dist = dist.clamp(min=self.eps)
        direct = vec / dist

        max_val, _ = dist.max(dim=-1)
        min_val, _ = dist.min(dim=-1)
        delta = (max_val - min_val).view(-1)
        delta = torch.where(delta == 0, torch.ones_like(delta), delta)
        dist = (dist - min_val.view(-1, 1, 1)) / delta.view(-1, 1, 1)

        return dist.relu() * direct

    def forward(self, vec: Tensor) -> Tensor:
        r"""Applies the layer normalization to the input tensor.

        Args:
            vec (torch.Tensor): The input tensor.
        """
        if vec.size(1) == 3:
            if self.norm_type == 'max_min':
                vec = self.max_min_norm(vec)
            return vec * self.weight.unsqueeze(0).unsqueeze(0)
        elif vec.size(1) == 8:
            vec1, vec2 = torch.split(vec, [3, 5], dim=1)
            if self.norm_type == 'max_min':
                vec1 = self.max_min_norm(vec1)
                vec2 = self.max_min_norm(vec2)
            vec = torch.cat([vec1, vec2], dim=1)
            return vec * self.weight.unsqueeze(0).unsqueeze(0)
        elif vec.size(1) == 15:
            vec1, vec2, vec3 = torch.split(vec, [3, 5, 7], dim=1)
            if self.norm_type == 'max_min':
                vec1 = self.max_min_norm(vec1)
                vec2 = self.max_min_norm(vec2)
                vec3 = self.max_min_norm(vec3)
            vec = torch.cat([vec1, vec2, vec3], dim=1)
            return vec * self.weight.unsqueeze(0).unsqueeze(0)
        elif vec.size(1) == 24:
            vec1, vec2, vec3, vec4 = torch.split(vec, [3, 5, 7, 9], dim=1)
            if self.norm_type == 'max_min':
                vec1 = self.max_min_norm(vec1)
                vec2 = self.max_min_norm(vec2)
                vec3 = self.max_min_norm(vec3)
                vec4 = self.max_min_norm(vec4)
            vec = torch.cat([vec1, vec2, vec3, vec4], dim=1)
            return vec * self.weight.unsqueeze(0).unsqueeze(0)
        elif vec.size(1) == 35:
            vec1, vec2, vec3, vec4, vec5 = torch.split(vec, [3, 5, 7, 9, 11], dim=1)
            if self.norm_type == 'max_min':
                vec1 = self.max_min_norm(vec1)
                vec2 = self.max_min_norm(vec2)
                vec3 = self.max_min_norm(vec3)
                vec4 = self.max_min_norm(vec4)
                vec5 = self.max_min_norm(vec5)
            vec = torch.cat([vec1, vec2, vec3, vec4, vec5], dim=1)
            return vec * self.weight.unsqueeze(0).unsqueeze(0)
        elif vec.size(1) == 48:
            vec1, vec2, vec3, vec4, vec5, vec6 = torch.split(vec, [3, 5, 7, 9, 11, 13], dim=1)
            if self.norm_type == 'max_min':
                vec1 = self.max_min_norm(vec1)
                vec2 = self.max_min_norm(vec2)
                vec3 = self.max_min_norm(vec3)
                vec4 = self.max_min_norm(vec4)
                vec5 = self.max_min_norm(vec5)
                vec6 = self.max_min_norm(vec6)
            vec = torch.cat([vec1, vec2, vec3, vec4, vec5, vec6], dim=1)
            return vec * self.weight.unsqueeze(0).unsqueeze(0)
        elif vec.size(1) == 63:
            vec1, vec2, vec3, vec4, vec5, vec6, vec7 = torch.split(vec, [3, 5, 7, 9, 11, 13, 15], dim=1)
            if self.norm_type == 'max_min':
                vec1 = self.max_min_norm(vec1)
                vec2 = self.max_min_norm(vec2)
                vec3 = self.max_min_norm(vec3)
                vec4 = self.max_min_norm(vec4)
                vec5 = self.max_min_norm(vec5)
                vec6 = self.max_min_norm(vec6)
                vec7 = self.max_min_norm(vec7)
            vec = torch.cat([vec1, vec2, vec3, vec4, vec5, vec6, vec7], dim=1)
            return vec * self.weight.unsqueeze(0).unsqueeze(0)
        elif vec.size(1) == 80:
            vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8 = torch.split(vec, [3, 5, 7, 9, 11, 13, 15, 17], dim=1)
            if self.norm_type == 'max_min':
                vec1 = self.max_min_norm(vec1)
                vec2 = self.max_min_norm(vec2)
                vec3 = self.max_min_norm(vec3)
                vec4 = self.max_min_norm(vec4)
                vec5 = self.max_min_norm(vec5)
                vec6 = self.max_min_norm(vec6)
                vec7 = self.max_min_norm(vec7)
                vec8 = self.max_min_norm(vec8)
            vec = torch.cat([vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8], dim=1)
            return vec * self.weight.unsqueeze(0).unsqueeze(0)
        elif vec.size(1) == 99:
            vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8, vec9 = torch.split(vec, [3, 5, 7, 9, 11, 13, 15, 17, 19], dim=1)
            if self.norm_type == 'max_min':
                vec1 = self.max_min_norm(vec1)
                vec2 = self.max_min_norm(vec2)
                vec3 = self.max_min_norm(vec3)
                vec4 = self.max_min_norm(vec4)
                vec5 = self.max_min_norm(vec5)
                vec6 = self.max_min_norm(vec6)
                vec7 = self.max_min_norm(vec7)
                vec8 = self.max_min_norm(vec8)
                vec9 = self.max_min_norm(vec9)
            vec = torch.cat([vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8, vec9], dim=1)
            return vec * self.weight.unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError('Invalid number of channels')



class Distance(torch.nn.Module):
    r"""Computes the pairwise distances between atoms in a molecule.

    This module computes the pairwise distances between atoms in a molecule,
    represented by their positions :obj:`pos`.
    The distances are computed only between points that are within a certain
    cutoff radius.

    Args:
        cutoff (float): The cutoff radius beyond
            which distances are not computed.
        max_num_neighbors (int, optional): The maximum number of neighbors
            considered for each point. (default: :obj:`32`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not
            include self-loops. (default: :obj:`True`)
    """
    def __init__(
        self,
        cutoff: float,
        max_num_neighbors: int = 32,
        add_self_loops: bool = True,
    ) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.add_self_loops = add_self_loops

    def forward(
        self,
        pos: Tensor,
        batch: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Computes the pairwise distances between atoms in the molecule.

        Args:
            pos (torch.Tensor): The positions of the atoms in the molecule.
            batch (torch.Tensor): A batch vector, which assigns each node to a
                specific example.

        Returns:
            edge_index (torch.Tensor): The indices of the edges in the graph.
            edge_weight (torch.Tensor): The distances between connected nodes.
            edge_vec (torch.Tensor): The vector differences between connected
                nodes.
        """
        edge_index = radius_graph(
            pos,
            r=self.cutoff,
            batch=batch,
            loop=self.add_self_loops,
            max_num_neighbors=self.max_num_neighbors,
        )
        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]

        if self.add_self_loops:
            mask = edge_index[0] != edge_index[1]
            edge_weight = torch.zeros(edge_vec.size(0), device=edge_vec.device)
            edge_weight[mask] = torch.norm(edge_vec[mask], dim=-1)
        else:
            edge_weight = torch.norm(edge_vec, dim=-1)

        return edge_index, edge_weight, edge_vec


class NeighborEmbedding(MessagePassing):
    r"""The :class:`NeighborEmbedding` module from the `"Enhancing Geometric
    Representations for Molecules with Equivariant Vector-Scalar Interactive
    Message Passing" <https://arxiv.org/abs/2210.16518>`_ paper.

    Args:
        hidden_channels (int): The number of hidden channels in the node
            embeddings.
        num_rbf (int): The number of radial basis functions.
        cutoff (float): The cutoff distance.
        max_z (int, optional): The maximum atomic numbers.
            (default: :obj:`100`)
    """
    def __init__(
        self,
        hidden_channels: int,
        num_rbf: int,
        cutoff: float,
        max_z: int = 100,
    ) -> None:
        super().__init__(aggr='add')
        self.embedding = Embedding(max_z, hidden_channels)
        self.distance_proj = Linear(num_rbf, hidden_channels)
        self.combine = Linear(hidden_channels * 2, hidden_channels)
        self.cutoff = CosineCutoff(cutoff)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets the parameters of the module."""
        self.embedding.reset_parameters()
        torch.nn.init.xavier_uniform_(self.distance_proj.weight)
        torch.nn.init.xavier_uniform_(self.combine.weight)
        self.distance_proj.bias.data.zero_()
        self.combine.bias.data.zero_()

    def forward(
        self,
        z: Tensor,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        edge_attr: Tensor,
    ) -> Tensor:
        r"""Computes the neighborhood embedding of the nodes in the graph.

        Args:
            z (torch.Tensor): The atomic numbers.
            x (torch.Tensor): The node features.
            edge_index (torch.Tensor): The indices of the edges.
            edge_weight (torch.Tensor): The weights of the edges.
            edge_attr (torch.Tensor): The edge features.

        Returns:
            x_neighbors (torch.Tensor): The neighborhood embeddings of the
                nodes.
        """
        mask = edge_index[0] != edge_index[1]
        if not mask.all():
            edge_index = edge_index[:, mask]
            edge_weight = edge_weight[mask]
            edge_attr = edge_attr[mask]

        C = self.cutoff(edge_weight)
        W = self.distance_proj(edge_attr) * C.view(-1, 1)

        x_neighbors = self.embedding(z)
        x_neighbors = self.propagate(edge_index, x=x_neighbors, W=W)
        x_neighbors = self.combine(torch.cat([x, x_neighbors], dim=1))
        return x_neighbors

    def message(self, x_j: Tensor, W: Tensor) -> Tensor:
        return x_j * W


class EdgeEmbedding(torch.nn.Module):
    r"""The :class:`EdgeEmbedding` module from the `"Enhancing Geometric
    Representations for Molecules with Equivariant Vector-Scalar Interactive
    Message Passing" <https://arxiv.org/abs/2210.16518>`_ paper.

    Args:
        num_rbf (int): The number of radial basis functions.
        hidden_channels (int): The number of hidden channels in the node
            embeddings.
    """
    def __init__(self, num_rbf: int, hidden_channels: int) -> None:
        super().__init__()
        self.edge_proj = Linear(num_rbf, hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets the parameters of the module."""
        torch.nn.init.xavier_uniform_(self.edge_proj.weight)
        self.edge_proj.bias.data.zero_()

    def forward(
        self,
        edge_index: Tensor,
        edge_attr: Tensor,
        x: Tensor,
    ) -> Tensor:
        r"""Computes the edge embeddings of the graph.

        Args:
            edge_index (torch.Tensor): The indices of the edges.
            edge_attr (torch.Tensor): The edge features.
            x (torch.Tensor): The node features.

        Returns:
            out_edge_attr (torch.Tensor): The edge embeddings.
        """
        x_j = x[edge_index[0]]
        x_i = x[edge_index[1]]
        return (x_i + x_j) * self.edge_proj(edge_attr)


class ViS_MP(MessagePassing):
    r"""The message passing module without vertex geometric features of the
    equivariant vector-scalar interactive graph neural network (ViSNet)
    from the `"Enhancing Geometric Representations for Molecules with
    Equivariant Vector-Scalar Interactive Message Passing"
    <https://arxiv.org/abs/2210.16518>`_ paper.

    Args:
        num_heads (int): The number of attention heads.
        hidden_channels (int): The number of hidden channels in the node
            embeddings.
        cutoff (float): The cutoff distance.
        vecnorm_type (str, optional): The type of normalization to apply to the
            vectors.
        trainable_vecnorm (bool): Whether the normalization weights are
            trainable.
        last_layer (bool, optional): Whether this is the last layer in the
            model. (default: :obj:`False`)
    """
    def __init__(
        self,
        num_heads: int,
        hidden_channels: int,
        cutoff: float,
        vecnorm_type: Optional[str],
        trainable_vecnorm: bool,
        last_layer: bool = False,
    ) -> None:
        super().__init__(aggr='add', node_dim=0)

        if hidden_channels % num_heads != 0:
            raise ValueError(
                f"The number of hidden channels (got {hidden_channels}) must "
                f"be evenly divisible by the number of attention heads "
                f"(got {num_heads})")

        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        self.head_dim = hidden_channels // num_heads
        self.last_layer = last_layer

        self.layernorm = LayerNorm(hidden_channels)
        self.vec_layernorm = VecLayerNorm(
            hidden_channels,
            trainable=trainable_vecnorm,
            norm_type=vecnorm_type,
        )

        self.act = torch.nn.SiLU()
        self.attn_activation = torch.nn.SiLU()

        self.cutoff = CosineCutoff(cutoff)

        self.vec_proj = Linear(hidden_channels, hidden_channels * 3, False)

        self.q_proj = Linear(hidden_channels, hidden_channels)
        self.k_proj = Linear(hidden_channels, hidden_channels)
        self.v_proj = Linear(hidden_channels, hidden_channels)
        self.dk_proj = Linear(hidden_channels, hidden_channels)
        self.dv_proj = Linear(hidden_channels, hidden_channels)

        self.s_proj = Linear(hidden_channels, hidden_channels * 2)
        if not self.last_layer:
            self.f_proj = Linear(hidden_channels, hidden_channels)
            self.w_src_proj = Linear(hidden_channels, hidden_channels, False)
            self.w_trg_proj = Linear(hidden_channels, hidden_channels, False)

        self.o_proj = Linear(hidden_channels, hidden_channels * 3)

        self.reset_parameters()

    @staticmethod
    def vector_rejection(vec: Tensor, d_ij: Tensor) -> Tensor:
        r"""Computes the component of :obj:`vec` orthogonal to :obj:`d_ij`.

        Args:
            vec (torch.Tensor): The input vector.
            d_ij (torch.Tensor): The reference vector.
        """
        vec_proj = (vec * d_ij.unsqueeze(2)).sum(dim=1, keepdim=True)
        return vec - vec_proj * d_ij.unsqueeze(2)

    def reset_parameters(self):
        r"""Resets the parameters of the module."""
        self.layernorm.reset_parameters()
        self.vec_layernorm.reset_parameters()
        torch.nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.k_proj.weight)
        self.k_proj.bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.s_proj.weight)
        self.s_proj.bias.data.zero_()

        if not self.last_layer:
            torch.nn.init.xavier_uniform_(self.f_proj.weight)
            self.f_proj.bias.data.zero_()
            torch.nn.init.xavier_uniform_(self.w_src_proj.weight)
            torch.nn.init.xavier_uniform_(self.w_trg_proj.weight)

        torch.nn.init.xavier_uniform_(self.vec_proj.weight)
        torch.nn.init.xavier_uniform_(self.dk_proj.weight)
        self.dk_proj.bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.dv_proj.weight)
        self.dv_proj.bias.data.zero_()

    def forward(
        self,
        x: Tensor,
        vec: Tensor,
        edge_index: Tensor,
        r_ij: Tensor,
        f_ij: Tensor,
        d_ij: Tensor,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        r"""Computes the residual scalar and vector features of the nodes and
        scalar featues of the edges.

        Args:
            x (torch.Tensor): The scalar features of the nodes.
            vec (torch.Tensor):The vector features of the nodes.
            edge_index (torch.Tensor): The indices of the edges.
            r_ij (torch.Tensor): The distances between connected nodes.
            f_ij (torch.Tensor): The scalar features of the edges.
            d_ij (torch.Tensor): The unit vectors of the edges

        Returns:
            dx (torch.Tensor): The residual scalar features of the nodes.
            dvec (torch.Tensor): The residual vector features of the nodes.
            df_ij (torch.Tensor, optional): The residual scalar features of the
                edges, or :obj:`None` if this is the last layer.
        """
        x = self.layernorm(x)
        vec = self.vec_layernorm(vec)

        q = self.q_proj(x).reshape(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(-1, self.num_heads, self.head_dim)
        dk = self.act(self.dk_proj(f_ij))
        dk = dk.reshape(-1, self.num_heads, self.head_dim)
        dv = self.act(self.dv_proj(f_ij))
        dv = dv.reshape(-1, self.num_heads, self.head_dim)

        vec1, vec2, vec3 = torch.split(self.vec_proj(vec),
                                       self.hidden_channels, dim=-1)
        vec_dot = (vec1 * vec2).sum(dim=1)

        x, vec_out = self.propagate(edge_index, q=q, k=k, v=v, dk=dk, dv=dv,
                                    vec=vec, r_ij=r_ij, d_ij=d_ij)

        o1, o2, o3 = torch.split(self.o_proj(x), self.hidden_channels, dim=1)
        dx = vec_dot * o2 + o3
        dvec = vec3 * o1.unsqueeze(1) + vec_out
        if not self.last_layer:
            df_ij = self.edge_updater(edge_index, vec=vec, d_ij=d_ij,
                                      f_ij=f_ij)
            return dx, dvec, df_ij
        else:
            return dx, dvec, None

    def message(self, q_i: Tensor, k_j: Tensor, v_j: Tensor, vec_j: Tensor,
                dk: Tensor, dv: Tensor, r_ij: Tensor,
                d_ij: Tensor) -> Tuple[Tensor, Tensor]:

        attn = (q_i * k_j * dk).sum(dim=-1)
        attn = self.attn_activation(attn) * self.cutoff(r_ij).unsqueeze(1)

        v_j = v_j * dv
        v_j = (v_j * attn.unsqueeze(2)).view(-1, self.hidden_channels)

        s1, s2 = torch.split(self.act(self.s_proj(v_j)), self.hidden_channels,
                             dim=1)
        vec_j = vec_j * s1.unsqueeze(1) + s2.unsqueeze(1) * d_ij.unsqueeze(2)

        return v_j, vec_j

    def edge_update(self, vec_i: Tensor, vec_j: Tensor, d_ij: Tensor,
                    f_ij: Tensor) -> Tensor:

        w1 = self.vector_rejection(self.w_trg_proj(vec_i), d_ij)
        w2 = self.vector_rejection(self.w_src_proj(vec_j), -d_ij)
        w_dot = (w1 * w2).sum(dim=1)
        df_ij = self.act(self.f_proj(f_ij)) * w_dot
        return df_ij

    def aggregate(
        self,
        features: Tuple[Tensor, Tensor],
        index: Tensor,
        ptr: Optional[torch.Tensor],
        dim_size: Optional[int],
    ) -> Tuple[Tensor, Tensor]:
        x, vec = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size)
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
        return x, vec


class ViS_MP_Vertex(ViS_MP):
    r"""The message passing module with vertex geometric features of the
    equivariant vector-scalar interactive graph neural network (ViSNet)
    from the `"Enhancing Geometric Representations for Molecules with
    Equivariant Vector-Scalar Interactive Message Passing"
    <https://arxiv.org/abs/2210.16518>`_ paper.

    Args:
        num_heads (int): The number of attention heads.
        hidden_channels (int): The number of hidden channels in the node
            embeddings.
        cutoff (float): The cutoff distance.
        vecnorm_type (str, optional): The type of normalization to apply to the
            vectors.
        trainable_vecnorm (bool): Whether the normalization weights are
            trainable.
        last_layer (bool, optional): Whether this is the last layer in the
            model. (default: :obj:`False`)
    """
    def __init__(
        self,
        num_heads: int,
        hidden_channels: int,
        cutoff: float,
        vecnorm_type: Optional[str],
        trainable_vecnorm: bool,
        last_layer: bool = False,
    ) -> None:
        super().__init__(num_heads, hidden_channels, cutoff, vecnorm_type,
                         trainable_vecnorm, last_layer)

        if not self.last_layer:
            self.f_proj = Linear(hidden_channels, hidden_channels * 2)
            self.t_src_proj = Linear(hidden_channels, hidden_channels, False)
            self.t_trg_proj = Linear(hidden_channels, hidden_channels, False)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets the parameters of the module."""
        super().reset_parameters()

        if not self.last_layer:
            if hasattr(self, 't_src_proj'):
                torch.nn.init.xavier_uniform_(self.t_src_proj.weight)
            if hasattr(self, 't_trg_proj'):
                torch.nn.init.xavier_uniform_(self.t_trg_proj.weight)

    def edge_update(self, vec_i: Tensor, vec_j: Tensor, d_ij: Tensor,
                    f_ij: Tensor) -> Tensor:

        w1 = self.vector_rejection(self.w_trg_proj(vec_i), d_ij)
        w2 = self.vector_rejection(self.w_src_proj(vec_j), -d_ij)
        w_dot = (w1 * w2).sum(dim=1)

        t1 = self.vector_rejection(self.t_trg_proj(vec_i), d_ij)
        t2 = self.vector_rejection(self.t_src_proj(vec_i), -d_ij)
        t_dot = (t1 * t2).sum(dim=1)

        f1, f2 = torch.split(self.act(self.f_proj(f_ij)), self.hidden_channels,
                             dim=-1)

        return f1 * w_dot + f2 * t_dot


class ViSNetBlock(torch.nn.Module):
    r"""The representation module of the equivariant vector-scalar
    interactive graph neural network (ViSNet) from the `"Enhancing Geometric
    Representations for Molecules with Equivariant Vector-Scalar Interactive
    Message Passing" <https://arxiv.org/abs/2210.16518>`_ paper.

    Args:
        lmax (int, optional): The maximum degree of the spherical harmonics.
            (default: :obj:`1`)
        vecnorm_type (str, optional): The type of normalization to apply to the
            vectors. (default: :obj:`None`)
        trainable_vecnorm (bool, optional):  Whether the normalization weights
            are trainable. (default: :obj:`False`)
        num_heads (int, optional): The number of attention heads.
            (default: :obj:`8`)
        num_layers (int, optional): The number of layers in the network.
            (default: :obj:`6`)
        hidden_channels (int, optional): The number of hidden channels in the
            node embeddings. (default: :obj:`128`)
        num_rbf (int, optional): The number of radial basis functions.
            (default: :obj:`32`)
        trainable_rbf (bool, optional): Whether the radial basis function
            parameters are trainable. (default: :obj:`False`)
        max_z (int, optional): The maximum atomic numbers.
            (default: :obj:`100`)
        cutoff (float, optional): The cutoff distance. (default: :obj:`5.0`)
        max_num_neighbors (int, optional): The maximum number of neighbors
            considered for each atom. (default: :obj:`32`)
        vertex (bool, optional): Whether to use vertex geometric features.
            (default: :obj:`False`)
    """
    def __init__(
        self,
        lmax: int = 1,
        vecnorm_type: Optional[str] = None,
        trainable_vecnorm: bool = False,
        num_heads: int = 8,
        num_layers: int = 6,
        hidden_channels: int = 128,
        num_rbf: int = 32,
        trainable_rbf: bool = False,
        max_z: int = 100,
        cutoff: float = 5.0,
        max_num_neighbors: int = 32,
        vertex: bool = False,
    ) -> None:
        super().__init__()

        self.lmax = lmax
        self.vecnorm_type = vecnorm_type
        self.trainable_vecnorm = trainable_vecnorm
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.num_rbf = num_rbf
        self.trainable_rbf = trainable_rbf
        self.max_z = max_z
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors

        self.embedding = Embedding(max_z, hidden_channels)
        self.distance = Distance(cutoff, max_num_neighbors=max_num_neighbors)
        self.sphere = Sphere(lmax=lmax)
        self.distance_expansion = ExpNormalSmearing(cutoff, num_rbf,
                                                    trainable_rbf)
        self.neighbor_embedding = NeighborEmbedding(hidden_channels, num_rbf,
                                                    cutoff, max_z)
        self.edge_embedding = EdgeEmbedding(num_rbf, hidden_channels)

        self.vis_mp_layers = torch.nn.ModuleList()
        vis_mp_kwargs = dict(
            num_heads=num_heads,
            hidden_channels=hidden_channels,
            cutoff=cutoff,
            vecnorm_type=vecnorm_type,
            trainable_vecnorm=trainable_vecnorm,
        )
        vis_mp_class = ViS_MP if not vertex else ViS_MP_Vertex
        for _ in range(num_layers - 1):
            layer = vis_mp_class(last_layer=False, **vis_mp_kwargs)
            self.vis_mp_layers.append(layer)
        self.vis_mp_layers.append(
            vis_mp_class(last_layer=True, **vis_mp_kwargs))

        self.out_norm = LayerNorm(hidden_channels)
        self.vec_out_norm = VecLayerNorm(
            hidden_channels,
            trainable=trainable_vecnorm,
            norm_type=vecnorm_type,
        )

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets the parameters of the module."""
        self.embedding.reset_parameters()
        self.distance_expansion.reset_parameters()
        self.neighbor_embedding.reset_parameters()
        self.edge_embedding.reset_parameters()
        for layer in self.vis_mp_layers:
            layer.reset_parameters()
        self.out_norm.reset_parameters()
        self.vec_out_norm.reset_parameters()

    def forward(
        self,
        data,
        z: Tensor,
        pos: Tensor,
        batch: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        r"""Computes the scalar and vector features of the nodes.

        Args:
            z (torch.Tensor): The atomic numbers.
            pos (torch.Tensor): The coordinates of the atoms.
            batch (torch.Tensor): A batch vector, which assigns each node to a
                specific example.

        Returns:
            x (torch.Tensor): The scalar features of the nodes.
            vec (torch.Tensor): The vector features of the nodes.
        """
        x = self.embedding(z)
        edge_index, edge_weight, edge_vec = self.distance(pos, batch)
        edge_attr = self.distance_expansion(edge_weight)
        mask = edge_index[0] != edge_index[1]
        edge_vec[mask] = edge_vec[mask] / torch.norm(edge_vec[mask],
                                                     dim=1).unsqueeze(1)
        edge_vec = edge_vec[:, [1,2,0]]
        edge_vec = self.sphere(edge_vec)
        x = self.neighbor_embedding(z, x, edge_index, edge_weight, edge_attr)
        data["node_attr"] = x
        vec = torch.zeros(x.size(0), ((self.lmax + 1)**2) - 1, x.size(1),
                          dtype=x.dtype, device=x.device)
        edge_attr = self.edge_embedding(edge_index, edge_attr, x)

        for attn in self.vis_mp_layers[:-1]:
            dx, dvec, dedge_attr = attn(x, vec, edge_index, edge_weight,
                                        edge_attr, edge_vec)
            x = x + dx
            vec = vec + dvec
            edge_attr = edge_attr + dedge_attr

        dx, dvec, _ = self.vis_mp_layers[-1](x, vec, edge_index, edge_weight,
                                             edge_attr, edge_vec)
        x = x + dx
        vec = vec + dvec

        x = self.out_norm(x)
        vec = self.vec_out_norm(vec)

        return x, vec


class GatedEquivariantBlock(torch.nn.Module):
    r"""Applies a gated equivariant operation to scalar features and vector
    features from the `"Enhancing Geometric Representations for Molecules with
    Equivariant Vector-Scalar Interactive Message Passing"
    <https://arxiv.org/abs/2210.16518>`_ paper.

    Args:
        hidden_channels (int): The number of hidden channels in the node
            embeddings.
        out_channels (int): The number of output channels.
        intermediate_channels (int, optional): The number of channels in the
            intermediate layer, or :obj:`None` to use the same number as
            :obj:`hidden_channels`. (default: :obj:`None`)
        scalar_activation (bool, optional): Whether to apply a scalar
            activation function to the output node features.
            (default: obj:`False`)
    """
    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        intermediate_channels: Optional[int] = None,
        scalar_activation: bool = False,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels

        if intermediate_channels is None:
            intermediate_channels = hidden_channels

        self.vec1_proj = Linear(hidden_channels, hidden_channels, bias=False)
        self.vec2_proj = Linear(hidden_channels, out_channels, bias=False)

        self.update_net = torch.nn.Sequential(
            Linear(hidden_channels * 2, intermediate_channels),
            torch.nn.SiLU(),
            Linear(intermediate_channels, out_channels * 2),
        )

        self.act = torch.nn.SiLU() if scalar_activation else None

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets the parameters of the module."""
        torch.nn.init.xavier_uniform_(self.vec1_proj.weight)
        torch.nn.init.xavier_uniform_(self.vec2_proj.weight)
        torch.nn.init.xavier_uniform_(self.update_net[0].weight)
        self.update_net[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.update_net[2].weight)
        self.update_net[2].bias.data.zero_()

    def forward(self, x: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Applies a gated equivariant operation to node features and vector
        features.

        Args:
            x (torch.Tensor): The scalar features of the nodes.
            v (torch.Tensor): The vector features of the nodes.
        """
        vec1 = torch.norm(self.vec1_proj(v), dim=-2)
        vec2 = self.vec2_proj(v)

        x = torch.cat([x, vec1], dim=-1)
        x, v = torch.split(self.update_net(x), self.out_channels, dim=-1)
        v = v.unsqueeze(1) * vec2

        if self.act is not None:
            x = self.act(x)

        return x, v


class EquivariantScalar(torch.nn.Module):
    r"""Computes final scalar outputs based on node features and vector
    features.

    Args:
        hidden_channels (int): The number of hidden channels in the node
            embeddings.
    """
    def __init__(self, hidden_channels: int) -> None:
        super().__init__()

        self.output_network = torch.nn.ModuleList([
            GatedEquivariantBlock(
                hidden_channels,
                hidden_channels // 2,
                scalar_activation=True,
            ),
            GatedEquivariantBlock(
                hidden_channels // 2,
                1,
                scalar_activation=False,
            ),
        ])

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets the parameters of the module."""
        for layer in self.output_network:
            layer.reset_parameters()

    def pre_reduce(self, x: Tensor, v: Tensor) -> Tensor:
        r"""Computes the final scalar outputs.

        Args:
            x (torch.Tensor): The scalar features of the nodes.
            v (torch.Tensor): The vector features of the nodes.

        Returns:
            out (torch.Tensor): The final scalar outputs of the nodes.
        """
        for layer in self.output_network:
            x, v = layer(x, v)

        return x + v.sum() * 0


class Atomref(torch.nn.Module):
    r"""Adds atom reference values to atomic energies.

    Args:
        atomref (torch.Tensor, optional):  A tensor of atom reference values,
            or :obj:`None` if not provided. (default: :obj:`None`)
        max_z (int, optional): The maximum atomic numbers.
            (default: :obj:`100`)
    """
    def __init__(
        self,
        atomref: Optional[Tensor] = None,
        max_z: int = 100,
    ) -> None:
        super().__init__()

        if atomref is None:
            atomref = torch.zeros(max_z, 1)
        else:
            atomref = torch.as_tensor(atomref)

        if atomref.ndim == 1:
            atomref = atomref.view(-1, 1)

        self.register_buffer('initial_atomref', atomref)
        self.atomref = Embedding(len(atomref), 1)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets the parameters of the module."""
        self.atomref.weight.data.copy_(self.initial_atomref)

    def forward(self, x: Tensor, z: Tensor) -> Tensor:
        r"""Adds atom reference values to atomic energies.

        Args:
            x (torch.Tensor): The atomic energies.
            z (torch.Tensor): The atomic numbers.
        """
        return x + self.atomref(z)


class ViSNet(torch.nn.Module):
    r"""A :pytorch:`PyTorch` module that implements the equivariant
    vector-scalar interactive graph neural network (ViSNet) from the
    `"Enhancing Geometric Representations for Molecules with Equivariant
    Vector-Scalar Interactive Message Passing"
    <https://arxiv.org/abs/2210.16518>`_ paper.

    Args:
        lmax (int, optional): The maximum degree of the spherical harmonics.
            (default: :obj:`1`)
        vecnorm_type (str, optional): The type of normalization to apply to the
            vectors. (default: :obj:`None`)
        trainable_vecnorm (bool, optional):  Whether the normalization weights
            are trainable. (default: :obj:`False`)
        num_heads (int, optional): The number of attention heads.
            (default: :obj:`8`)
        num_layers (int, optional): The number of layers in the network.
            (default: :obj:`6`)
        hidden_channels (int, optional): The number of hidden channels in the
            node embeddings. (default: :obj:`128`)
        num_rbf (int, optional): The number of radial basis functions.
            (default: :obj:`32`)
        trainable_rbf (bool, optional): Whether the radial basis function
            parameters are trainable. (default: :obj:`False`)
        max_z (int, optional): The maximum atomic numbers.
            (default: :obj:`100`)
        cutoff (float, optional): The cutoff distance. (default: :obj:`5.0`)
        max_num_neighbors (int, optional): The maximum number of neighbors
            considered for each atom. (default: :obj:`32`)
        vertex (bool, optional): Whether to use vertex geometric features.
            (default: :obj:`False`)
        atomref (torch.Tensor, optional): A tensor of atom reference values,
            or :obj:`None` if not provided. (default: :obj:`None`)
        reduce_op (str, optional): The type of reduction operation to apply
            (:obj:`"sum"`, :obj:`"mean"`). (default: :obj:`"sum"`)
        mean (float, optional): The mean of the output distribution.
            (default: :obj:`0.0`)
        std (float, optional): The standard deviation of the output
            distribution. (default: :obj:`1.0`)
        derivative (bool, optional): Whether to compute the derivative of the
            output with respect to the positions. (default: :obj:`False`)
    """
    def __init__(
        self,
        order: int = 1,
        vecnorm_type: Optional[str] = None,
        trainable_vecnorm: bool = False,
        num_heads: int = 8,
        num_layers: int = 6,
        embedding_dimension: int = 128,
        num_rbf: int = 32,
        trainable_rbf: bool = False,
        max_z: int = 100,
        max_radius: float = 5.0,
        max_num_neighbors: int = 32,
        vertex: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.representation_model = ViSNetBlock(
            lmax=order,
            vecnorm_type=vecnorm_type,
            trainable_vecnorm=trainable_vecnorm,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_channels=embedding_dimension,
            num_rbf=num_rbf,
            trainable_rbf=trainable_rbf,
            max_z=max_z,
            cutoff=max_radius,
            max_num_neighbors=max_num_neighbors,
            vertex=vertex,
        )
        self.irreps_node_embedding = construct_o3irrps(embedding_dimension, order=order)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets the parameters of the module."""
        self.representation_model.reset_parameters()

    def forward(
        self,
        data,
        **kwargs,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""Computes the energies or properties (forces) for a batch of
        molecules.

        Args:
            z (torch.Tensor): The atomic numbers.
            pos (torch.Tensor): The coordinates of the atoms.
            batch (torch.Tensor): A batch vector,
                which assigns each node to a specific example.

        Returns:
            y (torch.Tensor): The energies or properties for each molecule.
            dy (torch.Tensor, optional): The negative derivative of energies.
        """

        z = data.atomic_numbers.long()
        pos = data.pos
        batch = data.batch
        if z.dim() == 2: # if z of shape num_atoms x 1
            z = z.squeeze() # squeeze to num_atoms

        x, v = self.representation_model(data,z, pos, batch)
        irreps = torch.cat([x.unsqueeze(1), v], dim=1)
        irreps = irreps.reshape(irreps.shape[0], -1)
        data["node_embedding"] = x
        data["node_vec"] = irreps
        data['ptr'] = torch.cat([torch.Tensor([0]).to(data["molecule_size"].device).int(),
                              torch.cumsum(data["molecule_size"],dim = 0)],dim = 0)
        
        data['natoms'] = scatter(torch.ones_like(data.batch), data.batch, dim=0, reduce='sum')


        return data