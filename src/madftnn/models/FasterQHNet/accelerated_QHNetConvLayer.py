import time

import torch.nn.functional as F
import sys    
sys.path.append('/home/yl2428/MADFT-NN/src') 
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch_cluster import radius_graph
from e3nn import o3
from torch_scatter import scatter
from e3nn.nn import FullyConnectedNet, Gate, Activation
from e3nn.o3 import Linear, TensorProduct, FullyConnectedTensorProduct
from madftnn.models.equiformer.graph_attention_transformer import (
    get_norm_layer,
    LinearRS,
    FullyConnectedTensorProductRescaleNorm, 
    FullyConnectedTensorProductRescaleNormSwishGate, 
    FullyConnectedTensorProductRescaleSwishGate, 
    DepthwiseTensorProduct,
    SeparableFCTP,
    Vec2AttnHeads, 
    AttnHeads2Vec,
    FeedForwardNetwork, 
    NodeEmbeddingNetwork, 
    ScaledScatter, 
    EdgeDegreeEmbeddingNetwork)
from madftnn.models.QHNet import get_feasible_irrep, get_nonlinear, NormGate, InnerProduct, ConvLayer
import numpy as np
import warnings
from e3nn.util.jit import compile_mode, compile
from madftnn.models.equiformer_v2.equiformer_v2_oc20 import TransBlockV2



class ConvLayerAcceleratedV1(torch.nn.Module):
    def __init__(
            self,
            irrep_in_node,
            irrep_hidden,
            irrep_out,
            sh_irrep,
            edge_attr_dim,
            node_attr_dim,
            invariant_layers=1,
            invariant_neurons=32,
            avg_num_neighbors=None,
            nonlinear='ssp',
            use_norm_gate=True,
            edge_wise=False,
    ):
        super(ConvLayerAcceleratedV1, self).__init__()
        self.avg_num_neighbors = avg_num_neighbors
        self.edge_attr_dim = edge_attr_dim
        self.node_attr_dim = node_attr_dim
        self.edge_wise = edge_wise

        self.irrep_in_node = irrep_in_node if isinstance(irrep_in_node, o3.Irreps) else o3.Irreps(irrep_in_node)
        self.irrep_hidden = irrep_hidden \
            if isinstance(irrep_hidden, o3.Irreps) else o3.Irreps(irrep_hidden)
        self.irrep_out = irrep_out if isinstance(irrep_out, o3.Irreps) else o3.Irreps(irrep_out)
        self.sh_irrep = sh_irrep if isinstance(sh_irrep, o3.Irreps) else o3.Irreps(sh_irrep)
        self.nonlinear_layer = get_nonlinear(nonlinear)

        self.irrep_tp_out_node, instruction_node = get_feasible_irrep(
            self.irrep_in_node, self.sh_irrep, self.irrep_hidden, tp_mode='uvu')
        self.tp_node = DepthwiseTensorProduct(self.irrep_in_node, self.sh_irrep, self.irrep_tp_out_node)

        self.fc_node = FullyConnectedNet(
            [self.edge_attr_dim] + invariant_layers * [invariant_neurons] + [self.tp_node.tp.weight_numel],
            self.nonlinear_layer
        )

        num_mul = 0
        for mul, ir in self.irrep_in_node:
            num_mul = num_mul + mul

        self.layer_l0 = FullyConnectedNet(
            [num_mul + self.irrep_in_node[0][0]] + invariant_layers * [invariant_neurons] + [self.tp_node.tp.weight_numel],
            self.nonlinear_layer
        )

        self.linear_out = LinearRS(
            irreps_in=self.irrep_tp_out_node,
            irreps_out=self.irrep_out,
        )

        self.use_norm_gate = use_norm_gate
        self.norm_gate = NormGate(self.irrep_in_node)
        self.irrep_linear_out, instruction_node = get_feasible_irrep(
            self.irrep_in_node, o3.Irreps("0e"), self.irrep_in_node)
        self.linear_node = LinearRS(
            irreps_in=self.irrep_in_node,
            irreps_out=self.irrep_linear_out,
        )
        self.linear_node_pre = LinearRS(
            irreps_in=self.irrep_in_node,
            irreps_out=self.irrep_linear_out,
        )
        self.inner_product = InnerProduct(self.irrep_in_node)
    
    @torch.compile(dynamic=True)
    def forward(self, data, x):
        edge_dst, edge_src = data.edge_index[0], data.edge_index[1]

        if self.use_norm_gate:
            pre_x = self.linear_node_pre(x)
            s0 = self.inner_product(pre_x[edge_dst], pre_x[edge_src])[:, self.irrep_in_node.slices()[0].stop:]
            s0 = torch.cat([pre_x[edge_dst][:, self.irrep_in_node.slices()[0]],
                            pre_x[edge_src][:, self.irrep_in_node.slices()[0]], s0], dim=-1)
            x = self.norm_gate(x)
            x = self.linear_node(x)
        else:
            s0 = self.inner_product(x[edge_dst], x[edge_src])[:, self.irrep_in_node.slices()[0].stop:]
            s0 = torch.cat([x[edge_dst][:, self.irrep_in_node.slices()[0]],
                            x[edge_src][:, self.irrep_in_node.slices()[0]], s0], dim=-1)

        self_x = x

        edge_features = self.tp_node(
            x[edge_src], data.edge_sh, self.fc_node(data.edge_attr) * self.layer_l0(s0))

        if self.edge_wise:
            out = edge_features
        else:
            out = scatter(edge_features, edge_dst, dim=0, dim_size=len(x))

        if self.irrep_in_node == self.irrep_out:
            out = out + self_x

        out = self.linear_out(out)
        return out




if __name__ == '__main__':   
    import torch
    import time

    # Assuming ConvLayerAcceleratedV1 and ConvLayer are defined elsewhere
    # Load data and models
    kwargs = torch.load('/home/yl2428/MADFT-NN/test/test_cases/ConvNetLayerTestCases/args.pth')
    data = torch.load('/home/yl2428/MADFT-NN/test/test_cases/ConvNetLayerTestCases/data.pth').to('cuda:2')
    x = torch.load('/home/yl2428/MADFT-NN/test/test_cases/ConvNetLayerTestCases/x.pth').to('cuda:2')

    # model_1 = ConvLayerAcceleratedV1(**kwargs).to('cuda:2')
    # model_2 = ConvLayer(**kwargs).to('cuda:2')
    # model_2 = torch.compile(model_2)
    model = SO2EquivariantGraphAttention(128, 128, 8, 32, 16, 128, 6, 2)


    # Print number of parameters
    print(f"Model 1 Parameters: {sum(p.numel() for p in model_1.parameters())}")
    print(f"Model 2 Parameters: {sum(p.numel() for p in model_2.parameters())}")

    # Function to measure model inference time with warm-up steps
    def measure_time(model, data, x, iterations=10000, warm_up_iterations=1000):
        torch.cuda.synchronize()  # Synchronize to ensure accurate timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Warm-up runs: Execute a few forward passes before the actual measurement to warm up the GPU
        with torch.no_grad():
            for _ in range(warm_up_iterations):
                _ = model(data, x)

        # Measurement runs
        start_event.record()
        with torch.no_grad():
            for _ in range(iterations):
                _ = model(data, x)
        end_event.record()
        torch.cuda.synchronize()  # Synchronize to wait for the last operation to complete

        # Calculate and return average time per iteration
        total_time_ms = start_event.elapsed_time(end_event)
        return total_time_ms / iterations

    # Measure and print inference times with warm-up included
    avg_time_1 = measure_time(model_1, data, x)
    avg_time_2 = measure_time(model_2, data, x)
    print(f"Average Inference Time for Model 1: {avg_time_1:.3f} ms")
    print(f"Average Inference Time for Model 2: {avg_time_2:.3f} ms")
