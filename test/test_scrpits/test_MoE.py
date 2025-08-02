import torch
import sys
sys.path.append('/home/yl2428/MADFT-NN/src')
from madftnn.models.moe_utils import MoEPairLayer
from madftnn.models.utils import  construct_o3irrps_base,construct_o3irrps
from e3nn import o3

def test_moe_pair():
    hs = 128
    order = 4
    num_fc_layer = 2
    irreps_node_embedding = construct_o3irrps_base(hs, order=order)
    hidden_irrep_base = o3.Irreps(irreps_node_embedding)
    sh_irrep = o3.Irreps.spherical_harmonics(lmax=order)
    radius_embed_dim = 16
    moe_pair = MoEPairLayer(4, 16, True, 2,                         
                       irrep_in_node=hidden_irrep_base,
                        irrep_bottle_hidden=hidden_irrep_base,
                        irrep_out=hidden_irrep_base,
                        sh_irrep=sh_irrep,
                        edge_attr_dim=radius_embed_dim,
                        node_attr_dim=hs,
                        invariant_layers=num_fc_layer,
                        invariant_neurons=hs,
                        resnet=True,).cuda()
    data = torch.load('/home/yl2428/MADFT-NN/test/test_cases/PairNetLayerTestCases/data.pth')
    node_attr = torch.load('/home/yl2428/MADFT-NN/test/test_cases/PairNetLayerTestCases/node_attr.pth')
    data = moe_pair(data, node_attr)

if __name__ == "__main__":
    test_moe_pair()