import sys
import torch
sys.path.append('/home/yl2428/MADFT-NN/src')
from madftnn.models.spherical_visnet import ViSNet

from torch_scatter import scatter


if __name__ == "__main__":
    model = ViSNet(order = 4).cpu()
    data = torch.load('/home/yl2428/MADFT-NN/test/test_cases/ConvNetLayerTestCases/data.pth').cpu()
    data = model(data)
    print(data["node_vec"].shape)
    