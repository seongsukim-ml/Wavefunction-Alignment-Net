import torch
import sys
sys.path.append('/home/yl2428/MADFT-NN/src')
from madftnn.training.module import OrbitalEnergyError

def test_orbital_energy():
    loss = OrbitalEnergyError(1, basis='def2-tzvp')
    data = torch.load('/home/yl2428/MADFT-NN/test/test_cases/OrbitalEnergyTestCases/batch_data.pth')
    data = loss.cal_loss(data)

if __name__ == "__main__":
    test_orbital_energy()
