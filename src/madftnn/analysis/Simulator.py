import os
import numpy as np

from ase import Atoms, units
from ase.io.trajectory import Trajectory
from ase.md.md import MolecularDynamics
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation

from Calculator import LSRMCalculator


def printenergy(a: Atoms, dyn: MolecularDynamics):
    """
    Function to print the potential, kinetic and total energy
    """
    epot = a.get_potential_energy()
    ekin = a.get_kinetic_energy()
    temp = a.get_temperature()
    print('Step %d: Epot = %.3feV  Ekin = %.3feV (T = %.3fK) '
          'Etot = %.3feV' % (dyn.nsteps, epot, ekin, temp, epot + ekin))

class BaseSimulator(object):

    def __init__(self, atoms: Atoms, log_path: str = None) -> None:
        self.atoms = atoms
        self.log_path = log_path
        self.atoms.set_pbc(False)

    def set_calculator(self, info):
        self.calculator = LSRMCalculator(info['model_path'], device=info['device'])
        self.atoms.calc = self.calculator
    
    def simulate(self, name, simulation_steps, time_step, record_per_steps):
        
        MaxwellBoltzmannDistribution(self.atoms, temperature_K=500, rng=np.random.default_rng(716))
        Stationary(self.atoms)
        ZeroRotation(self.atoms)
        MolDyn = VelocityVerlet(self.atoms, timestep=time_step * units.fs)
        MolDyn.attach(printenergy, interval=record_per_steps, a=self.atoms, dyn=MolDyn)
        traj = Trajectory(os.path.join(self.log_path, f'{name}-traj.traj'), 'w', self.atoms)
        MolDyn.attach(traj.write, interval=record_per_steps)
        MolDyn.run(simulation_steps)
        print("Simulation finished!")