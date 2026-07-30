"""ASE-compatible Morse pair-potential calculator."""

from __future__ import annotations

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress

from .._lib import addr, lib
from ..neighborlist import neighbor_list as mojo_neighbor_list


def fcut(r: np.ndarray, r0: float, r1: float) -> np.ndarray:
    s = 1.0 - (r - r0) / (r1 - r0)
    return (s >= 1.0) + ((s > 0.0) & (s < 1.0)) * (
        6.0 * s**5 - 15.0 * s**4 + 10.0 * s**3
    )


def fcut_d(r: np.ndarray, r0: float, r1: float) -> np.ndarray:
    s = 1.0 - (r - r0) / (r1 - r0)
    return -(
        ((s > 0.0) & (s < 1.0))
        * (30.0 * s**4 - 60.0 * s**3 + 30.0 * s**2)
        / (r1 - r0)
    )


class MorsePotential(Calculator):
    implemented_properties = ["energy", "energies", "free_energy", "forces", "stress"]
    default_parameters = {
        "epsilon": 1.0,
        "rho0": 6.0,
        "r0": 1.0,
        "rcut1": 1.9,
        "rcut2": 2.7,
    }
    nolabel = True

    def __init__(self, neighbor_list=mojo_neighbor_list, **kwargs):
        self.neighbor_list = neighbor_list
        Calculator.__init__(self, **kwargs)

    def calculate(
        self, atoms=None, properties=["energy"], system_changes=all_changes
    ):
        Calculator.calculate(self, atoms, properties, system_changes)
        epsilon = float(self.parameters.epsilon)
        rho0 = float(self.parameters.rho0)
        r0 = float(self.parameters.r0)
        rcut1 = float(self.parameters.rcut1) * r0
        rcut2 = float(self.parameters.rcut2) * r0
        natoms = len(self.atoms)

        pair_i, distances, vectors = self.neighbor_list(
            "idD", self.atoms, rcut2
        )
        energies = np.zeros(natoms)
        forces = np.zeros((natoms, 3))
        stress = np.zeros((3, 3))
        if natoms and len(pair_i):
            lib().mase_morse(
                addr(pair_i),
                addr(distances),
                addr(vectors),
                len(pair_i),
                natoms,
                epsilon,
                rho0,
                r0,
                rcut1,
                rcut2,
                addr(energies),
                addr(forces),
                addr(stress),
            )

        energy = energies.sum()
        self.results["energy"] = energy
        self.results["free_energy"] = energy
        self.results["energies"] = energies
        self.results["forces"] = forces
        if self.atoms.cell.rank == 3:
            self.results["stress"] = full_3x3_to_voigt_6_stress(
                stress / self.atoms.get_volume()
            )
