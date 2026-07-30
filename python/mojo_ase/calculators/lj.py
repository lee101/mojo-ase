"""ASE-compatible Lennard-Jones calculator."""

from __future__ import annotations

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress

from .._lib import addr, lib
from ..neighborlist import neighbor_list


class LennardJones(Calculator):
    implemented_properties = [
        "energy",
        "energies",
        "forces",
        "free_energy",
        "stress",
        "stresses",
    ]
    default_parameters = {
        "epsilon": 1.0,
        "sigma": 1.0,
        "rc": None,
        "ro": None,
        "smooth": False,
    }
    nolabel = True

    def __init__(self, **kwargs):
        Calculator.__init__(self, **kwargs)
        if self.parameters.rc is None:
            self.parameters.rc = 3 * self.parameters.sigma
        if self.parameters.ro is None:
            self.parameters.ro = 0.66 * self.parameters.rc

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        if properties is None:
            properties = self.implemented_properties
        Calculator.calculate(self, atoms, properties, system_changes)

        natoms = len(self.atoms)
        sigma = float(self.parameters.sigma)
        epsilon = float(self.parameters.epsilon)
        rc = float(self.parameters.rc)
        ro = float(self.parameters.ro)
        pair_i, vectors = neighbor_list(
            "iD", self.atoms, np.nextafter(rc, np.inf)
        )
        energies = np.zeros(natoms)
        forces = np.zeros((natoms, 3))
        stresses = np.zeros((natoms, 3, 3))
        if natoms and len(pair_i):
            lib().mase_lennard_jones(
                addr(pair_i),
                addr(vectors),
                len(pair_i),
                natoms,
                sigma,
                epsilon,
                rc,
                ro,
                int(bool(self.parameters.smooth)),
                addr(energies),
                addr(forces),
                addr(stresses),
            )

        energy = energies.sum()
        self.results["energy"] = energy
        self.results["free_energy"] = energy
        self.results["energies"] = energies
        self.results["forces"] = forces
        if self.atoms.cell.rank == 3:
            voigt = full_3x3_to_voigt_6_stress(stresses)
            self.results["stress"] = voigt.sum(axis=0) / self.atoms.get_volume()
            self.results["stresses"] = voigt / self.atoms.get_volume()


def cutoff_function(r, rc, ro):
    r = np.asarray(r)
    return np.where(
        r < ro,
        1.0,
        np.where(
            r < rc,
            (rc - r) ** 2 * (rc + 2 * r - 3 * ro) / (rc - ro) ** 3,
            0.0,
        ),
    )


def d_cutoff_function(r, rc, ro):
    r = np.asarray(r)
    return np.where(
        r < ro,
        0.0,
        np.where(r < rc, 6 * (rc - r) * (ro - r) / (rc - ro) ** 3, 0.0),
    )
