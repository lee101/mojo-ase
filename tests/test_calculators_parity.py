import numpy as np
import pytest
from ase import Atoms
from ase.calculators.lj import LennardJones as ASELennardJones
from ase.calculators.lj import cutoff_function as ase_cutoff_function
from ase.calculators.morse import MorsePotential as ASEMorsePotential
from ase.calculators.morse import fcut as ase_fcut
from ase.calculators.morse import fcut_d as ase_fcut_d

from mojo_ase.calculators.lj import (
    LennardJones,
    cutoff_function,
    d_cutoff_function,
)
from ase.calculators.lj import d_cutoff_function as ase_d_cutoff_function
from mojo_ase.calculators.morse import MorsePotential, fcut, fcut_d


@pytest.fixture
def dense_atoms():
    rng = np.random.default_rng(5)
    axis = np.arange(3) * 1.15
    positions = np.array(np.meshgrid(axis, axis, axis, indexing="ij")).reshape(3, -1).T
    positions += rng.normal(0.0, 0.03, positions.shape)
    return Atoms("Ar" * len(positions), positions, cell=[8.0] * 3, pbc=True)


def compare_calculators(atoms, ours, theirs, *, atomic_stress=False):
    left = atoms.copy()
    right = atoms.copy()
    left.calc = ours
    right.calc = theirs
    assert left.get_potential_energy() == pytest.approx(
        right.get_potential_energy(), rel=5e-13, abs=5e-11
    )
    assert np.allclose(left.get_forces(), right.get_forces(), atol=3e-11)
    assert np.allclose(left.get_stress(), right.get_stress(), atol=3e-12)
    assert np.allclose(
        left.get_potential_energies(), right.get_potential_energies(), atol=3e-12
    )
    if atomic_stress:
        assert np.allclose(left.get_stresses(), right.get_stresses(), atol=3e-12)


def test_lennard_jones_shifted(dense_atoms):
    compare_calculators(
        dense_atoms, LennardJones(), ASELennardJones(), atomic_stress=True
    )


def test_lennard_jones_smooth_custom_parameters(dense_atoms):
    kwargs = dict(epsilon=0.7, sigma=1.05, rc=3.1, ro=1.9, smooth=True)
    compare_calculators(
        dense_atoms,
        LennardJones(**kwargs),
        ASELennardJones(**kwargs),
        atomic_stress=True,
    )


def test_lennard_jones_nonperiodic_has_no_stress(dense_atoms):
    atoms = dense_atoms.copy()
    atoms.pbc = False
    atoms.cell = np.zeros((3, 3))
    atoms.calc = LennardJones()
    atoms.get_forces()
    assert "stress" not in atoms.calc.results
    reference = atoms.copy()
    reference.calc = ASELennardJones()
    assert atoms.get_potential_energy() == pytest.approx(reference.get_potential_energy())
    assert np.allclose(atoms.get_forces(), reference.get_forces(), atol=3e-11)


def test_morse_default(dense_atoms):
    compare_calculators(dense_atoms, MorsePotential(), ASEMorsePotential())


def test_morse_custom_parameters(dense_atoms):
    kwargs = dict(epsilon=0.8, rho0=5.2, r0=1.1, rcut1=1.7, rcut2=2.5)
    compare_calculators(
        dense_atoms, MorsePotential(**kwargs), ASEMorsePotential(**kwargs)
    )


def test_public_cutoff_helpers_match_ase():
    values = np.linspace(0.2, 10.0, 101)
    assert np.allclose(
        cutoff_function(values, 9.0, 4.0),
        ase_cutoff_function(values, 9.0, 4.0),
    )
    assert np.allclose(
        d_cutoff_function(values, 9.0, 4.0),
        ase_d_cutoff_function(values, 9.0, 4.0),
    )
    distances = np.linspace(0.2, 3.0, 101)
    assert np.allclose(fcut(distances, 1.9, 2.7), ase_fcut(distances, 1.9, 2.7))
    assert np.allclose(
        fcut_d(distances, 1.9, 2.7), ase_fcut_d(distances, 1.9, 2.7)
    )
