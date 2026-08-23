"""Neighbor-list results are compared tuple-for-tuple with ASE 3.x."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk, molecule
from ase.neighborlist import (
    NewPrimitiveNeighborList as ASENewPrimitiveNeighborList,
    PrimitiveNeighborList as ASEPrimitiveNeighborList,
)
from ase.neighborlist import first_neighbors as ase_first_neighbors
from ase.neighborlist import neighbor_list as ase_neighbor_list
from ase.neighborlist import primitive_neighbor_list as ase_primitive_neighbor_list

from mojo_ase.neighborlist import (
    NewPrimitiveNeighborList,
    PrimitiveNeighborList,
    build_neighbor_list,
    first_neighbors,
    natural_cutoffs,
    neighbor_list,
    primitive_neighbor_list,
)


def canonical(result):
    i, j, d, D, S = result
    if not len(i):
        return i, j, d, D, S
    order = np.lexsort((S[:, 2], S[:, 1], S[:, 0], j, i))
    return i[order], j[order], d[order], D[order], S[order]


def assert_neighbor_parity(got, expected):
    got = canonical(got)
    expected = canonical(expected)
    assert np.array_equal(got[0], expected[0])
    assert np.array_equal(got[1], expected[1])
    assert np.array_equal(got[4], expected[4])
    assert np.allclose(got[2], expected[2], atol=2e-14)
    assert np.allclose(got[3], expected[3], atol=2e-14)


def test_molecule_scalar_cutoff():
    atoms = molecule("CH3CH2OH")
    assert_neighbor_parity(
        neighbor_list("ijdDS", atoms, 1.85),
        ase_neighbor_list("ijdDS", atoms, 1.85),
    )


@pytest.mark.parametrize("n", [1, 2, 7, 8, 9, 15, 16, 17])
@pytest.mark.parametrize("self_interaction", [False, True])
def test_nonperiodic_simd_tail(n, self_interaction):
    rng = np.random.default_rng(91)
    positions = rng.uniform(-1.0, 1.0, size=(n, 3))
    args = ("ijdDS", [False] * 3, np.eye(3), positions, 0.9)
    assert_neighbor_parity(
        primitive_neighbor_list(*args, self_interaction=self_interaction),
        ase_primitive_neighbor_list(*args, self_interaction=self_interaction),
    )


@pytest.mark.parametrize("n", [1, 7, 8, 9, 15, 16, 17])
@pytest.mark.parametrize("self_interaction", [False, True])
def test_periodic_simd_tail(n, self_interaction):
    rng = np.random.default_rng(92)
    positions = rng.uniform(0.0, 8.0, size=(n, 3))
    args = ("ijdDS", [True] * 3, np.eye(3) * 8.0, positions, 1.7)
    assert_neighbor_parity(
        primitive_neighbor_list(*args, self_interaction=self_interaction),
        ase_primitive_neighbor_list(*args, self_interaction=self_interaction),
    )


@pytest.mark.parametrize("n", [60, 61])
def test_parallel_candidate_threshold(n):
    rng = np.random.default_rng(93)
    positions = rng.uniform(0.0, 20.0, size=(n, 3))
    args = ("ijdDS", [True] * 3, np.eye(3) * 20.0, positions, 0.8)
    assert_neighbor_parity(
        primitive_neighbor_list(*args), ase_primitive_neighbor_list(*args)
    )


def test_periodic_bulk_scalar_cutoff():
    atoms = bulk("Cu", cubic=True) * (2, 2, 2)
    assert_neighbor_parity(
        neighbor_list("ijdDS", atoms, 4.2),
        ase_neighbor_list("ijdDS", atoms, 4.2),
    )


@pytest.mark.parametrize("pbc", [[True, True, True], [True, False, True]])
def test_triclinic_outside_cell(pbc):
    rng = np.random.default_rng(12)
    cell = np.array([[4.0, 0.3, 0.2], [0.4, 3.5, 0.1], [0.2, 0.5, 4.2]])
    positions = rng.uniform(-3.0, 7.0, size=(15, 3))
    args = ("ijdDS", pbc, cell, positions, 2.4)
    assert_neighbor_parity(
        primitive_neighbor_list(*args),
        ase_primitive_neighbor_list(*args),
    )


def test_cutoff_larger_than_cell_returns_multiple_images():
    positions = np.array([[0.0, 0.0, 0.0]])
    cell = np.eye(3) * 2.0
    args = ("ijdDS", [True, True, True], cell, positions, 3.1)
    got = primitive_neighbor_list(*args)
    expected = ase_primitive_neighbor_list(*args)
    assert len(got[0]) == 18
    assert_neighbor_parity(got, expected)


def test_per_atom_radii():
    rng = np.random.default_rng(3)
    positions = rng.uniform(0.0, 6.0, size=(18, 3))
    radii = rng.uniform(0.5, 1.4, size=len(positions))
    args = ("ijdDS", [False] * 3, np.eye(3), positions, radii)
    assert_neighbor_parity(
        primitive_neighbor_list(*args),
        ase_primitive_neighbor_list(*args),
    )


def test_pair_dictionary_with_symbols_and_numbers():
    atoms = molecule("CH3CH2OH")
    cutoff = {("H", "H"): 1.1, ("C", "H"): 1.3, (6, 6): 1.85, ("O", "H"): 1.2}
    assert_neighbor_parity(
        neighbor_list("ijdDS", atoms, cutoff),
        ase_neighbor_list("ijdDS", atoms, cutoff),
    )


def test_scaled_positions_and_self_interaction():
    rng = np.random.default_rng(4)
    cell = np.array([[3.0, 0.2, 0.0], [0.4, 3.4, 0.1], [0.2, 0.3, 3.8]])
    scaled = rng.uniform(-0.5, 1.5, size=(10, 3))
    kwargs = dict(self_interaction=True, use_scaled_positions=True)
    args = ("ijdDS", [True] * 3, cell, scaled, 1.7)
    assert_neighbor_parity(
        primitive_neighbor_list(*args, **kwargs),
        ase_primitive_neighbor_list(*args, **kwargs),
    )


def test_empty_and_requested_return_shapes():
    result = primitive_neighbor_list(
        "iDS", [False] * 3, np.zeros((3, 3)), np.empty((0, 3)), 1.0
    )
    assert result[0].shape == (0,)
    assert result[1].shape == (0, 3)
    assert result[2].shape == (0, 3)


def test_invalid_quantity_matches_ase_error():
    with pytest.raises(ValueError, match="Unsupported quantity"):
        primitive_neighbor_list(
            "x", [False] * 3, np.eye(3), np.zeros((1, 3)), 1.0
        )


@pytest.mark.parametrize("cutoff", [np.nan, np.inf, [0.5, np.nan]])
def test_nonfinite_cutoffs_are_rejected(cutoff):
    positions = np.zeros((2, 3))
    with pytest.raises(ValueError, match="finite"):
        primitive_neighbor_list(
            "i", [False] * 3, np.eye(3), positions, cutoff
        )


def test_pair_cutoff_validates_atomic_numbers_before_indexing():
    with pytest.raises(ValueError, match="atomic numbers"):
        primitive_neighbor_list(
            "i",
            [False] * 3,
            np.eye(3),
            np.zeros((2, 3)),
            {(1, 1): 1.0},
            numbers=[1],
        )


def test_natural_cutoffs_matches_ase_values():
    from ase.neighborlist import natural_cutoffs as ase_natural_cutoffs

    atoms = molecule("H2O")
    assert natural_cutoffs(atoms, mult=1.2, H=0.8) == pytest.approx(
        ase_natural_cutoffs(atoms, mult=1.2, H=0.8)
    )


def test_first_neighbors_matches_ase():
    first = np.array([0, 0, 0, 2, 2, 5, 5, 5, 5])
    assert np.array_equal(first_neighbors(7, first), ase_first_neighbors(7, first))
    assert np.array_equal(first_neighbors(4, []), ase_first_neighbors(4, []))


@pytest.mark.parametrize("bothways", [False, True])
def test_new_primitive_neighbor_list(bothways):
    atoms = bulk("Cu", cubic=True)
    cutoffs = np.full(len(atoms), 1.45)
    ours = NewPrimitiveNeighborList(
        cutoffs, skin=0.2, self_interaction=False, bothways=bothways, sorted=True
    )
    theirs = ASENewPrimitiveNeighborList(
        cutoffs, skin=0.2, self_interaction=False, bothways=bothways, sorted=True
    )
    assert ours.update(atoms.pbc, atoms.cell, atoms.positions)
    assert theirs.update(atoms.pbc, atoms.cell, atoms.positions)
    for atom in range(len(atoms)):
        oi, os = ours.get_neighbors(atom)
        ti, ts = theirs.get_neighbors(atom)
        ours_rows = np.column_stack((oi, os))
        their_rows = np.column_stack((ti, ts))
        ours_order = np.lexsort(ours_rows.T[::-1])
        their_order = np.lexsort(their_rows.T[::-1])
        assert np.array_equal(ours_rows[ours_order], their_rows[their_order])


def test_primitive_neighbor_list_class():
    atoms = molecule("H2O")
    cutoffs = np.array([0.8, 0.3, 0.3])
    ours = PrimitiveNeighborList(cutoffs, skin=0.0, bothways=True)
    theirs = ASEPrimitiveNeighborList(cutoffs, skin=0.0, bothways=True)
    ours.update(atoms.pbc, atoms.cell.complete(), atoms.positions)
    theirs.update(atoms.pbc, atoms.cell.complete(), atoms.positions)
    for atom in range(len(atoms)):
        got = ours.get_neighbors(atom)
        expected = theirs.get_neighbors(atom)
        got_rows = np.column_stack(got)
        expected_rows = np.column_stack(expected)
        assert sorted(map(tuple, got_rows)) == sorted(map(tuple, expected_rows))


def test_neighbor_list_skin_cache_contract():
    atoms = molecule("H2O")
    cutoffs = np.ones(len(atoms))
    nl = NewPrimitiveNeighborList(cutoffs, skin=0.3)
    assert nl.update(atoms.pbc, atoms.cell.complete(), atoms.positions)
    moved = atoms.positions.copy()
    moved[0, 0] += 0.1
    assert not nl.update(atoms.pbc, atoms.cell.complete(), moved)
    moved[0, 0] += 0.3
    assert nl.update(atoms.pbc, atoms.cell.complete(), moved)


def test_neighbor_list_wrapper_connectivity():
    atoms = molecule("H2O")
    nl = build_neighbor_list(
        atoms,
        cutoffs=np.array([0.8, 0.3, 0.3]),
        skin=0.0,
        self_interaction=False,
        bothways=True,
    )
    dense = nl.get_connectivity_matrix(sparse=False)
    assert dense.shape == (3, 3)
    assert np.array_equal(dense, dense.T)
    assert dense.sum() == 4
    sparse = nl.get_connectivity_matrix(sparse=True)
    assert np.array_equal(sparse.toarray(), dense)
    assert nl.nupdates == 1
    assert np.array_equal(nl.get_neighbors(0)[0], np.array([1, 2]))
