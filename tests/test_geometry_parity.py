import numpy as np
import pytest
from ase.geometry import find_mic as ase_find_mic
from ase.geometry import get_distances as ase_get_distances

from mojo_ase.geometry import find_mic, get_distances
from mojo_ase.neighborlist import mic


@pytest.mark.parametrize("pbc", [True, [True, False, True]])
def test_find_mic_triclinic(pbc):
    rng = np.random.default_rng(7)
    cell = np.array([[2.0, 0.0, 0.0], [1.2, 1.8, 0.0], [0.4, 0.5, 2.2]])
    vectors = rng.normal(size=(200, 3)) * 8.0
    got, got_lengths = find_mic(vectors, cell, pbc)
    expected, expected_lengths = ase_find_mic(vectors, cell, pbc)
    assert np.allclose(got, expected, atol=1e-13)
    assert np.allclose(got_lengths, expected_lengths, atol=1e-13)
    assert np.allclose(mic(vectors, cell, pbc), expected, atol=1e-13)


def test_find_mic_single_vector_shape():
    got, length = find_mic([3.1, -0.2, 0.0], np.eye(3) * 2.0)
    expected, expected_length = ase_find_mic([3.1, -0.2, 0.0], np.eye(3) * 2.0)
    assert got.shape == (3,)
    assert np.allclose(got, expected)
    assert length == pytest.approx(expected_length)


def test_get_distances_self_matrix():
    rng = np.random.default_rng(9)
    positions = rng.normal(size=(20, 3))
    cell = np.array([[3.0, 0.1, 0.0], [0.5, 3.2, 0.0], [0.2, 0.4, 3.5]])
    got = get_distances(positions, cell=cell, pbc=True)
    expected = ase_get_distances(positions, cell=cell, pbc=True)
    assert np.allclose(got[0], expected[0], atol=1e-13)
    assert np.allclose(got[1], expected[1], atol=1e-13)


def test_get_distances_two_sets_nonperiodic():
    rng = np.random.default_rng(10)
    first = rng.normal(size=(7, 3))
    second = rng.normal(size=(11, 3))
    got = get_distances(first, second)
    expected = ase_get_distances(first, second)
    assert np.allclose(got[0], expected[0])
    assert np.allclose(got[1], expected[1])
