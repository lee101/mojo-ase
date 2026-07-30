"""Geometry functions matching the covered part of :mod:`ase.geometry`."""

from __future__ import annotations

import numpy as np
from ase.cell import Cell
from ase.geometry import complete_cell, minkowski_reduce
from ase.utils import pbc2pbc

from ._lib import addr, f64, lib


def find_mic(v, cell, pbc=True):
    """Return minimum-image vectors and their lengths."""
    ase_cell = Cell(cell)
    periodic = ase_cell.any(1) & pbc2pbc(pbc)
    values = np.asarray(v, dtype=np.float64)
    single = values.ndim == 1
    vectors = f64(np.atleast_2d(values))

    if not periodic.any():
        result = vectors.copy()
        lengths = np.linalg.norm(result, axis=1)
    else:
        reduced, _ = minkowski_reduce(ase_cell, pbc=periodic)
        reduced = f64(complete_cell(reduced))
        inverse = f64(np.linalg.inv(reduced))
        result = np.empty_like(vectors)
        lengths = np.empty(len(vectors), dtype=np.float64)
        if len(vectors):
            px, py, pz = periodic.astype(np.int64)
            lib().mase_minimum_image(
                addr(vectors),
                addr(reduced),
                addr(inverse),
                addr(result),
                addr(lengths),
                len(vectors),
                int(px),
                int(py),
                int(pz),
            )

    if single:
        return result[0], lengths[0]
    return result, lengths


def get_distances(p1, p2=None, cell=None, pbc=None):
    """Return distance vectors and lengths, with ASE-compatible shapes."""
    p1 = np.atleast_2d(p1)
    if p2 is None:
        n1 = len(p1)
        ind1, ind2 = np.triu_indices(n1, k=1)
        vectors = p1[ind2] - p1[ind1]
    else:
        p2 = np.atleast_2d(p2)
        vectors = (p2[np.newaxis, :, :] - p1[:, np.newaxis, :]).reshape((-1, 3))

    if (cell is None) != (pbc is None):
        raise ValueError("cell or pbc must be both set or both be None")
    if cell is None:
        vectors = np.asarray(vectors, dtype=np.float64)
        lengths = np.linalg.norm(vectors, axis=1)
    else:
        vectors, lengths = find_mic(vectors, cell, pbc)

    if p2 is None:
        vector_matrix = np.zeros((n1, n1, 3))
        vector_matrix[(ind1, ind2)] = vectors
        vector_matrix -= vector_matrix.transpose(1, 0, 2)
        distance_matrix = np.zeros((n1, n1))
        distance_matrix[(ind1, ind2)] = lengths
        distance_matrix += distance_matrix.T
        return vector_matrix, distance_matrix

    return (
        vectors.reshape((-1, len(p2), 3)),
        lengths.reshape((-1, len(p2))),
    )
