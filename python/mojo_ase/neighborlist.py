"""Neighbor-list API compatible with the covered subset of ASE."""

from __future__ import annotations

import math

import numpy as np
from ase.data import atomic_numbers, covalent_radii
from ase.geometry import complete_cell

from ._lib import addr, f64, i64, lib
from .geometry import find_mic


def natural_cutoffs(atoms, mult=1, **kwargs):
    return [
        kwargs.get(atom.symbol, covalent_radii[atom.number] * mult)
        for atom in atoms
    ]


def build_neighbor_list(atoms, cutoffs=None, **kwargs):
    if cutoffs is None:
        cutoffs = natural_cutoffs(atoms)
    result = NeighborList(cutoffs, **kwargs)
    result.update(atoms)
    return result


def mic(dr, cell, pbc=True):
    return find_mic(dr, cell, pbc)[0]


def _empty_result(quantities: str):
    types = {
        "i": (int, (0,)),
        "j": (int, (0,)),
        "D": (float, (0, 3)),
        "d": (float, (0,)),
        "S": (int, (0, 3)),
    }
    values = []
    for quantity in quantities:
        if quantity not in types:
            raise ValueError("Unsupported quantity specified.")
        dtype, shape = types[quantity]
        values.append(np.empty(shape, dtype=dtype))
    return values[0] if len(values) == 1 else tuple(values)


def _pair_cutoff(cutoff, numbers, n):
    if isinstance(cutoff, dict):
        if not cutoff:
            return 0.0, None
        if numbers is not None:
            raw_numbers = np.asarray(numbers)
            if raw_numbers.ndim != 1 or len(raw_numbers) != n:
                raise ValueError(
                    f"Wrong number of atomic numbers: {raw_numbers.size} != {n}"
                )
            if not np.issubdtype(raw_numbers.dtype, np.integer):
                raise TypeError("atomic numbers must have an integer dtype")
            numbers = i64(raw_numbers)
        normalized = {}
        for (left, right), value in cutoff.items():
            left = atomic_numbers.get(left, left)
            right = atomic_numbers.get(right, right)
            value = float(value)
            if not np.isfinite(value):
                raise ValueError("cutoffs must be finite")
            normalized[(int(left), int(right))] = value
        return max(normalized.values()), ("dict", normalized, numbers)

    values = np.asarray(cutoff)
    if values.ndim == 0:
        value = float(values)
        if not np.isfinite(value):
            raise ValueError("cutoffs must be finite")
        return value, None
    radii = f64(values).reshape(-1)
    if len(radii) != n:
        raise ValueError(f"Wrong number of cutoff radii: {len(radii)} != {n}")
    if not np.isfinite(radii).all():
        raise ValueError("cutoffs must be finite")
    return (2.0 * float(radii.max()) if n else 0.0), ("radii", radii)


def primitive_neighbor_list(
    quantities,
    pbc,
    cell,
    positions,
    cutoff,
    numbers=None,
    self_interaction=False,
    use_scaled_positions=False,
    max_nbins=1e6,
):
    """Compute ASE-style directed neighbor tuples in Mojo.

    ``max_nbins`` is accepted for API compatibility.  This implementation
    allocates exact-size output buffers and does not allocate spatial bins.
    """
    del max_nbins
    for quantity in quantities:
        if quantity not in "ijdDS":
            raise ValueError("Unsupported quantity specified.")

    periodic = np.broadcast_to(np.asarray(pbc, dtype=bool), (3,)).copy()
    cell_array = f64(np.asarray(cell).reshape(3, 3))
    positions_array = f64(np.asarray(positions).reshape((-1, 3)))
    if not np.isfinite(cell_array).all() or not np.isfinite(positions_array).all():
        raise ValueError("cell and positions must be finite")
    n = len(positions_array)
    if n == 0:
        return _empty_result(quantities)

    if use_scaled_positions:
        positions_array = f64(positions_array @ cell_array)

    max_cutoff, filtering = _pair_cutoff(cutoff, numbers, n)
    if max_cutoff < 0:
        max_cutoff = 0.0

    base_shift = None
    ranges = np.zeros(3, dtype=np.int64)
    if periodic.any():
        base_shift = np.zeros((n, 3), dtype=np.int64)
        completed = complete_cell(cell_array)
        scaled = np.linalg.solve(completed.T, positions_array.T).T
        base_shift[:, periodic] = np.floor(scaled[:, periodic]).astype(np.int64)
        reciprocal = np.linalg.pinv(cell_array).T
        reciprocal_norms = np.linalg.norm(reciprocal, axis=1)
        face_distances = np.divide(
            1.0,
            reciprocal_norms,
            out=np.ones(3),
            where=reciprocal_norms > 0,
        )
        ranges[periodic] = np.ceil(
            max_cutoff / face_distances[periodic]
        ).astype(np.int64)

    image_count = math.prod(2 * int(value) + 1 for value in ranges)
    candidate_count = n * n * image_count
    if candidate_count > np.iinfo(np.int64).max:
        raise OverflowError("neighbor search exceeds the supported 64-bit size")

    offsets = np.empty(n, dtype=np.int64)
    if base_shift is None:
        base_shift = offsets
    count = lib().mase_neighbor_count(
        addr(positions_array),
        addr(cell_array),
        addr(base_shift),
        n,
        int(ranges[0]),
        int(ranges[1]),
        int(ranges[2]),
        int(bool(self_interaction)),
        max_cutoff * max_cutoff,
        addr(offsets),
    )
    if count < 0 or count > candidate_count:
        raise RuntimeError(f"invalid neighbor count returned by Mojo: {count}")
    if count == 0:
        return _empty_result(quantities)

    pair_i = np.empty(count, dtype=np.int64)
    pair_j = np.empty(count, dtype=np.int64)
    distances = np.empty(count, dtype=np.float64)
    vectors = np.empty((count, 3), dtype=np.float64)
    shifts = np.empty((count, 3), dtype=np.int64)
    lib().mase_neighbor_fill(
        addr(positions_array),
        addr(cell_array),
        addr(base_shift),
        n,
        int(ranges[0]),
        int(ranges[1]),
        int(ranges[2]),
        int(bool(self_interaction)),
        max_cutoff * max_cutoff,
        addr(offsets),
        addr(pair_i),
        addr(pair_j),
        addr(distances),
        addr(vectors),
        addr(shifts),
    )

    if filtering is not None:
        if filtering[0] == "radii":
            radii = filtering[1]
            mask = distances < radii[pair_i] + radii[pair_j]
        else:
            _, pair_values, atom_numbers = filtering
            if atom_numbers is None:
                mask = np.ones(count, dtype=bool)
            else:
                atom_numbers = atom_numbers.reshape(-1)
                limits = np.zeros(count)
                for (left, right), value in pair_values.items():
                    if left == right:
                        selected = (
                            (atom_numbers[pair_i] == left)
                            & (atom_numbers[pair_j] == right)
                        )
                    else:
                        selected = (
                            ((atom_numbers[pair_i] == left) & (atom_numbers[pair_j] == right))
                            | ((atom_numbers[pair_i] == right) & (atom_numbers[pair_j] == left))
                        )
                    limits[selected] = value
                mask = distances < limits
        pair_i = pair_i[mask]
        pair_j = pair_j[mask]
        distances = distances[mask]
        vectors = vectors[mask]
        shifts = shifts[mask]

    available = {
        "i": pair_i,
        "j": pair_j,
        "d": distances,
        "D": vectors,
        "S": shifts,
    }
    result = [available[quantity] for quantity in quantities]
    return result[0] if len(result) == 1 else tuple(result)


def neighbor_list(
    quantities,
    a,
    cutoff,
    self_interaction=False,
    max_nbins=1e6,
):
    return primitive_neighbor_list(
        quantities,
        a.pbc,
        a.get_cell(complete=True),
        a.positions,
        cutoff,
        numbers=a.numbers,
        self_interaction=self_interaction,
        max_nbins=max_nbins,
    )


def first_neighbors(natoms, first_atom):
    first_atom = i64(first_atom).reshape(-1)
    if len(first_atom) == 0:
        return np.zeros(natoms + 1, dtype=int)
    return np.searchsorted(first_atom, np.arange(natoms + 1), side="left")


class NewPrimitiveNeighborList:
    def __init__(
        self,
        cutoffs,
        skin=0.3,
        sorted=False,
        self_interaction=True,
        bothways=False,
        use_scaled_positions=False,
    ):
        self.cutoffs = np.asarray(cutoffs, dtype=float) + skin
        self.skin = skin
        self.sorted = sorted
        self.self_interaction = self_interaction
        self.bothways = bothways
        self.nupdates = 0
        self.use_scaled_positions = use_scaled_positions

    def update(self, pbc, cell, positions, numbers=None):
        positions = np.asarray(positions)
        if self.nupdates == 0:
            self.build(pbc, cell, positions, numbers=numbers)
            return True
        changed = (
            (self.pbc != pbc).any()
            or (self.cell != cell).any()
            or len(self.positions) != len(positions)
        )
        if not changed and len(positions):
            changed = (
                ((self.positions - positions) ** 2).sum(axis=1).max()
                > self.skin**2
            )
        if changed:
            self.build(pbc, cell, positions, numbers=numbers)
            return True
        return False

    def build(self, pbc, cell, positions, numbers=None):
        self.pbc = np.array(pbc, dtype=bool, copy=True)
        self.cell = np.array(cell, dtype=float, copy=True)
        self.positions = np.array(positions, dtype=float, copy=True)
        if len(self.cutoffs) != len(self.positions):
            raise ValueError(
                f"Wrong number of cutoff radii: {len(self.cutoffs)} != {len(self.positions)}"
            )
        pair_first, pair_second, offsets = primitive_neighbor_list(
            "ijS",
            self.pbc,
            self.cell,
            self.positions,
            self.cutoffs,
            numbers=numbers,
            self_interaction=self.self_interaction,
            use_scaled_positions=self.use_scaled_positions,
        )

        if len(self.positions) and not self.bothways:
            ox, oy, oz = offsets.T
            mask = oz > 0
            mask &= oy == 0
            mask |= oy > 0
            mask &= ox == 0
            mask |= ox > 0
            mask |= (pair_first <= pair_second) & (offsets == 0).all(axis=1)
            pair_first = pair_first[mask]
            pair_second = pair_second[mask]
            offsets = offsets[mask]

        if len(self.positions) and self.sorted:
            order = np.lexsort((pair_second, pair_first))
            pair_first = pair_first[order]
            pair_second = pair_second[order]
            offsets = offsets[order]

        self.pair_first = pair_first
        self.pair_second = pair_second
        self.offset_vec = offsets
        self.first_neigh = first_neighbors(len(self.positions), pair_first)
        self.neighbors = [
            pair_second[self.first_neigh[i] : self.first_neigh[i + 1]]
            for i in range(len(self.positions))
        ]
        self.displacements = [
            offsets[self.first_neigh[i] : self.first_neigh[i + 1]]
            for i in range(len(self.positions))
        ]
        self.nupdates += 1

    def get_neighbors(self, atom):
        return (
            self.pair_second[self.first_neigh[atom] : self.first_neigh[atom + 1]],
            self.offset_vec[self.first_neigh[atom] : self.first_neigh[atom + 1]],
        )


class PrimitiveNeighborList(NewPrimitiveNeighborList):
    pass


class NeighborList:
    def __init__(
        self,
        cutoffs,
        skin=0.3,
        sorted=False,
        self_interaction=True,
        bothways=False,
        primitive=PrimitiveNeighborList,
    ):
        self.nl = primitive(
            cutoffs,
            skin,
            sorted,
            self_interaction=self_interaction,
            bothways=bothways,
        )

    def update(self, atoms):
        return self.nl.update(
            atoms.pbc, atoms.get_cell(complete=True), atoms.positions
        )

    def get_neighbors(self, atom):
        if self.nl.nupdates <= 0:
            raise RuntimeError("Must call update(atoms) on your neighborlist first!")
        return self.nl.get_neighbors(atom)

    def get_connectivity_matrix(self, sparse=True):
        return get_connectivity_matrix(self.nl, sparse)

    @property
    def nupdates(self):
        return self.nl.nupdates


def get_connectivity_matrix(nl, sparse=True):
    if nl.nupdates <= 0:
        raise RuntimeError("Must call update(atoms) on your neighborlist first!")
    natoms = len(nl.cutoffs)
    if sparse:
        from scipy import sparse as scipy_sparse

        matrix = scipy_sparse.dok_matrix((natoms, natoms), dtype=np.int8)
    else:
        matrix = np.zeros((natoms, natoms), dtype=np.int8)
    for atom in range(natoms):
        matrix[atom, nl.get_neighbors(atom)[0]] = 1
    return matrix
