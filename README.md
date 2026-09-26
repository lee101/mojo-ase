# mojo-ase

`mojo-ase` ports the compute-intensive neighbor geometry and classical
pair-force core of the Atomic Simulation Environment (ASE) to Mojo. It keeps
ASE's Python names, signatures, array conventions, `Atoms` interoperability,
calculator result keys, and stress conventions for the covered subset.

This is an independent open-source implementation, not a fork of ASE. ASE
remains a dependency for its `Atoms` and `Calculator` interfaces, atomic data,
and Minkowski cell reduction.

## Coverage

The following APIs are implemented:

- `mojo_ase.neighborlist.primitive_neighbor_list`
- `mojo_ase.neighborlist.neighbor_list`
- `natural_cutoffs`, `build_neighbor_list`, `mic`, and `first_neighbors`
- `NewPrimitiveNeighborList`, `PrimitiveNeighborList`, and `NeighborList`
- `get_connectivity_matrix`
- `mojo_ase.geometry.find_mic` and `get_distances`
- `mojo_ase.calculators.lj.LennardJones`
- `mojo_ase.calculators.morse.MorsePotential`
- ASE's public Lennard-Jones and Morse cutoff helper functions

Neighbor lists support scalar, element-pair dictionary, and per-atom radius
cutoffs; Cartesian or scaled coordinates; arbitrary triclinic cells; mixed
periodic axes; positions outside the primary cell; self interaction; and
cutoffs large enough to return multiple images of the same atom. Returned
lists use ASE's directed-pair semantics, strict cutoff comparison, quantity
codes (`i`, `j`, `d`, `D`, `S`), and shift-vector identity
`D = positions[j] - positions[i] + S @ cell`.

The port does not cover the `Atoms` implementation, file formats, structure
builders, optimizers, electronic-structure calculators, graph-distance
helpers, or the rest of ASE's geometry module. The neighbor search currently
uses an exact quadratic scan with bounded periodic-image enumeration.
Consequently it is effective for small and medium atomistic systems and can be
faster there, but ASE's cell-binned implementation has better asymptotic
scaling for very large sparse systems. `max_nbins` is accepted for signature
compatibility but is not used because this implementation allocates no bins.

## Install

Install the pinned Mojo toolchain and Python dependencies, then build the
shared library:

```bash
pixi install
pixi run build
```

The repository's Pixi activation puts `python/` on `PYTHONPATH`. A normal
Python package can also be created from `pyproject.toml`; deployments may set
`MOJO_ASE_LIB` to an already-built `libmojo-ase.so`.

## Usage

```python
from ase.build import bulk
from mojo_ase.calculators.lj import LennardJones
from mojo_ase.neighborlist import neighbor_list

atoms = bulk("Ar", "fcc", a=1.62, cubic=True) * (2, 2, 2)

i, j, distances, shifts = neighbor_list("ijdS", atoms, cutoff=3.0)
print(len(i), distances.min(), shifts.shape)

atoms.calc = LennardJones(
    epsilon=0.8, sigma=1.0, rc=3.0, ro=1.9, smooth=True
)
print(atoms.get_potential_energy())
print(atoms.get_forces().shape)
```

The example runs after `pixi run build` with:

```bash
pixi run python example.py
```

## Correctness

The test suite compares complete neighbor tuples against ASE 3.29.0 for
molecules, periodic crystals, triclinic cells, partial periodicity, pair and
per-atom cutoffs, scaled coordinates, out-of-cell positions, and multiple
periodic images. It also compares MIC vectors, distance matrices, neighbor-list
object behavior, energies, atomic energies, forces, stress, and per-atom
stress for both calculators.

```bash
pixi run test
```

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz using
Python 3.13.14, ASE 3.29.0, and NumPy 2.5.1. Times are the best of five warm
runs. `ASE / Mojo` above one means `mojo-ase` was faster. These are measured
results, not estimates.

| case | mojo-ase | ASE | ASE / Mojo | result |
|---|---:|---:|---:|---|
| neighbor_list, 600 atoms nonperiodic | 2.888 ms | 26.311 ms | 9.11x | faster |
| neighbor_list, 108-atom periodic Cu | 0.963 ms | 6.681 ms | 6.94x | faster |
| find_mic, 200k triclinic vectors | 31.081 ms | 2265.510 ms | 72.89x | faster |
| LennardJones forces, 108 atoms | 2.022 ms | 60.094 ms | 29.72x | faster |
| MorsePotential forces, 108 atoms | 1.876 ms | 268.421 ms | 143.10x | faster |

Results will vary with CPU, compiler, structure, cutoff, and system size. Run
the benchmark on the target machine rather than treating this table as a
universal speed claim.

No GPU path is included. The only benchmark below 5x at baseline was the small,
branch-heavy periodic neighbor search, where transfer and launch overhead would
dominate. The arithmetic-heavy MIC and force cases were already more than 5x
faster than ASE and were deliberately left outside the optimization scope.

## How it works

`src/kernels.mojo` is one compilation unit exported as a C ABI shared library.
`build/build.sh` invokes `mojo build --emit shared-lib` and writes
`dist/libmojo-ase.so`. The Python layer loads it with `ctypes`.

NumPy owns every input, output, and scratch allocation. C-contiguous
`float64` and `int64` buffers cross the ABI as integer addresses because Mojo
exports cannot be parametric over pointer origins. Mojo reconstructs
`UnsafePointer[..., AnyOrigin[mut=True]]` values inside each wrapper. Positions,
cell matrices, displacement vectors, forces, and stresses are row-major;
neighbor outputs are structure-of-arrays buffers. No Mojo allocation crosses
the FFI boundary.

Neighbor construction uses a count pass followed by an exact-size fill pass.
Periodic positions are represented by an integer base shift, and the kernel
enumerates only the cell-image range geometrically capable of intersecting the
cutoff. Candidate distances use native-width SIMD with a scalar tail. Periodic
count and fill work is split by atom across up to 16 Python worker threads once a
search passes `MIN_PARALLEL_WORK` (2^22) candidate checks; atom ranges write
disjoint output slots, so the split is exact. Measured here: 1.9x at n=400,
3.4x at n=1372, 6.6x at n=2916, and 0.97x at n=256, which is where the floor
sits. Smaller and nonperiodic searches remain serial.
Force kernels consume the directed neighbor arrays and accumulate
ASE-compatible per-atom energy, force, and virial contributions in one pass.
