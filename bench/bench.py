"""Benchmark mojo-ase against ASE on identical structures and requests."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np
import ase
from ase import Atoms
from ase.build import bulk
from ase.calculators.lj import LennardJones as ASELennardJones
from ase.calculators.morse import MorsePotential as ASEMorsePotential
from ase.geometry import find_mic as ase_find_mic
from ase.neighborlist import neighbor_list as ase_neighbor_list

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"
    ),
)

from mojo_ase.calculators.lj import LennardJones  # noqa: E402
from mojo_ase.calculators.morse import MorsePotential  # noqa: E402
from mojo_ase.geometry import find_mic  # noqa: E402
from mojo_ase.neighborlist import neighbor_list  # noqa: E402


def best_time(function, repeat=5):
    function()
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def calculator_call(calculator_type, atoms, kwargs):
    def run():
        calculator = calculator_type(**kwargs)
        calculator.calculate(atoms)
        return calculator.results["forces"]

    return run


def cases():
    rng = np.random.default_rng(2026)

    cloud = Atoms(
        "Ar600",
        positions=rng.uniform(0.0, 50.0, size=(600, 3)),
        cell=[50.0, 50.0, 50.0],
        pbc=False,
    )
    yield (
        "neighbor_list, 600 atoms nonperiodic",
        lambda: neighbor_list("ijdD", cloud, 3.0),
        lambda: ase_neighbor_list("ijdD", cloud, 3.0),
    )

    crystal = bulk("Cu", cubic=True) * (3, 3, 3)
    crystal.rattle(0.05, rng=rng)
    yield (
        "neighbor_list, 108-atom periodic Cu",
        lambda: neighbor_list("ijdD", crystal, 3.2),
        lambda: ase_neighbor_list("ijdD", crystal, 3.2),
    )

    cell = np.array([[4.0, 0.3, 0.2], [0.4, 3.5, 0.1], [0.2, 0.5, 4.2]])
    vectors = rng.normal(size=(200_000, 3)) * 8.0
    yield (
        "find_mic, 200k triclinic vectors",
        lambda: find_mic(vectors, cell, True),
        lambda: ase_find_mic(vectors, cell, True),
    )

    pair_atoms = bulk("Ar", "fcc", a=1.62, cubic=True) * (3, 3, 3)
    pair_atoms.rattle(0.015, rng=rng)
    lj_kwargs = dict(epsilon=0.8, sigma=1.0, rc=3.0, ro=1.9, smooth=True)
    yield (
        "LennardJones forces, 108 atoms",
        calculator_call(LennardJones, pair_atoms, lj_kwargs),
        calculator_call(ASELennardJones, pair_atoms, lj_kwargs),
    )

    morse_kwargs = dict(epsilon=0.8, rho0=5.2, r0=1.0, rcut1=1.9, rcut2=2.7)
    yield (
        "MorsePotential forces, 108 atoms",
        calculator_call(MorsePotential, pair_atoms, morse_kwargs),
        calculator_call(ASEMorsePotential, pair_atoms, morse_kwargs),
    )


def main():
    print(
        f"Machine: {cpu_name()}; Python {platform.python_version()}; "
        f"ASE {ase.__version__}; NumPy {np.__version__}"
    )
    print()
    print("| case | mojo-ase | ASE | ASE / Mojo | result |")
    print("|---|---:|---:|---:|---|")
    for name, ours, upstream in cases():
        mojo_time = best_time(ours)
        ase_time = best_time(upstream)
        ratio = ase_time / mojo_time
        result = "faster" if ratio > 1.0 else "slower"
        print(
            f"| {name} | {mojo_time * 1000:.3f} ms | "
            f"{ase_time * 1000:.3f} ms | {ratio:.2f}x | {result} |"
        )


if __name__ == "__main__":
    main()
