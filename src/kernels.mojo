"""ASE-compatible neighbor geometry and pair-potential kernels.

All storage is owned by NumPy.  Exported functions receive integer addresses
and reconstruct explicitly-originated pointers inside the C ABI boundary.
"""

from std.math import exp, floor, sqrt
from std.runtime import initialize_runtime
from std.sys.info import simd_width_of

comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]


def fp(address: Int) -> FPtr:
    return FPtr(unsafe_from_address=address)


def ip(address: Int) -> IPtr:
    return IPtr(unsafe_from_address=address)


def neighbor_count_one(
    positions: FPtr,
    cell: FPtr,
    base_shift: IPtr,
    n: Int,
    i: Int,
    rx: Int,
    ry: Int,
    rz: Int,
    self_interaction: Bool,
    cutoff2: Float64,
) -> Int:
    comptime W = simd_width_of[DType.float64]()
    var count = 0
    var pix = positions[3 * i]
    var piy = positions[3 * i + 1]
    var piz = positions[3 * i + 2]
    var base_ix = Int(base_shift[3 * i])
    var base_iy = Int(base_shift[3 * i + 1])
    var base_iz = Int(base_shift[3 * i + 2])
    var j = 0
    while j + W <= n:
        var rawx = (positions + 3 * j).strided_load[width=W](3) - pix
        var rawy = (positions + 3 * j + 1).strided_load[width=W](3) - piy
        var rawz = (positions + 3 * j + 2).strided_load[width=W](3) - piz
        var base_jx = (base_shift + 3 * j).strided_load[width=W](3)
        var base_jy = (base_shift + 3 * j + 1).strided_load[width=W](3)
        var base_jz = (base_shift + 3 * j + 2).strided_load[width=W](3)
        for sx0 in range(-rx, rx + 1):
            var sx = SIMD[DType.int64, W](base_ix + sx0) - base_jx
            for sy0 in range(-ry, ry + 1):
                var sy = SIMD[DType.int64, W](base_iy + sy0) - base_jy
                for sz0 in range(-rz, rz + 1):
                    var sz = SIMD[DType.int64, W](base_iz + sz0) - base_jz
                    var dx = (
                        rawx + sx.cast[DType.float64]() * cell[0]
                        + sy.cast[DType.float64]() * cell[3]
                        + sz.cast[DType.float64]() * cell[6]
                    )
                    var dy = (
                        rawy + sx.cast[DType.float64]() * cell[1]
                        + sy.cast[DType.float64]() * cell[4]
                        + sz.cast[DType.float64]() * cell[7]
                    )
                    var dz = (
                        rawz + sx.cast[DType.float64]() * cell[2]
                        + sy.cast[DType.float64]() * cell[5]
                        + sz.cast[DType.float64]() * cell[8]
                    )
                    var r2 = dx * dx + dy * dy + dz * dz
                    count += Int(
                        r2.lt(cutoff2).select(
                            SIMD[DType.int64, W](1), SIMD[DType.int64, W](0)
                        ).reduce_add()
                    )
                    if (
                        not self_interaction and sx0 == 0 and sy0 == 0
                        and sz0 == 0 and j <= i and i < j + W
                        and 0.0 < cutoff2
                    ):
                        count -= 1
        j += W
    while j < n:
        var rawx = positions[3 * j] - pix
        var rawy = positions[3 * j + 1] - piy
        var rawz = positions[3 * j + 2] - piz
        for sx0 in range(-rx, rx + 1):
            var sx = base_ix - Int(base_shift[3 * j]) + sx0
            for sy0 in range(-ry, ry + 1):
                var sy = base_iy - Int(base_shift[3 * j + 1]) + sy0
                for sz0 in range(-rz, rz + 1):
                    var sz = base_iz - Int(base_shift[3 * j + 2]) + sz0
                    if self_interaction or i != j or sx != 0 or sy != 0 or sz != 0:
                        var dx = (
                            rawx + Float64(sx) * cell[0]
                            + Float64(sy) * cell[3] + Float64(sz) * cell[6]
                        )
                        var dy = (
                            rawy + Float64(sx) * cell[1]
                            + Float64(sy) * cell[4] + Float64(sz) * cell[7]
                        )
                        var dz = (
                            rawz + Float64(sx) * cell[2]
                            + Float64(sy) * cell[5] + Float64(sz) * cell[8]
                        )
                        if dx * dx + dy * dy + dz * dz < cutoff2:
                            count += 1
        j += 1
    return count


def neighbor_count_one_nonperiodic(
    positions: FPtr,
    n: Int,
    i: Int,
    self_interaction: Bool,
    cutoff2: Float64,
) -> Int:
    comptime W = simd_width_of[DType.float64]()
    var pix = positions[3 * i]
    var piy = positions[3 * i + 1]
    var piz = positions[3 * i + 2]
    var count = 0
    var j = 0
    while j + W <= n:
        var dx = (positions + 3 * j).strided_load[width=W](3) - pix
        var dy = (positions + 3 * j + 1).strided_load[width=W](3) - piy
        var dz = (positions + 3 * j + 2).strided_load[width=W](3) - piz
        var r2 = dx * dx + dy * dy + dz * dz
        var selected = r2.lt(cutoff2)
        count += Int(
            selected.select(
                SIMD[DType.int64, W](1), SIMD[DType.int64, W](0)
            ).reduce_add()
        )
        if (
            not self_interaction and j <= i and i < j + W
            and 0.0 < cutoff2
        ):
            count -= 1
        j += W
    while j < n:
        var dx = positions[3 * j] - pix
        var dy = positions[3 * j + 1] - piy
        var dz = positions[3 * j + 2] - piz
        if (self_interaction or i != j) and dx * dx + dy * dy + dz * dz < cutoff2:
            count += 1
        j += 1
    return count


def neighbor_count_range(
    positions: FPtr,
    cell: FPtr,
    base_shift: IPtr,
    n: Int,
    first: Int,
    last: Int,
    rx: Int,
    ry: Int,
    rz: Int,
    self_interaction: Bool,
    cutoff2: Float64,
    offsets: IPtr,
):
    """Raw per-atom neighbour counts for atoms [first, last)."""
    var nonperiodic = rx == 0 and ry == 0 and rz == 0
    for i in range(first, last):
        if nonperiodic:
            offsets[i] = Int64(
                neighbor_count_one_nonperiodic(
                    positions, n, i, self_interaction, cutoff2
                )
            )
        else:
            offsets[i] = Int64(
                neighbor_count_one(
                    positions, cell, base_shift, n, i, rx, ry, rz,
                    self_interaction, cutoff2,
                )
            )


def offsets_to_starts(offsets: IPtr, n: Int) -> Int:
    """Turn raw counts in [0, n) into exclusive prefix starts; return the total."""
    comptime W = simd_width_of[DType.float64]()
    var vector_total = SIMD[DType.int64, W](0)
    var i = 0
    while i + W <= n:
        vector_total += offsets.load[width=W](i)
        i += W
    var total = Int(vector_total.reduce_add())
    while i < n:
        total += Int(offsets[i])
        i += 1

    var start = 0
    for atom in range(n):
        var count = Int(offsets[atom])
        offsets[atom] = Int64(start)
        start += count
    return total


def neighbor_fill_one(
    positions: FPtr,
    cell: FPtr,
    base_shift: IPtr,
    n: Int,
    i: Int,
    rx: Int,
    ry: Int,
    rz: Int,
    self_interaction: Bool,
    cutoff2: Float64,
    pair_i: IPtr,
    pair_j: IPtr,
    distances: FPtr,
    vectors: FPtr,
    shifts: IPtr,
) -> Int:
    comptime W = simd_width_of[DType.float64]()
    var k = 0
    var pix = positions[3 * i]
    var piy = positions[3 * i + 1]
    var piz = positions[3 * i + 2]
    var base_ix = Int(base_shift[3 * i])
    var base_iy = Int(base_shift[3 * i + 1])
    var base_iz = Int(base_shift[3 * i + 2])
    var j = 0
    while j + W <= n:
        var rawx = (positions + 3 * j).strided_load[width=W](3) - pix
        var rawy = (positions + 3 * j + 1).strided_load[width=W](3) - piy
        var rawz = (positions + 3 * j + 2).strided_load[width=W](3) - piz
        var base_jx = (base_shift + 3 * j).strided_load[width=W](3)
        var base_jy = (base_shift + 3 * j + 1).strided_load[width=W](3)
        var base_jz = (base_shift + 3 * j + 2).strided_load[width=W](3)
        for sx0 in range(-rx, rx + 1):
            var sx = SIMD[DType.int64, W](base_ix + sx0) - base_jx
            for sy0 in range(-ry, ry + 1):
                var sy = SIMD[DType.int64, W](base_iy + sy0) - base_jy
                for sz0 in range(-rz, rz + 1):
                    var sz = SIMD[DType.int64, W](base_iz + sz0) - base_jz
                    var dx = (
                        rawx + sx.cast[DType.float64]() * cell[0]
                        + sy.cast[DType.float64]() * cell[3]
                        + sz.cast[DType.float64]() * cell[6]
                    )
                    var dy = (
                        rawy + sx.cast[DType.float64]() * cell[1]
                        + sy.cast[DType.float64]() * cell[4]
                        + sz.cast[DType.float64]() * cell[7]
                    )
                    var dz = (
                        rawz + sx.cast[DType.float64]() * cell[2]
                        + sy.cast[DType.float64]() * cell[5]
                        + sz.cast[DType.float64]() * cell[8]
                    )
                    var r2 = dx * dx + dy * dy + dz * dz
                    var selected = r2.lt(cutoff2)
                    for lane in range(W):
                        var atom_j = j + lane
                        if (
                            selected[lane]
                            and (
                                self_interaction or i != atom_j
                                or sx[lane] != 0 or sy[lane] != 0 or sz[lane] != 0
                            )
                        ):
                            pair_i[k] = Int64(i)
                            pair_j[k] = Int64(atom_j)
                            distances[k] = sqrt(r2[lane])
                            vectors[3 * k] = dx[lane]
                            vectors[3 * k + 1] = dy[lane]
                            vectors[3 * k + 2] = dz[lane]
                            shifts[3 * k] = sx[lane]
                            shifts[3 * k + 1] = sy[lane]
                            shifts[3 * k + 2] = sz[lane]
                            k += 1
        j += W
    while j < n:
        var rawx = positions[3 * j] - pix
        var rawy = positions[3 * j + 1] - piy
        var rawz = positions[3 * j + 2] - piz
        for sx0 in range(-rx, rx + 1):
            var sx = base_ix - Int(base_shift[3 * j]) + sx0
            for sy0 in range(-ry, ry + 1):
                var sy = base_iy - Int(base_shift[3 * j + 1]) + sy0
                for sz0 in range(-rz, rz + 1):
                    var sz = base_iz - Int(base_shift[3 * j + 2]) + sz0
                    if self_interaction or i != j or sx != 0 or sy != 0 or sz != 0:
                        var dx = (
                            rawx + Float64(sx) * cell[0]
                            + Float64(sy) * cell[3] + Float64(sz) * cell[6]
                        )
                        var dy = (
                            rawy + Float64(sx) * cell[1]
                            + Float64(sy) * cell[4] + Float64(sz) * cell[7]
                        )
                        var dz = (
                            rawz + Float64(sx) * cell[2]
                            + Float64(sy) * cell[5] + Float64(sz) * cell[8]
                        )
                        var r2 = dx * dx + dy * dy + dz * dz
                        if r2 < cutoff2:
                            pair_i[k] = Int64(i)
                            pair_j[k] = Int64(j)
                            distances[k] = sqrt(r2)
                            vectors[3 * k] = dx
                            vectors[3 * k + 1] = dy
                            vectors[3 * k + 2] = dz
                            shifts[3 * k] = Int64(sx)
                            shifts[3 * k + 1] = Int64(sy)
                            shifts[3 * k + 2] = Int64(sz)
                            k += 1
        j += 1
    return k


def neighbor_fill_one_nonperiodic(
    positions: FPtr,
    n: Int,
    i: Int,
    self_interaction: Bool,
    cutoff2: Float64,
    pair_i: IPtr,
    pair_j: IPtr,
    distances: FPtr,
    vectors: FPtr,
    shifts: IPtr,
) -> Int:
    comptime W = simd_width_of[DType.float64]()
    var pix = positions[3 * i]
    var piy = positions[3 * i + 1]
    var piz = positions[3 * i + 2]
    var k = 0
    var j = 0
    while j + W <= n:
        var dx = (positions + 3 * j).strided_load[width=W](3) - pix
        var dy = (positions + 3 * j + 1).strided_load[width=W](3) - piy
        var dz = (positions + 3 * j + 2).strided_load[width=W](3) - piz
        var r2 = dx * dx + dy * dy + dz * dz
        var selected = r2.lt(cutoff2)
        for lane in range(W):
            var atom_j = j + lane
            if selected[lane] and (self_interaction or i != atom_j):
                pair_i[k] = Int64(i)
                pair_j[k] = Int64(atom_j)
                distances[k] = sqrt(r2[lane])
                vectors[3 * k] = dx[lane]
                vectors[3 * k + 1] = dy[lane]
                vectors[3 * k + 2] = dz[lane]
                shifts[3 * k] = 0
                shifts[3 * k + 1] = 0
                shifts[3 * k + 2] = 0
                k += 1
        j += W
    while j < n:
        var dx = positions[3 * j] - pix
        var dy = positions[3 * j + 1] - piy
        var dz = positions[3 * j + 2] - piz
        var r2 = dx * dx + dy * dy + dz * dz
        if (self_interaction or i != j) and r2 < cutoff2:
            pair_i[k] = Int64(i)
            pair_j[k] = Int64(j)
            distances[k] = sqrt(r2)
            vectors[3 * k] = dx
            vectors[3 * k + 1] = dy
            vectors[3 * k + 2] = dz
            shifts[3 * k] = 0
            shifts[3 * k + 1] = 0
            shifts[3 * k + 2] = 0
            k += 1
        j += 1
    return k


def neighbor_fill_range(
    positions: FPtr,
    cell: FPtr,
    base_shift: IPtr,
    offsets: IPtr,
    n: Int,
    first: Int,
    last: Int,
    rx: Int,
    ry: Int,
    rz: Int,
    self_interaction: Bool,
    cutoff2: Float64,
    pair_i: IPtr,
    pair_j: IPtr,
    distances: FPtr,
    vectors: FPtr,
    shifts: IPtr,
):
    """Write the pairs owned by atoms [first, last); their slots are disjoint."""
    var nonperiodic = rx == 0 and ry == 0 and rz == 0
    for i in range(first, last):
        var start = Int(offsets[i])
        if nonperiodic:
            _ = neighbor_fill_one_nonperiodic(
                positions, n, i, self_interaction, cutoff2,
                pair_i + start, pair_j + start, distances + start,
                vectors + 3 * start, shifts + 3 * start,
            )
        else:
            _ = neighbor_fill_one(
                positions, cell, base_shift, n, i, rx, ry, rz,
                self_interaction, cutoff2, pair_i + start, pair_j + start,
                distances + start, vectors + 3 * start, shifts + 3 * start,
            )


def minimum_image(
    vectors: FPtr,
    cell: FPtr,
    inverse: FPtr,
    result: FPtr,
    lengths: FPtr,
    n: Int,
    px: Int,
    py: Int,
    pz: Int,
):
    for i in range(n):
        var vx = vectors[3 * i]
        var vy = vectors[3 * i + 1]
        var vz = vectors[3 * i + 2]
        var fx = vx * inverse[0] + vy * inverse[3] + vz * inverse[6]
        var fy = vx * inverse[1] + vy * inverse[4] + vz * inverse[7]
        var fz = vx * inverse[2] + vy * inverse[5] + vz * inverse[8]
        if px != 0:
            fx -= floor(fx)
        if py != 0:
            fy -= floor(fy)
        if pz != 0:
            fz -= floor(fz)
        var bx = fx * cell[0] + fy * cell[3] + fz * cell[6]
        var by = fx * cell[1] + fy * cell[4] + fz * cell[7]
        var bz = fx * cell[2] + fy * cell[5] + fz * cell[8]
        var best2 = 1.7976931348623157e308
        var bestx = 0.0
        var besty = 0.0
        var bestz = 0.0
        for sx in range(-px, px + 1):
            for sy in range(-py, py + 1):
                for sz in range(-pz, pz + 1):
                    var dx = (
                        bx + Float64(sx) * cell[0]
                        + Float64(sy) * cell[3] + Float64(sz) * cell[6]
                    )
                    var dy = (
                        by + Float64(sx) * cell[1]
                        + Float64(sy) * cell[4] + Float64(sz) * cell[7]
                    )
                    var dz = (
                        bz + Float64(sx) * cell[2]
                        + Float64(sy) * cell[5] + Float64(sz) * cell[8]
                    )
                    var r2 = dx * dx + dy * dy + dz * dz
                    if r2 < best2:
                        best2 = r2
                        bestx = dx
                        besty = dy
                        bestz = dz
        result[3 * i] = bestx
        result[3 * i + 1] = besty
        result[3 * i + 2] = bestz
        lengths[i] = sqrt(best2)


def lennard_jones(
    pair_i: IPtr,
    vectors: FPtr,
    npairs: Int,
    natoms: Int,
    sigma: Float64,
    epsilon: Float64,
    rc: Float64,
    ro: Float64,
    smooth: Bool,
    energies: FPtr,
    forces: FPtr,
    stresses: FPtr,
):
    for i in range(natoms):
        energies[i] = 0.0
        for c in range(3):
            forces[3 * i + c] = 0.0
        for c in range(9):
            stresses[9 * i + c] = 0.0
    var rc2 = rc * rc
    var ro2 = ro * ro
    var e0c6 = sigma / rc
    e0c6 = e0c6 * e0c6
    e0c6 = e0c6 * e0c6 * e0c6
    var e0 = 4.0 * epsilon * (e0c6 * e0c6 - e0c6)
    for k in range(npairs):
        var i = Int(pair_i[k])
        var dx = vectors[3 * k]
        var dy = vectors[3 * k + 1]
        var dz = vectors[3 * k + 2]
        var r2 = dx * dx + dy * dy + dz * dz
        if r2 > rc2:
            continue
        var c6 = sigma * sigma / r2
        c6 = c6 * c6 * c6
        var c12 = c6 * c6
        var pair_energy = 4.0 * epsilon * (c12 - c6)
        var pair_force = -24.0 * epsilon * (2.0 * c12 - c6) / r2
        if smooth:
            var fc = 1.0
            var dfc = 0.0
            if r2 >= ro2:
                var span = rc2 - ro2
                fc = (
                    (rc2 - r2) * (rc2 - r2)
                    * (rc2 + 2.0 * r2 - 3.0 * ro2) / (span * span * span)
                )
                dfc = 6.0 * (rc2 - r2) * (ro2 - r2) / (span * span * span)
            pair_force = fc * pair_force + 2.0 * dfc * pair_energy
            pair_energy *= fc
        else:
            pair_energy -= e0
        var fx = pair_force * dx
        var fy = pair_force * dy
        var fz = pair_force * dz
        energies[i] += 0.5 * pair_energy
        forces[3 * i] += fx
        forces[3 * i + 1] += fy
        forces[3 * i + 2] += fz
        stresses[9 * i] += 0.5 * fx * dx
        stresses[9 * i + 1] += 0.5 * fx * dy
        stresses[9 * i + 2] += 0.5 * fx * dz
        stresses[9 * i + 3] += 0.5 * fy * dx
        stresses[9 * i + 4] += 0.5 * fy * dy
        stresses[9 * i + 5] += 0.5 * fy * dz
        stresses[9 * i + 6] += 0.5 * fz * dx
        stresses[9 * i + 7] += 0.5 * fz * dy
        stresses[9 * i + 8] += 0.5 * fz * dz


def morse(
    pair_i: IPtr,
    distances: FPtr,
    vectors: FPtr,
    npairs: Int,
    natoms: Int,
    epsilon: Float64,
    rho0: Float64,
    r0: Float64,
    rcut1: Float64,
    rcut2: Float64,
    energies: FPtr,
    forces: FPtr,
    stress: FPtr,
):
    for i in range(natoms):
        energies[i] = 0.0
        for c in range(3):
            forces[3 * i + c] = 0.0
    for c in range(9):
        stress[c] = 0.0
    var span = rcut2 - rcut1
    for k in range(npairs):
        var i = Int(pair_i[k])
        var r = distances[k]
        if r >= rcut2:
            continue
        var ef = exp(rho0 * (1.0 - r / r0))
        var pair_energy = epsilon * ef * (ef - 2.0)
        var de = (-2.0 * epsilon * rho0 / r0) * ef * (ef - 1.0)
        var fc = 1.0
        var dfc = 0.0
        if r > rcut1:
            var s = 1.0 - (r - rcut1) / span
            var s2 = s * s
            var s3 = s2 * s
            var s4 = s3 * s
            var s5 = s4 * s
            fc = 6.0 * s5 - 15.0 * s4 + 10.0 * s3
            dfc = -(30.0 * s4 - 60.0 * s3 + 30.0 * s2) / span
        de = de * fc + pair_energy * dfc
        energies[i] += 0.5 * pair_energy * fc
        var scale = de / r
        var dx = vectors[3 * k]
        var dy = vectors[3 * k + 1]
        var dz = vectors[3 * k + 2]
        var fx = scale * dx
        var fy = scale * dy
        var fz = scale * dz
        forces[3 * i] += fx
        forces[3 * i + 1] += fy
        forces[3 * i + 2] += fz
        stress[0] += 0.5 * dx * fx
        stress[1] += 0.5 * dx * fy
        stress[2] += 0.5 * dx * fz
        stress[3] += 0.5 * dy * fx
        stress[4] += 0.5 * dy * fy
        stress[5] += 0.5 * dy * fz
        stress[6] += 0.5 * dz * fx
        stress[7] += 0.5 * dz * fy
        stress[8] += 0.5 * dz * fz


@export("mase_neighbor_count_range")
def mase_neighbor_count_range(
    positions: Int,
    cell: Int,
    base_shift: Int,
    n: Int,
    first: Int,
    last: Int,
    rx: Int,
    ry: Int,
    rz: Int,
    self_interaction: Int,
    cutoff2: Float64,
    offsets: Int,
) abi("C"):
    neighbor_count_range(
        fp(positions), fp(cell), ip(base_shift), n, first, last, rx, ry, rz,
        self_interaction != 0, cutoff2, ip(offsets),
    )


@export("mase_offsets_to_starts")
def mase_offsets_to_starts(offsets: Int, n: Int) abi("C") -> Int:
    return offsets_to_starts(ip(offsets), n)


@export("mase_neighbor_fill_range")
def mase_neighbor_fill_range(
    positions: Int,
    cell: Int,
    base_shift: Int,
    n: Int,
    first: Int,
    last: Int,
    rx: Int,
    ry: Int,
    rz: Int,
    self_interaction: Int,
    cutoff2: Float64,
    offsets: Int,
    pair_i: Int,
    pair_j: Int,
    distances: Int,
    vectors: Int,
    shifts: Int,
) abi("C"):
    neighbor_fill_range(
        fp(positions), fp(cell), ip(base_shift), ip(offsets), n, first, last,
        rx, ry, rz, self_interaction != 0, cutoff2, ip(pair_i), ip(pair_j),
        fp(distances), fp(vectors), ip(shifts),
    )


@export("mase_minimum_image")
def mase_minimum_image(
    vectors: Int,
    cell: Int,
    inverse: Int,
    result: Int,
    lengths: Int,
    n: Int,
    px: Int,
    py: Int,
    pz: Int,
) abi("C"):
    initialize_runtime()
    minimum_image(
        fp(vectors), fp(cell), fp(inverse), fp(result), fp(lengths),
        n, px, py, pz,
    )


@export("mase_lennard_jones")
def mase_lennard_jones(
    pair_i: Int,
    vectors: Int,
    npairs: Int,
    natoms: Int,
    sigma: Float64,
    epsilon: Float64,
    rc: Float64,
    ro: Float64,
    smooth: Int,
    energies: Int,
    forces: Int,
    stresses: Int,
) abi("C"):
    initialize_runtime()
    lennard_jones(
        ip(pair_i), fp(vectors), npairs, natoms, sigma, epsilon, rc, ro,
        smooth != 0, fp(energies), fp(forces), fp(stresses),
    )


@export("mase_morse")
def mase_morse(
    pair_i: Int,
    distances: Int,
    vectors: Int,
    npairs: Int,
    natoms: Int,
    epsilon: Float64,
    rho0: Float64,
    r0: Float64,
    rcut1: Float64,
    rcut2: Float64,
    energies: Int,
    forces: Int,
    stress: Int,
) abi("C"):
    initialize_runtime()
    morse(
        ip(pair_i), fp(distances), fp(vectors), npairs, natoms, epsilon,
        rho0, r0, rcut1, rcut2, fp(energies), fp(forces), fp(stress),
    )
