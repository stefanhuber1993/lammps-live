"""Geometry and per-frame state primitives shared by force fields, scenarios,
observables and the verifier.

Everything here is pure numpy -- no LAMMPS import -- so scenarios, energy
decompositions and observables are all unit-testable without spinning up a
simulation. That is deliberate: the whole point of splitting the force field's
energy expression out of the system class is to be able to evaluate it on a
handmade configuration and compare it against what LAMMPS computed.
"""
import itertools
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass(frozen=True)
class Box:
    """A simulation cell: bounds plus which axes are periodic.

    `periodic` drives both the LAMMPS `boundary` command and the minimum-image
    convention used by the pair list / energy decomposition, so the two can
    never disagree -- a class of bug the old per-system code was wide open to
    (each system hand-wrote both its `boundary p p f` string and its own
    `d[:, 0] -= Lx * round(...)` wrapping).
    """
    lo: tuple
    hi: tuple
    periodic: tuple = (False, False, False)

    @classmethod
    def cube(cls, side, periodic=(False, False, False)):
        h = side / 2.0
        return cls((-h, -h, -h), (h, h, h), periodic)

    @classmethod
    def centered(cls, lx, ly, lz, periodic=(False, False, False)):
        return cls((-lx / 2.0, -ly / 2.0, -lz / 2.0),
                   (lx / 2.0, ly / 2.0, lz / 2.0), periodic)

    @property
    def lengths(self):
        return tuple(h - l for l, h in zip(self.lo, self.hi))

    @property
    def center(self):
        return tuple(0.5 * (l + h) for l, h in zip(self.lo, self.hi))

    def boundary_command(self):
        return "boundary " + " ".join("p" if p else "f" for p in self.periodic)

    def region_command(self, name="box"):
        (xlo, ylo, zlo), (xhi, yhi, zhi) = self.lo, self.hi
        return (f"region {name} block {xlo} {xhi} {ylo} {yhi} {zlo} {zhi} "
                f"units box")

    def bounds_3d(self):
        """(xlo, xhi, ylo, yhi, zlo, zhi) -- the renderer's box-outline form."""
        return (self.lo[0], self.hi[0], self.lo[1], self.hi[1],
                self.lo[2], self.hi[2])

    def corners(self):
        """The eight corner points, for camera framing."""
        return np.array([(x, y, z) for x in (self.lo[0], self.hi[0])
                         for y in (self.lo[1], self.hi[1])
                         for z in (self.lo[2], self.hi[2])], dtype=float)

    def minimum_image(self, d):
        """Wrap separation vectors (..., 3) into the minimum image on every
        periodic axis. Returns a new array; non-periodic axes pass through."""
        d = np.asarray(d, dtype=float).copy()
        for axis, (per, length) in enumerate(zip(self.periodic, self.lengths)):
            if per and length > 0.0:
                d[..., axis] -= length * np.round(d[..., axis] / length)
        return d

    def wrap(self, positions):
        """Fold positions into [lo, hi) on every periodic axis."""
        p = np.asarray(positions, dtype=float).copy()
        for axis, (per, length) in enumerate(zip(self.periodic, self.lengths)):
            if per and length > 0.0:
                p[:, axis] = (p[:, axis] - self.lo[axis]) % length + self.lo[axis]
        return p


@dataclass
class FrameState:
    """One frame's particle state, as plain arrays.

    `directors` is the per-particle unit orientation vector (LAMMPS `mu` for
    dipole atom styles), or None for force fields whose particles have no
    orientation. Already normalized on construction by from_lammps.
    """
    positions: np.ndarray            # (N, 3)
    directors: np.ndarray = None     # (N, 3) unit vectors, or None
    types: np.ndarray = None         # (N,) int
    ids: np.ndarray = None           # (N,) LAMMPS atom ids
    box: Box = None

    def __len__(self):
        return len(self.positions)


@dataclass
class PairData:
    """The particle pairs within a cutoff, with their separation vectors.

    Built once per analysis frame and handed to every consumer (the force
    field's energy decomposition, observables), because building it is the
    expensive part -- the old code built a KD-tree twice per frame in
    mesomem_hex (once for the pulled bead, once for the whole-patch total) and
    once more per system for the RDF.

    Because energies come back per-pair, one pass serves both the "energy of
    this one bead" panel (select pairs touching it) and the "energy of the whole
    system" panel (sum everything). Previously those were two separate O(pairs)
    passes over the same configuration.
    """
    a: np.ndarray      # (M,) first index
    b: np.ndarray      # (M,) second index
    d: np.ndarray      # (M, 3) minimum-image r_a - r_b
    r: np.ndarray      # (M,) |d|
    # What fraction of the system's particles this list was built over. 1.0 for a
    # list over everything; below that the analysis subsampled to stay inside its
    # frame budget (see Analysis.MAX_PAIR_BEADS), and every extensive quantity
    # read off these pairs is short by a known factor: a pair sum by dilution^2,
    # since a pair survives only if BOTH its particles were drawn, and a
    # per-particle mean by dilution. Carried here rather than passed alongside so
    # that a consumer cannot be handed a diluted list without being told.
    dilution: float = 1.0

    def __len__(self):
        return len(self.r)

    def touching(self, index):
        """Boolean mask of the pairs that involve particle `index`."""
        return (self.a == index) | (self.b == index)

    @classmethod
    def empty(cls):
        z = np.zeros(0)
        return cls(np.zeros(0, dtype=int), np.zeros(0, dtype=int),
                   np.zeros((0, 3)), z)


def build_pairs(positions, cutoff, box=None):
    """Unique particle pairs separated by less than `cutoff`, minimum-imaged.

    Uses a KD-tree over the PERIODIC subspace when the box has periodic axes,
    then filters on the exact minimum-image 3D distance. Restricting the tree to
    the periodic axes is what makes one code path cover all three MesoMem
    geometries: a separation measured in a subspace is never larger than the
    full 3D separation, so the tree's answer is a superset of the true within-
    cutoff pairs and the exact filter afterwards makes it precise. (This is the
    trick the old sheet system used by hand -- a periodic 2D tree on (x, y) for
    a 3D monolayer -- generalized.)

    With no periodic axes the tree runs on all three coordinates directly.
    """
    positions = np.asarray(positions, dtype=float)
    if len(positions) < 2 or cutoff <= 0.0:
        return PairData.empty()

    from scipy.spatial import cKDTree

    per_axes = []
    if box is not None:
        per_axes = [i for i, p in enumerate(box.periodic)
                    if p and box.lengths[i] > 0.0]

    if per_axes:
        lengths = [box.lengths[i] for i in per_axes]
        # cKDTree's toroidal topology needs coordinates strictly inside [0, L).
        # The modulo alone is not enough: a coordinate a hair BELOW the lower
        # bound (LAMMPS hands back x = -9e-16 for an atom nominally at 0) wraps to
        # a value that rounds to exactly L, which cKDTree rejects. Under
        # periodicity L and 0 are the same point, so fold it back.
        cols = []
        for i in per_axes:
            w = (positions[:, i] - box.lo[i]) % box.lengths[i]
            w[w >= box.lengths[i]] = 0.0
            cols.append(w)
        pts = np.column_stack(cols)
        tree = cKDTree(pts, boxsize=lengths)
    else:
        tree = cKDTree(positions)

    pairs = tree.query_pairs(cutoff, output_type="ndarray")
    if not len(pairs):
        return PairData.empty()

    a, b = pairs[:, 0], pairs[:, 1]
    d = positions[a] - positions[b]
    if box is not None:
        d = box.minimum_image(d)
    r = np.linalg.norm(d, axis=1)
    keep = (r < cutoff) & (r > 1e-9)
    return PairData(a[keep], b[keep], d[keep], r[keep])


def normalize_rows(v, eps=1e-9):
    """Row-wise unit vectors, leaving (near-)zero rows untouched rather than
    dividing by ~0 -- LAMMPS hands back a zero `mu` for a particle whose dipole
    was never set."""
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.where(n < eps, 1.0, n)
    return v / n


def segment_distance(offsets, axis, length):
    """Distance from each offset vector to a line SEGMENT through the origin.

    The segment runs `length` along the unit `axis`, centred at the origin, so
    `offsets` are positions measured from the segment's centre (minimum-imaged by
    the caller if the cell is periodic). This is the geometry the rod pair style
    is built on -- the closest point on the axis is at s = clamp(offset.axis,
    +-L/2) -- and it is shared so the Python energy expression and the
    observables cannot drift from each other, or from the C++.

    `axis` is one (3,) vector or one per offset.
    """
    offsets = np.asarray(offsets, dtype=float)
    axis = np.atleast_2d(np.asarray(axis, dtype=float))
    along = np.einsum("ij,ij->i", offsets, np.broadcast_to(axis, offsets.shape))
    s = np.clip(along, -0.5 * length, 0.5 * length)
    return np.linalg.norm(offsets - s[:, None] * axis, axis=1)


def principal_normal(points):
    """Unit normal of the best-fit plane through `points` (the smallest
    principal axis of the centred second-moment matrix), sign-fixed to point
    along +z.

    Shared by the sheet/patch "keep the membrane face-on" housekeeping forces
    and the nematic-order observable, which independently derived it before.
    """
    p = np.asarray(points, dtype=float)
    q = p - p.mean(axis=0)
    _evals, evecs = np.linalg.eigh(q.T @ q)   # ascending
    n = evecs[:, 0]
    return -n if n[2] < 0.0 else n


class PlacementFailed(RuntimeError):
    """`random_points_min_separation` could not separate the points it was asked
    for. Its message names the way out (fewer particles, a bigger cell, or a
    smaller separation), because every one of those is a number in a playground
    file."""


def random_points_min_separation(n, box, min_sep, rng, rounds=80):
    """`n` uniformly random points in a periodic `box`, no two closer than
    `min_sep`. Returns (positions, rounds_used).

    WHY THIS EXISTS AND IS NOT `create_atoms random ... overlap`. LAMMPS will do
    exactly this, and its cost is the reason the remote demos were unpleasant to
    stand in front of: measured on this repo's 50,000-bead cell, `create_atoms
    random 50000 <seed> box overlap 0.9 maxtry 200` takes 47 SECONDS, against
    0.00 s for the same command with the overlap check left off. That one command
    was essentially the whole of a remote build -- and of a remote Reset, which is
    the same work again (44 s measured; see remote/client.py). Nothing about it is
    the GPU's fault or the network's: the insertion runs on the host and rejects
    candidates one at a time.

    THE METHOD is dart-throwing done in bulk. Sample all `n` at once, find every
    pair closer than `min_sep`, resample one member of each, repeat. At the
    densities these scenarios run (the 50k cell's exclusion spheres occupy about
    1.5% of it) roughly a tenth of the points conflict on the first pass and the
    rest converge in a handful more, so the whole thing is a few passes of numpy
    over the array. The distribution is the same one LAMMPS produces -- uniform,
    conditioned on the separation -- which is what makes this a speed change and
    not a physics one.

    The pair search is a cell hash at exactly `min_sep`, so a point's only possible
    conflicts are in its own cell and the 26 around it. Distances are
    minimum-imaged: the cell is periodic, and two points either side of a face are
    as close as they look.
    """
    lo = np.asarray(box.lo, dtype=float)
    lengths = np.asarray(box.lengths, dtype=float)
    pos = rng.random((int(n), 3)) * lengths
    if min_sep <= 0.0 or len(pos) < 2:
        return pos + lo, 0
    for used in range(1, int(rounds) + 1):
        crowded = _crowded_indices(pos, lengths, float(min_sep))
        if not len(crowded):
            return pos + lo, used
        pos[crowded] = rng.random((len(crowded), 3)) * lengths
    raise PlacementFailed(
        f"could not place {n} points at least {min_sep} apart in a "
        f"{lengths[0]:.1f} x {lengths[1]:.1f} x {lengths[2]:.1f} cell within "
        f"{rounds} rounds ({len(crowded)} still crowded). The cell is too full for "
        f"this separation: use fewer particles, a bigger cell (a lower volume "
        f"fraction), or a smaller overlap distance.")


def _crowded_indices(pos, lengths, min_sep):
    """Indices to resample: one member of every pair closer than `min_sep`.

    ONE member, the higher-indexed one, rather than both -- resampling both would
    throw away a point that has nothing wrong with it except its neighbour, which
    on a first pass with a tenth of the points in conflict roughly doubles the work
    per round for no gain in convergence.

    The cell grid is sized so a cell is at least `min_sep` across, which is what
    makes "own cell plus the 26 neighbours" a complete search. Cells are addressed
    by a single integer key and the points sorted by it, so each cell's members are
    a contiguous run; `members` is that run table, padded with -1, and the loop
    below walks the 27 offsets against it.
    """
    n = len(pos)
    ncell = np.maximum(np.floor(lengths / min_sep).astype(np.int64), 1)
    cell = np.floor(pos / lengths * ncell).astype(np.int64)
    # A point exactly on the upper face would index one cell past the end.
    np.clip(cell, 0, ncell - 1, out=cell)
    key = (cell[:, 0] * ncell[1] + cell[:, 1]) * ncell[2] + cell[:, 2]
    order = np.argsort(key, kind="stable")
    ucell, start, counts = np.unique(key[order], return_index=True,
                                     return_counts=True)
    width = int(counts.max())
    members = np.full((len(ucell), width), -1, dtype=np.int64)
    rank = np.arange(len(order), dtype=np.int64) - np.repeat(start, counts)
    members[np.repeat(np.arange(len(ucell), dtype=np.int64), counts), rank] = order

    idx = np.arange(n, dtype=np.int64)
    cut_sq = min_sep * min_sep
    crowded = []
    for offset in itertools.product((-1, 0, 1), repeat=3):
        nkey_cell = (cell + np.asarray(offset, dtype=np.int64)) % ncell
        nkey = ((nkey_cell[:, 0] * ncell[1] + nkey_cell[:, 1]) * ncell[2]
                + nkey_cell[:, 2])
        slot = np.minimum(np.searchsorted(ucell, nkey), len(ucell) - 1)
        # A neighbour cell that holds nothing is not in `ucell` at all, and
        # searchsorted lands it on some other cell -- so the key has to be checked
        # rather than trusted.
        occupied = ucell[slot] == nkey
        for k in range(width):
            j = np.where(occupied, members[slot, k], -1)
            live = j >= 0
            if not live.any():
                continue
            i, jj = idx[live], j[live]
            d = pos[jj] - pos[i]
            d -= lengths * np.round(d / lengths)
            close = (np.einsum("ij,ij->i", d, d) < cut_sq) & (i != jj)
            if close.any():
                crowded.append(np.maximum(i[close], jj[close]))
    if not crowded:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(crowded))


def hex_lattice_2d(n_cols, n_rows, a):
    """(N, 2) hexagonal lattice sites, centred on the origin.

    Rows run along x at spacing `a`; successive rows step a*sqrt(3)/2 in y and
    offset a/2 in x -- the arrangement that tiles a periodic rectangular cell of
    size (n_cols*a) x (n_rows*a*sqrt(3)/2), which is why the sheet scenario can
    size its periodic box straight from these counts.
    """
    dy = a * math.sqrt(3.0) / 2.0
    xs = np.arange(n_cols, dtype=float) * a
    pts = []
    for j in range(n_rows):
        xoff = (a / 2.0) if (j % 2) else 0.0
        pts.append(np.column_stack([xs + xoff, np.full(n_cols, j * dy)]))
    pts = np.vstack(pts)
    return pts - pts.mean(axis=0)


def hex_ring_2d(n_rings, a):
    """(N, 2) points of a hexagonal patch: a centre site plus `n_rings` closed
    rings of neighbours around it, at nearest-neighbour spacing `a`. Ring 1 is
    the classic six-neighbour hexagon (the 7-bead patch).

    Ordered centre-first, then ring by ring, and WITHIN each ring by increasing
    polar angle starting from +x. Scenarios build their bond lists from these
    indices (spokes 0-k, then the closed ring k -> k+1), so the angular ordering
    is part of the contract, not an accident of the loop.
    """
    e1 = np.array([a, 0.0])
    e2 = np.array([a * 0.5, a * math.sqrt(3.0) / 2.0])
    shells = {}
    for i in range(-n_rings, n_rings + 1):
        for j in range(-n_rings, n_rings + 1):
            if i == 0 and j == 0:
                continue
            # Hexagonal (graph) distance on the triangular lattice: sites at
            # distance k form the k-th closed ring.
            dist = max(abs(i), abs(j), abs(i + j))
            if dist <= n_rings:
                shells.setdefault(dist, []).append(i * e1 + j * e2)
    pts = [np.zeros(2)]
    for k in sorted(shells):
        ring = shells[k]
        ring.sort(key=lambda p: math.atan2(p[1], p[0]) % (2.0 * math.pi))
        pts.extend(ring)
    return np.array(pts, dtype=float)


# --- closed surfaces and closed chains ----------------------------------------
# The two geometries a vesicle-with-polymer scenario is made of. Both are here
# rather than in scenario.py for the same reason as the lattices above: they are
# pure geometry, and the tests that pin them (tests/test_vesicle_polymer.py) want
# nothing to do with LAMMPS.

# Twelve vertices of a regular icosahedron on the unit sphere, and its twenty
# faces. Written out rather than derived: it is a fixed object, and the winding of
# the faces (counter-clockwise seen from outside) is what makes every face normal
# below point OUT, which is the whole reason the vesicle's directors come out
# right without a per-face sign check.
_PHI = (1.0 + math.sqrt(5.0)) / 2.0
_ICO_VERTS = np.array([
    (-1, _PHI, 0), (1, _PHI, 0), (-1, -_PHI, 0), (1, -_PHI, 0),
    (0, -1, _PHI), (0, 1, _PHI), (0, -1, -_PHI), (0, 1, -_PHI),
    (_PHI, 0, -1), (_PHI, 0, 1), (-_PHI, 0, -1), (-_PHI, 0, 1),
], dtype=float)
_ICO_VERTS /= np.linalg.norm(_ICO_VERTS, axis=1, keepdims=True)
_ICO_FACES = np.array([
    (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
    (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
    (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
    (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
], dtype=int)


def icosphere_faces(nu):
    """Face centres and outward normals of a geodesic sphere, on the unit sphere.

    Returns (centres (20 nu^2, 3), normals (20 nu^2, 3)). Each of the
    icosahedron's twenty faces is split into nu^2 sub-triangles by subdividing
    its edges and projecting the result onto the sphere; the centres are where
    the beads go and the normals -- radial, since the surface is a sphere -- are
    their directors.

    A geodesic sphere rather than any of the easier constructions (a lat/long
    grid, a Fibonacci spiral) because what the membrane needs is a nearly
    UNIFORM triangulation: a lat/long grid crowds its poles by an unbounded
    factor, and a bilayer force field reads that as two very compressed patches
    that immediately buckle. Here the worst-case spacing varies by about 20% over
    the whole sphere, which the membrane simply relaxes out.
    """
    nu = max(1, int(nu))
    centres = []
    for face in _ICO_FACES:
        a, b, c = _ICO_VERTS[face]
        # Barycentric grid over the face, projected out to the sphere. p[i, j]
        # exists for i + j <= nu.
        i, j = np.meshgrid(np.arange(nu + 1), np.arange(nu + 1), indexing="ij")
        w = (nu - i - j) / nu
        grid = (w[..., None] * a + (i / nu)[..., None] * b
                + (j / nu)[..., None] * c)
        grid /= np.maximum(np.linalg.norm(grid, axis=-1, keepdims=True), 1e-12)
        # The nu^2 sub-triangles: nu(nu+1)/2 pointing the same way as the parent
        # face, nu(nu-1)/2 pointing the other way.
        up = [(i0, j0) for i0 in range(nu) for j0 in range(nu - i0)]
        down = [(i0, j0) for i0 in range(nu) for j0 in range(nu - i0 - 1)]
        for (i0, j0) in up:
            centres.append((grid[i0, j0] + grid[i0 + 1, j0] + grid[i0, j0 + 1]) / 3.0)
        for (i0, j0) in down:
            centres.append((grid[i0 + 1, j0] + grid[i0, j0 + 1]
                            + grid[i0 + 1, j0 + 1]) / 3.0)
    centres = np.array(centres, dtype=float)
    normals = centres / np.linalg.norm(centres, axis=1, keepdims=True)
    return centres, normals


def nearest_neighbour_distances(points):
    """Distance from every point to its nearest OTHER point, as an (N,) array.

    Pure numpy, and that is the point of it. This is reached from a scenario's
    `build`, and `build` runs on the REMOTE server, which is given numpy and
    nothing else -- scipy is a client-side dependency, because the analysis is
    what needs it (see `build_pairs`). A `cKDTree` on this path is a
    ModuleNotFoundError that arrives after the cluster has already allocated the
    node, which is exactly how it was found.

    Exact, not an estimate. The points are sorted by z and each is compared only
    against those within `band` of it on that axis. Since |dz| <= |dr|, anything
    the band excludes is further away than the band itself, so a distance that
    comes out BELOW the band is provably the nearest neighbour -- and if any does
    not, the band doubles and the sweep runs again. The band starts at a few
    times the spacing a uniform cloud of this size would have, which for the
    geodesic spheres this is called on settles in a pass or two.
    """
    p = np.asarray(points, dtype=float)
    n = len(p)
    if n < 2:
        return np.zeros(n)

    order = np.argsort(p[:, 2], kind="stable")
    q = p[order]
    z = q[:, 2]
    span = float(z[-1] - z[0])
    band = span * 4.0 / math.sqrt(n)
    if not band > 0.0:
        # A cloud flat in z (or one so thin the ratio underflows): compare
        # everything against everything and be done in one pass, rather than
        # doubling a band of zero forever.
        band = span
    best = np.zeros(n)

    while True:
        # Rows per block, so that one distance matrix stays a few million
        # entries however wide the band has grown.
        share = 1.0 if span <= 0.0 else min(1.0, 2.0 * band / span)
        rows = max(1, min(256, int(4.0e6 / max(1.0, n * share))))
        s = 0
        while s < n:
            e = min(s + rows, n)
            lo = int(np.searchsorted(z, z[s] - band, side="left"))
            hi = int(np.searchsorted(z, z[e - 1] + band, side="right"))
            diff = q[s:e, None, :] - q[None, lo:hi, :]
            d2 = np.einsum("ijk,ijk->ij", diff, diff)
            # lo <= s and hi >= e always (the band is non-negative), so every row
            # contains its own point, at the column its own index maps to.
            d2[np.arange(e - s), np.arange(s, e) - lo] = np.inf
            best[s:e] = np.sqrt(d2.min(axis=1))
            s = e
        # Every distance inside the band means every one of them is the true
        # nearest; a band as wide as the cloud has already compared everything.
        if best.max() < band or band >= span:
            break
        band *= 2.0

    out = np.empty(n)
    out[order] = best
    return out


@lru_cache(maxsize=32)
def icosphere_spacing(nu):
    """Mean nearest-neighbour distance between the faces of `icosphere_faces(nu)`
    on the UNIT sphere -- what a chosen bead spacing has to be divided by to get
    the vesicle's radius.

    MEASURED, not approximated. The flat-triangle estimate (1/(nu sqrt(3)), times
    the icosahedron's edge) is several per cent out, because projecting the
    subdivision onto the sphere stretches the sub-triangles unevenly -- and this
    number scales the radius of every vesicle built from it, so a few per cent
    here is a membrane that starts a few per cent off its relaxed packing.

    The spread it is a mean OF is not small: the faces nearest one of the
    icosahedron's own twelve vertices are packed about 1.5x more tightly than the
    ones at a parent face's centre, which is the price of this construction and is
    still far better than the unbounded crowding a lat/long grid puts at its
    poles. The membrane is a fluid and relaxes it out within a few hundred steps.

    Cached, because `VesiclePolymer.radius` is asked for it on every camera fit and
    every ring-count query, and at the shipped size this is a sweep over 18,000
    points.
    """
    nu = max(1, int(nu))
    centres, _ = icosphere_faces(nu)
    if len(centres) < 2:
        return 1.0
    return float(nearest_neighbour_distances(centres).mean())


def lattice_ring(nx, ny, nz):
    """A closed self-avoiding walk visiting every site of an nx x ny x nz cubic
    lattice exactly once, as (N, 3) integer coordinates in walk order.

    A Hamiltonian CYCLE, so the chain it describes is a ring: consecutive sites
    (and the last with the first) are one lattice step apart, and no two sites
    coincide. That is exactly the guarantee a polymer's initial configuration
    needs and the one a random walk cannot give -- every bond is at the bond
    length, and no two beads are closer than one, so the chain can be handed
    straight to a FENE bond and a repulsive core with nothing to push apart
    first.

    `ny` and `nx * ny` must be even; the construction needs the return corridor
    to come out on the right side (see below), and the standard fix is to make
    the grid even. Raises rather than returning something subtly not closed.
    """
    nx, ny, nz = int(nx), int(ny), int(nz)
    if min(nx, ny, nz) < 2:
        raise ValueError(f"a ring needs at least 2 sites per axis, got {nx}x{ny}x{nz}")
    if ny % 2:
        raise ValueError(f"lattice_ring needs an even ny (got {ny}): with an odd "
                         f"one the snake ends on the wrong side of the grid and "
                         f"the walk cannot close")
    # 1. A Hamiltonian cycle on the nx x ny FLOOR. Row 0 is the outward leg;
    #    rows 1.. are snaked over columns 1.., and column 0 is the return
    #    corridor that brings the walk back to the start.
    floor = [(x, 0) for x in range(nx)]
    for y in range(1, ny):
        cols = range(nx - 1, 0, -1) if y % 2 else range(1, nx)
        floor += [(x, y) for x in cols]
    floor += [(0, y) for y in range(ny - 1, 0, -1)]
    # 2. Extrude it: walk the floor cycle, running up and down the z column over
    #    each cell in turn. The cycle closes because the floor cycle has an even
    #    number of cells, so the last column is descended and ends at z = 0
    #    next to where the walk began.
    out = []
    for k, (x, y) in enumerate(floor):
        zs = range(nz) if k % 2 == 0 else range(nz - 1, -1, -1)
        out += [(x, y, z) for z in zs]
    return np.array(out, dtype=int)
