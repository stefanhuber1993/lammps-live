"""Which beads are stuck to which -- the aggregate each one belongs to, held
still enough to paint with.

WHAT IT IS FOR: the third bead colouring. The director banding says which way a
bead points and the energy ramp says how bound it is; neither says WHAT IT IS
PART OF. On the assembly box that is the whole story -- a gas coarsens into
patches, patches into membranes, membranes merge -- and it is invisible in a
picture where every aggregate is painted the same. Give each one its own colour
and the coarsening reads directly: the count of colours falls, the patches of
colour grow, and a merge is two colours becoming one.

THREE PROBLEMS, IN THE ORDER THEY BITE:

  1. WHAT IS A CLUSTER. Connected components of the "within `cutoff` of each
     other" graph, which is the standard cluster analysis for an aggregating
     fluid (LAMMPS spells it `compute cluster/atom`) and is the only definition
     that costs O(N). A graph CUT -- spectral, modularity, k-means on the
     coordinates -- would answer a different and worse question here: it insists
     on splitting a single connected membrane into pieces, because a cut always
     exists, whereas what the eye wants to know is exactly whether the material
     is one piece or two. Connectivity is also the physically meaningful line:
     the aggregates in this model ARE connected components of the attraction
     range.

  2. PERIODICITY. A cell that wraps has no "edge", and an aggregate straddling a
     face is one object, not two. Every separation goes through the minimum image
     (Box.minimum_image), and the cell list wraps with it, so a membrane sitting
     across a seam comes out as one cluster with one colour -- which is the whole
     point of drawing the periodic images next to it.

  3. IDENTITY OVER TIME. This is the hard one, and it is not a clustering problem
     at all: components have no names. Recomputing them every frame gives a
     labelling that is correct and completely unstable -- roots renumber, a surface bead
     rattles in and out of the cutoff, two aggregates touch for one frame -- and
     painting straight from it strobes. `ClusterTracker` below is the part that
     turns a sequence of anonymous labellings into per-bead COLOUR SLOTS that
     stay put: see its docstring for the three mechanisms.

Nothing here is physics, and nothing here feeds back into the simulation: it is
one more drawn quantity, filtered like the drawn positions and the drawn
energies (see smoothing.py), and it runs on the SMOOTHED coordinates for the same
reason they do -- the thermal rattle is what pushes a bead across the cutoff.
"""
import math

import numpy as np


# The cutoff, as a multiple of a bead DIAMETER (2 * spec.atom_radius_A). Two
# beads are "in contact" if their centres are inside this. Quoted in diameters
# rather than as an absolute length so it means the same thing on the paper's
# sigma = 1 beads as it would on anything else, and so no playground has to carry
# a number for it.
#
# 1.25 sits in the gap the model leaves: the condensed phase packs at 0.8-1.0
# sigma (the sheet's hexagonal lattice is a = 0.8) and its second shell is out at
# ~1.4, while a bead in the phi = 0.1 gas has its nearest neighbour around 2.
# So this joins everything touching, joins nothing that is merely nearby, and has
# most of a bead diameter of slack either side before it would start doing either.
CONTACT_CUTOFF_DIAMETERS = 1.25

# How many distinct colours there are to hand out. This layer never sees a colour,
# so it holds only the COUNT; the colours themselves are ui/theme.CLUSTER_COLORS,
# which must offer exactly this many (a test asserts it). Ten is what a
# categorical palette can hold with every pair still telling itself apart on a
# small, fogged, depth-blurred bead; past that the extra colours are
# indistinguishable and the picture is worse for having them.
#
# TEN COLOURS DOES NOT MEAN TEN CLUSTERS. A dilute cell holds hundreds of
# aggregates, and naming ten of them while greying the rest answers the question
# for a twentieth of the picture. So colours REPEAT -- what is guaranteed is not
# that a colour identifies one aggregate, but that it identifies one aggregate
# LOCALLY: no cluster ever wears the colour of any of its nearest neighbours (see
# NEIGHBOUR_COUNT). That is the cartographer's guarantee, and it is the one the
# eye actually uses -- four colours suffice to draw every country on a map
# distinctly, because nobody confuses two reds on opposite sides of it.
PALETTE_SLOTS = 10

# How many of the nearest other clusters a cluster must differ in colour from.
# Fewer than PALETTE_SLOTS, always, so a legal colour is guaranteed to exist and
# the assignment can never fail back to greying something.
#
# Four, because the constraint is SYMMETRISED (see neighbour_graph) and so costs
# about twice its nominal count in forbidden colours -- eight of ten, typically,
# leaving two to load-balance across. Four is also about the number of blobs that
# can plausibly be read as touching one in 3D, which is what "adjacent" has to
# mean here: this layer has no camera and cannot know what overlaps what on
# screen. Raising it tightens the guarantee and flattens the colour balance;
# lowering it does the reverse.
NEIGHBOUR_COUNT = 4

# How far out a cluster looks for the neighbours it must differ from, in
# multiples of the typical spacing between clusters -- (cell volume / cluster
# count)^(1/3). Because it is quoted in spacings, the number of candidates it
# turns up is the same at any density (about ninety), which is what makes the
# search cost linear in the cluster count rather than quadratic.
#
# It is a candidate filter, not the constraint -- that is still the
# NEIGHBOUR_COUNT nearest of whatever it finds. But it is also, deliberately, the
# scale at which two clusters stop being worth distinguishing: a cluster whose
# nearest neighbour is more than a couple of typical spacings away comes back
# with nothing and is free to take any colour, which is right. Sharing a colour
# with something that far off is the case sharing is FOR.
NEIGHBOUR_REACH = 1.5

# A ceiling on how many clusters are coloured at once. A cost guard only: the
# whole assignment is linear in the cluster count, measured on a 50k cell at
# 62 ms for a thousand clusters, 100 ms for 2400 and 129 ms for 3700, so this
# bounds the worst frame at roughly 200 ms rather than the 400 the same cell
# would reach if every group of four beads in it were its own aggregate. No scene
# here comes near it -- the dilute remote cell runs at one to two thousand -- and
# what is dropped is always the smallest, since clusters are claimed
# largest-first.
MAX_TRACKED_CLUSTERS = 6000

# The smallest group of beads worth giving a colour to, whatever the scene. Below
# four a "cluster" is a monomer, a pair or a triangle -- which every gas is mostly
# made of by count, and which says nothing about grouping that the picture does
# not already show.
MIN_CLUSTER_SIZE = 4

# ... and never more than this fraction of the scene, which only ever binds on a
# tiny one: four loose beads need something to be loose IN, and on the seven-bead
# patch there is no such thing -- the patch IS the aggregate.
MIN_CLUSTER_FRACTION = 1.0 / 3.0

# THE REAL THRESHOLD IS DENSITY-DEPENDENT, and getting that wrong was the first
# version's mistake. A group of six beads is a remarkable thing in a dilute gas
# and pure noise in a dense one: at the contact cutoff a random gas glues beads
# together for free, and how big those free clumps get climbs violently with
# density, because it is the run-up to percolation. Measured on random
# configurations of 30k beads, as the size below which 95% of all
# beads-in-chance-clumps sit:
#
#     phi    nbar     chance clumps reach
#     0.01   0.16           2
#     0.02   0.31           3
#     0.04   0.63           5
#     0.07   1.09           9
#     0.10   1.56          21
#     0.13   2.03          71     (percolating: the gas is one cluster)
#
# where nbar is the mean number of other beads inside one bead's contact cutoff,
# rho * (4/3) pi rc^3 -- the only parameter the chance-clump distribution has.
# Those sizes are close to a straight line in log s against nbar, hence the two
# constants below; the fit is used rather than the table because no playground
# sits on a sampled density.
#
# What it buys, at the two the playgrounds do sit on. The paper's box (nbar 1.53)
# gets a threshold of 24, so a freshly randomised cell -- whose biggest chance
# clump is about 14 -- comes up entirely grey and lights up as real aggregates
# grow past it. The dilute remote cell (nbar 0.31) gets the floor of 4, so the
# small aggregates that form at that density are all named instead of the picture
# going uniformly grey. One constant could not do both: 24 makes the dilute cell
# useless and 4 turns the dense one into confetti, which is exactly the pair of
# failures this replaced.
CHANCE_CLUMP_SCALE = 1.3
CHANCE_CLUMP_GROWTH = 1.9

# How many LABELLINGS (not frames -- see ClusterTracker.RECLUSTER_EVERY) a bead
# has to disagree with its committed slot before the slot changes. This is the
# hysteresis that kills the flicker of a surface bead skipping in and out of the
# cutoff: it has to leave and stay left.
COMMIT_HOLD = 3

# A component keeps a slot it overlaps by at least this fraction of its own
# beads. Below it the overlap is an accident -- a bridge that formed this
# labelling -- and the component is treated as new. Generous on purpose: a
# cluster that grows by half is still that cluster.
MATCH_MIN_OVERLAP = 0.30

_MAX_LABEL_ITERS = 64


def contact_cutoff(bead_radius):
    """The contact distance for beads of this radius. `None` -> the mesoscale
    default of half a sigma, which is what every 3D playground here uses."""
    return CONTACT_CUTOFF_DIAMETERS * 2.0 * float(bead_radius or 0.5)


# ---- the geometry: connected components of the contact graph -----------------

def _cell_grid(positions, box, cutoff):
    """Bin the beads into a cell list of cells at least `cutoff` across.

    Returns (cell coords per bead, counts per cell, sorted bead order, cell start
    offsets, cells per axis, whether each axis wraps). Periodic axes are
    binned over the CELL, non-periodic ones over the beads' own extent, so a
    patch floating in a large box costs cells only where it actually is.
    """
    n = len(positions)
    lo = np.empty(3)
    span = np.empty(3)
    wrap = np.zeros(3, dtype=bool)
    for axis in range(3):
        periodic = bool(box.periodic[axis]) and box.lengths[axis] > 0.0
        wrap[axis] = periodic
        if periodic:
            lo[axis] = box.lo[axis]
            span[axis] = box.lengths[axis]
        else:
            lo[axis] = positions[:, axis].min()
            span[axis] = max(float(np.ptp(positions[:, axis])), 1e-9)

    # At least one cell, and never so many that a cell is thinner than the
    # cutoff -- the 3x3x3 stencil below only reaches one cell out, so a thinner
    # cell would silently drop contacts.
    ncell = np.maximum(1, np.floor(span / cutoff).astype(int))
    # Two cells on a wrapping axis means the +1 and -1 neighbours are the same
    # cell, which the half-stencil then visits twice: harmless for connectivity
    # (a duplicated edge joins what was already joined) and not worth a branch.

    frac = (positions - lo) / span
    if np.any(wrap):
        frac[:, wrap] %= 1.0
    idx = np.clip((frac * ncell).astype(int), 0, ncell - 1)
    flat = (idx[:, 0] * ncell[1] + idx[:, 1]) * ncell[2] + idx[:, 2]

    order = np.argsort(flat, kind="stable").astype(np.int32)
    counts = np.bincount(flat, minlength=int(ncell.prod()))
    starts = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=starts[1:])
    return idx, counts, order, starts, ncell, wrap


def _ragged_gather(bead, k, start, order):
    """Expand "bead b pairs with the k[b] beads stored from start[b]" into flat
    (i, j) index arrays. The one loop-free way to walk a cell list in numpy: the
    within-group counter is an arange minus the group's own offset into it."""
    total = int(k.sum())
    if total == 0:
        return None, None
    i = np.repeat(bead, k)
    base = np.cumsum(k) - k
    within = np.arange(total, dtype=np.int64) - np.repeat(base, k)
    j = order[np.repeat(start, k) + within]
    return i, j


def contact_pairs(positions, box, cutoff):
    """Every pair of beads within `cutoff`, minimum-image, as (i, j) arrays.

    Pairs may repeat and may run either way round; both are harmless downstream
    (a duplicate edge joins what is already joined) and de-duplicating them would
    cost more than the pairs do. A bead is never paired with ITSELF, which is not
    free to promise: on a cell grid only one or two cells across -- which happens
    whenever the cutoff is a sizeable fraction of the cell, and always when this
    is run over cluster centres rather than beads -- a wrapped stencil offset
    lands back on the cell it came from, and the naive product of the two
    includes every bead against itself. Connectivity does not care, but a caller
    ranking each point's nearest OTHERS very much does.

    THE COORDINATES ARE COMPARED IN SINGLE PRECISION, one axis at a time. A
    stencil of cutoff-sized cells offers about six candidate pairs for every real
    one -- the sphere of radius rc inside the 27 rc^3 the stencil reaches -- so
    this loop is the whole cost of a labelling, and it is memory-bound rather
    than arithmetic-bound. Walking the axes as three contiguous 1-D columns
    instead of gathering (N, 3) rows halves the traffic again. Single precision
    is worth ~1e-5 sigma of slop on which side of the cutoff a bead falls, which
    is thermal noise next to the hysteresis this feeds.
    """
    positions = np.asarray(positions, dtype=float)
    n = len(positions)
    if n < 2:
        return np.empty(0, np.int32), np.empty(0, np.int32)
    idx, counts, order, starts, ncell, wrap = _cell_grid(positions, box, cutoff)

    # Half of the 3x3x3 stencil -- the lexicographically positive offsets --
    # plus the cell itself. The other half is the same pairs the other way round.
    offsets = [(0, 0, 0)]
    offsets += [(dx, dy, dz)
                for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
                if (dx, dy, dz) > (0, 0, 0)]

    cols = [np.ascontiguousarray(positions[:, a], dtype=np.float32)
            for a in range(3)]
    lengths = [np.float32(box.lengths[a]) for a in range(3)]
    halves = [np.float32(0.5 * box.lengths[a]) for a in range(3)]
    beads = np.arange(n, dtype=np.int32)
    cut2 = np.float32(cutoff * cutoff)

    # Whether a wrapped offset can land back on its own cell (see the docstring):
    # only then is the self-pair filter below worth its comparison.
    folds = bool(np.any(wrap & (ncell < 3)))
    out_i, out_j = [], []
    for off in offsets:
        nb = idx + np.asarray(off)
        keep = None
        for axis in range(3):
            if wrap[axis]:
                nb[:, axis] %= ncell[axis]
            else:
                inside = (nb[:, axis] >= 0) & (nb[:, axis] < ncell[axis])
                keep = inside if keep is None else (keep & inside)
        nflat = (nb[:, 0] * ncell[1] + nb[:, 1]) * ncell[2] + nb[:, 2]
        if keep is None:
            k = counts[nflat]
        else:
            nflat = np.where(keep, nflat, 0)
            k = np.where(keep, counts[nflat], 0)
        i, j = _ragged_gather(beads, k, starts[nflat], order)
        if i is None:
            continue
        if off == (0, 0, 0):
            same = i < j            # each within-cell pair once, and no self-pairs
            i, j = i[same], j[same]
            if not len(i):
                continue
        elif folds:
            other = i != j          # the wrapped-onto-itself case
            i, j = i[other], j[other]
            if not len(i):
                continue
        r2 = None
        for axis in range(3):
            d = cols[axis][j]
            d -= cols[axis][i]
            if wrap[axis]:
                # The minimum image, without the round(): a separation coming out
                # of a one-cell stencil is never more than a box away, so the two
                # sides are the only cases there are.
                d -= lengths[axis] * (d > halves[axis])
                d += lengths[axis] * (d < -halves[axis])
            d *= d
            r2 = d if r2 is None else r2 + d
        close = r2 < cut2
        if np.any(close):
            out_i.append(i[close])
            out_j.append(j[close])
    if not out_i:
        return np.empty(0, np.int32), np.empty(0, np.int32)
    return np.concatenate(out_i), np.concatenate(out_j)


def connected_labels(n, i, j):
    """Connected components of the graph on n nodes with edges (i, j).

    Shiloach-Vishkin: every node offers the smallest label it can see, that
    minimum is hooked onto the ROOT of the tree the node currently sits in, and
    then every pointer is followed to its root again. Repeat until nothing moves.

    HOOKING ONTO THE ROOT RATHER THAN ONTO THE NODE is the whole algorithm. Write
    the minimum onto the node itself and the label crawls one edge per round, so a
    membrane 200 beads across takes ~200 rounds to agree on a colour; hooking onto
    the root contracts whole trees at once and it takes four. Measured on a 50k
    box: 30 rounds and 40 ms the naive way, 4 rounds and 9 ms this way.

    Returns a label per node; nodes of a component share one, and it is the
    smallest node index in that component.
    """
    label = np.arange(n, dtype=np.int32)
    if not len(i):
        return label
    # Both directions once, grouped by source. The grouping is a property of the
    # graph, not of the labels, so it is sorted once and reused every round.
    src = np.concatenate([i, j]).astype(np.int32, copy=False)
    dst = np.concatenate([j, i]).astype(np.int32, copy=False)
    order = np.argsort(src, kind="stable")
    src, dst = src[order], dst[order]
    first = np.flatnonzero(np.r_[True, src[1:] != src[:-1]])
    nodes = src[first]

    for _ in range(_MAX_LABEL_ITERS):
        # What each node can see, including itself...
        offer = np.minimum(label[nodes], np.minimum.reduceat(label[dst], first))
        # ... hooked onto its root, so a whole tree moves together. Grouping the
        # offers by root is a second sort, over the nodes rather than the edges.
        roots = label[nodes]
        by_root = np.argsort(roots, kind="stable")
        roots, offer = roots[by_root], offer[by_root]
        heads = np.flatnonzero(np.r_[True, roots[1:] != roots[:-1]])
        new = label.copy()
        new[roots[heads]] = np.minimum(new[roots[heads]],
                                       np.minimum.reduceat(offer, heads))
        while True:                      # follow every pointer to its root
            jumped = new[new]
            if np.array_equal(jumped, new):
                break
            new = jumped
        if np.array_equal(new, label):
            return label
        label = new
    return label


def cluster_labels(positions, box, cutoff):
    """Connected components of the contact graph, one label per bead."""
    positions = np.asarray(positions, dtype=float)
    return connected_labels(len(positions), *contact_pairs(positions, box, cutoff))


def cluster_geometry(positions, labels, rows, box):
    """Centre and radius of each cluster in `rows`, minimum-image safe.

    Used to decide which clusters are NEAR each other, which is what stops two of
    them being handed the same colour. Two numbers per cluster rather than the
    full bead set, because "are these two blobs adjacent" is a coarse question and
    the answer has to be cheap at a thousand of them.

    THE CENTRE IS A CIRCULAR MEAN on every wrapping axis, the same construction
    RandomFill.housekeeping uses and for the same reason: an ordinary mean is
    meaningless on a torus, and a cluster straddling a face would average to the
    empty middle of the box and be declared adjacent to everything there. Map the
    coordinate onto the unit circle, average the unit vectors, read the angle
    back, and the answer is translation-covariant as a centre of mass must be.

    The radius is the RMS distance from that centre -- a radius of gyration, not
    an enclosing radius, so one stray bead on a tether does not make a compact
    droplet claim half the cell. It is what lets a large flat membrane count its
    neighbours from its own SURFACE rather than from its middle, which for an
    extended object is most of the difference between a useful answer and none.
    """
    positions = np.asarray(positions, dtype=float)
    n_rows = len(rows)
    rank = np.full(labels.max() + 1 if len(labels) else 1, -1, dtype=np.int64)
    rank[rows] = np.arange(n_rows)
    row_of_bead = rank[labels]
    keep = row_of_bead >= 0
    row_of_bead, pts = row_of_bead[keep], positions[keep]
    counts = np.bincount(row_of_bead, minlength=n_rows).astype(float)
    counts[counts == 0] = 1.0

    centre = np.empty((n_rows, 3))
    for axis in range(3):
        column = pts[:, axis]
        length = box.lengths[axis]
        if box.periodic[axis] and length > 0.0:
            theta = 2.0 * math.pi * (column - box.lo[axis]) / length
            mean_c = np.bincount(row_of_bead, np.cos(theta), n_rows) / counts
            mean_s = np.bincount(row_of_bead, np.sin(theta), n_rows) / counts
            angle = np.arctan2(mean_s, mean_c) % (2.0 * math.pi)
            centre[:, axis] = box.lo[axis] + length * angle / (2.0 * math.pi)
        else:
            centre[:, axis] = np.bincount(row_of_bead, column, n_rows) / counts

    offset = box.minimum_image(pts - centre[row_of_bead])
    spread = np.bincount(row_of_bead, np.einsum("ij,ij->i", offset, offset), n_rows)
    return centre, np.sqrt(spread / counts)


def neighbour_graph(centre, radius, box, count):
    """Each cluster's nearest others, as a symmetric graph in CSR form:
    (starts, other, separation), sorted by source.

    Separation is measured between SURFACES -- centre distance less both radii of
    gyration -- so a small droplet sitting against a big membrane is that
    membrane's neighbour, which is the whole point: centre-to-centre would rank a
    droplet halfway across the cell as closer than one touching the membrane's
    edge.

    A rank rather than a distance threshold, deliberately. A threshold needs a
    length, and there is no length here that means the same thing on a 20-sigma
    cell holding one membrane and a 109-sigma cell holding a thousand droplets;
    "the four nearest" means the same thing on both. It is also the more stable of
    the two over time -- a cluster drifting past a fixed radius flips in and out
    of the constraint, while a rank changes only when clusters overtake each
    other.

    SYMMETRISED, which is not a detail. "The four nearest" is not a symmetric
    relation: a cluster at the edge of a crowd of them counts a distant one among
    its four, while that one has four closer of its own. Constraining only the
    direction that happens to be listed lets exactly those fringe pairs come out
    the same colour, sitting next to each other -- which was the visible bug in
    the directed version.

    WHY THIS DOES NOT COMPARE EVERY PAIR. It did, and the cost is quadratic in the
    cluster count: 31 ms at a thousand clusters, 127 ms at two and a half, on top
    of a labelling that already costs 76 ms at 50k beads -- and the dilute cell
    this exists for is exactly the one that holds thousands. So the candidates come
    from the same cell-list pair search the beads themselves go through, run over
    the cluster CENTRES at a reach of a few typical cluster spacings. The ranking
    is unchanged -- still the four nearest -- because the reach only limits which
    pairs are ranked, and it is set from the cell volume and the cluster count, so
    it is about ninety candidates per cluster whatever the density. A cluster
    isolated enough to have none simply carries no constraints, which is correct:
    there is nothing near it to be confused with.
    """
    k = len(centre)
    count = min(count, k - 1)
    empty = (np.zeros(k + 1, np.int64), np.zeros(0, np.int64), np.zeros(0))
    if count <= 0:
        return empty
    volume = float(np.prod(box.lengths))
    spacing = (volume / k) ** (1.0 / 3.0) if volume > 0.0 else 1.0
    # Plus the two largest radii, because the reach is applied to CENTRES while
    # the ranking is by surfaces: a big membrane's nearest neighbour by surface
    # can be several of its own radii away by centre.
    reach = NEIGHBOUR_REACH * spacing + 2.0 * float(radius.max(initial=0.0))
    src, dst = contact_pairs(centre, box, reach)
    if not len(src):
        return empty

    # Both directions, so every cluster ranks every candidate it takes part in.
    u = np.concatenate([src, dst]).astype(np.int64)
    v = np.concatenate([dst, src]).astype(np.int64)
    delta = box.minimum_image(centre[v] - centre[u])
    gap = np.sqrt(np.einsum("ij,ij->i", delta, delta)) - radius[u] - radius[v]

    # Nearest first within each source. Sorting on v as well is what makes the
    # duplicate pairs the cell list is allowed to emit land next to each other,
    # so one comparison drops them -- a duplicate would otherwise eat one of the
    # four constraint slots and give the colour away.
    order = np.lexsort((v, gap, u))
    u, v, gap = u[order], v[order], gap[order]
    if len(u) > 1:
        keep = np.r_[True, (u[1:] != u[:-1]) | (v[1:] != v[:-1])]
        u, v, gap = u[keep], v[keep], gap[keep]
    starts = np.flatnonzero(np.r_[True, u[1:] != u[:-1]])
    within = np.arange(len(u)) - np.repeat(starts, np.diff(np.r_[starts, len(u)]))
    near = within < count
    u, v, gap = u[near], v[near], gap[near]

    # ... and now symmetrise the truncated relation, and group by source.
    u, v = np.concatenate([u, v]), np.concatenate([v, u])
    gap = np.concatenate([gap, gap])
    order = np.argsort(u, kind="stable")
    out = np.zeros(k + 1, dtype=np.int64)
    np.cumsum(np.bincount(u, minlength=k), out=out[1:])
    return out, v[order], gap[order]


# ---- the identity: anonymous labellings -> stable colour slots ---------------

class ClusterTracker:
    """Per-bead colour slots that survive the clusters being recomputed.

    A labelling is anonymous: it says which beads are together, not which
    aggregate this is. Painted straight, the picture strobes -- roots renumber
    when a bead leaves, a surface bead flickers across the cutoff, two membranes
    brush past each other for a frame. Four mechanisms, each aimed at one of
    those:

      MATCHING answers "which of the things I was already painting is this?".
        Components are matched by overlap to the slots their beads already hold,
        LARGEST FIRST: a cluster that gained or lost beads keeps its colour, and
        when two merge the survivor is the LARGER one's colour -- so the big
        aggregate on screen never changes, and only the smaller one that ran into
        it does. A split is the same rule read backwards: the bigger fragment
        keeps the colour, the smaller becomes new.

      SEPARATION answers "and what about the ones I have never seen before?".
        There are ten colours and a dilute cell holds hundreds of aggregates, so
        colours have to repeat; what must not repeat is a colour NEAR one that
        already wears it (neighbour_graph, and _claim). Among the colours that
        clears, the least-used one wins, so the palette spreads over the box
        rather than one hue taking a third of it.

      HYSTERESIS answers "is it really?". A bead's slot only changes once it has
        disagreed for COMMIT_HOLD labellings running, so a bead rattling across
        the cutoff, or a one-frame bridge between two aggregates, never reaches
        the screen at all. It costs the truth about half a second of lag, which
        on a process that takes minutes is not a cost.

      RATE answers "how often". Aggregates coarsen over thousands of timesteps;
        there is nothing in a labelling recomputed every frame that was not in
        the one before it. Recomputing every RECLUSTER_EVERY frames spends a
        tenth of the effort and, because the slots between recomputes are simply
        held, changes nothing about the picture. The colour crossfade that
        actually smooths the transition runs every frame, in the renderer -- it
        is an animation, not an analysis.

    The slots handed back are indices into whatever palette the caller draws
    with, or -1 for a bead in nothing worth naming -- a monomer, a pair, a
    triangle (MIN_CLUSTER_SIZE). Nothing here knows what colour a slot is.
    """

    # Frames between labellings, and the bead count one labelling is worth. A
    # labelling costs O(N) -- measured at 2.5 ms for the 1500-bead assembly box,
    # 8 ms at 6000, and 90 ms at the remote box's 50k with a thousand clusters in
    # it (of which ~10 ms is the colour assignment; the rest is finding the
    # components at all). So a fixed cadence would be free on the small scenes and
    # ruinous on the large one. Pacing it by size instead spends a bounded ~2-3 ms
    # per frame everywhere: every 5 frames up to 7500 beads, every 33 at 50k. What
    # a fixed cadence would buy back is the one frame in 33 that carries the whole
    # labelling -- 90 ms on the remote box as measured here, and 185 ms once it has
    # coarsened into clumps -- which was a visible hitch about every second and a
    # half.
    #
    # THAT IS NO LONGER A FRAME'S PROBLEM, which is what the pacing above is now
    # for rather than the whole answer. On the remote playground the labelling runs
    # on a thread of its own (remote/client.py, FrameAnalysis), so what the pacing
    # buys there is a bounded share of a core -- 185 ms every 1.6 s -- instead of a
    # bounded share of a frame. The stepper thread, which this comment used to name
    # as the fix, is NOT one: the app joins it before it draws, so work parked there
    # is a frame the window does not get.
    RECLUSTER_EVERY = 5
    BEADS_PER_LABELLING = 1500

    def __init__(self, cutoff, n_slots=PALETTE_SLOTS, min_size=MIN_CLUSTER_SIZE):
        """`min_size` is the ABSOLUTE floor; the scene-relative cap is applied on
        top of it per labelling (see _min_size)."""
        self.n_slots = int(n_slots)
        self.cutoff = float(cutoff)
        self.min_size = int(min_size)
        self.reset()

    def reset(self):
        """Forget everything. The next call reseeds -- committed straight away,
        with no hysteresis to work through, so a rebuilt system comes up in its
        real colours rather than fading in from the last one's."""
        self._slots = None            # (N,) committed slot per bead, -1 = gas
        self._hold = None             # (N,) labellings the pending slot has held
        self._last_used = np.zeros(self.n_slots, dtype=np.int64)
        self._tick = 0
        self._frame = -1

    def slots(self, positions, box, frame):
        """The committed slot of every bead, in the caller's own bead order.

        `frame` is the app's frame counter: it is what paces the recompute, and
        passing the same one twice (two readouts in one frame) costs nothing.
        """
        positions = np.asarray(positions, dtype=float)
        n = len(positions)
        if self._slots is None or len(self._slots) != n:
            self._seed(n)
        elif frame - self._frame < self._period(n) and frame >= self._frame:
            return self._slots.copy()
        if n:
            self._frame = frame
            self._advance(positions, box)
        return self._slots.copy()

    def _min_size(self, n, box):
        """The smallest component this scene will call a cluster: the bigger of
        the absolute floor and the chance-clump size at this cell's density, and
        never so big that it would grey out the whole scene."""
        volume = float(np.prod(box.lengths))
        density = n / volume if volume > 0.0 else 0.0
        n_bar = density * (4.0 / 3.0) * math.pi * self.cutoff ** 3
        chance = CHANCE_CLUMP_SCALE * math.exp(CHANCE_CLUMP_GROWTH * n_bar)
        scene = max(self.min_size, int(n * MIN_CLUSTER_FRACTION))
        return int(min(max(self.min_size, round(chance)), scene))

    def _period(self, n):
        """Frames to wait before labelling again, at this bead count."""
        return max(self.RECLUSTER_EVERY, n // self.BEADS_PER_LABELLING)

    def _seed(self, n):
        self._slots = np.full(n, -1, dtype=np.int32)
        self._hold = np.zeros(n, dtype=np.int32)
        self._last_used[:] = 0
        self._tick = 0
        self._frame = -1

    def _advance(self, positions, box):
        self._tick += 1
        labels = cluster_labels(positions, box, self.cutoff)
        pending = self._assign_slots(labels, positions, box)
        first = self._tick == 1
        if first:
            self._slots = pending
            self._hold[:] = 0
            return
        # Hysteresis: a bead has to want the same new slot COMMIT_HOLD labellings
        # running before it gets it. `_hold` counts the run, and any change of
        # mind -- including changing back -- restarts it.
        differs = pending != self._slots
        self._hold = np.where(differs, self._hold + 1, 0)
        commit = differs & (self._hold >= COMMIT_HOLD)
        self._slots = np.where(commit, pending, self._slots).astype(np.int32)
        self._hold = np.where(commit, 0, self._hold)

    def _assign_slots(self, labels, positions, box):
        """One slot per bead from one labelling, matched against what is on
        screen. Returns the PENDING assignment -- what the hysteresis then has to
        agree to."""
        n = len(labels)
        pending = np.full(n, -1, dtype=np.int32)
        sizes = np.bincount(labels, minlength=n)
        big = np.flatnonzero(sizes >= self._min_size(n, box))
        if not len(big):
            return pending
        # Largest first: the biggest thing on screen gets first claim on the
        # colour it already has, which is what makes a merge recolour the small
        # aggregate rather than the large one. It is also what the cost guard
        # cuts against, so what it drops is always the least visible.
        big = big[np.argsort(-sizes[big])][:MAX_TRACKED_CLUSTERS]

        # Who holds which slot now, per component, in one pass: a 2D histogram of
        # (component, committed slot) over the beads that have a slot at all.
        held = self._slots
        counted = held >= 0
        overlap = np.zeros((len(big), self.n_slots), dtype=np.int64)
        rank = np.full(n, -1, dtype=np.int64)     # component label -> row in `big`
        rank[big] = np.arange(len(big))
        if np.any(counted):
            row = rank[labels[counted]]
            keep = row >= 0
            if np.any(keep):
                np.add.at(overlap, (row[keep], held[counted][keep]), 1)

        centre, radius = cluster_geometry(positions, labels, big, box)
        starts, other, gap = neighbour_graph(centre, radius, box, NEIGHBOUR_COUNT)

        assigned = np.full(len(big), -1, dtype=np.int64)
        load = np.zeros(self.n_slots, dtype=np.int64)
        for row in range(len(big)):
            lo, hi = starts[row], starts[row + 1]
            slot = self._claim(overlap[row], sizes[big[row]],
                               assigned[other[lo:hi]], gap[lo:hi], load)
            assigned[row] = slot
            load[slot] += 1
            self._last_used[slot] = self._tick
        # One lookup, not one scan per cluster: `labels == label` inside the loop
        # is O(N) per cluster, which is nothing at ten of them and 60 million
        # element-comparisons at a thousand.
        slot_of_label = np.full(len(sizes), -1, dtype=np.int32)
        slot_of_label[big] = assigned
        return slot_of_label[labels]

    def _claim(self, overlap, size, neighbour_slots, neighbour_gap, load):
        """The colour this cluster should wear.

        Three preferences in order, and the first is the one that matters: KEEP
        WHAT YOU ARE ALREADY WEARING. The rest is about a cluster that has no
        colour yet, or whose colour has just been taken by something bigger.
        """
        # How close the nearest neighbour wearing each colour is -- infinity for a
        # colour none of them wears. Written as a distance rather than a
        # forbidden/allowed flag so that running out of colours DEGRADES instead
        # of failing: with nothing free, the widest of the near separations wins
        # and the repeat is put as far away as this cluster's neighbourhood
        # allows. A boolean would have to either give up the guarantee or invent a
        # colour.
        nearest = np.full(self.n_slots, np.inf)
        for slot, gap in zip(neighbour_slots, neighbour_gap):
            if slot >= 0 and gap < nearest[slot]:
                nearest[slot] = gap
        allowed = nearest >= nearest.max()

        # 1. The colour most of it already wears, if that is really most of it.
        best = int(np.argmax(np.where(allowed, overlap, -1)))
        if allowed[best] and overlap[best] >= max(1, MATCH_MIN_OVERLAP * size):
            return best
        # 2. Otherwise the colour in least use so far, so the palette spreads
        #    evenly over the box instead of one hue taking a third of it, and
        #    among equals the one longest unused -- which keeps a colour just
        #    vacated by a merge from being handed straight to something else.
        options = np.flatnonzero(allowed)
        options = options[load[options] == load[options].min()]
        return int(options[np.argmin(self._last_used[options])])
