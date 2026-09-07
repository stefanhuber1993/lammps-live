"""A declarative layer for exploring force fields interactively.

A playground file names a FORCE FIELD, a SCENARIO and a MODE:

    from lammps_live.playground import *

    PLAYGROUND = Playground(
        name="MesoMem membrane patch (3D)",
        force_field="mesomem",
        scenario=hex_patch(n_rings=1),
        mode="game",
        control=Control(atom="first", plane="xz", leash=(2.8, 2.8)),
        observables=["nematic_S", "mean_tilt_deg"],
        presets={"floppy": {"k_tilt": 2.0, "k_splay": 0.1}},
    )

The division of labour, and the answer to "what goes in the file and what goes in
the GUI": a force field declares its parameters once, and every LIVE one becomes a
slider automatically. Anything STRUCTURAL -- particle counts, box size, boundary
conditions -- stays in the file, because changing it means rebuilding the
simulation and a slider there would destroy the state you were watching. See
params.py for the tiers.

  forcefield.py   the potential: its parameters, its LAMMPS commands, and one
                  vectorized expression of its energy (used for the live
                  breakdown panels and for verify.py's cross-check against
                  LAMMPS)
  scenario.py     geometry, cell, relaxation, per-frame housekeeping -- pure
                  numpy, so it is testable with no LAMMPS instance
  modes.py        game (a controlled particle, leash, haptics) vs sim (Play /
                  Pause / Reset). Chosen at run time, so both work on any
                  playground
  observables.py  named quantities plotted live and logged by a sweep, on a
                  frame budget
  system.py       composes the four into the MDSystem the app already speaks
"""
from .forcefield import ForceField, register
from .observables import observable
from .params import Param, ParamSet, Tier, structural
from .scenario import (
    BeadAndPartner, Composite, HexPatch, HexSheet, RandomFill, RodOnSheet,
    Scenario, ScenarioBuild, VesiclePolymer, bead_and_partner, compose,
    hex_patch, hex_sheet, random_fill, rod_on_sheet, vesicle_polymer,
)
from ..mdsystem import ForceFeedbackProfile
from .deposition import Deposition2D, IonicSlab2D, deposition_2d, ionic_slab_2d
from .spec import Control, Lesson, Playground, Thesis
from .state import Box, FrameState, PairData, build_pairs
from .thermostat import CSVR, Langevin, Thermostat

__all__ = [
    "Playground", "Control", "Lesson", "Thesis",
    "ForceFeedbackProfile",
    "Deposition2D", "deposition_2d", "IonicSlab2D", "ionic_slab_2d",
    "Thermostat", "Langevin", "CSVR",
    "Scenario", "ScenarioBuild", "HexPatch", "HexSheet", "RandomFill",
    "RodOnSheet", "VesiclePolymer", "BeadAndPartner", "Composite",
    "hex_patch", "hex_sheet", "random_fill", "rod_on_sheet", "vesicle_polymer",
    "bead_and_partner", "compose",
    "ForceField", "register",
    "Param", "ParamSet", "Tier", "structural",
    "Box", "FrameState", "PairData", "build_pairs",
    "observable",
]
