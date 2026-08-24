/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

// Contributing author: Pietro Sillano (TU Delft), 2025

#ifdef PAIR_CLASS
// clang-format off
PairStyle(mesomem,PairMesoMem);
// clang-format on

#else

#ifndef LMP_PAIR_mesomem_H
#define LMP_PAIR_mesomem_H

#include "pair.h"

namespace LAMMPS_NS {

class PairMesoMem : public Pair {
 public:
  PairMesoMem(LAMMPS *lmp);
  ~PairMesoMem() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  double init_one(int, int) override;
  void write_restart(FILE *) override;
  void read_restart(FILE *) override;
  void write_restart_settings(FILE *) override;
  void read_restart_settings(FILE *) override;
  void write_data(FILE *) override;
  void write_data_all(FILE *) override;
  void init_style() override;

 protected:
  double **cut;
// double **cutsq;
  double **sigma, **eps;
  double **ktilt, **ksplay;
  double **weight_rcut;
  double **zeta;
  double cut_global;
  double **c0; // for spont curvature
  double **splay_symmetry; // 0..1: blend signed<->|ninj| in the splay term (see coeff/compute)

  // DERIVED PER-TYPE-PAIR CONSTANTS. Every one of these was being re-derived
  // inside the pair loop, per pair, from numbers that depend only on the two
  // TYPES -- of which a membrane deck has one or two. Hoisting them into
  // init_one() is what turns a divide, a sqrt and a libm pow() per pair into a
  // table lookup; see compute() for the measured cost of each.
  double **inv_span;       // 1 / (cut - sigma): the cosine branch's r -> g scale
  double **inv_wr;         // 1 / weight_rcut
  double **rga_sq;         // (weight_rcut / 2)^2, the weight function's width^2
  int **zt_exp;            // 2*zeta - 1 when that is a whole number, else -1


virtual void allocate();
};

} // namespace LAMMPS_NS

#endif
#endif
