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

// Contributing author: Pietro Sillano (TU Delft), 2026

// 07-02-2026 branched from working pair_membrane_sillano_gemini3_radial.cpp (19/12/2025 last edit). it is the potential I used for the paper anaysis.

// in this code we will change:
// - splay to the new form
// - we will introduce curvature as distance dependent term

// - new name for the pair: membrane_sillano_v2
// - new parameter: c0
// - update tilt with new form
// - update Usplay, splay radial force and splay torques with new form



#define INCLUDE_RADIAL

#include "pair_membrane_sillano_v2.h"
#include <cmath>
#include <cstring>
#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
// (math_extra.h not needed: manual cross products below)
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"



using namespace LAMMPS_NS;



/* ---------------------------------------------------------------------- */

PairMesoMem::PairMesoMem(LAMMPS *lmp) : Pair(lmp), eps(nullptr), sigma(nullptr), cut(nullptr)

{
  writedata = 1;
  single_enable = 0;
}

/* ---------------------------------------------------------------------- */

PairMesoMem::~PairMesoMem()
{
  if (copymode) return;
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(cut);
    memory->destroy(sigma);
    memory->destroy(eps);
    memory->destroy(ktilt);
    memory->destroy(ksplay);
    memory->destroy(weight_rcut);
    memory->destroy(zeta);
    memory->destroy(c0);
    memory->destroy(splay_symmetry);

    memory->destroy(inv_span);
    memory->destroy(inv_wr);
    memory->destroy(rga_sq);
    memory->destroy(zt_exp);

  }
}

/* ---------------------------------------------------------------------- */

void PairMesoMem::allocate()
{
  allocated = 1;
  int np1 = atom->ntypes + 1;

  memory->create(setflag, np1, np1, "pair:setflag");
  for (int i = 1; i < np1; i++)
    for (int j = i; j < np1; j++) setflag[i][j] = 0;

  memory->create(cutsq, np1, np1, "pair:cutsq");
  memory->create(cut, np1, np1, "pair:cut");
  memory->create(sigma, np1, np1, "pair:sigma");
  memory->create(eps, np1, np1, "pair:eps");
  memory->create(ktilt, np1, np1, "pair:ktilt");
  memory->create(ksplay, np1, np1, "pair:ksplay");
  memory->create(weight_rcut, np1, np1, "pair:weight_rcut");
  memory->create(zeta, np1, np1, "pair:zeta");
  memory->create(c0, np1, np1, "pair:c0"); // Allocate c0 array
  memory->create(splay_symmetry, np1, np1, "pair:splay_symmetry");

  memory->create(inv_span, np1, np1, "pair:inv_span");
  memory->create(inv_wr, np1, np1, "pair:inv_wr");
  memory->create(rga_sq, np1, np1, "pair:rga_sq");
  memory->create(zt_exp, np1, np1, "pair:zt_exp");
  // ZEROED, unlike the coefficient arrays above. Those are read only for a type
  // pair init_one() has been through, and under `pair_style hybrid` this style's
  // sub-list carries only the type pairs assigned to it -- so an unassigned entry
  // is unreachable either way. But memory->create leaves whatever was in the
  // allocation, and the difference between an unreachable zero and an unreachable
  // inf is the difference between a bug that stays a bug and one that becomes a
  // NaN in somebody's forces.
  for (int i = 0; i < np1; i++)
    for (int j = 0; j < np1; j++) {
      inv_span[i][j] = inv_wr[i][j] = rga_sq[i][j] = 0.0;
      zt_exp[i][j] = -1;
    }

}

/* ---------------------------------------------------------------------- */

void PairMesoMem::settings(int narg, char **arg)
{
  if (narg != 1) error->all(FLERR, "Illegal pair_style command");
  cut_global = utils::numeric(FLERR, arg[0], false, lmp);

  #ifdef INCLUDE_RADIAL
      error->warning(FLERR, "INCLUDE_RADIAL is ON");
  #else
      error->warning(FLERR, "INCLUDE_RADIAL is OFF");
  #endif

  // Reset cutoffs that have been explicitly set
  if (allocated) {
    int i, j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i; j <= atom->ntypes; j++)
        if (setflag[i][j]) cut[i][j] = cut_global;
  }
}

/* ----------------------------------------------------------------------
   Set coefficients for one or more type pairs
   Args: sigma, eps, ktilt, ksplay, cut, weight_rcut, zeta, c0, [splay_symmetry]

   The trailing splay_symmetry argument is optional (0..1, default 0.0) so that
   existing 10-argument callers keep the original signed-splay behaviour; see
   compute() for what the parameter does.
------------------------------------------------------------------------- */

void PairMesoMem::coeff(int narg, char **arg)
{
  if (narg != 10 && narg != 11) error->all(FLERR, "Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo, ihi, jlo, jhi;
  utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
  utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);

  double sigma_one = utils::numeric(FLERR, arg[2], false, lmp);
  double eps_one = utils::numeric(FLERR, arg[3], false, lmp);
  double ktilt_one = utils::numeric(FLERR, arg[4], false, lmp);
  double ksplay_one = utils::numeric(FLERR, arg[5], false, lmp);
  double cut_one = utils::numeric(FLERR, arg[6], false, lmp);
  double weight_rcut_one = utils::numeric(FLERR, arg[7], false, lmp);
  double zeta_one = utils::numeric(FLERR, arg[8], false, lmp);
  double c0_one = utils::numeric(FLERR, arg[9], false, lmp); // spontaneous curvature
  // Optional 11th arg: splay symmetry weight (0 -> signed dot product, the
  // original form; 1 -> |dot product|). Defaults to 0 for 10-arg callers.
  double splay_sym_one = (narg == 11) ? utils::numeric(FLERR, arg[10], false, lmp) : 0.0;


  if (weight_rcut_one > cut_one || weight_rcut_one > cut_global) {
    error->all(FLERR, "Orientation cutoff w_c > isotropic distance cutoff r_c");
  }

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo, i); j <= jhi; j++) {
      sigma[i][j] = sigma_one;
      eps[i][j] = eps_one;
      ktilt[i][j] = ktilt_one;
      ksplay[i][j] = ksplay_one;
      cut[i][j] = cut_one;
      weight_rcut[i][j] = weight_rcut_one;
      zeta[i][j] = zeta_one;
      c0[i][j] = c0_one; // Store c0
      splay_symmetry[i][j] = splay_sym_one;


      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR, "Incorrect args for pair coefficients");
}

/* ---------------------------------------------------------------------- */

void PairMesoMem::init_style()
{
  // Requirement: atoms must have orientation (mu) and torque
  if (!atom->q_flag || !atom->mu_flag || !atom->torque_flag)
    error->all(FLERR, "Pair membrane_sillano requires atom attributes q, mu, torque");

  neighbor->request(this, instance_me);
}

/* ---------------------------------------------------------------------- */

double PairMesoMem::init_one(int i, int j)
{
  // Strict Manual Mixing:
  // If the user did not set coefficients for this pair, we error out.
  // Automatic mixing for ktilt/ksplay/zeta is not physically defined here.

  if (setflag[i][j] == 0) {
    error->all(FLERR, "All pair coeffs must be set manually for pair_style membrane_sillano");
  }

  eps[j][i] = eps[i][j];
  sigma[j][i] = sigma[i][j];
  ktilt[j][i] = ktilt[i][j];
  ksplay[j][i] = ksplay[i][j];
  weight_rcut[j][i] = weight_rcut[i][j];
  zeta[j][i] = zeta[i][j];
  cut[j][i] = cut[i][j];
  c0[j][i] = c0[i][j]; // Copy c0
  splay_symmetry[j][i] = splay_symmetry[i][j];

  // --- the derived constants the pair loop reads (see the header) -----------
  // The cosine branch's r -> g map is g = (pi/2)(r - sigma)/(cut - sigma), so
  // what it needs per pair is the reciprocal of that span; a zero span (cut ==
  // sigma) leaves no attractive branch at all, and 0 keeps g there rather than
  // producing an inf.
  double span = cut[i][j] - sigma[i][j];
  inv_span[i][j] = inv_span[j][i] = (span > 0.0) ? 1.0 / span : 0.0;
  double wr = weight_rcut[i][j];
  inv_wr[i][j] = inv_wr[j][i] = (wr > 0.0) ? 1.0 / wr : 0.0;
  rga_sq[i][j] = rga_sq[j][i] = 0.25 * wr * wr;
  // THE EXPONENT. The attractive branch is -eps cos(g)^(2 zeta), and its
  // derivative carries cos(g)^(2 zeta - 1) -- a libm pow() per pair, which is
  // the single most expensive thing in the loop. But 2 zeta - 1 is a whole
  // number for every HALF-INTEGER zeta, which is every value anyone runs (the
  // paper's is 5, i.e. an exponent of 9), and a whole-number power is a handful
  // of multiplies by squaring. So the integer is worked out once, here, and -1
  // means "this zeta is not a half-integer, use pow()" -- which is only ever
  // the case for a moment while a slider is being dragged through one.
  double e = 2.0 * zeta[i][j] - 1.0;
  double rounded = nearbyint(e);
  int whole = (rounded >= 0.0) && (rounded <= 64.0) && (fabs(e - rounded) < 1.0e-12);
  zt_exp[i][j] = zt_exp[j][i] = whole ? (int) rounded : -1;

  return cut[i][j];
}

/* ---------------------------------------------------------------------- */

void PairMesoMem::compute(int eflag, int vflag)
{
  int i, j, ii, jj, inum, jnum, itype, jtype;
  double xtmp, ytmp, ztmp, delx, dely, delz, evdwl, rsq, r, inv_r;
  int *ilist, *jlist, *numneigh, **firstneigh;

  evdwl = 0.0;
  double Utot = 0.0;
  ev_init(eflag, vflag);

  double **x = atom->x;
  double **f = atom->f;
  double **mu = atom->mu;
  double **torque = atom->torque;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // Local vector registers
  double ni[3], nj[3], rhat[3];
  double fx, fy, fz, tx, ty, tz, tx_j, ty_j, tz_j;

  // Math vars
  double inv_mag_i, inv_mag_j;
  double factor_lj;

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    // OPTIMIZATION: Cache inverse magnitude of I once per neighbor list
    inv_mag_i = 1.0 / mu[i][3];
    ni[0] = mu[i][0] * inv_mag_i;
    ni[1] = mu[i][1] * inv_mag_i;
    ni[2] = mu[i][2] * inv_mag_i;

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;
      jtype = type[j];

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx * delx + dely * dely + delz * delz;

      if (rsq < cutsq[itype][jtype]) {
        r = sqrt(rsq);
        inv_r = 1.0 / r;

        rhat[0] = delx * inv_r;
        rhat[1] = dely * inv_r;
        rhat[2] = delz * inv_r;

        // --- 1. Isotropic (LJ/Cos) Force ---
        // Calculated for all pairs within r_cut
        double eps_val = eps[itype][jtype];
        double sigma_val = sigma[itype][jtype]; // we are not using sigma*1.12 because we want to be transparent in the code!
        double rmin = sigma_val;
        double eps_lj = 0.0;
        double Ulj = 0.0;

        if (r < rmin) {
          double t = sigma_val * inv_r;
          double t2 = t * t;
          double t4 = t2 * t2;
          Ulj = eps_val * (t4 - 2.0 * t2);
          eps_lj = 4.0 * eps_val * inv_r * (t4 - t2);
        } else {
          // THE ATTRACTIVE BRANCH, AND THE HOT PATH: rmin is sigma and the
          // cutoff is 2.5 sigma, so on a membrane at its equilibrium spacing
          // this is where nearly every pair lands. Measured on the 3600-bead rod
          // playground, the isotropic term alone was 35 of the 44 ms a 16-step
          // chunk took -- more than the whole tilt/splay block -- and it was
          // spending it on four things it did not have to: a sqrt of cutsq, a
          // divide, a sin(), and a libm pow(). All four are gone below; the
          // numbers are unchanged to within a few ULP, which the energy
          // cross-check against the Python expression (--verify) pins.
          double zt = zeta[itype][jtype];
          // The span's reciprocal is a per-TYPE-PAIR constant (see init_one),
          // and `cut` is what `sqrt(cutsq)` was recovering.
          double dg_dr = M_PI * 0.5 * inv_span[itype][jtype];
          double g = dg_dr * (r - rmin);

          double cos_t = cos(g);
          // sin(g) WITHOUT A SECOND TRANSCENDENTAL. g runs over [0, pi/2] as r
          // runs over [rmin, cut], so sin(g) >= 0 there and is exactly
          // sqrt(1 - cos^2). The fmax guards the last bit of rounding at
          // g = pi/2, where cos^2 can come out a hair above 1.
          double sin_t = sqrt(fmax(0.0, 1.0 - cos_t * cos_t));

          // cos^(2 zeta - 1), by squaring when the exponent is a whole number
          // -- which it is for every half-integer zeta, i.e. every value anyone
          // runs. See init_one for how the exponent is classified.
          double cos_pow;
          int e = zt_exp[itype][jtype];
          if (e >= 0) {
            cos_pow = 1.0;
            double base = cos_t;
            for (int k = e; k; k >>= 1) {
              if (k & 1) cos_pow *= base;
              base *= base;
            }
          } else {
            cos_pow = pow(cos_t, 2.0 * zt - 1.0);
          }

          double cos_2zt = cos_pow * cos_t;

          Ulj = -eps_val * cos_2zt;

          // dU/dg * dg/dr
          double dU_dg = eps_val * (2.0 * zt) * cos_pow * sin_t;
          eps_lj = -dU_dg * dg_dr;
        }

        // Initialize total forces/torques with just LJ part
        fx = eps_lj * rhat[0];
        fy = eps_lj * rhat[1];
        fz = eps_lj * rhat[2];
        tx = ty = tz = 0.0;
        tx_j = ty_j = tz_j = 0.0;

        // --- 2. Anisotropic (Tilt/Splay) Force ---
        double wr = weight_rcut[itype][jtype];

        if (r < wr) {
            // --- A. Weight Calculation ---
            // Both the reciprocal of wr and (wr/2)^2 are per-type-pair
            // constants; see init_one.
            double r_wr = r * inv_wr[itype][jtype];

            // D = (r/wc)^4
            double r_wr_2 = r_wr * r_wr;
            double r_wr_4 = r_wr_2 * r_wr_2;
            double denom_w = r_wr_4 - 1.0; // This is always negative for r < wr

            double w = 0.0;

            // REPLACEMENT LOGIC:
            // Only calculate if we are safely away from the singularity (denom_w < -1e-14).
            // If denom_w is closer to 0 than that, the exp() result is mathematically 0.0 anyway.
            double rga_sq_ij = rga_sq[itype][jtype];
            if (denom_w < -1e-14) {
                double val_exp = (r * r) / (rga_sq_ij * denom_w);
                w = exp(val_exp);
            }
            // Else: w remains 0.0, avoiding division by tiny denom_w


            // --- B. Vector Normalization ---
            inv_mag_j = 1.0 / mu[j][3];
            nj[0] = mu[j][0] * inv_mag_j;
            nj[1] = mu[j][1] * inv_mag_j;
            nj[2] = mu[j][2] * inv_mag_j;

            // --- C. Dot Products ---
            double nirhat = ni[0]*rhat[0] + ni[1]*rhat[1] + ni[2]*rhat[2];
            double njrhat = nj[0]*rhat[0] + nj[1]*rhat[1] + nj[2]*rhat[2];
            double ninj   = ni[0]*nj[0]   + ni[1]*nj[1]   + ni[2]*nj[2];

            // --- D. Tilt Calculation (Extended with C0) ---
            double rh_x_ni[3], rh_x_nj[3];
            // Cross product rhat x ni
            rh_x_ni[0] = rhat[1]*ni[2] - rhat[2]*ni[1];
            rh_x_ni[1] = rhat[2]*ni[0] - rhat[0]*ni[2];
            rh_x_ni[2] = rhat[0]*ni[1] - rhat[1]*ni[0];
            // Cross product rhat x nj
            rh_x_nj[0] = rhat[1]*nj[2] - rhat[2]*nj[1];
            rh_x_nj[1] = rhat[2]*nj[0] - rhat[0]*nj[2];
            rh_x_nj[2] = rhat[0]*nj[1] - rhat[1]*nj[0];

            // sin_a2 = 0.5 * r * c0
            double sin_a2 = 0.5 * r * c0[itype][jtype];

            double kt = ktilt[itype][jtype];

            // Diff terms: ni.r + sin(alpha/2)
            double diff_i = nirhat + sin_a2;
            double diff_j = njrhat - sin_a2; // check if sign needs to be the same

            double Utilt = 0.5 * kt * (diff_i * diff_i + diff_j * diff_j);

            // Vector u = n - (n.r)r
            double ui[3], uj[3];
            ui[0] = ni[0] - nirhat*rhat[0];
            ui[1] = ni[1] - nirhat*rhat[1];
            ui[2] = ni[2] - nirhat*rhat[2];

            uj[0] = nj[0] - njrhat*rhat[0];
            uj[1] = nj[1] - njrhat*rhat[1];
            uj[2] = nj[2] - njrhat*rhat[2];

            // Tilt Force term (angular part only)
            double ft_pref = -kt * inv_r;

            double tilt_fx = ft_pref * (diff_i * ui[0] + diff_j * uj[0]);
            double tilt_fy = ft_pref * (diff_i * ui[1] + diff_j * uj[1]);
            double tilt_fz = ft_pref * (diff_i * ui[2] + diff_j * uj[2]);

            // Accumulate Tilt Force scaled by weight
            fx += tilt_fx * w;
            fy += tilt_fy * w;
            fz += tilt_fz * w;


            // --- E. Splay Calculation ---
            //
            // Splay penalises neighbouring directors that are not aligned. In the
            // original form the penalty is built on the SIGNED dot product
            //   ninj = ni . nj,
            // driving (ninj - 1) -> 0, i.e. it rewards ni == nj (parallel) and
            // maximally penalises ni == -nj (antiparallel). A membrane where the
            // two leaflets' normals point opposite ways is therefore punished.
            //
            // splay_symmetry s in [0,1] smoothly makes the term blind to that
            // sign by blending toward the ABSOLUTE (normed) dot product:
            //   ninj_eff = (1 - s) * ninj + s * |ninj|.
            // At s = 0 this is exactly the original ninj (nothing changes). At
            // s = 1 the penalty is built on |ninj|, so parallel AND antiparallel
            // directors both sit at the energy minimum -- the system no longer
            // cares which way a director points, only that it lies along the same
            // axis as its neighbours. Intermediate s interpolates continuously.
            //
            // The blend is applied to the recurring "deviation" expression
            //   dev = ninj_eff - 1 + 2*sin(alpha/2)^2   (the last term is the c0
            // spontaneous-curvature offset, sign-independent). Because
            //   d(ninj_eff)/d(ninj) = (1 - s) + s * sign(ninj) =: dsym,
            // every director gradient (the splay torque) picks up the factor dsym,
            // while the purely radial c0 contribution -- which comes from the
            // sin(alpha/2)^2 term, not from ninj -- is unchanged.
            double ks = ksplay[itype][jtype];
            double ss = splay_symmetry[itype][jtype];

            double abs_ninj = fabs(ninj);
            double ninj_eff = (1.0 - ss) * ninj + ss * abs_ninj;
            double sgn_ninj = (ninj >= 0.0) ? 1.0 : -1.0;
            double dsym = (1.0 - ss) + ss * sgn_ninj;   // d(ninj_eff)/d(ninj)

            double ni_x_nj[3];
            ni_x_nj[0] = ni[1]*nj[2] - ni[2]*nj[1];
            ni_x_nj[1] = ni[2]*nj[0] - ni[0]*nj[2];
            ni_x_nj[2] = ni[0]*nj[1] - ni[1]*nj[0];

            double splay_dev = ninj_eff - 1.0 + 2.0 * (sin_a2 * sin_a2);
            double Usplay = 0.5 * ks * splay_dev * splay_dev;

            // --- F. Radial Correction (Energy Conservation) ---
            // This force is purely radial (along rhat) and repulsive
            // Derived from - (Utilt + Usplay) * (dw/dr)

            double U_ang_sum = Utilt + Usplay;



            // Apply radial correction
            #ifdef INCLUDE_RADIAL
              if (w > 0){
              // Factor = w * [ 2 * (D+1) ] / [ rga^2 * (D-1)^2 ]
              
              // 8 Feb 2026 removed the minus sign in rad_numerator
              double rad_numerator = 2.0 * w * (r_wr_4 + 1.0) * r;
              double rad_denominator = rga_sq_ij * denom_w * denom_w;

              double f_rad_mag = U_ang_sum * (rad_numerator / rad_denominator);

              fx += f_rad_mag * rhat[0];
              fy += f_rad_mag * rhat[1];
              fz += f_rad_mag * rhat[2];

              // --- Radial part coming from tilt term ---
              double f_rad_tilt = - 0.5 * kt * (diff_i * c0[itype][jtype] + diff_j * c0[itype][jtype]);
              fx += w * f_rad_tilt * rhat[0];
              fy += w * f_rad_tilt * rhat[1];
              fz += w * f_rad_tilt * rhat[2];

            // --- Radial part coming from splay term ---
            // r-derivative of Usplay through sin(alpha/2) = 0.5*r*c0 only, so it
            // uses splay_dev but NOT dsym (the symmetry blend touches ninj, not r).
              double f_rad_splay = - ks * splay_dev * (c0[itype][jtype] * c0[itype][jtype] * r);
              fx += w * f_rad_splay * rhat[0];
              fy += w * f_rad_splay * rhat[1];
              fz += w * f_rad_splay * rhat[2];
              }
            #endif


            // --- G. Torques ---
            // Torque on I. dsym carries the splay-symmetry blend into the
            // director gradient: d(Usplay)/d(ninj) = ks * splay_dev * dsym.
            double splay_pref = ks * splay_dev * dsym;

            tx += w * (kt * diff_i * rh_x_ni[0] + splay_pref * ni_x_nj[0]);
            ty += w * (kt * diff_i * rh_x_ni[1] + splay_pref * ni_x_nj[1]);
            tz += w * (kt * diff_i * rh_x_ni[2] + splay_pref * ni_x_nj[2]);

            // Torque on J
            tx_j += w * (kt * diff_j * rh_x_nj[0] - splay_pref * ni_x_nj[0]);
            ty_j += w * (kt * diff_j * rh_x_nj[1] - splay_pref * ni_x_nj[1]);
            tz_j += w * (kt * diff_j * rh_x_nj[2] - splay_pref * ni_x_nj[2]);



            // Accumulate Energy
            if (eflag) evdwl = Ulj + w * U_ang_sum;
        }
        else {
            if (eflag) evdwl = Ulj;
        }

        // --- FINAL APPLY ---
        if (eflag) evdwl *= factor_lj;

        // Apply Factor LJ to forces/torques
        fx *= factor_lj; fy *= factor_lj; fz *= factor_lj;
        tx *= factor_lj; ty *= factor_lj; tz *= factor_lj;
        tx_j *= factor_lj; ty_j *= factor_lj; tz_j *= factor_lj;

        f[i][0] += fx;
        f[i][1] += fy;
        f[i][2] += fz;
        torque[i][0] += tx;
        torque[i][1] += ty;
        torque[i][2] += tz;

        if (newton_pair || j < nlocal) {
          f[j][0] -= fx;
          f[j][1] -= fy;
          f[j][2] -= fz;
          torque[j][0] += tx_j;
          torque[j][1] += ty_j;
          torque[j][2] += tz_j;
        }

        // Use ev_tally_xyz for non-central forces (virial correction)
        if (evflag) ev_tally_xyz(i, j, nlocal, newton_pair, evdwl, 0.0,
                                 fx, fy, fz, delx, dely, delz);
      }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}



// double PairMesoMem::single(int /*i*/, int /*j*/, int itype, int jtype, double rsq, double /*factor_coul*/, double factor_lj, double &fforce)
// {
//   double r, dr, aexp, bexp;

//   r = sqrt(rsq);
//   dr = r - r0[itype][jtype];
//   aexp = biga0[itype][jtype] * exp(-alpha[itype][jtype] * r);
//   bexp = biga1[itype][jtype] * exp(-beta[itype][jtype] * dr * dr);

//   fforce = ;
//   return U;
// }


/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairMesoMem::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  for (int i = 1; i <= atom->ntypes; i++) {
    for (int j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j], sizeof(int), 1, fp);
      if (setflag[i][j]) {
        fwrite(&sigma[i][j], sizeof(double), 1, fp);
        fwrite(&eps[i][j], sizeof(double), 1, fp);
        fwrite(&ktilt[i][j], sizeof(double), 1, fp);
        fwrite(&ksplay[i][j], sizeof(double), 1, fp);
        fwrite(&cut[i][j], sizeof(double), 1, fp);
        fwrite(&weight_rcut[i][j], sizeof(double), 1, fp);
        fwrite(&zeta[i][j], sizeof(double), 1, fp);
        fwrite(&c0[i][j], sizeof(double), 1, fp); // Write c0
        fwrite(&splay_symmetry[i][j], sizeof(double), 1, fp);

      }
    }
  }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairMesoMem::read_restart(FILE *fp)
{
  read_restart_settings(fp);
  allocate();

  for (int i = 1; i <= atom->ntypes; i++) {
    for (int j = i; j <= atom->ntypes; j++) {
      if (comm->me == 0) utils::sfread(FLERR, &setflag[i][j], sizeof(int), 1, fp, nullptr, error);
      MPI_Bcast(&setflag[i][j], 1, MPI_INT, 0, world);
      if (setflag[i][j]) {
        if (comm->me == 0) {

          utils::sfread(FLERR, &sigma[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &eps[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &ktilt[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &ksplay[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &cut[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &weight_rcut[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &zeta[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &c0[i][j], sizeof(double), 1, fp, nullptr, error); // Read c0
          utils::sfread(FLERR, &splay_symmetry[i][j], sizeof(double), 1, fp, nullptr, error);

        }

        MPI_Bcast(&sigma[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&eps[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&ktilt[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&ksplay[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&cut[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&weight_rcut[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&zeta[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&splay_symmetry[i][j], 1, MPI_DOUBLE, 0, world);

      }
    }
  }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairMesoMem::write_restart_settings(FILE *fp)
{
  fwrite(&cut_global, sizeof(double), 1, fp);
  fwrite(&offset_flag, sizeof(int), 1, fp);
  fwrite(&mix_flag, sizeof(int), 1, fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairMesoMem::read_restart_settings(FILE *fp)
{
  if (comm->me == 0) {

    utils::sfread(FLERR, &cut_global, sizeof(double), 1, fp, nullptr, error);
    utils::sfread(FLERR, &offset_flag, sizeof(int), 1, fp, nullptr, error);
    utils::sfread(FLERR, &mix_flag, sizeof(int), 1, fp, nullptr, error);
  }

  MPI_Bcast(&cut_global, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&offset_flag, 1, MPI_INT, 0, world);
  MPI_Bcast(&mix_flag, 1, MPI_INT, 0, world);
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void PairMesoMem::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    fprintf(fp, "%d %g %g %g %g %g %g %g\n", i, sigma[i][i], eps[i][i], ktilt[i][i],ksplay[i][i], cut[i][i], weight_rcut[i][i], zeta[i][i]);
}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void PairMesoMem::write_data_all(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      fprintf(fp, "%d %d %g %g %g %g %g %g %g \n", i, j, sigma[i][i], eps[i][i], ktilt[i][j],ksplay[i][j], cut[i][i], weight_rcut[i][j], zeta[i][i]);
}
