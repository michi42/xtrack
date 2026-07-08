# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

"""
Tune FOOTPRINTS of different bunch families in the LHC collision scenario
(6.8 TeV squeezed flat optics, head-on + BBLR), with the per-bunch
self-consistent multi-bunch beam-beam closed solution.

Workflow (multi-threaded OpenMP kernels by default, ``LHC_OMP=0`` for
serial):

1. Solve the multi-bunch problem on the sector-map machine (as in
   ``002``/``004``): 2 iterations with the orbit-only ``fast_orbit`` twiss,
   then 2 more with ``dynamic_beta=True`` (per-bunch effective sizes
   recomputed from the live betas each iteration; the element state carries
   over between the two calls, so this continues the same iteration).
2. Transfer the converged solution back to the FULL THICK lattice of beam 1:
   the same multi-bunch beam-beam lenses are installed at the (still
   present) encounter markers and loaded with the final per-bunch orbits
   and dynamic sizes of beam 2, plus the (bunch-averaged) dynamic own sizes
   of beam 1.
3. Generate tune footprints on this thick lattice for 12 equidistant
   bunches along the longest train (first and last included), covering the
   PACMAN transition from the train head through the fully-surrounded
   center to the tail. Each footprint tracks particles with ``zeta`` frozen at the
   bunch's slot label, so the multi-bunch lenses apply that bunch's actual
   encounters (head-on + long-range, around its own closed orbit). Before
   tracking, the lenses are switched to ``coherent=False``: the footprint
   particles are INDIVIDUAL protons that see the field of the opposing
   bunches with their own sizes (incoherent, weak-strong), while the closed
   solution was obtained with the coherent rigid-bunch (convolved-size)
   kicks. The footprints use the linear rescale on the ``beambeam_scale``
   knob (bound to the ``scale_strength`` of all lenses at installation, as
   in the xfields beam-beam config tools): computed at weak beam-beam and
   extrapolated linearly to full strength, which avoids resonance-distorted
   footprints.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt

import xobjects as xo
import xtrack as xt
import lhc_mb_common as mb

# multi-threaded CPU kernels by default in this demo
omp = os.environ.get('LHC_OMP', 'auto')
sim = mb.LHCMultibunchBB.collision(context=(
    xo.ContextCpu() if omp in ('0', '', 'serial')
    else xo.ContextCpu(omp_num_threads='auto' if omp == 'auto' else int(omp))))

# ----------------------------------------------------------------------------
# Build the machine (as in 002/004)
# ----------------------------------------------------------------------------
env, line_b1, line_b2 = sim.load()
slot_len = line_b1.get_length() / sim.N_SLOTS
b_h_dist = slot_len / 2.0
gamma0 = line_b1.particle_ref.gamma0[0]
beta0 = line_b1.particle_ref.beta0[0]

sim.install_markers(line_b1, mirror=False, b_h_dist=b_h_dist)
sim.install_markers(line_b2, mirror=True, b_h_dist=b_h_dist)
geom, meta = sim.compute_geometry(line_b1, line_b2, b_h_dist, slot_len)

print('  building second-order maps between the beam-beam markers...')
red_b1 = line_b1.get_line_with_second_order_maps(split_at=sim.marker_names_b1)
red_b2 = line_b2.get_line_with_second_order_maps(split_at=sim.marker_names_b2)
for rl in (red_b1, red_b2):
    rl.twiss_default['method'] = '4d'
    rl.build_tracker(_context=sim.context)

scheme_b1, scheme_b2 = mb.load_scheme()
slots_b1, slots_b2 = mb.all_filled_slots(scheme_b1, scheme_b2)
print(f'  populated bunches: B1 = {len(slots_b1)}, B2 = {len(slots_b2)}')

bb_b1 = sim.install_bb(red_b1, False, geom, len(slots_b2), gamma0, beta0)
bb_b2 = sim.install_bb(red_b2, True, geom, len(slots_b1), gamma0, beta0)

# ----------------------------------------------------------------------------
# 1) Self-consistent solve: 2 iterations orbit-only, 2 with dynamic beta
# ----------------------------------------------------------------------------
print('Self-consistent solve (2 iterations fast_orbit):')
t0 = time.time()
sim.solve_self_consistent(red_b1, red_b2, bb_b1, bb_b2,
                          slots_b1, slots_b2, geom, n_iter=2)
print('Self-consistent solve (2 more iterations with dynamic beta):')
mbtw_b1, mbtw_b2 = sim.solve_self_consistent(
    red_b1, red_b2, bb_b1, bb_b2, slots_b1, slots_b2, geom, n_iter=2,
    dynamic_beta=True)
print(f'  total solve time: {time.time() - t0:.1f} s')

# ----------------------------------------------------------------------------
# 2) Transfer the converged solution to the full thick lattice of beam 1
# ----------------------------------------------------------------------------
print('Installing the converged lenses on the full thick lattice (B1)...')
bb_thick_b1 = sim.install_bb(line_b1, False, geom, len(slots_b2),
                             gamma0, beta0)
sizes_b2 = sim.effective_sigmas(mbtw_b2, sim.marker_names_b2, gamma0)
sizes_b1 = sim.effective_sigmas(mbtw_b1, sim.marker_names_b1, gamma0)
sim.update_opposing(bb_thick_b1, mbtw_b2, slots_b2, sim.marker_names_b2,
                     geom, sigmas_other=sizes_b2, sigmas_own=sizes_b1)

# incoherent footprints: individual particles see the field of the opposing
# bunches with their OWN sizes (weak-strong), not the convolved coherent
# kick used for the rigid-bunch closed solution
for bb in bb_thick_b1.values():
    bb.coherent = False

zeta_b1 = np.array(slots_b1) * sim.ZETA_PER_SLOT

# ----------------------------------------------------------------------------
# 3) Footprints for the bunch families of the longest train
# ----------------------------------------------------------------------------
# longest contiguous filled run of beam 1
filled = np.asarray(scheme_b1) > 0
best_len = best_start = cur_len = cur_start = 0
for s in range(sim.N_SLOTS):
    if filled[s]:
        cur_start = s if cur_len == 0 else cur_start
        cur_len += 1
        if cur_len > best_len:
            best_len, best_start = cur_len, cur_start
    else:
        cur_len = 0
train = list(range(best_start, best_start + best_len))
print(f'  longest train: {best_len} bunches starting at slot {best_start}')

# 12 equidistant bunches along the train (first and last included)
pos_in_train = np.round(np.linspace(0, len(train) - 1, 12)).astype(int)
family = [(train[k], f'train bunch {k + 1}') for k in pos_in_train]

# cross-check the transferred lattice on the family bunches
zeta_fam = np.array([sl for sl, _ in family]) * sim.ZETA_PER_SLOT
mb_thick = line_b1.twiss_multibunch(zeta_bunches=zeta_fam,
                                    mode='fast_orbit', show_progress=False)
idx_fam = np.searchsorted(slots_b1, [sl for sl, _ in family])
dq_check = mb.wrap_frac_tune(mb_thick.qx - np.asarray(mbtw_b1.qx)[idx_fam])
print(f'  transfer check (thick vs sector maps, family bunches): '
      f'max |dqx| = {np.max(np.abs(dq_check)):.2e}')

print('Footprints on the thick lattice:')
footprints = {}
for sl, label in family:
    t0 = time.time()
    # linear rescale on the beam-beam strength (as in the xmask footprint
    # example): the footprint is computed at weak beam-beam
    # (beambeam_scale = v0 and v0 + dv) and extrapolated linearly to the
    # actual strength, avoiding resonance-distorted footprints.
    fp = line_b1.get_footprint(
        nemitt_x=sim.nemitt, nemitt_y=sim.nemitt,
        r_range=(0.3, 6), theta_range=(0.15, np.pi / 2 - 0.15),
        freeze_longitudinal=True, zeta0=sl * sim.ZETA_PER_SLOT,
        linear_rescale_on_knobs=[xt.LinearRescale(knob_name='beambeam_scale', v0=0.0, dv=0.15)])
    footprints[sl] = dict(label=label, qx=fp.qx, qy=fp.qy)
    print(f'  slot {sl:4d} ({label:14s}): {time.time() - t0:.1f} s  '
          f'qx [{fp.qx.min():.4f}, {fp.qx.max():.4f}]')

import pandas as pd
pd.to_pickle(footprints, os.path.join(mb.HERE, 'footprints_coll.pkl'))
print('saved footprints_coll.pkl')

# ----------------------------------------------------------------------------
# Plot: footprints colored by the bunch position in the train
# ----------------------------------------------------------------------------
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

fig, ax = plt.subplots(figsize=(9, 8))
cmap = plt.get_cmap('viridis')
norm = Normalize(vmin=1, vmax=len(train))
for k, (sl, _) in zip(pos_in_train, family):
    fp = footprints[sl]
    color = cmap(norm(k + 1))
    ax.plot(fp['qx'], fp['qy'], color=color, lw=1)
    ax.plot(fp['qx'].T, fp['qy'].T, color=color, lw=1)
ax.set_xlabel(r'$q_x$')
ax.set_ylabel(r'$q_y$')
ax.set_title('Per-bunch tune footprints, LHC collision (head-on + BBLR)')
fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax,
             label='bunch position in train')
plt.tight_layout()
plt.show()


