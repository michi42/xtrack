# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

"""
Multi-bunch beam-beam on the LHC at injection, sped up with second-order maps.

Same physics as ``000_lhc_multibunch_bb.py`` (long-range beam-beam at injection
-- separation bumps kept on, so no head-on collisions -- with per-bunch
self-consistent closed solution), but the machine between the beam-beam
encounters is replaced by second-order Taylor maps:

    reduced line  =  [ map(arc), BB, map(arc), BB, ... ]

The beam-beam elements stay EXACT; only the arcs are approximated (to second
order, as in pytrain's sector maps). The reduced line has ~a few hundred
elements instead of ~13000, so each per-bunch twiss is orders of magnitude
faster -- fast enough to solve ALL bunches of the filling scheme.

The script prints a timing comparison of one full-lattice twiss vs one
reduced-line twiss. The same CpuContext is used throughout (via the single
environment), so the small reduced-line kernel is compiled once and reused for
both beams and every per-bunch twiss.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt

import _lhc_mb_common as C

N_ITER = int(os.environ.get('LHC_NITER', '3'))
ALL_BUNCHES = os.environ.get('LHC_ALL', '1') == '1'  # False -> bounded subset
WINDOW = int(os.environ.get('LHC_WINDOW', '48'))

# ----------------------------------------------------------------------------
# Load, install markers, get per-encounter geometry (cached)
# ----------------------------------------------------------------------------
env, line_b1, line_b2 = C.load_lhc()
slot_len = line_b1.get_length() / C.N_SLOTS
b_h_dist = slot_len / 2.0
gamma0 = line_b1.particle_ref.gamma0[0]
beta0 = line_b1.particle_ref.beta0[0]

C.install_markers(line_b1, mirror=False, b_h_dist=b_h_dist)
C.install_markers(line_b2, mirror=True, b_h_dist=b_h_dist)

geom, meta = C.get_geometry(line_b1, line_b2, b_h_dist, slot_len)

# ----------------------------------------------------------------------------
# Timing comparison and reduced-line construction
# ----------------------------------------------------------------------------
print('  building second-order maps between the beam-beam markers...')
red_b1 = line_b1.get_line_with_second_order_maps(split_at=C.MARKER_NAMES_B1)
red_b2 = line_b2.get_line_with_second_order_maps(split_at=C.MARKER_NAMES_B2)
for rl in (red_b1, red_b2):
    rl.twiss_default['method'] = '4d'

# Timing comparison (warm: compile each kernel once before timing)
line_b1.twiss(); red_b1.twiss()
t0 = time.time(); line_b1.twiss(); t_full = time.time() - t0
t0 = time.time(); tw_red = red_b1.twiss(); t_red = time.time() - t0
print(f'  one full-lattice twiss: {t_full:.3f} s ({len(line_b1.element_names)} elements)')
print(f'  one reduced-line twiss: {t_red:.3f} s ({len(red_b1.element_names)} elements)'
      f'  ->  speed-up x{t_full/t_red:.1f}')
# The reduced line reproduces the FRACTIONAL tunes (its integer tune is not
# meaningful -- the maps carry phase advance only modulo 2 pi).
print(f'  fractional tunes reduced/full: '
      f'qx {tw_red.qx % 1:.5f} / {meta["bare_qx_b1"] % 1:.5f}   '
      f'qy {tw_red.qy % 1:.5f} / {meta["bare_qy_b1"] % 1:.5f}')

# ----------------------------------------------------------------------------
# Filling: all bunches (feasible thanks to the fast reduced twiss)
# ----------------------------------------------------------------------------
scheme_b1, scheme_b2 = C.load_scheme()
if ALL_BUNCHES:
    slots_b1, slots_b2 = C.all_filled_slots(scheme_b1, scheme_b2)
else:
    slots_b1, slots_b2 = C.windowed_slots(scheme_b1, scheme_b2, geom, WINDOW)
print(f'  populated bunches: B1 = {len(slots_b1)}, B2 = {len(slots_b2)}')

# Install beam-beam elements in the reduced lines (markers were kept)
bb_b1 = C.install_bb(red_b1, False, geom, len(slots_b2), gamma0, beta0)
bb_b2 = C.install_bb(red_b2, True, geom, len(slots_b1), gamma0, beta0)

print('Self-consistent solve on the reduced (second-order-map) lines:')
t0 = time.time()
mbtw_b1, mbtw_b2 = C.solve_self_consistent(
    red_b1, red_b2, bb_b1, bb_b2, slots_b1, slots_b2, geom, n_iter=N_ITER)
print(f'  solve time ({len(slots_b1)}+{len(slots_b2)} bunches, {N_ITER} iters): '
      f'{time.time() - t0:.1f} s')

# Reference the tune shift to each reduced line's own bare tune (on the
# fractional-tune circle: fast-mode twiss returns fractional tunes)
tw_red2 = red_b2.twiss()
dqx_b1 = C.wrap_frac_tune(mbtw_b1.qx - tw_red.qx)
print(f"\nB1 tune shift: dqx in [{dqx_b1.min():.2e}, {dqx_b1.max():.2e}]")

# Save per-bunch results of both beams as DataFrames
df_b1 = C.results_dataframe(mbtw_b1, slots_b1, tw_red.qx, tw_red.qy,
                            ip='ip1', reverse=False)
df_b2 = C.results_dataframe(mbtw_b2, slots_b2, tw_red2.qx, tw_red2.qy,
                            ip='ip1', reverse=True)
out_b1 = os.path.join(C.HERE, 'results_b1.pkl')
out_b2 = os.path.join(C.HERE, 'results_b2.pkl')
df_b1.to_pickle(out_b1)
df_b2.to_pickle(out_b2)
print(f'saved {out_b1}\nsaved {out_b2}')

C.plot_results(slots_b1, mbtw_b1, tw_red.qx, tw_red.qy,
               title_suffix='  [second-order maps]')
plt.show()
