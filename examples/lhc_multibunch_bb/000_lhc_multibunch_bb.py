# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

"""
Multi-bunch beam-beam on the FULL (thick) LHC lattice at injection (450 GeV).

Head-on and long-range beam-beam elements (BeamBeamBiGaussianMultibunch2D) are
installed at IP1/2/5/8 and the per-bunch closed solution (closed orbit + tunes)
of the two multi-bunch beams is found self-consistently. See ``_lhc_mb_common.py``
for the model (encounter geometry, convolved sizes, closed-orbit + survey
separation) which follows pytrain / TRAIN.

At nominal injection the separation bumps are kept ON, so the beams do not
collide head-on: the effect is long-range beam-beam (BBLR) only.

This "direct" variant twisses the full thick line once per bunch. The companion example
``001_multibunch_sectormaps_bb.py`` replaces the arcs by second-order maps.
"""
import os

import matplotlib.pyplot as plt

import _lhc_mb_common as C

N_ITER = int(os.environ.get('LHC_NITER', '3'))
ALL_BUNCHES = os.environ.get('LHC_ALL', '1') == '1'  # False -> bounded subset
WINDOW = int(os.environ.get('LHC_WINDOW', '48'))

# ----------------------------------------------------------------------------
env, line_b1, line_b2 = C.load_lhc()
slot_len = line_b1.get_length() / C.N_SLOTS
b_h_dist = slot_len / 2.0
gamma0 = line_b1.particle_ref.gamma0[0]
beta0 = line_b1.particle_ref.beta0[0]

C.install_markers(line_b1, mirror=False, b_h_dist=b_h_dist)
C.install_markers(line_b2, mirror=True, b_h_dist=b_h_dist)

geom, meta = C.compute_geometry(line_b1, line_b2, b_h_dist, slot_len)
for ip in C.IPS:
    print(f'  IP{ip}: head-on offset = {geom[f"bb_ip{ip}_ho"]["offset"]} slots')

# Filling: bounded subset with all-IP pairings
scheme_b1, scheme_b2 = C.load_scheme()
if ALL_BUNCHES:
    slots_b1, slots_b2 = C.all_filled_slots(scheme_b1, scheme_b2)
else:
    slots_b1, slots_b2 = C.windowed_slots(scheme_b1, scheme_b2, geom, WINDOW)
print(f'  populated bunches: B1 = {len(slots_b1)}, B2 = {len(slots_b2)}')

# Install beam-beam elements in the full lines
bb_b1 = C.install_bb(line_b1, False, geom, len(slots_b2), gamma0, beta0)
bb_b2 = C.install_bb(line_b2, True, geom, len(slots_b1), gamma0, beta0)

print('Self-consistent solve on the full thick lattice:')
import time
t0 = time.time()
mbtw_b1, mbtw_b2 = C.solve_self_consistent(
    line_b1, line_b2, bb_b1, bb_b2, slots_b1, slots_b2, geom, n_iter=N_ITER)
print(f'  solve time ({len(slots_b1)}+{len(slots_b2)} bunches, {N_ITER} iters): '
      f'{time.time() - t0:.1f} s')

print(f"\nB1 tune shift: dqx in [{(mbtw_b1.qx-meta['bare_qx_b1']).min():.2e}, "
      f"{(mbtw_b1.qx-meta['bare_qx_b1']).max():.2e}]")

df_b1 = C.results_dataframe(mbtw_b1, slots_b1, meta['bare_qx_b1'], meta['bare_qy_b1'],
                            ip='ip1', reverse=False)
df_b2 = C.results_dataframe(mbtw_b2, slots_b2, meta['bare_qx_b2'], meta['bare_qy_b2'],
                            ip='ip1', reverse=True)
out_b1 = os.path.join(C.HERE, 'results_b1_full.pkl')
out_b2 = os.path.join(C.HERE, 'results_b2_full.pkl')
df_b1.to_pickle(out_b1)
df_b2.to_pickle(out_b2)
print(f'saved {out_b1}\nsaved {out_b2}')

C.plot_results(slots_b1, mbtw_b1, meta['bare_qx_b1'], meta['bare_qy_b1'],
               title_suffix='  [full lattice]')
plt.show()
