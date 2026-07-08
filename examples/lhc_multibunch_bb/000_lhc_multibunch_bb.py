# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

"""
Multi-bunch beam-beam on the FULL (thick) LHC lattice in collisions (6.8 TeV,
fully squeezed R2025aRP 15 cm flat optics, end-of-levelling knobs).

Head-on and long-range beam-beam elements (BeamBeamBiGaussianMultibunch2D) are
installed at IP1/2/5/8 and the per-bunch closed solution (closed orbit + tunes)
of the two multi-bunch beams is found self-consistently. See ``lhc_mb_common.py``
for the model (encounter geometry, convolved sizes, closed-orbit + survey
separation) which follows pytrain / TRAIN. The beams collide head-on at
IP1/IP5 (levelling offsets at IP2/IP8), so the effect is head-on + BBLR.

This "direct" variant twisses the full thick line (no sector-map reduction).
The companion example ``002_multibunch_sectormaps_collisions.py`` replaces
the arcs by second-order maps (much faster).
"""
import os

import matplotlib.pyplot as plt

import lhc_mb_common as mb

sim = mb.LHCMultibunchBB.collision()

N_ITER = int(os.environ.get('LHC_NITER', '3'))
ALL_BUNCHES = os.environ.get('LHC_ALL', '0') == '1'  # False -> bounded subset
WINDOW = int(os.environ.get('LHC_WINDOW', '48'))

# ----------------------------------------------------------------------------
env, line_b1, line_b2 = sim.load()
slot_len = line_b1.get_length() / sim.N_SLOTS
b_h_dist = slot_len / 2.0

sim.install_markers(line_b1, mirror=False, b_h_dist=b_h_dist)
sim.install_markers(line_b2, mirror=True, b_h_dist=b_h_dist)

geom, meta = sim.compute_geometry(line_b1, line_b2, b_h_dist, slot_len)
for ip in sim.ips:
    print(f'  IP{ip}: head-on offset = {geom[f"bb_ip{ip}_ho"]["offset"]} slots')

# Filling: bounded subset with all-IP pairings
scheme_b1, scheme_b2 = mb.load_scheme()
if ALL_BUNCHES:
    slots_b1, slots_b2 = mb.all_filled_slots(scheme_b1, scheme_b2)
else:
    slots_b1, slots_b2 = sim.windowed_slots(scheme_b1, scheme_b2, WINDOW)
print(f'  populated bunches: B1 = {len(slots_b1)}, B2 = {len(slots_b2)}')

# Install beam-beam elements in the full lines
bb_b1 = sim.install_bb(line_b1, False, len(slots_b2))
bb_b2 = sim.install_bb(line_b2, True, len(slots_b1))

print('Self-consistent solve on the full thick lattice:')
import time
t0 = time.time()
mbtw_b1, mbtw_b2 = sim.solve_self_consistent(
    line_b1, line_b2, bb_b1, bb_b2, slots_b1, slots_b2, n_iter=N_ITER)
print(f'  solve time ({len(slots_b1)}+{len(slots_b2)} bunches, {N_ITER} iters): '
      f'{time.time() - t0:.1f} s')

dqx_b1 = mb.wrap_frac_tune(mbtw_b1.qx - meta['bare_qx_b1'])
print(f"\nB1 tune shift: dqx in [{dqx_b1.min():.2e}, {dqx_b1.max():.2e}]")

df_b1 = mb.results_dataframe(mbtw_b1, slots_b1, meta['bare_qx_b1'], meta['bare_qy_b1'],
                            ip='ip1', reverse=False)
df_b2 = mb.results_dataframe(mbtw_b2, slots_b2, meta['bare_qx_b2'], meta['bare_qy_b2'],
                            ip='ip1', reverse=True)
out_b1 = os.path.join(mb.HERE, 'results_b1_coll_full.pkl')
out_b2 = os.path.join(mb.HERE, 'results_b2_coll_full.pkl')
df_b1.to_pickle(out_b1)
df_b2.to_pickle(out_b2)
print(f'saved {out_b1}\nsaved {out_b2}')

mb.plot_results(slots_b1, mbtw_b1, meta['bare_qx_b1'], meta['bare_qy_b1'],
               title_suffix='  [full thick lattice, collision]')
plt.show()
