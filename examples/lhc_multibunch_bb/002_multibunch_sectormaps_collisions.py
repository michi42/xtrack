# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

"""
Multi-bunch beam-beam on the LHC in COLLISION (6.8 TeV, fully squeezed
R2025aRP 15 cm flat optics, levelling-style knobs), with second-order maps.

Scenario following LHC 2025/2026 physics at end of levelling: head-on collisions
at IP1/IP5 (flat optics, H crossing in 1, V crossing in 5, separations off),
levelling offsets at IP2/IP8, spectrometers/solenoids on, octupoles powered,
tunes/chromaticity matched to 62.316/60.322 and Q' = 10. 1.1e11 p/bunch,
2.3 um normalized emittance, full 2460-bunch filling scheme.

The flattened collision optics and the pytrain reference are provided in
``test_data/lhc_2024`` (regenerate with
``test_data/lhc_2024/pytrain/regenerate_collision.py``, pytrain venv +
cpymad + acc-models-lhc checkout).
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt

import lhc_mb_common as mb

sim = mb.LHCMultibunchBB.collision()

N_ITER = int(os.environ.get('LHC_NITER', '6'))
ALL_BUNCHES = os.environ.get('LHC_ALL', '1') == '1'
WINDOW = int(os.environ.get('LHC_WINDOW', '48'))
COMPUTE_OPTICS_PARAMS = os.environ.get('COMPUTE_OPTICS_PARAMS', '1') == '1'

# ----------------------------------------------------------------------------
env, line_b1, line_b2 = sim.load()
slot_len = line_b1.get_length() / sim.N_SLOTS
b_h_dist = slot_len / 2.0
gamma0 = line_b1.particle_ref.gamma0[0]
beta0 = line_b1.particle_ref.beta0[0]

sim.install_markers(line_b1, mirror=False, b_h_dist=b_h_dist)
sim.install_markers(line_b2, mirror=True, b_h_dist=b_h_dist)

geom, meta = sim.compute_geometry(line_b1, line_b2, b_h_dist, slot_len)
print(f'  bare tunes B1 {meta["bare_qx_b1"]:.5f}/{meta["bare_qy_b1"]:.5f}  '
      f'B2 {meta["bare_qx_b2"]:.5f}/{meta["bare_qy_b2"]:.5f}')

print('  building second-order maps between the beam-beam markers...')
red_b1 = line_b1.get_line_with_second_order_maps(split_at=sim.marker_names_b1)
red_b2 = line_b2.get_line_with_second_order_maps(split_at=sim.marker_names_b2)
for rl in (red_b1, red_b2):
    rl.twiss_default['method'] = '4d'
    rl.build_tracker(_context=sim.context)
tw_red = red_b1.twiss()
tw_red2 = red_b2.twiss()

scheme_b1, scheme_b2 = mb.load_scheme()
if ALL_BUNCHES:
    slots_b1, slots_b2 = mb.all_filled_slots(scheme_b1, scheme_b2)
else:
    slots_b1, slots_b2 = sim.windowed_slots(scheme_b1, scheme_b2, geom, WINDOW)
print(f'  populated bunches: B1 = {len(slots_b1)}, B2 = {len(slots_b2)}')

bb_b1 = sim.install_bb(red_b1, False, geom, len(slots_b2), gamma0, beta0)
bb_b2 = sim.install_bb(red_b2, True, geom, len(slots_b1), gamma0, beta0)

print('Self-consistent solve (head-on + long-range):')
t0 = time.time()
mbtw_b1, mbtw_b2 = sim.solve_self_consistent(
    red_b1, red_b2, bb_b1, bb_b2, slots_b1, slots_b2, geom, n_iter=N_ITER)
print(f'  solve time ({len(slots_b1)}+{len(slots_b2)} bunches, {N_ITER} iters): '
      f'{time.time() - t0:.1f} s')

if COMPUTE_OPTICS_PARAMS:
    print('Final mode="fast" twiss (per-bunch optics + global quantities):')
    t0 = time.time()
    mbtw_b1 = red_b1.twiss_multibunch(
        zeta_bunches=np.array(slots_b1) * sim.ZETA_PER_SLOT, mode='fast')
    mbtw_b2 = red_b2.twiss_multibunch(
        zeta_bunches=np.array(slots_b2) * sim.ZETA_PER_SLOT, mode='fast')
    print(f'  final twiss (both beams): {time.time() - t0:.1f} s')

dqx_b1 = mb.wrap_frac_tune(mbtw_b1.qx - tw_red.qx)
print(f"\nB1 tune shift: dqx in [{dqx_b1.min():.2e}, {dqx_b1.max():.2e}]")

df_b1 = mb.results_dataframe(mbtw_b1, slots_b1, tw_red.qx, tw_red.qy,
                            ip='ip1', reverse=False)
df_b2 = mb.results_dataframe(mbtw_b2, slots_b2, tw_red2.qx, tw_red2.qy,
                            ip='ip1', reverse=True)
df_b1.to_pickle(os.path.join(mb.HERE, 'results_b1_coll.pkl'))
df_b2.to_pickle(os.path.join(mb.HERE, 'results_b2_coll.pkl'))
print('saved results_b1_coll.pkl / results_b2_coll.pkl')

mb.plot_results(slots_b1, mbtw_b1, tw_red.qx, tw_red.qy,
               title_suffix='  [collision, 6.8 TeV]')
if COMPUTE_OPTICS_PARAMS:
    mb.plot_global_quantities(slots_b1, mbtw_b1, slots_b2, mbtw_b2)
plt.show()
