# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

"""
Multi-threaded (OpenMP) CPU kernels for the multi-bunch beam-beam machinery.

Builds the collision machine on second-order maps (as
``002_multibunch_sectormaps_collisions.py``), populates the beam-beam
elements with one self-consistent iteration, then times the batched
multibunch twiss of beam 1 -- both the orbit-only ``fast_orbit`` mode (used
in the solver loop) and the optics-carrying ``fast`` mode -- with the SERIAL
and the MULTI-THREADED CPU kernels, and reports the speed-up.

All the multibunch examples accept the environment variable ``LHC_OMP``
(unset/``0`` -> serial kernels; ``auto`` or a thread count -> OpenMP
kernels); prebuilt kernels exist for both flavours, so no compilation is
triggered either way. Here ``LHC_OMP`` defaults to ``auto``.
"""

import os
import time
import numpy as np

import xobjects as xo
import lhc_mb_common as mb

omp = os.environ.get('LHC_OMP', 'auto')
ctx_mt = xo.ContextCpu(omp_num_threads='auto' if omp == 'auto' else int(omp))
sim = mb.LHCMultibunchBB.collision(context=ctx_mt)
n_threads = os.cpu_count() if ctx_mt.omp_num_threads == 'auto' \
    else ctx_mt.omp_num_threads
print(f'multi-threaded context: {ctx_mt.omp_num_threads} '
      f'({n_threads} threads)')

# ----------------------------------------------------------------------------
# Build the machine (as in 002) and populate the beam-beam elements
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

print('Populating the beam-beam elements (one solve iteration):')
sim.solve_self_consistent(red_b1, red_b2, bb_b1, bb_b2,
                          slots_b1, slots_b2, geom, n_iter=1)

# ----------------------------------------------------------------------------
# Timing: batched multibunch twiss of B1, serial vs multi-threaded kernels
# ----------------------------------------------------------------------------
zeta_b1 = np.array(slots_b1) * sim.ZETA_PER_SLOT
contexts = (('serial', xo.ContextCpu()),
            (f'openmp ({n_threads} threads)', ctx_mt))

print(f'\nBatched multibunch twiss of B1 ({len(slots_b1)} bunches):')
for mode in ('fast_orbit', 'fast'):
    timings = []
    for label, ctx in contexts:
        red_b1.discard_tracker()
        red_b1.build_tracker(_context=ctx)
        red_b1.twiss_multibunch(zeta_bunches=zeta_b1[:16], mode=mode,
                                show_progress=False)   # warm-up
        t0 = time.time()
        red_b1.twiss_multibunch(zeta_bunches=zeta_b1, mode=mode,
                                show_progress=False)
        timings.append(time.time() - t0)
        print(f'  mode={mode!r:13s} {label:22s}: {timings[-1]:7.1f} s')
    print(f'  mode={mode!r:13s} speed-up: x{timings[0] / timings[1]:.2f}')
