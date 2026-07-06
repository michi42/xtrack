# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

"""
Impact of DYNAMIC BETA on the multi-bunch beam-beam closed solution in the
LHC collision scenario (6.8 TeV squeezed flat optics, head-on + BBLR).

The beam-beam elements take the effective (convolved) transverse sizes of
the colliding bunch pairs. By default these are STATIC, computed once from
the bare optics: ``sigma^2 = (beta_b1 + beta_b2) * nemitt / gamma0``. But
head-on beam-beam changes the per-bunch beta functions at the encounters
(dynamic beta, ~10% spread of beta* in this scenario), which in turn changes
the sizes, the kicks, and hence the per-bunch closed solution.

This script solves the same machine twice:

1. static sizes (as ``002_multibunch_sectormaps_collisions.py``);
2. ``dynamic_beta=True``: at every iteration the per-bunch effective sizes
   of all encounters are recomputed from the LIVE per-bunch betas of both
   beams (requires the optics-carrying mode='fast' twiss in the loop).

and compares per-bunch tunes, orbits and beta* at IP1.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt

import lhc_mb_common as mb

sim = mb.LHCMultibunchBB.collision()

N_ITER = int(os.environ.get('LHC_NITER', '3'))

# ----------------------------------------------------------------------------
# Build the machine (as in 002)
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
tw_red = red_b1.twiss()
tw_red2 = red_b2.twiss()

scheme_b1, scheme_b2 = mb.load_scheme()
slots_b1, slots_b2 = mb.all_filled_slots(scheme_b1, scheme_b2)
print(f'  populated bunches: B1 = {len(slots_b1)}, B2 = {len(slots_b2)}')

bb_b1 = sim.install_bb(red_b1, False, geom, len(slots_b2), gamma0, beta0)
bb_b2 = sim.install_bb(red_b2, True, geom, len(slots_b1), gamma0, beta0)


def reset_bb():
    """Forget the opposing-beam state (fresh solve) and restore the static
    sizes. (Sizes set through the element method: the line[] View does not
    support slice assignment on array fields.)"""
    for bb_dict, mirror in ((bb_b1, False), (bb_b2, True)):
        for name, bb in bb_dict.items():
            bb.num_other_bunches = 0
            e = geom[name]
            n_cap = len(bb.other_beam_zeta)
            bb._set_per_bunch('other_beam_sigma_x',
                              np.sqrt(e['betx'] * sim.nemitt / gamma0), n_cap)
            bb._set_per_bunch('other_beam_sigma_y',
                              np.sqrt(e['bety'] * sim.nemitt / gamma0), n_cap)


# ----------------------------------------------------------------------------
# Solve with static sizes, then with dynamic beta
# ----------------------------------------------------------------------------
results = {}
for label, dynamic_beta in (('static', False), ('dynamic beta', True)):
    reset_bb()
    print(f'Self-consistent solve ({label}):')
    t0 = time.time()
    # mode='fast' in both runs (dynamic_beta forces it anyway) so the
    # returned tables carry the per-bunch optics and the cost is comparable
    mbtw_b1, mbtw_b2 = sim.solve_self_consistent(
        red_b1, red_b2, bb_b1, bb_b2, slots_b1, slots_b2, geom,
        n_iter=N_ITER, dynamic_beta=dynamic_beta)
    print(f'  solve time ({N_ITER} iters): {time.time() - t0:.1f} s')
    results[label] = (mbtw_b1, mbtw_b2)

# ----------------------------------------------------------------------------
# Compare per-bunch tunes, orbit and beta* at IP1 (B1)
# ----------------------------------------------------------------------------
mk_b1 = mb.marker_name('bb_ip1_ho', False)


def extract(mbtw):
    return dict(
        qx=np.asarray(mbtw.qx_frac), qy=np.asarray(mbtw.qy_frac),
        x=np.array([tw['x', mk_b1] for tw in mbtw]),
        betx=np.array([tw['betx', mk_b1] for tw in mbtw]),
        bety=np.array([tw['bety', mk_b1] for tw in mbtw]),
    )


stat = extract(results['static'][0])
dyn = extract(results['dynamic beta'][0])

df_b1 = mb.results_dataframe(results['dynamic beta'][0], slots_b1,
                             tw_red.qx, tw_red.qy, ip='ip1', reverse=False)
df_b2 = mb.results_dataframe(results['dynamic beta'][1], slots_b2,
                             tw_red2.qx, tw_red2.qy, ip='ip1', reverse=True)
df_b1.to_pickle(os.path.join(mb.HERE, 'results_b1_coll_dynbeta.pkl'))
df_b2.to_pickle(os.path.join(mb.HERE, 'results_b2_coll_dynbeta.pkl'))
print('saved results_b1_coll_dynbeta.pkl / results_b2_coll_dynbeta.pkl')

dqx = mb.wrap_frac_tune(dyn['qx'] - stat['qx'])
dqy = mb.wrap_frac_tune(dyn['qy'] - stat['qy'])
print(f"\nDynamic-beta impact (B1):")
print(f"  tune change dqx in [{dqx.min():+.2e}, {dqx.max():+.2e}] "
      f"(rms {dqx.std():.2e})")
print(f"  tune change dqy in [{dqy.min():+.2e}, {dqy.max():+.2e}] "
      f"(rms {dqy.std():.2e})")
print(f"  orbit change at IP1: rms {np.std(dyn['x'] - stat['x'])*1e9:.1f} nm")
print(f"  betx* at IP1: static [{stat['betx'].min():.4f}, "
      f"{stat['betx'].max():.4f}] m, dynamic [{dyn['betx'].min():.4f}, "
      f"{dyn['betx'].max():.4f}] m")

fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

axs[0].plot(slots_b1, dqx * 1e3, '.', ms=3, label=r'$\Delta q_x$')
axs[0].plot(slots_b1, dqy * 1e3, '.', ms=3, label=r'$\Delta q_y$')
axs[0].set_ylabel(r'tune change [$10^{-3}$]')
axs[0].set_title('Per-bunch effect of dynamic beta on the closed solution '
                 '(B1, collision)')
axs[0].legend()

axs[1].plot(slots_b1, (dyn['x'] - stat['x']) * 1e9, '.', ms=3, label='x')
axs[1].set_ylabel('orbit change at IP1 [nm]')
axs[1].legend()

axs[2].plot(slots_b1, stat['betx'], '.', ms=3, label=r'$\beta_x^*$ static')
axs[2].plot(slots_b1, dyn['betx'], '.', ms=3,
            label=r'$\beta_x^*$ dynamic beta')
axs[2].plot(slots_b1, stat['bety'], '.', ms=3, label=r'$\beta_y^*$ static')
axs[2].plot(slots_b1, dyn['bety'], '.', ms=3,
            label=r'$\beta_y^*$ dynamic beta')
axs[2].set_ylabel(r'$\beta^*$ at IP1 [m]')
axs[2].set_xlabel('25 ns slot')
axs[2].legend(ncol=2, fontsize=8)

plt.tight_layout()
plt.show()
