# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

"""
Plot the per-bunch beam-beam results of both beams saved by
``001_multibunch_sectormaps_bb.py`` (``results_b1.pkl`` / ``results_b2.pkl``).

Top row: per-bunch tune offsets (beam-beam tune shift) dqx, dqy for B1 and B2.
Bottom row: per-bunch closed-orbit offsets (deviation from the beam-averaged
orbit, which removes the common crossing/separation bump) dx, dy at IP1.

If pytrain result DataFrames are present (``results_b1_pytrain.pkl`` /
``results_b2_pytrain.pkl``, same format), they are overlaid for comparison.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
df_b1 = pd.read_pickle(os.path.join(HERE, 'results_b1.pkl'))
df_b2 = pd.read_pickle(os.path.join(HERE, 'results_b2.pkl'))

# optional pytrain overlay
pt_b1 = pt_b2 = None
if os.path.exists(os.path.join(HERE, 'results_b1_pytrain.pkl')):
    pt_b1 = pd.read_pickle(os.path.join(HERE, 'results_b1_pytrain.pkl'))
    pt_b2 = pd.read_pickle(os.path.join(HERE, 'results_b2_pytrain.pkl'))

fig, axs = plt.subplots(2, 2, figsize=(13, 8), sharex=True)


def panel(ax, col, scale, ylabel, title):
    if pt_b1 is not None:
        bb = np.full((3564,), np.nan)
        bb[pt_b1.index] = pt_b1[col] * scale
        ax.plot(bb, '-', ms=3, color='C2',
                alpha=1, label='B1 pytrain')
        bb = np.full((3564,), np.nan)
        bb[pt_b2.index] = pt_b2[col] * scale
        ax.plot(bb, '-', ms=3, color='C3',
                alpha=1, label='B2 pytrain')
    ax.plot(df_b1.index, df_b1[col] * scale, '.', ms=3, color='C0', label='B1 xsuite')
    ax.plot(df_b2.index, df_b2[col] * scale, '.', ms=3, color='C1', label='B2 xsuite')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(ncol=2, fontsize=8)


panel(axs[0, 0], 'dqx', 1e3, r'$\Delta q_x$ [$10^{-3}$]',
      r'Per-bunch beam-beam tune offset $\Delta q_x$')
panel(axs[0, 1], 'dqy', 1e3, r'$\Delta q_y$ [$10^{-3}$]',
      r'Per-bunch beam-beam tune offset $\Delta q_y$')
panel(axs[1, 0], 'dx', 1e6, r'orbit offset x at IP1 [$\mu$m]',
      'Per-bunch beam-beam closed-orbit offset (x)')
panel(axs[1, 1], 'dy', 1e6, r'orbit offset y at IP1 [$\mu$m]',
      'Per-bunch beam-beam closed-orbit offset (y)')
if pt_b1 is not None:   # pytrain tune tails are large (unconverged in slicing)
    axs[0, 0].set_ylim(-4, 4)
    axs[0, 1].set_ylim(-4, 4)
axs[1, 0].set_xlabel('25 ns slot')
axs[1, 1].set_xlabel('25 ns slot')

plt.suptitle('LHC injection multi-bunch beam-beam (BBLR): xsuite vs pytrain')
plt.tight_layout()
plt.show()
