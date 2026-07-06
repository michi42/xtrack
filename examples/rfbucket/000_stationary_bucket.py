import xtrack as xt
import xpart as xp
import numpy as np

env = xt.load(['../../test_data/sps_thick/sps.seq',
                '../../test_data/sps_thick/lhc_q20.str'])
line = env.sps
line.set_particle_ref('proton', p0c=26e9)

line.append('timedelay', xt.TimeDelay(shift_zeta=0.1))

env['actcse.31632'].frequency = 200e6
env['actcse.31632'].voltage = 4.5e6
env['actcse.31632'].phase = np.pi

tw = line.twiss6d()

p = xp.generate_matched_gaussian_bunch(
    line=line,
    num_particles=100_000,
    nemitt_x=2e-6,
    nemitt_y=2e-6,
    sigma_z=12e-2)

rfb = line._get_bucket()

z_separatrix = np.linspace(rfb.z_left, rfb.z_right, 1000)
delta_separatrix = rfb.separatrix(z_separatrix)
delta_separatrix_neg = rfb.separatrix(z_separatrix, sgn=-1)

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.plot(p.zeta, p.delta, '.', markersize=0.5, alpha=0.5)
plt.plot(z_separatrix, delta_separatrix)
plt.plot(z_separatrix, delta_separatrix_neg)
plt.plot(tw.zeta[0], tw.delta[0], 'x')
plt.show()
