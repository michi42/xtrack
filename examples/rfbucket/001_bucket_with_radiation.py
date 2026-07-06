import xtrack as xt
import xpart as xp
import numpy as np
import xobjects as xo

np.random.seed(12345)

sigma_z = 12e-2
shift_zeta = -0.1

env = xt.load(['../../test_data/sps_thick/sps.seq',
                '../../test_data/sps_thick/lhc_q20.str'])
line = env.sps
line.set_particle_ref('positron', p0c=30e9)

line.cycle('qf.30010')

env['actcse.31632'].frequency = 200e6
env['actcse.31632'].voltage = 100e6
env['actcse.31632'].phase = np.pi

line.configure_radiation(model='mean')
line.compensate_radiation_energy_loss()
tw_rad = line.twiss6d()