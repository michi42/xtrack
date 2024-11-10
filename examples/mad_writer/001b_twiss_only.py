import numpy as np
import xtrack as xt

line = xt.Line.from_json(
    '../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')

tw = line.twiss(method='4d')

mng = line.to_madng(sequence_name='lhcb1')

line._xdeps_vref._owner.mng = mng

tw_columns = ['s', 'beta11', 'beta22', 'alfa11', 'alfa22',
              'x', 'px', 'y', 'py', 't', 'pt',
              'dx', 'dy', 'dpx', 'dpy', 'mu1', 'mu2']
columns = tw_columns
send_cmd = f'py:send({{mtbl.{", mtbl.".join(columns)}}})'

mng.send('''
local lhc = MADX.lhcb1

-- twiss with RDTs
local mtbl = twiss {sequence=lhc, method=4, mapdef=2, implicit=true, nslice=3}

-- send columns to Python
'''
+ send_cmd)

out = mng.recv()
out_dct = {k: v for k, v in zip(columns, out)}

# Add to table
assert len(out[0]) == len(tw) + 1
for nn in tw_columns:
    tw[nn+'_ng'] = out_dct[nn][:-1]

# dct = {k: v[:-1] for k, v in zip(colums, out)}
# dct['name'] = tw.name
# tng = xt.Table(dct)

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)

plt.plot(tw.s, np.abs(tw.f4000), label='f4000')
plt.plot(tw.s, np.abs(tw.f2020), label='f2020')
plt.plot(tw.s, np.abs(tw.f1120), label='f1120')
plt.plot(tw.s, np.abs(tw.f3100), label='f3100')
plt.xlabel('s [m]')
plt.ylabel(r'|f_{jklm}|')
plt.legend()

plt.show()