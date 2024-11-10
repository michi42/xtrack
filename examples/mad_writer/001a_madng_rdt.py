import numpy as np
import xtrack as xt

line = xt.Line.from_json(
    '../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')

tw = line.twiss(method='4d')

mng = line.to_madng(sequence_name='lhcb1')

rdts = ["f4000", "f3100", "f2020", "f1120"]
colums = ['s', 'beta11'] + rdts
rdt_cmd = 'local rdts = {"' + '", "'.join(rdts) + '"}'
send_cmd = f'py:send({{mtbl.{", mtbl.".join(colums)}}})'

mng.send('''
local damap in MAD
local lhc = MADX.lhcb1

-- list of octupolar RDTs
'''
+ rdt_cmd +
'''
-- create phase-space damap at 4th order
local X0 = damap {nv=6, mo=4}

-- twiss with RDTs
local mtbl = twiss {sequence=lhc, X0=X0, trkrdt=rdts, info=2, saverdt=true}

-- send columns to Python
'''
+ send_cmd)

out = mng.recv()

# Add to table
assert len(out[0]) == len(tw) + 1
for ii, nn in enumerate(colums):
    if nn in tw.keys():
        nn = nn + '_ng'
    tw[nn] = out[ii][:-1]

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