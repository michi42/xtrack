import numpy as np
import xtrack as xt

line = xt.Line.from_json(
    '../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')

sequence_name='dummy'
mng = line.to_madng(sequence_name=sequence_name)
mng._sequence_name = sequence_name

line._xdeps_vref._owner.mng = mng

rdts = ["f4000", "f3100", "f2020", "f1120", 'f1001']

def _tw_ng(line, rdts=[], tw=None):
    if tw is None:
        tw = line.twiss(method='4d')
    tw_columns = ['s', 'beta11', 'beta22', 'alfa11', 'alfa22',
                'x', 'px', 'y', 'py', 't', 'pt',
                'dx', 'dy', 'dpx', 'dpy', 'mu1', 'mu2']

    columns = tw_columns + rdts
    rdt_cmd = 'local rdts = {"' + '", "'.join(rdts) + '"}'
    send_cmd = f'py:send({{mtbl.{", mtbl.".join(columns)}}})'

    if len(rdts) > 0:
        mng_script = ('''
        local damap in MAD
        '''
        f'local seq = MADX.{mng._sequence_name}'
        '''
        -- list of RDTs
        '''
        + rdt_cmd +
        '''
        -- create phase-space damap at 4th order
        local X0 = damap {nv=6, mo=4}

        -- twiss with RDTs
        local mtbl = twiss {sequence=seq, X0=X0, trkrdt=rdts, info=2, saverdt=true}

        -- send columns to Python
        '''
        + send_cmd)
    else:
        mng_script = ('''
        local damap in MAD
        '''
        f'local seq = MADX.{mng._sequence_name}'
        '''

        -- twiss with RDTs
        local mtbl = twiss {sequence=seq, method=4, mapdef=2, implicit=true, nslice=3}

        -- send columns to Python
        '''
        + send_cmd)

    mng.send(mng_script)

    out = mng.recv()
    out_dct = {k: v for k, v in zip(columns, out)}

    # Add to table
    assert len(out[0]) == len(tw) + 1
    for nn in tw_columns:
        tw[nn+'_ng'] = out_dct[nn][:-1]
    for nn in rdts:
        tw[nn] = out_dct[nn][:-1]

    return tw

xt.Line._tw_ng = _tw_ng

tw = line._tw_ng(rdts=rdts)

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