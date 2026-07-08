# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

"""Cross-check of the LHC multi-bunch beam-beam machinery against pytrain
(TRAIN): builds the sector-map model (as in ``examples/lhc_multibunch_bb``,
minimal logic duplicated below so the test is self-contained) for both the
injection (BBLR only) and the collision (6.8 TeV squeezed, head-on + BBLR)
scenarios -- all 2460+2460 bunches -- and compares the per-bunch closed-orbit
deviations and tune shifts at IP1 with the stored pytrain references
(``test_data/lhc_2024/pytrain/pytrain_{injection,collision}.json``,
regenerated with the scripts in the same directory).

Model (following pytrain / TRAIN): long-range encounter ``n`` sits at
``n * b_h_dist`` from the IP (``b_h_dist`` = half a 25 ns slot); beam-1 bunch
``b1`` meets beam-2 bunch ``b1 + offset`` with ``offset = round(2 * (s -
s_IP1) / slot_len) mod N_SLOTS``; the coherent (rigid-bunch) kick uses the
convolved pair size (``coherent=True`` elements convolve each beam's own
``sigma = sqrt(beta * nemitt / gamma0)``); the beam separation is the
closed-orbit difference plus the SIGNED geometric survey separation of the
two rings. Bunches are labelled by ``zeta = slot * ZETA_PER_SLOT`` and the
elements pair them via ``zeta_offset``/``zeta_period``.

NOTE: slow (several minutes per scenario, dominated by building the
second-order maps of both beams).
"""

import json
import pathlib

import numpy as np
import pytest
from scipy.constants import c as clight

import xobjects as xo
import xtrack as xt
import xfields as xf

test_data_folder = pathlib.Path(
    __file__).parent.joinpath('../test_data').absolute()
lhc_data = test_data_folder / 'lhc_2024'

N_SLOTS = 3564
ZETA_PER_SLOT = 25e-9 * clight   # per-bunch zeta label spacing [m]
IPS = (1, 2, 5, 8)
NPARASITIC = 45                  # long-range encounters per IP side

SCENARIOS = {
    'injection': dict(p0c=450e9, bunch_intensity=1.8e11, nemitt=1.5e-6,
                      optics='injection_optics.madx', n_iter=3),
    'collision': dict(p0c=6800e9, bunch_intensity=1.1e11, nemitt=2.3e-6,
                      optics='collision_optics_15cm_flat_2026.madx',
                      n_iter=4),
}

# pytrain is a thin (slicefactor 8) model solved with second-order beam-beam
# maps, xsuite an exact-kick model built from a thick lattice, so the two
# never agree to machine precision:
# - orbits: sub-nm to few-nm agreement in collision; at injection a few
#   pytrain bunches show um-level slicing artifacts (measured max 2.5e-5).
# - tune shifts: at injection the pytrain per-bunch tunes are not converging
#   well; in collision the second-order map truncation of the head-on kick
#   gives a ~2% shift (max 5.1e-4).
TOLERANCES = {
    #             orbit atol   tune atol   tune rms
    'injection': dict(orbit=5e-5, tune=1e-1, tune_rms=1e-2),
    'collision': dict(orbit=2e-7, tune=2e-3, tune_rms=1e-3),
}


def _wrap_frac_tune(v):
    """Tune difference on the fractional-tune circle, wrapped to
    (-0.5, 0.5]."""
    return (np.asarray(v) + 0.5) % 1.0 - 0.5


def _encounter_specs():
    """Yield (name, ip, signed_n); signed_n == 0 is the head-on encounter."""
    for ip in IPS:
        yield f'bb_ip{ip}_ho', ip, 0
        for n in range(1, NPARASITIC + 1):
            yield f'bb_ip{ip}_r{n:02d}', ip, +n
            yield f'bb_ip{ip}_l{n:02d}', ip, -n


def _marker_name(name, mirror):
    return name + ('_b2' if mirror else '_b1')


ENC_NAMES = [n for n, _, _ in _encounter_specs()]
MARKER_NAMES = {False: [_marker_name(n, False) for n in ENC_NAMES],
                True: [_marker_name(n, True) for n in ENC_NAMES]}


def _load_lines(p0c, optics):
    env = xt.load(str(lhc_data / 'lhc.seq'), format='madx',
                  reverse_lines=['lhcb2'])
    for ln in (env.lhcb1, env.lhcb2):
        ln.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, p0c=p0c)
    env.vars.load(str(lhc_data / optics))
    for ln in (env.lhcb1, env.lhcb2):
        ln.twiss_default['method'] = '4d'
        ln.cycle(name_first_element='ip3', inplace=True)  # no IP at s=0
    return env.lhcb1, env.lhcb2


def _install_markers(line, mirror, b_h_dist):
    # the reversed beam-2 line mirrors left/right so a marker name is the
    # same physical point in both beams
    env = line.env
    places = []
    for name, ip, sn in _encounter_specs():
        at = (-sn if mirror else sn) * b_h_dist + 1e-6
        places.append(env.place(
            env.new(_marker_name(name, mirror), xt.Marker),
            at=at, from_=f'ip{ip}'))
    line.insert(places)


def _compute_geometry(line_b1, line_b2, slot_len):
    """Per-encounter slot offset, convolved betas and SIGNED survey
    separation (TRAIN convention: sign from the direction of the separation
    vector relative to the ring azimuth; flips between IP1/5 and IP2/8)."""
    tw1 = line_b1.twiss()
    tw2 = line_b2.twiss()
    sv1 = line_b1.survey()
    sv2 = line_b2.survey()

    s_ip1 = tw1['s', 'ip1']
    ip_pos = {}
    for ip in IPS:
        ip_pos[ip] = (
            np.array([sv1['X', f'ip{ip}'], sv1['Y', f'ip{ip}'],
                      sv1['Z', f'ip{ip}']]),
            np.array([sv2['X', f'ip{ip}'], sv2['Y', f'ip{ip}'],
                      sv2['Z', f'ip{ip}']]))
    r1 = sv1.rows[MARKER_NAMES[False]]
    r2 = sv2.rows[MARKER_NAMES[True]]
    m1 = np.stack([r1.X, r1.Y, r1.Z], axis=1)
    m2 = np.stack([r2.X, r2.Y, r2.Z], axis=1)

    length = line_b1.get_length()
    geom = {}
    for j, (name, ip, sn) in enumerate(_encounter_specs()):
        n1, n2 = MARKER_NAMES[False][j], MARKER_NAMES[True][j]
        s_marker = tw1['s', n1]
        offset = int(round(2 * (s_marker - s_ip1) / slot_len)) % N_SLOTS
        s1 = m1[j] - ip_pos[ip][0]
        s2 = m2[j] - ip_pos[ip][1]
        s2 = np.array([-s2[0], s2[1], -s2[2]])   # B2 survey rotated 180 deg
        d = s1 - s2
        ang = np.arctan2(d[2], d[0]) - 2 * np.pi * s_marker / length
        ang = (ang + np.pi) % (2 * np.pi) - np.pi
        xsign = 1.0 if abs(ang) <= np.pi / 2 else -1.0
        geom[name] = dict(
            offset=offset,
            betx_b1=float(tw1['betx', n1]),
            bety_b1=float(tw1['bety', n1]),
            betx_b2=float(tw2['betx', n2]),
            bety_b2=float(tw2['bety', n2]),
            sep_x=float(np.hypot(d[0], d[2]) * xsign),
            sep_y=float(d[1]),
        )
    return geom


def _install_bb(line, mirror, geom, n_other, nemitt, gamma0, beta0):
    env = line.env
    places = []
    names = []
    for name, ip, sn in _encounter_specs():
        e = geom[name]
        # beam1 pairs b2 = b1 + offset; beam2 pairs b1 = b2 - offset
        zoff = (-e['offset'] if mirror else e['offset']) * ZETA_PER_SLOT
        # coherent rigid-bunch kick: the element convolves this beam's own
        # size with each opposing bunch's size
        own, oth = ('b2', 'b1') if mirror else ('b1', 'b2')
        bb = xf.BeamBeamBiGaussianMultibunch2D(
            num_bunches=n_other,
            zeta_offset=zoff, zeta_match_tol=0.4 * ZETA_PER_SLOT,
            zeta_period=N_SLOTS * ZETA_PER_SLOT,   # pairing wraps the ring
            other_beam_q0=1.0, other_beam_beta0=beta0,
            coherent=True,
            sigma_x=np.sqrt(e[f'betx_{own}'] * nemitt / gamma0),
            sigma_y=np.sqrt(e[f'bety_{own}'] * nemitt / gamma0),
            other_beam_sigma_x=np.sqrt(e[f'betx_{oth}'] * nemitt / gamma0),
            other_beam_sigma_y=np.sqrt(e[f'bety_{oth}'] * nemitt / gamma0),
            _context=line._context)
        elname = _marker_name(name, mirror) + '_bb'
        places.append(env.place(elname, bb, at=_marker_name(name, mirror)))
        names.append((name, elname))
    line.insert(places)
    return {name: line[elname] for name, elname in names}


def _update_opposing(bb_dict, mbtw_other, zeta_other, marker_names_other,
                     geom, p0c, bunch_intensity):
    """Write the opposing beam's per-bunch orbit + survey separation into the
    beam-beam elements. Between the two (opposite-parity) beam lines x flips"""
    # mbtw['x', names] resolves the marker rows once and slices the numpy
    # columns of all bunch tables (fast multi-element access)
    xs = -mbtw_other['x', marker_names_other]
    ys = mbtw_other['y', marker_names_other]
    p = xt.Particles(p0c=p0c, mass0=xt.PROTON_MASS_EV, q0=1.0,
                     x=np.zeros(len(zeta_other)), y=np.zeros(len(zeta_other)),
                     zeta=zeta_other, weight=bunch_intensity)
    for j, name in enumerate(ENC_NAMES):
        p.x[:] = xs[:, j] - geom[name]['sep_x']
        p.y[:] = ys[:, j] - geom[name]['sep_y']
        bb_dict[name].update_from_other_beam(p)


def _run_xsuite_scenario(scenario):
    par = SCENARIOS[scenario]
    line_b1, line_b2 = _load_lines(par['p0c'], par['optics'])
    slot_len = line_b1.get_length() / N_SLOTS
    b_h_dist = slot_len / 2.0
    gamma0 = line_b1.particle_ref.gamma0[0]
    beta0 = line_b1.particle_ref.beta0[0]

    _install_markers(line_b1, mirror=False, b_h_dist=b_h_dist)
    _install_markers(line_b2, mirror=True, b_h_dist=b_h_dist)
    geom = _compute_geometry(line_b1, line_b2, slot_len)

    red_b1 = line_b1.get_line_with_second_order_maps(
        split_at=MARKER_NAMES[False])
    red_b2 = line_b2.get_line_with_second_order_maps(
        split_at=MARKER_NAMES[True])
    for rl in (red_b1, red_b2):
        rl.twiss_default['method'] = '4d'
    tw_red = red_b1.twiss()
    tw_red2 = red_b2.twiss()

    with open(lhc_data / '25ns_2460b_2448_2092_2239_144bpi_20inj.json') as f:
        scheme = json.load(f)
    slots_b1 = np.where(np.array(scheme['schemebeam1']) > 0)[0]
    slots_b2 = np.where(np.array(scheme['schemebeam2']) > 0)[0]
    zeta_b1 = slots_b1 * ZETA_PER_SLOT
    zeta_b2 = slots_b2 * ZETA_PER_SLOT

    bb_b1 = _install_bb(red_b1, False, geom, len(slots_b2), par['nemitt'],
                        gamma0, beta0)
    bb_b2 = _install_bb(red_b2, True, geom, len(slots_b1), par['nemitt'],
                        gamma0, beta0)

    # self-consistent solve: iterate the fast multibunch twiss on both
    # beams, feeding each beam's orbits into the other beam's elements
    for _ in range(par['n_iter']):
        mbtw_b1 = red_b1.twiss_multibunch(zeta_bunches=zeta_b1,
                                          mode='fast_orbit',
                                          show_progress=False)
        mbtw_b2 = red_b2.twiss_multibunch(zeta_bunches=zeta_b2,
                                          mode='fast_orbit',
                                          show_progress=False)
        _update_opposing(bb_b1, mbtw_b2, zeta_b2, MARKER_NAMES[True], geom,
                         p0c=par['p0c'], bunch_intensity=par['bunch_intensity'])
        _update_opposing(bb_b2, mbtw_b1, zeta_b1, MARKER_NAMES[False], geom,
                         p0c=par['p0c'], bunch_intensity=par['bunch_intensity'])

    def extract(mbtw, slots, bare_qx, bare_qy, mirror):
        marker = _marker_name('bb_ip1_ho', mirror)
        x = mbtw['x', marker]
        if mirror:
            x = -x   # reversed beam-2 line -> physical frame
        y = mbtw['y', marker]
        return dict(slots=slots,
                    dx=x - x.mean(), dy=y - y.mean(),
                    dqx=_wrap_frac_tune(mbtw.qx - bare_qx),
                    dqy=_wrap_frac_tune(mbtw.qy - bare_qy))

    return (extract(mbtw_b1, slots_b1, tw_red.qx, tw_red.qy, mirror=False),
            extract(mbtw_b2, slots_b2, tw_red2.qx, tw_red2.qy, mirror=True))


@pytest.mark.parametrize('scenario', ['injection', 'collision'])
def test_lhc_multibunch_train(scenario):
    with open(test_data_folder / 'lhc_2024' / 'pytrain'
              / f'pytrain_{scenario}.json') as fid:
        ref = json.load(fid)
    tol = TOLERANCES[scenario]

    res_b1, res_b2 = _run_xsuite_scenario(scenario)

    for beam, res in (('b1', res_b1), ('b2', res_b2)):
        rr = ref[beam]
        assert np.array_equal(res['slots'], np.asarray(rr['slots']))

        # per-bunch closed-orbit deviation (from the bunch average) at IP1
        xo.assert_allclose(res['dx'], np.asarray(rr['dx']),
                           rtol=0, atol=tol['orbit'])
        xo.assert_allclose(res['dy'], np.asarray(rr['dy']),
                           rtol=0, atol=tol['orbit'])

        # per-bunch beam-beam tune shifts (fractional-tune circle), larger
        # tolerance: pytrain tunes carry thin-slicing / map-truncation errors
        for col in ('dqx', 'dqy'):
            dq_xs = res[col]
            dq_pt = _wrap_frac_tune(np.asarray(rr[col]))
            xo.assert_allclose(dq_xs, dq_pt, rtol=0, atol=tol['tune'])
            rms = float(np.sqrt(np.mean(
                _wrap_frac_tune(dq_xs - dq_pt) ** 2)))
            assert rms < tol['tune_rms'], (
                f'{scenario} {beam} {col}: RMS tune-shift difference '
                f'{rms:.2e} exceeds {tol["tune_rms"]:.0e}')
