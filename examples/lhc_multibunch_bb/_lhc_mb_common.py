# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

"""
Shared helpers for the LHC multi-bunch beam-beam examples
(``000_lhc_multibunch_bb.py`` and ``001_multibunch_sectormaps_bb.py``).

Both examples install 2D beam-beam elements
(``xfields.BeamBeamBiGaussianMultibunch2D``) for head-on and long-range (LR)
encounters at IP1/2/5/8 and find the per-bunch closed solution of the two
multi-bunch beams self-consistently. They differ only in the line they twiss:
the first uses the full thick LHC line, the second replaces the arcs between the
beam-beam elements by second-order Taylor maps (much faster twiss).

Model (following pytrain / TRAIN):

* Encounter geometry: LR encounter ``n`` sits at ``n * b_h_dist`` from the IP,
  ``b_h_dist = L / n_slots / 2`` (half a 25 ns slot). Beam-1 bunch ``b1`` meets
  beam-2 bunch ``b2 = b1 + offset`` with
  ``offset = round(2 * (s_marker - s_IP1) / slot_len)`` (mod n_slots) -> 0 at
  IP1 and IP5, ~891 at IP2, ~2670 at IP8.
* Convolved (coherent) size: ``sigma = sqrt(eps_n * (beta_b1 + beta_b2) / gamma)``
  reproduced with ``other_beam_betx = beta_b1 + beta_b2`` and normalized
  emittance ``eps_n``.
* Beam separation = closed-orbit difference (crossing AND separation bumps, from
  the live twiss) PLUS the geometric survey separation of the two rings. The
  latter is ~0 in the common vacuum chamber and rises through the recombination
  region to the 194 mm arc separation, which is what makes the LR encounters
  beyond ~D1 negligible. The nominal injection separation bumps are kept ON, so
  the beams do not collide head-on: the effect studied here is long-range (BBLR).

The per-bunch macroparticle is labelled by its 25 ns slot through
``zeta = slot * ZETA_PER_SLOT``; the ``zeta_offset`` of each element encodes the
encounter slot offset so ``twiss_multibunch`` pairs the right bunches.
"""

import os
import json
import numpy as np

import xtrack as xt
import xfields as xf

# ----------------------------------------------------------------------------
# Physics / machine configuration
# ----------------------------------------------------------------------------
P0C = 450e9                 # injection momentum [eV]
BUNCH_INTENSITY = 1.8e11    # protons per bunch
NEMITT = 1.5e-6             # normalized emittance [m rad], both planes/beams

IPS = [int(v) for v in os.environ.get('LHC_IPS', '1,2,5,8').split(',')]
NPARASITIC = int(os.environ.get('LHC_NPAR', '45'))  # LR encounters per IP side
N_SLOTS = 3564

ZETA_PER_SLOT = 1e-3        # abstract per-bunch zeta label spacing [m]

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', '..', 'test_data', 'lhc_2024')
SCHEME_FILE = os.path.join(DATA, '25ns_2460b_2448_2092_2239_144bpi_20inj.json')
_IPTAG = ''.join(str(i) for i in IPS)
GEOM_CACHE = os.path.join(HERE, f'_lhc_bb_geometry_ip{_IPTAG}_np{NPARASITIC}.json')


# ----------------------------------------------------------------------------
# Encounter bookkeeping
# ----------------------------------------------------------------------------
def encounter_specs():
    """Yield (name, ip, signed_n); signed_n == 0 is the head-on encounter."""
    for ip in IPS:
        yield f'bb_ip{ip}_ho', ip, 0
        for n in range(1, NPARASITIC + 1):
            yield f'bb_ip{ip}_r{n:02d}', ip, +n
            yield f'bb_ip{ip}_l{n:02d}', ip, -n


def marker_name(name, mirror):
    return name + ('_b2' if mirror else '_b1')


ENC_NAMES = [n for n, _, _ in encounter_specs()]
MARKER_NAMES_B1 = [marker_name(n, False) for n in ENC_NAMES]
MARKER_NAMES_B2 = [marker_name(n, True) for n in ENC_NAMES]


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_lhc():
    """Load both LHC beams at injection. Returns (env, line_b1, line_b2).

    The nominal injection optics are used as-is: the separation bumps are kept ON
    (as at real injection), so the beams are separated at the IPs and DO NOT
    collide head-on. The beam-beam effect is then long-range only (BBLR), which
    is what this study targets. The head-on elements are still installed but see
    the full IP separation and contribute negligibly.

    All lines share the environment's single context, so the tracking kernel is
    compiled once per distinct line structure and reused."""
    with open(os.path.join(DATA, 'lhc.seq')) as fid:
        seq_text = fid.read().replace(' at=', 'at:=')
    env = xt.load(string=seq_text, format='madx', reverse_lines=['lhcb2'])
    env.lhcb1.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, p0c=P0C)
    env.lhcb2.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, p0c=P0C)
    env.vars.load(os.path.join(DATA, 'injection_optics.madx'))
    for ln in (env.lhcb1, env.lhcb2):
        ln.twiss_default['method'] = '4d'
        # no IP at the s=0 boundary (IP1 is at s=0 otherwise)
        ln.cycle(name_first_element='ip3', inplace=True)
    return env, env.lhcb1, env.lhcb2


def install_markers(line, mirror, b_h_dist):
    """Install the beam-beam markers. For the reversed beam-2 line the left/right
    side is mirrored so a given marker name is the same physical point in both
    beams."""
    env = line.env
    places = []
    for name, ip, sn in encounter_specs():
        at = (-sn if mirror else sn) * b_h_dist + 1e-6
        places.append(env.place(env.new(marker_name(name, mirror), xt.Marker),
                                at=at, from_=f'ip{ip}'))
    line.insert(places)


# ----------------------------------------------------------------------------
# Encounter geometry (offset, convolved betas, survey separation) with caching
# ----------------------------------------------------------------------------
def _survey_positions(sv, names):
    r = sv.rows[names]
    return np.stack([r.X, r.Y, r.Z], axis=1)   # (n, 3)


def compute_geometry(line_b1, line_b2, b_h_dist, slot_len):
    """Compute per-encounter offset, convolved betas and geometric survey
    separation. Expensive (twiss + survey of both full beams); cached to JSON."""
    print('  twiss + survey of both beams (one-off, compiles kernels)...')
    tw1 = line_b1.twiss()
    tw2 = line_b2.twiss()
    sv1 = line_b1.survey()
    sv2 = line_b2.survey()

    s_ip1 = tw1['s', 'ip1']
    # survey position of each IP for both beams (looked up once per IP)
    ip_pos1 = {ip: np.array([sv1['X', f'ip{ip}'], sv1['Y', f'ip{ip}'],
                             sv1['Z', f'ip{ip}']]) for ip in IPS}
    ip_pos2 = {ip: np.array([sv2['X', f'ip{ip}'], sv2['Y', f'ip{ip}'],
                             sv2['Z', f'ip{ip}']]) for ip in IPS}
    m1 = _survey_positions(sv1, MARKER_NAMES_B1)
    m2 = _survey_positions(sv2, MARKER_NAMES_B2)

    length = line_b1.get_length()
    geom = {}
    for j, (name, ip, sn) in enumerate(encounter_specs()):
        n1, n2 = MARKER_NAMES_B1[j], MARKER_NAMES_B2[j]
        s_marker = tw1['s', n1]
        offset = int(round(2 * (s_marker - s_ip1) / slot_len)) % N_SLOTS
        # Geometric (survey) separation of the two rings at this encounter, in
        # beam 1's frame. Beam-2 survey is rotated 180 deg about the vertical
        # (X,Z -> -X,-Z), then the SIGNED horizontal separation is obtained as in
        # TRAIN/pytrain: the horizontal-plane distance times a sign from the
        # direction of the separation vector relative to the ring azimuth
        # (2*pi*s/L). This sign flips between IP1/IP5 and IP2/IP8 because the
        # latter are rotated in the global frame -- it must be right for the
        # long-range orbit.
        s1 = m1[j] - ip_pos1[ip]
        s2 = m2[j] - ip_pos2[ip]
        s2 = np.array([-s2[0], s2[1], -s2[2]])
        d = s1 - s2
        ang = np.arctan2(d[2], d[0]) - 2 * np.pi * s_marker / length
        ang = (ang + np.pi) % (2 * np.pi) - np.pi
        xsign = 1.0 if abs(ang) <= np.pi / 2 else -1.0
        sep_x = float(np.hypot(d[0], d[2]) * xsign)
        sep_y = float(d[1])
        geom[name] = dict(
            ip=ip,
            offset=offset,
            betx=float(tw1['betx', n1] + tw2['betx', n2]),
            bety=float(tw1['bety', n1] + tw2['bety', n2]),
            sep_x=sep_x, sep_y=sep_y,
        )
    meta = dict(bare_qx_b1=float(tw1.qx), bare_qy_b1=float(tw1.qy),
                bare_qx_b2=float(tw2.qx), bare_qy_b2=float(tw2.qy))
    return geom, meta


def get_geometry(line_b1, line_b2, b_h_dist, slot_len, use_cache=True):
    if use_cache and os.path.exists(GEOM_CACHE):
        print(f'  loading cached encounter geometry from {GEOM_CACHE}')
        with open(GEOM_CACHE) as fid:
            blob = json.load(fid)
        return blob['geom'], blob['meta']
    geom, meta = compute_geometry(line_b1, line_b2, b_h_dist, slot_len)
    with open(GEOM_CACHE, 'w') as fid:
        json.dump(dict(geom=geom, meta=meta), fid, indent=1)
    return geom, meta


# ----------------------------------------------------------------------------
# Filling scheme
# ----------------------------------------------------------------------------
def load_scheme():
    with open(SCHEME_FILE) as fid:
        scheme = json.load(fid)
    return np.array(scheme['schemebeam1']), np.array(scheme['schemebeam2'])


def all_filled_slots(scheme_b1, scheme_b2):
    return (sorted(np.where(scheme_b1 > 0)[0].tolist()),
            sorted(np.where(scheme_b2 > 0)[0].tolist()))


def windowed_slots(scheme_b1, scheme_b2, geom, window):
    """A bounded subset: a reference window plus the windows it collides with at
    every distinct head-on offset (so all four IPs get realistic pairings)."""
    ho_offsets = sorted({geom[f'bb_ip{ip}_ho']['offset'] for ip in IPS})
    # Use the longest contiguous filled run of beam 1 as the reference window
    # (the filling is made of batches separated by gaps, so a window may be
    # shorter than requested). Its LR/IP partners in beam 2 may or may not be
    # filled -> realistic PACMAN structure at the window edges.
    filled = scheme_b1 > 0
    best_len = best_start = cur_len = cur_start = 0
    for s in range(N_SLOTS):
        if filled[s]:
            cur_start = s if cur_len == 0 else cur_start
            cur_len += 1
            if cur_len > best_len:
                best_len, best_start = cur_len, cur_start
        else:
            cur_len = 0
    window = min(window, best_len)
    ref_start = best_start
    cand = set()
    for o in ho_offsets:
        for shift in (o, -o):
            for k in range(window):
                cand.add((ref_start + shift + k) % N_SLOTS)
    return (sorted(s for s in cand if scheme_b1[s]),
            sorted(s for s in cand if scheme_b2[s]))


# ----------------------------------------------------------------------------
# Beam-beam element installation and self-consistent solve
# ----------------------------------------------------------------------------
def install_bb(line, mirror, geom, n_other, gamma0, beta0):
    """Install one beam-beam element per encounter. Returns {enc_name: element}."""
    env = line.env
    places = []
    names = []
    for name, ip, sn in encounter_specs():
        e = geom[name]
        # beam1 pairs b2 = b1 + offset (zeta_offset=+offset);
        # beam2 pairs b1 = b2 - offset (zeta_offset=-offset)
        zoff = (-e['offset'] if mirror else e['offset']) * ZETA_PER_SLOT
        bb = xf.BeamBeamBiGaussianMultibunch2D(
            num_bunches=max(n_other, 1),
            zeta_offset=zoff, zeta_match_tol=0.4 * ZETA_PER_SLOT,
            # ring is periodic in slots; encounter offsets are stored mod
            # N_SLOTS, so the pairing must wrap around the ring
            zeta_period=N_SLOTS * ZETA_PER_SLOT,
            other_beam_q0=1.0, other_beam_beta0=beta0, other_beam_gamma0=gamma0,
            other_beam_nemitt_x=NEMITT, other_beam_nemitt_y=NEMITT,
            other_beam_betx=e['betx'], other_beam_bety=e['bety'],
            _context=line._context)
        elname = marker_name(name, mirror) + '_bb'
        places.append(env.place(elname, bb, at=marker_name(name, mirror)))
        names.append((name, elname))
    line.insert(places)
    return {name: line[elname] for name, elname in names}


def _update_opposing(bb_dict, mbtw_other, slots_other, marker_names_other, geom,
                     target_is_b2):
    """Write the opposing beam's per-bunch orbit + geometric survey separation
    into the beam-beam elements, in the frame of the line that holds them.

    Between the two (opposite-parity) beam lines x flips and y does not. Matching
    TRAIN/pytrain (beam1 sees the opponent at co - sep, beam2 at co + sep), and
    accounting for the beam-2 line x-flip, the survey separation enters as
    ``-sep_x`` in x for BOTH beams, and as ``-sep_y`` (beam 1) / ``+sep_y``
    (beam 2) in y (``sep_y`` is ~0 for the LHC)."""
    xs = -np.array([tw.rows[marker_names_other].x for tw in mbtw_other])
    ys = np.array([tw.rows[marker_names_other].y for tw in mbtw_other])
    zeta_other = np.array(slots_other) * ZETA_PER_SLOT
    # NOTE: encounter slot offsets are stored mod N_SLOTS (a left LR at -n is
    # stored as N_SLOTS-n; the IP2/IP8 pairings wrap around the ring for part of
    # the bunches). The elements handle this via their `zeta_period` (set in
    # install_bb), which makes the pairing periodic in the bunch-label axis.
    y_sep_sign = 1.0 if target_is_b2 else -1.0
    for j, name in enumerate(ENC_NAMES):
        p = xt.Particles(p0c=P0C, mass0=xt.PROTON_MASS_EV, q0=1.0,
                         x=xs[:, j] - geom[name]['sep_x'],
                         y=ys[:, j] + y_sep_sign * geom[name]['sep_y'],
                         zeta=zeta_other, weight=BUNCH_INTENSITY)
        bb_dict[name].update_from_other_beam(p)


def solve_self_consistent(line_b1, line_b2, bb_b1, bb_b2,
                          slots_b1, slots_b2, geom, n_iter=3, chrom=False):
    """Iterate twiss_multibunch on both beams, feeding each beam's per-bunch
    closed orbit into the other beam's elements. Returns (mbtw_b1, mbtw_b2)."""
    zeta_b1 = np.array(slots_b1) * ZETA_PER_SLOT
    zeta_b2 = np.array(slots_b2) * ZETA_PER_SLOT
    mbtw_b1 = mbtw_b2 = None
    for it in range(n_iter):
        mbtw_b1 = line_b1.twiss_multibunch(zeta_bunches=zeta_b1, chrom=chrom)
        mbtw_b2 = line_b2.twiss_multibunch(zeta_bunches=zeta_b2, chrom=chrom)
        _update_opposing(bb_b1, mbtw_b2, slots_b2, MARKER_NAMES_B2, geom,
                         target_is_b2=False)
        _update_opposing(bb_b2, mbtw_b1, slots_b1, MARKER_NAMES_B1, geom,
                         target_is_b2=True)
        print(f'  iteration {it}: B1 qx spread {np.ptp(mbtw_b1.qx):.2e}, '
              f'B2 qx spread {np.ptp(mbtw_b2.qx):.2e}')
    return mbtw_b1, mbtw_b2


# ----------------------------------------------------------------------------
# Results as a DataFrame
# ----------------------------------------------------------------------------
def results_dataframe(mbtw, slots, bare_qx, bare_qy, ip='ip1', reverse=False):
    """Per-bunch results as a pandas DataFrame, indexed by 25 ns slot.

    Columns: qx, qy (per-bunch tunes), dqx, dqy (beam-beam tune shift vs the
    bare tune), x, y (closed orbit at the head-on marker of ``ip``, in the
    physical frame -- for the reversed beam-2 line pass ``reverse=True`` to flip
    x). ``dx``/``dy`` are the per-bunch orbit deviations from the beam average.
    """
    import pandas as pd
    marker = marker_name(f'bb_{ip}_ho', reverse)
    x = np.array([tw['x', marker] for tw in mbtw]) * (-1.0 if reverse else 1.0)
    y = np.array([tw['y', marker] for tw in mbtw])

    def _wrap_frac(v):
        # tune difference on the fractional-tune circle (fast-mode twiss
        # returns fractional tunes while the bare reference may carry an
        # integer part)
        return (np.asarray(v) + 0.5) % 1.0 - 0.5

    df = pd.DataFrame({
        'slot': np.asarray(slots),
        'qx': mbtw.qx, 'qy': mbtw.qy,
        'dqx': _wrap_frac(mbtw.qx - bare_qx),
        'dqy': _wrap_frac(mbtw.qy - bare_qy),
        'x': x, 'y': y,
        'dx': x - x.mean(), 'dy': y - y.mean(),
    }).set_index('slot')
    return df


# ----------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------
def plot_results(slots_b1, mbtw_b1, bare_qx, bare_qy, title_suffix=''):
    import matplotlib.pyplot as plt
    qx0, qy0 = bare_qx, bare_qy
    co_x = np.array([tw['x', marker_name('bb_ip1_ho', False)] for tw in mbtw_b1])
    co_y = np.array([tw['y', marker_name('bb_ip1_ho', False)] for tw in mbtw_b1])
    # per-bunch orbit deviation from the bunch-averaged orbit (removes the common
    # crossing/separation-bump orbit, leaving the bunch-by-bunch beam-beam part)
    dco_x = (co_x - co_x.mean()) * 1e6
    dco_y = (co_y - co_y.mean()) * 1e6
    fig, axs = plt.subplots(2, 1, figsize=(9, 7))
    axs[0].plot(slots_b1, (mbtw_b1.qx - qx0) * 1e3, '.', label=r'$\Delta q_x$')
    axs[0].plot(slots_b1, (mbtw_b1.qy - qy0) * 1e3, '.', label=r'$\Delta q_y$')
    axs[0].set_xlabel('25 ns slot')
    axs[0].set_ylabel(r'beam-beam tune shift [$10^{-3}$]')
    axs[0].set_title('LHC injection: per-bunch beam-beam tune shift (B1)'
                     + title_suffix)
    axs[0].legend()
    axs[1].plot(slots_b1, dco_x, '.', label='x')
    axs[1].plot(slots_b1, dco_y, '.', label='y')
    axs[1].set_xlabel('25 ns slot')
    axs[1].set_ylabel('orbit dev. from mean at IP1 [$\\mu$m]')
    axs[1].set_title('Per-bunch beam-beam closed-orbit deviation at IP1 (B1)')
    axs[1].legend()
    plt.tight_layout()
    return fig
