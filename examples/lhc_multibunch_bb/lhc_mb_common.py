# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

"""
Shared helpers for the LHC multi-bunch beam-beam examples and the regression
test ``xtrack/tests/test_lhc_multibunch_train.py``.

The examples install 2D beam-beam elements
(``xfields.BeamBeamBiGaussianMultibunch2D``) for head-on and long-range (LR)
encounters at IP1/2/5/8 and find the per-bunch closed solution of the two
multi-bunch beams self-consistently -- either on the full thick lines or on
reduced lines where the arcs between the encounters are second-order Taylor
maps.

All scenario configuration (beam momentum, bunch intensity, emittance, optics
file, IP set, number of LR encounters) lives in an :class:`LHCMultibunchBB`
instance; use the presets ``LHCMultibunchBB.injection()`` /
``LHCMultibunchBB.collision()``. The module holds no mutable state, so it can
be imported normally from several scenarios in one process.

Model (following pytrain / TRAIN):

* Encounter geometry: LR encounter ``n`` sits at ``n * b_h_dist`` from the IP,
  ``b_h_dist = L / n_slots / 2`` (half a 25 ns slot). Beam-1 bunch ``b1`` meets
  beam-2 bunch ``b2 = b1 + offset`` with
  ``offset = round(2 * (s_marker - s_IP1) / slot_len)`` (mod n_slots) -> 0 at
  IP1 and IP5, ~891 at IP2, ~2670 at IP8.
* Coherent (rigid-bunch) kick with the convolved pair size:
  ``sigma_eff = sqrt(eps_n * (beta_b1 + beta_b2) / gamma)``, obtained by the
  elements (``coherent=True``) from each beam's own sizes
  ``sigma = sqrt(eps_n * beta / gamma)``.
* Beam separation = closed-orbit difference (crossing AND separation bumps, from
  the live twiss) PLUS the geometric survey separation of the two rings. The
  latter is ~0 in the common vacuum chamber and rises through the recombination
  region to the 194 mm arc separation, which is what makes the LR encounters
  beyond ~D1 negligible.

The per-bunch macroparticle is labelled by its 25 ns slot through
``zeta = slot * ZETA_PER_SLOT``; the ``zeta_offset`` of each element encodes the
encounter slot offset so ``twiss_multibunch`` pairs the right bunches.
"""

import os
import json
import numpy as np
from scipy.constants import c as clight


import xobjects as xo
import xtrack as xt
import xfields as xf

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', '..', 'test_data', 'lhc_2024')
SCHEME_FILE = os.path.join(DATA, '25ns_2460b_2448_2092_2239_144bpi_20inj.json')


def marker_name(name, mirror):
    return name + ('_b2' if mirror else '_b1')


def wrap_frac_tune(v):
    """Tune difference on the fractional-tune circle, wrapped to (-0.5, 0.5]
    (fast-mode twiss returns fractional tunes while the bare reference may
    carry an integer part)."""
    return (np.asarray(v) + 0.5) % 1.0 - 0.5


class LHCMultibunchBB:
    """One multi-bunch beam-beam scenario: beam parameters, optics file and
    encounter layout, plus all the machinery to build and solve it.

    Use the presets::

        sim = LHCMultibunchBB.injection()   # 450 GeV, BBLR only
        sim = LHCMultibunchBB.collision()   # 6.8 TeV squeezed, head-on + BBLR

    ``ips`` / ``nparasitic`` default from the environment variables
    ``LHC_IPS`` / ``LHC_NPAR`` (falling back to 1,2,5,8 / 45).

    ``context`` selects the CPU context for all trackers and beam-beam
    elements; default from the environment variable ``LHC_OMP`` (unset/``0``
    -> serial; a thread count or ``auto`` -> multi-threaded OpenMP kernels,
    e.g. ``LHC_OMP=auto``). Prebuilt kernels exist for both flavours.
    """

    N_SLOTS = 3564
    ZETA_PER_SLOT = 25e-9 * clight   # per-bunch zeta spacing [m]

    def __init__(self, p0c, bunch_intensity, nemitt, optics_file,
                 ips=None, nparasitic=None, context=None):
        if context is None:
            omp = os.environ.get('LHC_OMP', '0')
            if omp in ('0', '', 'serial'):
                context = xo.ContextCpu()
            else:
                context = xo.ContextCpu(
                    omp_num_threads=('auto' if omp == 'auto' else int(omp)))
        self.context = context
        if ips is None:
            ips = [int(v) for v in
                   os.environ.get('LHC_IPS', '1,2,5,8').split(',')]
        if nparasitic is None:
            nparasitic = int(os.environ.get('LHC_NPAR', '45'))
        self.p0c = p0c
        energy = np.sqrt(p0c**2 + xt.PROTON_MASS_EV**2)
        self.gamma0 = energy / xt.PROTON_MASS_EV
        self.beta0 = p0c / energy
        self.bunch_intensity = bunch_intensity
        self.nemitt = nemitt
        self.optics_file = optics_file
        self.geom = None    # set by compute_geometry
        self.meta = None
        self.ips = list(ips)
        self.nparasitic = nparasitic
        self.enc_names = [n for n, _, _ in self.encounter_specs()]
        self.marker_names_b1 = [marker_name(n, False) for n in self.enc_names]
        self.marker_names_b2 = [marker_name(n, True) for n in self.enc_names]

    @classmethod
    def injection(cls, **kwargs):
        """LHC injection: 450 GeV, 1.8e11 p/bunch, 1.5 um. The nominal
        injection optics are used as-is: the separation bumps are kept ON (as
        at real injection), so the beams DO NOT collide head-on and the
        effect is long-range only (BBLR). The head-on elements are still
        installed but act as "long-range" at the full IP separation."""
        return cls(p0c=450e9, bunch_intensity=1.8e11, nemitt=1.5e-6,
                   optics_file=os.path.join(DATA, 'injection_optics.madx'),
                   **kwargs)

    @classmethod
    def collision(cls, **kwargs):
        """LHC collision: 6.8 TeV, 1.1e11 p/bunch, 2.3 um, fully squeezed
        R2025aRP 15 cm flat optics with end-of-levelling knobs, tunes/chroma
        matched to 62.316/60.322, Q' = 10 (head-on at IP1/5 + BBLR).

        The squeezed optics of acc-models-lhc generate their knobs via MAD-X
        matching, which xsuite cannot execute directly. Instead the FLATTENED
        optics ``collision_optics_15cm_flat_2026.madx`` (the complete numeric
        MAD-X global-variable state after optics + knobs + matching, produced
        by ``test_data/lhc_2024/pytrain/regenerate_collision.py``) is loaded
        on top of the sequence, exactly like ``injection_optics.madx``."""
        return cls(p0c=6800e9, bunch_intensity=1.1e11, nemitt=2.3e-6,
                   optics_file=os.path.join(
                       DATA, 'collision_optics_15cm_flat_2026.madx'),
                   **kwargs)

    # ------------------------------------------------------------------
    # Encounter bookkeeping
    # ------------------------------------------------------------------
    def encounter_specs(self):
        """Yield (name, ip, signed_n); signed_n == 0 is the head-on
        encounter."""
        for ip in self.ips:
            yield f'bb_ip{ip}_ho', ip, 0
            for n in range(1, self.nparasitic + 1):
                yield f'bb_ip{ip}_r{n:02d}', ip, +n
                yield f'bb_ip{ip}_l{n:02d}', ip, -n

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load(self):
        """Load both LHC beams (sequence + this scenario's optics file).
        Returns (env, line_b1, line_b2); the beam-2 line is reversed."""
        env = xt.load(os.path.join(DATA, 'lhc.seq'), format='madx',
                      reverse_lines=['lhcb2'])
        env.lhcb1.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV,
                                              p0c=self.p0c)
        env.lhcb2.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV,
                                              p0c=self.p0c)
        env.vars.load(self.optics_file)
        for ln in (env.lhcb1, env.lhcb2):
            ln.twiss_default['method'] = '4d'
            # no IP at the s=0 boundary (IP1 is at s=0 otherwise)
            ln.cycle(name_first_element='ip3', inplace=True)
            ln.build_tracker(_context=self.context)
        return env, env.lhcb1, env.lhcb2

    def install_markers(self, line, mirror, b_h_dist):
        """Install the beam-beam markers. For the reversed beam-2 line the
        left/right side is mirrored so a given marker name is the same
        physical point in both beams."""
        env = line.env
        places = []
        for name, ip, sn in self.encounter_specs():
            at = (-sn if mirror else sn) * b_h_dist + 1e-6
            places.append(env.place(
                env.new(marker_name(name, mirror), xt.Marker),
                at=at, from_=f'ip{ip}'))
        line.insert(places)

    # ------------------------------------------------------------------
    # Encounter geometry (offset, convolved betas, survey separation)
    # ------------------------------------------------------------------
    @staticmethod
    def _survey_positions(sv, names):
        r = sv.rows[names]
        return np.stack([r.X, r.Y, r.Z], axis=1)   # (n, 3)

    def compute_geometry(self, line_b1, line_b2, b_h_dist, slot_len):
        """Compute per-encounter offset, convolved betas and geometric survey
        separation (twiss + survey of both full beams, ~10 s with prebuilt
        kernels)."""
        print('  twiss + survey of both beams...')
        tw1 = line_b1.twiss()
        tw2 = line_b2.twiss()
        sv1 = line_b1.survey()
        sv2 = line_b2.survey()

        s_ip1 = tw1['s', 'ip1']
        # survey position of each IP for both beams (looked up once per IP)
        ip_pos1 = {ip: np.array([sv1['X', f'ip{ip}'], sv1['Y', f'ip{ip}'],
                                 sv1['Z', f'ip{ip}']]) for ip in self.ips}
        ip_pos2 = {ip: np.array([sv2['X', f'ip{ip}'], sv2['Y', f'ip{ip}'],
                                 sv2['Z', f'ip{ip}']]) for ip in self.ips}
        m1 = self._survey_positions(sv1, self.marker_names_b1)
        m2 = self._survey_positions(sv2, self.marker_names_b2)

        length = line_b1.get_length()
        geom = {}
        for j, (name, ip, sn) in enumerate(self.encounter_specs()):
            n1, n2 = self.marker_names_b1[j], self.marker_names_b2[j]
            s_marker = tw1['s', n1]
            offset = int(round(2 * (s_marker - s_ip1) / slot_len)) % self.N_SLOTS
            # Geometric (survey) separation of the two rings at this
            # encounter, in beam 1's frame. Beam-2 survey is rotated 180 deg
            # about the vertical (X,Z -> -X,-Z), then the SIGNED horizontal
            # separation is obtained as in TRAIN/pytrain: the horizontal-
            # plane distance times a sign from the direction of the
            # separation vector relative to the ring azimuth (2*pi*s/L).
            # This sign flips between IP1/IP5 and IP2/IP8 because the latter
            # are rotated in the global frame -- it must be right for the
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
                betx_b1=float(tw1['betx', n1]),
                bety_b1=float(tw1['bety', n1]),
                betx_b2=float(tw2['betx', n2]),
                bety_b2=float(tw2['bety', n2]),
                sep_x=sep_x, sep_y=sep_y,
            )
        meta = dict(bare_qx_b1=float(tw1.qx), bare_qy_b1=float(tw1.qy),
                    bare_qx_b2=float(tw2.qx), bare_qy_b2=float(tw2.qy))
        self.geom = geom
        self.meta = meta
        return geom, meta

    # ------------------------------------------------------------------
    # Filling scheme
    # ------------------------------------------------------------------
    def windowed_slots(self, scheme_b1, scheme_b2, window):
        """A bounded subset: a reference window plus the windows it collides
        with at every distinct head-on offset (so all IPs get realistic
        pairings)."""
        ho_offsets = sorted({self.geom[f'bb_ip{ip}_ho']['offset']
                             for ip in self.ips})
        # Use the longest contiguous filled run of beam 1 as the reference
        # window (the filling is made of batches separated by gaps, so a
        # window may be shorter than requested). Its LR/IP partners in beam 2
        # may or may not be filled -> realistic PACMAN structure at the
        # window edges.
        filled = scheme_b1 > 0
        best_len = best_start = cur_len = cur_start = 0
        for s in range(self.N_SLOTS):
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
                    cand.add((ref_start + shift + k) % self.N_SLOTS)
        return (sorted(s for s in cand if scheme_b1[s]),
                sorted(s for s in cand if scheme_b2[s]))

    # ------------------------------------------------------------------
    # Beam-beam element installation and self-consistent solve
    # ------------------------------------------------------------------
    def install_bb(self, line, mirror, n_other):
        """Install one beam-beam element per encounter.
        Returns {enc_name: element}."""
        env = line.env
        places = []
        names = []
        for name, ip, sn in self.encounter_specs():
            e = self.geom[name]
            # beam1 pairs b2 = b1 + offset (zeta_offset=+offset);
            # beam2 pairs b1 = b2 - offset (zeta_offset=-offset)
            zoff = (-e['offset'] if mirror else e['offset']) \
                * self.ZETA_PER_SLOT
            # coherent (rigid-bunch) kick: the element convolves this beam's
            # own size with each opposing bunch's size, sigma_eff^2 =
            # sigma_own^2 + sigma_other^2 (statically equivalent to the
            # previous (beta_b1 + beta_b2) * nemitt / gamma0)
            own, oth = ('b2', 'b1') if mirror else ('b1', 'b2')
            bb = xf.BeamBeamBiGaussianMultibunch2D(
                num_bunches=max(n_other, 1),
                zeta_offset=zoff, zeta_match_tol=0.4 * self.ZETA_PER_SLOT,
                # ring is periodic in slots; encounter offsets are stored mod
                # N_SLOTS, so the pairing must wrap around the ring
                zeta_period=self.N_SLOTS * self.ZETA_PER_SLOT,
                other_beam_q0=1.0, other_beam_beta0=self.beta0,
                coherent=True,
                sigma_x=np.sqrt(e[f'betx_{own}'] * self.nemitt / self.gamma0),
                sigma_y=np.sqrt(e[f'bety_{own}'] * self.nemitt / self.gamma0),
                other_beam_sigma_x=np.sqrt(
                    e[f'betx_{oth}'] * self.nemitt / self.gamma0),
                other_beam_sigma_y=np.sqrt(
                    e[f'bety_{oth}'] * self.nemitt / self.gamma0),
                _context=line._context)
            elname = marker_name(name, mirror) + '_bb'
            places.append(env.place(elname, bb, at=marker_name(name, mirror)))
            names.append((name, elname))
        line.insert(places)
        # 'beambeam_scale' knob scaling the strength of all lenses of this
        # line (as in the xfields beam-beam config tools), e.g. for footprints
        # with linear rescale on the beam-beam strength
        line.vars['beambeam_scale'] = 1.0
        for name, elname in names:
            line.element_refs[elname].scale_strength = line.vars['beambeam_scale']
        return {name: line[elname] for name, elname in names}

    def compute_sigmas(self, mbtw, marker_names):
        """Per-encounter transverse sizes of a beam's bunches from their
        LIVE per-bunch beta functions (dynamic beta). Returns (sigma_x,
        sigma_y), each of shape (n_bunches, n_encounters). The convolution
        of the two beams' sizes is done inside the elements
        (``coherent=True``)."""
        eg = self.nemitt / self.gamma0   # same convention as the static case
        # mbtw['betx', names] resolves the marker rows once and slices the
        # numpy columns of all bunch tables (fast multi-element access)
        sigma_x = np.sqrt(mbtw['betx', marker_names] * eg)
        sigma_y = np.sqrt(mbtw['bety', marker_names] * eg)
        return sigma_x, sigma_y

    def update_opposing(self, bb_dict, mbtw_other, slots_other,
                         marker_names_other, sigmas_other=None,
                         sigmas_own=None):
        """Write the opposing beam's per-bunch orbit + geometric survey
        separation into the beam-beam elements, in the frame of the line that
        holds them; optionally also update the sizes (dynamic beta):
        ``sigmas_other`` = the opposing bunches' per-bunch sizes
        (n_other, n_enc), ``sigmas_own`` = this beam's bunch sizes
        (n_own, n_enc) -- the elements hold ONE own size per encounter, so
        the bunch AVERAGE is stored (the bunch-by-bunch resolution enters
        through the opposing beam's per-bunch sizes of the partner
        elements).

        Between the two (opposite-parity) beam lines x flips and y does not.
        Matching TRAIN/pytrain (beam1 sees the opponent at co - sep, beam2 at
        co + sep), and accounting for the beam-2 line x-flip, the survey
        separation enters as ``-sep_x`` in x for BOTH beams."""
        # mbtw['x', names] resolves the marker rows once and slices the
        # numpy columns of all bunch tables (fast multi-element access)
        xs = -mbtw_other['x', marker_names_other]
        ys = mbtw_other['y', marker_names_other]
        zeta_other = np.array(slots_other) * self.ZETA_PER_SLOT
        # NOTE: encounter slot offsets are stored mod N_SLOTS (a left LR at
        # -n is stored as N_SLOTS-n; the IP2/IP8 pairings wrap around the
        # ring for part of the bunches). The elements handle this via their
        # `zeta_period` (set in install_bb), which makes the pairing periodic
        # in the bunch-label axis.
        # one reusable Particles object (same zeta/weight for all encounters)
        p = xt.Particles(p0c=self.p0c, mass0=xt.PROTON_MASS_EV, q0=1.0,
                         x=np.zeros(len(zeta_other)),
                         y=np.zeros(len(zeta_other)),
                         zeta=zeta_other, weight=self.bunch_intensity)
        for j, name in enumerate(self.enc_names):
            p.x[:] = xs[:, j] - self.geom[name]['sep_x']
            p.y[:] = ys[:, j] - self.geom[name]['sep_y']
            kw = {}
            if sigmas_other is not None:
                kw = dict(other_beam_sigma_x=sigmas_other[0][:, j],
                          other_beam_sigma_y=sigmas_other[1][:, j])
            bb = bb_dict[name]
            if sigmas_own is not None:
                bb.sigma_x = sigmas_own[0][:, j].mean()
                bb.sigma_y = sigmas_own[1][:, j].mean()
            bb.update_from_other_beam(p, **kw)

    def solve_self_consistent(self, line_b1, line_b2, bb_b1, bb_b2,
                              slots_b1, slots_b2, n_iter=3, chrom=False,
                              twiss_mode='fast_orbit', show_progress=True,
                              dynamic_beta=False):
        """Iterate twiss_multibunch on both beams, feeding each beam's
        per-bunch closed orbit into the other beam's elements. Returns
        (mbtw_b1, mbtw_b2).

        The iteration only feeds back orbits, so by default the orbit-only
        fast twiss is used (roughly half the cost of the optics-carrying
        default); pass ``twiss_mode='fast'`` to get optics in the returned tables.

        With ``dynamic_beta=True`` the per-bunch effective (convolved) sizes
        of every encounter are recomputed at each iteration from the LIVE
        per-bunch beta functions of both beams (dynamic beta) instead of
        staying at their static values. This requires the optics-carrying
        twiss, so mode='fast' is forced."""
        if dynamic_beta and twiss_mode == 'fast_orbit':
            twiss_mode = 'fast'
        zeta_b1 = np.array(slots_b1) * self.ZETA_PER_SLOT
        zeta_b2 = np.array(slots_b2) * self.ZETA_PER_SLOT
        mbtw_b1 = mbtw_b2 = None
        for it in range(n_iter):
            mbtw_b1 = line_b1.twiss_multibunch(
                zeta_bunches=zeta_b1, chrom=chrom, mode=twiss_mode,
                show_progress=show_progress)
            mbtw_b2 = line_b2.twiss_multibunch(
                zeta_bunches=zeta_b2, chrom=chrom, mode=twiss_mode,
                show_progress=show_progress)
            sizes_b1 = sizes_b2 = None
            if dynamic_beta:
                # per-bunch sizes of each beam at ITS OWN markers: used both
                # as the opposing-beam sizes of the other beam's elements and
                # (bunch-averaged) as the own sizes of this beam's elements
                sizes_b1 = self.compute_sigmas(mbtw_b1,
                                               self.marker_names_b1)
                sizes_b2 = self.compute_sigmas(mbtw_b2,
                                               self.marker_names_b2)
            self.update_opposing(bb_b1, mbtw_b2, slots_b2,
                                  self.marker_names_b2,
                                  sigmas_other=sizes_b2, sigmas_own=sizes_b1)
            self.update_opposing(bb_b2, mbtw_b1, slots_b1,
                                  self.marker_names_b1,
                                  sigmas_other=sizes_b1, sigmas_own=sizes_b2)
            if show_progress:
                print(f'  iteration {it}: '
                      f'B1 qx spread {np.ptp(mbtw_b1.qx):.2e}, '
                      f'B2 qx spread {np.ptp(mbtw_b2.qx):.2e}')
        return mbtw_b1, mbtw_b2


# ----------------------------------------------------------------------------
# Filling scheme (scenario-independent)
# ----------------------------------------------------------------------------
def load_scheme():
    with open(SCHEME_FILE) as fid:
        scheme = json.load(fid)
    return np.array(scheme['schemebeam1']), np.array(scheme['schemebeam2'])


def all_filled_slots(scheme_b1, scheme_b2):
    return (sorted(np.where(scheme_b1 > 0)[0].tolist()),
            sorted(np.where(scheme_b2 > 0)[0].tolist()))


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
    x = mbtw['x', marker] * (-1.0 if reverse else 1.0)
    y = mbtw['y', marker]

    df = pd.DataFrame({
        'slot': np.asarray(slots),
        'qx': mbtw.qx, 'qy': mbtw.qy,
        'dqx': wrap_frac_tune(mbtw.qx - bare_qx),
        'dqy': wrap_frac_tune(mbtw.qy - bare_qy),
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
    co_x = mbtw_b1['x', marker_name('bb_ip1_ho', False)]
    co_y = mbtw_b1['y', marker_name('bb_ip1_ho', False)]
    # per-bunch orbit deviation from the bunch-averaged orbit (removes the common
    # crossing/separation-bump orbit, leaving the bunch-by-bunch beam-beam part)
    dco_x = (co_x - co_x.mean()) * 1e6
    dco_y = (co_y - co_y.mean()) * 1e6
    fig, axs = plt.subplots(2, 1, figsize=(9, 7))
    axs[0].plot(slots_b1, wrap_frac_tune(mbtw_b1.qx - qx0) * 1e3, '.',
                label=r'$\Delta q_x$')
    axs[0].plot(slots_b1, wrap_frac_tune(mbtw_b1.qy - qy0) * 1e3, '.',
                label=r'$\Delta q_y$')
    axs[0].set_xlabel('25 ns slot')
    axs[0].set_ylabel(r'beam-beam tune shift [$10^{-3}$]')
    axs[0].set_title('LHC: per-bunch beam-beam tune shift (B1)'
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


def plot_global_quantities(slots_b1, mbtw_b1, slots_b2, mbtw_b2):
    """Bunch-by-bunch orbit at IP1, beta* at IP1, tunes, chromaticity and
    coupling |C-| of both beams, from mode='fast' MultiBunchTwiss results
    (which carry per-bunch optics and global quantities)."""
    import matplotlib.pyplot as plt
    mk = {False: marker_name('bb_ip1_ho', False),
          True: marker_name('bb_ip1_ho', True)}

    def at_ip1(mbtw, col, mirror):
        return mbtw[col, mk[mirror]]

    fig, axs = plt.subplots(3, 2, figsize=(13, 10), sharex=True)

    ax = axs[0, 0]   # orbit deviation at IP1 (physical frame for both beams)
    for slots, mbtw, mirror, lab in [(slots_b1, mbtw_b1, False, 'B1'),
                                     (slots_b2, mbtw_b2, True, 'B2')]:
        sgn = -1.0 if mirror else 1.0
        x = sgn * at_ip1(mbtw, 'x', mirror)
        y = at_ip1(mbtw, 'y', mirror)
        ax.plot(slots, (x - x.mean()) * 1e6, '.', ms=3, label=f'{lab} x')
        ax.plot(slots, (y - y.mean()) * 1e6, '.', ms=3, label=f'{lab} y')
    ax.set_ylabel(r'orbit dev. at IP1 [$\mu$m]')
    ax.set_title('Per-bunch closed-orbit deviation at IP1')
    ax.legend(ncol=2, fontsize=8)

    ax = axs[0, 1]   # beta* at IP1
    for slots, mbtw, mirror, lab in [(slots_b1, mbtw_b1, False, 'B1'),
                                     (slots_b2, mbtw_b2, True, 'B2')]:
        ax.plot(slots, at_ip1(mbtw, 'betx', mirror), '.', ms=3,
                label=fr'{lab} $\beta_x^*$')
        ax.plot(slots, at_ip1(mbtw, 'bety', mirror), '.', ms=3,
                label=fr'{lab} $\beta_y^*$')
    ax.set_ylabel(r'$\beta^*$ at IP1 [m]')
    ax.set_title('Per-bunch $\\beta^*$ at IP1 (dynamic beta)')
    ax.legend(ncol=2, fontsize=8)

    ax = axs[1, 0]   # fractional tunes
    for slots, mbtw, lab in [(slots_b1, mbtw_b1, 'B1'),
                             (slots_b2, mbtw_b2, 'B2')]:
        ax.plot(slots, mbtw.qx_frac, '.', ms=3, label=f'{lab} $q_x$')
        ax.plot(slots, mbtw.qy_frac, '.', ms=3, label=f'{lab} $q_y$')
    ax.set_ylabel('fractional tune')
    ax.set_title('Per-bunch tunes')
    ax.legend(ncol=2, fontsize=8)

    ax = axs[1, 1]   # chromaticity
    for slots, mbtw, lab in [(slots_b1, mbtw_b1, 'B1'),
                             (slots_b2, mbtw_b2, 'B2')]:
        ax.plot(slots, mbtw.dqx, '.', ms=3, label=f"{lab} $q'_x$")
        ax.plot(slots, mbtw.dqy, '.', ms=3, label=f"{lab} $q'_y$")
    ax.set_ylabel("chromaticity $q'$")
    ax.set_title('Per-bunch chromaticity')
    ax.legend(ncol=2, fontsize=8)

    ax = axs[2, 0]   # coupling
    for slots, mbtw, lab in [(slots_b1, mbtw_b1, 'B1'),
                             (slots_b2, mbtw_b2, 'B2')]:
        ax.plot(slots, mbtw.c_minus, '.', ms=3, label=lab)
    ax.set_xlabel('25 ns slot')
    ax.set_ylabel('$|C^-|$')
    ax.set_title('Per-bunch coupling (closest tune approach)')
    ax.legend(fontsize=8)

    axs[2, 1].axis('off')
    axs[1, 1].set_xlabel('25 ns slot')
    axs[1, 1].tick_params(labelbottom=True)
    plt.suptitle('Per-bunch optics & global quantities '
                 '(mode="fast" multibunch twiss)')
    plt.tight_layout()
    return fig
