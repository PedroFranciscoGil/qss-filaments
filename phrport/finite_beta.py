"""Finite-beta coupling of the coil subproblem to the equilibrium (opt-in, PHRConfig.finite_beta).

The coils must cancel the plasma's own normal field on the LCFS: the coil residual is
(B_coil + B_plasma).n / |B_coil + B_plasma| with B_plasma from virtual casing on the VMEX state of the
boundary (phrport.plasma_field).  This module holds the CONTEXT (the VMEX state whose plasma field the
coil solver currently sees) and the FIELD TERM of the envelope gradient,

    d J*/ds |_field = < dJ/dB_plasma , dB_plasma/ds >
                    = sum_j e_j [ <v_state, dx_j> + <v_params, dp_j> ],

with (v_state, v_params) one reverse pass of B_plasma(state, params) contracted with the coil
cotangent dJ/dB_plasma, and (dx_j, dp_j) the state / parameter tangents of the free boundary DOFs that
the exact Jacobian already solves for (gauge-continued map, the map our Taylor tests validate).

Schedule (phr.py): the context is the state of the last Jacobian point (an accepted iterate) and is
held fixed through the trial evaluations of a least-squares subproblem; at every Jacobian evaluation
it moves to the new point (exact plasma field there), the coils are re-solved warm and the field term
is added to the coil row.  Front points / final coil stages always use their own exact plasma field.

Current consistency (opt-in, profile phr.fb_current_from_G): the coil problem fixes the total coil current
T (coils.total_current), while B_plasma is absolute (VMEX field at the deck's phiedge).  Coils that hold the
equilibrium carry the linked current 2 pi |G_edge| / mu0 (Ampere on a toroidal loop outside the plasma), i.e.
lam_G = pi |G_edge| / (mu0 nfp T) times T.  B.n/|B| being invariant under a common scaling, the coil problem
then sees B_plasma / lam_G; lam_G(state, params) is differentiated in the field term.

jit (PHRConfig.fb_jit, default "vjp"): the context grid, the provider interpolation and the field term (one
reverse pass + the tangent contraction) run as jitted functions.  They take every array as an argument and close
only over static objects (VMEX input, implicit config, precision plan), so they compile once per continuation step.
"off" is the eager path; "check" uses the eager results and writes the jitted ones' differences to
fb_jit_check.jsonl.  If compiling or running a jitted piece raises (compile error, XLA / CUDA RESOURCE_EXHAUSTED),
one warning is logged and that step falls back to the eager path; the next step tries jit again.
"""
from __future__ import annotations

import time

import numpy as np

import jax
import jax.numpy as jnp

from . import equilibrium as E
from .plasma_field import PlasmaField


def g_edge(state, rt):
    """VMEC rbtor = 1.5 bvco(ns) - 0.5 bvco(ns-1) (edge R B_tor, T m) of a VMEX state (traceable)."""
    from vmex.core.statephysics import _field_chain
    from vmex.core.fields import surface_currents
    _, _, _, f, _ = _field_chain(state, rt)
    return surface_currents(bsubu=f.bsubu, bsubv=f.bsubv, trig=rt.trig, s=jnp.asarray(rt.setup.s_full),
                            signgs=rt.setup.signgs).rbtor


def current_scale(state, rt, nfp, total_current):
    """lam_G: required / configured coil current (see the module docstring)."""
    return jnp.pi * jnp.abs(g_edge(state, rt)) / (4e-7 * jnp.pi * nfp * total_current)


def _tdot(a, b):
    return sum(jnp.vdot(x, y) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)))


class FiniteBetaCoupler:
    def __init__(self, nfp, *, n=32, digits_value=6, digits_grad=4, log=None, current_from_G=None,
                 total_current=None, jit="vjp", check_path=None):
        import os
        # virtual-casing source chunk: PHRPORT_FB_CHUNK (default 1024; use 128 on GPUs / hosts with <= 16 GB)
        chunk = int(os.environ.get("PHRPORT_FB_CHUNK", "1024"))
        self.nfp = int(nfp)
        self.current_from_G = bool(E._PROB.get("phr", "fb_current_from_G", False) if current_from_G is None
                                   else current_from_G)
        self.total_current = float(E._PROB.get("coils", "total_current", 1.25e6) if total_current is None
                                   else total_current)
        self.lam = 1.0
        self.pf_val = PlasmaField(nfp, n, digits_value, chunk_size=chunk)
        self.pf_grad = PlasmaField(nfp, n, digits_grad, chunk_size=chunk)
        self.log = log or (lambda *a, **k: None)
        self.key = None
        self._grid = None
        if jit not in ("off", "vjp", "check"):
            raise ValueError(f"fb jit mode {jit!r}")
        self.jit = jit
        self.check_path = check_path
        self._jfn = {}              # (kind, id(inp), id(imp_cfg), id(plan)) -> (inp, imp_cfg, plan, jitted fn)
        self._interp_j = None
        self._jit_failed = set()    # (id(inp), id(imp_cfg)) of the steps that fell back to eager
        self._ctx_mode = None       # that key for the current context (provider fallback)
        self.last_check_g = None    # check mode: the jitted field term of the last call
        self.stats = dict(contexts=0, t_context=0.0, field_grads=0, t_field_grad=0.0, jit_traces={},
                          jit_fallbacks=0)

    # ---- jitted pieces (fb_jit != "off")
    def _traced(self, kind):
        self.stats["jit_traces"][kind] = self.stats["jit_traces"].get(kind, 0) + 1

    def _jit_on(self, mode):
        return self.jit != "off" and mode not in self._jit_failed

    def _jit_fail(self, what, mode, e):
        """Jitted `what` raised: eager path for the rest of this step (mode = (id(inp), id(imp_cfg)))."""
        self._jit_failed.add(mode)
        self._jfn = {k: v for k, v in self._jfn.items() if k[1:3] != mode}
        self._interp_j = None
        self.stats["jit_fallbacks"] += 1
        msg = " ".join(f"{type(e).__name__}: {e}".split())[:300]
        self.log(f"  [WARNING] finite-beta jit {what} failed ({msg}); eager virtual casing for the rest of this mode")
        self._check_write(dict(kind="fallback", what=what, error=msg))

    def _jitted(self, kind, inp, imp_cfg):
        from vmex.core import implicit as imp
        pf = self.pf_val if kind == "ctx" else self.pf_grad
        k = (kind, id(inp), id(imp_cfg), id(pf.plan))
        hit = self._jfn.get(k)
        if hit is not None:
            return hit[3]
        cur, nfp, T = self.current_from_G, self.nfp, self.total_current
        if kind == "ctx":
            def fn(state, params):
                self._traced("ctx")
                rt = imp.runtime_from_params(params, imp_cfg)
                grid = pf.grid_cyl(pf.surface_data(inp, state, rt))
                return grid, (current_scale(state, rt, nfp, T) if cur else jnp.ones(()))
        else:
            def fn(state, params, cot, TH, PH, dxs, tps):
                self._traced("vjp")

                def Fb(state, params):
                    rt = imp.runtime_from_params(params, imp_cfg)
                    v = jnp.sum(cot * pf.at(inp, state, TH, PH, rt))
                    return v / current_scale(state, rt, nfp, T) if cur else v
                _, vjp = jax.vjp(Fb, state, params)
                vs, vp = vjp(jnp.ones(()))
                return jax.vmap(lambda dx, tp: _tdot(vs, dx) + _tdot(vp, tp))(dxs, tps)
        f = jax.jit(fn)
        self._jfn[k] = (inp, imp_cfg, pf.plan, f)
        return f

    def _check_write(self, rec):
        if self.check_path:
            import json
            with open(self.check_path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")

    # ---- context: the state whose plasma field the coil solver sees
    def set_context(self, inp, imp_cfg, state, params, key):
        if key is not None and key == self.key:
            return
        from vmex.core import implicit as imp
        t0 = time.time()
        mode = (id(inp), id(imp_cfg))
        self._ctx_mode = mode
        with E.vmex_device_ctx():
            if self.pf_val.plan is None:
                self.pf_val.make_plan(self.pf_val.surface_data(inp, state, imp.runtime_from_params(params, imp_cfg)))
            grid = gj = None
            lam = lj = 1.0
            if self._jit_on(mode):
                try:
                    gj, lj = self._jitted("ctx", inp, imp_cfg)(state, params)
                    gj, lj = np.asarray(jax.device_get(gj)), float(lj)
                except Exception as e:  # noqa: BLE001
                    self._jit_fail("context", mode, e)
                    gj = None
            t1 = time.time()
            if gj is None or self.jit == "check":
                rt = imp.runtime_from_params(params, imp_cfg)
                sd = self.pf_val.surface_data(inp, state, rt)
                grid = np.asarray(jax.device_get(self.pf_val.grid_cyl(sd)))
                lam = float(current_scale(state, rt, self.nfp, self.total_current)) if self.current_from_G else 1.0
            if self.jit == "check" and gj is not None:
                self._check_write(dict(kind="context", n=self.stats["contexts"], t_jit=t1 - t0,
                                       t_eager=time.time() - t1,
                                       rel_grid=float(np.max(np.abs(gj - grid)) / max(np.max(np.abs(grid)), 1e-300)),
                                       bitwise=bool(np.array_equal(gj, grid)), traces=dict(self.stats["jit_traces"])))
            self._grid = gj if grid is None else grid
            self.lam = lj if grid is None else lam
        self.key = key
        dt = (time.time() - t1) if (self.jit == "check" and gj is not None) else time.time() - t0
        self.stats["contexts"] += 1
        self.stats["t_context"] += dt
        self.log(f"  [finite-beta] plasma field context updated (virtual casing, {self.pf_val.digits} digits"
                 + (f"; current scale lam_G = {self.lam:.6f}" if self.current_from_G else "")
                 + ("" if grid is not None and self.jit != "check" else f"; jit {self.jit}") + f") [{dt:.1f}s]")

    def provider(self, boundary, TH, PH):
        """CoilSolver.set_plasma provider: Cartesian B_plasma at boundary angles (TH, PH)."""
        if self._grid is None:
            raise RuntimeError("finite-beta: no plasma-field context set")
        if self.jit == "vjp" and self._jit_on(self._ctx_mode):
            pf = self.pf_val
            try:
                if self._interp_j is None:
                    def interp(grid, TH, PH):
                        self._traced("interp")
                        return pf.interp_cart(grid, TH, PH)
                    self._interp_j = jax.jit(interp)
                return np.asarray(self._interp_j(jnp.asarray(self._grid), jnp.asarray(TH), jnp.asarray(PH))) / self.lam
            except Exception as e:  # noqa: BLE001
                self._jit_fail("provider interpolation", self._ctx_mode, e)
        return np.asarray(self.pf_val.interp_cart(jnp.asarray(self._grid), TH, PH)) / self.lam

    # ---- field term of the envelope gradient
    def field_grad(self, inp, imp_cfg, ex, cot, TH, PH):
        """d c0/d(free DOFs) through B_plasma, at the exact-Jacobian point ex (needs its state tangents)."""
        from vmex.core import implicit as imp
        if getattr(ex, "state_tangents", None) is None:
            raise ValueError("finite-beta field term needs the exact Jacobian's state tangents")
        t0 = time.time()
        cot = jnp.asarray(cot)
        TH, PH = jnp.asarray(TH), jnp.asarray(PH)
        with E.vmex_device_ctx():
            if self.pf_grad.plan is None:
                self.pf_grad.make_plan(self.pf_grad.surface_data(inp, ex.state,
                                                                 imp.runtime_from_params(ex.params, imp_cfg)))

            mode = (id(inp), id(imp_cfg))
            g = gj = None
            if self._jit_on(mode):
                try:
                    gj = self._jitted("vjp", inp, imp_cfg)(ex.state, ex.params, cot, TH, PH,
                                                           ex.state_tangents, ex.param_tangents)
                    gj = np.asarray(jax.device_get(gj), dtype=np.float64)
                except Exception as e:  # noqa: BLE001
                    self._jit_fail("field term", mode, e)
                    gj = None
            t1 = time.time()
            if gj is None or self.jit == "check":
                def Fb(state, params):
                    rt = imp.runtime_from_params(params, imp_cfg)
                    v = jnp.sum(cot * self.pf_grad.at(inp, state, TH, PH, rt))
                    if self.current_from_G:
                        v = v / current_scale(state, rt, self.nfp, self.total_current)
                    return v
                _, vjp = jax.vjp(Fb, ex.state, ex.params)
                vs, vp = vjp(jnp.ones(()))
                g = jax.vmap(lambda dx, tp: _tdot(vs, dx) + _tdot(vp, tp))(ex.state_tangents, ex.param_tangents)
                g = np.asarray(jax.device_get(g), dtype=np.float64)
            self.last_check_g = None
            if self.jit == "check" and gj is not None:
                ng = max(float(np.linalg.norm(g)), 1e-300)
                self._check_write(dict(kind="field", n=self.stats["field_grads"], ndof=int(g.size), t_jit=t1 - t0,
                                       t_eager=time.time() - t1, rel_vjp=float(np.linalg.norm(gj - g) / ng),
                                       maxrel_vjp=float(np.max(np.abs(gj - g)) / max(float(np.max(np.abs(g))), 1e-300)),
                                       bitwise_vjp=bool(np.array_equal(gj, g)), traces=dict(self.stats["jit_traces"])))
                self.last_check_g = gj
            jit_used = g is None
            g = gj if g is None else g
        dt = (time.time() - t1) if (self.jit == "check" and gj is not None) else time.time() - t0
        self.stats["field_grads"] += 1
        self.stats["t_field_grad"] += dt
        self.log(f"  [finite-beta] field term of the envelope gradient: |g_field|={np.linalg.norm(g):.3e} "
                 f"[{dt:.1f}s" + (", jit vjp" if jit_used else (", jit check" if self.jit == "check" else "")) + "]")
        return g


def standalone_coupler(boundary, step, *, n=32, digits_value=6, digits_grad=4, log=None, jit="vjp"):
    """A coupler whose context is the cold-solved state of `boundary` (two-stage / post-processing)."""
    from vmex.core import implicit as imp
    from .exact_jacobian import step_context
    eq = E.evaluate(boundary, step, return_state=True)
    if getattr(eq, "vmex_state", None) is None:
        raise RuntimeError(f"no converged state for the plasma field: {eq.message}")
    cp = FiniteBetaCoupler(E.NFP, n=n, digits_value=digits_value, digits_grad=digits_grad, log=log, jit=jit)
    ctx = step_context(step)
    icfg = imp.make_config(ctx.inp, multigrid=True)
    with E.vmex_device_ctx():
        state = eq.vmex_state.spectral_state()
        params = imp.params_from_input(E.vmec_input(boundary, step))
    cp.set_context(ctx.inp, icfg, state, params, ("standalone", step))
    return cp
