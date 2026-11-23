"""phrport.trf_port -- port of scipy.optimize.least_squares(method='trf') for the
unbounded, linear-loss, dense-callable-Jacobian case used by least_squares_mpi_solve_qss,
with two OPT-IN hooks that scipy does not expose:

  * Delta0      : initial trust radius (scaled space, i.e. ||D^-1 p|| <= Delta with D = x_scale)
                  instead of scipy's ||x0 / x_scale|| (or 1 if that is 0).  Used by
                  PHRConfig.carry_trust_radius.
  * stall_stop  : stop after `stall_stop` CONSECUTIVE rejected trials once the trust radius
                  (as an x-space bound, Delta * max(x_scale)) has collapsed below
                  stall_delta_factor * xtol * (xtol + ||x||).  The current (accepted) x, cost,
                  fun, jac, grad are returned exactly as scipy returns them when max_nfev is
                  exhausted after rejected trials (status 0); result.stall_stopped = True.

With Delta0=None and stall_stop=0 the port is the scipy 1.18 code path verbatim
(least_squares front-end -> VectorFunction -> trf -> trf_no_bounds; the helper functions are
imported from scipy itself), so it is bit-identical to scipy.optimize.least_squares
scipy is NOT modified.

Extra result fields: Delta_final (scaled radius at exit), Delta_hist (radius in force at
each trial), max_accepted_step_h (largest accepted ||step_h||), n_rejected_tail
(consecutive rejected trials at exit), stall_stopped.
"""
# Derived from SciPy (scipy/optimize/_lsq/least_squares.py and trf.py), BSD-3-Clause License,
# Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers. All rights reserved.
# The full license text is in THIRD_PARTY_NOTICES.md.
import numpy as np
from numpy.linalg import norm
from scipy.linalg import svd, qr
from scipy.sparse.linalg import lsmr
from scipy.optimize import OptimizeResult
from scipy.optimize._differentiable_functions import VectorFunction
from scipy.optimize._lsq.least_squares import (
    TERMINATION_MESSAGES, _WrapArgsKwargs, prepare_bounds, check_x_scale, check_tolerance)
from scipy.optimize._lsq.common import (
    solve_lsq_trust_region, solve_trust_region_2d, minimize_quadratic_1d, build_quadratic_1d,
    evaluate_quadratic, right_multiplied_operator, compute_grad, compute_jac_scale,
    check_termination, update_tr_radius, scale_for_robust_loss_function,
    print_header_nonlinear, print_iteration_nonlinear)

__all__ = ["least_squares_trf", "trf_no_bounds_port", "STALL_STOP_MESSAGE"]

STALL_STOP_MESSAGE = ("Stall-stop: {k} consecutive rejected trials with trust radius "
                      "{delta:.3e} < {thr:.3e}; returning the current iterate.")


def least_squares_trf(fun, x0, jac, *, ftol=1e-8, xtol=1e-8, gtol=1e-8, x_scale=None,
                      max_nfev=None, verbose=0, tr_solver=None, tr_options=None,
                      Delta0=None, stall_stop=0, stall_delta_factor=10.0, **unsupported):
    """scipy.optimize.least_squares(fun, x0, jac=jac, method='trf', loss='linear', bounds=None)
    with the opt-in hooks Delta0 / stall_stop (see module docstring)."""
    method = unsupported.pop("method", "trf")
    loss = unsupported.pop("loss", "linear")
    bounds = unsupported.pop("bounds", (-np.inf, np.inf))
    if method != "trf" or loss != "linear" or unsupported:
        raise ValueError(f"trf_port supports method='trf', loss='linear' only "
                         f"(got method={method!r}, loss={loss!r}, extra={sorted(unsupported)})")
    if not callable(jac):
        raise ValueError("trf_port needs a callable jac")
    if verbose not in [0, 1, 2]:
        raise ValueError("`verbose` must be in [0, 1, 2].")
    if max_nfev is not None and max_nfev <= 0:
        raise ValueError("`max_nfev` must be None or positive integer.")
    x0 = np.atleast_1d(x0).astype(float)
    if x0.ndim > 1:
        raise ValueError("`x0` must have at most 1 dimension.")
    lb, ub = prepare_bounds(bounds, x0.shape[0])
    if not (np.all(lb == -np.inf) and np.all(ub == np.inf)):
        raise ValueError("trf_port supports unbounded problems only")
    x_scale = check_x_scale(x_scale, x0, method)
    if isinstance(x_scale, str):
        raise ValueError("trf_port does not support x_scale='jac'")
    ftol, xtol, gtol = check_tolerance(ftol, xtol, gtol, method)
    if tr_options is None:
        tr_options = {}

    # --- least_squares front-end (VectorFunction memoisation exactly as scipy) ---
    fun_wrapped = _WrapArgsKwargs(fun)
    jac_wrapped = _WrapArgsKwargs(jac)

    def _dummy_hess(x, *args):
        return x

    vector_fun = VectorFunction(fun_wrapped, x0, jac_wrapped, _dummy_hess,
                                finite_diff_rel_step=None, finite_diff_jac_sparsity=None,
                                finite_diff_bounds=bounds, workers=None)
    f0 = vector_fun.fun(x0)
    J0 = vector_fun.jac(x0)
    if f0.ndim != 1:
        raise ValueError(f"`fun` must return at most 1-d array_like. f0.shape: {f0.shape}")
    if not np.all(np.isfinite(f0)):
        raise ValueError("Residuals are not finite in the initial point.")
    n, m = x0.size, f0.size
    initial_cost = 0.5 * np.dot(f0, f0)
    if J0.shape != (m, n):
        raise ValueError(f"The return value of `jac` has wrong shape: expected {(m, n)}, "
                         f"actual {J0.shape}.")
    if not isinstance(J0, np.ndarray):
        raise ValueError("trf_port needs a dense Jacobian")
    if tr_solver is None:
        tr_solver = 'exact'

    result = trf_no_bounds_port(
        vector_fun.fun, vector_fun.jac, x0, f0, J0, ftol, xtol, gtol, max_nfev, x_scale,
        None, tr_solver, tr_options.copy(), verbose,
        Delta0=Delta0, stall_stop=stall_stop, stall_delta_factor=stall_delta_factor)

    result.message = TERMINATION_MESSAGES[result.status]
    if result.stall_stopped:
        result.message = STALL_STOP_MESSAGE.format(**result.stall_info)
    result.success = result.status > 0
    if verbose >= 1:
        print(result.message)
        print(f"Function evaluations {result.nfev}, initial cost {initial_cost:.4e}, "
              f"final cost {result.cost:.4e}, "
              f"first-order optimality {result.optimality:.2e}.")
    return result


def trf_no_bounds_port(fun, jac, x0, f0, J0, ftol, xtol, gtol, max_nfev, x_scale,
                       loss_function, tr_solver, tr_options, verbose, callback=None,
                       Delta0=None, stall_stop=0, stall_delta_factor=10.0):
    """scipy.optimize._lsq.trf.trf_no_bounds (scipy 1.18.1) + the Delta0 / stall_stop hooks.
    Lines marked [port] are the only additions."""
    x = x0.copy()

    f = f0
    f_true = f.copy()
    nfev = 1

    J = J0
    njev = 1
    m, n = J.shape

    if loss_function is not None:
        rho = loss_function(f)
        cost = 0.5 * np.sum(rho[0])
        J, f = scale_for_robust_loss_function(J, f, rho)
    else:
        cost = 0.5 * np.dot(f, f)

    g = compute_grad(J, f)

    jac_scale = isinstance(x_scale, str) and x_scale == 'jac'
    if jac_scale:
        scale, scale_inv = compute_jac_scale(J)
    else:
        scale, scale_inv = x_scale, 1 / x_scale

    Delta = norm(x0 * scale_inv)
    if Delta == 0:
        Delta = 1.0
    if Delta0 is not None:                                    # [port] carried radius
        Delta = float(Delta0)
        if not (np.isfinite(Delta) and Delta > 0):
            raise ValueError(f"Delta0 must be finite and > 0, got {Delta0!r}")

    if tr_solver == 'lsmr':
        reg_term = 0
        damp = tr_options.pop('damp', 0.0)
        regularize = tr_options.pop('regularize', True)

    if max_nfev is None:
        max_nfev = x0.size * 100

    alpha = 0.0  # "Levenberg-Marquardt" parameter

    termination_status = None
    iteration = 0
    step_norm = None
    actual_reduction = None

    stalled = False                                           # [port]
    stall_info = {}                                           # [port]
    n_rej = 0                                                 # [port] consecutive rejections
    Delta_hist = []                                           # [port]
    max_acc_step_h = 0.0                                      # [port]
    x_bound_factor = float(np.max(scale)) if np.ndim(scale) else float(scale)   # [port]

    if verbose == 2:
        print_header_nonlinear()

    while True:
        g_norm = norm(g, ord=np.inf)
        if g_norm < gtol:
            termination_status = 1

        if verbose == 2:
            print_iteration_nonlinear(iteration, nfev, cost, actual_reduction,
                                      step_norm, g_norm)

        if termination_status is not None or nfev == max_nfev or stalled:   # [port] stalled
            break

        d = scale
        g_h = d * g

        if tr_solver == 'exact':
            J_h = J * d
            U, s, V = svd(J_h, full_matrices=False)
            V = V.T
            uf = U.T.dot(f)
        elif tr_solver == 'lsmr':
            J_h = right_multiplied_operator(J, d)

            if regularize:
                a, b = build_quadratic_1d(J_h, g_h, -g_h)
                to_tr = Delta / norm(g_h)
                ag_value = minimize_quadratic_1d(a, b, 0, to_tr)[1]
                reg_term = -ag_value / Delta**2

            damp_full = (damp**2 + reg_term)**0.5
            gn_h = lsmr(J_h, f, damp=damp_full, **tr_options)[0]
            S = np.vstack((g_h, gn_h)).T
            S, _ = qr(S, mode='economic')
            JS = J_h.dot(S)
            B_S = np.dot(JS.T, JS)
            g_S = S.T.dot(g_h)

        n_rej = 0                                             # [port]
        actual_reduction = -1
        while actual_reduction <= 0 and nfev < max_nfev:
            if tr_solver == 'exact':
                step_h, alpha, n_iter = solve_lsq_trust_region(
                    n, m, uf, s, V, Delta, initial_alpha=alpha)
            elif tr_solver == 'lsmr':
                p_S, _ = solve_trust_region_2d(B_S, g_S, Delta)
                step_h = S.dot(p_S)

            predicted_reduction = -evaluate_quadratic(J_h, g_h, step_h)
            step = d * step_h
            x_new = x + step
            Delta_hist.append(float(Delta))                   # [port]
            f_new = fun(x_new)
            nfev += 1

            step_h_norm = norm(step_h)

            if not np.all(np.isfinite(f_new)):
                Delta = 0.25 * step_h_norm
                n_rej += 1                                    # [port]
                if _stall(stall_stop, n_rej, Delta, x_bound_factor, stall_delta_factor,
                          xtol, x, stall_info):               # [port]
                    stalled = True
                    break
                continue

            # Usual trust-region step quality estimation.
            if loss_function is not None:
                cost_new = loss_function(f_new, cost_only=True)
            else:
                cost_new = 0.5 * np.dot(f_new, f_new)
            actual_reduction = cost - cost_new

            Delta_new, ratio = update_tr_radius(
                Delta, actual_reduction, predicted_reduction,
                step_h_norm, step_h_norm > 0.95 * Delta)

            step_norm = norm(step)
            termination_status = check_termination(
                actual_reduction, cost, step_norm, norm(x), ratio, ftol, xtol)
            if termination_status is not None:
                break

            alpha *= Delta / Delta_new
            Delta = Delta_new

            if actual_reduction <= 0:                         # [port] rejected trial
                n_rej += 1
                if _stall(stall_stop, n_rej, Delta, x_bound_factor, stall_delta_factor,
                          xtol, x, stall_info):
                    stalled = True
                    break

        if actual_reduction > 0:
            x = x_new
            max_acc_step_h = max(max_acc_step_h, float(step_h_norm))   # [port]

            f = f_new
            f_true = f.copy()

            cost = cost_new

            J = jac(x)
            njev += 1

            if loss_function is not None:
                rho = loss_function(f)
                J, f = scale_for_robust_loss_function(J, f, rho)

            g = compute_grad(J, f)

            if jac_scale:
                scale, scale_inv = compute_jac_scale(J, scale_inv)
        else:
            step_norm = 0
            actual_reduction = 0

        iteration += 1

        if callback is not None:      # least_squares_qss never passes one (kept for fidelity)
            from scipy._lib._util import _call_callback_maybe_halt
            intermediate_result = OptimizeResult(
                x=x, fun=f_true, nit=iteration, nfev=nfev)
            intermediate_result["cost"] = cost

            if _call_callback_maybe_halt(
                callback, intermediate_result
            ):
                termination_status = -2
                break

    if termination_status is None:
        termination_status = 0

    active_mask = np.zeros_like(x)
    return OptimizeResult(
        x=x, cost=cost, fun=f_true, jac=J, grad=g, optimality=g_norm,
        active_mask=active_mask, nfev=nfev, njev=njev,
        status=termination_status,
        # [port] extras
        Delta_final=float(Delta), Delta_hist=Delta_hist, max_accepted_step_h=max_acc_step_h,
        n_rejected_tail=int(n_rej), stall_stopped=bool(stalled), stall_info=stall_info)


def _stall(stall_stop, n_rej, Delta, x_bound_factor, factor, xtol, x, info):
    """[port] stall-stop test after a rejected trial (Delta already updated)."""
    if not stall_stop or n_rej < stall_stop:
        return False
    thr = factor * xtol * (xtol + norm(x))
    delta_x = Delta * x_bound_factor          # bound on the x-space norm of the next step
    if delta_x < thr:
        info.update(k=int(n_rej), delta=float(delta_x), thr=float(thr))
        return True
    return False
