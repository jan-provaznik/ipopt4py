#!/bin/env python

# 2019 - 2022 Jan Provaznik (jan@provaznik.pro)
#
# COIN-OR IPOPT interface for constrained non-linear optimization.

from ._bridge import (
    minimize
        as _minimize,
    Result
)

def minimize (evalf, evalg, gradf, gradg, xstart, xlimit, glimit, options = [], finite_diff_method = '2-point'):
    '''
    Find local minimum using COIN-OR Interior Point Optimizer IPOPT.

    Parameters
    ----------
    evalf : callable
        Objective function.
    evalg : callable
        Constraints.
    gradf : callable or null
        First derivatives of the objective function.
    gradg : callable or null
        First derivatives of the constraint functions (Jacobi matrix).
    xstart : numpy.ndarray
        Starting point.
    xlimit : pair of iterables
        Limits on the parameters. 
        The 1st iterable should define the lower limits,
        the 2nd iterable should define the upper limits.
    glimit : pair of iterables
        Limits on the constraints. 
        The 1st iterable should define the lower limits,
        the 2nd iterable should define the upper limits.
    options : list of strings
        Options to be passed down to the IPOPT engine.
    finite_diff_method : string
        Either a '2-point' or '3-point' difference scheme.

    Returns
    -------
    Result
        A structure giving the result of the optimization endeavour.
    '''

    if not callable(gradf):
        import numpy
        from scipy.optimize._differentiable_functions import ScalarFunction

        evalf = _newx_wrapper(evalf)
        proxyf = ScalarFunction(evalf,
            xstart, (), finite_diff_method, _dummy_function, 
            None, xlimit
        )

        evalf = _skip_newx_wrapper(proxyf.fun)
        gradf = _skip_newx_wrapper(proxyf.grad)

    if not callable(gradg):
        import numpy
        from scipy.optimize._differentiable_functions import VectorFunction

        evalg = _newx_wrapper(evalg)
        proxyg = VectorFunction(evalg,
            xstart, finite_diff_method, _dummy_function,
            None, None, xlimit, False
        )

        evalg = _skip_newx_wrapper(proxyg.fun)
        gradg = _skip_newx_wrapper(proxyg.jac)

    # Off we go!

    xlimlo, xlimhi = xlimit
    glimlo, glimhi = glimit
    xcount, gcount = len(xlimlo), len(glimlo)

    return _minimize(
        evalf, gradf,
        evalg, gradg,
        xstart,
        xcount, xlimlo, xlimhi,
        gcount, glimlo, glimhi,
        options
    )

def _dummy_function (* args):
    return None

def _newx_wrapper (function):
    def wrapped (point):
        return function(1, point)
    return wrapped

def _skip_newx_wrapper (function):
    def wrapped (skip, point):
        return function(point)
    return wrapped

