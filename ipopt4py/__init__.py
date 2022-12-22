#!/bin/env python

# 2019 - 2022 Jan Provaznik (jan@provaznik.pro)
#
# COIN-OR IPOPT interface for constrained non-linear optimization.

def minimize (evalf, evalg, gradf, gradg, xstart, xlimit, glimit, options = []):
    '''
    Find local minimum using COIN-OR Interior Point Optimizer IPOPT.

    Parameters
    ----------
    evalf : callable
        Objective function, evalf(isnew, point) where isnew indicates whether
        the point changed between calls to {eval,grad}{f,g} arguments.
    evalg : callable
        Constraint (vector) function, evalg(isnew, point).
    gradf : callable or string or null
        First derivatives of the objective function, gradf(isnew, point).
        If not a callable, the gradient is approximted using a finite
        difference algorithm from SciPy. A string value can be used to indicate
        the differentiation scheme, either '2-point' or '3-point' method. The
        central difference ('3-point') method is used by default.
    gradg : callable or null
        First derivatives of the constraint functions (Jacobi matrix).
        If null, approximated with finite difference algorithms by IPOPT.
        Achieved by setting the jacobian_approximation option.
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

    Returns
    -------
    Result
        A structure giving the result of the optimization endeavour.
    '''

    from ._bridge import Result, minimize as _minimize

    # Handle the gradient of the objective function.

    if not callable(gradf):
        import numpy
        from scipy.optimize._differentiable_functions import ScalarFunction

        if gradf == '2-point' or gradf == '3-point':
            finite_diff_method = gradf
        else:
            finite_diff_method = '3-point'

        # @todo We should do the isnew-based memoization ourselves here.

        evalf = _wrap_evalf(evalf)
        proxy = ScalarFunction(evalf,
            xstart, (), finite_diff_method, _none_function, 
            None, xlimit
        )

        evalf = _wrap_proxy(proxy.fun)
        gradf = _wrap_proxy(proxy.grad)

    if not callable(gradg):
        options = options + [
            'jacobian_approximation finite-difference-values'
        ]

    # Extract limits imposed on the search area and the constraints

    xlimlo, xlimhi = xlimit
    glimlo, glimhi = glimit

    if len(xlimlo) != len(xlimhi):
        raise ValueError('Mismatched lengths of lower and upper limits on search area (xlimit pair).')

    if len(glimlo) != len(glimhi):
        raise ValueError('Mismatched lengths of lower and upper limits on constraints (glimit pair).')

    xcount, gcount = len(xlimlo), len(glimlo)

    # Off we go!

    return _minimize(
        evalf, gradf,
        evalg, gradg,
        xstart,
        xcount, xlimlo, xlimhi,
        gcount, glimlo, glimhi,
        options
    )

def _none_function (* args):
    return None

def _wrap_evalf (evalf):
    def wrapper (point):
        return evalf(1, point)
    return wrapper

def _wrap_proxy (function):
    def wrapper (isnew, point):
        return function(point)
    return wrapper

