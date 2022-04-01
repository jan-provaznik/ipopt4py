import ipopt4py
import numpy

# KOPACEK (978-80-7378-282-5)
# Chapter 9, problem 27

def evalf (isnew, point):
    return numpy.prod(point)
def evalg (isnew, point):
    return numpy.array([
        numpy.linalg.norm(point, 2),
        numpy.sum(point)
    ])
def gradf (isnew, point):
    return numpy.array([
        point[1] * point[2],
        point[0] * point[2],
        point[0] * point[1]
    ])
def gradg (isnew, point):
    return numpy.array([
        [ 2 * point[0], 2 * point[1], 2 * point[2] ],
        [ 1, 1, 1 ]
    ])

# Initial, starting point
xstart = [ 1, 0, 0 ]

# Boundary configuration (parametric region)
xlimlo = [ -5, -5, -5 ]
xlimhi = [  5,  5,  5 ]
xlimit = (xlimlo, xlimhi)

# Boundary configuration (constraints)
glimlo = [ 1, 0 ]
glimhi = [ 1, 0 ]
glimit = (glimlo, glimhi)

# Options
options = [
    "print_level 0"
]

# Bang!
res = ipopt4py.minimize(
    evalf, evalg,
    gradf, gradg,
    xstart,
    xlimit, glimit,
    options
)

# Report on the results!
print('IS SUCCESS?', res.success)
print()
print('status code', res.status)
print('status text', res.message)
print()
print('iters', res.niter)
print()
print('xval', res.xval)
print('fval', res.fval)
print('gval', res.gval)

