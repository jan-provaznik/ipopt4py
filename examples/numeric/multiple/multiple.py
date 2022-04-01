import ipopt4py
import numpy

# Optimization problem definition.
def evalf (isnew, point):
    return numpy.linalg.norm(point - 1, 2)
def evalg (isnew, point):
    return numpy.array([
        numpy.linalg.norm(point, 2),
        3 - point[0] - point[1]
    ])

# Options
options = [
    "print_level 0"
]

# Boundary configuration (region)
xlimlo = [ 0, 0 ]
xlimhi = [ 5, 5 ]
xlimit = (xlimlo, xlimhi)

# Boundary configuration (constraints)
glimlo = [ 0, - numpy.inf ]
glimhi = [ 3, 0 ]
glimit = (glimlo, glimhi)

# Initial, starting point
xstart = [ 1.5, 1.5 ]

# Bang!
res = ipopt4py.minimize(
    evalf, evalg,
    0, 0,
    xstart,
    xlimit, glimit,
    options
)

# Report on the results?

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


