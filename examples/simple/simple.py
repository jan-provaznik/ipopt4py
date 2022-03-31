import ipopt4py
import numpy

# Optimization problem definition.

def evalf (isnew, point):
    return numpy.linalg.norm(point - 1, 2)

def gradf (isnew, point):
    value = evalf(isnew, point)
    return (point - 1) / numpy.linalg.norm(point - 1, 2)

def evalg (isnew, point):
    return numpy.array([
        numpy.linalg.norm(point, 2)
    ])
        
def gradg (isnew, point):
    return numpy.array([
        point[0] / numpy.linalg.norm(point, 2),
        point[1] / numpy.linalg.norm(point, 2)
    ])

# Options
options = [
    "print_level 0"
]

# Boundary configuration (region)
xcount = 2
xlimlo = [ 0, 0 ]
xlimhi = [ 5, 5 ]

# Boundary configuration (constraints)
gcount = 1
glimlo = [ 2 ]
glimhi = [ numpy.inf ]

# Initial, starting point
xstart = [ 1, 3 ]

# Bang!
res = ipopt4py.minimize(
    evalf, gradf, 
    evalg, gradg, 
    xstart,
    xcount, xlimlo, xlimhi, 
    gcount, glimlo, glimhi,
    options
)

# Report on the results?

print('IS SUCCESS?', res.success)
print()

print('status code', res.status)
print('status text', res.message)
print()

print('xval', res.xval)
print('fval', res.fval)
print('gval', res.gval)


