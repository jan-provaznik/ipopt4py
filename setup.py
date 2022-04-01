#!/usr/bin/env python3
#
# 2021 - 2022 Jan Provaznik (jan@provaznik.pro)
#
# Let's see how poorly this goes.

import setuptools

VERSION = '0.1.1'

# Let setuptools fill in the blanks.

bridge = setuptools.Extension(
    name = 'ipopt4py._bridge',
    sources = [ 'ipopt4py/_bridge/bridge.cxx' ],
    libraries = [ 'boost_python3', 'boost_numpy3', 'ipopt' ],
    define_macros = [ ('IPOPT4PY_VERSION', '"{}"'.format(VERSION)) ],
    extra_compile_args = [ '-std=c++17', '-Wextra', '-pthread', '-Wno-sign-compare' ],
    language = 'c++'
)

# Yes, yes, yes!

setuptools.setup(
    name = 'ipopt4py',
    version = VERSION,
    description = 'Basic interface for the COIN-OR IPOPT 3.14 library.',
    author = 'Jan Provaznik',
    author_email = 'jan@provaznik.pro',
    url = 'https://provaznik.pro/ipopt4py',
    license = 'LGPL',

    install_requires = [
        'numpy >= 1.22',
        'scipy >= 1.8.0'
    ],
    ext_modules = [ bridge ],
    packages = [ 'ipopt4py' ]
)

