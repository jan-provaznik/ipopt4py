#!/usr/bin/env python3
#
# 2021 - 2022 Jan Provaznik (jan@provaznik.pro)
#
# Let's see how poorly this goes.

import setuptools
import sys
import os, os.path

VERSION = '0.0.1'

# Let setuptools fill in the blanks.

ipopt4py = setuptools.Extension(
    name = 'ipopt4py',
    sources = [ 'ipopt4py/ipopt4py.cxx' ],
    libraries = [ 'boost_python3', 'boost_numpy3', 'ipopt' ],
    define_macros = [ ('IPOPT4PY_VERSION', '"{}"'.format(VERSION)) ],
    extra_compile_args = [ '-std=c++17', '-Wextra', '-pthread', '-Wno-sign-compare' ],
    language = 'c++'
)

# Yes, yes, yes!

setuptools.setup(
    name = 'ipopt4py',
    version = VERSION,
    description = 'Basic interface for COIN-OR IPOPT 3.14 library.',
    author = 'Jan Provaznik',
    author_email = 'jan@provaznik.pro',
    url = 'https://provaznik.pro/ipopt4py',
    license = 'LGPL',
    ext_modules = [ ipopt4py ]
)

