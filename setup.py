#!/usr/bin/env python3

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['lfd_dmp', 'dmpbbo', 'lfd_smoother'],
    package_dir={'lfd_dmp': 'src/lfd_dmp', 
                 'dmpbbo':'src/dmpbbo/dmpbbo',
                 'lfd_smoother':'src/lfd_smoothing/lfd_smoother'}
)

setup(**d)
