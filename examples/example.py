#!/usr/bin/env python

#from mpi4py import MPI
from pathlib import Path
import time
#rank = comm.Get_rank()
import numpy as np
#import pymultinest
import warnings
from astropy.io import fits
from threeML import *


file=load_analysis_results('/home/polpy/polpy_test/polpy/examples/fit_results_161218B.fits')

fig=file.corner_plot()

fig.savefig('corner_plot.png')
