import numpy as np
import astropy as ap
import scipy as sp

from astropy.io import fits

#### creating, updating and changing fits files with astropy

# reading in the file
folder = 'synth_cubes/' # folder I store the cubes in
new_folder = 'synth_cubes_mod/'
file_name = 'image_lines_incl30_02-planar-wa0.fits' # original fits
file_name_mod = 'image_lines_incl30_02-planar-wa0_mod.fits' # modified fits

# fixes the HZ vs Hz unit and missing RESTFRQ keyword 
with fits.open(folder+file_name) as hdul:
    hdul.info()
    hdr = hdul[0].header

    # changed
    hdr['BUNIT'] = 'Jy/beam' # because we're convolving
    hdr['CUNIT3'] = 'Hz' 

    # 17/03/2026: should work now?

    # added
    # hdr['EXTEND'] = 'T' --> don't do this maually (according to astropy doc.) If you are certain there's no extensions, leave it out. 
    hdr['RESTFRQ'] = (2.305380000000E+11, '[Hz] Line rest frequency') # copied from HD163 file
    
    # beam info
    hdr['BZERO'] = 0.000000000000E+00
    hdr['BSCALE'] = 1.000000000000E+00
    hdr['BMAJ'] = 4.166666666667E-05
    hdr['BMIN'] = 4.166666666667E-05
    hdr['BPA'] = 0.000000000000E+00
 
    print(hdr['CUNIT3'])
    print(hdr['RESTFRQ'])
    hdul.writeto(new_folder+file_name_mod, overwrite=True)

# test test - DO THE VERIFY, V. IMPORTANTO, or Discminer will throw a fit (pun very much)
with fits.open(new_folder+file_name_mod) as hdul:
    hdul.verify('fix')
    hdr = hdul[0].header
    print(hdr['RESTFRQ'])