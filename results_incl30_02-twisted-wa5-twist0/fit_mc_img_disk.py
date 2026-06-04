from discminer.core import Data
from discminer.disc2d import Model

import numpy as np
import matplotlib.pyplot as plt

from astropy import units as u
import emcee

from argparse import ArgumentParser

parser = ArgumentParser(prog='Handle emcee backend', description='Handle emcee backend')
parser.add_argument('-b', '--backend', default=1, type=int, choices=[0, 1], help="If 0, create new backend. If 1, reuse existing backend")
args = parser.parse_args()

#*********************
#SOME DEFINITIONS
#*********************
foldername = 'Linas_cubes/fits-and-analysis/incl30_02-twisted-wa5-twist0'
diskname = 'disk_name'
file_data = f'../image_lines_incl30_02-twisted-wa5-twist0_mod_noisy_convolved_clipped_downsamp_8pix.fits'
tag_out = 'synthcube_12co_incl30-02-twisted-wa5-twist0_mod' #PREFERRED FORMAT: disc_mol_chan_program_extratags
tag_in = tag_out

nwalkers = 256
nsteps = 7000

dpc = 150.*u.pc
au_to_m = u.au.to('m')

#*********
# GRIDDING
#*********
downsamp_pro = 2 # downsampling for prototype
downsamp_fit = 8 # downsampling for MCMC fit
downsamp_factor = (downsamp_fit/downsamp_pro)**2 # rewuired to correct intensity normalisation for prototype


#*********
#READ DATA
#*********
datacube = Data(file_data, dpc) #Read data and convert to Cube object
vchannels = datacube.vchannels
Rmax = 500 * u.au # Needs to be well-above disk size; estimate from CARTA extent


#****************************
#INIT MODEL AND PRESCRIPTIONS
#****************************
model = Model(datacube, Rmax, Rmin=0, prototype = False) # set inner Rmin to 0, can also be 1 (then it starts from 1 beam size)

# plot definitions, but not needed for the fit
xmax = model.grid['xmax']
xlim = 1.2*xmax/au_to_m
extent = np.array([-xmax, xmax, -xmax, xmax])/au_to_m

hypot_func = lambda x, y: np.srt(x**2 + y**2) # not needed, but takes care of a VSCode error notice

# fyi: we calculate everything in SI, but convert back to au for output. So, input is au, output is au, but calcs are in SI (I think, given the meters)
def intensity_powerlaw_rout(coord, I0=30.0, R0=100*au_to_m, p=-0.4, z0=100*au_to_m, q=0.3, Rout=200):
    """ Standard intensity powerlaw that stops at Rout. 
    """
    if 'R' not in coord.keys(): R = hypot_func(coord['x'], coord['y'])
    else: R = coord['R']
    z = coord['z']
    A = I0*R0**-p*z0**-q # some norm. constant
    Ieff = np.where(R<=Rout*au_to_m, A*R**p*np.abs(z)**q, 0.0) # Effective int. within a radius of Rout
    return Ieff

def intensity_powerlaw_rbreak(coord, I0=30.0, p0=-0.4, p1=-0.4, z0=100, q=0.3, Rbreak=20, Rout=500):
    """ An intensity powerlaw with two components for disks with a sudden drop in intensity near the center (at Rbreak).  
    """
    if 'R' not in coord.keys(): R = hypot_func(coord['x'], coord['y'])
    else: R = coord['R']
    z = coord['z']
    z0*=au_to_m
    Rout*=au_to_m
    Rbreak*=au_to_m
    A = I0*Rbreak**-p0*z0**-q
    B = I0*Rbreak**-p1*z0**-q
    Ieff = np.where(R<=Rbreak, A*R**p0*np.abs(z)**q, B*R**p1*np.abs(z)**q)
    ind = R>Rout
    Ieff[ind] = 0.0
    return Ieff

def z_upper(coord, z0, p, Rb, q, R0=100):
    R = coord['R']/au_to_m
    return au_to_m*(z0*(R/R0)**p*np.exp(-(R/Rb)**q))

def z_lower(coord, z0, p, Rb, q, R0=100):
    R = coord['R']/au_to_m
    return -au_to_m*(z0*(R/R0)**p*np.exp(-(R/Rb)**q))

model.z_upper_func = z_upper
model.z_lower_func = z_lower
model.velocity_func = model.keplerian_vertical # i.e. vrot = sqrt(GM/r**3)*R
model.line_profile = model.line_profile_bell # choice of line profile - use Bell except if you're REALLY certain it's a gaussian.
model.intensity_func = intensity_powerlaw_rout # choice of intensity powerlaw

#If not redefined, intensity and linewidth are powerlaws 
 #of R and z by default, whereas lineslope is constant.
  #See Table 1 of discminer paper 1 (izquierdo+2021).

  
#*************
# PREPARE PARS
#*************
#best_fit_pars = np.loadtxt('./log_pars_%s_cube_150walkers_10000steps.txt'%(tag_in))[1]
#Mstar, vsys, incl, PA, I0, p, q, Rout, L0, pL, qL, Ls, pLs, z0_upper, p_upper, Rb_upper, q_upper, z0_lower, p_lower, Rb_lower, q_lower = best_fit_pars

# velocity
vel_sign = -1 # Rotation direction (Izquierdo et al. 2025); pos. inclination = clockwise, neg. is anti-clockwise
Mstar = 1.0 # M_sol
vsys = 0.0 # moment map, km/s

# Orientation
incl = -0.524 # positive = tilted towards us along the redshifted axis
PA = 0.0 # Joe: offset by about 270 deg; positive = anti-clockwise; wrong sign gives a 90 degree rotation clockwise for some reason. Also divide by 2pi if PA > 2pi
xc = 0.0
yc = 0.0

# intensity
I0 = 0.005 # norm. cte, Jy/pix
p = -1.5 # R-power
q = 0.5 # z-power
# Rbreak = 100. # radius for powerlaw break, default is 100 au
Rout = 400 # extend of the disk, au

# line width
L0 = 0.4 # norm. cte, km/s
pL = -0.4 # R-power
qL = 0.4 # z-power

# line slope
Ls = 2.0 # norm. cte, Jy/pix per km/s (? check this) 
pLs = 0.3 # R-power

# emission surface params
z0_upper, p_upper, Rb_upper, q_upper = 40, 1.0, 240, 3.0 # norm. cte, R-power1 (flare), taper point radius, R-power2 (exp. taper); au
z0_lower, p_lower, Rb_lower, q_lower = 30, 1.0, 240, 3.0 # same definition as upper surface

p0 = [Mstar, vsys,                              #Velocity
      PA, xc, yc,                         #Orientation
      I0, p, q, Rout,                           #Intensity
      L0, pL, qL,                               #Line width
      Ls, pLs,                                  #Line slope
      z0_upper, p_upper, Rb_upper, q_upper,     #Upper surface height
      z0_lower, p_lower, Rb_lower, q_lower      #Lower surface height
]


#****************************
#SET FREE PARS AND BOUNDARIES
#****************************
# If True, parameter is allowed to vary freely.
#  If float, parameter is fixed to the value provided.

model.mc_params['velocity']['vel_sign'] = vel_sign 
model.mc_params['velocity']['Mstar'] = True 
model.mc_params['velocity']['vsys'] = True 
model.mc_params['orientation']['incl'] = incl
model.mc_params['orientation']['PA'] = True
model.mc_params['orientation']['xc'] = True
model.mc_params['orientation']['yc'] = True
model.mc_params['intensity'] = {'I0': True, 'p': True, 'q': True, 'Rout': True}
model.mc_params['linewidth']['L0'] = True
model.mc_params['linewidth']['p'] = True
model.mc_params['linewidth']['q'] = True
model.mc_params['lineslope']['Ls'] = True
model.mc_params['lineslope']['p'] = True
model.mc_params['height_upper'] = {'z0': True, 'p': True, 'Rb': True, 'q': True}
model.mc_params['height_lower'] = {'z0': True, 'p': True, 'Rb': True, 'q': True}
                                   
# Boundaries of user-defined attributes must be defined here.
# Boundaries of attributes existing in discminer can be modified here, otherwise default values are taken.

model.mc_boundaries['velocity']['vsys'] = (-5, 5)
model.mc_boundaries['orientation']['incl'] = (-1.2, 1.2)
model.mc_boundaries['intensity']['I0'] = (0, 0.05)
model.mc_boundaries['intensity']['Rout'] = (0, 550)
model.mc_boundaries['intensity']['p'] = (-5.0, 5.0)
model.mc_boundaries['intensity']['q'] = (0, 5.0)
model.mc_boundaries['linewidth']['L0'] = (0.005, 5.0)
model.mc_boundaries['linewidth']['p'] = (-5.0, 5.0)
model.mc_boundaries['linewidth']['q'] = (-5.0, 5.0)
model.mc_boundaries['lineslope']['Ls'] = (0.005, 20)
model.mc_boundaries['lineslope']['p'] = (-5.0, 5.0)
model.mc_boundaries['lineslope']['q'] = (-5.0, 5.0)
model.mc_boundaries['height_upper']['z0'] = (0, 200)
model.mc_boundaries['height_upper']['p'] = (0, 5)
model.mc_boundaries['height_upper']['Rb'] = (0, 550)
model.mc_boundaries['height_upper']['q'] = (0, 15)
model.mc_boundaries['height_lower']['z0'] = (0, 200)
model.mc_boundaries['height_lower']['p'] = (0, 5)
model.mc_boundaries['height_lower']['Rb'] = (0, 550)
model.mc_boundaries['height_lower']['q'] = (0, 15)

"""
#****************************
# SIDE-BY-SIDE PLOT - PRE-FITDiscminer
# comment RUN MCMC if doing this
#****************************

# plot side-by-side
model.params['velocity']['vel_sign'] = vel_sign 
model.params['velocity']['Mstar']    = Mstar
model.params['velocity']['vsys']     = vsys

model.params['orientation']['incl'] = incl
model.params['orientation']['PA']   = PA
model.params['orientation']['xc']   = xc
model.params['orientation']['yc']   = yc

model.params['intensity'] = {'I0': I0, 'p': p, 'q': q, 'Rout': Rout}

model.params['linewidth']['L0'] = L0
model.params['linewidth']['p']  = pL
model.params['linewidth']['q']  = qL

model.params['lineslope']['Ls'] = Ls
model.params['lineslope']['p']  = pLs

model.params['height_upper'] = {'z0': z0_upper, 'p': p_upper, 'Rb': Rb_upper, 'q': q_upper}
model.params['height_lower'] = {'z0': z0_lower, 'p': p_lower, 'Rb': Rb_lower, 'q': q_lower}

#**************************
#MAKE MODEL (2D ATTRIBUTES)
#**************************
modelcube = model.make_model(make_convolve=False)
modelcube.filename = 'cube_model_%s.fits'%tag_out
if model.beam_info is not None:
    modelcube.convert_to_tb(writefits=True)

datacube.filename = 'cube_data_%s.fits'%tag_out
if model.beam_info is not None:
    datacube.convert_to_tb(writefits=True)

#**********************
#VISUALISE CHANNEL MAPS
#**********************
# modelcube.show(compare_cubes=[datacube], extent=extent, int_unit='Intensity [K]', show_beam=True, surface_from=model)
modelcube.show_side_by_side(datacube, int_unit='Intensity [K]', show_beam=True,  surface_from=model) 
"""


#********
#RUN MCMC
#********
# Set up the emcee backend
# saves the stuff here
if __name__ == '__main__': # this fixes a weird bug. Indent is important.
    filename = "backend_%s.h5"%tag_out
    backend = None

    #try and except statement failing with FileNotFoundError/OSError
    if args.backend:
        #Succesive runs
        backend = emcee.backends.HDFBackend(filename)
    else:
        #First run: Initialise backend
        backend = emcee.backends.HDFBackend(filename)
        backend.reset(nwalkers, len(p0))

    print("Backend Initial size: {0} steps".format(backend.iteration))
        
        # Noise in each pixel is stddev of intensity from first and last 5 channels 
    noise = np.std( np.append(datacube.data[:5,:,:], datacube.data[-5:,:,:], axis=0), axis=0) 
        
    #Run Emcee - the actual fitting
    model.run_mcmc(datacube.data, vchannels,
           p0_mean=p0, nwalkers=nwalkers, nsteps=nsteps,
           backend=backend,
           tag=tag_out,
           nthreads=86, # If not specified considers maximum possible number of cores
           frac_stats=0.1, # analogous to the burnin time, chooses which part to take for the corner plot
           frac_stddev=1e-3, # estimates stdev 
           noise_stddev=noise) 
    print("Backend Final size: {0} steps".format(backend.iteration))


