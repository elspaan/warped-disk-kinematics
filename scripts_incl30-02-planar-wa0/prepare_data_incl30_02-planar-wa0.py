from discminer.core import Data
import astropy.units as u

file_data = 'image_lines_incl30_02-planar-wa0_mod_noisy_convolved'
dpc = 150.*u.pc
#**********************
#DATACUBE FOR PROTOTYPE
#**********************
print(file_data+'.fits')
datacube = Data(file_data+'.fits', dpc)
datacube.clip(channels={"interval": [20, 170]}, overwrite=True) # can also clip along the velocity axis using e.g. channels={"interval": [15, 115]})
datacube.downsample(2, tag='_2pix') # Downsample cube and add tag to filename

#**********************
#DATACUBE FOR MCMC FIT
#**********************
datacube = Data(file_data+'_clipped.fits', dpc)
datacube.downsample(8, tag='_8pix')
