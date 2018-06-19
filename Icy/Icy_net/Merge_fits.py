#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 13:53:27 2018

@author: bordieremma
"""

#Create a file with the 3 fits files  (SAGE=SAGESpec_OAGB+SAGESpec_CAGB+SAGESpec_RSG)

import matplotlib.pyplot as plt
import astropy
import numpy as np
from astropy.io import fits
from astropy.table import Table

direc='/Users/bordieremma/Documents/Magistere_2/STAGE_TAIPEI/CODES/FITS/SAGE/'

t1 = fits.open(direc+'SAGESpec_OAGB.fits')
hdu=t1[1]
header=hdu.header
data=hdu.data
SAGE_name_1=data['SAGE_SPEC_CLASS']
SAGE_wvl_1=data['WAVE']
SAGE_flux_1=data['FLUX']
SAGE_dflux_1=data['DFLUX']
SAGE_subtype_1=data['SUBTYPE']


t2 = fits.open(direc+'SAGESpec_CAGB.fits')
hdu2=t2[1]
header=hdu2.header
data2=hdu2.data
SAGE_name_2=data2['SAGE_SPEC_CLASS']
SAGE_wvl_2=data2['WAVE']
SAGE_flux_2=data2['FLUX']
SAGE_dflux_2=data2['DFLUX']
SAGE_subtype_2=data2['SUBTYPE']


t3 = fits.open(direc+'SAGESpec_RSG.fits')
hdu3=t3[1]
header=hdu3.header
data3=hdu3.data
SAGE_name_3=data3['SAGE_SPEC_CLASS']
SAGE_wvl_3=data3['WAVE']
SAGE_flux_3=data3['FLUX']
SAGE_dflux_3=data3['DFLUX']
SAGE_subtype_3=data3['SUBTYPE']

#new = hstack([t1, t2,t3])
#new.write('SAGE.fits')


SAGE_wvl=[]
SAGE_wvl.extend(SAGE_wvl_1)
SAGE_wvl.extend(SAGE_wvl_2)
SAGE_wvl.extend(SAGE_wvl_3)

SAGE_name=[]
SAGE_name.extend(SAGE_name_1)
SAGE_name.extend(SAGE_name_2)
SAGE_name.extend(SAGE_name_3)

SAGE_flux=[]
SAGE_flux.extend(SAGE_flux_1)
SAGE_flux.extend(SAGE_flux_2)
SAGE_flux.extend(SAGE_flux_3)

SAGE_dflux=[]
SAGE_dflux.extend(SAGE_dflux_1)
SAGE_dflux.extend(SAGE_dflux_2)
SAGE_dflux.extend(SAGE_dflux_3)

SAGE_subtype=[]
SAGE_subtype.extend(SAGE_subtype_1)
SAGE_subtype.extend(SAGE_subtype_2)
SAGE_subtype.extend(SAGE_subtype_3)

c1=fits.Column(name='Name', array=np.array(SAGE_name),format='23A')
c2=fits.Column(name='Wave', array=np.array(SAGE_wvl), format='365D')
c3=fits.Column(name='Flux', array=np.array(SAGE_flux), format='365D')
c4=fits.Column(name='Dflux', array=np.array(SAGE_dflux),  format='365D')
c5=fits.Column(name='Subtype', array=np.array(SAGE_subtype),format='4A')


t=fits.BinTableHDU.from_columns([c1,c2, c3,c4,c5])
t.writeto('SAGE2.fits')


direc1='/Users/bordieremma/Documents/Magistere_2/STAGE_TAIPEI/CODES/ICY/'

SAGE2 = fits.open(direc1+'SAGE2.fits')
hdu=SAGE2[1]
header=hdu.header
data=hdu.data
print(header)

SAGE_Name=data['Name']
SAGE_Wvl=data['Wave']
SAGE_Flux_1=data['Flux']
SAGE_Dflux_1=data['Dflux']
SAGE_Subtype_1=data['Subtype']



#t=Table(np.asarray(SAGE_name), names='Name')
#t.write('Name.fits', format='fits')

'''
SAGE=((SAGE_name),(SAGE_wvl),(SAGE_flux),(SAGE_dflux),(SAGE_subtype))
SAGE1=open('SAGE1.fits',"w")
SAGE1.write(SAGE)
#c1=fits.Column(name='Name', array=np.array(SAGE_name),format='K')
#c5=fits.Column(name='Subtype', array=np.array(SAGE_subtype),format='K')
SAGE1.close()'''



               
