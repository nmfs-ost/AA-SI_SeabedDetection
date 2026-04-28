"""
qview_EK-echograms.py

visualize EK echograms
reads netCDF4 format files

jech
"""

import echopype as ep
#from echopype import open_raw
#import echoregions as er
from pathlib import Path
#from sys import exit
#from dask.distributed import Client
import xarray as xr
from matplotlib.pyplot import figure, show, subplots_adjust, get_cmap, cm
#import matplotlib.ticker as mtick
import matplotlib.dates as mdates
#import hvplot.xarray
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import json
#import re
from datetime import datetime, time, timedelta
#import itertools


def convert_to_Sv(ed, wf='CW', encode='power'):
    '''
    apply calibrations and compute Sv
    it seems that for data collected with EK60 GPTs, but with EK80 software, you
    must specify CW and power.
    input: Echopype object
           wf - waveform mode, string, CW=continuous wave
           encode - encode mode, string, power=power
    Output: xarray with Sv data variable
    '''

    print('Converting to Sv with waveform {0} and encoding {1}'.format(wf, encode))
    #Sv = ep.calibrate.compute_Sv(ed, waveform_mode='CW', encode_mode='power')
    Sv = ep.calibrate.compute_Sv(ed, waveform_mode=wf, encode_mode=encode)

    return Sv


def get_frequency_list(Sv):
    '''
    get a list of the available frequencies in the Sv data
    this returns the frequency list in original order, i.e., the order of the
    frequencies in the data file, 
    input: Sv xarray data set or data array with "frequency_nominal" 
           as a dimension or coordinate
    outut: frequency list in original order
           frequency list sorted by low to high frequency
           index list that maps the original order to a sorted order
    '''

    f_Hz = list(Sv.frequency_nominal.values)
    f_Hz_sorted = sorted(f_Hz)
    print('f_Hz: ', f_Hz)
    fidx = [f_Hz.index(i) for i in f_Hz_sorted]

    return(f_Hz, f_Hz_sorted, fidx)


def display_echograms(xrds, nrow=2, ncol=1):
    '''
    Display echograms on the monitor using imshow
    input:  xrds, xarray data set wtih the code variable
            display_variable, variable to display
                              must be 'Sv' or 'PWcode'
    output: None
    ### this is hard coded for now to 2 plots per page
    '''
    
    # get a list of frequencies
    f_Hz, f_Hz_sorted, fidx = get_frequency_list(xrds)
    nf = len(f_Hz)
    f_kHz = [f/1000 for f in f_Hz]

    # default to range
    rv = 'range_sample'
    display_var = 'Sv'

    # range/depth limits
    # the range values are the upper limit of the range/depth bin
    ymin = min(xrds[rv].values)
    ymax = max(xrds[rv].values)
    ylims = [ymin, ymax]

    # time limits
    # xmit time interval
    xmin = min(xrds.ping_time.values)
    xmax = max(xrds.ping_time.values)
    xlims = [xmin, xmax]

    # convert xlimits from time to number for the x-axis
    xlims = mdates.date2num(xlims)

    # display extent
    dextent = [xlims[0], xlims[1], ylims[1], ylims[0]] 

    # display limits
    # default values
    dBmin = -90
    dBmax = -30

    # other parameters
    colormap = 'viridis'
    ptitle = 'Sv Echograms'

    nrow = 2
    ncol = 1
    n2pg, n1pg = divmod(nf, nrow)
    ct = 0
    for i in range(n2pg):
        fig, (ax0, ax1), = plt.subplots(nrows=nrow, ncols=ncol, figsize=(9,9))
        #fig.subplots_adjust(hspace = 0.3)
        cset0 = ax0.imshow(xrds[display_var].sel(frequency_nominal=f_Hz[fidx[ct]]).transpose(), 
                           cmap=colormap, vmin=dBmin, vmax=dBmax, aspect='auto', 
                           interpolation='none', extent=dextent)
        ax0.set_xlabel('Date-Time')
        ax0.set_ylabel('Depth (m)')
        ax0.set_title(str(f_kHz[fidx[ct]])+' kHz')
        ax0.xaxis_date()
        date_format = mdates.DateFormatter('%Y/%m/%d\n%H:%M:%S')
        ax0.xaxis.set_major_formatter(date_format)
        fig.autofmt_xdate()
        cbar = fig.colorbar(cset0, ax=ax0, shrink=0.75)
        cbar.set_label('dB re $m^2 m^{-3}$')
        ct += 1
        cset1 = ax1.imshow(xrds[display_var].sel(frequency_nominal=f_Hz[fidx[ct]]).transpose(), 
                           cmap=colormap, vmin=dBmin, vmax=dBmax, aspect='auto', 
                           interpolation='none', extent=dextent)
        ax1.set_xlabel('Date-Time')
        ax1.set_ylabel('Depth (m)')
        ax1.set_title(str(f_kHz[fidx[ct]])+' kHz')
        ax1.xaxis_date()
        date_format = mdates.DateFormatter('%Y/%m/%d\n%H:%M:%S')
        ax1.xaxis.set_major_formatter(date_format)
        fig.autofmt_xdate()
        cbar = fig.colorbar(cset1, ax=ax1, shrink=0.75)
        cbar.set_label('dB re $m^2 m^{-3}$')
        plt.suptitle(ptitle)
        plt.show(block=False)
        ct += 1
    if (n1pg > 0):
        fig, (ax0, ax1), = plt.subplots(nrows=nrow, ncols=ncol, figsize=(9,9))
        #fig.subplots_adjust(hspace = 0.3)
        cset0 = ax0.imshow(xrds[display_var].sel(frequency_nominal=f_Hz[fidx[ct]]).transpose(), 
                           cmap=colormap, vmin=dBmin, vmax=dBmax, aspect='auto', 
                           interpolation='none', extent=dextent)
        ax0.set_xlabel('Date-Time')
        ax0.set_ylabel('Depth (m)')
        ax0.set_title(str(f_kHz[fidx[ct]])+' kHz')
        ax0.xaxis_date()
        date_format = mdates.DateFormatter('%Y/%m/%d\n%H:%M:%S')
        ax0.xaxis.set_major_formatter(date_format)
        fig.autofmt_xdate()
        cbar = fig.colorbar(cset0, ax=ax0, shrink=0.75)
        cbar.set_label('dB re $m^2 m^{-3}$')
        plt.suptitle(ptitle)
        plt.show(block=False)

# path to the files - this assumes all files are in the same directory
ncpath = Path('/home/user/DATA/netCDF4_Files')
# list of files
ncfiles = ['D20090405-T114914.nc']
# Niki - ncfiles = ['SetteSE2403Bigeye-D20240320-T032338.nc']
# create the echo data object
edlist = []
for k in ncfiles:
    edlist.append(ep.open_converted(str(Path(ncpath / k))))
ed = ep.combine_echodata(edlist)
sonar_model = ed['Sonar'].attrs['sonar_model']
waveform = 'CW'
encode = 'power' 

# generate Sv
Sv = convert_to_Sv(ed, wf=waveform, encode=encode)

# new/old? version of Echopype has beam as a dimension, select beam=1
# assume there is only one beam for now
if [dim for dim in Sv.Sv.dims if dim == 'beam']:
    Sv = Sv.sel(beam='1').drop_vars('beam')

Sv = ep.consolidate.swap_dims_channel_frequency(Sv)

# display echograms

display_echograms(Sv, nrow=2, ncol=1)