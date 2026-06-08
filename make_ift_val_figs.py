import flow
import numpy as np
import pandas as pd
import os
import sys
import subprocess
import shutil
import matplotlib.pyplot as plt
from flow import FlowProject, directives
from pymser import pymser
import warnings
import panedr
from block_average.block_average import block_average
import glob

###Make IFT validation figures showing interfaces

def calc_mass_dens(density, mode = "liq"):
    #Find the region attributed to liquid density
    mass_dens_z = density[:, 1]
    find_liq_slab = find_bulk_liq_index(density, mode = mode)
    range_for_liq_dens = find_liq_slab[0]
    range_org_liq = find_liq_slab[1]
    median_dens_liq = find_liq_slab[3]
    #Calculate the density
    prop_vals = mass_dens_z[range_for_liq_dens] #kg/m^3
    fig, ax = plt.subplots()

    ax.scatter(density[:, 0], density[:, 1], label='Density', color='blue')
    ax.axhline(y=median_dens_liq, color='g', linestyle='--', label='Median Bulk Liq. Density')

    ax.set_xlabel('Z (nm)')
    ax.set_ylabel('Density (kg/m^3)')

    med_dens_pt = find_liq_slab[4]

    ax.scatter(density[range_for_liq_dens, 0],
            density[range_for_liq_dens, 1],
            label='Bulk Liquid',
            color='red')

    # ax.scatter(density[range_org_liq, 0], density[range_org_liq, 1], color='green')

    # ax.scatter(density[med_dens_pt, 0],
    #         density[med_dens_pt, 1],
    #         label='Median Bulk Liq. Density',
    #         color='orange')

    ax.legend(loc='lower center',
          bbox_to_anchor=(0.5, 1.02),
          ncol=3,
          frameon=False)
    # print(np.mean(prop_vals))
    # print("Liq Dens Range")
    # print(max(density[range_for_liq_dens, 0]), min(density[range_for_liq_dens, 0]))
    return prop_vals, fig
from scipy.signal import find_peaks
from findpeaks import findpeaks

def find_bulk_liq_index(density, mode = "liq"):
    #Use np.diff to approximate the 1st derivative
    ES_numdens_z = density[:,1]
    x = density[:,0]
    dy = np.gradient(ES_numdens_z, x)
    #Use findpeaks to find the peaks and valleys, interpolating to get as close as possible
    fp = findpeaks(lookahead=1, interpolate=10, verbose=0)
    results = fp.fit(dy)["df_interp"]

    all_peaks = results[results['peak'] | results['valley']].index.values
    #get the highest peak and the lowest valley, these are the interfaces
    y_vals = results['y'].iloc[all_peaks]
    peak_index = all_peaks[np.argmax(y_vals)]
    valley_index = all_peaks[np.argmin(y_vals)]

    interfaces = [peak_index, valley_index]

    #Divide the indices by 10 to get the correct index for the density based on interpolation
    interfaces = [int(i/10) for i in interfaces]

    #Get the range of indices for the bulk liquid density regardless of whether interfaces have shifted
    if interfaces[0] < interfaces[1]:
        range_org_liq = list(range(interfaces[0], interfaces[1] + 1))
    else:
        range_org_liq = list(range(interfaces[0], len(results)//10)) + list(range(0, interfaces[1] + 1))

    #Get the indecies of the vapor range (the opposite of range_org_liq)
    if interfaces[0] < interfaces[1]:
        range_org_vapor = list(range(interfaces[1] + 1, len(results)//10)) + list(range(0, interfaces[0]))
    else:
        range_org_vapor = list(range(interfaces[1] + 1, interfaces[0]))

    if_r = [peak_index, valley_index]

    # #Divide the indices by 10 to get the correct index for the density based on interpolation
    median_dens_liq = np.median(ES_numdens_z[range_org_liq])
    differences = np.abs(ES_numdens_z - median_dens_liq)
    med_dens_pt = np.argmin(differences)

    median_dens_vap = np.mean(ES_numdens_z[range_org_vapor])
    differences_vap = np.abs(ES_numdens_z - median_dens_vap)
    med_dens_pt_vap = np.argmin(differences_vap)

    # Find the first and last index where density >= median
    valid_idx = np.where(ES_numdens_z[range_org_liq] >= median_dens_liq)[0]
    if valid_idx.size > 0:
        start = valid_idx[0]
        end = valid_idx[-1] + 1  # +1 to include the last valid index
        range_for_liq_dens = range_org_liq[start:end]
    else:
        range_for_liq_dens = np.array([], dtype=int)

    median_dens_liq = np.median(ES_numdens_z[range_for_liq_dens])
    differences = np.abs(ES_numdens_z - median_dens_liq)

    #Do the same for vapor
    valid_idx = np.where(ES_numdens_z[range_org_vapor] <= median_dens_vap)[0]
    if valid_idx.size > 0:
        start = valid_idx[0]
        end = valid_idx[-1] + 1 # +1 to include the last valid index
        range_for_vapor_dens = range_org_vapor[start:end]
    else:
        range_for_vapor_dens = np.array([], dtype=int)

    median_dens_vap = np.mean(ES_numdens_z[range_for_vapor_dens])
    differences_vap = np.abs(ES_numdens_z - median_dens_vap)
    # print("Vap Dens Range")
    idx1_vap = range_org_vapor[0]
    idx2_vap = range_org_vapor[-1]
    # print(density[idx1_vap, 0], density[idx2_vap, 0])

    if mode == "liq":
        return range_for_liq_dens, range_org_liq, ES_numdens_z, median_dens_liq, med_dens_pt
    elif mode == "vap":
        return range_for_vapor_dens, range_org_vapor, ES_numdens_z, median_dens_vap, med_dens_pt_vap

#for all IFT things
#Get all files that match a certain pattern
files = sorted(glob.glob("Build_GPs/vle_iters/workspace/*/calc_props/ift_liq_dens.xvg"))
files2 = sorted(glob.glob("Opt_ES/ift_val_opt/workspace/*/calc_props/ift_liq_dens.xvg"))
files3 = sorted(glob.glob("Opt_ES/ift_val_no_opt/workspace/*/calc_props/ift_liq_dens.xvg"))
all_files = files + files2 + files3
for file in all_files:
    try:
        density = np.loadtxt(file, comments=["#", "@"]) #Gly weird
        save_fig_loc = file.replace("calc_props/ift_liq_dens.xvg", "calc_props/ift_prod_dens.png")
        #If the file exists, pass otherwise create it
        if os.path.exists(save_fig_loc):
            continue
        prop_vals, fig = calc_mass_dens(density, "liq")
        #Save the figure to Build_GPs/vle_iters/workspace/042907986f66368d7fd5e2536a646d49/calc_props
        fig.savefig(save_fig_loc, dpi=300, bbox_inches='tight')
    except:
        pass