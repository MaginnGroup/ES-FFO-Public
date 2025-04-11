import signac
import numpy as np
import unyt as u
from scipy.stats import qmc
import pandas as pd
import sys
import glob
import os

sys.path.append("../")
from utils.molec_class_files import esolvs
from fffit.fffit.utils import values_scaled_to_real

sys.path.remove("../")

from utils.molec_class_files import esolvs

# Load class properies for each training molecule
mol_names = ["EG"]  # , "Gly", "ACN", "MeOH", "DMSO", "THF", "DCM", "DEC", "DMF"]
molec_dict = esolvs.make_dict(mol_names)


def unpack_molec_values(class_data, state_point, sample):
    """
    Unpacks scaled sample values given the molecule under study
    """
    param_names = class_data.param_names
    for i, param in enumerate(param_names):
        # Unpack the sample, set to 0.0 if the value is less than 1e-14
        if sample[i] > 1.0 * 1e-14:
            sample_use = sample[i]
        else:
            sample_use = 0.0

        state_point[param] = sample_use

    return state_point


def determine_density_iter(molec_name):
    # Check the analysis folder for analysis/MolName/density-iter-X folders
    # Find the highest density-iter-X folder
    files = sorted(glob.glob("analysis/" + molec_name + "/params-iter-*.csv"))
    if len(files) == 0:
        dens_iter = 1
    else:
        # Get the highest density-iter-X folder from the last character of the last file minus the .csv part
        base = os.path.splitext(os.path.basename(files[-1]))[0]
        dens_iter = base[-1]
    return dens_iter


nsteps_nvt1 = 100000  # 100ps
nsteps_npt = 500000  # 500ps (minimum)
nsteps_nvt2 = 100000  # 100ps
nsteps_intereq = 15000000  # 15 ns (minimum)
nsteps_interprod = 50000000  # 50 ns
nmols = 1000  # Number of molecules in the system
aspect_ratio = 3.0  # Aspect ratio of the box


def init_project():
    # Loop over all molecules
    for molec_name, molec_data in molec_dict.items():
        # Determine Density iter based off of the analysis folder
        # For now use dens-iter = 1
        dens_iter = determine_density_iter(molec_name)

        # Initialize project
        project = signac.init_project("runs")

        # Use GenLHS samples to generate LHS samples in the analysis folder
        # Load the lhs_samples and bounds
        bounds = molec_data.param_bounds
        lhs_samples = pd.read_csv(
            "analysis/" + molec_name + "/params-iter-" + str(dens_iter) + ".csv",
            index_col=0,
        )
        # Convert scaled latin hypercube samples to physical values
        scaled_params = values_scaled_to_real(lhs_samples, bounds)

        # Define temps (from constants files)
        temps = list(molec_data.expt_Pvap.keys())
        for temp in [temps[0]]:
            liq_density = molec_data.expt_liq_density[temp]
            for sample in scaled_params[0].reshape(1, -1):
                # Define the state point w/ unchanging characteristics
                state_point = {
                    "mol_name": molec_name,
                    "dens-iter": dens_iter,
                    "smiles": molec_data.smiles_str,
                    "T": float((temp * u.K).in_units(u.K).value),  # K
                    "P": float(molec_data.expt_Pvap[temp]),  # bar
                    "rho_liq": liq_density,  # kg/m^3
                    "nmols": nmols,  # Number of molecules
                    "nsteps_nvt1": nsteps_nvt1,
                    "nsteps_npt": nsteps_npt,
                    "nsteps_nvt2": nsteps_nvt2,
                    "nsteps_intereq": nsteps_intereq,
                    "nsteps_interprod": nsteps_interprod,
                    "cutoff": float(6 * np.max(molec_data.bounds_sig)),
                }

                state_point = unpack_molec_values(molec_data, state_point, sample)

                job = project.open_job(state_point)
                job.init()


if __name__ == "__main__":
    init_project()
