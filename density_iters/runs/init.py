import signac
import numpy as np
import unyt as u
from scipy.stats import qmc
import pandas as pd
import sys
import glob
import os

sys.path.append("../..")
from utils.molec_class_files import esolvs
from fffit.fffit.utils import values_scaled_to_real

sys.path.remove("../..")

from utils.molec_class_files import esolvs

# Load class properies for each training molecule
mol_names = ["EG", "Gly", "ACN", "MeOH", "DMSO", "THF", "DCM", "DEC"]
molec_dict = esolvs.make_dict(mol_names)


def unpack_molec_values(class_data, state_point, sample):
    """
    Unpacks scaled sample values given the molecule under study
    """
    param_names = class_data.param_names
    for i, param in enumerate(param_names):
        if "sigma" in param:
            state_point[param] = float((sample[i] * u.Angstrom).in_units(u.nm).value)
        elif "epsilon" in param:
            state_point[param] = float(
                (sample[i] * u.K * u.kb).in_units("kJ/mol").value
            )
        state_point[param] = sample[i]

    return state_point


def determine_density_iter(molec_name):
    # Check the analysis folder for analysis/MolName/density-iter-X folders
    # Find the highest density-iter-X folder
    files = sorted(glob.glob("../analysis/" + molec_name + "/dens-iter-*"))
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
nsteps_intereq = 25000000  # 25 ns (minimum)
nsteps_interprod = 50000000  # 50 ns


def init_project():
    # Loop over all molecules
    for molec_name, molec_data in molec_dict.items():
        # Determine Density iter based off of the analysis folder
        # For now use dens-iter = 1
        dens_iter = determine_density_iter(molec_name)

        # Initialize project
        project = signac.init_project("dens-iter-" + str(dens_iter))

        # Use GenLHS samples to generate LHS samples in the analysis folder
        # Load the lhs_samples and bounds
        bounds = molec_data.param_bounds
        lhs_samples = pd.read_csv(
            "../analysis/" + molec_name + "/dens-iter-" + str(dens_iter) + ".csv",
            index_col=0,
        )
        # Convert scaled latin hypercube samples to physical values
        scaled_params = values_scaled_to_real(lhs_samples, bounds)

        # Define temps (from constants files)
        temps = list(molec_data.expt_Pvap.keys())
        for temp in temps:
            liq_density = molec_data.expt_liq_density[temp]
            vap_density = molec_data.expt_vap_density[temp]
            avg_density = (liq_density + vap_density) / 2
            for sample in scaled_params:
                # Define the state point w/ unchanging characteristics
                state_point = {
                    "mol_name": molec_name,
                    "dens-iter": dens_iter,
                    "smiles": molec_data.smiles_str,
                    "T": float((temp * u.K).in_units(u.K).value),
                    "P": float(molec_data.expt_Pvap[temp]),
                    "rho_avg": avg_density,  # kg/m^3
                    "nsteps_nvt1": nsteps_nvt1,
                    "nsteps_npt": nsteps_npt,
                    "nsteps_nvt2": nsteps_nvt2,
                    "nsteps_intereq": nsteps_intereq,
                    "nsteps_interprod": nsteps_interprod,
                }

                state_point = unpack_molec_values(molec_data, state_point, sample)

                # print(state_point)

                # job = project.open_job(state_point)
                # job.init()


if __name__ == "__main__":
    init_project()
