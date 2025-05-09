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
mol_names = ["R125"] #["EG" , "Gly", "ACN", "MeOH", "DMSO", "THF", "DCM", "DEC", "DMF"]
molec_dict = esolvs.make_dict(mol_names)


def unpack_molec_values(class_data, state_point, sample):
    """
    Unpacks scaled sample values given the molecule under study
    """
    param_names = class_data.param_names
    max_sigma = 0
    for i, param in enumerate(param_names):
        # Unpack the sample, set to 0.0 if the value is less than 1e-14
        if sample[i] > 1.0 * 1e-14:
            sample_use = sample[i]
        else:
            sample_use = 0.0

        if "sigma" in param:
            if sample_use > max_sigma:
                max_sigma = sample_use

        state_point[param] = sample_use

    return state_point, max_sigma


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


nsteps_nvt_eq = 100000  # 100ps
nsteps_npzzat_eq = 5000000  # 5 ns
nsteps_npzzat_prod = 10000000  # 10 ns
nsteps_fl_eq = 100000  # 100ps
nsteps_npt_pre_eq = 500000  # 500ps
nsteps_npt_eq = 500000  # 500ps (minimum)
nsteps_npt_prod = 2500000  # 2.5 ns
nsteps_nvt_prod = 100000  # 100 ps
nsteps_intereq = 30000000  # 15 ns (minimum)
nsteps_interprod = 40000000  # 30 ns
n_particles = 10000  # Number of particles in the system
nmols = 1000  # Number of molecules in the system
aspect_ratio = 3.0  # Aspect ratio of the box


def init_project():
    # Loop over all molecules
    for molec_name, molec_data in molec_dict.items():
        # Determine Density iter based off of the analysis folder
        # For now use dens-iter = 1
        dens_iter = determine_density_iter(molec_name)

        # Initialize project
        project = signac.init_project("npt_core_test")

        # Use GenLHS samples to generate LHS samples in the analysis folder
        # Load the lhs_samples and bounds
        bounds = molec_data.param_bounds
        lhs_samples = pd.read_csv(
            "analysis/" + molec_name + "/params-iter-" + str(dens_iter) + ".csv",
            index_col=0,
        )
        # Convert scaled latin hypercube samples to physical values
        scaled_params = values_scaled_to_real(lhs_samples, bounds)
        #Make the GAFF param_set (test)
        scaled_params = molec_data.A_kJmol_to_nm_Kkb(molec_data.gaff_params)
        scaled_params = np.array(list(scaled_params.values())).reshape(1,-1)
        # nmols = int(n_particles/molec_data.n_atoms) #Number of molecules in the system
        # Define temps (from constants files)
        temps = list(molec_data.expt_Pvap.keys())
        for temp in [temps[-3]]:
            liq_density = molec_data.expt_liq_density[temp]
            vap_density = molec_data.expt_vap_density[temp]
            max_vd = molec_data.expt_vap_density[max(temps)]
            min_ld = molec_data.expt_liq_density[max(temps)]
            rho_thresh = (max_vd + min_ld)/2.0
            rho_avg = (liq_density + vap_density) / 2.0
            for sample in scaled_params[0].reshape(1, -1):
                # Define the state point w/ unchanging characteristics
                state_point = {
                    "mol_name": molec_name,
                    "dens-iter": dens_iter,
                    "smiles": molec_data.smiles_str,
                    "T": float((temp * u.K).in_units(u.K).value),  # K
                    "P": float(molec_data.expt_Pvap[temp]),  # bar
                    "rho_liq": liq_density,  # kg/m^3
                    "rho_thresh": rho_thresh,  # kg/m^3
                    # "rho_avg": rho_avg,  # kg/m^3
                    "mol_wt": molec_data.molecular_weight,  # g/mol
                    # "nmols": nmols,  # Number of molecules
                    "aspect_ratio": aspect_ratio,  # Aspect ratio of the box
                    "nsteps_nvt_eq": nsteps_nvt_eq,
                    # "nsteps_npzzat_eq": nsteps_npzzat_eq,
                    # "nsteps_npzzat_prod": nsteps_npzzat_prod,
                    # "nsteps_fl_eq": nsteps_fl_eq,
                    # "nsteps_npt_pre_eq": nsteps_npt_pre_eq,
                    "nsteps_npt_eq": nsteps_npt_eq,
                    "nsteps_npt_prod": nsteps_npt_prod,
                    # "nsteps_nvt_prod": nsteps_nvt_prod,
                    "nsteps_intereq": nsteps_intereq,
                    "nsteps_interprod": nsteps_interprod,
                    "max_sigma" : np.max(molec_data.bounds_sig)
                }

                state_point, max_sigma = unpack_molec_values(molec_data, state_point, sample)
                # state_point["max_sigma"] = max_sigma

                job = project.open_job(state_point)
                job.init()


if __name__ == "__main__":
    init_project()
