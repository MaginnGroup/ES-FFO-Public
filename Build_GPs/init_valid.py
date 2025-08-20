import signac
import pandas as pd
import sys
import glob
import os
import unyt as u
import numpy as np

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from utils.molec_class_files import esolvs
from utils.molec_class_files import esolvs

# Load class properies for each training molecule
mol_names = [
    "EG",
    "MeOH",
]  # ["EG" , "Gly", "ACN", "MeOH", "DMSO", "THF", "DCM", "DEC", "DMF"]
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


def get_best_param_set(molec_name):
    # Check the analysis folder for analysis/MolName/vle_iters folders
    # Find the highest params-iter-X.csv file
    pareto_sets = pd.read_csv(f"analysis/{molec_name}/vle_iters/iter-1/final-params.csv", header = 0, index_col = 0)

    #Get the row where the mapd_surf_tens column is lowest
    best_row = pareto_sets.loc[pareto_sets['mapd_surf_tens'].idxmin()]
    #Return the array of all parameters (ignore mapd columns)
    param_set = best_row.drop(labels=[col for col in best_row.index if "mapd" in col]).values
    return param_set


# Load class properies for each training molecule
mol_names = ["EG" , "MeOH"]
molec_dict = esolvs.make_dict(mol_names)

num_restarts = 3  # Number of restarts for replications
n_vap = 160  # number of molecules in vapor phase
n_liq = 640

# Initialize project
project = signac.init_project("vle_val")

def init_project():
    # Loop over all molecules
    for molec_name, molec_data in molec_dict.items():
        # Define temps (from constants files)
        temps = list(molec_data.expt_Pvap.keys())

        # Run at vapor pressure (from constants file)
        press = molec_data.expt_Pvap

        # For each restart
        for restart in range(num_restarts):
            # Loop over temperatures
            for temp in temps:
                # Theoretically, we could examine more than just the best
                # Define the initial state point
                state_point = {
                    "mol_name": molec_name,
                    "mol_weight": molec_data.molecular_weight,  # amu
                    "smiles": molec_data.smiles_str,
                    "N_atoms": molec_data.n_atoms,
                    "T": float(temp),  # K
                    "P": float(press[temp]),  # bar
                    "N_vap": n_vap,
                    "N_liq": n_liq,
                    "expt_liq_density": molec_data.expt_liq_density[temp],  # kg/m^3
                    "nsteps_nvt": 2500,
                    "nsteps_npt": 5000,
                    "nsteps_gemc_eq": 50000, #We will actually be using sweeps and not steps as units here
                    "nsteps_gemc_prod": 100000, #Sweeps
                    "restart": restart + 1,
                    
                }
                param_set = get_best_param_set(molec_name)
                #Save SP for lowest error param set for EG and MeOH
                state_point, max_sigma = unpack_molec_values(molec_data, state_point, param_set)
                state_point["max_sigma"] = max_sigma

                # Initialize the GAFF jobs
                job = project.open_job(state_point)
                job.init()

if __name__ == "__main__":
    init_project()
