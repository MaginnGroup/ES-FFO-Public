import signac
import numpy as np
import unyt as u
from scipy.stats import qmc
import pandas as pd
import sys
import glob
import os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from utils.molec_class_files import esolvs
from fffit.fffit.utils import values_scaled_to_real

# Load class properies for each training molecule
mol_names = ["DMSO", "DEC", "DMF"] #["EG" , "Gly", "ACN", "MeOH", "DMSO", "THF", "DCM", "DEC", "DMF"]
molec_dict = esolvs.make_dict(mol_names)

def calc_nmols(sp):
    """
    Calculate the number of molecules in the system based on the density and box length
    """
    density = sp["rho_liq"]
    #Calculatue box lengths from system density and 3.0*cutoff (3.6 = 3.0*1.2 A)
    xy_len = 3.6
    new_V = xy_len**3 #Square box
    #Calculate the number of molecules from the new volume and the given density
    nmols = int(np.floor(density*1000*6.022*1e23*new_V/(sp["mol_wt"]*1e27)))

    return sp, nmols


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


def determine_iter(molec_name):
    # Check the analysis folder for analysis/MolName/density-iter-X folders
    # Find the highest density-iter-X folder
    files = sorted(glob.glob("analysis/" + molec_name + "/ld_iters/params-iter-*.csv"))
    if len(files) == 0:
        iter = 1
    else:
        # Get the highest density-iter-X folder from the last character of the last file minus the .csv part
        base = os.path.splitext(os.path.basename(files[-1]))[0]
        iter = base[-1]
    return int(iter)


nsteps_nvt_eq = 100000  # 100ps
nsteps_npt_eq = 500000  # 500ps (minimum)
nsteps_npt_prod = 2500000  # 2.5 ns
aspect_ratio = 3.0  # Aspect ratio of the box


def init_project():
    # Loop over all molecules
    for molec_name, molec_data in molec_dict.items():
        # Determine Density iter based off of the analysis folder
        dens_iter = determine_iter(molec_name)

        # Initialize project
        project = signac.init_project("ld_iters")

        # Use GenLHS samples to generate LHS samples in the analysis folder
        # Load the lhs_samples and bounds
        bounds = molec_data.param_bounds
        lhs_samples = pd.read_csv(
            "analysis/" + molec_name + "/ld_iters/params-iter-" + str(dens_iter) + ".csv",
            index_col=0,
        )
        # Convert scaled latin hypercube samples to physical values
        scaled_params = values_scaled_to_real(lhs_samples, bounds)
        #Make the GAFF param_set (test)
        # scaled_params = molec_data.A_kJmol_to_nm_Kkb(molec_data.gaff_params)
        # scaled_params = np.array(list(scaled_params.values())).reshape(1,-1)

        # Define temps (from constants files)
        temps = list(molec_data.expt_Pvap.keys())
        for temp in temps:
            liq_density = molec_data.expt_liq_density[temp]
            vap_density = molec_data.expt_vap_density[temp]
            max_vd = molec_data.expt_vap_density[max(temps)]
            min_ld = molec_data.expt_liq_density[max(temps)]
            rho_thresh = (max_vd + min_ld)/2.0
            rho_avg = (liq_density + vap_density) / 2.0
            for sample in scaled_params:
                # Define the state point w/ unchanging characteristics
                state_point = {
                    "mol_name": molec_name,
                    "iter": dens_iter,
                    "smiles": molec_data.smiles_str,
                    "T": float((temp * u.K).in_units(u.K).value),  # K
                    "P": float(molec_data.expt_Pvap[temp]),  # bar
                    "rho_liq": liq_density,  # kg/m^3
                    "rho_thresh": rho_thresh,  # kg/m^3
                    "mol_wt": molec_data.molecular_weight,  # g/mol
                    "aspect_ratio": aspect_ratio,  # Aspect ratio of the box
                    "nsteps_nvt_eq": nsteps_nvt_eq,
                    "nsteps_npt_eq": nsteps_npt_eq,
                    "nsteps_npt_prod": nsteps_npt_prod,
                    "max_sigma" : np.max(molec_data.bounds_sig)
                }

                state_point, max_sigma = unpack_molec_values(molec_data, state_point, sample)
                state_point, nmols = calc_nmols(state_point)
                state_point["nmols"] = nmols
                #Optionally set max_sigma to the max sigma value from the FF rather than the max from the bounds
                # state_point["max_sigma"] = max_sigma

                job = project.open_job(state_point)
                job.init()

if __name__ == "__main__":
    init_project()
