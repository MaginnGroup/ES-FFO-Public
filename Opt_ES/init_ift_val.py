import signac
import numpy as np
import unyt as u
from scipy.stats import qmc
import pandas as pd
import sys
import glob
import os
import pickle

from utilsOpt import opt_atom_types

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from utils.molec_class_files import esolvs
from fffit.fffit.utils import values_scaled_to_real, values_real_to_scaled

from utils.molec_class_files import esolvs

at_numbers = [0, 1]
obj_choice = "ExpVal"
gen_FF_mols = ["EG", "Gly", "MeOH"]
num_restarts = 3
# Load class properies for each training molecule
mol_names = [
    "EG",
    "Gly",
    "MeOH",
    "DMSO",
    "DEC",
    "DMF",
]  # ["EG" , "Gly", "ACN", "MeOH", "DMSO", "THF", "DCM", "DEC", "DMF"]
molec_dict = esolvs.make_dict(mol_names)


def calc_nmols(sp):
    """
    Calculate the number of molecules in the system based on the density and box length
    """
    nmols = 2000  # Use no fewer than 2000 molecules (8000 particles)
    density = sp["rho_liq"]
    # Calculate the box lengths from the system density using 2000 molecules
    V = (nmols * sp["mol_wt"] * 1e27) / (density * 1000 * 6.022 * 1e23)
    xy_len = (V / sp["aspect_ratio"]) ** (1 / 3)

    # If 2000 molecules is not enough to satisfy xy_len > 13.2*max_sigma
    if xy_len < 13.2 * sp["max_sigma"]:
        # Calculatue box lengths from system density and 13.2*max_sigma
        xy_len = 13.2 * sp["max_sigma"]
        new_V = sp["aspect_ratio"] * xy_len**3
        # Calculate the number of molecules from the new volume and the given density
        nmols = int(
            np.floor(density * 1000 * 6.022 * 1e23 * new_V / (sp["mol_wt"] * 1e27))
        )

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
    # Check the analysis folder for analysis/MolName/vle_iters folders
    # Find the highest params-iter-X.csv file
    files = sorted(glob.glob("../Build_GPs/analysis/" + molec_name + "/vle_iters/params-iter-*.csv"))
    if len(files) == 0:
        iter = 1
    else:
        # Get the highest params-iter-X.csv file from the last character of the file (before the .csv)
        base = os.path.splitext(os.path.basename(files[-1]))[0]
        iter = base[-1]
    return int(iter)


def get_gp_models(molec_name, vle_iter_num):
    # For the 1st VLE iteration, load the best GP models from the LD iterations
    if vle_iter_num == 1:
        files = sorted(
            glob.glob(f"../Build_GPs/analysis/{molec_name}/ld_iters/iter-*/best_gp_models.pkl")
        )
    # For all other VLE iterations, load the best GP models from the VLE iterations
    else:
        files = sorted(
            glob.glob(f"../Build_GPs/analysis/{molec_name}/vle_iters/iter-*/best_gp_models.pkl")
        )
    # Load the last file (most recent)
    with open(files[-1], "rb") as f:
        gp_model = pickle.load(f)
        ld_model = gp_model["sim_liq_density"]
    return ld_model


def get_ld_est(gp_model, temps, samples):
    # Get the LD estimate for the given molecule
    samples_repeat = samples.loc[np.repeat(samples.index, len(temps))].reset_index(
        drop=True
    )
    # Add temperature column
    samples_repeat["temperature"] = np.tile(temps, len(samples))
    # Order the samples by temperature then by the rest of the parameters
    samples_repeat = samples_repeat.sort_values(by=["temperature"])
    # Get the LD estimate
    samples_array = samples_repeat.to_numpy()
    ld_est, var_est = gp_model.predict_f(samples_array)
    return ld_est


nsteps_nvt_eq = 100000  # 100ps
nsteps_npzzat_eq = 500000  # 500 ps
nsteps_npzzat_prod = 2500000  # 2.5 ns
# nsteps_npt_eq = 500000  # 500ps (minimum)
# nsteps_npt_prod = 2500000  # 2.5 ns
nsteps_intereq = 40000000  # 40 ns (minimum)
nsteps_interprod = 40000000  # 40 ns
aspect_ratio = 3.0  # Aspect ratio of the box


def init_project():
    # Loop over all molecules
    for molec_name, molec_data in molec_dict.items():
        if os.path.exists("../Build_GPs/analysis/" + molec_name + "/vle_iters/params-iter-1.csv"):
            # Determine iter based off of the analysis folder
            vle_iter = determine_iter(molec_name)

            # Initialize project
            project = signac.init_project("ift_val")

            # Get the parameter bounds for the molecule
            bounds = molec_data.param_bounds
            # Define temps (from constants files)
            temps = list(molec_data.expt_Pvap.keys())
            temp_bnds = molec_data.temperature_bounds("expt_liq_density")
            scaled_temps = values_real_to_scaled(temps, temp_bnds).flatten()
            
            for at_number in at_numbers:
                molec_names = [molec_name] if at_number == 0 else gen_FF_mols
                #Only make jobs for molecule if using distinct ATs or if the molecules is part of the generalized FF
                if at_number == 0 or molec_name in gen_FF_mols:
                    # Load samples from the opt_at_params Results folder
                    analyzer = opt_atom_types.Analyze_opt_res(molec_names, at_number, 1, obj_choice)
                    unique_real = pd.read_csv(analyzer.use_dir_name + "/unique_best_set.csv", header=0, index_col=0).values
                    best_idx = 0 if at_number == 0 else 2  # Use only the best set
                    unique_best = unique_real[best_idx]  # Use only the best set
                    # If using generalized atom types, transform the parameters to the distinct parameters for the molecule
                    if at_number > 0:
                        param_matrix = analyzer.at_class.get_transformation_matrix({molec_name: molec_data})
                        best_nm_k = unique_best.reshape(-1, 1).T @ param_matrix
                    else:
                        best_nm_k = unique_best.reshape(1, -1)

                    #Scale distinct sample between 0 and 1
                    new_samples_scl = values_real_to_scaled(best_nm_k, bounds)
                    new_samples = pd.DataFrame(new_samples_scl, columns=molec_data.param_names)

                    # Load the GP models for the given molecule and get the LD estimates
                    ld_model = get_gp_models(molec_name, vle_iter)
                    ld_bnds = molec_data.liq_density_bounds
                    ld_est_scl = get_ld_est(ld_model, scaled_temps, new_samples)
                    ld_est_real = values_scaled_to_real(ld_est_scl, ld_bnds).flatten()
                    ld_estimates = np.around(
                        ld_est_real.reshape(len(temps), len(new_samples)), 2
                    )

                    # Convert scaled samples to physical values
                    scaled_params = values_scaled_to_real(new_samples, bounds)
                    # Make the GAFF param_set (test)
                    # scaled_params = molec_data.A_kJmol_to_nm_Kkb(molec_data.gaff_params)
                    # scaled_params = np.array(list(scaled_params.values())).reshape(1,-1)

                    for i, temp in enumerate(temps):
                        max_vd = molec_data.expt_vap_density[max(temps)]
                        min_ld = molec_data.expt_liq_density[max(temps)]
                        rho_thresh = (max_vd + min_ld) / 2.0
                        for j, sample in enumerate(scaled_params):
                            # Get the LD estimate for the given sample
                            liq_density_est = ld_estimates[i, j]
                            # Optionally, use the lower density between the estimate and the experimental value to calculate initial volume/box length
                            # liq_density = np.minimum(
                            #     liq_density_est, molec_data.expt_liq_density[temp]
                            # )
                            liq_density = liq_density_est
                            # Define the state point w/ unchanging characteristics
                            state_point = {
                                "mol_name": molec_name,
                                "atom_type": at_number,
                                "obj_choice": obj_choice,
                                "param_set": best_idx + 1,
                                "smiles": molec_data.smiles_str,
                                "T": float((temp * u.K).in_units(u.K).value),  # K
                                "P": float(molec_data.expt_Pvap[temp]),  # bar
                                "rho_liq": liq_density,  # kg/m^3
                                "rho_thresh": rho_thresh,  # kg/m^3
                                "mol_wt": molec_data.molecular_weight,  # g/mol
                                "aspect_ratio": aspect_ratio,  # Aspect ratio of the box
                                "nsteps_nvt_eq": nsteps_nvt_eq,
                                "nsteps_npzzat_eq": nsteps_npzzat_eq,
                                "nsteps_npzzat_prod": nsteps_npzzat_prod,
                                "nsteps_intereq": nsteps_intereq,
                                "nsteps_interprod": nsteps_interprod,
                                "max_sigma": np.max(molec_data.bounds_sig),
                            }
                            # Calculate the number of molecules in the system based on the density and box length (defined by max_sigma)
                            state_point, max_sigma = unpack_molec_values(
                                molec_data, state_point, sample
                            )
                            state_point, nmols = calc_nmols(state_point)
                            state_point["nmols"] = nmols
                            # Optionally define max_sigma in the state point as the highest value of the parameters
                            # state_point["max_sigma"] = max_sigma
                            for restart in range(num_restarts):
                                state_point["restart"] = restart + 1
                                # Open the job and initialize if it doesn't already exist
                                job = project.open_job(state_point)
                                job.init()
        else:
            print(
                f"Skipping {molec_name} as it is not ready for VLE iters. No params-iter-1.csv found."
            )


if __name__ == "__main__":
    init_project()
