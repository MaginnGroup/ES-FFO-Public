import signac
import numpy as np
import unyt as u
import sys
import pandas as pd
import os
sys.path.append("..")
from utils.molec_class_files import esolvs
sys.path.remove("..")
from utilsOpt import opt_atom_types

# Load class properies for each training molecule
mol_names = ["EG", "Gly", "MeOH", "DMSO", "DEC", "DMF"]
gen_FF_mols = ["EG", "Gly", "MeOH"]
molec_dict = esolvs.make_dict(mol_names)

at_numbers = [0] #0 is distinct ATs, 1 is CHO, 2 is GAFF
num_restarts = 3  # Number of restarts for replications
n_vap = 160  # number of molecules in vapor phase
n_liq = 640
obj_choice = "ExpVal"  # Objective to consider
mode = "opt" #no_opt or opt

# Initialize project
proj_name = "gemc_val" if mode == None else f"gemc_val_{mode}"
project = signac.init_project(proj_name)

def unpack_molec_values(molec_name, at_class, sample, state_point):
    """
    Unpacks sckaled sample values given the molecule under study
    """
    max_sigma = 0
    # Unpack the sample according to atom typing scheme mapping dictionary
    molec_map_dict = at_class.molec_map_dicts[molec_name]
    param_names = molec_map_dict.keys()
    # Get param names in order of original mapping
    order = molec_data.param_names
    index_mapping = {elem: idx for idx, elem in enumerate(order)}
    # Add params based on the order they show up in given the mapping
    for param in param_names:
        state_point[param] = sample[index_mapping[param]].item()
        if "sigma" in param:
            if state_point[param] > max_sigma:
                max_sigma = state_point[param]
    return state_point, max_sigma

def get_gaff_sp(molec_data, state_point):
    """
    Unpacks the GAFF state point
    """
    param_names = molec_data.keys()
    # Get param names in order of original mapping
    for param in param_names:
        state_point[param] = molec_data.gaff_params[param]
    return state_point

def sp_within_bounds(analyzer):
    """
    Check if the state point parameters are within the molecule's bounds
    """
    param_bounds, param_names = analyzer.get_param_bnds_names()
    #Get the max sigma from the bounds and names
    sigma_vals = [v for n, v in zip(param_names, param_bounds) if "sigma" in n]
    max_sigma = np.max(sigma_vals)
    param_bnds = analyzer.values_real_to_pref(param_bounds.T).T
    all_molec_dir = analyzer.use_dir_name
    if os.path.exists(all_molec_dir / "unique_best_set.csv"):
        all_df = pd.read_csv(all_molec_dir / "unique_best_set.csv", header=0)
    #Get the best set where no bound is approached
    param_vals = all_df.to_numpy()
    # Find which bounds are different
    lower_bnd = param_bnds[:, 0]
    upper_bnd = param_bnds[:, 1]
    dif_bnds = lower_bnd != upper_bnd
    # Check closeness to bounds for params that have variable bounds
    close_to_lower = np.isclose(param_vals[:,dif_bnds], lower_bnd[dif_bnds])
    close_to_upper = np.isclose(param_vals[:,dif_bnds], upper_bnd[dif_bnds])
    close_any = np.logical_or(close_to_lower, close_to_upper)
    # A "valid" row has no True in close_any
    valid_rows = ~close_any.any(axis=1)
    # Pick first valid row or the first row if none are valid
    best_idx = np.argmax(valid_rows) if valid_rows.any() else 0
    return best_idx

# Loop over all molecules
for molec_name, molec_data in molec_dict.items():
    # Define temps (from constants files)
    temps = list(molec_data.expt_Pvap.keys())

    # Run at vapor pressure (from constants file)
    press = molec_data.expt_Pvap

    # Theoretically, we could examine more than just the best
    # Define the initial state point
    state_point = {
        "mol_name": molec_name,
        "obj_choice": obj_choice,
        "mol_weight": molec_data.molecular_weight,  # amu
        "smiles": molec_data.smiles_str,
        "N_atoms": molec_data.n_atoms,
        "N_vap": n_vap,
        "N_liq": n_liq,
        "nsteps_nvt": 2500,
        "nsteps_npt": 5000,
        "nsteps_gemc_eq": 50000, #We will actually be using sweeps and not steps as units here
        "nsteps_gemc_prod": 100000, #Sweeps            
    }

    #Save GAFF params to state point
    # state_point_gaff = state_point.copy()
    # state_point_gaff["atom_type"] = "GAFF"
    # state_point_gaff["obj_choice"] = "None"
    # state_point_gaff = get_gaff_sp(molec_data, state_point_gaff)
    # # Initialize the GAFF jobs
    # job = project.open_job(state_point_gaff)
    # job.init()

    #Loop over all other atom typing schemes
    for at_number in at_numbers:
        molec_names = [molec_name] if at_number == 0 else gen_FF_mols
        #Only make jobs for molecule if using distinct ATs or if the molecules is part of the generalized FF
        if at_number == 0 or molec_name in gen_FF_mols:
            state_point["atom_type"] = at_number
            setup = opt_atom_types.Problem_Setup(molec_names, at_number, obj_choice)
            param_bounds, param_names = setup.get_param_bnds_names()
            #Get the max sigma from the bounds and names
            sigma_vals = [v for n, v in zip(param_names, param_bounds) if "sigma" in n]
            max_sigma = np.max(sigma_vals)
            param_bnds =setup.values_real_to_pref(param_bounds)
            all_molec_dir = setup.use_dir_name
            if mode == "opt" or mode == None:
                if os.path.exists(all_molec_dir / "unique_best_set.csv"):
                    all_df = pd.read_csv(all_molec_dir / "unique_best_set.csv", header=0)
                else:
                    break
                # Get best set
                if at_number == 0:
                    best_idx = sp_within_bounds(setup)
                else:
                    #Get the best set where no bound is approached
                    param_vals = all_df.to_numpy()
                    # Find which bounds are different
                    lower_bnd = param_bnds[:, 0]
                    upper_bnd = param_bnds[:, 1]
                    dif_bnds = lower_bnd != upper_bnd
                    # Check closeness to bounds for params that have variable bounds
                    close_to_lower = np.isclose(param_vals[:,dif_bnds], lower_bnd[dif_bnds])
                    close_to_upper = np.isclose(param_vals[:,dif_bnds], upper_bnd[dif_bnds])
                    close_any = np.logical_or(close_to_lower, close_to_upper)
                    # A "valid" row has no True in close_any
                    valid_rows = ~close_any.any(axis=1)
                    # Pick first valid row or the first row if none are valid
                    best_idx = np.argmax(valid_rows) if valid_rows.any() else 0
                all_best_real = setup.values_pref_to_real(all_df.iloc[best_idx].values)
            else:
                #If not using optimized parameter sets, putt the lowest IFT error set from IFT pareto iterations
                # Load all IFT pareto sets and choose the one with the lowest ST error
                pareto_sets = pd.read_csv(f"../Build_GPs/analysis/{molec_name}/vle_iters/iter-1/final-params.csv", header = 0, index_col = 0)
                #Get the row where the mapd_surf_tens column is lowest
                best_idx = pareto_sets['mapd_surf_tens'].idxmin()
                best_row = pareto_sets.loc[best_idx]
                #Return the array of all parameters (ignore mapd columns)
                all_best_real = best_row.drop(labels=[col for col in best_row.index if "mapd" in col]).values

            state_point["param_set"] = best_idx + 1
            
            # Parameters in units nm and kJ/mol
            if at_number > 0:
                param_matrix = setup.at_class.get_transformation_matrix({molec_name: molec_data})
                scaled_params = all_best_real.reshape(-1, 1).T @ param_matrix
                train_mol_str = "-".join(gen_FF_mols)
            else:
                scaled_params = all_best_real.reshape(1, -1)
                train_mol_str = molec_name
            
            state_point["train_mol_str"] = train_mol_str

            # Loop over temperatures
            for temp in temps:
                #Only make jobs for the first and last temp for speed
                if temp == temps[0] or temp == temps[-1]:
                    state_point["T"] = float(temp)  # K
                    state_point["P"] =float(press[temp])  # bar
                    state_point["expt_liq_density"] = molec_data.expt_liq_density[temp]  # kg/m^3
                    # For each restart
                    for restart in range(num_restarts):
                        state_point["restart"] = restart + 1 
                        # Loop over all scaled samples
                        for sample in scaled_params:
                            state_point, max_sigma_ff = unpack_molec_values(
                                    molec_name, setup.at_class, sample, state_point
                                ) 
                            state_point["max_sigma"] = max_sigma
                            job = project.open_job(state_point)
                            job.init()