import signac
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymser
from matplotlib.backends.backend_pdf import PdfPages

from utils.molec_class_files import esolvs
from Build_GPs.utils.signac import get_signac_results, save_signac_results
from Build_GPs.utils.id_new_samples import new_samples_vle, find_pareto, new_samples_ld, check_mse_10
from Build_GPs.utils.models import get_best_models
from Build_GPs.utils.plot import plot_gp_examples, plot_sim_exp
import pickle

from fffit.fffit.utils import values_real_to_scaled, values_scaled_to_real, values_scaled_to_real_tf, values_real_to_scaled_tf
from fffit.fffit.plot import plot_model_vs_exp
from utils.molec_class_files import esolvs
from Opt_ES.utilsOpt import opt_atom_types
from make_svd_sensitivity import run_sensitivity_analysis

import matplotlib.pyplot as plt
import numpy as np

import os
import tensorflow as tf

def sp_within_bounds(analyzer):
    """
    Check if the state point parameters are within the molecule's bounds
    """
    param_bounds, param_names = analyzer.get_param_bnds_names()
    #Get the max sigma from the bounds and names
    sigma_vals = [v for n, v in zip(param_names, param_bounds) if "sigma" in n]
    max_sigma = np.max(sigma_vals)
    param_bnds2 = analyzer.values_real_to_pref(param_bounds.T).T
    all_molec_dir = analyzer.use_dir_name
    if os.path.exists(os.path.join(all_molec_dir, "best_per_run.csv")):
        all_df = pd.read_csv(os.path.join(all_molec_dir, "best_per_run.csv"), header=0, index_col=0)
    #Get the best set where no bound is approached
    all_data = all_df
    first_param_name = param_names[0] + "_cum"
    last_param_name = param_names[-1] + "_cum"
    param_vals = all_data.loc[:, first_param_name:last_param_name].values
    # Find which bounds are different
    lower_bnd = param_bnds2[:, 0]
    upper_bnd = param_bnds2[:, 1]
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

from Build_GPs.utils.models import get_exp_data, loo_model_perform

###Create Estimability Analysis Tables
# The SVD sensitivity analysis lives in make_svd_sensitivity.py so that it can be
# reproduced on its own, without the simulation-workflow dependencies used below.
# Running it here regenerates the singular values, right singular vectors, the full
# decomposition, and the q_j rankings for every molecule and both temperature modes.
mol_names = ["DEC", "DMF", "DMSO", "EG", "Gly", "MeOH"]
molec_dict = esolvs.make_dict(mol_names)

run_sensitivity_analysis(mol_names=mol_names)

# Work from the repo's Opt_ES directory, resolved relative to this script, rather than a
# hard-coded cluster path. The code below reads "../Build_GPs/..." and resolves GP pickles
# relative to this working directory.
_script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(_script_dir, "Opt_ES"))


###Compare GP Prediction Accuracy to MAPD of final FF model

#For each of the 6 solvents
data_dict = []
#Load GP models
max_mapd = 0
for mol_name in mol_names:
    at_num = 0 
    setup = opt_atom_types.Problem_Setup([mol_name], at_num, "ExpVal")
    # Set parameter set of interest (in this case get the best parameter set)
    x_label = "best_set"
    all_molec_dir = setup.use_dir_name
    GPs = setup.all_gp_dict[mol_name]
    for prop_name in ["liq_density", "surf_tens"]:
        GP_model = GPs[f"sim_{prop_name}"]
        mol_data = molec_dict[mol_name]

        #get the MAPD of final FF model predictions on test set
        #load from Opt_ES/analysis/at_00/MeOH/ExpVal/opt_res/ms_val_opt/error_data.csv
        error_data = pd.read_csv(f"{setup.use_dir_name}/ms_val_opt/error_data.csv")
        ff_mapd = error_data[f'mapd_{prop_name}'].max()
        exp_data, property_bounds, property_name = get_exp_data(mol_data, prop_name)
        # print(f"{mol_name} {prop_name} GP MAPD: {GP_mapd}, FF MAPD: {ff_mapd}")

        #Or get x and y data from the org parameter sets and property values used to train the GP model
        #Get parameter sets and property values for best parameter set
        ms_data = pd.read_csv(f"{setup.use_dir_name}/ms_val_opt/ms_data.csv")
        param_bounds, param_names = setup.get_param_bnds_names()
        first_param_name = param_names[0]
        last_param_name = param_names[-1]
        real_best_set = ms_data.loc[:, first_param_name:last_param_name].values
        real_temp = ms_data.loc[:, "temperature"].values
        real_prop = ms_data.loc[:, f"sim_{prop_name}"].values
        #scale the parameter sets to be between 0 and 1 using the parameter bounds
        scaled_best_set = values_real_to_scaled(real_best_set, param_bounds)
        temperature_bounds = mol_data.temperature_bounds(f"expt_{prop_name}")
        scaled_temp = values_real_to_scaled(real_temp, temperature_bounds)
        scaled_prop = values_real_to_scaled(real_prop, property_bounds)
        x_data = np.hstack((scaled_best_set, scaled_temp))
        y_data = scaled_prop
        
        #Calculate MAPD of GP model predictions on the parameter sets and property values used to train the GP model
        __, GP_mapd = loo_model_perform({prop_name: GP_model}, x_data, y_data, property_bounds)
        
        data_dict.append([mol_name, prop_name, property_name, GP_mapd, ff_mapd])
        if np.maximum(GP_mapd, ff_mapd) > max_mapd:
            max_mapd = np.maximum(GP_mapd, ff_mapd)
    
#Plot the results on a scatter plot with MAPD of GP on x-axis and MAPD of FF on y-axis. 2 subplots for each property
data_df = pd.DataFrame(data_dict, columns=["Molecule", "Property", "Property Name", "GP_MAPD", "FF_MAPD"])
fig, axes = plt.subplots(1, 2, figsize=(12,6))
for i, property_name in enumerate(data_df["Property Name"].unique()):
    label_type = "o" if i == 0 else "s"
    prop_data = data_df[data_df["Property Name"] == property_name]
    colors = plt.cm.tab10.colors
    for j in range(len(prop_data)):
        #If the molecule is not Glycerol, plot it as a point, if it is Glycerol, skip the next color
        # if prop_data["Molecule"].iloc[j] != "Gly":
        axes[i].scatter(prop_data["GP_MAPD"].iloc[j], prop_data["FF_MAPD"].iloc[j], 
                        label=prop_data["Molecule"].iloc[j], 
                        color=colors[j % len(colors)], marker=label_type, s=150, alpha = 0.5)
    # axes[i].set_ylabel("GP-Opt Simulated MAPD", fontsize = 18)
    axes[i].set_title(f"{property_name.split('/')[0]} MAPD Comparison", fontsize = 24)
    if i == 1:
        axes[i].set_ylim(0, max_mapd*1.05)
        # axes[i].set_xlabel(f"GP-Predicted MAPD", fontsize = 18)
    else:
        axes[i].legend(loc="lower right", fontsize = 18, ncol=2)
    axes[i].tick_params("y", direction="inout", which="both", length=7)
    axes[i].tick_params("y", which="major", length=14)
    axes[i].tick_params("x", pad=15)
    #Increase font size of ticks    
    axes[i].tick_params(axis='both', which='major', labelsize=14)
fig.supylabel('FF simulation vs. experiment, MAPD/%', fontsize=18)
fig.supxlabel('GP prediction vs. FF simulation, MAPD/%', fontsize=18)

plt.tight_layout()

#Save the figure
save_path = os.path.join("analysis", "AT-0", "ms_val_opt", 'MAPD_comp_bestFF.png')
fig.savefig(save_path)


### Compare the performance of the GP model and the FF models 
at_number = 0
seed = 1
obj_choice = "ExpVal"

def get_best_set_data(molec_name, mode = "sing"):
    # Check the analysis folder for analysis/MolName/vle_iters folders
    # Find the highest params-iter-X.csv file
    pareto_sets = pd.read_csv(f"../Build_GPs/analysis/{molec_name}/vle_iters/iter-1/final-params.csv", header = 0, index_col = 0)
    all_data = pd.read_csv(f"../Build_GPs/analysis/{molec_name}/vle_iters/all_results.csv", header = 0, index_col = 0)
    #Get the row where the mapd_surf_tens column is lowest
    best_row = pareto_sets.loc[pareto_sets['mapd_surf_tens'].idxmin()]
    #Return the array of all parameters (ignore mapd columns)
    param_set = best_row.drop(labels=[col for col in best_row.index if "mapd" in col])
    #Find the final parameters with the lowest surface tension
    param_set = pd.DataFrame(param_set).T
    mask = (all_data[param_set.columns] == param_set.iloc[0]).all(axis=1)

    #Apply mask
    all_data = all_data[mask]
    all_data = all_data.sort_values(by='temperature', ascending=True)
    all_data = pd.DataFrame(all_data)
    return all_data


# Loop over molecules
molec_list = ["EG", "MeOH", "Gly", "DMSO", "DEC", "DMF"] #["EG", "MeOH", "Gly", "DMSO", "DMF", "DEC"]

for mode in ["sing", "all"]:
    for molec_name in molec_list:
        pdf = PdfPages(f"analysis/at_00/{molec_name}/ExpVal/opt_res/prop_pred/gp_vs_sim_vs_exp_{mode}.pdf")
        visual = opt_atom_types.Vis_Results(molec_name.split("-") , at_number, seed, obj_choice)
        param_bnds, param_names = visual.get_param_bnds_names()
        mol_data = molec_dict[molec_name]
        best_ift_data = get_best_set_data(molec_name)
        all_ms_data = pd.read_csv(f"../Build_GPs/analysis/{molec_name}/vle_iters/all_results.csv", header=0, index_col=0)
        best_ms_data = pd.read_csv(f"analysis/at_00/{molec_name}/ExpVal/opt_res/ms_val/ms_data.csv", header=0, index_col=0)
        
        if len(best_ms_data) > 0:
            #drop all rows after the first
            best_ms_data = best_ms_data.iloc[:15]

        ##Rename any columns in best_ms_data to remove the string "sim_" if the column has it
        best_ms_data = best_ms_data.rename(columns={c: c.replace("sim_", "") for c in best_ms_data.columns if "sim_" in c})
        all_ms_data = all_ms_data.rename(columns={c: c.replace("sim_", "") for c in all_ms_data.columns if "sim_" in c})

        first_param_name = param_names[0]
        last_param_name = param_names[-1]

        group_cols = ["temperature"] + param_names
        best_ms_data = (
            best_ms_data.groupby(group_cols)
            .agg(
                liq_density=("liq_density", "mean"),
                liq_density_unc=("liq_density", "std"),
                surf_tens=("surf_tens", "mean"),
                surf_tens_unc=("surf_tens", "std"),
            )
            .reset_index())
        # print("Best Data ST Unc mean:", molec_name, np.mean(best_ms_data["surf_tens_unc"]))
        all_ms_data = (
            all_ms_data.groupby(group_cols)
            .agg(
                liq_density=("liq_density", "mean"),
                liq_density_unc=("liq_density", "std"),
                surf_tens=("surf_tens", "mean"),
                surf_tens_unc=("surf_tens", "std"),
            )
            .reset_index())

        # Get GPs associated with each molecule
        molec_gps_dict = visual.all_train_gp_dict[molec_name]

        # for i in range(len(best_ms_data)):
        ms_test_set = best_ms_data.loc[0, first_param_name:last_param_name].values
        # print("MS Test Set:", ms_test_set)
        test_params = visual.get_best_results(molec_name, theta_guess=ms_test_set.reshape(1,-1))
        #remove the GAFF key if it exists
        if "GAFF" in test_params:
            del test_params["GAFF"]

        # Loop over gps (1 per property)
        for key in list(molec_gps_dict.keys()):
            key_nosim = key.replace("sim_", "")
            if mode == "all":
                data_labels = [ "GP-Opt (Sim)", "Base (Sim)", "All ST Sets (Sim)",]
                data = [best_ms_data, best_ift_data, all_ms_data]
            elif mode == "sing":
                data_labels = [ "GP-Opt (Sim)", "Base (Sim)",]
                data = [best_ms_data, best_ift_data]
            #Prepare other data to plot
            other_data = {}
            for d_label, df in zip(data_labels, data):
                other_data[d_label] = df[["temperature", f"{key_nosim}", f"{key_nosim}_unc"]]

            # other_data = {f"ST (Sim)": best_ift_data[["temperature", f"{key_nosim}", f"{key_nosim}_unc"]], f"Optimized (Sim)": best_ms_data[["temperature", f"{key_nosim}", f"{key_nosim}_unc"]]}
            # Set label
            # Get GP associated with property
            gp_model = molec_gps_dict[key]
            # Get X and Y data and bounds associated with the GP
            exp_data, y_bounds, y_names = visual.get_exp_data(mol_data, key)
            x_data = np.array(list(exp_data.keys()))
            y_data = np.array(list(exp_data.values()))

            # Plot test vs train for each parameter set
            pdf.savefig(plot_model_vs_exp(
                {molec_name: gp_model},
                test_params,
                exp_data,
                mol_data.temperature_bounds(),
                y_bounds,
                plot_bounds=mol_data.temperature_bounds(),
                property_name=y_names,
                other_data = other_data
                ), bbox_inches='tight')   # save one figure at a time
        pdf.close()