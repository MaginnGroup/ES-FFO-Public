import signac
import sys
import os
from pathlib import Path
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import matplotlib.pyplot as plt


# Now import using package structure relative to ES-FFO root
from utils.molec_class_files import esolvs
from Build_GPs.utils.signac import get_signac_results, save_signac_results
from Build_GPs.utils.id_new_samples import new_samples_vle, find_pareto, check_mse_10
from Build_GPs.utils.models import get_best_models
from Build_GPs.utils.plot import plot_gp_examples, plot_sim_exp
from utils.prep_ms_data import prepare_df_errors, prepare_df_props
import glob 
import shutil

def determine_iter(molec_name):
    # Check the analysis folder for analysis/MolName/vle_iters folders
    # Find the highest params-iter-X.csv file
    files = sorted(glob.glob("analysis/" + molec_name + "/ld_iters/params-iter-*.csv"))
    if len(files) == 0:
        iter = 1
    else:
        # Get the highest params-iter-X.csv file from the last character of the file (before the .csv)
        base = os.path.splitext(os.path.basename(files[-1]))[0]
        iter = base[-1]
    return int(iter) - 1

os.chdir("/groups/ed/group_members/Montana_Carlozo/ES-FFO/Build_GPs")

### Get pareto set info for IFT iters for each molecule

#Set iters to analyze and properties to analyze
iters = [1]  # Change me as needed
property_names = ["liq_density", "surf_tens"]  # Change me as needed
all_names = ["EG", "Gly", "MeOH", "DMSO", "DEC", "DMF"]

for mol in all_names:
    mol_names = [mol]
    #Set seeds and preferences
    cl_shuffle_seed = 1  # classifier
    gp_shuffle_seed = 42  # GP seed 30 for Gly (36)
    dist_seed = 1  # Distance seed
    mapd_le = 10
    save_csv = False
    save_fig = False
    verbose = True
    show_num = 5
    mode = f"best_{show_num}" #"pareto" #"sing", "pareto", or "all"
    mode = "pareto"



    ##############################################################################
    ##############################################################################
    if mode == "pareto" or mode =="all":
        def get_best_set_data(molec_name, mode="pareto"):
            # Check the analysis folder for analysis/MolName/vle_iters folders
            # Find the highest params-iter-X.csv file
            pareto_sets = pd.read_csv(f"analysis/{molec_name}/vle_iters/iter-1/pareto-params.csv", header = 0, index_col = 0)
            all_data = pd.read_csv(f"analysis/{molec_name}/vle_iters/all_results.csv", header = 0, index_col = 0)
            #Get the row where the mapd_surf_tens column is lowest
            #Return the array of all parameters (ignore mapd columns)
            param_set = pareto_sets.drop(columns=[col for col in pareto_sets.columns if any(x in col for x in ["mapd", "mse", "mae"])])
            common_cols = list(set(param_set.columns) & set(all_data.columns))
            # Filter all_data to rows that match any row in param_set on common_cols
            pareto_data = all_data.merge(param_set[common_cols].drop_duplicates(), on=common_cols)
            new_data = pd.DataFrame(pareto_data)

            #Uncomment last line to get all IFT sets
            if mode == "all":
                new_data = all_data

            
            return new_data
    elif mode == "sing":
        def get_best_set_data(molec_name, mode = "sing"):
            # Check the analysis folder for analysis/MolName/vle_iters folders
            # Find the highest params-iter-X.csv file
            pareto_sets = pd.read_csv(f"analysis/{molec_name}/vle_iters/iter-1/final-params.csv", header = 0, index_col = 0)
            all_data = pd.read_csv(f"analysis/{molec_name}/vle_iters/all_results.csv", header = 0, index_col = 0)
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
    else:
        def get_best_set_data(molec_name, mode = mode):
            # Check the analysis folder for analysis/MolName/vle_iters folders
            # Find the highest params-iter-X.csv file
            molec_dict = esolvs.make_dict(mol_names)
            molecule = molec_dict[molec_name] 
            # pareto_sets = pd.read_csv(f"analysis/{molec_name}/ld_iters/mse-less10-full.csv", header = 0, index_col = 0)
            #Map exp data to density
            all_data = pd.read_csv(f"analysis/{molec_name}/vle_iters/all_results.csv", header = 0, index_col = 0)
            ld_threshold = (min(list(molecule.expt_liq_density.values()))+ max(list(molecule.expt_vap_density.values()))) / 2
            df_all, df_liquid, df_vapor = prepare_df_props(all_data, molecule, ld_threshold, scale=False)
                        #Calculate MAPD between expt_liq_density and liq_density for each param set, 
            # Prepare error data to find pareto points
            pareto_sets = prepare_df_errors(df_liquid, molec_dict, molec_name)
            prop_sort = "mapd_" + props
            if "worst" in mode:
                top_sets = pareto_sets.sort_values(by=prop_sort).tail(show_num)
            else:
                top_sets = pareto_sets.sort_values(by=prop_sort).head(show_num)
            
            print("Top sets: ", top_sets)
            # Drop non-parameter columns
            exclude = ["mse", "mpd", "mapd", "mae"]

            param_cols = [c for c in top_sets.columns if not any(substr in c for substr in exclude)]
            top_sets_params = top_sets[param_cols]
            # print(top_sets_params)

            # Build a mask: True if a row in all_data matches ANY of the top 15 rows
            mask = np.zeros(len(all_data), dtype=bool)
            for _, row in top_sets_params.iterrows():
                mask |= (all_data[param_cols] == row).all(axis=1)

            # Filter
            all_data = all_data[mask].sort_values(by='temperature', ascending=True)
            all_data = pd.DataFrame(all_data)
            # print(all_data.columns)
            return all_data

    def make_pareto_plots(molec_name, mode="pareto", err_met="mapd"):
        # Check the analysis folder for analysis/MolName/vle_iters folders
        # Find the highest params-iter-X.csv file
        pareto_sets = pd.read_csv(f"analysis/{molec_name}/vle_iters/iter-1/pareto-params.csv", header = 0, index_col = 0)


        pareto_sets_errs = pareto_sets[[col for col in pareto_sets.columns if any(x in col for x in ["mapd", "mse", "mae", "mpd"])]]
        
        #Plot err_met_liq_density vs err_met_surf_tens for each row in pareto_sets
        fig, ax = plt.subplots(figsize=(6,6))
        for i in range(len(pareto_sets_errs)):
            ax.scatter(pareto_sets_errs.iloc[i][f"{err_met}_liq_density"], pareto_sets_errs.iloc[i][f"{err_met}_surf_tens"], s=100, label="Pareto Points")
        ax.tick_params("both", direction="in", which="both", length=4, labelsize=20, pad=10)
        ax.tick_params("both", which="major", length=8)
        ax.xaxis.set_ticks_position("both")
        ax.yaxis.set_ticks_position("both")
        ax.set_xlabel(rf"{err_met.upper()} $\rho_l$/%", fontsize=32)
        ax.set_ylabel(rf"{err_met.upper()} $\gamma$/%", fontsize=32)
        # ax.set_title(f"Pareto Front for {molec_name}", fontsize=32)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2.0)
        # ax.legend()
        # plt.show()
        return fig

    ###For each molecule, make a plot of the pareto front for IFT iteration data  
    #Get Project
    iter_type = "vle_iters" 
    project = signac.get_project(iter_type)
    #Load class properies for each molecule in the FF
    molec_dict = esolvs.make_dict(mol_names)

    # Save DataFrame of all molecule data for each iteration
    df_all_molec = get_signac_results(project, molec_dict, property_names)
    df_all_molec = save_signac_results(df_all_molec, iter_type, save_csv)

    #Check pareto efficient samples for each molecule to see if there is one with < mapd_le (10)% error in all properties
    all_final_params = find_pareto(df_all_molec, molec_dict, property_names, mapd_le)

    for key, value in all_final_params.items():
        #If there are, we have the final parameters
        if len(value) > 0:
            print(f"{key}: Final parameters:")
            # print(value)
            param_names = list(molec_dict[key].param_names)
            #Make a fxn in utils.plot to plot predictions vs exp data for LD and ST
            dir_name = f"analysis/{key}/{iter_type}"
            os.makedirs(dir_name, exist_ok=True)
            pdf_name = os.path.join(dir_name , f"prop_preds_{mode}.pdf")
            pdf = PdfPages(pdf_name)

            fig2 = make_pareto_plots(key, mode="pareto", err_met="mapd")
            pdf.savefig(fig2, bbox_inches='tight')
            
            for props in ["liq_density", "surf_tens"]:
                best_data = get_best_set_data(key, mode) 
                #Calculate MAE for each property and plot
                best_data = get_best_set_data(key, mode)

                          
                fig = plot_sim_exp(molec_dict[key], best_data, props)
                pdf.savefig(fig, bbox_inches='tight')   # save one figure at a time
            #Make plots of MAE for rho vs gamma  


            pdf.close()


### Get pareto set info for LD iters for each molecule


props = ["liq_density"]
property_names = ["liq_density"]  # Change me as needed
all_names = ["EG","Gly", "MeOH", "DMSO", "DEC", "DMF"]

for mol in all_names:
    mol_names = [mol]
    show_num = 5
    mode_best_worst = "best"
    mode = f"{mode_best_worst}_{show_num}" #"sing", "pareto", or "all"
    mode = "pareto"


    ##############################################################################
    ##############################################################################
    if mode == "all":
        def get_best_set_data(molec_name):
            # Check the analysis folder for analysis/MolName/vle_iters folders
            # Find the highest params-iter-X.csv file
            iter = determine_iter(molec_name)
            all_data = pd.read_csv(f"analysis/{molec_name}/ld_iters/iter-{iter}/results.csv", header = 0, index_col = 0)

            return all_data
    elif mode == "sing":
        def get_best_set_data(molec_name):
            # Check the analysis folder for analysis/MolName/vle_iters folders
            # Find the highest params-iter-X.csv file
            pareto_sets = pd.read_csv(f"analysis/{molec_name}/ld_iters/mse-less10-full.csv", header = 0, index_col = 0)
            all_data = pd.read_csv(f"analysis/{molec_name}/ld_iters/all_results.csv", header = 0, index_col = 0)
            #Get the row where the mapd_surf_tens column is lowest
            best_row = pareto_sets.loc[pareto_sets['mapd'].idxmin()]
            #Return the array of all parameters (ignore mapd columns)
            param_set = best_row.drop(labels=[col for col in best_row.index if col in ["mse", "mpd", "mapd"]])
            #Find the final parameters with the lowest surface tension
            param_set = pd.DataFrame(param_set).T
            mask = (all_data[param_set.columns] == param_set.iloc[0]).all(axis=1)

            #Apply mask
            all_data = all_data[mask]
            all_data = all_data.sort_values(by='temperature', ascending=True)
            all_data = pd.DataFrame(all_data)
            return all_data
        
    else:
        def get_best_set_data(molec_name, mode = mode_best_worst):
            # Check the analysis folder for analysis/MolName/vle_iters folders
            # Find the highest params-iter-X.csv file
            molec_dict = esolvs.make_dict(mol_names)
            molecule = molec_dict[molec_name] 
            #Map exp data to density
            all_data = pd.read_csv(f"analysis/{molec_name}/ld_iters/all_results.csv", header = 0, index_col = 0)
            ld_threshold = (min(list(molecule.expt_liq_density.values()))+ max(list(molecule.expt_vap_density.values()))) / 2
            df_all, df_liquid, df_vapor = prepare_df_props(all_data, molecule, ld_threshold, scale=False)
                        #Calculate MAPD between expt_liq_density and liq_density for each param set, 
            # Prepare error data to find pareto points
            pareto_sets = prepare_df_errors(df_liquid, molec_dict, molec_name)
            prop_sort = "mapd_" + props
            if mode == "worst":
                top_sets = pareto_sets.sort_values(by=prop_sort).tail(show_num)
            else:
                top_sets = pareto_sets.sort_values(by=prop_sort).head(show_num)
            
            # Drop non-parameter columns
            exclude = ["mse", "mpd", "mapd", "mae"]

            param_cols = [c for c in top_sets.columns if not any(substr in c for substr in exclude)]
            top_sets_params = top_sets[param_cols]

            # Build a mask: True if a row in all_data matches ANY of the top 15 rows
            mask = np.zeros(len(all_data), dtype=bool)
            for _, row in top_sets_params.iterrows():
                mask |= (all_data[param_cols] == row).all(axis=1)

            # Filter
            all_data = all_data[mask].sort_values(by='temperature', ascending=True)
            all_data = pd.DataFrame(all_data)
            return all_data
            
    # #Set iters to analyze and properties to analyze
    property_names = ["liq_density"]  # Change me as needed
    # Set seeds and preferences
    dist_seed = 1  # Distance seed
    mse_less_10_thresh = 25
    save_csv = False



    ##############################################################################
    ##############################################################################
    # Get Project
    iter_type = "ld_iters"
    project = signac.get_project(iter_type)
    molec_dict = esolvs.make_dict(mol_names)

    # Save DataFrame of all molecule data for each iteration
    df_all_molec = get_signac_results(project, molec_dict, property_names)
    # df_all_molec = save_signac_results(df_all_molec, iter_type, True)

    # Check the MSE of the new samples
    mse_less10 = check_mse_10(
        df_all_molec, molec_dict, mse_less_10_thresh, dist_seed, save_csv)

    # Find the next samples to run if fewer than 25 samples have MSE less than 10
    for key, value in mse_less10.items():
        #If there are, we have the final parameters
        if len(value) > 0:
            for props in ["liq_density"]:
                best_data = get_best_set_data(key)
                param_names = list(molec_dict[key].param_names)
                #Make a fxn in utils.plot to plot predictions vs exp data for LD and ST
                dir_name = f"analysis/{key}/{iter_type}/"
                os.makedirs(dir_name, exist_ok=True)
                pdf_name = os.path.join(dir_name , f"prop_preds_{mode}.pdf")
                pdf = PdfPages(pdf_name)

                # for (param_vals), group_df in best_data.groupby(param_names):
                fig = plot_sim_exp(molec_dict[key], best_data, props)
                pdf.savefig(fig, bbox_inches='tight')   # save one figure at a time

            pdf.close()