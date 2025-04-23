import signac
import sys
import pandas as pd
import numpy as np
from utils.id_new_samples import (
    prep_df_density,
    build_classifier,
    fit_gp_model,
    rank_vl_samples,
    vis_top_samples,
    get_next_iter_params,
    classify_samples,
    prepare_df_density,
)
from utils.molec_class_files import esolvs
from utils.id_pareto import prepare_df_dens_errors, select_final_pareto

sys.path.append("../")
from fffit.fffit.pareto import find_pareto_set, is_pareto_efficient

def save_signac_results(project, data_dict, prop_names, save_csv=True):
    """Save the signac results to a CSV file.

    Parameters
    ----------
    projects : list of signac.Project
        signac projects to load
    data_dict : dictionary
        dictionary of molecule names and data from esolvs.py
    prop_names : set
        set of property names
    save_csv : bool, default True
        Whether to save the results to a CSV file
    """
    if type(param_names) not in (list, tuple):
        raise TypeError("param_names must be a list or tuple")
    if type(property_names) not in (list, tuple):
        raise TypeError("property_names must be a list or tuple")

    # Group by project_name and molecules
    job_groupby = tuple(("mol_name", "dens-iter"))
    property_names = tuple(property_names)
    
    print(f"Extracting the following properties: {property_names}")  

    all_data_dict = {}

    # Loop over all jobs in project and group by mol name and density iter
    for (mol_name, dens_iter), job_group in project.groupby(job_groupby):
        data = [] # Store data here before converting to dataframe
        # Get the unique param sets for each molecule
        param_names = data_dict[mol_name].param_names
        #Loop over each parameter set in the group
        for param_vals, job_group_params in job_group.groupby(param_names):
            #Loop over all jobs (temperatures) in the group
            for job in job_group_params:
                # Extract the parameters into a dict
                new_row = {
                    name: param for (name, param) in zip(param_names, param_vals)
                }

                # Extract the temperature for each job.
                # Assumes temperature increments >= 1 K
                temperature = round(job.sp.T)
                new_row["temperature"] = temperature

                # Extract property values. Insert N/A if not found
                for property_name in property_names:
                    try:
                        property_ = job.doc[property_name]
                        new_row[property_name] = property_
                    except KeyError:
                        print(f"Job failed: {job.id}")
                        new_row[property_name] = np.nan
                
                data.append(new_row)

        #Create data from dict
        df = pd.DataFrame(data)

        #Add data to all_data_dict
        # If the molecule name is already in the dictionary, concatenate the dataframes
        if mol_name in all_data_dict:
            all_data_dict[mol_name] = pd.concat([all_data_dict[mol_name], df])
        # If the molecule name is not in the dictionary, add the dataframe
        else:
            all_data_dict[mol_name] = df

        # Save to csv file for record-keeping
        if save_csv:
            csv_name = "density_iters/analysis/" +  mol_name  + "/results-iter-" + str(dens_iter) + ".csv"
            df.to_csv(csv_name)
    
    # Save all data to a single CSV file
    for mol_name, data in all_data_dict.items():
        if save_csv:
            # Save each molecule data to a separate CSV file
            csv_name = "density_iters/analysis/" + mol_name + "/all_results.csv"
            data.to_csv(csv_name)

    return all_data_dict

def find_new_samples(all_df_data, verbose = True, save_fig=False, cl_shuffle_seed = 1, gp_shuffle_seed = 42, dist_seed = 1):
    #Loop over all molecules:
    next_iter_params_all = {}
    for mol_name, data in all_df_data.items():

        df_csv = all_df_data[mol_name]
        iter_num = df_csv["dens-iter"].max()

        ### Step 1: Prepare df_density
        df_iter1, df_liquid, root_dir = prep_df_density(mol_name, data, df_csv)

        ### Step 2: Fit classifier and GP models
        classifier = build_classifier(df_iter1, root_dir, data, cl_shuffle_seed, verbose, save_fig)

        ### Fit GP Model
        models = fit_gp_model(df_liquid, data, gp_shuffle_seed=gp_shuffle_seed)

        ### Step 3: Find new parameters for MD simulations
        # SVM to classify hypercube regions as liquid or vapor
        latin_hypercube = np.genfromtxt(
            root_dir + "LHS_500000.csv",
            delimiter=",",
            skip_header=1,
        )[:, 1:]
        liquid_samples, vapor_samples = classify_samples(latin_hypercube, classifier)
        top_liquid_samples, top_vapor_samples = rank_vl_samples(liquid_samples, vapor_samples, models, data, verbose)
        

        #### Find and Visualize Low MSE parameter sets
        top_liq, top_vap = vis_top_samples(top_liquid_samples, top_vapor_samples, data, root_dir, iter_num, save_fig)

        #### Get next set of 200 samples
        target_total = 200
        next_iter_params, final_sample_file = get_next_iter_params(top_liq, top_vap, data, root_dir, iter_num, target_total, dist_seed, verbose)
        next_iter_params.to_csv(final_sample_file)
        next_iter_params_all[mol_name] = next_iter_params
    return next_iter_params_all

def find_pareto(all_df_data):
    #Loop over all molecules:
    all_final_params = {}
    for mol_name, data in all_df_data.items():
        root_dir = "density-iters/analysis/" + mol_name + "/"
        #Get all data from last iteration
        df_csv = all_df_data[mol_name]
        iter_num = df_csv["dens-iter"].max()
        #Get only data from the last iteration
        df_this_iter = df_csv[df_csv["dens-iter"] == iter_num]

        #Prepare error data to find pareto points
        df_paramsets = prepare_df_dens_errors(df_this_iter, mol_name, root_dir, iter_num)
        result, pareto_points, dominated_points = find_pareto_set(
            df_paramsets.filter(["mse_liq_density", "mse_surf_tens"]).values,
            is_pareto_efficient
        )
        
        df_paramsets = df_paramsets.join(pd.DataFrame(result, columns=["is_pareto"]))

        file_name = root_dir + "dens-iter-" + str(iter_num) + f"-pareto-params.csv"
        df_paramsets[df_paramsets["is_pareto"] == True].to_csv(file_name)

        df_final = select_final_pareto(df_paramsets, root_dir, iter_num)
    all_final_params[mol_name] = df_final
    return all_final_params

def plot_gp_examples(all_df_data, next_iter_params_all):
    #Get all data
    next_iter_params_all = {}
    for mol_name, data in all_df_data.items():
        df_csv = all_df_data[mol_name]
        iter_num = df_csv["dens-iter"].max()
        ld_threshold = data.expt_rhoc
        df_all, df_liq, df_vap = prepare_df_density(df_csv, data, ld_threshold)
