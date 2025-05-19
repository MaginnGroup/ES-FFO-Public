import signac
import os
import sys
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import pickle

# sys.path.append("../")
from utils.id_new_samples import (
    prep_df_density,
    build_classifier,
    rank_vl_samples,
    vis_top_samples,
    get_next_iter_params,
    classify_samples,
    rank_vle_samples, 
    get_next_vle_params, 
    prepare_df_props
)
from utils.id_pareto import prepare_df_errors, select_final_pareto
from utils.plotfig_gp_examples import plot_gp_slices, plot_test_sets, get_exp_data, plot_model_performance, get_prop_best_model
from fffit.fffit.pareto import find_pareto_set, is_pareto_efficient

def get_signac_results(project, data_dict, prop_names):
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
    if type(prop_names) not in (list, tuple):
        raise TypeError("prop_names must be a list or tuple")

    # Group by project_name, molecules, and iteration
    job_groupby = ["mol_name", "iter"]
    property_names = prop_names
    
    print(f"Extracting the following properties: {property_names}")  

    all_data_dict = {}

    project_df = project.to_dataframe()#.sort_values(by=[job_groupby])
    project_df.columns = [col.replace('sp.', '') for col in project_df.columns]
    project_df["job"] = project_df.index
    project_df.reset_index(drop=True, inplace=True)

    # Loop over all jobs in project and group by mol name and density iter
    for (mol_name, dens_iter), job_group in project_df.groupby(job_groupby):
        data = [] # Store data here before converting to dataframe
        # Get the unique param sets for each molecule
        param_names = list(data_dict[mol_name].param_names)
        
        #Loop over each parameter set in the group
        for (param_vals), group_df in job_group.groupby(param_names):
            for row in range(len(group_df)):
                new_job = group_df.sort_values(by=["T"]).iloc[row]
                job = project.open_job(id=new_job["job"])
                # Extract the parameters into a dict
                new_row = {
                    name: param for (name, param) in zip(param_names, param_vals)
                }
                # Extract the temperature for each job.
                # Assumes temperature increments >= 1 K
                temperature = job.sp.T
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
        df["iter"] = dens_iter

        #Add data to all_data_dict
        # If the molecule name is already in the dictionary, concatenate the dataframes
        if mol_name in all_data_dict:
            all_data_dict[mol_name] = pd.concat([all_data_dict[mol_name], df])
        # If the molecule name is not in the dictionary, add the dataframe
        else:
            all_data_dict[mol_name] = df

    return all_data_dict

def save_signac_results(all_data_dict, iter_type = "ld_iters", save_csv=True):
    #Loop over all molecules
    for mol_name, all_data_df in all_data_dict.items():
        #Save all data to one file for easy access
        if save_csv:
            csv_name_all = f"analysis/{mol_name}/{iter_type}/all_results.csv"
            all_data_df.to_csv(csv_name_all)
        #Group by iteration number
        grouped = all_data_df.groupby("iter")
        for iter, group_df in grouped:
            #Remove the iter column from the group_df
            group_df = group_df.drop(columns=["iter"])
            # Save to csv file for record-keeping
            if save_csv:
                dir_name = f"analysis/{mol_name}/{iter_type}/iter-{str(iter)}"
                # Create the directory if it doesn't exist
                os.makedirs(dir_name, exist_ok=True)
                csv_name = os.path.join(dir_name , "results.csv")
                group_df.to_csv(csv_name)
    return all_data_dict

def new_samples_vle(all_df_data, data_dict, verbose = True, save_fig=False, gp_shuffle_seed = 42, dist_seed = 1):

    max_mse = 25**2 #(kg/m^3)^2
    target_total = 25
    #Loop over all molecules:
    next_iter_params_all = {}
    for mol_name, df_csv in all_df_data.items():
        data = data_dict[mol_name]
        # df_csv = all_df_data[mol_name]
        iter_num = df_csv["iter"].max()
        iter_type = "vle_iters"
        ld_threshold = 0
        root_dir = f"analysis/{mol_name}/"

        ### Step 1: Prepare df_density
        df_all, df_liquid, df_vapor = prepare_df_props(df_csv, data, ld_threshold)

        #Get LD root directory
        root_dir_vle = os.path.join(root_dir, iter_type)

        ### Fit GP Model
        path_gps = f"{root_dir_vle}/iter-{str(iter_num)}"

        models_best, all_models, dir_train_test = get_prop_best_model(df_liquid, data, path_gps, gp_shuffle_seed)

        ### Step 3: Find new parameters for MD simulations
        # SVM to classify hypercube regions as liquid or vapor
        LHS_file = os.path.join(root_dir, "LHS_500000.csv")
        latin_hypercube = np.genfromtxt(
            LHS_file,
            delimiter=",",
            skip_header=1,
        )[:, 1:]
        
        
        #### Find Low MSE parameter sets
        vle_samples = rank_vle_samples(latin_hypercube, models_best, data, verbose)
        result, pareto_points, dominated_points = find_pareto_set(
        vle_samples.drop(columns=list(data.param_names)).values, is_pareto_efficient)
        vle_samples = vle_samples.join(pd.DataFrame(result, columns=["is_pareto"]))

        pareto_points = vle_samples[vle_samples["is_pareto"] == True]

        # Get the best row for each property from pareto_points
        mse_columns = [col for col in pareto_points.columns if "mse" in col]
        best_vals = [pareto_points.sort_values(col).iloc[[0]] for col in mse_columns]
        new_points = pd.concat(best_vals)

        # Drop the selected points from pareto_points to ensure new samples are selected
        pareto_points.drop(index=new_points.index, inplace=True)
        print(f"A total of {len(pareto_points)} pareto efficient points were found.")

        next_iter_params, final_sample_file = get_next_vle_params(pareto_points, data, root_dir_vle, iter_num, target_total, dist_seed, verbose)
        next_iter_params.to_csv(final_sample_file)
        next_iter_params_all[mol_name] = next_iter_params
    return next_iter_params_all 


def find_new_samples(all_df_data, data_dict, verbose = True, save_fig=False, cl_shuffle_seed = 1, gp_shuffle_seed = 42, dist_seed = 1):
    #Loop over all molecules:
    next_iter_params_all = {}
    for mol_name, df_csv in all_df_data.items():
        data = data_dict[mol_name]
        # df_csv = all_df_data[mol_name]
        iter_num = df_csv["iter"].max()
        iter_type = "ld_iters"

        ### Step 1: Prepare df_density
        df_iter1, df_liquid, root_dir = prep_df_density(mol_name, data, df_csv)

        #Get LD root directory
        root_dir_ld = os.path.join(root_dir, iter_type)
        ### Step 2: Fit classifier and GP models
        classifier = build_classifier(df_iter1, root_dir_ld, data, cl_shuffle_seed, verbose, save_fig)

        ### Fit GP Model
        path_gps = f"{root_dir_ld}/iter-{str(iter_num)}"
        models_best, all_models, dir_train_test = get_prop_best_model(df_liquid, data, path_gps, gp_shuffle_seed)

        ### Step 3: Find new parameters for MD simulations
        # SVM to classify hypercube regions as liquid or vapor
        LHS_file = os.path.join(root_dir, "LHS_500000.csv")
        latin_hypercube = np.genfromtxt(
            LHS_file,
            delimiter=",",
            skip_header=1,
        )[:, 1:]
        liquid_samples, vapor_samples = classify_samples(latin_hypercube, classifier)
        top_liquid_samples, top_vapor_samples = rank_vl_samples(liquid_samples, vapor_samples, models_best, data, verbose)

        #### Find and Visualize Low MSE parameter sets
        top_liq, top_vap = vis_top_samples(top_liquid_samples, top_vapor_samples, data, root_dir_ld, iter_num, save_fig)

        #### Get next set of 200 samples
        target_total = 200
        next_iter_params, final_sample_file = get_next_iter_params(top_liq, top_vap, data, root_dir_ld, iter_num, target_total, dist_seed, verbose)
        next_iter_params.to_csv(final_sample_file)
        next_iter_params_all[mol_name] = next_iter_params
    return next_iter_params_all

def find_pareto(all_df_data, data_dict):
    #Loop over all molecules:
    all_final_params = {}
    for mol_name, df_csv in all_df_data.items():
        root_dir = f"analysis/{mol_name}/"
        root_dir_vle = os.path.join(root_dir, "vle_iters")
        #Get all data from last iteration
        # df_csv = all_df_data[mol_name]
        iter_num = df_csv["iter"].max()
        ld_threshold = 0
        data = data_dict[mol_name]

        #Get all result data
        df_all, df_liquid, df_vapor = prepare_df_props(df_csv, data, ld_threshold, scale=False)
        
        #Prepare error data to find pareto points
        df_paramsets = prepare_df_errors(df_all, mol_name, root_dir, iter_num)
        mse_columns = [col for col in df_paramsets.columns if "mse" in col]
        result, pareto_points, dominated_points = find_pareto_set(
            df_paramsets.filter(mse_columns).values,
            is_pareto_efficient)
        
        df_paramsets = df_paramsets.join(pd.DataFrame(result, columns=["is_pareto"]))
        pareto_points = df_paramsets[df_paramsets["is_pareto"] == True]
        print(f"A total of {len(pareto_points)} pareto efficient points were found.")

        # Create the directory if it doesn't exist and store pareto points
        dir_name = os.path.join(root_dir_vle, "iter-" + str(iter_num))
        os.makedirs(dir_name, exist_ok=True)
        file_name = os.path.join(dir_name , "pareto-params.csv")
        pareto_points.to_csv(file_name)

        df_final = select_final_pareto(pareto_points, root_dir, iter_num)
    all_final_params[mol_name] = df_final
    return all_final_params

def get_best_models(all_df_data, data_dict, iter_type = "ld_iters", gp_shuffle_seed = 42, save_fig=False):
    #Get all data
    models_molecs = {}
    for mol_name, df_csv in all_df_data.items():
        data = data_dict[mol_name]
        ld_threshold = (min(list(data.expt_liq_density.values())) + max(list(data.expt_vap_density.values())))/2
        # df_csv = all_df_data[mol_name]
        iter_num = df_csv["iter"].max()

        dir_name = f"analysis/{mol_name}/{iter_type}/iter-{str(iter_num)}"
        os.makedirs(dir_name, exist_ok=True)
        if save_fig:
            pdf_name = os.path.join(dir_name , "fig_gp_examples.pdf")
            pdf = PdfPages(pdf_name)
        else:
            pdf = None

        df_all, df_liq, df_vapor = prepare_df_props(df_csv, data, ld_threshold)

        path_gps = os.path.join(dir_name, "gp_models.pkl")
        models_best, all_models, dir_train_test = get_prop_best_model(df_liq, data, path_gps, gp_shuffle_seed)
            
        models_molecs[mol_name] = models_best

    dir2 = f"analysis/all_mols/{iter_type}/iter-{str(iter_num)}"
    with open(dir2 + "/gp_models.pkl", "wb") as f:
        pickle.dump(models_molecs, f)

    return models_molecs

def plot_gp_examples(all_df_data, data_dict, iter_type = "ld_iters", gp_shuffle_seed = 42, save_fig=False):
    #Get all data
    for mol_name, df_csv in all_df_data.items():
        data = data_dict[mol_name]
        ld_threshold = (min(list(data.expt_liq_density.values())) + max(list(data.expt_vap_density.values())))/2
        # df_csv = all_df_data[mol_name]
        iter_num = df_csv["iter"].max()

        dir_name = f"analysis/{mol_name}/{iter_type}/iter-{str(iter_num)}"
        os.makedirs(dir_name, exist_ok=True)
        pdf_name = os.path.join(dir_name , "fig_gp_examples.pdf")
        pdf = PdfPages(pdf_name)

        df_all, df_liq, df_vapor = prepare_df_props(df_csv, data, ld_threshold)
        path_gps = os.path.join(dir_name, "gp_models.pkl")
        models_best, all_models, dir_train_test = get_prop_best_model(df_liq, data, path_gps, gp_shuffle_seed)
        
        for prop_name, models in all_models.items():
            # Load data
            exp_data, property_bounds, name = get_exp_data(data, prop_name)
            df_x_train = pd.read_csv(os.path.join(dir_train_test, f"{prop_name}_x_train.csv"), header = 1, index_col = False)
            df_y_train = pd.read_csv(os.path.join(dir_train_test, f"{prop_name}_y_train.csv"), header = 1, index_col = False)
            df_x_test = pd.read_csv(os.path.join(dir_train_test, f"{prop_name}_x_test.csv"), header = 1, index_col = False)
            df_y_test = pd.read_csv(os.path.join(dir_train_test, f"{prop_name}_y_test.csv"), header = 1, index_col = False)
            df_x_all = pd.concat([df_x_train, df_x_test], ignore_index=True)
            df_y_all = pd.concat([df_y_train, df_y_test], ignore_index=True)

            #Plot model performance
            plot_model_performance(models, df_x_all, df_y_all, property_bounds, pdf, xylim=None, save_fig=False)
            plot_model_performance(models, df_x_train, df_y_train, property_bounds, pdf, xylim=None, save_fig=False)
            plot_model_performance(models, df_x_test, df_y_test, property_bounds, pdf, xylim=None, save_fig=False)

        for prop_name, models in all_models.items():
            #Plot test sets
            df_x_test = pd.read_csv(os.path.join(dir_train_test, f"{prop_name}_x_test.csv"), header = 1, index_col = False)
            if len(df_x_test) > 0:
                x_test = df_x_test.to_numpy()
                plot_test_sets(models, x_test, df_liq, data, pdf, prop_name)
            #Plot GP slices
            plot_gp_slices(models, data, prop_name, pdf) 

        pdf.close()