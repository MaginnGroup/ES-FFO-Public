import signac
import os
import sys
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import pickle

# sys.path.append("../")
from utils.prep_ms_data import (
    prepare_df_props
)
from utils.id_pareto import prepare_df_errors, select_final_pareto
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
        df_paramsets = prepare_df_errors(df_all, mol_name)
        #Save data to csv
        dir_name = root_dir_vle + "iter-" + str(iter_num) + "/"
        os.makedirs(dir_name, exist_ok=True)
        csv_name = os.path.join(dir_name, "result_errors.csv")
        df_paramsets.to_csv(csv_name)


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



def select_final_pareto(df_pareto, root_dir, iter_num):
    # Filter for parameter sets with less than 5 % error in all properties
    df_final = df_pareto.drop(
        columns=[
            "sim_liq_density",
            "sim_surf_tens",
            "mse_surf_tens",
            "mse_liq_density",
            "mae_surf_tens",
            "mae_liq_density",
            "is_pareto",
        ]
    )

    ### Choosing Final Parameter Sets (R-32)
    # Filter for parameter sets with less than 5 % error in all properties
    df_final = df_final[
        (df_final["mape_surf_tens"] <= 5.0)
        & (df_final["mape_liq_density"] <= 5.0)
    ]

    # Save CSV files
    dir_name = root_dir + "iter-" + str(iter_num) + "/"
    os.makedirs(dir_name, exist_ok=True)
    csv_name = os.path.join(dir_name, "final-params.csv")
    df_final.to_csv(csv_name)
    csv_name = root_dir + "final-params.csv"
    df_final.to_csv(csv_name)

    return df_final