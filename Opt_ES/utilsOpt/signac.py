import signac
import os
import sys
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import pickle
from .opt_atom_types import Problem_Setup
#Import opt_atom_types
# sys.path.append("../")

def get_signac_results(project_dict, data_dict):
    """Save the signac results to a CSV file.

    Parameters
    ----------
    project_dict : dictionary of {name: [signac.Project, list(prop_names)]}
        signac projects to load and property names to pull from each
    data_dict : dictionary
        dictionary of molecule names and data from esolvs.py
    prop_names : list
        list of property names to extract from this project
    save_csv : bool, default True
        Whether to save the results to a CSV file

    Returns
    -------
    all_data_dict : dict
        Dictionary of all dataframes for each molecule
    """
    # project_names = list(project_dict.keys())
    project_dict = {key: value for key, value in project_dict.items() if value[0] is not None}
    project_names = list(project_dict.keys())
    # property_names = list(project_dict.values())

    data_dicts = {}
    for key, values in project_dict.items():
        project, property_names = values
        # Group by molecules, atom type, and restart
        job_groupby = ["mol_name", "atom_type", "restart", "obj_choice"]

        print(f"Extracting the following properties: {property_names}")

        #Creat dict to store data for each molecule and atom type
        all_data_dict = {} #file_save, df

        project_df = project.to_dataframe()#.sort_values(by=[job_groupby])
        project_df.columns = [col.replace('sp.', '') for col in project_df.columns]
        project_df["job"] = project_df.index
        project_df.reset_index(drop=True, inplace=True)

        # Loop over all jobs in project and group by mol name, at number, restart, and obj choice
        for (mol_name, at_num, restart, obj_choice), job_group in project_df.groupby(job_groupby):
            train_mol_str = job_group["train_mol_str"].values[0]
            train_mols = train_mol_str.split("-") if "-" in train_mol_str else [train_mol_str]
            if mol_name in list(data_dict.keys()):
                data = [] # Store data here before converting to dataframe
                #Make atom type setup
                setup = Problem_Setup(train_mols, at_num, obj_choice)
                param_bounds, param_names = setup.get_param_bnds_names()
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
                                # print(f"Job failed: {job.id}")
                                new_row[property_name] = np.nan
                            try:
                                property_unc = job.doc[property_name + "_unc"]
                                new_row[property_name + "_unc"] = property_unc
                            except KeyError:
                                # print(f"Job failed: {job.id}")
                                new_row[property_name + "_unc"] = np.nan
                        
                        data.append(new_row)

                #Create data from dict
                df = pd.DataFrame(data)

                df["restart"] = restart
                df["molecule"] = mol_name

                #Add data to all_data_dict
                # If the data directory is already in the dictionary, concatenate the dataframes
                dir_name = os.path.join(setup.use_dir_name, "ms_val")
                if dir_name in all_data_dict:
                    all_data_dict[dir_name] = pd.concat([all_data_dict[dir_name], df])
                # If the data directory is not in the dictionary, add the dataframe
                else:
                    all_data_dict[dir_name] = df
                
            else:
                print(f"Warning: {mol_name} not found in data_dict. Skipping.")
        data_dicts[key] = all_data_dict

    #Merge dfs in data_dicts for which the files are the same
    all_data_dict = {}
    #Get data dicts for each project
    list_data_dicts = list(data_dicts.values())
    
    # Find common files between projects
    if len(list_data_dicts) == 1:
        common_files = set(list_data_dicts[0].keys())
    elif len(list_data_dicts) > 1:
        common_files = set.intersection(*(set(d.keys()) for d in list_data_dicts))

    #For each file, merge the dataframes from each project
    for file in common_files:
        dfs_to_merge = [d[file] for d in list_data_dicts]
        merged_df = dfs_to_merge[0]
        if len(dfs_to_merge) > 1:
            for df in dfs_to_merge[1:]:
                merged_df = pd.merge(merged_df, df, how="outer")
        all_data_dict[file] = merged_df

    # gemc_dict = data_dict(project_names[0])
    # ift_dict = data_dict(project_names[1])

    # common_files = set(ift_dict.keys()) & set(gemc_dict.keys())
    # for file in common_files:
    #     df1, df2 = ift_dict[file], gemc_dict[file]
    #     all_data_dict[file] = pd.merge(df1, df2, how="outer")

    return all_data_dict

def save_signac_results(all_data_dict, save_csv=True):
    """
    Save the signac results to a CSV file.

    Parameters
    ----------
    all_data_dict : dict
        Dictionary of all dataframes for each molecule
    iter_type : str
        Type of iteration to save
    save_csv : bool, default True
        Whether to save the results to a CSV file

    Returns
    -------
    all_data_dict : dict
        Dictionary of all dataframes for each molecule
    """
    #Loop over all molecules
    for data_dir, all_data_df in all_data_dict.items():
        #Save all data to one file for easy access
        os.makedirs(data_dir, exist_ok=True)
        if save_csv:
            csv_name_all = f"{data_dir}/raw_data.csv"
            all_data_df.to_csv(csv_name_all)

    return all_data_dict