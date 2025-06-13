import signac
import os
import sys
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import pickle

# sys.path.append("../")

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

    Returns
    -------
    all_data_dict : dict
        Dictionary of all dataframes for each molecule
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
        if mol_name in list(data_dict.keys()):
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
                            # print(f"Job failed: {job.id}")
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
        else:
            print(f"Warning: {mol_name} not found in data_dict. Skipping.")

    return all_data_dict

def save_signac_results(all_data_dict, iter_type = "ld_iters", save_csv=True):
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