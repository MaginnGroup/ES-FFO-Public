import numpy as np
import pandas as pd

def save_signac_results(project, param_names, property_names, csv_name = None):
    """Save the signac results to a CSV file.

    Parameters
    ----------
    project : signac.Project
        signac project to load
    param_names : set
        molecule name
    property_names : set
        set of property names
    csv_name : string
        name of csv file to save results

    Returns
    -------
    df : pandas.DataFrame
        DataFrame of all data for each molecule
    """
    # if type(param_names) not in (list, tuple):
    #     raise TypeError("param_names must be a list or tuple")
    if type(property_names) not in (list, tuple):
        raise TypeError("property_names must be a list or tuple")

    job_groupby = param_names #tuple(param_names)
    property_names = tuple(property_names)

    print(f"Extracting the following properties: {property_names}")

    # Store data here before converting to dataframe
    data = []

    # Loop over all jobs in project and group by parameter sets
    for params, job_group in project.groupby(job_groupby):

        for job in job_group:
            
            # Extract the parameters into a dict
            # new_row = {
            #     name: param for (name, param) in zip(job_groupby, params)
            # }
            new_row = {"molecule": job.sp.mol_name}

            # Extract the temperature for each job.
            # Assumes temperature increments >= 1 K
            temperature = round(job.sp.T)
            new_row["temperature"] = temperature
            if hasattr(job.sp, "restart"):
                new_row["restart"] = job.sp.restart
            else:
                new_row["restart"] = 1

            job_fail_stat = False
            # Extract property values. Insert N/A if not found
            for property_name in property_names:
                if "gemc_failed" in job.doc.keys() and job.doc.gemc_failed == True:
                    job_fail_stat = True
                    new_row[property_name] = np.nan
                else:
                    try:
                        property_ = job.doc[property_name]
                        new_row[property_name] = property_
                    except KeyError:
                        job_fail_stat = True
                        new_row[property_name] = np.nan
            if job_fail_stat:
                print(f"Job {job.id} in project {project} failed. Molecule {job.sp.mol_name} at T = {temperature} K.")

            data.append(new_row)

    # Save to csv file for record-keeping
    df = pd.DataFrame(data)

    #sort by molecule and temperature -- added by Ning Wang
    df.sort_values(by=["molecule", "temperature", "restart"], ignore_index=True, inplace=True)
    #Filer out temperatures which only have one restart
    df = df.groupby(["molecule", "temperature"]).filter(lambda x: len(x) > 1)

    if csv_name != None:
        df.to_csv(csv_name)

    return df
