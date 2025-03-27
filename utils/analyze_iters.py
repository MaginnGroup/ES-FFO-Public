import signac
import sys
import pandas as pd
import numpy as np
from fffit.signac import save_signac_results
from utils.r41 import R41Constants


# Load class properies for each training and testing molecule
R41 = r41.R41Constants()

molec_dict = {
    "R41": R41,
}


def _get_molec_dicts():
    # Load class properies for each molecule
    from utils.molec_class_files import r41  # import all the class files

    R41 = r41.R41Constants()

    # Create a dictionary with all of the data
    molec_dict = {
        "R41": R41,
    }
    return molec_dict


def _get_class_from_molecule(molecule_name):
    molec_dict = _get_molec_dicts()
    return {molecule_name: molec_dict[molecule_name]}


def save_signac_results(project, property_names, csv_name=None):
    """Save the signac results to a CSV file.

    Parameters
    ----------
    project : signac.Project
        signac project to load
    property_names : set
        set of property names
    csv_name : string
        name of csv file to save results
    """
    df_all_molec = {}

    if type(property_names) not in (list, tuple):
        raise TypeError("property_names must be a list or tuple")

    # From Project, group by molecule name
    molec_groupby = project.groupby("mol_name")
    for molec, molec_group in molec_groupby:
        # Get the parameter names from utils/molec_class_files/
        molec_dict = _get_class_from_molecule(molec)
        param_names = molec_dict[molec].param_names
        job_groupby = param_names  # tuple(param_names)
        property_names = tuple(property_names)

        print(f"Extracting the following properties: {property_names}")

        # Store data here before converting to dataframe
        data = []

        # Loop over all jobs in project and group by parameter sets
        for params, job_group in molec_group.groupby(job_groupby):
            for job in job_group:
                # Extract the parameters into a dict
                new_row = {name: param for (name, param) in zip(job_groupby, params)}

                # Extract the temperature for each job.
                # Assumes temperature increments >= 1 K
                temperature = round(job.sp.T)
                new_row["temperature"] = temperature

                job_fail_stat = False
                # Extract property values. Insert N/A if not found
                for property_name in property_names:
                    try:
                        property_ = job.doc[property_name]
                        new_row[property_name] = property_
                    except KeyError:
                        job_fail_stat = True
                        new_row[property_name] = np.nan
                if job_fail_stat:
                    print(
                        f"Job {job.id} in project {project} failed. Molecule {job.sp.mol_name} at T = {temperature} K."
                    )

                data.append(new_row)

        # Save to csv file for record-keeping
        df = pd.DataFrame(data)
        sortby_list = list(param_names) + ["temperature"]
        # sort by parameter, and temperature
        df.sort_values(
            by=sortby_list,
            ignore_index=True,
            inplace=True,
        )

        df_all_molec[molec] = df

        if csv_name != None:
            # Save CSV to a molecules folder in analysis
            df.to_csv(csv_name)

    return df


def extract_density(project_name, criteria_dict):
    # Get project path
    project = signac.get_project(project_name)

    run_path = "/scratch365/mcarlozo/HFC-FFO/r41/run/"
    itername = "r41-density-iter" + str(iternum)
    project_path = run_path + itername
    csv_name = "csv/" + itername + "-results.csv"

    property_names = ["density", "surf_ten"]
    project = signac.get_project(project_path)

    save_signac_results(project, R41.param_names, property_names, csv_name)
