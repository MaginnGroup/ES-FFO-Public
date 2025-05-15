# Imports
from utils.molec_class_files import r14, r32, r50, r125, r134a, r143a, r170
from utils import atom_type, opt_atom_types
import numpy as np
import unyt as u
import pandas as pd
import os
import copy
import scipy
import signac
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Set params for what you want to analyze
save_data = True  # Data to save
obj_choice = "ExpVal"  # Objective to consider
at_number = 6  # atom type to consider
seed = 1  # Seed to use
molec_names = [
    "R14",
    "R32",
    "R50",
    "R170",
    "R125",
    "R134a",
    "R143a",
    "R41",
]  # Training data to consider

# Get best_run data saved in one csv from all jobs
project = signac.get_project("opt_at_params")
filtered_jobs = project.find_jobs({"obj_choice": obj_choice, "atom_type": at_number})
grouped_jobs = filtered_jobs.groupby("training_molecules")
for statepoint_value, group in grouped_jobs:
    # print(statepoint_value, group)
    unsorted_df = None
    save_path = None
    for i, job in enumerate(group):
        # If the best run file exists
        if os.path.exists(job.fn("best_run.csv")):
            # For each group of training molecules, get the first job to get the path to the directory
            if save_path is None:
                save_path = job.document.dir_name

            # Read the file and concatenate the data
            df_best_run = pd.read_csv(job.fn("best_run.csv"), header=0, index_col=False)
            # On the 1st iteration where we have data, create the df
            if unsorted_df is None:
                unsorted_df = df_best_run
            # Otherwise append to it
            else:
                unsorted_df = pd.concat([unsorted_df, df_best_run], ignore_index=True)

    if unsorted_df is not None:
        # Sort the data by the minimum objective value
        all_df = unsorted_df.sort_values(by="Min Obj", ascending=True).reset_index(
            drop=True
        )
        # Save all the best sets in appropriate folder for each set of training molecules
        all_df.to_csv(
            os.path.join(save_path, "best_per_run.csv"), index=False, header=True
        )


# Create visualization object
visual = opt_atom_types.Vis_Results(molec_names, at_number, seed, obj_choice)
# Set parameter set of interest (in this case get the best parameter set)
x_label = "best_set"
all_molec_dir = visual.use_dir_name
path_best_sets = os.path.join(all_molec_dir, "best_per_run.csv")
assert os.path.exists(path_best_sets), "best_per_run.csv not found in directory"
all_df = pd.read_csv(path_best_sets, header=0)
first_param_name = visual.at_class.at_names[0] + "_min"
last_param_name = visual.at_class.at_names[-1] + "_min"
all_sets = all_df.loc[:, first_param_name:last_param_name].values
unique_best_sets = visual.get_unique_sets(
    all_sets, save_data=save_data, save_label=x_label
)

# Loop over unique parameter sets
for i in range(unique_best_sets.shape[0]):
    x_label_set = x_label + "_" + str(i + 1)
    best_set = unique_best_sets.iloc[i, :].values
    best_real = visual.values_pref_to_real(copy.copy(best_set))

    # Get Property Predictions for all training molecules
    molec_names_all = list(visual.all_train_molec_data.keys())
    visual.comp_paper_full_ind(molec_names_all, save_label=x_label_set)

    # Calculate MAPD for predictions and save results
    df = visual.calc_MAPD_best(
        molec_names_all,
        theta_guess=best_real,
        save_data=save_data,
        save_label=x_label_set,
    )


for i in range(unique_best_sets.shape[0]):
    x_label_set = x_label + "_" + str(i + 1)
    best_set = unique_best_sets.iloc[i, :].values
    best_real = visual.values_pref_to_real(copy.copy(best_set))
    # Gat Jac and Hess Approximations
    scale_theta = True
    jac = visual.approx_jac(best_real, scale_theta, save_data, x_label=x_label_set)
    hess = visual.approx_hess(best_real, scale_theta, save_data, x_label=x_label_set)
    eigval, eigvec = scipy.linalg.eig(hess)
    if save_data == True:
        eig_val_path = os.path.join(
            all_molec_dir / "hess_approx", "EigVals_" + x_label_set
        )
        eig_vec_path = os.path.join(
            all_molec_dir / "hess_approx", "EigVecs_" + x_label_set
        )
        eigval = [np.real(num) for num in eigval]
        np.savetxt(eig_val_path, eigval, delimiter=",")
        np.savetxt(eig_vec_path, eigvec, delimiter=",")
"""
    # Plot optimization result heat maps
    visual.plot_obj_hms(best_set, x_label_set)

# Plot atom_type scheme results
# at_schemes = [11,12,13,14]
# if len(at_schemes) > 1 and isinstance(at_schemes, (list,np.ndarray)):
#     at_str = '-'.join(at_schemes.sort())
# else:
#     at_str = at_schemes[0]
# pdf = PdfPages('Results/at_schemes_' + at_str + '.pdf')
# pdf.savefig(visual.plot_at_MSE(molec_names, at_schemes), bbox_inches='tight')
# #Close figures
# plt.close()
# pdf.close()
"""
