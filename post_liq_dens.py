import signac
import sys

from utils.analyze_iters import save_signac_results, find_new_samples, find_pareto

#Set iters to analyze and properties to analyze
iters = [1]  # Change me as needed
property_names = ["density", "surf_tens"]  # Change me as needed

#Set seeds and preferences
cl_shuffle_seed = 1  # classifier
gp_shuffle_seed = 42  # GP seed
dist_seed = 1  # Distance seed
save_csv = False
save_fig = False

##############################################################################
##############################################################################
#Get Project
base_name = "density_iters/runs/"
project = signac.get_project(base_name)

# Save DataFrame of all molecule data for each iteration
df_all_molec = save_signac_results(project, property_names, save_csv)
next_samples = find_new_samples(df_all_molec, verbose = True, save_fig=False, cl_shuffle_seed = 1, gp_shuffle_seed = 42, dist_seed = 1)
if max(iters) > 1:
    all_final_params = find_pareto(df_all_molec)
    for key, value in all_final_params.items():
        if len(value) > 0:
            print(f"Final parameters for {key}:")
            print(value)
        else:
            print(f"No final parameters found for {key}. Move to iteration {max(iters) + 1}")
