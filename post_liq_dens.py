import signac
import sys
from utils.molec_class_files import esolvs
from utils.analyze_iters import save_signac_results, find_new_samples, find_pareto, plot_gp_examples

#Set iters to analyze and properties to analyze
iters = [1]  # Change me as needed
property_names = ["density", "surf_tens"]  # Change me as needed
mol_names = ["EG", "Gly", "ACN", "MeOH", "DMSO", "THF", "DCM", "DEC", "DMF"] # Change me as needed


#Set seeds and preferences
cl_shuffle_seed = 1  # classifier
gp_shuffle_seed = 42  # GP seed
dist_seed = 1  # Distance seed
save_csv = False
save_fig = False
verbose = True


##############################################################################
##############################################################################
#Get Project
base_name = "density_iters/runs/"
project = signac.get_project(base_name)
molec_dict = esolvs.make_dict(mol_names)

# Save DataFrame of all molecule data for each iteration
df_all_molec = save_signac_results(project, molec_dict, property_names, save_csv)
all_prop_models = plot_gp_examples(df_all_molec, molec_dict, gp_shuffle_seed, save_fig)
next_samples = find_new_samples(df_all_molec, molec_dict, verbose, save_fig, cl_shuffle_seed, gp_shuffle_seed, dist_seed)
if max(iters) > 1:
    all_final_params = find_pareto(df_all_molec)
    for key, value in all_final_params.items():
        if len(value) > 0:
            print(f"Final parameters for {key}:")
            print(value)
        else:
            print(f"No final parameters found for {key}. Move to iteration {max(iters) + 1}")
