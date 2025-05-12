import signac
import sys
from utils.molec_class_files import esolvs
from utils.analyze_iters import get_signac_results, check_mse_10, save_signac_results, find_new_samples, find_pareto, plot_gp_examples
from utils.id_new_samples import check_mse_10

#Set iters to analyze and properties to analyze
iters = [1]  # Change me as needed
property_names = ["liq_density", "surf_tens"]  # Change me as needed
mol_names = ["EG", "Gly", "ACN", "MeOH", "DMSO", "THF", "DCM", "DEC", "DMF"] # Change me as needed


#Set seeds and preferences
cl_shuffle_seed = 1  # classifier
gp_shuffle_seed = 42  # GP seed
dist_seed = 1  # Distance seed
iter_type = "dens_iters"  # Change me as needed
mse_less_10_thresh = 25
save_csv = False
save_fig = False
verbose = True


##############################################################################
##############################################################################
#Get Project
base_name = "vle_iters/" #Name of the project folder
project = signac.get_project(base_name)
molec_dict = esolvs.make_dict(mol_names)

# Save DataFrame of all molecule data for each iteration
df_all_molec = get_signac_results(project, molec_dict, property_names)
df_all_molec = save_signac_results(df_all_molec, iter_type, save_csv)
all_prop_models = plot_gp_examples(df_all_molec, molec_dict, iter_type, gp_shuffle_seed, save_fig)
next_samples = find_new_samples(df_all_molec, molec_dict, verbose, save_fig, cl_shuffle_seed, gp_shuffle_seed, dist_seed)
mse_less10 = check_mse_10(df_all_molec, molec_dict, iter_type, mse_less_10_thresh, dist_seed, save_csv)
# if max(iters) > 1:
#     #Check whether results 
#     all_final_params = find_pareto(df_all_molec)
#     for key, value in all_final_params.items():
#         if len(value) > 0:
#             print(f"Final parameters for {key}:")
#             print(value)
#         else:
#             print(f"No final parameters found for {key}. Move to iteration {max(iters) + 1}")
