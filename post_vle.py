import signac
import sys
from utils.molec_class_files import esolvs
from utils.analyze_iters import get_signac_results, save_signac_results, new_samples_vle, find_pareto

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
next_samples = new_samples_vle(df_all_molec, molec_dict, verbose = True, save_fig=False, gp_shuffle_seed = 42, dist_seed = 1)
if max(iters) > 1:
    #Check whether results 
    all_final_params = find_pareto(df_all_molec, molec_dict)
    for key, value in all_final_params.items():
        if len(value) > 0:
            print(f"Final parameters for {key}:")
            print(value)
        else:
            print(f"No final parameters found for {key}. Move to iteration {max(iters) + 1}")
