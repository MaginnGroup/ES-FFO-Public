import signac
import sys

sys.path.append("..")
from utils.molec_class_files import esolvs
sys.path.remove("..")

from utils.signac import get_signac_results, save_signac_results
from utils.id_new_samples import new_samples_vle, find_pareto
from utils.models import get_best_models, plot_gp_examples



#Set iters to analyze and properties to analyze
iters = [1]  # Change me as needed
property_names = ["liq_density", "surf_tens"]  # Change me as needed
mol_names = ["EG", "Gly", "ACN", "MeOH", "DMSO", "THF", "DCM", "DEC", "DMF"] # Change me as needed


#Set seeds and preferences
cl_shuffle_seed = 1  # classifier
gp_shuffle_seed = 42  # GP seed
dist_seed = 1  # Distance seed
mse_less_10_thresh = 25
save_csv = False
save_fig = False
verbose = True


##############################################################################
##############################################################################
#Get Project
iter_type = "vle_iters" 
project = signac.get_project(iter_type)
#Load class properies for each molecule in the FF
molec_dict = esolvs.make_dict(mol_names)

# Save DataFrame of all molecule data for each iteration
df_all_molec = get_signac_results(project, molec_dict, property_names)
df_all_molec = save_signac_results(df_all_molec, iter_type, save_csv)
#Make and save best GP models for all molecules and properties and plot GP examples
models_molecs = get_best_models(df_all_molec, molec_dict, iter_type, gp_shuffle_seed, save_fig)
plot_gp_examples(df_all_molec, molec_dict, iter_type, gp_shuffle_seed, save_fig)
#Check pareto efficient samples for each molecule to see if there is one with < 5% error in all properties
all_final_params = find_pareto(df_all_molec, molec_dict)
for key, value in all_final_params.items():
    #If there are, we have the final parameters
    if len(value) > 0:
        print(f"{key}: Final parameters:")
        print(value)
    #Otherwise we need to move to the next iteration
    else:
        print(f"{key} : No final parameters found. Move to iteration {max(iters) + 1}")
        next_samples = new_samples_vle(df_all_molec, molec_dict, verbose = True, save_fig=False, gp_shuffle_seed = 42, dist_seed = 1)

