import signac
import sys
import os
from pathlib import Path

root_path = Path(__file__).resolve().parents[1]  # ES-FFO directory (two levels up from this script)
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

# Now import using package structure relative to ES-FFO root
from utils.molec_class_files import esolvs
from Build_GPs.utils.signac import get_signac_results, save_signac_results
from Build_GPs.utils.id_new_samples import new_samples_ld, check_mse_10
from Build_GPs.utils.models import get_best_models
from Build_GPs.utils.plot import plot_gp_examples

print(f"Current working dir: {os.getcwd()}")
print(f"Script location: {Path(__file__).parent}")

# #Set iters to analyze and properties to analyze
# iters = [1]  # Change me as needed
property_names = ["liq_density"]  # Change me as needed
mol_names = ["EG", "Gly", "ACN", "MeOH", "DMSO", "THF", "DCM", "DEC", "DMF"] # Change me as needed


#Set seeds and preferences
cl_shuffle_seed = 1  # classifier
gp_shuffle_seed = 42  # GP seed
dist_seed = 1  # Distance seed
mse_less_10_thresh = 25
save_csv = True
save_fig = True
verbose = True


##############################################################################
##############################################################################
#Get Project
iter_type = "ld_iters"
project = signac.get_project(iter_type)
molec_dict = esolvs.make_dict(mol_names)

# Save DataFrame of all molecule data for each iteration
df_all_molec = get_signac_results(project, molec_dict, property_names)
df_all_molec = save_signac_results(df_all_molec, iter_type, save_csv)
#Make and save best GP models for all molecules and properties and plot GP examples
models_molecs = get_best_models(df_all_molec, molec_dict, iter_type, gp_shuffle_seed)
plot_gp_examples(df_all_molec, molec_dict, iter_type, gp_shuffle_seed, save_fig)
#Check the MSE of the new samples
mse_less10 = check_mse_10(df_all_molec, molec_dict, mse_less_10_thresh, dist_seed, save_csv)
#Find the next samples to run if fewer than 25 samples have MSE less than 10
for key, value in mse_less10.items():
    if len(value) >= mse_less_10_thresh:
        print(f"{key} : {len(value)} samples have MSE less than {mse_less_10_thresh}. Proceed to VLE.")
    else:
        print(f"{key} : Total samples with MSE less than {mse_less_10_thresh}, {value}")
        next_samples = new_samples_ld(df_all_molec, molec_dict, verbose, save_fig, cl_shuffle_seed, gp_shuffle_seed, dist_seed)
        
