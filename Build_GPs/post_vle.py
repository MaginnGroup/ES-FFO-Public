import signac
import sys
import os
from pathlib import Path
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# root_path = Path(__file__).resolve().parents[1]  # ES-FFO directory (two levels up from this script)
# if str(root_path) not in sys.path:
#     sys.path.insert(0, str(root_path))

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),  ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Now import using package structure relative to ES-FFO root
from utils.molec_class_files import esolvs
from Build_GPs.utils.signac import get_signac_results, save_signac_results
from Build_GPs.utils.id_new_samples import new_samples_vle, find_pareto
from Build_GPs.utils.models import get_best_models
from Build_GPs.utils.plot import plot_gp_examples, plot_sim_exp

print(f"Current working dir: {os.getcwd()}")
print(f"Script location: {Path(__file__).parent}")

#Set iters to analyze and properties to analyze
iters = [1]  # Change me as needed
property_names = ["liq_density", "surf_tens"]  # Change me as needed
mol_names = ["EG", "Gly", "MeOH", "DMSO", "DEC", "DMF"]

#Set seeds and preferences
cl_shuffle_seed = 1  # classifier
gp_shuffle_seed = 42  # GP seed
dist_seed = 1  # Distance seed
mapd_le = 10
save_csv = True
save_fig = True
verbose = True

##############################################################################
##############################################################################
def get_best_set_data(molec_name):
    # Check the analysis folder for analysis/MolName/vle_iters folders
    # Find the highest params-iter-X.csv file
    pareto_sets = pd.read_csv(f"analysis/{molec_name}/vle_iters/iter-1/final-params.csv", header = 0, index_col = 0)
    all_data = pd.read_csv(f"analysis/{molec_name}/vle_iters/all_results.csv", header = 0, index_col = 0)
    #Get the row where the mapd_surf_tens column is lowest
    best_row = pareto_sets.loc[pareto_sets['mapd_surf_tens'].idxmin()]
    #Return the array of all parameters (ignore mapd columns)
    param_set = best_row.drop(labels=[col for col in best_row.index if "mapd" in col])
    #Find the final parameters with the lowest surface tension
    param_set = pd.DataFrame(param_set).T
    mask = (all_data[param_set.columns] == param_set.iloc[0]).all(axis=1)

    # Apply mask
    all_data = all_data[mask]
    all_data = all_data.sort_values(by='temperature', ascending=True)
    all_data = pd.DataFrame(all_data)
    return all_data

#Get Project
iter_type = "vle_iters" 
project = signac.get_project(iter_type)
#Load class properies for each molecule in the FF
molec_dict = esolvs.make_dict(mol_names)

# Save DataFrame of all molecule data for each iteration
df_all_molec = get_signac_results(project, molec_dict, property_names)
df_all_molec = save_signac_results(df_all_molec, iter_type, save_csv)

#Check pareto efficient samples for each molecule to see if there is one with < mapd_le (10)% error in all properties
all_final_params = find_pareto(df_all_molec, molec_dict, property_names, mapd_le)

#Make and save best GP models for all molecules and properties and plot GP examples
models_molecs = get_best_models(df_all_molec, molec_dict, iter_type, gp_shuffle_seed)
plot_gp_examples(df_all_molec, molec_dict, iter_type, gp_shuffle_seed, save_fig)

for key, value in all_final_params.items():
    #If there are, we have the final parameters
    if len(value) > 0:
        print(f"{key}: Final parameters:")
        print(value)
        best_data = get_best_set_data(key)
        #Make a fxn in utils.plot to plot predictions vs exp data for LD and ST
        dir_name = f"analysis/{key}/{iter_type}"
        os.makedirs(dir_name, exist_ok=True)
        pdf_name = os.path.join(dir_name , "prop_preds.pdf")
        pdf = PdfPages(pdf_name)
        for props in ["liq_density", "surf_tens"]:
            fig = plot_sim_exp(molec_dict[key], best_data, props)
            pdf.savefig(fig, bbox_inches='tight')   # save one figure at a time
        pdf.close()

    #Otherwise we need to move to the next iteration
    else:
        print(f"{key} : No final parameters found. Move to iteration {max(iters) + 1}")
        next_samples = new_samples_vle(df_all_molec, molec_dict, verbose, gp_shuffle_seed, dist_seed)