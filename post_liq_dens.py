import signac
import sys

from utils.analyze_iters import save_signac_results

base_name = "density_iters/runs/"
iters = [1]  # Change me as needed
project = signac.get_project(base_name)
property_names = ["density", "surf_ten"]  # Change me as needed
save_csv = False
cl_shuffle_seed = 1  # classifier
gp_shuffle_seed = 42  # GP seed
dist_seed = 1  # Distance seed
save_fig = False

##############################################################################
##############################################################################
# Save DataFrame of all molecule data for each iteration
df_all_molec = save_signac_results(project, property_names, save_csv)
