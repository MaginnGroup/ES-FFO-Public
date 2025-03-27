import signac
import sys

from fffit.signac import save_signac_results
from utils.r41 import R41Constants


project_name = "dens-iter-1"
property_names = ["density", "surf_ten"]
save_csv = False
cl_shuffle_seed = 1  # classifier
gp_shuffle_seed = 42  # GP seed
dist_seed = 1  # Distance seed
save_fig = False

##############################################################################
##############################################################################
# Save DataFrame of all molecule data
df_all_molec = save_signac_results(project_name, property_names, save_csv)
