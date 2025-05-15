#Imports
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

#Set params for what you want to analyze
save_data = True #Data to save
obj_choice = "ExpVal" #Objective to consider
at_number = 6 #atom type to consider
seed = 1 #Seed to use
molec_names = ["R14", "R32", "R50", "R170", "R125", "R134a", "R143a", "R41"] #Training data to consider

#Create visualization and opt_ats object
visual = opt_atom_types.Vis_Results(molec_names, at_number, seed, obj_choice)
repeats = 1
opt_ats = opt_atom_types.Opt_ATs(molec_names, at_number, repeats, seed, obj_choice)
#Set parameter set of interest (in this case get the best parameter set)
x_label = "best_set"
all_molec_dir = opt_ats.use_dir_name
path_best_sets = os.path.join(all_molec_dir, "best_per_run.csv")
assert os.path.exists(path_best_sets), "best_per_run.csv not found in directory"
all_df = pd.read_csv(path_best_sets, header = 0)
first_param_name = opt_ats.at_class.at_names[0] + "_min"
last_param_name = opt_ats.at_class.at_names[-1] + "_min"
all_sets = all_df.loc[:, first_param_name:last_param_name].values
unique_best_sets = visual.get_unique_sets(all_sets, save_data, save_label=x_label)

#Loop over unique parameter sets
for i in range(unique_best_sets.shape[0]):
    x_label_unique = x_label + "_" + str(i+1)
    # x_label_rcc = "rcc_set_" + str(i+1)
    best_set = unique_best_sets.iloc[i,:].values
    best_real = opt_ats.values_pref_to_real(copy.copy(best_set))

    #Get sensitivity analysis results for each best set
    ranked_indices, n_data, at_names_ranked = opt_ats.rank_parameters(best_real, save_data, x_label_unique)

    #Get RCC Analysis
    # opt_num_param, rcc, loss_data, opt_params =opt_ats.estimate_opt(best_real, ranked_indices, 
    #                                                                n_data, save_data, x_label_rcc)

    # #Opt params pref to real
    # opt_params = opt_ats.values_pref_to_real(opt_params)
    # #Get Property Predictions for all training molecules
    # molec_names_all = list(visual.all_train_molec_data.keys())
    # visual.comp_paper_full_ind(molec_names_all, opt_params, save_label=x_label_rcc)

    # #Calculate MAPD for predictions and save results
    # MAPD_rcc = visual.calc_MAPD_best(molec_names_all, opt_params, save_data, x_label_rcc)