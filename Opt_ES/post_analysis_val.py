#Imports
import signac
import sys

import numpy as np
import pandas as pd
import os
import copy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
sys.path.append("..")
from utils.molec_class_files import esolvs
from utils.prep_ms_data import prepare_df_props, prepare_df_errors
sys.path.remove("..")

from utilsOpt.plot import plot_vle_envelopes, plot_surf_tens, plot_pvap_hvap, plot_err_each_prop, plot_err_avg_props

#After jobs are finished
#save signac results for each atom for a given atom typing scheme and number of training parameters
from utilsOpt import atom_type
from utilsOpt.signac import save_signac_results, get_signac_results

#Change me as needed
obj_choice = "ExpVal"
at_numbers = [0,1] #0 is distinct ATs
#Dictionary of all molecules of interest
mol_names = ["EG" , "Gly", "MeOH", "DMSO", "DEC", "DMF"]

### Do not modify below this point ###

def get_label_from_fn(file_name):
    for s in file_name.split("/"):
        if s.startswith("at_"):
            at_num = int(s.split("_")[1])
            label = atom_type.make_atom_type_class(at_num).scheme_plot_name
            break
        elif s in ff_names:
            #Otherwise create the label from a literature name
            idx = ff_names.index(s)
            label = ff_labels[idx]
            break
        else:
            label = "Unknown"
    return label
#Load class properies for each training and testing molecule
molec_dict = esolvs.make_dict(mol_names)
prop_dict = {} #Store dfs for FF property comparison {"file":df}
error_dict = {} #Store dfs for FF error comparison {"file":df}
#Can filter out specific param sets here if needed
gemc_proj = signac.get_project("gemc_val")
ift_proj = signac.get_project("ift_val")
gemc_props = ["vap_density", "Hvap", "Pvap", "liq_enthalpy", "vap_enthalpy"]
ift_props = ["liq_density", "surf_tens"]

#Get Results from GEMC and IFT Simulations
project_dict = {"gemc":[gemc_proj, gemc_props],
                "ift": [ift_proj, ift_props]}
all_df_data = get_signac_results(project_dict, molec_dict)
all_df_data = save_signac_results(all_df_data)

#Calculate MAPD and MSE for each T point
for file, df_molec in all_df_data.items():
    #Determine if file contains data for > 1 molecule (generalized FFs)
    file_pieces = file.split("/")
    train_mol_str = file_pieces[2]#This will be train_mol_str
    train_moles = train_mol_str.split("-")
    if len(train_moles) > 1:
        #Prepare and save data for each individual molecule when genFF is being evaluated
        for molec in train_moles:
            molec_data = copy.copy(df_molec[df_molec['molecule'] == molec])
            df_all, df_liq, df_vap = prepare_df_props(molec_data, molec_dict[molec], ld_threshold = 0, scale = False)
            file_save = file.replace(train_mol_str, molec)
            data_loc = os.path.join(file_save, "ms_data.csv")
            df_all.to_csv(data_loc)
            prop_dict[file_save] = df_all
            #Save error data as well
            df_paramsets = prepare_df_errors(df_all, molec_dict)
            df_paramsets["molecule"] = molec
            df_paramsets.to_csv(os.path.join(file_save, "error_data.csv"))
            error_dict[file_save] = df_paramsets
    else:
        #Save single molecule propery data
        df_all, df_liq, df_vap = prepare_df_props(df_molec, molec_dict[molec], ld_threshold = 0, scale = False)
        data_loc = os.path.join(file, "ms_data.csv")
        df_all.to_csv(data_loc)
        #Save single molecule errors
        df_paramsets = prepare_df_errors(df_all, molec_dict)
        df_paramsets["molecule"] = molec
        df_paramsets.to_csv(os.path.join(file, "error_data.csv"))
        error_dict[file] = df_paramsets
    

#Get Results from Literature Data (for comparison)
#Load csvs for Opt_FF, GAFF, NW, Trappe, and Potoff, and BBFF
ff_names = ["Potoff", "TraPPE", "Wang_FFO", "BBFF"]
ff_labels = ["Potoff et al.", "TraPPE", "Previous Work", ""]
# for ff_name, ff_label in zip(ff_names, ff_labels):
    #Get FF data
    #Loop over each molecule and create property and error data
    # for mol_name, df_data in df_dict.items():
    #     prepare_df_props(df_csv, molecule, liquid_density_threshold, scale=True)
    #     prop_dict[file] = df_all
    #     error_lit = prepare_df_errors(df_data, data_dict, mol_name)
    #     error_dict[file] = error_lit

#Plot VLE, Hvap/Pvap, and ST
full_at_dir = os.path.join("analysis", "AT-" + "".join(map(str, at_numbers)), "ms_val")
os.makedirs(full_at_dir, exist_ok=True)
pdf_vle = PdfPages(os.path.join(full_at_dir ,"vle.pdf"))
pdf_hpvap = PdfPages(os.path.join(full_at_dir ,"h_p_vap.pdf"))
pdf_st = PdfPages(os.path.join(full_at_dir ,"surf_tens.pdf"))


#For each molecule
molecules = df_paramsets['molecule'].unique().tolist()
for molec in molecules:
    #Get the data for the molecule from each FF if it exists
    one_molec_dict = {molec: molec_dict[molec]}
    ff_molec_dict = {}
    for file_name, df_ff in prop_dict.items():
        #Create a label for the FF from the AT:
        label = get_label_from_fn(file_name) #+ "_" + molec
        #Add molecule data to dict for plotting
        df_molec = copy.copy(df_ff[df_ff['molecule'] == molec])
        ff_molec_dict[label] = df_molec
    
    #Plot Vle, Hvap, and Pvap and save to different pdfs
    pdf_vle.savefig(plot_vle_envelopes(one_molec_dict, ff_molec_dict), bbox_inches='tight')
    # plt.show()
    plt.close()
    pdf_hpvap.savefig(plot_pvap_hvap(one_molec_dict, ff_molec_dict), bbox_inches='tight')
    plt.close()
    pdf_st.savefig(plot_surf_tens(one_molec_dict, ff_molec_dict), bbox_inches='tight')
    plt.close()
#Close figures    
pdf_vle.close()
pdf_hpvap.close()
pdf_st.close()

#Get error dict labels ready
df_err_dict = {}
molec_names = mol_names
for file, data_df_err in error_dict.items():
    #Load the error data for the file
    #Remove all columns not related to error metrics
    label_base = get_label_from_fn(file)
    label_molec = data_df_err["molecule"].values[0]
    label = label_base + "_" + label_molec
    err_data = data_df_err.filter(regex="mapd|mse|mae")
    df_err_dict[label] = err_data

#For genFF plot MAPD breakdown for all molecules by property
error_objs = ["mae", "mapd"]
for error_obj in error_objs:
    #Make error Plots
    if len(at_numbers) == 1:
        at_class = atom_type.make_atom_type_class(at_numbers[0])
        full_at_dir = os.path.join("analysis", at_class.scheme_name, obj_choice, "ms_val")
    else:
        full_at_dir = os.path.join("analysis", "AT-" + "".join(map(str, at_numbers)), obj_choice, "ms_val")
    os.makedirs(full_at_dir, exist_ok=True)
    pdf_MAPD = PdfPages(os.path.join(full_at_dir , error_obj.upper() + ".pdf"))
    #For each molecule
    save_name = os.path.join(full_at_dir, error_obj + "_props.png")

    #These functions will need to change since there is no training/testing set. But keep the same format to them
    pdf_MAPD.savefig(plot_err_each_prop(molec_names, df_err_dict, obj = error_obj, save_name=save_name), bbox_inches='tight')
    plt.close()
    # pdf_MAPD.savefig(plot_err_avg_props(molec_names, df_err_dict, obj = error_obj), bbox_inches='tight')
    # plt.close()
    # #Close figures 
    pdf_MAPD.close() 
