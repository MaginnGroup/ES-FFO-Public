#Imports
import signac
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import os
import copy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

root_path = (
    Path(__file__).resolve().parents[1]
)  # ES-FFO directory (two levels up from this script)
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

# Now import using package structure relative to ES-FFO root
from utils.molec_class_files import esolvs
from utils.prep_ms_data import estimate_hvaps, prepare_df_props, prepare_df_errors
from Opt_ES.utilsOpt.plot import plot_vle_envelopes, plot_misc_prop, plot_pvap_hvap, plot_err_each_prop, plot_err_avg_props
from Opt_ES.utilsOpt import atom_type
from Opt_ES.utilsOpt.signac import save_signac_results, get_signac_results

print(f"Current working dir: {os.getcwd()}")
print(f"Script location: {Path(__file__).parent}")

#After jobs are finished
#save signac results for each atom for a given atom typing scheme and number of training parameters
opt_status = "opt"

#Change me as needed
obj_choice = "ExpVal"
at_numbers = [0] #0 is distinct ATs
#Dictionary of all molecules of interest
mol_names = ["EG" , "Gly", "MeOH", "DMSO", "DEC", "DMF"] #["EG" , "Gly", "MeOH", "DMSO", "DEC", "DMF"]

### Do not modify below this point ###
def get_label_from_fn(file_name):
    for s in file_name.split("/"):
        if s.startswith("at_"):
            at_num = int(s.split("_")[1])
            label = atom_type.make_atom_type_class(at_num).scheme_plot_name
            break
        else:
            label = "Unknown"
    return label
#Load class properies for each training and testing molecule
molec_dict = esolvs.make_dict(mol_names)
prop_dict = {} #Store dfs for FF property comparison {"file":df}
error_dict = {} #Store dfs for FF error comparison {"file":df}
#Can filter out specific param sets here if needed
gemc_proj = signac.get_project(f"gemc_val_{opt_status}")
gemc_props = ["vap_density", "Hvap", "Pvap", "liq_enthalpy", "vap_enthalpy"]
try:
    ift_proj = signac.get_project(f"ift_val_{opt_status}")
    ift_props = ["liq_density", "surf_tens", "diff_coeff"]
except:
    ift_proj = None
    ift_props = []
    gemc_props.append("liq_density")

#Get Results from GEMC and IFT Simulations
project_dict = {"gemc":[gemc_proj, gemc_props],
                "ift": [ift_proj, ift_props]}
all_df_data = get_signac_results(project_dict, molec_dict, proj_name=opt_status)
all_df_data = save_signac_results(all_df_data)

#Calculate MAPD and MSE for each T point
for file, df_molec in all_df_data.items():
    #Determine if file contains data for > 1 molecule (generalized FFs)
    file_pieces = file.split("/")
    train_mol_str = file_pieces[9]#This will be train_mol_str
    # print(file_pieces, train_mol_str)
    train_moles = train_mol_str.split("-")
    #Prepare and save data for each individual molecule when genFF is being evaluated
    for molec in train_moles:
        if len(train_moles) > 1:
            #Save individual molecule property data
            molec_data = copy.copy(df_molec[df_molec['molecule'] == molec])
            df_all, df_liq, df_vap = prepare_df_props(molec_data, molec_dict[molec], 0, scale = False)
            file_save = file.replace(train_mol_str, molec)
        else:
            #Save single molecule propery data
            df_all, df_liq, df_vap = prepare_df_props(df_molec, molec_dict[molec], 0, scale = False)
            file_save = file

        #Save df for Hvap estimates
        df_H_est = estimate_hvaps(df_all, molec_dict, molec)
        data_H_loc = os.path.join(file_save, "Hvap_estimates.csv")
        df_H_est.to_csv(data_H_loc)

        #Save prop data
        data_loc = os.path.join(file_save, "ms_data.csv")
        df_all.to_csv(data_loc)
        prop_dict[file_save] = df_all
        #Save molecule errors
        df_paramsets = prepare_df_errors(df_all, molec_dict, molec)
        df_paramsets["molecule"] = molec
        df_paramsets.to_csv(os.path.join(file_save, "error_data.csv"))
        error_dict[file_save] = df_paramsets

#Plot VLE, Hvap/Pvap, and ST
full_at_dir = os.path.join("analysis", "AT-" + "".join(map(str, at_numbers)), f"ms_val_{opt_status}")
os.makedirs(full_at_dir, exist_ok=True)
pdf_vle = PdfPages(os.path.join(full_at_dir ,"vle.pdf"))
pdf_hpvap = PdfPages(os.path.join(full_at_dir ,"h_p_vap.pdf"))
pdf_st = PdfPages(os.path.join(full_at_dir ,"surf_tens.pdf"))
pdf_diff = PdfPages(os.path.join(full_at_dir ,"diff_coeff.pdf"))

#### Add literature FF data to prop_dict
# file_save_lit = "analysis/lit_ff_data.csv"
# lit_data = pd.read_csv(file_save_lit, header=0)

# for molec_name in mol_names:
#     #Get the data from the original workflow for the best NW parameter set
#     pareto_sets = pd.read_csv(f"../Build_GPs/analysis/{molec_name}/vle_iters/iter-1/final-params.csv", header = 0, index_col = 0)
#     all_data = pd.read_csv(f"../Build_GPs/analysis/{molec_name}/vle_iters/iter-1/results.csv", header = 0, index_col = 0)
#     #Get the row where the mapd_surf_tens column is lowest
#     best_row = pareto_sets.loc[pareto_sets['mapd_surf_tens'].idxmin()]
#     #Return the array of all parameters (ignore mapd columns)
#     param_set = best_row.drop(labels=[col for col in best_row.index if "mapd" in col])
#     #Find the final parameters with the lowest surface tension
#     param_set = pd.DataFrame(param_set).T
#     mask = (all_data[param_set.columns] == param_set.iloc[0]).all(axis=1)

#     #Apply mask
#     all_data = all_data[mask]
#     all_data = all_data.sort_values(by='temperature', ascending=True)
#     all_data = pd.DataFrame(all_data)
#     #Add molecule column
#     all_data['molecule'] = molec_name
#     #Add ref name and short name columns
#     all_data['ref_name'] = 'Wang et. al.'
#     all_data['short_name'] = 'NW'
#     #Add sim to any column name that contains" liq_" or "surf_"
#     for col in all_data.columns:
#         if "liq_" in col or "surf_" in col:
#             all_data = all_data.rename(columns={col: "sim_" + col})
#     #Return the array of all parameters (ignore mapd columns)
#     lit_data = pd.concat([lit_data, all_data.reindex(columns=lit_data.columns)], ignore_index=True)

# #Save lit data with sim columns for future use
# lit_data.to_csv("analysis/lit_ff_data_w_NW.csv", index=False)

####Add old FF data to lit_data
file_save_lit = "analysis/lit_ff_data.csv"
# lit_data = pd.read_csv(file_save_lit, header=0)
# for molec_name in mol_names:
#     df_old_FF = pd.read_csv(f"analysis_old/at_00/{molec_name}/ExpVal/opt_res/ms_val/ms_data.csv", header = 0, index_col = 0)
#     #Drop all column without sim_ in the name or "temperature" or "molecule"
#     cols_to_keep = [col for col in df_old_FF.columns if "sim_" in col or col in ["temperature", "molecule"]]
#     df_old_FF = df_old_FF[cols_to_keep]
#     df_old_FF['ref_name'] = 'Old Opt FF'
#     df_old_FF['short_name'] = 'Old Opt FF'
#     lit_data = pd.concat([lit_data, df_old_FF.reindex(columns=lit_data.columns)], ignore_index=True)
# lit_data.to_csv("analysis/lit_ff_data_w_oldFF.csv", index=False)

# file_save_lit = "analysis/lit_ff_data_w_oldFF.csv"
lit_data = pd.read_csv(file_save_lit, header=0)
for molec_name in mol_names:
    other_opt = "no_opt" if opt_status == "opt" else "opt"
    ref_name = "IFT FF" if opt_status == "opt" else "Opt FF"
    df_old_FF = pd.read_csv(f"analysis/at_00/{molec_name}/ExpVal/opt_res/ms_val_{other_opt}/ms_data.csv", header = 0, index_col = 0)
    #Drop all column without sim_ in the name or "temperature" or "molecule"
    cols_to_keep = [col for col in df_old_FF.columns if "sim_" in col or col in ["temperature", "molecule"]]
    df_old_FF = df_old_FF[cols_to_keep]
    df_old_FF['ref_name'] = ref_name
    df_old_FF['short_name'] = ref_name
    lit_data = pd.concat([lit_data, df_old_FF.reindex(columns=lit_data.columns)], ignore_index=True)
lit_data.to_csv(f"analysis/lit_ff_data_w_{other_opt}.csv", index=False)
lit_data_error = prepare_df_errors(lit_data, molec_dict, molec_name)
lit_data_error.to_csv("analysis/lit_error_data.csv")

 #Save df for Hvap estimates
h_est_lit_data = estimate_hvaps(lit_data, molec_dict, molec)
h_est_lit_data.to_csv(f"analysis/lit_Hvap_est_w_{other_opt}.csv", index=False)

#For each file in error dict, add the data to the lit data if it is not already there (if it is not already in prop dict)
new_lit_data = copy.copy(lit_data_error)
for file_err, df_err in error_dict.items():
    df_err["ref_name"] = "IFT FF" if opt_status == "no_opt" else "Opt FF"
    #Add the data to the lit data if it is not already there (if it is not already in prop dict)
    new_lit_data = pd.concat([new_lit_data, df_err], join="inner", ignore_index=True)
#Sort by molecule and remove rows where columns other than ref_name and molecule are NaN
ref_order = ["Opt FF", "IFT FF"]
new_lit_data["ref_name"] = pd.Categorical(
    new_lit_data["ref_name"],
    categories=ref_order + sorted(set(new_lit_data["ref_name"]) - set(ref_order)),
    ordered=True,)
new_lit_data = new_lit_data.sort_values(["molecule", "ref_name"])
new_lit_data = new_lit_data.dropna(subset=[col for col in new_lit_data.columns if col not in ["ref_name", "molecule"]], how='all').reset_index(drop=True)
new_lit_data.to_csv("analysis/comp_err_data.csv")
    
#For each molecule
molecules = mol_names #df_paramsets['molecule'].unique().tolist()
for molec in molecules:
    #Get the data for the molecule from each FF if it exists
    one_molec_dict = {molec: molec_dict[molec]}
    ff_molec_dict = {}
    for file_name, df_ff in prop_dict.items():
        if molec in file_name:
            #Add molecule data to dict for plotting
            df_molec = copy.copy(df_ff[df_ff['molecule'] == molec])
            #Create a label for the FF from the AT:
            label = get_label_from_fn(file_name) #+ "_" + molec
            ff_molec_dict[label] = df_molec

    lit_data_molec = copy.copy(lit_data[lit_data['molecule'] == molec])
    groups = lit_data_molec.groupby('ref_name')
    for name, group in groups:
        ff_molec_dict[name] = group
    
    #Plot Vle, Hvap, and Pvap and save to different pdfs
    pdf_vle.savefig(plot_vle_envelopes(one_molec_dict, copy.deepcopy(ff_molec_dict)), bbox_inches='tight', orientation='portrait')
    plt.close()
    pdf_hpvap.savefig(plot_pvap_hvap(one_molec_dict, copy.deepcopy(ff_molec_dict)), bbox_inches='tight')
    plt.close()
    pdf_st.savefig(plot_misc_prop(one_molec_dict, copy.deepcopy(ff_molec_dict), prop_name="surf_tens"), bbox_inches='tight')
    plt.close() 
    pdf_diff.savefig(plot_misc_prop(one_molec_dict, copy.deepcopy(ff_molec_dict), prop_name="diff_coeff"), bbox_inches='tight')
    plt.close()
    pdf_vle.savefig(plot_misc_prop(one_molec_dict, copy.deepcopy(ff_molec_dict), prop_name="liq_density"), bbox_inches='tight')
    plt.close()
    pdf_vle.savefig(plot_misc_prop(one_molec_dict, copy.deepcopy(ff_molec_dict), prop_name="vap_density"), bbox_inches='tight')
    plt.close()
        
#Close figures    
pdf_vle.close()
pdf_hpvap.close()
pdf_st.close()
pdf_diff.close()

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
#TO DO: Need to modify plotting functions since there is no training/testing split here
error_objs = ["mae", "mapd"]
for error_obj in error_objs:
    #Make error Plots
    if len(at_numbers) == 1:
        at_class = atom_type.make_atom_type_class(at_numbers[0])
        full_at_dir = os.path.join("analysis", at_class.scheme_name, obj_choice, f"ms_val_{opt_status}")
    else:
        full_at_dir = os.path.join("analysis", "AT-" + "".join(map(str, at_numbers)), obj_choice, f"ms_val_{opt_status}")
    os.makedirs(full_at_dir, exist_ok=True)
    pdf_MAPD = PdfPages(os.path.join(full_at_dir , error_obj.upper() + ".pdf"))
    #For each molecule
    save_name = os.path.join(full_at_dir, error_obj + "_props")

    #These functions will need to change since there is no training/testing set. But keep the same format to them
    pdf_MAPD.savefig(plot_err_each_prop(molec_names, df_err_dict, obj = error_obj, save_name=save_name), bbox_inches='tight')
    plt.close()
    # pdf_MAPD.savefig(plot_err_avg_props(molec_names, df_err_dict, obj = error_obj), bbox_inches='tight')
    # plt.close()
    # #Close figures 
    pdf_MAPD.close() 
