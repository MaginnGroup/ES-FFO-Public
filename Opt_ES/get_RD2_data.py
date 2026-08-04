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

#Change me as needed
#Dictionary of all molecules of interest
mol_names = ["EG" , "Gly", "MeOH", "DMSO", "DEC", "DMF"] #["EG" , "Gly", "MeOH", "DMSO", "DEC", "DMF"]

#Load class properies for each training and testing molecule
molec_dict = esolvs.make_dict(mol_names)

#Make empty dataframe with 5 columns, molecule, T, rho_l, rho_v, Pvap, Hvap
df = pd.DataFrame(columns=["Molecule", "T (K)", "rho_l (kg/m^3)", "rho_v (kg/m^3)", "Pvap (kPa)"])

#Make a pandas df for the properties of interest for each molecule at each temperature and pressure
#For each molecule, get the properties of interest and make a pandas df for each property
for mol_name in mol_names:
    print(f"Making tables for {mol_name}")
    molec = molec_dict[mol_name]
    #Get molecule properties for each temperature and pressure
    T = molec.expt_liq_density.keys()
    #if mol = MeoH or EG, only use T < 450
    # if mol_name in ["MeOH", "EG"]:
    #     T = [t for t in T if t < 450]
    liq_density = [molec.expt_liq_density[t] for t in T]
    vap_density = [molec.expt_vap_density[t] for t in T]
    Pvap = [molec.expt_Pvap[t]*100 for t in T]  #Convert from bar to kPa
    #Add the properties to the dataframe
    for t, rho_l, rho_v, pvap in zip(T, liq_density, vap_density, Pvap):
        df = pd.concat([df, pd.DataFrame({"Molecule": [mol_name], "T (K)": [t], "rho_l (kg/m^3)": [rho_l], "rho_v (kg/m^3)": [rho_v], "Pvap (kPa)": [pvap]})], ignore_index=True)

#Save the dataframe to a csv file
df.to_csv("all_properties.csv", index=False)
    