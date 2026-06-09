import sys
import os
import numpy as np
import pandas as pd
import math
import matplotlib
import matplotlib.pyplot as plt
import seaborn
from scipy.stats import linregress
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
from matplotlib.ticker import LogLocator, MultipleLocator, AutoMinorLocator, MaxNLocator
from fffit.fffit.utils import values_real_to_scaled, values_scaled_to_real, variances_scaled_to_real
from fffit.fffit.plot import plot_model_performance, plot_model_vs_test, plot_slices_temperature, plot_slices_params, plot_model_vs_exp, plot_obj_contour
from utils.molec_class_files import esolvs

mol_names = ["EG" , "Gly", "ACN", "MeOH", "DMSO", "THF", "DCM", "DEC", "DMF"]
molec_dict = esolvs.make_dict(mol_names)

#Deprecated
# def prepare_df_vle(df_csv, molec_dict, csv_name = None, drop_one = False):
#     """Prepare a pandas dataframe for fitting a GP model to density data

#     Performs the following actions:
#        - Renames "liq_density" to "sim_liq_density" (units kg/m3 assumed)
#        - Renames "vap_density" to "sim_vap_density" (units kg/m3 assumed)
#        - Renames "Pvap" to "sim_Pvap" (units bar assumed)
#        - Removes "liq_enthalpy" and "vap_enthalpy" and adds "sim_Hvap" (units kJ/kg returned)
#             - Units kJ/mol assumed for Hvap and kJ/kg for Hvap kJ/kg

#     Parameters
#     ----------
#     df_csv : pd.DataFrame
#         The dataframe as loaded from a CSV file with the signac results
#     molec_dict : {"Rxx": RxxConstants, ...}
#         A dictionary mapping molecule names to classes
#     n_molecules : int
#         The number of molecules in the simulation

#     Returns
#     -------
#     df_all : pd.DataFrame
#         The dataframe with scaled parameters and MD/expt. properties
#     """

#     def rename_col(df, property_name, units):
#         prop_units = property_name + " " + units
#         if property_name == "temperature":
#             sim_str = ""
#         else:
#             sim_str = "sim_"
#         if prop_units in df.columns:
#             df.rename(columns={prop_units: sim_str + property_name}, inplace=True)
#         elif property_name in df.columns:
#             df.rename(columns={property_name: sim_str + property_name}, inplace=True)
#         else:
#             raise ValueError(f"df must contain either {property_name} or {prop_units}")
        
#         return df
    
#     # # Convert Hvap to kJ/kg if in kJ/mol
#     # if "Hvap" in df_csv.columns:
#     #     #Add Hvap in kJ/kg
#     #     df_csv["Hvap kJ/kg"] = df_csv["Hvap"]*1000.0/df_csv["molecule"].apply(
#     #         lambda molec: molec_dict[molec].molecular_weight)
#     #     #And drop kJ/mol column
#     #     df_csv.drop("Hvap", axis=1)
    
#     # Rename properties to MD
#     props = ["liq_density", "vap_density", "Pvap", "Hvap", "temperature"]
#     units = ["kg/m3", "kg/m3", "bar", "kJ/kg", "K"]
#     for prop, unit in zip(props, units):
#         rename_col(df_csv, prop, unit)
        
#     #sort by molecule and temperature -- added by Ning Wang
#     df_csv.dropna(subset=["sim_liq_density", "sim_vap_density"], how = "any", inplace=True)
#     df_csv.sort_values(by=["molecule", "temperature"], inplace=True)

#     #Add Tc and Rhoc predictions
#     Tc, rhoc = calc_critical(df_csv)
#     df_csv["sim_Tc"] = Tc
#     df_csv["sim_rhoc"] = rhoc

#     #Drop any molecule and temperature with only 1 data point
#     if drop_one:
#         df_csv = df_csv.groupby(["molecule", "temperature"]).filter(lambda x: len(x) > 1)

#     if csv_name != None:
#         df_csv.to_csv(csv_name)
           
#     return df_csv

#Moved to id_new_samples.py
# def calc_critical(df):
#     """Compute the critical temperature and density

#     Accepts a dataframe with "T_K", "rholiq_kgm3" and "rhovap_kgm3"
#     Returns the critical temperature (K) and density (kg/m3)

#     Computes the critical properties with the law of rectilinear diameters
#     """
#     Tc = []
#     rhoc = []
#     for group, values in df.groupby(['molecule']):    
#         #Need to group by molecule and do this for each molecule
#         temps = values["temperature"].values
#         liq_density = values["sim_liq_density"].values
#         vap_density = values["sim_vap_density"].values

#         #Check that all temps are not the same
#         if all(x == temps[0] for x in temps):
#             Tc += [np.nan]*len(temps)
#             rhoc += [np.nan]*len(temps)
#         else:
#             # Critical Point (Law of rectilinear diameters)
#             slope1, intercept1, r_value1, p_value1, std_err1 = linregress(
#                 temps,(liq_density + vap_density) / 2.0,)

#             try:
#                 slope2, intercept2, r_value2, p_value2, std_err2 = linregress(
#                     temps,(liq_density - vap_density)**(1/0.32),)
#             except:
#                 slope2, intercept2, r_value2, p_value2, std_err2 = linregress(
#                     temps,abs((liq_density - vap_density))**(1/0.32),)

#             Tc_mol = np.abs(intercept2 / slope2)
#             rhoc_mol = intercept1 + slope1 * Tc_mol

#             # if len(temps) == 5:
#             Tc += list([Tc_mol])*len(temps)
#             rhoc += list([rhoc_mol])*len(temps)
        
#     return Tc, rhoc

#Integrated into prepare_df_errors
# def prepare_df_vle_errors(df, molec_dict, csv_name = None):
#     """Create a dataframe with mean square error (mse) and mean absolute
#     percent error (mape) for each unique parameter set. The critical
#     temperature and density are also evaluated.

#     Parameters
#     ----------
#     df : pandas.Dataframe
#         per simulation results
#     molecule : R143a
#         molecule class with bounds/experimental data

#     Returns
#     -------
#     df_new : pandas.Dataframe
#         dataframe with one row per parameter set and including
#         the MSE and MAPD for liq_density, vap_density, pvap, hvap,
#         critical temperature, critical density
#     """
#     new_data = []

#     #sort by molecule and temperature -- added by Ning Wang
#     df=df.sort_values(by=["molecule", "temperature"])
#     molecules = df['molecule'].unique().tolist()
#     for group, values in df.groupby(['molecule']):
#         new_quantities = {}
#         #The molecule is listed as the first value in the group
#         molecule = molec_dict[values["molecule"].values[0]]
#         if group[0] not in ["R134", "R152"] and len(values) > 0:
#             # Temperatures
#             temps = values["temperature"].values

#             #Add experimental data (if not R134, 143 or R152)
#             values["expt_liq_density"] = values["temperature"].apply(
#                 lambda temp: molecule.expt_liq_density[int(temp)])
#             values["expt_vap_density"] = values["temperature"].apply(
#                 lambda temp: molecule.expt_vap_density[int(temp)] )
#             values["expt_Pvap"] = values["temperature"].apply(
#                 lambda temp: molecule.expt_Pvap[int(temp)])
#             values["expt_Hvap"] = values["temperature"].apply(
#                 lambda temp: molecule.expt_Hvap[int(temp)])
#             # Critical Point (Law of rectilinear diameters)
#             values["expt_Tc"] =  molecule.expt_Tc
#             values["expt_rhoc"] = molecule.expt_rhoc
        
#             def calculate_objs(expt_values, sim_values, property_name, molecule_name):
#                 try:
#                     fin_sim = sim_values[np.isfinite(sim_values)]
#                     fin_expt = expt_values[np.isfinite(sim_values)]
#                     mse = mean_squared_error(fin_expt, fin_sim)
#                     mapd = mean_absolute_percentage_error(fin_expt, fin_sim) * 100.0
#                     mae = mean_absolute_error(fin_expt, fin_sim)
#                 except ValueError as e:
#                     print(f"Error in calculating {property_name} for {molecule_name}: {e}. Setting MSE, MAE, and MAPD to NaN")
#                     print("Exp", expt_values, "\n Sim", sim_values)
#                     mse, mapd, mae = np.nan, np.nan, np.nan
#                 return mse, mapd, mae

#             for prop in ["liq_density", "vap_density", "Pvap", "Hvap"]:
#                 mse, mapd, mae = calculate_objs(values["expt_" + prop], values["sim_" + prop], prop, group[0])
#                 new_quantities["mse_" + prop] = mse
#                 new_quantities["mapd_" + prop] = mapd
#                 new_quantities["mae_" + prop] = mae

#             for prop in ["Tc", "rhoc"]:
#                 mse, mapd, mae = calculate_objs(np.array([values["expt_" + prop].values[0]]), np.array([values["sim_" + prop].values[0]]), prop, group[0])
#                 new_quantities["mse_" + prop] = mse
#                 new_quantities["mapd_" + prop] = mapd
#                 new_quantities["mae_" + prop] = mae
#         else:
#             for prop in ["liq_density", "vap_density", "Pvap", "Hvap", "Tc", "rhoc"]:
#                 new_quantities["mse_" + prop] = np.nan
#                 new_quantities["mapd_" + prop] = np.nan
#                 new_quantities["mae_" + prop] = np.nan
        
#         data_to_append = list(group) + list(new_quantities.values())
#         # print(data_to_append)
#         new_data.append(data_to_append)

#     columns = list(["molecule"]) + list(new_quantities.keys())
#     new_df = pd.DataFrame(new_data, columns=columns)

#     if csv_name != None:
#         new_df.to_csv(csv_name)

#     return new_df

# def get_min_max(curr_min, curr_max, new_vals, std_dev = None):
#     if isinstance(new_vals, float):
#         new_vals = [new_vals]
#     if std_dev is not None:
#         min_new_val = np.maximum(np.nanmin(new_vals - 2 * std_dev), 1e-6) #Avoid negative values for Pvap
#         max_new_val = np.nanmax(new_vals + 2 * std_dev)
#     else:
#         min_new_val = np.nanmin(new_vals)
#         max_new_val = np.nanmax(new_vals)
#     # print(min_new_val, max_new_val)
#     if min_new_val < curr_min and np.isfinite(min_new_val):
#         curr_min = min_new_val
#     if max_new_val > curr_max:
#         curr_max = max_new_val
#     return curr_min, curr_max


def get_min_max(curr_min, curr_max, new_vals, std_dev=None):
    """
    Update the minimum and maximum values based on new values and standard deviation.
    
    Parameters
    ----------
    curr_min : float
        Current minimum value.
    curr_max : float
        Current maximum value.
    new_vals : array-like
        New values to consider for updating the min and max.
    std_dev : array-like, optional
        Standard deviation of the new values. If provided, it will be used to adjust the min and max.
    
    Returns
    -------
    curr_min : float
        Updated minimum value.
    curr_max : float
        Updated maximum value.
    """
    # Ensure new_vals is iterable
    if isinstance(new_vals, (float, int)):
        new_vals = [new_vals]

    # Convert to NumPy array for easier handling
    new_vals = np.array(new_vals)
    
    # Filter finite values to avoid issues with NaN or Inf
    finite_indices = np.where(np.isfinite(new_vals))[0]
    valid_vals = new_vals[finite_indices]
    
    if valid_vals.size == 0:  # If no valid values exist, return current bounds
        return curr_min, curr_max

    # Compute adjusted min and max
    if std_dev is not None:
        valid_stds = std_dev[finite_indices]
        adjusted_vals = valid_vals - 1.96 * valid_stds
        min_new_val = np.nanmin(adjusted_vals)  # Avoid negative Pvap
        max_new_val = np.nanmax(valid_vals + 1.96 * valid_stds)
    else:
        min_new_val = np.nanmin(valid_vals)
        max_new_val = np.nanmax(valid_vals)
    
    # Update curr_min and curr_max
    if min_new_val < curr_min and np.isfinite(min_new_val):
        curr_min = min_new_val
    if max_new_val > curr_max and np.isfinite(max_new_val):
        curr_max = max_new_val
    
    return curr_min, curr_max

def plot_misc_prop(molec_dict, df_ff_dict, prop_name):
    """
    Plot a specific property for a given molecule and force field

    Parameters
    ----------
    molec_dict : dict
        Dictionary containing the molecule data
    df_ff_dict : dict
        Dictionary containing the force field data
    prop_name : str
        The name of the property to plot

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot
    """
    molec = list(molec_dict.keys())[0]
    mol_data = molec_dict[molec]
    # Plot VLE envelopes
    fig, ax2 = plt.subplots(1, 1, figsize=(6,6))    
    
    df_keys, df_ffs =  zip(*df_ff_dict.items())
    df_labels = list(df_keys)
    df_ff_list = list(df_ffs)

    key_map = {"Martinez-Jimenez et al.": ('gray', 's', 1, False),
               "Jorgensen": ('tab:orange', '>', 1, False),
               "Gonzalez-Salgado & Vega": ('tab:green', 'p', 1, False),
               "Chen et al.": ('purple', 'd', 1, False),
               "Stubbs et al.": ('purple', 'd', 1, False),
               "Huang et al.": ('gray', 's', 1, False),
               "Jahn et al.": ('gray', 's', 1, False),
               "Caleman et al.": ('gray', 's', 1, False),
               "Vahid & Maginn": ('tab:orange', '>', 1, False),
               "Chalaris & Samios": ('tab:green', 'p', 1, False),
               "Senapati": ('gray', 's', 1, False),
               "Borin & Skaf": ('tab:green', 'p', 1, False),
               "Garcia-Melgarejo et al.": ('gray', 's', 1, False),
               "Luo et al.": ('tab:orange', '>', 1, False),
               "Wang et al.": ('tab:magenta', 'D', 1, False),
               "Old Opt FF": ('tab:blue', '+', 1, False),
               "IFT FF": ('tab:red', '^', 1, False),
               "Opt FF": ('tab:red', '^', 1, False),
               }

    cmap = plt.get_cmap("cool")  # Get the rainbow colormap
    num_lit = sum("AT" not in key for key in df_labels)
    df_colors = [cmap(i) for i in np.linspace(0, 1, len(df_ffs)-num_lit)]

    for i, key in enumerate(df_labels):
        if "AT-" in key: #color, marker, z_order)
            key_map[key] = (df_colors[i], "o", len(df_ff_list), True)

    # df_labels, df_ffs = ["This Work", "GAFF", "Potoff et al.", "TraPPE", "Wang et al.", "Befort et al." ]
    # df_colors = ['blue', 'gray', '#0989d9', 'red', 'green','purple']
    # df_markers = ['o', 's', '^', '*', 'p', 'd']
    # df_z_order = [6,3,2,1,5,4]
    prop_data = getattr(mol_data, "expt_" + prop_name)

    if prop_name == "Pvap":
        for key in prop_data.keys():
            prop_data[key] = prop_data[key]*100 #Convert from bar to kPa for plotting

    #Initialize min and max values
    min_temp = min(prop_data.keys())
    max_temp = max(prop_data.keys())
    min_st = min(prop_data.values())
    max_st = max(prop_data.values())
    # else:
    #     for df in df_ff_list:
    #         if df is not None:
    #             min_temp = min(df["temperature"].values)
    #             max_temp = max(df["temperature"].values)
    #             min_st = min(df["sim_" + prop_name].values)
    #             max_st = max(df["sim_" + prop_name].values)
    #             print(f"Plotting {prop_name} for {molec}, initial min/max: {min_st}/{max_st}, {min_temp}/{max_temp}")
    #             break

    

    for i in range(len(df_ff_list)):
        df_label = df_labels[i]
        df_ff = df_ff_list[i]

        df_color, df_marker, df_z_order, show_df = key_map.get(df_label, ('black', 'o', 2, True))
        
        # df_label = df_labels[i] if df_labels[i] != "" else "Previous Work"
        
        if df_ff is not None and "Vahid" not in df_label: #and show_df:
            min_temp, max_temp = get_min_max(min_temp, max_temp, df_ff["temperature"].values)
            all_props = ["sim_" + prop_name]
            # grouped = df_ff.groupby(["temperature", "atom_type"])[all_props]
            grouped = df_ff.groupby(["temperature"])[all_props]
            x_props = ["sim_" + prop_name]
            if df_ff["sim_" + prop_name].isnull().all():
                continue
            # Calculate mean and standard deviation for each group
            means = grouped.mean().reset_index()
            stds = grouped.std(ddof=0).reset_index()

            for x_prop in x_props:
                #Set new max and mins
                if prop_name == "Pvap": #Convert from bar to kPa for plotting -- multiply by 100
                    #multiply by 10**9 
                    means[x_prop] = means[x_prop]*100
                    stds[x_prop] = stds[x_prop]*100
                min_st, max_st = get_min_max(min_st, max_st, means[x_prop].values, stds[x_prop].values)
                # print(min_st, max_st)
                # #Plot opt_scheme_ms vle curve
                if df_label == "AT-Dis":
                    df_label = "GP-Opt"
                elif df_label == "IFT FF":
                    df_label = "Base" #"Lowest " + r"$\gamma$" + " MAPD FF"
                ax2.errorbar(means["temperature"], means[x_prop],yerr=1.96*stds[x_prop],
                            color=df_color,markersize=10, linestyle='None', marker = df_marker, alpha=0.5, 
                            zorder = df_z_order, label = df_label)

    #Plot experimental data
    # if molec in ["MeOH", "EG"]:
    #     keys = np.array(list(prop_data.keys()))
    #     vals = np.array(list(prop_data.values()))
    #     mask = keys < 430
    #     ax2.scatter(keys[mask], vals[mask],
    #     color="black",marker="x",linewidths=2,s=100,label="Experiment", zorder = len(df_ff_list)+1)
    # else:
    ax2.scatter(prop_data.keys(), prop_data.values(),
        color="black",marker="x",linewidths=2,s=100,label="Experiment", zorder = len(df_ff_list)+1)

    #Set Axes
    #Use a log10 scale for diff_coeff
    if prop_name == "diff_coeff":
        ax2.set_yscale("log")
        # ax2.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax2.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
        # pyplot.locator_params(nbins=4)
    else:
        # ax2.set_ylim(min_st*0.95,max_st*1.05)
        #Set 5 ticks on y axis
        ax2.yaxis.set_major_locator(MaxNLocator(nbins=6))

    
    # print(f"Final min/max for {prop_name} for {molec}: {min_st}/{max_st}, {min_temp}/{max_temp}")
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax2.xaxis.set_minor_locator(AutoMinorLocator(4))
    # if molec in ["MeOH", "EG"]:
    #     max_temp =430
    ax2.set_xlim(min_temp*0.95, max_temp*1.05)
    
    ax2.tick_params("both", direction="in", which="both", length=4, labelsize=22, pad=10)
    ax2.tick_params("both", which="major", length=8)
    ax2.xaxis.set_ticks_position("both")
    ax2.yaxis.set_ticks_position("both")

    ax2.set_xlabel(r"$T$/K", fontsize=32, labelpad=10)
    titles = {"surf_tens": r"$\gamma$/mN$\cdot$m$^{-1}$", # r"$\mathregular{\gamma}$ (mN/m)"
              "liq_density": r"$\rho_{l}$/kg$\cdot$m$^{-3}$",
              "vap_density": r"$\mathregular{\rho_{v}}$/kg$\cdot$m$^{-3}$",
              "Pvap": r"$P_{vap}$/kPa",
              "Hvap": r"$H_{vap}$/kJ$\cdot$kg$^{-1}$",
              "diff_coeff": r"D (m$^2$/s)"}
    if prop_name in titles:
        ax2.set_ylabel(titles[prop_name], fontsize=32, labelpad=10)
    else:
        ax2.set_ylabel(prop_name, fontsize=32, labelpad=10)
    for axis in ['top','bottom','left','right']:
        ax2.spines[axis].set_linewidth(2.0)

    #Get legends and handles
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.78), ncol=2, fontsize=20, handletextpad=0.1, markerscale=0.9, edgecolor="dimgrey")
    # ax2.legend(loc="lower left", bbox_to_anchor=(-0.16, 1.03), ncol=2, fontsize=22, handletextpad=0.1, markerscale=0.9, edgecolor="dimgrey")
    if prop_name == "diff_coeff":
        #Put text in lower right
        ax2.text(0.60,  0.15, molec, fontsize=30, transform=ax2.transAxes)
    else:
        ax2.text(0.60,  0.82, molec, fontsize=30, transform=ax2.transAxes)
    fig.subplots_adjust(bottom=0.2, top=0.75, left=0.15, right=0.95, wspace=0.55)

    return fig

def plot_vle_envelopes(molec_dict, df_ff_dict, save_name = None):
    """
    Plot the density VLE envelopes for a given molecule and force field

    Parameters
    ----------
    molec_dict : dict
        Dictionary containing the molecule data
    df_ff_dict : dict
        Dictionary containing the force field data

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot
    """
    molec = list(molec_dict.keys())[0]
    mol_data = molec_dict[molec]
    # Plot VLE envelopes
    fig, ax2 = plt.subplots(1, 1, figsize=(6,6.1))    
    
    df_keys, df_ffs =  zip(*df_ff_dict.items())
    df_labels = list(df_keys)
    df_ff_list = list(df_ffs)

    key_map = {"Martinez-Jimenez et al.": ('gray', 's', 1),
               "Jorgensen": ('tab:orange', '>', 1),
               "Gonzalez-Salgado & Vega": ('tab:green', 'p', 1),
               "Chen et al.": ('purple', 'd', 1),
               "Stubbs et al.": ('purple', 'd', 1),
               "Huang et al.": ('gray', 's', 1),
               "Jahn et al.": ('gray', 's', 1),
               "Caleman et al.": ('gray', 's', 1),
               "Vahid & Maginn": ('tab:orange', '>', 1),
               "Chalaris & Samios": ('tab:green', 'p', 1),
               "Senapati": ('gray', 's', 1),
               "Borin & Skaf": ('tab:green', 'p', 1),
               "Garcia-Melgarejo et al.": ('gray', 's', 1),
               "Luo et al.": ('tab:orange', '>', 1),
               "Wang et al.": ('magenta', 'D', 1),
               "Old Opt FF": ('tab:blue', '+', 1),
               "IFT FF": ('tab:red', '^', 1),
               "Opt FF": ('tab:red', '^', 1),
               }
    
    cmap = plt.get_cmap("cool")  # Get the rainbow colormap
    num_lit = sum("AT" not in key for key in df_labels)
    df_colors = [cmap(i) for i in np.linspace(0, 1, len(df_ffs)-num_lit)]

    for i, key in enumerate(df_labels):
        if "AT-" in key: #color, marker, z_order)
            key_map[key] = (df_colors[i], "o", len(df_ff_list))

    # df_colors = [cmap(i) for i in np.linspace(0, 1, len(df_ffs)-5)] + ['gray', 'brown', 'orange', 'olive', 'olive']
    # df_labels, df_ffs = ["This Work", "GAFF", "Potoff et al.", "TraPPE", "Wang et al.", "Befort et al." ]
    # df_colors = ['blue', 'gray', '#0989d9', 'red', 'green','purple']
    # df_markers = ['o', 's', '^', '*', 'p', 'd']
    # df_z_order = [6,3,2,1,5,4]

    #Initialize min and max values
    min_temp = min(mol_data.expt_liq_density.keys())
    max_rho = max(mol_data.expt_liq_density.values())
    max_temp = max(mol_data.expt_liq_density.keys())
    min_rho = min(mol_data.expt_liq_density.values())
    # max_temp = mol_data.expt_Tc
    # min_rho = min(mol_data.expt_vap_density.values())
    # else:
    #     for df in df_ff_list:
    #         if df is not None:
    #             min_temp = min(df["temperature"].values)
    #             max_temp = max(df["temperature"].values)
    #             min_rho = min(df["sim_vap_density"].values)
    #             max_rho = max(df["sim_liq_density"].values)
    #             break
    liq_data_present = False
    vap_data_present = False
    label_prop = None
    for i in range(len(df_ff_list)):
        df_label = df_labels[i]
        df_ff = df_ff_list[i]

        df_color, df_marker, df_z_order = key_map[df_label]
        # df_label = df_labels[i] if df_labels[i] != "" else "Previous Work"
        
        if df_ff is not None and "Vahid" not in df_label:
            #Check that there are data points for vapor density
            all_props = ["sim_liq_density", "sim_vap_density", "sim_Tc", "sim_rhoc"]
            x_props = []
            has_vap = True
            has_liq = True
            label_prop = df_label
            # grouped = df_ff.groupby(["temperature", "atom_type"])[all_props]
            #Check that there are data points for liquid density for all df
            if df_ff["sim_liq_density"].isnull().all():
                has_liq = False
                label_prop = df_label
            else:
                x_props.append("sim_liq_density")
                liq_data_present = True
            #Check that there are data points for vapor density for all df
            if df_ff["sim_vap_density"].isnull().all():
                has_vap = False
                label_prop = df_label
            else:
                x_props.append("sim_vap_density")
                vap_data_present = True
    
            grouped = df_ff.groupby(["temperature"])[all_props]

            # Calculate mean and standard deviation for each group
            means = grouped.mean().reset_index()
            stds = grouped.std(ddof=0).reset_index()

            min_temp, max_temp = get_min_max(min_temp, max_temp, means["temperature"].values)

            for x_prop in x_props:
                min_rho, max_rho = get_min_max(min_rho, max_rho, means[x_prop].values, stds[x_prop].values)
                            
                # #Plot opt_scheme_ms vle curve
                if label_prop == "AT-Dis":
                    label_prop = "GP-Opt"
                elif label_prop == "IFT FF":
                    label_prop = "Base" #"Lowest " + r"$\gamma$" + " MAPD FF"
                ax2.errorbar(means[x_prop], means["temperature"], xerr=1.96*stds[x_prop],
                            color=df_color,markersize=10, linestyle='None', marker = df_marker, alpha=0.5, 
                            zorder = df_z_order, label=label_prop)

            #Plot critical points if available
            if has_vap and has_liq and molec != "DMSO":
                if df_label == "AT-Dis":
                    df_label = "GP-Opt"
                elif df_label == "IFT FF":
                    df_label = "Base" #"Lowest " + r"$\gamma$" + " MAPD FF"
                min_rho, max_rho = get_min_max(min_rho, max_rho, means["sim_rhoc"].values, stds["sim_rhoc"].values)
                min_temp, max_temp = get_min_max(min_temp, max_temp, means["sim_Tc"].values, stds["sim_Tc"].values)
                try:
                    ax2.errorbar(means["sim_rhoc"].dropna().iloc[0],means["sim_Tc"].dropna().iloc[0], xerr=1.96*stds["sim_rhoc"].dropna().iloc[0],
                            color=df_color,markersize=10, linestyle='None', marker = df_marker, alpha=0.5, 
                            zorder = df_z_order, label = df_label)
                except:
                    pass

    #Plot experimental data
    if liq_data_present or (not liq_data_present and not vap_data_present):
        ax2.scatter(mol_data.expt_liq_density.values(),mol_data.expt_liq_density.keys(),
            color="black",marker="x",linewidths=2,s=100,label="Experiment", zorder = 7)
    if vap_data_present or (not liq_data_present and not vap_data_present):
        ax2.scatter(mol_data.expt_vap_density.values(),mol_data.expt_vap_density.keys(),
            color="black",marker="x",linewidths=2,s=100, zorder = 7)
    if liq_data_present and vap_data_present and molec != "DMSO":
        ax2.scatter(mol_data.expt_rhoc, mol_data.expt_Tc, color="black", marker="x", linewidths=2, 
                    s=100, zorder = len(df_ff_list)+1)

    #Set Axes
    # ax2.set_xlim(min_rho*0.95,max_rho*1.05)
    # number_of_ticks = int(np.ceil((ax2.get_xlim()[1] - ax2.get_xlim()[0]) / 500))
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=4))
    # if number_of_ticks > 2:
    #     ax2.xaxis.set_major_locator(MultipleLocator(500))
    # else:
    #     ax2.xaxis.set_major_locator(MultipleLocator(200))
    ax2.xaxis.set_minor_locator(AutoMinorLocator(4))
    
    # ax2.set_ylim(min_temp*0.95, max_temp*1.05)
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=5))
    # ax2.yaxis.set_major_locator(MultipleLocator(40))
    ax2.yaxis.set_minor_locator(AutoMinorLocator(4))
    
    ax2.tick_params("both", direction="in", which="both", length=4, labelsize=20, pad=10)
    ax2.tick_params("both", which="major", length=8)
    ax2.xaxis.set_ticks_position("both")
    ax2.yaxis.set_ticks_position("both")

    ax2.set_ylabel(r"$T$/K", fontsize=32, labelpad=10)
    ax2.set_xlabel(r"$\rho$/kg$\cdot$m$^{-3}$", fontsize=32, labelpad=10)
    for axis in ['top','bottom','left','right']:
        ax2.spines[axis].set_linewidth(2.0)

    if molec not in ["R14", "R50", "R170", "R116"]:
        #Substitute mole string R w/ HFC
        molec = molec.replace("R","HFC")
    # handles, labels = ax2.get_legend_handles_labels()
    # for h in handles: h.set_linestyle("")

    # Collect handles and labels
    handles, labels = ax2.get_legend_handles_labels()

    # Remove duplicates while preserving order
    unique = dict()
    for h, l in zip(handles, labels):
        if l not in unique:
            unique[l] = h

    ax2.text(0.65,  0.82, molec, fontsize=30, transform=ax2.transAxes)
    fig.subplots_adjust(bottom=0.2, top=0.75, left=0.15, right=0.95, wspace=0.55)
    fig.legend(unique.values(), unique.keys(), loc="lower center", bbox_to_anchor=(0.5, 0.78), ncol=2, fontsize=20, handletextpad=0.1, markerscale=0.9, edgecolor="dimgrey")


    return fig

    # if save_name is not None:
    #     path = os.path.join(save_name, "vle_plt.png")
    #     fig.savefig(path,dpi=300)

def plot_pvap_hvap(molec_dict, df_ff_dict, save_name = None):
    """
    Plot the Hvap and Pvap values for a given molecule and force field

    Parameters
    ----------
    molec_dict : dict
        Dictionary containing the molecule data
    df_ff_dict : dict
        Dictionary containing the force field data

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot
    """
    molec = list(molec_dict.keys())[0]
    mol_data = molec_dict[molec]
    # Plot Pvap and Hvap
    
    df_keys, df_ffs =  zip(*df_ff_dict.items())
    df_labels = list(df_keys)
    df_ff_list = list(df_ffs)

    key_map = {"Martinez-Jimenez et al.": ('gray', 's', 1),
               "Jorgensen": ('tab:orange', '>', 1),
               "Gonzalez-Salgado & Vega": ('tab:green', 'p', 1),
               "Chen et al.": ('purple', 'd', 1),
               "Stubbs et al.": ('purple', 'd', 1),
               "Huang et al.": ('gray', 's', 1),
               "Jahn et al.": ('gray', 's', 1),
               "Caleman et al.": ('gray', 's', 1),
               "Vahid & Maginn": ('tab:orange', '>', 1),
               "Chalaris & Samios": ('tab:green', 'p', 1),
               "Senapati": ('gray', 's', 1),
               "Borin & Skaf": ('tab:green', 'p', 1),
               "Garcia-Melgarejo et al.": ('gray', 's', 1),
               "Luo et al.": ('tab:orange', '>', 1),
               "Wang et al.": ('magenta', 'D', 1),
               "Old Opt FF": ('tab:blue', '+', 1),
               "IFT FF": ('tab:red', '^', 1),
               "Opt FF": ('tab:red', '^', 1),
               }
    
    cmap = plt.get_cmap("cool")  # Get the rainbow colormap
    num_lit = sum("AT" not in key for key in df_labels)
    df_colors = [cmap(i) for i in np.linspace(0, 1, len(df_ffs)-num_lit)]

    for i, key in enumerate(df_labels):
        if "AT-" in key: #color, marker, z_order)
            key_map[key] = (df_colors[i], "o", len(df_ff_list))

    # df_labels = ["This Work", "GAFF", "Potoff et al.", "TraPPE", "Wang et al.", "Befort et al." ]
    # df_colors = ['blue', 'gray', '#0989d9', 'red', 'green','purple']
    # df_markers = ['o', 's', '^', '*', 'p', 'd']
    # df_z_order = [6,3,2,1,5,4]

    #Initialize min and max values
    min_temp = min(np.array(list(mol_data.expt_Pvap.keys())))
    max_temp = max(np.array(list(mol_data.expt_Pvap.keys())))
    min_pvap = min(np.log(np.array(list(mol_data.expt_Pvap.values()))*100)) #Convert from bar to kPa for plotting
    max_pvap = max(np.log(np.array(list(mol_data.expt_Pvap.values()))*100)) #Convert from bar to kPa for plotting
    # else:
    #     for df in df_ff_list:
    #         if df is not None:
    #             min_temp = min(df["temperature"].values)
    #             max_temp = max(df["temperature"].values)
    #             pvap_data = df["sim_Pvap"].values
    #             finite_pvap = pvap_data[np.isfinite(np.log(pvap_data))]
    #             min_pvap = np.nanmin(np.log(finite_pvap)) if finite_pvap.size > 0 else 0
    #             max_pvap = np.nanmax(np.log(df["sim_Pvap"].values))
    #             break

    min_hvap = min(mol_data.expt_Hvap.values())
    max_hvap = max(mol_data.expt_Hvap.values())
    # else:
    #     for df in df_ff_list:
    #         if df is not None:
    #             hvap_data = df["sim_Hvap"].values
    #             finite_hvap = hvap_data[np.isfinite(hvap_data)]
    #             min_hvap = np.min(finite_hvap) if finite_hvap.size > 0 else 0
    #             max_hvap = max(finite_hvap) if finite_hvap.size > 0 else 0
    #             break

    # Plot Pvap / Hvap
    fig, axs = plt.subplots(nrows=1, ncols=2,figsize=(12.2,6))
    #fig, ax1 = plt.subplots(1, 1, figsize=(6,6))

    #Loop over dfs of given ff results
    for i in range(len(df_ff_list)):
        df_label = df_labels[i]
        df_ff = df_ff_list[i]

        #Convert from bar to kPa for plotting
        if df_ff is not None and "Vahid" not in df_label:
            df_ff["sim_Pvap"] = df_ff["sim_Pvap"]*100


        df_color, df_marker, df_z_order = key_map[df_label]
        
        # df_label = df_labels[i] if df_labels[i] != "" else "Previous Work"

        if df_ff is not None and "Vahid" not in df_label:
            #Check if there are data points for Pvap and Hvap
            # x_props = ["sim_Pvap", "sim_Hvap"]
            df_ff.replace("", np.nan, inplace=True)
            # df_ff.dropna(subset=["sim_Pvap", "sim_Hvap"], inplace=True)
            #Check that there are data points for hvap for all df
            x_props = []
            has_pvap = True
            has_hvap = True
            #Check that there are data points for liquid density for all df
            if df_ff["sim_Pvap"].isnull().all():
                has_pvap = False
                label_prop = df_label
            else:
                x_props.append("sim_Pvap")
            #Check that there are data points for vapor density for all df
            if df_ff["sim_Hvap"].isnull().all():
                has_hvap = False
                label_prop = df_label
            else:
                x_props.append("sim_Hvap")

            # grouped = df_ff.groupby(["temperature", "atom_type"])[x_props]
            grouped = df_ff.groupby(["temperature"])[x_props]
            
            # Calculate mean and standard deviation for each group
            # grouped = grouped.replace("", np.nan)
            means = grouped.mean().reset_index()
            stds = grouped.std(ddof=0).reset_index()

            # print(df_label, molec)
            # print(means["sim_Pvap"].values, stds["sim_Pvap"].values)
            # print(len(means["sim_Pvap"].values), len(stds["sim_Pvap"].values))

            min_temp, max_temp = get_min_max(min_temp, max_temp, means["temperature"].values)
            
            
            #Plot 1/T vs log(Pvap) 
            #Plot if not all nan
            if has_pvap:
                finite_indices = np.where(means["sim_Pvap"].values > 0)[0]
                log_Pvap_finite =  np.log(means["sim_Pvap"].values[finite_indices])
                if len(log_Pvap_finite) > 0:
                    std_log_pvap = (stds["sim_Pvap"].values/means["sim_Pvap"].values)[finite_indices]
                    temps_finite = means["temperature"].values[finite_indices]
                    # print(df_label, molec)
                    # print(log_Pvap_finite, std_log_pvap)
                    # print(min_pvap, max_pvap)
                    min_pvap, max_pvap = get_min_max(min_pvap, max_pvap, log_Pvap_finite, std_log_pvap)
                    if df_label == "AT-Dis":
                        df_label = "GP-Opt"
                    elif df_label == "IFT FF":
                        df_label = "Base" #"Lowest " + r"$\gamma$" + " MAPD FF"
                    axs[0].errorbar(1000/temps_finite, log_Pvap_finite, yerr = 1.96*std_log_pvap,
                                color=df_color, markersize=10, linestyle='None', marker = df_marker, alpha=0.5, 
                                zorder = df_z_order,label = df_label)
                    # axs[0].scatter(1/means["temperature"], np.log(means["sim_Pvap"]), color=df_colors[i], 
                    #             s=70,alpha=0.5, label = df_label, marker = df_marker,
                    #             zorder = df_z_order)
            #Plot T vs Hvap
            if has_hvap and not np.all(np.isnan(means["sim_Hvap"].values)):
                # print(means["sim_Hvap"].values, stds["sim_Hvap"].values)
                finite_indices = np.isfinite(means["sim_Hvap"].values)
                Hvap_finite =  means["sim_Hvap"].values[finite_indices]
                std_hvap = stds["sim_Hvap"].values[finite_indices]
                temps_finite = means["temperature"].values[finite_indices]
                min_hvap, max_hvap = get_min_max(min_hvap, max_hvap, Hvap_finite, std_hvap)
                axs[1].errorbar(temps_finite, Hvap_finite, yerr=1.96*std_hvap,
                            color=df_color, markersize=10, linestyle='None', marker = df_marker, alpha=0.5, 
                            zorder = df_z_order,label = df_label)

        
    #Plot experimental pvap (kPa)
    axs[0].scatter(1000/np.array(list(mol_data.expt_Pvap.keys())),
                    np.log(np.array(list(mol_data.expt_Pvap.values()))*100),
        color="black",marker="x",label="Experiment",s=100,zorder = len(df_ff_list)+1)
    #Plot experimental Hvap
    axs[1].scatter(mol_data.expt_Hvap.keys(),mol_data.expt_Hvap.values(),
        color="black",marker="x",label="Experiment",s=100, zorder = len(df_ff_list)+1)

    #Set axes details
    # axs[0].set_xlim((1/max_temp)*0.95,(1/min_temp)*1.05)
    # axs[0].xaxis.set_major_locator(MultipleLocator(40))
    # axs[0].xaxis.set_minor_locator(AutoMinorLocator(4))

    min_mult = 1.05 if min_pvap <= -1 else 0.95
    max_mult = 1.05 if max_pvap >= 1 else 0.95

    # axs[0].set_ylim(min_pvap * min_mult, max_pvap * max_mult)
    # axs[0].yaxis.set_major_locator(MultipleLocator(10))
    # axs[0].yaxis.set_minor_locator(AutoMinorLocator(5))

    axs[0].tick_params("both", direction="in", which="both", length=4, labelsize=20, pad=10)
    axs[0].tick_params("both", which="major", length=8)
    axs[0].xaxis.set_ticks_position("both")
    axs[0].yaxis.set_ticks_position("both")

    axs[0].set_xlabel(r"$1000\cdot T^{-1}$" + r"/$\mathregular{K^{-1}}$", fontsize=32, labelpad=10)
    axs[0].set_ylabel(r"$\mathregular{ln}(P_{vap}$/kPa)", fontsize=32, labelpad=10)

    # axs[1].set_xlim(min_temp*0.95,max_temp*1.05)
    # axs[1].xaxis.set_major_locator(MultipleLocator(40))
    # axs[1].xaxis.set_minor_locator(AutoMinorLocator(4))

    # axs[1].set_ylim(min_hvap*0.95, max_hvap*1.05)
    # axs[1].yaxis.set_major_locator(MultipleLocator(100))
    # axs[1].yaxis.set_minor_locator(AutoMinorLocator(5))

    axs[1].tick_params("both", direction="in", which="both", length=4, labelsize=20, pad=10)
    axs[1].tick_params("both", which="major", length=8)
    axs[1].xaxis.set_ticks_position("both")
    axs[1].yaxis.set_ticks_position("both")

    axs[1].set_xlabel(r"$T$/K", fontsize=32, labelpad=10)
    axs[1].set_ylabel(r"$\Delta H_{vap}$/kJ$\cdot$kg$^{-1}$", fontsize=32, labelpad=10)

    if molec not in ["R14", "R50", "R170", "R116"]:
        #Substitute mole string R w/ HFC
        molec = molec.replace("R","HFC")
    axs[0].text(0.08, 0.15, molec, fontsize=30, transform=axs[0].transAxes)

    for axis in ['top','bottom','left','right']:
        axs[0].spines[axis].set_linewidth(2.0)
        axs[1].spines[axis].set_linewidth(2.0)

    #Get unique labels for legend from axs 0 and 1
    # Collect handles and labels from each subplot
    handles0, labels0 = axs[0].get_legend_handles_labels()
    handles1, labels1 = axs[1].get_legend_handles_labels()

    # Combine them
    handles = handles0 + handles1
    labels = labels0 + labels1

    # Remove duplicates while preserving order
    unique = dict()
    for h, l in zip(handles, labels):
        if l not in unique:
            unique[l] = h

    fig.legend(unique.values(), unique.keys(), loc="lower center", bbox_to_anchor=(0.5, 0.88), ncol=2, fontsize=20, handletextpad=0.1, markerscale=0.8, edgecolor="dimgrey")

    fig.subplots_adjust(bottom=0.15, top=0.85, left=0.15, right=0.85, wspace=0.55, hspace=0.5)

    return fig
    # if save_name is not None:
    #     path = os.path.join(save_name, "h_p_vap_plt.png")
    #     fig.savefig(path,dpi=300)

def plot_err_each_prop(molec_names, err_path_dict, obj = 'mapd', save_name = None):
    """
    Plot the error for each property for a given molecule and force field

    Parameters
    ----------
    molec_names : list
        List of molecule names
    err_path_dict : dict
        Dictionary containing the error data for each force field
    obj : str
        The type of error to plot (e.g., 'mapd' or 'mae')
    save_name : str, optional
        The name of the file to save the plot to
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot
    """
    props = ["liq_density", "vap_density", "Pvap", "Hvap", "surf_tens"]
    cols = [obj + "_" + prop for prop in props]
    if obj == "mae":
        names = [r"\rho_l" + r"/kg$\cdot$m$^{-3}$", r"\rho_v" + r"/kg$\cdot$m$^{-3}$", r"$P_{vap}$" + r"/bar", r"$\Delta H_{vap}$" + r"/kJ$\cdot$kg$^{-1}$", r"$\gamma$" + r"/mN$\cdot$m$^{-1}$"]
    else:
        names = [r"\rho_l", r"\rho_v", r"$P_{vap}$", r"$\Delta H_{vap}$", r"$\gamma$"]
    # cols = [item for item in cols for _ in range(2)]
    # names = [item for item in names for _ in range(2)]
    
    df_keys, df_ffs =  zip(*err_path_dict.items())
    df_labels = list(df_keys)
    df_mse_list = list(df_ffs)

    train_molecs = ["EG","Gly", "MeOH", "DMSO","DEC","DMF"]
    results = []
    for label, df in zip(df_labels, df_mse_list):
        # Get columns that exist in the dataframe
        # existing_cols = list(set(cols).intersection(df.columns))
        existing_cols = [c for c in cols if c in df.columns]
        if len(existing_cols) > 0:
            # Compute mean error for available columns
            avg_errors = df[existing_cols].mean()
            # Store results in list as dictionary
            result_row = {"method": label}
            result_row.update(avg_errors.to_dict())
            results.append(result_row)

    # Convert to pandas DataFrame
    results_df = pd.DataFrame(results)
    #Sort alphabetically by Method
    # results_df = results_df.sort_values(by="method").reset_index(drop=True)
    if save_name is not None:
        results_df.to_csv(save_name + ".csv", index=False)
    # for label, df in zip(df_labels, df_mse_list):
    #     # Get the columns which exitst in the dataframe
    #     existing_cols = list(set(cols).intersection(df.columns))
    #     print(existing_cols)
    #     if len(existing_cols) > 0:
    #         #Get the average errors for training and testing molecules for each property
    #         avg_errors = df[existing_cols].mean()
    #         print(f"Average {obj} for all molecules using {label}:")
    #         print(avg_errors)

    cmap = plt.get_cmap("cool")  # Get the rainbow colormap
    #Choose color basede on FF label (ATs diff colors Dis=blue, 1=red, 2=orange, literature =gray
    def get_color(label):
        if "AT-Dis" in label:
            return "blue"
        elif "AT-3" in label:
            return 'red'
        elif "AT-4" in label:
            return 'orange'
        else:
            return 'gray'

    #Get indeces where train molecules are in all molecules
    # len_train = len(set(molec_names).intersection(train_molecs))
    len_train = len([m for m in molec_names if m in train_molecs])
    left_indices = np.arange(len_train)
    right_indices =  np.arange(len_train, len(molec_names))

    if len(right_indices) > 0:
        axs_col = 2
        cols = [item for item in cols for _ in range(2)]
        names = [item for item in names for _ in range(2)]
    else:
        axs_col = 1
        cols = [item for item in cols]
        names = [item for item in names]

    fig, axs = plt.subplots(len(props), axs_col, figsize=(3*len(df_mse_list), 3*len(props)), sharex = False)
    # Plot each column in a subplot
    label_set = False
    for i, (ax, column, name) in enumerate(zip(axs.flatten(), cols, names)):
        bar_width = 0.1
        max_val_f = 0

        if axs_col == 1:
            indices = left_indices
            mol_names = molec_names[:len_train]
        else:
            indices = right_indices
            mol_names = molec_names[len_train:]

        for j, df in enumerate([results_df]):
            if column in df.columns:
                if j < len(df_mse_list):
                    max_val = np.nanmax(df[column].values)
                    max_val_f = max(max_val, max_val_f)
                    max_val_f = max_val_f if not np.isnan(max_val_f) else 1
            label=df_labels[j].split("_")[0] if not label_set else None
            ax.bar(indices + j*bar_width, df[column], bar_width, label=label, color = get_color(df_labels[j])) #df[column].iloc[indices]
            
        ax.set_ylim(0, max_val_f*1.05)
        ax.set_title(name, fontsize = 24) 
        ax.set_xticks(indices + bar_width)
        ax.tick_params(axis='y', labelsize=20)

        molec_names_use = []
        for molec in mol_names:
            if molec not in ["R14", "R50", "R170", "R116"]:
                #Substitute mole string R w/ HFC
                molec_names_use.append(molec.replace("R","HFC"))
            else:
                molec_names_use.append(molec)

        ax.set_xticklabels(molec_names_use, fontsize=20)
    
    if axs_col == 2:
        handles, labels = axs[0, 0].get_legend_handles_labels()
    else:
        handles, labels = axs[0].get_legend_handles_labels()
    #Drop duplicate labels
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fontsize = 20)

    # Adjust layout
    fig.supxlabel('Molecule', fontsize = 24)
    if obj == "mapd":
        titley = obj.upper() + " (%)"
    else:
        titley = obj.upper()
    fig.supylabel(titley, fontsize = 24)

    # Add Training and Testing labels
    if axs_col == 2:
        fig.text(0.075, 0.99, "Training Set", ha="left", va="top", fontsize=20)
        fig.text(0.99, 0.99, "Testing Set", ha="right", va="top", fontsize=20)

    #Explain missing HFC-143 data
#     fig.text(
#     0.85, 0.1,  # Adjust these coordinates based on the image placement
#     "Experimental\n Data\n Unavailable",
#     fontsize=20,
#     color="black",
#     ha="center",  # Center horizontally
#     va="center",  # Center vertically
#     bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3")
# )

    plt.tight_layout(rect=[0.01, 0.0, 1, 1])
    #Save figure to jpg
    # if save_name is not None:
    #     plt.savefig(save_name + ".png", bbox_inches='tight')
    # Show the plot
    return fig

# def plot_err_each_prop(molec_names, err_path_dict, obj = 'mapd', save_name = None):
#     """
#     Plot the error for each property for a given molecule and force field

#     Parameters
#     ----------
#     molec_names : list
#         List of molecule names
#     err_path_dict : dict
#         Dictionary containing the error data for each force field
#     obj : str
#         The type of error to plot (e.g., 'mapd' or 'mae')
#     save_name : str, optional
#         The name of the file to save the plot to
    
#     Returns
#     -------
#     fig : matplotlib.figure.Figure
#         The figure object containing the plot
#     """
#     props = ["liq_density", "vap_density", "Pvap", "Hvap", "surf_tens"]
#     cols = [obj + "_" + prop for prop in props]
#     if obj == "mae":
#         names = ["Liquid Density " + r"$(kg/m^3)$", "Vapor Density " + r"$(kg/m^3)$", "Vapor Pressure " + r"$(bar)$", "Heat of Vaporization " + r"$(kJ/kg)$", "Surface Tension " + r"$(mN/m)$"]
#     else:
#         names = ["Liquid Density", "Vapor Density", "Vapor Pressure", "Heat of Vaporization", "Surface Tension"]
#     # cols = [item for item in cols for _ in range(2)]
#     # names = [item for item in names for _ in range(2)]
    
#     df_keys, df_ffs =  zip(*err_path_dict.items())
#     df_labels = list(df_keys)
#     df_mse_list = list(df_ffs)

#     train_molecs = ["EG","Gly", "MeOH", "DMSO","DEC","DMF"]
#     results = []
#     for label, df in zip(df_labels, df_mse_list):
#         # Get columns that exist in the dataframe
#         # existing_cols = list(set(cols).intersection(df.columns))
#         existing_cols = [c for c in cols if c in df.columns]
#         if len(existing_cols) > 0:
#             # Compute mean error for available columns
#             avg_errors = df[existing_cols].mean()
#             # Store results in list as dictionary
#             result_row = {"method": label}
#             result_row.update(avg_errors.to_dict())
#             results.append(result_row)

#     # Convert to pandas DataFrame
#     results_df = pd.DataFrame(results)
#     if save_name is not None:
#         results_df.to_csv(save_name + ".csv", index=False)
#     # for label, df in zip(df_labels, df_mse_list):
#     #     # Get the columns which exitst in the dataframe
#     #     existing_cols = list(set(cols).intersection(df.columns))
#     #     print(existing_cols)
#     #     if len(existing_cols) > 0:
#     #         #Get the average errors for training and testing molecules for each property
#     #         avg_errors = df[existing_cols].mean()
#     #         print(f"Average {obj} for all molecules using {label}:")
#     #         print(avg_errors)

#     cmap = plt.get_cmap("cool")  # Get the rainbow colormap
#     #Choose color basede on FF label (ATs diff colors Dis=blue, 1=red, 2=orange, literature =gray
#     def get_color(label):
#         if "AT-Dis" in label:
#             return "blue"
#         elif "AT-3" in label:
#             return 'red'
#         elif "AT-4" in label:
#             return 'orange'
#         else:
#             return 'gray'

#     #Get indeces where train molecules are in all molecules
#     # len_train = len(set(molec_names).intersection(train_molecs))
#     len_train = len([m for m in molec_names if m in train_molecs])
#     left_indices = np.arange(len_train)
#     right_indices =  np.arange(len_train, len(molec_names))

#     if len(right_indices) > 0:
#         axs_col = 2
#         cols = [item for item in cols for _ in range(2)]
#         names = [item for item in names for _ in range(2)]
#     else:
#         axs_col = 1
#         cols = [item for item in cols]
#         names = [item for item in names]

#     print(len(props), axs_col)

#     fig, axs = plt.subplots(len(props), axs_col, figsize=(6*len(props), 8*axs_col), sharex = False)
#     # Plot each column in a subplot
#     for i, (ax, column, name) in enumerate(zip(axs.flatten(), cols, names)):
#         bar_width = 0.1
#         max_val_f = 0

#         if axs_col == 1:
#             indices = left_indices
#             mol_names = molec_names[:len_train]
#         else:
#             indices = right_indices
#             mol_names = molec_names[len_train:]
#         print(mol_names)
#         for j, df in enumerate(df_mse_list):
#             if column in df.columns:
#                 if j < len(df_mse_list):
#                     max_val = np.nanmax(df[column].values)
#                     max_val_f = max(max_val, max_val_f)
#                     max_val_f = max_val_f if not np.isnan(max_val_f) else 1
#                 ax.bar(indices + j*bar_width, df[column].iloc[indices], bar_width, label=df_labels[j].split("_")[0], color = get_color(df_labels[j]))
            
#         ax.set_ylim(0, max_val_f*1.05)
#         ax.set_title(name, fontsize = 24) 
#         ax.set_xticks(indices + bar_width)
#         ax.tick_params(axis='y', labelsize=20)

#         molec_names_use = []
#         for molec in mol_names:
#             if molec not in ["R14", "R50", "R170", "R116"]:
#                 #Substitute mole string R w/ HFC
#                 molec_names_use.append(molec.replace("R","HFC"))
#             else:
#                 molec_names_use.append(molec)

#         ax.set_xticklabels(molec_names_use, fontsize=20)
    
#     if axs_col == 2:
#         handles, labels = axs[0, 0].get_legend_handles_labels()
#     else:
#         handles, labels = axs[0].get_legend_handles_labels()
#     fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=3, fontsize = 20)

#     # Adjust layout
#     fig.supxlabel('Molecule', fontsize = 24)
#     if obj == "mapd":
#         titley = obj.upper() + " (%)"
#     else:
#         titley = obj.upper()
#     fig.supylabel(titley, fontsize = 24)

#     # Add Training and Testing labels
#     fig.text(0.075, 0.99, "Training Set", ha="left", va="top", fontsize=20)
#     if axs_col == 2:
#         fig.text(0.99, 0.99, "Testing Set", ha="right", va="top", fontsize=20)

#     #Explain missing HFC-143 data
# #     fig.text(
# #     0.85, 0.1,  # Adjust these coordinates based on the image placement
# #     "Experimental\n Data\n Unavailable",
# #     fontsize=20,
# #     color="black",
# #     ha="center",  # Center horizontally
# #     va="center",  # Center vertically
# #     bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3")
# # )

#     plt.tight_layout(rect=[0.01, 0.0, 1, 1])
#     #Save figure to jpg
#     # if save_name is not None:
#     #     plt.savefig(save_name + ".png", bbox_inches='tight')
#     # Show the plot
#     return fig

def plot_err_avg_props(molec_names, err_path_dict, obj = 'mapd', save_name = None):
    """
    Plot the average error for each property for a given molecule and force field

    Parameters
    ----------
    molec_names : list
        List of molecule names
    err_path_dict : dict
        Dictionary containing the error data for each force field
    obj : str
        The type of error to plot (e.g., 'mapd' or 'mae')
    save_name : str, optional
        The name of the file to save the plot to

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot
    """
    #Load our results, Gaff results, and old result MAPD values
    # df_labels = ["This Work", "GAFF", "Wang et al.", "Befort et al." ]
    # df_colors = ['blue', 'gray', 'green','purple']
    # props = ["liq_density", "vap_density", "Pvap", "Hvap", "Tc", "rhoc"]
    # cols = [obj + "_" + prop for prop in props]
    # df_mse_list = []
    # for key in list(MSE_path_dict.keys()):
    #     df_mse = pd.read_csv(MSE_path_dict[key], header = 0, index_col = "molecule")
    #     df_mse_list.append(df_mse.reindex(molec_names))

    props = ["liq_density", "vap_density", "Pvap", "Hvap", "surf_tens"]
    cols = [obj + "_" + prop for prop in props]
    # names = ["Liquid Density " + r"$(kg/m^3)$", "Vapor Density " + r"$(kg/m^3)$", "Vapor Pressure " + r"$(bar)$", "Heat of Vaporization " + r"$(kJ/kg)$", "Surface Tension " + r"$(mN/m)$"]
    names = [r"\rho_l" + r"/kg$\cdot$m$^{-3}$", r"\rho_v" + r"/kg$\cdot$m$^{-3}$", r"$P_{vap}$" + r"/bar", r"$\Delta H_{vap}$" + r"/kJ$\cdot$kg$^{-1}$", r"$\gamma$" + r"/mN$\cdot$m$^{-1}$"]
    cols = [item for item in cols for _ in range(2)]
    names = [item for item in names for _ in range(2)]
    
    df_keys, df_ffs =  zip(*err_path_dict.items())
    df_labels = list(df_keys)
    df_mse_list = list(df_ffs)

    cmap = plt.get_cmap("cool")  # Get the rainbow colormap
    df_colors = [cmap(i) for i in np.linspace(0, 1, len(df_ffs)-3)] + ['gray', 'olive', 'olive']

    train_molecs = ["EG" , "Gly", "ACN", "MeOH", "DMSO", "THF", "DCM", "DEC", "DMF"]
    #Get indeces where train molecules are in all molecules
    len_train = len(set(molec_names).intersection(train_molecs))
    left_labels = molec_names[:len_train]
    right_labels = molec_names[len_train:]

    # #Get Avg MAPD values for each molecule and each property + get min and max values
    df_avg_list = []
    for df in df_mse_list:
        df_avg = df[cols].agg(['mean', 'min', 'max'], axis=1)
        df_avg.columns = [obj, 'Min', 'Max']
        df_avg_list.append(df_avg.reindex(molec_names))

    #Merge the dataframes
    merged_df = pd.concat(df_avg_list, axis=1, keys=df_labels)
    #Group by molecule and take average and print
    #Split into  dfs baed on train_molecs
    merged_df_train = merged_df.loc[merged_df.index.isin(train_molecs)]
    merged_df_test = merged_df.loc[~merged_df.index.isin(train_molecs)]

    def compute_average_mapd(df, mapd_columns, obj):
        # Select the columns that contain 'mapd' for each scheme
        mapd_columns = [col for col in df.columns if obj in col]
        # # Calculate the average MAPD for each scheme
        average_mapd = df[mapd_columns].mean().sort_values().reset_index().iloc[:, [0, -1]]
        #Ignore objective column
        average_mapd.columns = ['Molecule', 'Average ' + obj.upper()]
        return average_mapd

    average_mapd = compute_average_mapd(merged_df, cols, obj)
    average_train_mapd = compute_average_mapd(merged_df_train, cols, obj)
    average_test_mapd = compute_average_mapd(merged_df_test, cols, obj)
    #Sort by average MAPD
    print("Overall Average:\n", average_mapd)
    print("Train Average:\n", average_train_mapd)
    print("Test Average:\n", average_test_mapd)

    # Plot the merged DataFrame
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(24, 8), sharex=False)

    for i in range(len(df_labels)):
        label = df_labels[i]
        color = df_colors[i]
        # y_err = [merged_df[label][obj] - merged_df[label]['Min'], merged_df[label]['Max'] - merged_df[label][obj]]
        y_err = [merged_df[label][obj] - merged_df[label]['Min'], merged_df[label]['Max'] - merged_df[label][obj]]
        y_err = np.array(y_err)  # Convert to numpy array for easier slicing
        # Plot for training data on the left subplot
        merged_df[label][obj].iloc[:len_train].plot(
            kind='bar', color=color, ax=ax_left, 
            yerr=y_err[:, :len_train],  # Slicing y_err for left subplot
            position=i, width=0.1, label=label, rot=0
        )
        
        # Plot for test data on the right subplot
        merged_df[label][obj].iloc[len_train:].plot(
            kind='bar', color=color, ax=ax_right, 
            yerr=y_err[:, len_train:],  # Slicing y_err for right subplot
            position=i, width=0.1, label=label, rot=0
        )
        # merged_df[label][obj].plot(kind='bar', color=color, ax=ax, yerr =y_err, position=i, width=0.1, label=label, rot = 0)

    ax_left.set_xlim(-0.4, len_train - 0.6)  # Adjust the xlim based on train set size
    ax_right.set_xlim(-0.4, len(merged_df.index) - len_train - 0.6)  # Adjust for test set size
    
    # ax.set_ylabel('Average ' + obj.upper())
    ax_right.legend(loc = 'upper right', fontsize = 20)
    ax_right.tick_params(axis='both', labelsize=20)
    ax_left.tick_params(axis='both', labelsize=20)

    # ax_left.set_xticklabels([])  # Removes x-axis labels on the left subplot
    # ax_right.set_xticklabels([])  # Removes x-axis labels on the right subplot
    ax_left.set_xlabel('')  # Ensure no x-axis label for the left subplot
    ax_right.set_xlabel('')  # Ensure no x-axis label for the right subplot

    # Add Training and Testing labels
    fig.text(0.05, 0.99, "Training Set", ha="left", va="top", fontsize=20)
    fig.text(0.99, 0.99, "Testing Set", ha="right", va="top", fontsize=20)

    fig.suptitle(obj.upper() + ' Comparison for Different Solvents', fontsize = 20)
    fig.supxlabel('Molecule', fontsize = 20)
    fig.supylabel('Average ' + obj.upper(), fontsize = 20)
    plt.tight_layout(rect=[0.01, 0.0, 1, 1])

    # Show the plot
    return fig