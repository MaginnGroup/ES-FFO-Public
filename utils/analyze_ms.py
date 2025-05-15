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
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from fffit.fffit.utils import values_real_to_scaled, values_scaled_to_real, variances_scaled_to_real
from fffit.fffit.plot import plot_model_performance, plot_model_vs_test, plot_slices_temperature, plot_slices_params, plot_model_vs_exp, plot_obj_contour
from .molec_class_files import r14, r32, r50, r125, r134a, r143a, r170, r41, r23, r161, r152a, r152, r134, r143, r116

R14 = r14.R14Constants()
R32 = r32.R32Constants()
R50 = r50.R50Constants()
R125 = r125.R125Constants()
R134a = r134a.R134aConstants()
R143a = r143a.R143aConstants()
R170 = r170.R170Constants()
R41 = r41.R41Constants()
R23 = r23.R23Constants()
R161 = r161.R161Constants()
R152a = r152a.R152aConstants()
R152 = r152.R152Constants()
R143 = r143.R143Constants()
R134 = r134.R134Constants()
R116 = r116.R116Constants()

molec_dict = {"R14": R14,
                "R32": R32,
                "R50": R50,
                "R125": R125,
                "R134a": R134a,
                "R143a": R143a,
                "R170": R170,
                "R41": R41,
                "R23": R23,
                "R161":R161,
                "R152a":R152a,
                "R152": R152,
                "R143": R143,
                "R134": R134,
                "R116": R116}

def block_average(data):
    """
    Calculate block averages

    Implements the technique of Flyvbjerg and Peterson
    J. Chem. Phys. 91, 461 (1989). Also described in
    Appendix D of Frenkel and Smit. This function implements
    equation D.3.4 of Frenkel and Smit.

    The data should be provided as a numpy ndarray with
    shape=(npoints,). The function performs N blocking
    operations, where the block size is 2^N and the
    maximum number of blocking operations is determined
    from the number of points in ``data``

    Parameters:
    -----------
    data : numpy.ndarray, shape=(npoints,)
        numpy array with shape (npoints,) where npoints
        is the number of data point in the sample

    Returns:
    -------
    means, vars_est, vars_err

    means : np.ndarray, shape=(n_avg_ops,)
        mean values calculated from different numbers
        of blocking operations
    vars_est : np.ndarray, shape=(n_avg_ops,)
        estimates of the variances of the average from different
        numbers of blocking operations
    vars_err: np.ndarray, shape=(n_avg_ops,)
        estimates of the error in the variances from different
        numbers of blocking operations
    """

    try:
        data = np.asarray(data)
    except:
        raise TypeError("data should be provided as a numpy.ndarray")

    means = []
    vars_est = []
    vars_err = []

    n_samples = data.shape[0]

    max_blocking_ops = 0
    block_length = 1
    while block_length < 1.0 / 4.0 * n_samples:
        max_blocking_ops += 1
        block_length = 2 ** max_blocking_ops

    # Calc stats for mulitple-of-two block lengths
    for m in range(max_blocking_ops):
        block_length = 2 ** m  # Number of datapoints in each block
        n_blocks = int(
            n_samples / block_length
        )  # Number of blocks we can get with given block size
        # Calculate the 'new' dataset by block averaging
        block_data = [
            np.mean(
                data[i * block_length : (i + 1) * block_length],
                dtype=np.float64,
            )
            for i in range(n_blocks)
        ]
        block_data = np.asarray(block_data, dtype=np.float64)
        # Calculate the mean of this new dataset
        mean = np.mean(block_data, dtype=np.float64)
        # Calculate the variance of this new dataset
        var = np.var(block_data, dtype=np.float64)
        var_err = math.sqrt(2.0 * var ** 2.0 / (n_blocks - 1) ** 3.0)

        # Save data for blocking op
        means.append(mean)
        vars_est.append(var / (n_blocks - 1))
        vars_err.append(var_err)

    return np.asarray(means), np.asarray(vars_est), np.asarray(vars_err)

def prepare_df_vle(df_csv, molec_dict, csv_name = None, drop_one = False):
    """Prepare a pandas dataframe for fitting a GP model to density data

    Performs the following actions:
       - Renames "liq_density" to "sim_liq_density" (units kg/m3 assumed)
       - Renames "vap_density" to "sim_vap_density" (units kg/m3 assumed)
       - Renames "Pvap" to "sim_Pvap" (units bar assumed)
       - Removes "liq_enthalpy" and "vap_enthalpy" and adds "sim_Hvap" (units kJ/kg returned)
            - Units kJ/mol assumed for Hvap and kJ/kg for Hvap kJ/kg

    Parameters
    ----------
    df_csv : pd.DataFrame
        The dataframe as loaded from a CSV file with the signac results
    molec_dict : {"Rxx": RxxConstants, ...}
        A dictionary mapping molecule names to classes
    n_molecules : int
        The number of molecules in the simulation

    Returns
    -------
    df_all : pd.DataFrame
        The dataframe with scaled parameters and MD/expt. properties
    """

    def rename_col(df, property_name, units):
        prop_units = property_name + " " + units
        if property_name == "temperature":
            sim_str = ""
        else:
            sim_str = "sim_"
        if prop_units in df.columns:
            df.rename(columns={prop_units: sim_str + property_name}, inplace=True)
        elif property_name in df.columns:
            df.rename(columns={property_name: sim_str + property_name}, inplace=True)
        else:
            raise ValueError(f"df must contain either {property_name} or {prop_units}")
        
        return df
    
    # Convert Hvap to kJ/kg if in kJ/mol
    if "Hvap" in df_csv.columns:
        #Add Hvap in kJ/kg
        df_csv["Hvap kJ/kg"] = df_csv["Hvap"]*1000.0/df_csv["molecule"].apply(
            lambda molec: molec_dict[molec].molecular_weight)
        #And drop kJ/mol column
        df_csv.drop("Hvap", axis=1)
    
    # Rename properties to MD
    props = ["liq_density", "vap_density", "Pvap", "Hvap", "temperature"]
    units = ["kg/m3", "kg/m3", "bar", "kJ/kg", "K"]
    for prop, unit in zip(props, units):
        rename_col(df_csv, prop, unit)
        
    #sort by molecule and temperature -- added by Ning Wang
    df_csv.dropna(subset=["sim_liq_density", "sim_vap_density"], how = "any", inplace=True)
    df_csv.sort_values(by=["molecule", "temperature"], inplace=True)

    #Add Tc and Rhoc predictions
    Tc, rhoc = calc_critical(df_csv)
    df_csv["sim_Tc"] = Tc
    df_csv["sim_rhoc"] = rhoc

    #Drop any molecule and temperature with only 1 data point
    if drop_one:
        df_csv = df_csv.groupby(["molecule", "temperature"]).filter(lambda x: len(x) > 1)

    if csv_name != None:
        df_csv.to_csv(csv_name)
           
    return df_csv

def calc_critical(df):
    """Compute the critical temperature and density

    Accepts a dataframe with "T_K", "rholiq_kgm3" and "rhovap_kgm3"
    Returns the critical temperature (K) and density (kg/m3)

    Computes the critical properties with the law of rectilinear diameters
    """
    Tc = []
    rhoc = []
    for group, values in df.groupby(['molecule']):    
        #Need to group by molecule and do this for each molecule
        temps = values["temperature"].values
        liq_density = values["sim_liq_density"].values
        vap_density = values["sim_vap_density"].values

        #Check that all temps are not the same
        if all(x == temps[0] for x in temps):
            Tc += [np.nan]*len(temps)
            rhoc += [np.nan]*len(temps)
        else:
            # Critical Point (Law of rectilinear diameters)
            slope1, intercept1, r_value1, p_value1, std_err1 = linregress(
                temps,(liq_density + vap_density) / 2.0,)

            try:
                slope2, intercept2, r_value2, p_value2, std_err2 = linregress(
                    temps,(liq_density - vap_density)**(1/0.32),)
            except:
                slope2, intercept2, r_value2, p_value2, std_err2 = linregress(
                    temps,abs((liq_density - vap_density))**(1/0.32),)

            Tc_mol = np.abs(intercept2 / slope2)
            rhoc_mol = intercept1 + slope1 * Tc_mol

            # if len(temps) == 5:
            Tc += list([Tc_mol])*len(temps)
            rhoc += list([rhoc_mol])*len(temps)
        
    return Tc, rhoc

def prepare_df_vle_errors(df, molec_dict, csv_name = None):
    """Create a dataframe with mean square error (mse) and mean absolute
    percent error (mape) for each unique parameter set. The critical
    temperature and density are also evaluated.

    Parameters
    ----------
    df : pandas.Dataframe
        per simulation results
    molecule : R143a
        molecule class with bounds/experimental data

    Returns
    -------
    df_new : pandas.Dataframe
        dataframe with one row per parameter set and including
        the MSE and MAPD for liq_density, vap_density, pvap, hvap,
        critical temperature, critical density
    """
    new_data = []

    #sort by molecule and temperature -- added by Ning Wang
    df=df.sort_values(by=["molecule", "temperature"])
    molecules = df['molecule'].unique().tolist()
    for group, values in df.groupby(['molecule']):
        new_quantities = {}
        #The molecule is listed as the first value in the group
        molecule = molec_dict[values["molecule"].values[0]]
        if group[0] not in ["R134", "R152"] and len(values) > 0:
            # Temperatures
            temps = values["temperature"].values

            #Add experimental data (if not R134, 143 or R152)
            values["expt_liq_density"] = values["temperature"].apply(
                lambda temp: molecule.expt_liq_density[int(temp)])
            values["expt_vap_density"] = values["temperature"].apply(
                lambda temp: molecule.expt_vap_density[int(temp)] )
            values["expt_Pvap"] = values["temperature"].apply(
                lambda temp: molecule.expt_Pvap[int(temp)])
            values["expt_Hvap"] = values["temperature"].apply(
                lambda temp: molecule.expt_Hvap[int(temp)])
            # Critical Point (Law of rectilinear diameters)
            values["expt_Tc"] =  molecule.expt_Tc
            values["expt_rhoc"] = molecule.expt_rhoc
        
            def calculate_objs(expt_values, sim_values, property_name, molecule_name):
                try:
                    fin_sim = sim_values[np.isfinite(sim_values)]
                    fin_expt = expt_values[np.isfinite(sim_values)]
                    mse = mean_squared_error(fin_expt, fin_sim)
                    mapd = mean_absolute_percentage_error(fin_expt, fin_sim) * 100.0
                    mae = mean_absolute_error(fin_expt, fin_sim)
                except ValueError as e:
                    print(f"Error in calculating {property_name} for {molecule_name}: {e}. Setting MSE, MAE, and MAPD to NaN")
                    print("Exp", expt_values, "\n Sim", sim_values)
                    mse, mapd, mae = np.nan, np.nan, np.nan
                return mse, mapd, mae

            for prop in ["liq_density", "vap_density", "Pvap", "Hvap"]:
                mse, mapd, mae = calculate_objs(values["expt_" + prop], values["sim_" + prop], prop, group[0])
                new_quantities["mse_" + prop] = mse
                new_quantities["mapd_" + prop] = mapd
                new_quantities["mae_" + prop] = mae

            for prop in ["Tc", "rhoc"]:
                mse, mapd, mae = calculate_objs(np.array([values["expt_" + prop].values[0]]), np.array([values["sim_" + prop].values[0]]), prop, group[0])
                new_quantities["mse_" + prop] = mse
                new_quantities["mapd_" + prop] = mapd
                new_quantities["mae_" + prop] = mae
        else:
            for prop in ["liq_density", "vap_density", "Pvap", "Hvap", "Tc", "rhoc"]:
                new_quantities["mse_" + prop] = np.nan
                new_quantities["mapd_" + prop] = np.nan
                new_quantities["mae_" + prop] = np.nan
        
        data_to_append = list(group) + list(new_quantities.values())
        # print(data_to_append)
        new_data.append(data_to_append)

    columns = list(["molecule"]) + list(new_quantities.keys())
    new_df = pd.DataFrame(new_data, columns=columns)

    if csv_name != None:
        new_df.to_csv(csv_name)

    return new_df

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


def plot_vle_envelopes(molec_dict, df_ff_dict, save_name = None):
    molec = list(molec_dict.keys())[0]
    mol_data = molec_dict[molec]
    # Plot VLE envelopes
    fig, ax2 = plt.subplots(1, 1, figsize=(6,6))    
    
    df_keys, df_ffs =  zip(*df_ff_dict.items())
    df_labels = list(df_keys)
    df_ff_list = list(df_ffs)

    cmap = plt.get_cmap("cool")  # Get the rainbow colormap
    df_colors = [cmap(i) for i in np.linspace(0, 1, len(df_ffs)-5)] + ['gray', 'brown', 'orange', 'olive', 'olive']
    # df_labels, df_ffs = ["This Work", "GAFF", "Potoff et al.", "TraPPE", "Wang et al.", "Befort et al." ]
    # df_colors = ['blue', 'gray', '#0989d9', 'red', 'green','purple']
    # df_markers = ['o', 's', '^', '*', 'p', 'd']
    # df_z_order = [6,3,2,1,5,4]

    #Initialize min and max values
    if molec not in ["R152", "R134"]:
        min_temp = min(mol_data.expt_liq_density.keys())
        max_temp = mol_data.expt_Tc
        min_rho = min(mol_data.expt_vap_density.values())
        max_rho = max(mol_data.expt_liq_density.values())
    else:
        for df in df_ff_list:
            if df is not None:
                min_temp = min(df["temperature"].values)
                max_temp = max(df["temperature"].values)
                min_rho = min(df["sim_vap_density"].values)
                max_rho = max(df["sim_liq_density"].values)
                break

    for i in range(len(df_ff_list)):
        df_label = df_labels[i]
        df_ff = df_ff_list[i]

        if "AT-" in df_label:
            df_z_order = len(df_ff_list)
            df_marker = "o"
        elif "GAFF" in df_label:
            df_z_order = 3
            df_marker = "s"
        elif "Potoff" in df_label:
            df_z_order = 2
            df_marker = "^"
        elif "TraPPE" in df_label:
            df_z_order = 1
            df_marker = "*"
        else:
            df_z_order = 4
            df_marker = "p"

        df_label = df_labels[i] if df_labels[i] != "" else "Previous Work"
        
        if df_ff is not None:
            all_props = ["sim_liq_density", "sim_vap_density", "sim_Tc", "sim_rhoc"]
            grouped = df_ff.groupby(["temperature", "atom_type"])[all_props]
            
            x_props = ["sim_liq_density", "sim_vap_density"]
            # Calculate mean and standard deviation for each group
            means = grouped.mean().reset_index()
            stds = grouped.std(ddof=0).reset_index()

            for x_prop in x_props:
                #Set new max and mins
                min_rho, max_rho = get_min_max(min_rho, max_rho, means[x_prop].values, stds[x_prop].values)
                
                # #Plot opt_scheme_ms vle curve
                ax2.errorbar(means[x_prop], means["temperature"], xerr=1.96*stds[x_prop],
                            color=df_colors[i],markersize=10, linestyle='None', marker = df_marker, alpha=0.5, 
                            zorder = df_z_order,)

            #Plot critical points
            min_temp, max_temp = get_min_max(min_temp, max_temp, means["sim_Tc"].values, stds["sim_Tc"].values)
            ax2.errorbar(means["sim_rhoc"].values[0],means["sim_Tc"].values[0], xerr=1.96*stds["sim_rhoc"].values[0],
                        color=df_colors[i],markersize=10, linestyle='None', marker = df_marker, alpha=0.5, 
                        zorder = df_z_order, label = df_label)

    #Plot experimental data
    if molec not in ["R152", "R134"]:
        ax2.scatter(mol_data.expt_liq_density.values(),mol_data.expt_liq_density.keys(),
            color="black",marker="x",linewidths=2,s=100,label="Experiment", zorder = 7)
        ax2.scatter(mol_data.expt_vap_density.values(),mol_data.expt_vap_density.keys(),
            color="black",marker="x",linewidths=2,s=100, zorder = 7)
        ax2.scatter(mol_data.expt_rhoc, mol_data.expt_Tc, color="black", marker="x", linewidths=2, 
                    s=100, zorder = 7)

    #Set Axes
    ax2.set_xlim(min_rho*0.95,max_rho*1.05)
    number_of_ticks = int(np.ceil((ax2.get_xlim()[1] - ax2.get_xlim()[0]) / 500))
    if number_of_ticks > 2:
        ax2.xaxis.set_major_locator(MultipleLocator(500))
    else:
        ax2.xaxis.set_major_locator(MultipleLocator(200))
    ax2.xaxis.set_minor_locator(AutoMinorLocator(4))
    
    ax2.set_ylim(min_temp*0.95, max_temp*1.05)
    ax2.yaxis.set_major_locator(MultipleLocator(40))
    ax2.yaxis.set_minor_locator(AutoMinorLocator(4))
    
    ax2.tick_params("both", direction="in", which="both", length=4, labelsize=26, pad=10)
    ax2.tick_params("both", which="major", length=8)
    ax2.xaxis.set_ticks_position("both")
    ax2.yaxis.set_ticks_position("both")

    ax2.set_ylabel("T (K)", fontsize=32, labelpad=10)
    ax2.set_xlabel(r"$\mathregular{\rho}$ (kg/m$\mathregular{^3}$)", fontsize=32, labelpad=15)
    for axis in ['top','bottom','left','right']:
        ax2.spines[axis].set_linewidth(2.0)

    if molec not in ["R14", "R50", "R170", "R116"]:
        #Substitute mole string R w/ HFC
        molec = molec.replace("R","HFC")
    # handles, labels = ax2.get_legend_handles_labels()
    # for h in handles: h.set_linestyle("")
    ax2.legend(loc="lower left", bbox_to_anchor=(-0.16, 1.03), ncol=2, fontsize=22, handletextpad=0.1, markerscale=0.9, edgecolor="dimgrey")
    ax2.text(0.60,  0.82, molec, fontsize=30, transform=ax2.transAxes)
    fig.subplots_adjust(bottom=0.2, top=0.75, left=0.15, right=0.95, wspace=0.55)

    return fig

    # if save_name is not None:
    #     path = os.path.join(save_name, "vle_plt.png")
    #     fig.savefig(path,dpi=300)

def plot_pvap_hvap(molec_dict, df_ff_dict, save_name = None):
    molec = list(molec_dict.keys())[0]
    mol_data = molec_dict[molec]
    # Plot Pvap and Hvap
    
    df_keys, df_ffs =  zip(*df_ff_dict.items())
    df_labels = list(df_keys)
    df_ff_list = list(df_ffs)

    cmap = plt.get_cmap("cool")  # Get the rainbow colormap
    df_colors = [cmap(i) for i in np.linspace(0, 1, len(df_ffs)-5)] + ['gray', 'brown', 'orange', 'olive', 'olive']

    # df_labels = ["This Work", "GAFF", "Potoff et al.", "TraPPE", "Wang et al.", "Befort et al." ]
    # df_colors = ['blue', 'gray', '#0989d9', 'red', 'green','purple']
    # df_markers = ['o', 's', '^', '*', 'p', 'd']
    # df_z_order = [6,3,2,1,5,4]

    #Initialize min and max values
    if molec not in ["R152", "R134"]:
        min_temp = max(np.array(list(mol_data.expt_Pvap.keys())))
        max_temp = mol_data.expt_Tc
        min_pvap = min(np.log(np.array(list(mol_data.expt_Pvap.values()))))
        max_pvap = max(np.log(np.array(list(mol_data.expt_Pvap.values()))))
    else:
        for df in df_ff_list:
            if df is not None:
                min_temp = min(df["temperature"].values)
                max_temp = max(df["temperature"].values)
                pvap_data = df["sim_Pvap"].values
                finite_pvap = pvap_data[np.isfinite(np.log(pvap_data))]
                min_pvap = np.nanmin(np.log(finite_pvap)) if finite_pvap.size > 0 else 0
                max_pvap = np.nanmax(np.log(df["sim_Pvap"].values))
                break

    if molec not in ["R152", "R134", "R143"]:
        min_hvap = min(mol_data.expt_Hvap.values())
        max_hvap = max(mol_data.expt_Hvap.values())
    else:
        for df in df_ff_list:
            if df is not None:
                hvap_data = df["sim_Hvap"].values
                finite_hvap = hvap_data[np.isfinite(hvap_data)]
                min_hvap = np.min(finite_hvap) if finite_hvap.size > 0 else 0
                max_hvap = max(finite_hvap) if finite_hvap.size > 0 else 0
                break

    # Plot Pvap / Hvap
    fig, axs = plt.subplots(nrows=1, ncols=2,figsize=(12,6))
    #fig, ax1 = plt.subplots(1, 1, figsize=(6,6))

    #Loop over dfs of given ff results
    for i in range(len(df_ff_list)):
        df_label = df_labels[i]
        df_ff = df_ff_list[i]

        if "AT-" in df_label:
            df_z_order = len(df_ff_list)
            df_marker = "o"
        elif "GAFF" in df_label:
            df_z_order = 3
            df_marker = "s"
        elif "Potoff" in df_label:
            df_z_order = 2
            df_marker = "^"
        elif "TraPPE" in df_label:
            df_z_order = 1
            df_marker = "*"
        else:
            df_z_order = 4
            df_marker = "p"
        
        df_label = df_labels[i] if df_labels[i] != "" else "Previous Work"

        if df_ff is not None: #and ("Potoff" in df_label or "GAFF" in df_label)
            x_props = ["sim_Pvap", "sim_Hvap"]
            df_ff.replace("", np.nan, inplace=True)
            df_ff.dropna(subset=["sim_Pvap", "sim_Hvap"], inplace=True)
            grouped = df_ff.groupby(["temperature", "atom_type"])[x_props]
            
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
            finite_indices = np.where(means["sim_Pvap"].values > 0)[0]
            log_Pvap_finite =  np.log(means["sim_Pvap"].values[finite_indices])
            if len(log_Pvap_finite) > 0:
                std_log_pvap = (stds["sim_Pvap"].values/means["sim_Pvap"].values)[finite_indices]
                temps_finite = means["temperature"].values[finite_indices]
                # print(df_label, molec)
                # print(log_Pvap_finite, std_log_pvap)
                # print(min_pvap, max_pvap)
                min_pvap, max_pvap = get_min_max(min_pvap, max_pvap, log_Pvap_finite, std_log_pvap)
                axs[0].errorbar(1/temps_finite, log_Pvap_finite, yerr = std_log_pvap,
                            color=df_colors[i], markersize=10, linestyle='None', marker = df_marker, alpha=0.5, 
                            zorder = df_z_order,label = df_label)
                # axs[0].scatter(1/means["temperature"], np.log(means["sim_Pvap"]), color=df_colors[i], 
                #             s=70,alpha=0.5, label = df_label, marker = df_marker,
                #             zorder = df_z_order)
            #Plot T vs Hvap
            if not np.all(np.isnan(means["sim_Hvap"].values)):
                # print(means["sim_Hvap"].values, stds["sim_Hvap"].values)
                finite_indices = np.isfinite(means["sim_Hvap"].values)
                Hvap_finite =  means["sim_Hvap"].values[finite_indices]
                std_hvap = stds["sim_Hvap"].values[finite_indices]
                temps_finite = means["temperature"].values[finite_indices]
                
                    
                min_hvap, max_hvap = get_min_max(min_hvap, max_hvap, Hvap_finite, std_hvap)
                # if "AT-" in df_label and molec == "R152":
                #     print(min_hvap, max_hvap)
                #     print(Hvap_finite, std_hvap)
                axs[1].errorbar(temps_finite, Hvap_finite, yerr=1.96*std_hvap,
                            color=df_colors[i], markersize=10, linestyle='None', marker = df_marker, alpha=0.5, 
                            zorder = df_z_order,)
                # axs[1].scatter(means["temperature"],means["sim_Hvap"], color=df_colors[i], 
                #             s=70,alpha=0.5, marker = df_marker,
                #             zorder = df_z_order)
        
    #Plot experimental pvap
    if molec not in ["R152", "R134"]:
        axs[0].scatter(1/np.array(list(mol_data.expt_Pvap.keys())),
                       np.log(np.array(list(mol_data.expt_Pvap.values()))),
            color="black",marker="x",label="Experiment",s=100,zorder = 7)
    #Plot experimental Hvap
    if molec not in ["R152", "R134", "R143"]:
        axs[1].scatter(mol_data.expt_Hvap.keys(),mol_data.expt_Hvap.values(),
            color="black",marker="x",label="Experiment",s=100, zorder = 7)

    #Set axes details
    axs[0].set_xlim((1/max_temp)*0.95,(1/min_temp)*1.05)
    # axs[0].xaxis.set_major_locator(MultipleLocator(40))
    # axs[0].xaxis.set_minor_locator(AutoMinorLocator(4))

    axs[0].set_ylim(min_pvap*0.8,max_pvap*1.05)
    # axs[0].yaxis.set_major_locator(MultipleLocator(10))
    # axs[0].yaxis.set_minor_locator(AutoMinorLocator(5))

    axs[0].tick_params("both", direction="in", which="both", length=2, labelsize=16, pad=10)
    axs[0].tick_params("both", which="major", length=4)
    axs[0].xaxis.set_ticks_position("both")
    axs[0].yaxis.set_ticks_position("both")

    axs[0].set_xlabel("1/T " + r"$\mathregular{K^{-1}}$", fontsize=16, labelpad=8)
    axs[0].set_ylabel(r"$\mathregular{ln(P_{vap})}$ (bar)", fontsize=16, labelpad=8)

    axs[1].set_xlim(min_temp*0.95,max_temp*1.05)
    # axs[1].xaxis.set_major_locator(MultipleLocator(40))
    # axs[1].xaxis.set_minor_locator(AutoMinorLocator(4))

    axs[1].set_ylim(min_hvap*0.95, max_hvap*1.05)
    # axs[1].yaxis.set_major_locator(MultipleLocator(100))
    # axs[1].yaxis.set_minor_locator(AutoMinorLocator(5))

    axs[1].tick_params("both", direction="in", which="both", length=2, labelsize=16, pad=10)
    axs[1].tick_params("both", which="major", length=4)
    axs[1].xaxis.set_ticks_position("both")
    axs[1].yaxis.set_ticks_position("both")

    axs[1].set_xlabel("T (K)", fontsize=16, labelpad=8)
    axs[1].set_ylabel(r"$\mathregular{\Delta H_{vap}}$ (kJ/kg)", fontsize=16, labelpad=8)

    if molec not in ["R14", "R50", "R170", "R116"]:
        #Substitute mole string R w/ HFC
        molec = molec.replace("R","HFC")
    axs[0].text(0.08, 0.3, molec, fontsize=20, transform=axs[0].transAxes)
    axs[0].legend(loc="lower left", bbox_to_anchor=(0.35, 1.05), ncol=3, fontsize=16, handletextpad=0.1, markerscale=0.8, edgecolor="dimgrey")

    fig.subplots_adjust(bottom=0.15, top=0.85, left=0.15, right=0.85, wspace=0.55, hspace=0.5)

    return fig
    # if save_name is not None:
    #     path = os.path.join(save_name, "h_p_vap_plt.png")
    #     fig.savefig(path,dpi=300)

def plot_err_each_prop(molec_names, err_path_dict, obj = 'mapd', save_name = None):
    props = ["liq_density", "vap_density", "Pvap", "Hvap"]
    cols = [obj + "_" + prop for prop in props]
    if obj == "mae":
        names = ["Liquid Density " + r"$(kg/m^3)$", "Vapor Density " + r"$(kg/m^3)$", "Vapor Pressure " + r"$(bar)$", "Heat of Vaporization " + r"$(kJ/kg)$"]
    else:
        names = ["Liquid Density", "Vapor Density", "Vapor Pressure", "Heat of Vaporization"]
    cols = [item for item in cols for _ in range(2)]
    names = [item for item in names for _ in range(2)]
    
    df_keys, df_ffs =  zip(*err_path_dict.items())
    df_labels = list(df_keys)
    df_mse_list = list(df_ffs)

    cmap = plt.get_cmap("cool")  # Get the rainbow colormap
    df_colors = [cmap(i) for i in np.linspace(0, 1, len(df_ffs)-3)] + ['gray', 'olive', 'olive']

    train_molecs = ["R14", "R32", "R50", "R170", "R125", "R134a", "R143a", "R41"]
    #Get indeces where train molecules are in all molecules
    len_train = len(set(molec_names).intersection(train_molecs))
    left_indices = np.arange(len_train)
    right_indices =  np.arange(len_train, len(molec_names))

    fig, axs = plt.subplots(4, 2, figsize=(24, 16), sharex = False)
    # Plot each column in a subplot
    for i, (ax, column, name) in enumerate(zip(axs.flatten(), cols, names)):
        bar_width = 0.1
        max_val_f = 0

        if i % 2 ==0:
            indices = left_indices
            mol_names = molec_names[:len_train]
        else:
            indices = right_indices
            mol_names = molec_names[len_train:]

        for j, df in enumerate(df_mse_list):
            if j < len(df_mse_list):
                max_val = np.nanmax(df[column].values)
                max_val_f = max(max_val, max_val_f)
            ax.bar(indices + j*bar_width, df[column].iloc[indices], bar_width, label=df_labels[j], color = df_colors[j])
        
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
    
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=3, fontsize = 20)

    # Adjust layout
    fig.supxlabel('Molecule', fontsize = 24)
    if obj == "mapd":
        titley = obj.upper() + " (%)"
    else:
        titley = obj.upper()
    fig.supylabel(titley, fontsize = 24)

    # Add Training and Testing labels
    fig.text(0.075, 0.99, "Training Set", ha="left", va="top", fontsize=20)
    fig.text(0.99, 0.99, "Testing Set", ha="right", va="top", fontsize=20)

    #Explain missing HFC-143 data
    fig.text(
    0.85, 0.1,  # Adjust these coordinates based on the image placement
    "Experimental\n Data\n Unavailable",
    fontsize=20,
    color="black",
    ha="center",  # Center horizontally
    va="center",  # Center vertically
    bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3")
)

    plt.tight_layout(rect=[0.01, 0.0, 1, 1])
    #Save figure to jpg
    if save_name is not None:
        plt.savefig(save_name, bbox_inches='tight')
    # Show the plot
    return fig

def plot_err_avg_props(molec_names, err_path_dict, obj = 'mapd', save_name = None):
    #Load our results, Gaff results, and old result MAPD values
    # df_labels = ["This Work", "GAFF", "Wang et al.", "Befort et al." ]
    # df_colors = ['blue', 'gray', 'green','purple']
    # props = ["liq_density", "vap_density", "Pvap", "Hvap", "Tc", "rhoc"]
    # cols = [obj + "_" + prop for prop in props]
    # df_mse_list = []
    # for key in list(MSE_path_dict.keys()):
    #     df_mse = pd.read_csv(MSE_path_dict[key], header = 0, index_col = "molecule")
    #     df_mse_list.append(df_mse.reindex(molec_names))

    props = ["liq_density", "vap_density", "Pvap", "Hvap"]
    cols = [obj + "_" + prop for prop in props]
    names = ["Liquid Density " + r"$(kg/m^3)$", "Vapor Density " + r"$(kg/m^3)$", "Vapor Pressure " + r"$(bar)$", "Heat of Vaporization " + r"$(kJ/kg)$"]
    cols = [item for item in cols for _ in range(2)]
    names = [item for item in names for _ in range(2)]
    
    df_keys, df_ffs =  zip(*err_path_dict.items())
    df_labels = list(df_keys)
    df_mse_list = list(df_ffs)

    cmap = plt.get_cmap("cool")  # Get the rainbow colormap
    df_colors = [cmap(i) for i in np.linspace(0, 1, len(df_ffs)-3)] + ['gray', 'olive', 'olive']

    train_molecs = ["R14", "R32", "R50", "R170", "R125", "R134a", "R143a", "R41"]
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

    fig.suptitle(obj.upper() + ' Comparison for Different Refrigerants', fontsize = 20)
    fig.supxlabel('Molecule', fontsize = 20)
    fig.supylabel('Average ' + obj.upper(), fontsize = 20)
    plt.tight_layout(rect=[0.01, 0.0, 1, 1])

    # Show the plot
    return fig