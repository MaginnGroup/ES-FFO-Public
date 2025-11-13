import signac
import os
import sys
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import pickle
from scipy.stats import linregress
import copy

from fffit.fffit.utils import values_real_to_scaled
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error

def prepare_df_props(df_csv, molecule, liquid_density_threshold, scale=True):
    """Prepare a pandas dataframe for fitting a GP model to density data

    Performs the following actions:
       - Renames "prop" to "sim_prop". Ex "liq_density" to "sim_liq_density"
       - Adds "expt_prop" for each property. Ex adds "expt_liq_density"
       - Adds "is_liquid"
       - Converts all values from physical values to scaled values if scale is True

    Parameters
    ----------
    df_csv : pd.DataFrame
        The dataframe as loaded from a CSV file with the signac results
    molecule : RXXConstants
        An instance of a molecule constants class
    liquid_density_threshold : float
        Density threshold (kg/m^3) for distinguishing liquid and vapor
    scale : bool
        Whether to scale the values between 0 and 1 or not. Default is True (scale values)

    Returns
    -------
    df_all : pd.DataFrame
        The dataframe with scaled parameters, temperature, density, and is_liquid
    df_liquid : pd.DataFrame
        `df_all` where `is_liquid` is True
    df_vapor : pd.DataFrame
        `df_all` where `is_liquid` is False
    """

    #Get a list of the column names that do not include sigma, epsilon, or temperature
    excluded_keywords = ["sigma", "epsilon", "temperature"]
    filtered_cols = [col for col in df_csv.columns if not any(k in col.lower() for k in excluded_keywords)]

    if "liq_density" not in filtered_cols:
        raise ValueError("df_csv must contain column 'liq_density'")
    if "temperature" not in df_csv.columns:
        raise ValueError("df_csv must contain column 'temperature'")
    for param in list(molecule.param_names):
        if param not in df_csv.columns:
            raise ValueError(f"df_csv must contain a column for parameter: '{param}'")
        
    if "vap_enthalpy" in df_csv.columns:
        df_csv.drop(columns="vap_enthalpy", inplace=True)
    if "liq_enthalpy" in df_csv.columns:
        df_csv.drop(columns="liq_enthalpy", inplace=True)
        
    # Add expt density and is_liquid
    df_all = copy.deepcopy(df_csv.rename(columns={"liq_density": "sim_liq_density"}))
    df_all["expt_liq_density"] = df_all["temperature"].map(molecule.expt_liq_density)
    df_all["is_liquid"] = df_all["sim_liq_density"] > liquid_density_threshold

    # Create scaling for all values
    scaling_info = {
    "temperature": molecule.temperature_bounds(),
    "sim_liq_density": molecule.liq_density_bounds,
    "expt_liq_density": molecule.liq_density_bounds,}

    # Optional columns and corresponding bounds + expt mappings
    optional_props = {
        "surf_tens": ("expt_surf_tens", molecule.surf_tens_bounds, molecule.expt_surf_tens),
        "Pvap": ("expt_Pvap", molecule.Pvap_bounds, molecule.expt_Pvap),
        "Hvap": ("expt_Hvap", molecule.Hvap_bounds, molecule.expt_Hvap),
        "vap_density": ("expt_vap_density", molecule.vap_density_bounds, molecule.expt_vap_density),
        "diff_coeff": ("expt_diff_coeff", molecule.diff_coeff_bounds, molecule.expt_diff_coeff),
    }

    # Add optional columns if they exist
    for old_col, (expt_col, bounds_func, expt_map) in optional_props.items():
        if old_col in df_all.columns and bounds_func is not None: #If the column exists and bounds are defined
            sim_col = "sim_" + old_col
            df_all.rename(columns={old_col: sim_col}, inplace=True)
            scaling_info[sim_col] = bounds_func
            scaling_info[expt_col] = bounds_func
            # Map experimental values to temperatures
            try:
                df_all[expt_col] = df_all["temperature"].map(expt_map)
            #If temperature does not exist in the mapping make it nan
            except KeyError: 
                df_all[expt_col] = np.nan

    if "sim_liq_density" in df_all.columns and "sim_vap_density" in df_all.columns:
        Tc, rhoc = calc_critical(df_all)
        df_all["sim_Tc"] = Tc
        df_all["sim_rhoc"] = rhoc
        df_all["expt_Tc"] = molecule.expt_Tc
        df_all["expt_rhoc"] = molecule.expt_rhoc

    if scale:
        # Scale param values
        df_all[list(molecule.param_names)] = values_real_to_scaled(
            df_all[list(molecule.param_names)], molecule.param_bounds
        )

        # Scale other properties
        for col, bounds_func in scaling_info.items():
            #Scale the column if we have bounds defined
            if bounds_func is not None:
                df_all[col] = values_real_to_scaled(df_all[col], bounds_func)
            #Delete the column if we do not have bounds defined
            else:
                df_all.drop(columns=col, inplace=True)

    # Split out vapor and liquid samples
    df_liquid = df_all[df_all["is_liquid"] == True]
    df_vapor = df_all[df_all["is_liquid"] == False]

    return df_all, df_liquid, df_vapor

def calc_critical(df):
    """
    Computes the critical temperature and density with the law of rectilinear diameters

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with columns "temperature", "sim_liq_density", "sim_vap_density"
    Returns
    -------
    Tc : list
        Critical temperature for each molecule
    rhoc : list
        Critical density for each molecule
    """
    Tc = []
    rhoc = []
    values = df
    # for group, values in df.groupby(['molecule']):    
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

def prepare_df_errors(df_data, data_dict, mol_name):
    """Create a dataframe with mean square error (mse) and mean absolute
    percent error (mape) for each unique parameter set. The critical
    temperature and density are also evaluated.

    Parameters
    ----------
    df_data : pandas.Dataframe
        all simulation results
    data_dict : EsolvsConstants
        Molecule class EsolvsConstants with bounds/experimental data
    mol_name : str
        name of the molecule to be evaluated

    Returns
    -------
    new_df : pandas.Dataframe
        dataframe with one row per parameter set and including
        the MSE and MAPD for liq_density, vap_density, pvap, hvap,
        critical temperature, critical density
    """
    molecule = data_dict[mol_name]
    #sort by molecule and temperature -- added by Ning Wang
    new_data = []
    if "iter" in df_data.columns:
        df = df_data.sort_values(by=["temperature", "iter"])
    else:
        df = df_data.sort_values(by=["temperature"])

    if list(molecule.param_names)[0] in df.columns:
        #For data with known parameters
        groupby_data = df.groupby(list(molecule.param_names))
        group_keys = list(molecule.param_names)
    else:
        #For literature csv data files
        groupby_data = df.groupby(["molecule"])
        group_keys = ["molecule"]
    #Sort by param names to be able to save these values
    for group, values in groupby_data:
        new_quantities = {}

        if len(values) > 0:
            # Temperatures
            temps = values["temperature"].values

            #Add experimental data
            all_props = {
                "liq_density": ("expt_liq_density", molecule.expt_liq_density),
                "surf_tens": ("expt_surf_tens", molecule.expt_surf_tens),
                "Pvap": ("expt_Pvap", molecule.expt_Pvap),
                "Hvap": ("expt_Hvap", molecule.expt_Hvap),
                "vap_density": ("expt_vap_density", molecule.expt_vap_density),
                "Tc": ("expt_Tc", molecule.expt_Tc),
                "rhoc": ("expt_rhoc", molecule.expt_rhoc),
                "diff_coeff": ("expt_diff_coeff", molecule.expt_diff_coeff),}
            
            # Add optional columns if they exist
            for old_col, (expt_col, expt_map) in all_props.items():
                #For columns that exist where the exp data is not already in the dataframe
                if "sim_" + old_col in values.columns and not "expt_" + old_col in values.columns:
                    #For Tc and rhoc, add a single value directly
                    if old_col in ["Tc", "rhoc"]:
                        values[expt_col] = np.array([expt_map])
                    #Otherwise map the values to the temperature
                    else:
                        try:
                            values[expt_col] = values["temperature"].map(expt_map)
                        #If temperature does not exist in the mapping make it nan
                        except KeyError: 
                            values[expt_col] = np.nan
        
            def calculate_objs(expt_values, sim_values, property_name, molecule_name):
                try:
                    #Find indeces where both expt and sim values are finite
                    finite_indices = np.isfinite(expt_values) & np.isfinite(sim_values)
                    fin_sim = sim_values[finite_indices]
                    fin_expt = expt_values[finite_indices]
                    mse = mean_squared_error(fin_expt, fin_sim)
                    mapd = mean_absolute_percentage_error(fin_expt, fin_sim) * 100.0
                    mae = mean_absolute_error(fin_expt, fin_sim)
                    pct_errors = (fin_sim - fin_expt)/fin_expt * 100.0
                    mpd = np.average(pct_errors)
                except ValueError as e:
                    print(f"Error in calculating {property_name} for {molecule_name}: {e}. Setting MSE, MAE, and MAPD to NaN")
                    # print("Exp", expt_values, "\n Sim", sim_values)
                    mse, mapd, mae, mpd = np.nan, np.nan, np.nan, np.nan
                return mse, mapd, mae, mpd

            for prop in ["liq_density", "surf_tens", "vap_density", "Pvap", "Hvap", "diff_coeff"]:
                if "sim_" + prop in values.columns:
                    mse, mapd, mae, mpd = calculate_objs(values["expt_" + prop], values["sim_" + prop], prop, mol_name)
                    new_quantities["mse_" + prop] = mse
                    new_quantities["mapd_" + prop] = mapd
                    new_quantities["mae_" + prop] = mae
                    new_quantities["mpd_" + prop] = mpd

            for prop in ["Tc", "rhoc"]:
                if "sim_" + prop in values.columns:
                    mse, mapd, mae, mpd = calculate_objs(np.array([values["expt_" + prop].values[0]]), np.array([values["sim_" + prop].values[0]]), prop, mol_name)
                    new_quantities["mse_" + prop] = mse
                    new_quantities["mapd_" + prop] = mapd
                    new_quantities["mae_" + prop] = mae
                    new_quantities["mpd_" + prop] = mpd

        else:
            for prop in list(all_props.keys()):
                new_quantities["mse_" + prop] = np.nan
                new_quantities["mapd_" + prop] = np.nan
                new_quantities["mae_" + prop] = np.nan
                new_quantities["mpd_" + prop] = np.nan
        
        data_to_append = list(group) + list(new_quantities.values())
        new_data.append(data_to_append)
 
    columns = group_keys + list(new_quantities.keys())
    new_df = pd.DataFrame(new_data, columns=columns)
        
    return new_df