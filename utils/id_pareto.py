import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error

def prepare_df_dens_errors(df, mol_name, root_dir, iter_num):
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
    df=df.sort_values(by=["temperature", "dens-iter"])
    for group, values in df.groupby(['dens-iter']):
        new_quantities = {}
        #The molecule is listed as the first value in the group
        molecule = mol_name
        if len(values) > 0:
            # Temperatures
            temps = values["temperature"].values

            #Add experimental data
            values["expt_liq_density"] = values["temperature"].apply(
                lambda temp: molecule.expt_liq_density[temp])
            values["expt_vap_density"] = values["temperature"].apply(
                lambda temp: molecule.expt_vap_density[temp] )
        
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

            for prop in ["liq_density", "surf_tens"]:
                mse, mapd, mae = calculate_objs(values["expt_" + prop], values["md_" + prop], prop, group[0])
                new_quantities["mse_" + prop] = mse
                new_quantities["mapd_" + prop] = mapd
                new_quantities["mae_" + prop] = mae

        else:
            for prop in ["liq_density", "surf_tens"]:
                new_quantities["mse_" + prop] = np.nan
                new_quantities["mapd_" + prop] = np.nan
                new_quantities["mae_" + prop] = np.nan
        
        data_to_append = list(group) + list(new_quantities.values())
        new_data.append(data_to_append)

    columns = list(["molecule"]) + list(new_quantities.keys())
    new_df = pd.DataFrame(new_data, columns=columns)

    dir_name = root_dir + "dens-iter-" + str(iter_num) + "/"
    os.makedirs(dir_name, exist_ok=True)
    csv_name = os.path.join(dir_name, "result_errors.csv")
    new_df.to_csv(csv_name)
        
    return new_df

def select_final_pareto(df_pareto, root_dir, iter_num):
    # Filter for parameter sets with less than 3 % error in all properties
    df_final = df_pareto.drop(
        columns=[
            "md_liq_density",
            "md_surf_tens",
            "mse_surf_tens",
            "mse_liq_density",
            "mae_surf_tens",
            "mae_liq_density",
            "is_pareto",
        ]
    )

    ### Choosing Final Parameter Sets (R-32)
    # Filter for parameter sets with less than 5 % error in all properties
    df_final = df_final[
        (df_final["mape_surf_tens"] <= 5.0)
        & (df_final["mape_liq_density"] <= 5.0)
    ]

    # Save CSV files
    dir_name = root_dir + "dens-iter-" + str(iter_num) + "/"
    os.makedirs(dir_name, exist_ok=True)
    csv_name = os.path.join(dir_name, "final-params.csv")
    df_final.to_csv(csv_name)
    csv_name = root_dir + "final-params.csv"
    df_final.to_csv(csv_name)

    return df_final