import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error

from fffit.fffit.utils import (
    values_real_to_scaled,
    values_scaled_to_real,
)

sys.path.append("../")

from utils.r134a import R134aConstants
from utils.id_new_samples import prepare_df_vle
from utils.analyze_samples import prepare_df_vle_errors
from utils.plot import plot_property, render_mpl_table

from fffit.pareto import find_pareto_set, is_pareto_efficient

R134a = R134aConstants()

import matplotlib._color_data as mcd

############################# QUANTITIES TO EDIT #############################
##############################################################################

iternum = 1

##############################################################################
##############################################################################

csv_path = "/scratch365/nwang2/ff_development/HFC_143a_FFO_FF/r134a/analysis/csv/"
in_csv_names = [
    "r134a-vle-iter" + str(i) + "-results.csv" for i in range(1, iternum + 1)
]
out_csv_name = "r134a-pareto-iter1.csv"

# Read files
df_csvs = [
    pd.read_csv(csv_path + in_csv_name, index_col=0)
    for in_csv_name in in_csv_names
]
df_csv = pd.concat(df_csvs)
df_all = prepare_df_vle(df_csv, R134a)

def main():
    # ID pareto points
    result, pareto_points, dominated_points = find_pareto_set(
        df_paramsets.filter(["mse_liq_density", "mse_surf_tens"]).values,
        is_pareto_efficient
    )
    df_paramsets = df_paramsets.join(pd.DataFrame(result, columns=["is_pareto"]))

    df_paramsets[df_paramsets["is_pareto"]==True].to_csv(csv_path + "/" + out_csv_name)


if __name__ == "__main__":
    main()

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

    csv_name = root_dir + "dens-iter-" + str(iter_num) + f"-pareto-params.csv"
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
            "mse_surf_tens",
            "mse_liq_density",
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
    csv_name = root_dir + "dens-iter-" + str(iter_num) + f"-final-params.csv"
    df_final.to_csv(csv_name)

    return df_final