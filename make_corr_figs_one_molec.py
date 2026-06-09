
import signac
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymser
from matplotlib.backends.backend_pdf import PdfPages

from utils.molec_class_files import esolvs
from Build_GPs.utils.signac import get_signac_results, save_signac_results
from Build_GPs.utils.id_new_samples import new_samples_vle, find_pareto, new_samples_ld, check_mse_10
from Build_GPs.utils.models import get_best_models
from Build_GPs.utils.plot import plot_gp_examples
import pickle

from fffit.fffit.utils import values_real_to_scaled, values_scaled_to_real, values_scaled_to_real_tf, values_real_to_scaled_tf
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import seaborn
from matplotlib import ticker
import re
import os

from Opt_ES.utilsOpt import opt_atom_types
import re
import glob

##Add plot formatting functions
def set_ticks_for_axis(ax, param_bounds, nticks):
    """Set the tick positions and labels on y axis for each plot

    Tick positions based on normalised data
    Tick labels are based on original data
    """
    min_val, max_val = param_bounds
    step = (max_val - min_val) / float(nticks-1)
    tick_labels = [round(min_val + step * i, 2) for i in range(nticks)]
    ticks = np.linspace(0, 1.0, nticks)
    ax.yaxis.set_ticks(ticks)
    ax.set_yticklabels(tick_labels, fontsize=16)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.tick_params("y", direction="inout", which="both", length=7)
    ax.tick_params("y", which="major", length=14)
    ax.tick_params("x", pad=15) 

def set_ticks_for_axis(ax, param_bounds, nticks):
    import numpy as np
    from matplotlib import ticker

    min_val, max_val = param_bounds

    # Base ticks
    ticks = np.linspace(0.0, 1.0, nticks)
    values = np.linspace(min_val, max_val, nticks)

    # --- FORCE zero tick if it lies in bounds ---
    if min_val < 0 < max_val:
        zero_tick = (0.0 - min_val) / (max_val - min_val)

        ticks = np.append(ticks, zero_tick)
        values = np.append(values, 0.0)

        # Sort consistently
        order = np.argsort(ticks)
        ticks = ticks[order]
        values = values[order]

    # Apply ticks
    ax.yaxis.set_ticks(ticks)
    y_labels_rounded = []
    for v in values:
        if abs(v) >= 1.0:
            y_labels_rounded.append(round(v, 1))
        else:
            y_labels_rounded.append(round(v, 2))
    ax.set_yticklabels(y_labels_rounded, fontsize=16)

    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.tick_params("y", direction="inout", which="both", length=7)
    ax.tick_params("y", which="major", length=14)
    ax.tick_params("x", pad=15)

#Add correlation creating functions
def calc_param_sums(df_analyze, data_class, mol_names, mode3 = "scl"):
    NM_TO_ANGSTROM = 10
    K_B = 0.008314 # J/MOL K
    KJMOL_TO_K = 1.0 / K_B

    param_names = list(data_class.param_names)
    project = signac.get_project("Build_GPs/vle_iters")
    #Grab system.top from any the first vle iter results
    jobs = project.find_jobs({"mol_name": mol_names[0], "iter": 1})
    job = None
    for i, job_first in enumerate(jobs):
        job = job_first
        break
    #Open system.top and read the atom types
    # Extract unique atom types from param names
    atom_types = set(p.split("_")[1] for p in param_names)

    # Initialize counts dictionary
    param_counts = {atype: 0 for atype in atom_types}

    # Load .top file
    top_file = job.fn("system.top")
    with open(top_file, "r") as f:
        lines = f.readlines()

    # Step 1: extract atoms section to count atom types
    # GROMACS .top atoms section starts with [ atoms ] and ends at next [ ... ]
    atoms_section = []
    inside_atoms = False
    for line in lines:
        line = line.strip()
        if line.startswith("[ atoms ]"):
            inside_atoms = True
            continue
        if inside_atoms:
            if line.startswith("[") or line == "":
                break
            atoms_section.append(line)

    # Parse atoms lines (assume standard GROMACS format: nr type resnr resid atom cgnr charge mass)
    for line in atoms_section:
        if line.startswith(";") or line == "":
            continue
        parts = re.split(r"\s+", line)
        atom_type = parts[1]  # second column is atom type
        if atom_type in param_counts:
            param_counts[atom_type] += 1
    #For each row, get the sum of the rows of sigma 
    sigma_cols = [c for c in df_analyze.columns if c.startswith("sigma")]
    epsilon_cols = [c for c in df_analyze.columns if c.startswith("epsilon")]
    #Scale sigma and epsilon between 0 and 1
    #Get data for sigma and epsilon in array form
    if mode3 == "scl":
        data = df_analyze[sigma_cols + epsilon_cols].values
        #Scale data
        data = values_real_to_scaled(data, data_class.param_bounds)
        data[data > 1e5] = 0
        df_analyze[sigma_cols + epsilon_cols] = data
        # print(df_analyze)
    elif mode3 == "from_scl":
        #Make values zero whenever bounds are less than 1e-8 apart
        data = df_analyze[sigma_cols + epsilon_cols].values
        param_bnds = np.array(data_class.param_bounds)  # shape (n_params, 2)
        lower_bnd = param_bnds[:, 0]
        upper_bnd = param_bnds[:, 1]
        fixed_cols = np.isclose(upper_bnd, lower_bnd, rtol=1e-8)
        # print("Fixed cols", fixed_cols, param_names)
        data[:, fixed_cols] = 0
        df_analyze[sigma_cols + epsilon_cols] = data
        # print("DATA", data)
    if mode3 == "to_real":
        data = df_analyze[sigma_cols + epsilon_cols].values
        #Scale data
        data = values_scaled_to_real(data, data_class.param_bounds)
        data[data < 1e-5] = 0
        df_analyze[sigma_cols + epsilon_cols] = data
        # print(df_analyze)

    #Calculate weighted sums
    df_analyze["sigma_sum"] = sum(df_analyze[col] * param_counts[col.split("_")[1]] for col in sigma_cols) #* NM_TO_ANGSTROM
    df_analyze["epsilon_sum"] = sum(df_analyze[col] * param_counts[col.split("_")[1]] for col in epsilon_cols) #* KJMOL_TO_K
    if mode3 == "real" or mode3 == "to_real":
        df_analyze["sigma_sum"] = df_analyze["sigma_sum"]* NM_TO_ANGSTROM
        df_analyze["epsilon_sum"] = df_analyze["epsilon_sum"]* KJMOL_TO_K
    return df_analyze, param_counts, param_names

def weighted_min_max(param_bounds_slice, param_names_slice, param_counts):
    x_min = np.sum([
        param_bounds_slice[i, 0] * param_counts[param_names_slice[i].split("_")[1]]
        for i in range(len(param_names_slice))
    ])
    x_max = np.sum([
        param_bounds_slice[i, 1] * param_counts[param_names_slice[i].split("_")[1]]
        for i in range(len(param_names_slice))
    ])
    return x_min, x_max

def get_corr_all_molecs(mol_names, mode, mode2 = "all", mode3 ="scl", err_met = "mpd", threshold=10):
    NM_TO_ANGSTROM = 10
    K_B = 0.008314 # J/MOL K
    KJMOL_TO_K = 1.0 / K_B
    import matplotlib.pyplot as plt 

    all_df_analysis = pd.DataFrame()
    x_min_sig = np.inf
    x_max_sig = -np.inf
    x_min_eps = np.inf
    x_max_eps = -np.inf
    for molec in mol_names:
        # ID the top ten by lowest average MAPE
        molec_dict = esolvs.make_dict(mol_names)
        data_class = molec_dict[molec]
        #Get params < 10
        if mode == "ld":
            if mode2 == "pareto":
                df = pd.read_csv("Build_GPs/analysis/" + molec + "/ld_iters/mse-less10-full.csv", header = 0, index_col=0)
            else:
                df_all_res = pd.read_csv("Build_GPs/analysis/"+molec+"/ld_iters/all_results.csv", header = 0, index_col=0)
                #For each group of param names and temperature, calculate the average mpd_liq_density
                #Remove all rows where 5 temperatures do not have ld results
                df_all_res = df_all_res.dropna().copy()
                if molec == "MeOH":
                    df_all_res = df_all_res[df_all_res["iter"] <= 1]
                df_all_res = df_all_res.groupby(list(data_class.param_names)).filter(lambda x: len(x) >= 5) 
                #Calculate the average mpd_liq_density for each group
                df_all_res["expt_liq_density"] = df_all_res["temperature"].apply(
                lambda x: data_class.expt_liq_density[x])
                df_all_res["pct_err"] = ((df_all_res["liq_density"] - df_all_res["expt_liq_density"]) / df_all_res["expt_liq_density"]) * 100
                df = (df_all_res.groupby(list(data_class.param_names)).agg(mpd=("pct_err", "mean")).reset_index())
            
        elif mode == "vle":
            props_pareto = ["liq_density", "surf_tens"] 
            if mode2 == "pareto":
                df_pareto = pd.read_csv("Build_GPs/analysis/" + molec + "/vle_iters/iter-1/pareto-params.csv", header = 0, index_col=0)
                #Get only the lowest mapd value row where pareto is true
                df_final = df_pareto.drop(columns="is_pareto")
            else:
                df_final = pd.read_csv("Build_GPs/analysis/" + molec + "/vle_iters/iter-1/result_errors.csv", header = 0, index_col=0)
            props_mse = ["mapd_" + prop for prop in props_pareto]
            df= df_final.copy()
            # df = df_final[df_final[props_mse].le(threshold).all(axis=1)

        df_analyze = df.copy()
        #For each row, get the sum of the rows of sigma 
        df_analyze, param_counts, param_names = calc_param_sums(df_analyze, data_class, mol_names, mode3 = mode3)

        if molec == "DEC" and mode == "vle" and mode2 == "all":
            df_analyze = df_analyze[df_analyze["sigma_sum"] > 5]

        #Drop columns not containing sum or mpd
        df_analyze = df_analyze.loc[:, df_analyze.columns.str.contains("sum|mpd")]
        df_analyze["Molecule"] = molec
        if mode == "ld":
            df_analyze.rename(columns={"mpd": "mpd_liq_density"}, inplace=True)
        
        all_df_analysis = pd.concat([all_df_analysis, df_analyze], ignore_index=True)
        
        data = df[list(data_class.param_names)].values
        data = values_real_to_scaled(data, data_class.param_bounds)
        param_bounds = data_class.param_bounds
        indx_mid = int(len(data_class.param_names) / 2)
        if mode3 == "scl":
            param_bounds = values_real_to_scaled(param_bounds.T, data_class.param_bounds).T #For consistency
        if mode3 != "scl":
            param_bounds[:indx_mid] = param_bounds[:indx_mid] * NM_TO_ANGSTROM
            param_bounds[indx_mid:] = param_bounds[indx_mid:] * KJMOL_TO_K
        # Split param_names to match sigma / epsilon
        sigma_names = param_names[:indx_mid]
        epsilon_names = param_names[indx_mid:]

        # Compute weighted sums
        x_min_sig_new, x_max_sig_new = weighted_min_max(param_bounds[:indx_mid], sigma_names, param_counts)
        x_min_eps_new, x_max_eps_new = weighted_min_max(param_bounds[indx_mid:], epsilon_names, param_counts)

        if x_min_sig_new < x_min_sig:
            x_min_sig = x_min_sig_new
        if x_max_sig_new > x_max_sig:
            x_max_sig = x_max_sig_new
        if x_min_eps_new < x_min_eps:
            x_min_eps = x_min_eps_new
        if x_max_eps_new > x_max_eps:
            x_max_eps = x_max_eps_new

    if mode == "ld":
        use_df = all_df_analysis.drop(columns="Molecule") #df_new
    else:
        use_df = all_df_analysis.drop(columns=["Molecule", "mpd_liq_density"])

    meth_corr = "spearman"

    corr_matrix = use_df.corr(method=meth_corr)  # or "spearman"    
    return all_df_analysis, corr_matrix, x_min_sig, x_max_sig, x_min_eps, x_max_eps

def get_corr_one_molec(mol_name, mode, mode2 = "all", mode3 ="scl", err_met = "mpd", threshold=10):
    mol_names = [mol_name]
    molec_dict = esolvs.make_dict(mol_names)
    data_class = molec_dict[mol_name]
    NM_TO_ANGSTROM = 10
    K_B = 0.008314 # J/MOL K
    KJMOL_TO_K = 1.0 / K_B

    x_min_sig = np.inf
    x_max_sig = -np.inf
    x_min_eps = np.inf
    x_max_eps = -np.inf

    # ID the top ten by lowest average MAPE
    #Get params < 10
    if mode == "ld":
        if mode2 == "pareto":
            df = pd.read_csv("Build_GPs/analysis/" + mol_name + "/ld_iters/mse-less10-full.csv", header = 0, index_col=0)
        else:
            df_all_res = pd.read_csv("Build_GPs/analysis/"+mol_name+"/ld_iters/all_results.csv", header = 0, index_col=0)
            #For each group of param names and temperature, calculate the average mpd_liq_density
            #Remove all rows where 5 temperatures do not have ld results
            df_all_res = df_all_res.dropna().copy()
            #Remove all rows where iter > 2
            if mol_name == "MeOH":
                df_all_res = df_all_res[df_all_res["iter"] <= 1]
            df_all_res = df_all_res.groupby(list(data_class.param_names)).filter(lambda x: len(x) >= 5) 
            #Calculate the average mpd_liq_density for each group
            df_all_res["expt_liq_density"] = df_all_res["temperature"].apply(
            lambda x: data_class.expt_liq_density[x])
            df_all_res["pct_err"] = ((df_all_res["liq_density"] - df_all_res["expt_liq_density"]) / df_all_res["expt_liq_density"]) * 100
            df = (df_all_res.groupby(list(data_class.param_names)).agg(mpd=("pct_err", "mean")).reset_index())
    elif mode == "vle":
        props_pareto = ["liq_density", "surf_tens"] 
        if mode2 == "pareto":
            df_pareto = pd.read_csv("Build_GPs/analysis/" + mol_name + "/vle_iters/iter-1/pareto-params.csv", header = 0, index_col=0)
            #Get only the lowest mapd value row where pareto is true
            df_final = df_pareto.drop(columns="is_pareto")
            #Get row where mapd_surf_tens is minimum
            # min_index = df_final["mapd_surf_tens"].idxmin()
            # df_final = df_final.loc[[min_index]]
        else:
            df_final = pd.read_csv("Build_GPs/analysis/" + mol_name + "/vle_iters/iter-1/result_errors.csv", header = 0, index_col=0)
        props_mse = ["mapd_" + prop for prop in props_pareto]
        df= df_final.copy()
        # df = df_final[df_final[props_mse].le(threshold).all(axis=1)

    df_analyze = df.copy()
    #For each row, get the sum of the rows of sigma 
    df_analyze, param_counts, param_names = calc_param_sums(df_analyze, data_class, mol_names, mode3 = mode3)
    if mode == "ld":
        df_analyze.rename(columns={"mpd": "mpd_liq_density"}, inplace=True)

    if mol_name == "DEC" and mode == "vle" and mode2 == "all":
        df_analyze = df_analyze[df_analyze["sigma_sum"] > 5]

    #Drop columns not containing sum or mpd
    df_new = df_analyze.loc[:, df_analyze.columns.str.contains("sigma|epsilon|mpd")].copy()
    df_analyze = df_analyze.loc[:, df_analyze.columns.str.contains("sum|mpd")]

    seaborn.set_palette('bright', n_colors=len(df))
    data = df[list(data_class.param_names)].values
    data = values_real_to_scaled(data, data_class.param_bounds)
    data_scl = data.copy()

    if mode == "ld":
        max_error = df[f"{err_met}"].max()
        min_error = df[f"{err_met}"].min()
        result_bounds = np.array([[min_error, np.maximum(20, max_error)]])
        results = values_real_to_scaled(df[[f"{err_met}"]].values, result_bounds)
    elif mode == "vle":
        if err_met == "mapd":
            result_bounds = np.array([[0, 20], [0,20]])
        elif err_met == "mpd":
            err_cols = [c for c in df.columns if err_met in c]
            ld_error = err_cols[0]
            st_error = err_cols[1]
            max_error_ld = df[ld_error].max()
            min_error_ld = df[ld_error].min()
            max_error_st = df[st_error].max()
            min_error_st = df[st_error].min()
            result_bounds = np.array([[min_error_ld, max_error_ld], [min_error_st, max_error_st]])
        results = values_real_to_scaled(df[[f"{err_met}_liq_density", f"{err_met}_surf_tens"]].values, result_bounds)
    param_bounds = data_class.param_bounds
    indx_mid = int(len(data_class.param_names) / 2)
    if mode3 == "scl":
        param_bounds = values_real_to_scaled(param_bounds.T, data_class.param_bounds).T #For consistency
    if mode3 != "scl":
        param_bounds[:indx_mid] = param_bounds[:indx_mid] * NM_TO_ANGSTROM
        param_bounds[indx_mid:] = param_bounds[indx_mid:] * KJMOL_TO_K

    def weighted_min_max(param_bounds_slice, param_names_slice, param_counts):
        x_min = np.sum([
            param_bounds_slice[i, 0] * param_counts[param_names_slice[i].split("_")[1]]
            for i in range(len(param_names_slice))
        ])
        x_max = np.sum([
            param_bounds_slice[i, 1] * param_counts[param_names_slice[i].split("_")[1]]
            for i in range(len(param_names_slice))
        ])
        return x_min, x_max

    # Split param_names to match sigma / epsilon
    sigma_names = param_names[:indx_mid]
    epsilon_names = param_names[indx_mid:]

    # Compute weighted sums
    x_min_sig, x_max_sig = weighted_min_max(param_bounds[:indx_mid], sigma_names, param_counts)
    x_min_eps, x_max_eps = weighted_min_max(param_bounds[indx_mid:], epsilon_names, param_counts)

    data = np.hstack((data, results))
    bounds = np.vstack((param_bounds, result_bounds))

    col_names = []
    for name in data_class.param_names:
        latex_name = lambda s: fr"$\{s.split('_',1)[0]}_{{{s.split('_',1)[1]}}}$" if '_' in s else fr"${s}$"
        col_names.append(latex_name(name))

    err_met_upper = err_met.upper()
    if mode == "ld":
        col_names += [err_met_upper + "\n" + r"$\rho_l$"] #r"$\rho_l_{\mathrm{sat}}$"
    elif mode == "vle":
        col_names += [err_met_upper + "\n" + r"$\rho_l$", err_met_upper + "\n" + r"$\gamma$"]
    # print("Column names: ", col_names)
    n_axis = len(col_names)
    assert data.shape[1] == n_axis
    x_vals = [i for i in range(n_axis)]

    return df_analyze, data, bounds, data_class, col_names, n_axis, x_vals, x_min_sig, x_max_sig, x_min_eps, x_max_eps

def plot_corr_one_molec(data, bounds, data_class, col_names, n_axis, x_vals, mode, mol_name):
    import matplotlib.pyplot as plt 
    from matplotlib import ticker

    mol_names = [mol_name]
    molec_dict = esolvs.make_dict(mol_names)
    data_class = molec_dict[mol_name]
    # Create (N-1) subplots along x axis
    fig, axes = plt.subplots(1, n_axis-1, sharey=False, figsize=(20,5))

    # print(data)
    # Plot each row
    for i, ax in enumerate(axes):
        for j, line in enumerate(data):
            # if j == 0 and i ==0:
            #     ax.plot(x_vals, line, alpha=0.45, color = "gray", label = "IFT FFs")
            # else:
            ax.plot(x_vals, line, alpha=0.45)

        ax.set_xlim([x_vals[i], x_vals[i+1]])


    for dim, ax in enumerate(axes):
        ax.xaxis.set_major_locator(ticker.FixedLocator([dim]))
        set_ticks_for_axis(ax, bounds[dim], nticks=6)
        if dim < 10:
            ax.set_xticklabels([col_names[dim]], fontsize=24)
        else:
            ax.set_xticklabels([col_names[dim]], fontsize=20)
        ax.set_ylim(-0.05,1.05)
        # Add white background behind labels
        for label in ax.get_yticklabels():
            label.set_bbox(
                dict(
                    facecolor='white',
                    edgecolor='none',
                    alpha=0.45,
                    boxstyle=mpatch.BoxStyle("round4")
                )
            )
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_linewidth(2.0)

    ax = axes[-1]
    ax.xaxis.set_major_locator(ticker.FixedLocator([n_axis-2, n_axis-1]))
    ax.set_xticklabels([col_names[-2], col_names[-1]], fontsize=20)

    ax = plt.twinx(axes[-1])
    #Include zero as a tick on the last 2 y axes
    ticks = ax.get_yticks()          # or get_xticks()
    ticks = np.unique(np.append(ticks, 0))
    ax.set_ylim(-0.05, 1.05)
    set_ticks_for_axis(ax, bounds[-1], nticks=6)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_linewidth(2.0)

    # Remove space between subplots
    plt.subplots_adjust(wspace=0, bottom=0.3)
    #Set title
    if mode == "ld":
        title_end = "LD Iters"
    elif mode == "vle":
        title_end = "ST Iter"
    plt.suptitle(f"LJ Parameter Sets for {mol_name} {title_end}", fontsize=20)
    return fig

def plot_corr_matrix_one_molec(df_analyze, data_class, x_min_sig, x_max_sig, x_min_eps, x_max_eps, mode, mol_name):
    import matplotlib.pyplot as plt
    fig2, axes = plt.subplots(1, 3, figsize=(15,5))
    ax1, ax2, ax3 = axes.flat[:3]
    if mode == "ld":
        for i in range(len(df_analyze)):
            ax1.plot(df_analyze["sigma_sum"].iloc[i], df_analyze["mpd_liq_density"].iloc[i], 'o', label="Liquid Density")
        ax1.set_xlabel(r"$\Sigma \sigma$/A", fontsize=16)
        ax1.set_ylabel(r"MPD $\rho_l$/%", fontsize=16)
        ax1.tick_params(axis='both', which='major', labelsize=14)
        ax1.set_xlim(x_min_sig, x_max_sig)

        for i in range(len(df_analyze)):
            ax2.plot(df_analyze["epsilon_sum"].iloc[i], df_analyze["mpd_liq_density"].iloc[i], 'o', label="Liquid Density")
        ax2.set_xlabel(r"$\Sigma \frac{\epsilon}{k_B}$/K", fontsize=16)
        ax2.set_ylabel(r"MPD $\rho_l$/%", fontsize=16)
        ax2.tick_params(axis='both', which='major', labelsize=14)
        ax2.set_xlim(x_min_eps, x_max_eps)

    if mode == "vle":
        for i in range(len(df_analyze)):
            ax2.plot(df_analyze["epsilon_sum"].iloc[i], df_analyze["mpd_surf_tens"].iloc[i], 'o', label="Surface Tension")
        ax2.set_xlabel(r"$\Sigma \frac{\epsilon}{k_B}$/K", fontsize=16)
        ax2.set_ylabel(r"MPD $\gamma$/%", fontsize=16)
        ax2.tick_params(axis='both', which='major', labelsize=14)
        ax2.set_xlim(x_min_eps, x_max_eps)
        for i in range(len(df_analyze)):
            ax1.plot(df_analyze["sigma_sum"].iloc[i], df_analyze["mpd_surf_tens"].iloc[i], 'o', label="Surface Tension")
        ax1.set_xlabel(r"$\Sigma \sigma$/A", fontsize=16)
        ax1.set_ylabel(r"MPD $\gamma$/%", fontsize=16)
        ax1.tick_params(axis='both', which='major', labelsize=14)
        ax1.set_xlim(x_min_sig, x_max_sig)


    for i in range(len(df_analyze)):
        ax3.plot(df_analyze["sigma_sum"].iloc[i], df_analyze["epsilon_sum"].iloc[i], 'o', label="Liquid Density")
    ax3.set_xlabel(r"$\Sigma \sigma$/A", fontsize=16)
    ax3.set_ylabel(r"$\Sigma \frac{\epsilon}{k_B}$/K", fontsize=16)
    ax3.set_ylim(x_min_eps, x_max_eps)
    ax3.tick_params(axis='both', which='major', labelsize=14)
    ax3.set_xlim(x_min_sig, x_max_sig)
    fig2.tight_layout()

    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    if mode == "ld":
        use_df = df_analyze #df_new
    else:
        use_df = df_analyze.drop(columns=["mpd_liq_density"])
    meth = "spearman"
    corr_matrix = use_df.corr(method=meth)  # or "spearman"

    import seaborn as sns
    import matplotlib.pyplot as plt
    size = 48/(len(corr_matrix))
    fig4 = plt.figure(figsize=(8,8))
    rename_map = {}
    for i, col in enumerate(corr_matrix.columns):
        # Map the current column name to the corresponding letter
        if "sigma_sum" in col:
            rename_map[col] = r"$\Sigma \sigma$/A"
        elif "epsilon_sum" in col:
            rename_map[col] = r"$\Sigma \frac{\epsilon}{k_B}$/K"
        elif "mpd_liq_density" in col:
            rename_map[col] = "MPD " + r"$\rho_l$" + "/%"
        elif "mpd_surf_tens" in col:
            rename_map[col] = "MPD " + r"$\gamma$" + "/%"
        else:
            rename_map[col] = col
    corr_matrix = corr_matrix.rename(columns=rename_map, index=rename_map)
    ax = sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, fmt=".2f", annot_kws={"size": size})
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=size) 
    ax.set_xlabel(ax.get_xlabel(), fontsize=size)
    ax.set_ylabel(ax.get_ylabel(), fontsize=size)
    ax.tick_params(axis='x', labelsize=size)
    ax.tick_params(axis='y', labelsize=size)
    
    plt.title("Correlation Matrix", fontsize=size)

    #Concatenate 
    wildcard = "1" if mol_name != "MeOH" else "1"
    lhs_param_df = pd.read_csv("Build_GPs/analysis/" + mol_name + "/" +mode +"_iters/params-iter-1.csv", header = 0, index_col=0) #
    csv_files = glob.glob(f"Build_GPs/analysis/{mol_name}/{mode}_iters/params-iter-{wildcard}.csv")
    # Read and concatenate all files
    lhs_param_df = pd.concat(
        [pd.read_csv(f, header=0, index_col=0) for f in csv_files],
        ignore_index=True
    ).drop_duplicates()

    mode3_corr = "from_scl" if mode3 == "scl" else "to_real"
    mol_names = mol_name.split("-")
    lhs_param_df, param_counts, param_names = calc_param_sums(lhs_param_df, data_class, mol_names, mode3 = mode3_corr)

    lhs_array = lhs_param_df[["sigma_sum", "epsilon_sum"]].to_numpy()
    analyze_array = df_analyze[["sigma_sum", "epsilon_sum"]].to_numpy()

    # Set tolerance
    tol = 1e-8  # adjust as needed

    # Create a boolean mask for each lhs row: True if it's far from all df_analyze rows
    mask = np.array([
        not np.any(np.all(np.isclose(row, analyze_array, rtol=tol, atol=tol), axis=1))
        for row in lhs_array
    ])

    # Keep only the "far" rows
    failure_points = lhs_param_df[mask].reset_index(drop=True)
    failure_points = failure_points.loc[:, lhs_param_df.columns.str.contains("sum")]
    lhs_param_df = lhs_param_df.loc[:, lhs_param_df.columns.str.contains("sum")]
    corr_mat_df = lhs_param_df
    corr_matrix2 = corr_mat_df.corr(method=meth)  # or "spearman"
    fig5 = plt.figure(figsize=(8,8))
    size = np.maximum(10,48/(len(corr_matrix2)))

    for i, col in enumerate(corr_matrix2.columns):
        # Map the current column name to the corresponding letter
        if "sigma_sum" in col:
            rename_map[col] = r"$\Sigma \sigma$/A"
        elif "epsilon_sum" in col:
            rename_map[col] = r"$\Sigma \frac{\epsilon}{k_B}$/K"
        elif "mpd_liq_density" in col:
            rename_map[col] = "MPD " + r"$\rho_l$" + "/%"
        elif "mpd_surf_tens" in col:
            rename_map[col] = "MPD " + r"$\gamma$" + "/%"
        else:
            rename_map[col] = col
    corr_matrix2 = corr_matrix2.rename(columns=rename_map, index=rename_map)
    ax = sns.heatmap(corr_matrix2, annot=True, cmap="coolwarm", center=0, fmt=".2f", annot_kws={"size": size})
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=size) 
    ax.set_xlabel(ax.get_xlabel(), fontsize=size)
    ax.set_ylabel(ax.get_ylabel(), fontsize=size)
    ax.tick_params(axis='x', labelsize=size)
    ax.tick_params(axis='y', labelsize=size)
    ax.set_title("Correlation Matrix (LHS Samples)", fontsize=size)
    # plt.show()

    fig3, ax3 = plt.subplots(1, 1, figsize=(5,5))
    ax3.plot(df_analyze["sigma_sum"], df_analyze["epsilon_sum"], 'o', label="Successful Points", alpha = 0.5)
    ax3.plot(failure_points["sigma_sum"], failure_points["epsilon_sum"], 'o', label="Failed Points", alpha = 0.5)
    ax3.set_xlabel(r"$\Sigma \sigma$/A", fontsize=16)
    ax3.set_ylabel(r"$\Sigma \frac{\epsilon}{k_B}$/K", fontsize=16)
    ax3.set_ylim(x_min_eps, x_max_eps)
    ax3.tick_params(axis='both', which='major', labelsize=14)
    ax3.set_xlim(x_min_sig, x_max_sig)
    fig3.legend(loc = 'upper center', fontsize=16, ncol=2, bbox_to_anchor=(0.5, 1.15)  )
    fig3.tight_layout()
    if mode == "ld":
        return fig2, fig3, fig4, fig5
    elif mode == "vle":
        return fig2, fig3, fig4
    

err_met = "mpd" # or mapd
mode3 = "scl"
threshold = 10
mol_names = ["EG", "MeOH", "Gly", "DMSO", "DMF", "DEC"] #Change me as needed

###Make correlation figures for LD and VLE iterations on a per molecule basis

os.chdir("/groups/ed/group_members/Montana_Carlozo/ES-FFO/")
matplotlib.rc("font", family="sans-serif")
matplotlib.rc("font", serif="Arial")
#Make pdf 
modes = ["ld", "vle"]
modes2 = ["all", "pareto"]
for mode in modes:
    for mode2 in modes2:
        for molec_name in mol_names:   
            full_at_dir = os.path.join(f"Build_GPs/analysis/{molec_name}/{mode}_iters")
            os.makedirs(full_at_dir, exist_ok=True)
            pdf_hpvap = PdfPages(os.path.join(full_at_dir ,f"{mode}_corr_{mode2}.pdf"))
            df_analyze, data, bounds, data_class, col_names, n_axis, x_vals, x_min_sig, x_max_sig, x_min_eps, x_max_eps = get_corr_one_molec(molec_name, mode, mode2, "real", err_met, threshold) 
            fig_par = plot_corr_one_molec(data, bounds, data_class, col_names, n_axis, x_vals, mode, molec_name)
            pdf_hpvap.savefig(fig_par, bbox_inches='tight')
            plt.close()
            df_analyze, data, bounds, data_class, col_names, n_axis, x_vals, x_min_sig, x_max_sig, x_min_eps, x_max_eps = get_corr_one_molec(molec_name, mode, mode2, mode3, err_met, threshold) 
            figs = plot_corr_matrix_one_molec(df_analyze, data_class, x_min_sig, x_max_sig, x_min_eps, x_max_eps, mode, mol_name = molec_name) 
            for fig in figs:
                pdf_hpvap.savefig(fig, bbox_inches='tight')
                plt.close()   
            pdf_hpvap.close()