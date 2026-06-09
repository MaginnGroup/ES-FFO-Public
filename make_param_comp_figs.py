#Code for atom type line plot
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
from Opt_ES.utilsOpt import opt_atom_types


from fffit.fffit.utils import values_real_to_scaled, values_scaled_to_real, values_scaled_to_real_tf, values_real_to_scaled_tf
import matplotlib
import matplotlib.patches as mpatch
from matplotlib import ticker

#Set matplotlib parameters
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

os.chdir("/groups/ed/group_members/Montana_Carlozo/ES-FFO/Opt_ES")


molec_names = ["EG", "MeOH", "Gly", "DMSO", "DMF", "DEC"] #Change me as needed

dir_name = f"analysis/AT-0/ms_val_opt/"
os.makedirs(dir_name, exist_ok=True)
pdf_name = os.path.join(dir_name , f"parallel_plot.pdf")
pdf = PdfPages(pdf_name)

for str in molec_names:
    at_num = 0
    err_met = "mpd"
    mol_names = str.split("-") # ["EG", "Gly", "ACN", "MeOH", "DMSO", "THF", "DCM", "DEC", "DMF"] Change me as needed

    molec_dict = esolvs.make_dict(mol_names)
    matplotlib.rc("font", family="sans-serif")
    matplotlib.rc("font", serif="Arial")

    NM_TO_ANGSTROM = 10
    K_B = 0.008314 # J/MOL K
    KJMOL_TO_K = 1.0 / K_B

    # ID the top ten by lowest average MAPE
    #Get params < 10
    # df = pd.read_csv("analysis/at_01/EG-Gly-MeOH/ExpVal/opt_res/best_per_run.csv", header = 0, index_col=0)
    def get_ms_best(visual, mol, param_bnds, param_names, all_molec_dir, err_met):
        path_best_sets = os.path.join(all_molec_dir, "best_per_run.csv")
        df = pd.read_csv(path_best_sets, header=0)
        first_param_name = param_names[0] + "_cum"
        last_param_name = param_names[-1] + "_cum"
        data = df.loc[0, first_param_name:last_param_name].values
        data = visual.values_pref_to_real(data.reshape(1,-1))
        data = values_real_to_scaled(data.reshape(1,-1), param_bnds)

        #Load the data for the param set found by NW Method (from ms_val/pareto_params.csv)
        pareto_path = f"../Build_GPs/analysis/{mol}/vle_iters/iter-1/result_errors.csv"
        pareto_df = pd.read_csv(pareto_path, header=0)
        min_obj_idx = abs(pareto_df[f"{err_met}_surf_tens"]).idxmin()
        min_obj_idx2 = (
        pareto_df[f"{err_met}_surf_tens"]
        .abs()
        .sort_values()
        .index[1]
    )
        pareto_params = pareto_df.loc[:, param_names[0]:param_names[-1]].values
        data_pareto = values_real_to_scaled(pareto_params, param_bnds)

        #Get the results from validation files
        path_to_opt_ms_val = os.path.join(all_molec_dir, "ms_val_opt/error_data.csv")
        df_val_res = pd.read_csv(path_to_opt_ms_val, header=0)
        results = df_val_res[[f"{err_met}_liq_density", f"{err_met}_surf_tens"]].values

        err_cols = [c for c in pareto_df.columns if err_met in c]
        result_bounds = np.zeros((len(err_cols),2))
        for col in err_cols:
            col_max = max(np.abs(pareto_df[col].values).max(), np.abs(df_val_res[col].values).max())
            if err_met == "mpd":
                col_min = -col_max
            else:
                col_min = 0
            result_bounds[err_cols.index(col)] = [col_min, col_max]        

        pareto_vals = pareto_df[[f"{err_met}_liq_density", f"{err_met}_surf_tens"]].values
        results_pareto = values_real_to_scaled(pareto_vals, result_bounds)
        results = values_real_to_scaled(results.reshape(1,-1), result_bounds).reshape(1,-1)
        # results = np.full(shape=len(results_pareto.T), fill_value=np.nan).reshape(1,-1)
        return data, data_pareto, results, results_pareto, result_bounds, min_obj_idx, min_obj_idx2

    visual = opt_atom_types.Vis_Results(mol_names, at_num, 1, "ExpVal")
    param_bnds, param_names = visual.get_param_bnds_names()
    # Set parameter set of interest (in this case get the best parameter set)
    x_label = "best_set"
    all_molec_dir = visual.use_dir_name
    data_best, data_pareto, results_best, results_pareto, result_bounds, min_obj_idx, min_obj_idx2= get_ms_best(visual, str, param_bnds, param_names, all_molec_dir, err_met="mpd")
    #Print results shapes
    data= np.vstack((data_best, data_pareto))
    results = np.vstack((results_best, results_pareto))

    param_bounds = param_bnds
    indx_mid = int(len(param_names) / 2)
    param_bounds[:indx_mid] = param_bounds[:indx_mid] * NM_TO_ANGSTROM
    param_bounds[indx_mid:] = param_bounds[indx_mid:] * KJMOL_TO_K

    data = np.hstack((data, results))
    bounds = np.vstack((param_bounds, result_bounds))

    col_names = []
    for name in param_names:
        latex_name = lambda s: fr"$\{s.split('_',1)[0]}_{{{s.split('_',1)[1]}}}$" if '_' in s else fr"${s}$"
        col_names.append(latex_name(name))

    err_met_upper = err_met.upper()
    col_names += [err_met_upper + "\n" + r"$\rho_l$", err_met_upper + "\n" + r"$\gamma$"]
    n_axis = len(col_names)
    assert data.shape[1] == n_axis
    x_vals = [i for i in range(n_axis)]

    # Create (N-1) subplots along x axis
    fig, axes = plt.subplots(1, n_axis-1, sharey=False, figsize=(20,5))

    # Plot each row
    for i, ax in enumerate(axes):
        for j, line in enumerate(data):
            # print(x_vals, line)
            if j ==0:
                if i ==len(axes)-1:
                    ax.plot(x_vals, line, alpha=1, label="GP-Opt", color="purple", linewidth=3.0, zorder=500)
                else:
                    ax.plot(x_vals, line, alpha=1, color="purple", linewidth=3.0, zorder=500)
                    
            elif j ==min_obj_idx+1:
                if i ==len(axes)-1:
                    ax.plot(x_vals, line, alpha=1, label="Base", color="red", linestyle = "--", linewidth=2.5, zorder=501)
                else:
                    ax.plot(x_vals, line, alpha=1, color="red", linestyle = "--", linewidth=2.5, zorder=501)
                    
            elif j ==min_obj_idx2+1:
                if i ==len(axes)-1:
                    ax.plot(x_vals, line, alpha=1, label="ST FF with Lowest MAPD " + r"$\rho_l$", color="dodgerblue", linestyle = "--", linewidth=2.5, zorder=501)
                else:
                    ax.plot(x_vals, line, alpha=1, color="dodgerblue", linestyle = "--", linewidth=2.5, zorder=501)

            elif j == len(data)-1 and i == len(axes)-1:
                ax.plot(x_vals, line, alpha=0.25, label="Other ST FFs", color="gray")
            #For the last line, add labels
            else:
                ax.plot(x_vals, line, alpha=0.25)
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
    #Put legend at top center of plot
    ax.xaxis.set_major_locator(ticker.FixedLocator([n_axis-2, n_axis-1]))
    ax.set_xticklabels([col_names[-2], col_names[-1]], fontsize=20)

    ax = plt.twinx(axes[-1])
    ax.set_ylim(-0.05, 1.05)
    set_ticks_for_axis(ax, bounds[-1], nticks=6)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_linewidth(2.0)
    
    # Remove space between subplots
    #Add Titile
    plt.subplots_adjust(wspace=0, bottom=0.3)

    # 2. Calculate the "Visual Center" of the combined axes
    # This finds the left edge of the first axis and the right edge of the last axis
    all_axes_bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    # Convert that bbox to figure coordinates
    fig_bbox = fig.transFigure.inverted().transform_bbox(all_axes_bbox)
    # The true horizontal center of your DATA area
    data_center = 100*(fig_bbox.x0 + fig_bbox.x1) / 2

    fig.suptitle(f"{str} - Comparison of GP-Optimized FF and Unoptimized ST Iter FFs", fontsize=24, y=1.10)
    leg = fig.legend(loc='upper center', bbox_to_anchor=(data_center, 1.03), ncol=4, fontsize=16, bbox_transform=fig.transFigure)
    pdf.savefig(fig, bbox_inches='tight')   # save one figure at a time

pdf.close()

