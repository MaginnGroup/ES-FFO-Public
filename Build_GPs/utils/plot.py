import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import matplotlib.ticker as ticker
import sys

sys.path.append("../..")
from utils.prep_ms_data import prepare_df_props
from fffit.fffit.utils import values_scaled_to_real
from fffit.fffit.plot import (
    plot_model_performance,
    plot_slices_temperature,
    plot_slices_params,
    plot_model_vs_test)
sys.path.remove("../..")

from .models import get_prop_best_model, get_exp_data

def plot_gp_examples(all_df_data, data_dict, iter_type = "ld_iters", gp_shuffle_seed = 42, save_fig=False):
    """
    Plot GP examples for each molecule and property
    Parameters
    ----------
    all_df_data : dict
        Dictionary of all dataframes for each molecule
    data_dict : dict
        Dictionary of all data for each molecule
    iter_type : str
        Type of iteration (ld_iters or vle_iters)
    gp_shuffle_seed : int
        Seed for GP shuffle
    save_fig : bool
        Whether to save the figure or not
    """
    #Get all data
    for mol_name, df_csv in all_df_data.items():
        data = data_dict[mol_name]
        ld_threshold = (min(list(data.expt_liq_density.values())) + max(list(data.expt_vap_density.values())))/2
        iter_num = df_csv["iter"].max()

        dir_name = f"analysis/{mol_name}/{iter_type}/iter-{str(iter_num)}"
        os.makedirs(dir_name, exist_ok=True)
        pdf_name = os.path.join(dir_name , "fig_gp_examples.pdf")
        pdf = PdfPages(pdf_name)

        df_all, df_liq, df_vapor = prepare_df_props(df_csv, data, ld_threshold)
        models_best, models_rq, all_models, dir_train_test, best_labels = get_prop_best_model(df_liq, data, dir_name, gp_shuffle_seed)
        
        for prop_name, models in all_models.items():
            # Load data
            exp_data, property_bounds, name = get_exp_data(data, prop_name)
            df_x_train = pd.read_csv(os.path.join(dir_train_test, f"{prop_name}_x_train.csv"), header = 0, index_col = 0)
            df_y_train = pd.read_csv(os.path.join(dir_train_test, f"{prop_name}_y_train.csv"), header = 0, index_col = 0)
            df_x_test = pd.read_csv(os.path.join(dir_train_test, f"{prop_name}_x_test.csv"), header = 0, index_col = 0)
            df_y_test = pd.read_csv(os.path.join(dir_train_test, f"{prop_name}_y_test.csv"), header = 0, index_col = 0)
            df_x_all = pd.concat([df_x_train, df_x_test], ignore_index=True)
            df_y_all = pd.concat([df_y_train, df_y_test], ignore_index=True)

            #Plot model performance
            best_model = {best_labels[prop_name]: models_best[prop_name]}
            title = f"{mol_name} {name} Iter {iter_num} - All Data"
            plot_model_performance(models, df_x_all, df_y_all, property_bounds, pdf, title, xylim=None, save_fig=save_fig)
            title = f"{mol_name} {name} Iter {iter_num} - Best All Data"
            plot_model_performance(best_model, df_x_all, df_y_all, property_bounds, pdf, title, xylim=None, save_fig=save_fig)
            title = f"{mol_name} {name} Iter {iter_num} - Training Data"
            plot_model_performance(models, df_x_train, df_y_train, property_bounds, pdf, title, xylim=None, save_fig=save_fig)
            title = f"{mol_name} {name} Iter {iter_num} - Best Training Data"
            plot_model_performance(best_model, df_x_train, df_y_train, property_bounds, pdf, title, xylim=None, save_fig=save_fig)
            title = f"{mol_name} {name} Iter {iter_num} - Testing Data"
            plot_model_performance(models, df_x_test, df_y_test, property_bounds, pdf, title, xylim=None, save_fig=save_fig)
            title = f"{mol_name} {name} Iter {iter_num} - Best Testing Data"
            plot_model_performance(best_model, df_x_test, df_y_test, property_bounds, pdf, title, xylim=None, save_fig=save_fig)
            
        # for prop_name, models in all_models.items():
        #     #Plot test sets
        #     df_x_test = pd.read_csv(os.path.join(dir_train_test, f"{prop_name}_x_test.csv"), header = 1, index_col = False)
        #     if len(df_x_test) > 0:
        #         x_test = df_x_test.to_numpy()
        #         plot_test_sets(models, x_test, df_liq, data, pdf, prop_name)
        #     #Plot GP slices
        #     plot_gp_slices(models, data, prop_name, pdf) 

        pdf.close()

def plot_model_performance(models, x_data, y_data, property_bounds, pdf, title = None, xylim=None, save_fig=False):
    """Plot the predictions vs. result for one or more GP models

    Parameters
    ----------
    models : dict { label : model }
        Each model to be plotted (value, GPFlow model) is provided
        with a label (key, string)
    x_data : np.array
        data to create model predictions for
    y_data : np.ndarray
        correct answer
    property_bounds : array-like
        bounds for scaling density between physical
        and dimensionless values
    xylim : array-like, shape=(2,), optional
        lower and upper x and y limits of the plot

    Returns
    -------
    matplotlib.Figure.figure
    """

    if isinstance(x_data, pd.DataFrame):
        x_data = x_data.to_numpy()
    if isinstance(y_data, pd.DataFrame):
        y_data = y_data.to_numpy()

    y_data_physical = values_scaled_to_real(y_data, property_bounds)
    min_xylim = np.min(y_data_physical)
    max_xylim = np.max(y_data_physical)

    fig, ax = plt.subplots()

    mse_min = np.inf
    mse_model = None
    for (label, model) in models.items():
        gp_mu, gp_var = model.predict_f(x_data)
        gp_mu_physical = values_scaled_to_real(gp_mu, property_bounds)
        ax.scatter(y_data_physical, gp_mu_physical, label=label, zorder=2.5, alpha=0.4)
        meansqerr = np.mean(
            (gp_mu_physical - y_data_physical.reshape(-1, 1)) ** 2
        )
        if meansqerr < mse_min:
            mse_min = meansqerr
            mse_model = model
        print("Model: {}. Mean squared err: {:.2e}".format(label, meansqerr))
        if np.min(gp_mu_physical) < min_xylim:
            min_xylim = np.min(gp_mu_physical)
        if np.max(gp_mu_physical) > max_xylim:
            max_xylim = np.max(gp_mu_physical)

    if xylim is None:
        xylim = [min_xylim, max_xylim]

    ax.plot(
        np.arange(xylim[0], xylim[1] + 100, 100),
        np.arange(xylim[0], xylim[1] + 100, 100),
        color="xkcd:blue grey",
        label="y=x",
    )

    if title is not None:
        ax.set_title(title, fontsize=12)
    ax.set_xlim(xylim[0], xylim[1])
    ax.set_ylim(xylim[0], xylim[1])
    ax.set_xlabel("Actual")
    ax.set_ylabel("Model Prediction")
    ax.legend()
    ax.set_aspect("equal", "box")

    if save_fig:
        pdf.savefig(fig)
    else:
        plt.show()
    plt.close(fig)

    return mse_model, label
    
def plot_gp_slices(models, data, property_name, pdf):
    """
    Plot the GP slices for a given property
    Parameters
    ----------
    models : dict
        Dictionary of models for each property
    data : object
        Data object containing the data for the property
    property_name : str
        Name of the property to plot
    pdf : PdfPages
        PdfPages object to save the plots to
    """
    exp_data, prop_bounds, prop_name = get_exp_data(data, property_name)
    # Plot temperature slices
    figs = plot_slices_temperature(
        models,
        data.n_params,
        data.temperature_bounds(),
        prop_bounds,
        property_name=prop_name,
    )

    try:
        for fig in figs:
            pdf.savefig(fig)
    except:
        pdf.savefig(figs)
    del figs

    # Plot parameter slices
    for param_name in data.param_names:
        figs = plot_slices_params(
            models,
            param_name,
            data.param_names,
            data.temperature_bounds()[0], # min temperature
            data.temperature_bounds(),
            prop_bounds,
            property_name=prop_name,
        )
        try:
            for fig in figs:
                pdf.savefig(fig)
        except:
            pdf.savefig(figs)
        del figs

def plot_test_sets(models, x_test, df_data, data, pdf, property_name):
    exp_data, prop_bounds, prop_name = get_exp_data(data, property_name)
    # Loop over test params
    for test_params in x_test[:,:data.n_params]:
        train_points = []
        test_points = []
        # Locate rows where parameter set == test parameter set
        matches = np.unique(np.where((df_data[list(data.param_names)] == test_params).all(axis=1))[0])
        # Loop over all matches -- these will be different temperatures
        for match in matches:
            # If the match (including T) is in the test set, then append to test points
            if np.where((df_data.values[match,:data.n_params+1] == x_test[:,:data.n_params+1]).all(axis=1))[0].shape[0] == 1:
                test_points.append([df_data["temperature"].iloc[match],df_data[property_name].iloc[match]])
            # Else append to train points
            else:
                train_points.append([df_data["temperature"].iloc[match],df_data[property_name].iloc[match]])

        pdf.savefig(
            plot_model_vs_test(
                models,
                test_params,
                np.asarray(train_points),
                np.asarray(test_points),
                data.temperature_bounds(),
                prop_bounds,
                property_name=prop_name
            )
        )

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

def plot_sim_exp(mol_data, df_data, property_name):
    
    """
    Plot the surface tension for a given molecule and force field

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
    # Plot VLE envelopes
    fig, ax2 = plt.subplots(1, 1, figsize=(6,6))    

    #Initialize min and max values
    exp_data, prop_bounds, prop_name = get_exp_data(mol_data, property_name)
    min_temp = min(exp_data.keys())
    max_temp = max(exp_data.keys())
    min_st = min(exp_data.values())
    max_st = max(exp_data.values())
    
    prop_vals = df_data[property_name].values
    prop_unc = df_data[property_name + "_unc"].values

    #Set new max and mins
    min_st, max_st = get_min_max(min_st, max_st, prop_vals, prop_unc)
    # #Plot opt_scheme_ms vle curve
    if len(df_data) == 1:
        ax2.errorbar(df_data["temperature"].values, prop_vals, yerr=1.96*prop_unc,
                    color="blue",markersize=10, linestyle='None', marker = "o", alpha=0.5, 
                    zorder = 1, label = "Simulation")
    else:
        param_names = list(mol_data.param_names)
        for i, ((param_vals), group_df) in enumerate(df_data.groupby(param_names)):
            label = "Simulation" if i == 0 else None
            prop_vals = group_df[property_name].values
            prop_unc = group_df[property_name + "_unc"].values
            ax2.errorbar(group_df["temperature"].values, prop_vals, yerr=1.96*prop_unc,
                         markersize=10, linestyle='None', marker = "o", alpha=0.5, 
                         zorder = 1, label = label)

    #Plot experimental data
    #For MeOH and EG ignore T data > 430
    if mol_data.name in ["MeOH", "EG"]:
        exp_data = {k: v for k, v in exp_data.items() if k <= 430}

    ax2.scatter(exp_data.keys(), exp_data.values(),
        color="black",marker="x",linewidths=2,s=100,label="Experiment", zorder = 2)

    #Set Axes
    # ax2.set_ylim(min_st*0.95,max_st*1.05)
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    # ax2.set_xlim(min_temp*0.95, max_temp*1.05)
    
    ax2.tick_params("both", direction="in", which="both", length=4, labelsize=20, pad=10)
    ax2.tick_params("both", which="major", length=8)
    ax2.xaxis.set_ticks_position("both")
    ax2.yaxis.set_ticks_position("both")

    ax2.set_xlabel("T (K)", fontsize=20, labelpad=10)
    # ax2.set_ylabel(r"$\mathregular{\gamma}$ (mN/m)", fontsize=32, labelpad=15)
    ax2.set_ylabel(prop_name, fontsize=20, labelpad=15)
    # for axis in ['top','bottom','left','right']:
    #     ax2.spines[axis].set_linewidth(2.0)

    # for h in handles: h.set_linestyle("")
    fig.legend(loc="upper center", ncol=2, fontsize=22, handletextpad=0.1, markerscale=0.9, edgecolor="dimgrey")
    ax2.text(0.60,  0.82, mol_data.name, fontsize=30, transform=ax2.transAxes)
    fig.subplots_adjust(bottom=0.2, top=0.85, left=0.15, right=0.95, wspace=0.55)

    return fig