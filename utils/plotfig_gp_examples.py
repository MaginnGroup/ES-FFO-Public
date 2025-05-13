import sys
import gpflow
import numpy as np
import gpflow
import matplotlib.pyplot as plt

# sys.path.append("../")

from fffit.fffit.models import run_gpflow_scipy
from fffit.fffit.utils import shuffle_and_split, values_scaled_to_real

from fffit.fffit.plot import (
    plot_model_performance,
    plot_slices_temperature,
    plot_slices_params,
    plot_model_vs_test,
)

def get_exp_data(molec_object, prop_key):
    """
    Helper function for getting experimental data and bounds

    Parameters
    ----------
    molec_object: Instance of RXXXXConstant() class. Class for refrigerant molecule data
    prop_key: str, The property key to get exp_data for. Valid Keys are "sim_vap_density", "sim_liq_density",
    "sim_Pvap", "sim_Hvap"

    Returns:
    --------
    exp_data: dict, dictionary of Temperature and property data
    property_bounds: array, array of bounds for the property data
    """
    # How to assert that we have a constants class?
    assert isinstance(prop_key, str), "prop_key must be a string"
    
    if "vap_density" in prop_key:
        exp_data = molec_object.expt_vap_density
        property_bounds = molec_object.vap_density_bounds
        property_name = "Vapor Density [kg/m^3]"
    elif "liq_density" in prop_key:
        exp_data = molec_object.expt_liq_density
        property_bounds = molec_object.liq_density_bounds
        property_name = "Liquid Density [kg/m^3]"
    elif "Pvap" in prop_key:
        exp_data = molec_object.expt_Pvap
        property_bounds = molec_object.Pvap_bounds
        property_name = "Vapor Pressure [bar]"
    elif "Hvap" in prop_key:
        exp_data = molec_object.expt_Hvap
        property_bounds = molec_object.Hvap_bounds
        property_name = "Enthalpy of Vaporization [kJ/kg]"
    elif "surf_tens" in prop_key:
        exp_data = molec_object.expt_surf_tens
        property_bounds = molec_object.surf_tens_bounds
        property_name = "Surface Tension [mN/m]"
    else:
        raise (
            ValueError,
            "all_gp_dict must contain a dict with keys vap_density, liq_density, Hvap, surf_tens, or, Pvap",
        )
    return exp_data, property_bounds, property_name


def fit_gp_models(df_data, data, property_name, pdf, gp_shuffle_seed = 1, save_fig = False):
    ### Fit GP Model to liquid density
    param_names = list(data.param_names) + ["temperature"]
    
    x_train, y_train, x_test, y_test = shuffle_and_split(
        df_data, param_names, property_name, shuffle_seed=gp_shuffle_seed, fraction_train=0.8
    )

    # Fit model
    models = {}

    models["RBF"] = run_gpflow_scipy(
        x_train,
        y_train,
        gpflow.kernels.RBF(lengthscales=np.ones(data.n_params + 1)),
    )

    models["Matern32"] = run_gpflow_scipy(
        x_train,
        y_train,
        gpflow.kernels.Matern32(lengthscales=np.ones(data.n_params + 1)),
    )

    models["Matern52"] = run_gpflow_scipy(
        x_train,
        y_train,
        gpflow.kernels.Matern52(lengthscales=np.ones(data.n_params + 1)),
    )
 
    # Plot model performance on train and test points
    exp_data, prop_bounds, prop_name = get_exp_data(data, property_name)
    if save_fig:
        pdf.savefig(plot_model_performance(models, x_train, y_train, prop_bounds))
        if len(x_test) > 0:
            pdf.savefig(plot_model_performance(models, x_test, y_test, prop_bounds))
            
    return models, x_train, y_train, x_test, y_test

def plot_model_performance(models, x_data, y_data, property_bounds, pdf, xylim=None, save_fig=False):
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

    ax.set_xlim(xylim[0], xylim[1])
    ax.set_ylim(xylim[0], xylim[1])
    ax.set_xlabel("Actual")
    ax.set_ylabel("Model Prediction")
    ax.legend()
    ax.set_aspect("equal", "box")

    if save_fig:
        pdf.savefig(fig)

    return mse_model
    
def plot_gp_slices(models, data, property_name, pdf):
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