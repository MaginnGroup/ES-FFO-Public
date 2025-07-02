import sys
import gpflow
import numpy as np
import gpflow
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import svm

# sys.path.append("../")
sys.path.append("../..")
from utils.prep_ms_data import prepare_df_props
from fffit.fffit.models import run_gpflow_scipy
from fffit.fffit.utils import shuffle_and_split, values_scaled_to_real
from fffit.fffit.plot import plot_model_performance

sys.path.remove("../..")


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


def get_prop_best_model(df_data, data, path_gps, gp_shuffle_seed=42):
    """
    Get the best GP model for a given property and save the training and test data to a file.

    Parameters
    ----------
    df_data : pd.DataFrame
        The dataframe containing the data for the property.
    data : object
        The data object containing the molecule information.
    path_gps : str
        The path to save the GP models and training/test data.
    gp_shuffle_seed : int, default 42
        The seed for shuffling the data.

    Returns
    -------
    models_best : dict
        The best GP models for each property.
    models_props : dict
        The GP models for each property.
    dir_train_test : str
        The directory where the training and test data is saved.
    """
    dir_train_test = path_gps + "/train_test_sets/"
    os.makedirs(dir_train_test, exist_ok=True)
    gp_model_path = os.path.join(path_gps, "gp_models.pkl")
    best_model_path = os.path.join(path_gps, "best_gp_models.pkl")
    best_model_path_act = os.path.join(path_gps, "rq_gp_models.pkl")

    if os.path.exists(gp_model_path):
        # Load the GP models from the file
        with open(gp_model_path, "rb") as f:
            models_props, best_labels = pickle.load(f)
    else:
        models_props = {}
        best_labels = {}
        property_names = [
            "sim_liq_density",
            "sim_surf_tens",
            "sim_vap_density",
            "sim_Pvap",
            "sim_Hvap",
        ]
        param_names = list(data.param_names) + ["temperature"]
        # Get the property names from the data
        for prop_name in property_names:
            # Make GP models for each property that exists
            if prop_name in df_data.columns:
                models, x_train, y_train, x_test, y_test = fit_gp_models(
                    df_data, data, prop_name, None, gp_shuffle_seed, False
                )
                exp_data, property_bounds, name = get_exp_data(data, prop_name)
                # The best model is the one with the lowest MSE on the test set
                model_best, best_label = eval_model_performance(
                    models, x_test, y_test, property_bounds
                )
                models_props[prop_name] = models
                best_labels[prop_name] = best_label
                # Save training and test data to a file
                df_xtrain = pd.DataFrame(x_train, columns=param_names)
                df_xtest = pd.DataFrame(x_test, columns=param_names)
                df_xtrain.to_csv(
                    f"{dir_train_test}/{prop_name}_x_train.csv", index=True
                )
                df_xtest.to_csv(f"{dir_train_test}/{prop_name}_x_test.csv", index=True)
                df_ytrain = pd.DataFrame(y_train, columns=[prop_name])
                df_ytest = pd.DataFrame(y_test, columns=[prop_name])
                df_ytrain.to_csv(
                    f"{dir_train_test}/{prop_name}_y_train.csv", index=True
                )
                df_ytest.to_csv(f"{dir_train_test}/{prop_name}_y_test.csv", index=True)
        with open(gp_model_path, "wb") as f:
            pickle.dump((models_props, best_labels), f)

    models_best = {
        prop: models_props[prop][best_labels[prop]] for prop in models_props.keys()
    }

    # Optionally, put the best_label here, but for now we use RQ as the best model
    models_rq = {prop: models_props[prop]["RQ"] for prop in models_props.keys()}

    if not os.path.exists(best_model_path):
        with open(best_model_path, "wb") as f:
            # Save the best models to a filewith open(gp_model_path, "wb") as f:
            pickle.dump(models_best, f)

    if not os.path.exists(best_model_path_act):
        with open(best_model_path_act, "wb") as f:
            # Save the best models to a filewith open(gp_model_path, "wb") as f:
            pickle.dump(models_rq, f)

    return models_best, models_rq, models_props, dir_train_test


def fit_gp_models(df_data, data, property_name, pdf, gp_shuffle_seed=1, save_fig=False):
    """
    Fit GP models to the given property data and plot the model performance.

    Parameters
    ----------
    df_data : pd.DataFrame
        The dataframe containing the data for the property.
    data : object
        The data object containing the molecule information.
    property_name : str
        The name of the property to fit the GP models to.
    pdf : PdfPages or None (None if save_fig is False)
        The PdfPages object to save the plots to.
    gp_shuffle_seed : int, default 1
        The seed for shuffling the data.
    save_fig : bool, default False
        Whether to save the plots to a file.

    Returns
    -------
    models : dict
        The fitted GP models for the given property.
    x_train : np.ndarray
        The training data for the GP models.
    y_train : np.ndarray
        The training labels for the GP models.
    x_test : np.ndarray
        The test data for the GP models.
    y_test : np.ndarray
        The test labels for the GP models.
    """
    gpConfig = {
        "useWhiteKernel": False,
        "trainLikelihood": True,
        "anisotropic": True,
        "mean_function": "Linear",
    }

    ### Fit GP Model to liquid density
    param_names = list(data.param_names) + ["temperature"]

    x_train, y_train, x_test, y_test = shuffle_and_split(
        df_data,
        param_names,
        property_name,
        shuffle_seed=gp_shuffle_seed,
        fraction_train=0.8,
    )

    # Fit model
    models = {}

    for kernel in ["RBF", "Matern32", "Matern52", "RQ"]:
        gpConfig["kernel"] = kernel
        models[kernel] = run_gpflow_scipy(x_train, y_train, gpConfig, restarts=3)

    # Plot model performance on train and test points
    exp_data, prop_bounds, prop_name = get_exp_data(data, property_name)
    if save_fig:
        pdf.savefig(plot_model_performance(models, x_train, y_train, prop_bounds))
        if len(x_test) > 0:
            pdf.savefig(plot_model_performance(models, x_test, y_test, prop_bounds))

    return models, x_train, y_train, x_test, y_test


def get_best_models(all_df_data, data_dict, iter_type="ld_iters", gp_shuffle_seed=42):
    """
    Get the best GP models for all molecules and properties and save them to a file.
    Parameters
    ----------
    all_df_data : dict
        Dictionary of dataframes for each molecule.
    data_dict : dict
        Dictionary of data objects for each molecule.
    iter_type : str, default "ld_iters"
        The type of iteration (e.g., "ld_iters", "vle_iters").
    gp_shuffle_seed : int, default 42
        The seed for shuffling the data.
    save_fig : bool, default False
        Whether to save the plots to a file.

    Returns
    -------
    models_molecs : dict
        Dictionary of best GP models for each molecule.
    """
    # Get all data
    models_molecs = {}
    for mol_name, df_csv in all_df_data.items():
        data = data_dict[mol_name]
        df_csv = df_csv.dropna().copy()  # Filter out rows with NaN values
        # Filter out rows with NaN values
        ld_threshold = (
            min(list(data.expt_liq_density.values()))
            + max(list(data.expt_vap_density.values()))
        ) / 2
        # df_csv = all_df_data[mol_name]
        iter_num = df_csv["iter"].max()

        dir_name = f"analysis/{mol_name}/{iter_type}/iter-{str(iter_num)}"
        os.makedirs(dir_name, exist_ok=True)

        df_all, df_liq, df_vapor = prepare_df_props(df_csv, data, ld_threshold)

        models_best, models_rq, all_models, dir_train_test = get_prop_best_model(
            df_liq, data, dir_name, gp_shuffle_seed
        )

        models_molecs[mol_name] = models_best

    # Save all models to a file if there are multiple molecules
    if len(list(all_df_data.keys())) > 1:
        names = "-".join(sorted(all_df_data.keys()))
        dir2 = f"analysis/{names}/{iter_type}/iter-{str(iter_num)}"
        os.makedirs(dir2, exist_ok=True)
        with open(dir2 + "/best_gp_models.pkl", "wb") as f:
            pickle.dump(models_molecs, f)

    return models_molecs


def build_classifier(
    df_iter1, root_dir, data, cl_shuffle_seed=1, verbose=True, save_fig=False
):
    """
    Classify samples as liquid or vapor using a SVM classifier.
    Parameters
    ----------
    df_iter1 : pd.DataFrame
        The dataframe for the first iteration
    root_dir : str
        The root directory for saving the results
    data : object
        The data object containing the molecule information
    cl_shuffle_seed : int, default 1
        The seed for shuffling the data
    verbose : bool, default True
        Whether to print the classifier accuracy
    save_fig : bool, default False
        Whether to save the confusion matrix figure
    Returns
    -------
    classifier : sklearn.svm.SVC
        The trained SVM classifier
    """

    # Create training/test set
    param_names = list(data.param_names) + ["temperature"]
    property_name = "is_liquid"
    x_train, y_train, x_test, y_test = shuffle_and_split(
        df_iter1, param_names, property_name, shuffle_seed=cl_shuffle_seed
    )

    clas_data_dir = root_dir + "/classifier_data/"
    os.makedirs(clas_data_dir, exist_ok=True)
    # Save classifier training and test data
    df_xtrain = pd.DataFrame(x_train, columns=param_names)
    df_xtest = pd.DataFrame(x_test, columns=param_names)
    df_xtrain.to_csv(f"{clas_data_dir}/classifier_x_train.csv", index=True)
    df_xtest.to_csv(f"{clas_data_dir}/classifier_x_test.csv", index=True)
    df_ytrain = pd.DataFrame(y_train, columns=[property_name])
    df_ytest = pd.DataFrame(y_test, columns=[property_name])
    df_ytrain.to_csv(f"{clas_data_dir}/classifier_y_train.csv", index=True)
    df_ytest.to_csv(f"{clas_data_dir}/classifier_y_test.csv", index=True)

    # Create and fit classifier
    # class_weight "balanced" used because there are fewer liquid than vapor samples in the LHS sets
    classifier = svm.SVC(kernel="rbf", class_weight="balanced")
    classifier.fit(x_train, y_train)
    test_score = classifier.score(x_test, y_test)
    if verbose:
        print(f"Classifer is {test_score*100.0}% accurate on the test set.")
    ConfusionMatrixDisplay.from_estimator(
        classifier, x_test, y_test, display_labels=["Vapor", "Liquid"]
    )
    if save_fig:
        plt.savefig(root_dir + "/classifier.pdf")
    return classifier


def eval_model_performance(models, x_data, y_data, property_bounds):
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

    mse_min = np.inf
    mse_model = None
    mse_label = None
    for label, model in models.items():
        gp_mu, gp_var = model.predict_f(x_data)
        gp_mu_physical = values_scaled_to_real(gp_mu, property_bounds)
        meansqerr = np.mean((gp_mu_physical - y_data_physical.reshape(-1, 1)) ** 2)
        if meansqerr < mse_min:
            mse_min = meansqerr
            mse_model = model
            mse_label = label
        print("Model: {}. Mean squared err: {:.2e}".format(label, meansqerr))

    return mse_model, mse_label
