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

from utils.prep_ms_data import prepare_df_props

from fffit.fffit.models import run_gpflow_scipy
from fffit.fffit.utils import shuffle_and_split, values_scaled_to_real

from fffit.fffit.plot import (
    plot_model_performance,
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

def get_prop_best_model(df_data, data, path_gps, gp_shuffle_seed=42):
    dir_train_test = path_gps + "/train_test_sets/"
    os.makedirs(dir_train_test, exist_ok=True)
    gp_model_path = os.path.join(path_gps, "gp_models.pkl")
    best_model_path = os.path.join(path_gps, "best_gp_models.pkl")

    if os.path.exists(gp_model_path):
        # Load the GP models from the file
        with open(gp_model_path, "rb") as f:
            models_props, best_labels = pickle.load(f)
    else:
        models_props = {}
        best_labels = {}
        property_names = ["sim_liq_density", "sim_surf_tens", "sim_vap_density", "sim_Pvap", "sim_Hvap"]
        param_names = list(data.param_names) + ["temperature"]
        # Get the property names from the data
        for prop_name in property_names:
            #Make GP models for each property that exists
            if prop_name in df_data.columns:
                models, x_train, y_train, x_test, y_test = fit_gp_models(df_data, data, prop_name, None, gp_shuffle_seed, False)
                exp_data, property_bounds, name = get_exp_data(data, prop_name)
                model_best, best_label = plot_model_performance(models, x_test, y_test, property_bounds, None, False)
                models_props[prop_name] = models
                best_labels[prop_name] = best_label
                # Save training and test data to a file
                df_xtrain = pd.DataFrame(x_train, columns=param_names)
                df_xtest = pd.DataFrame(x_test, columns=param_names)
                df_xtrain.to_csv(f"{dir_train_test}/{prop_name}_x_train", index=False)
                df_xtest.to_csv(f"{dir_train_test}/{prop_name}_x_test", index=False)
                df_ytrain = pd.DataFrame(y_train, columns=[prop_name])
                df_ytest = pd.DataFrame(y_test, columns=[prop_name])
                df_ytrain.to_csv(f"{dir_train_test}/{prop_name}_y_train", index=False)
                df_ytest.to_csv(f"{dir_train_test}/{prop_name}_y_test", index=False)
        with open(gp_model_path, "wb") as f:
            pickle.dump((models_props, best_labels), f)

    models_best = {prop : models_props[prop][best_labels[prop]] for prop in models_props.keys()}

    if not os.path.exists(best_model_path):
        with open(best_model_path, "wb") as f:
            # Save the best models to a filewith open(gp_model_path, "wb") as f:
            pickle.dump(models_best, f)

    return models_best, models_props, dir_train_test

def fit_gp_models(df_data, data, property_name, pdf, gp_shuffle_seed = 1, save_fig = False):

    gpConfig={'useWhiteKernel':False,
            'trainLikelihood':True,
            'anisotropic':True,
            'mean_function':"Linear"}
    
    ### Fit GP Model to liquid density
    param_names = list(data.param_names) + ["temperature"]
    
    x_train, y_train, x_test, y_test = shuffle_and_split(
        df_data, param_names, property_name, shuffle_seed=gp_shuffle_seed, fraction_train=0.8
    )

    # Fit model
    models = {}

    for kernel in ["RBF", "Matern32", "Matern52", "RQ"]:
        gpConfig["kernel"] = kernel
        models[kernel] = run_gpflow_scipy(x_train, y_train, gpConfig, restarts = 1)
 
    # Plot model performance on train and test points
    exp_data, prop_bounds, prop_name = get_exp_data(data, property_name)
    if save_fig:
        pdf.savefig(plot_model_performance(models, x_train, y_train, prop_bounds))
        if len(x_test) > 0:
            pdf.savefig(plot_model_performance(models, x_test, y_test, prop_bounds))
            
    return models, x_train, y_train, x_test, y_test

def get_best_models(all_df_data, data_dict, iter_type = "ld_iters", gp_shuffle_seed = 42, save_fig=False):
    #Get all data
    models_molecs = {}
    for mol_name, df_csv in all_df_data.items():
        data = data_dict[mol_name]
        ld_threshold = (min(list(data.expt_liq_density.values())) + max(list(data.expt_vap_density.values())))/2
        # df_csv = all_df_data[mol_name]
        iter_num = df_csv["iter"].max()

        dir_name = f"analysis/{mol_name}/{iter_type}/iter-{str(iter_num)}"
        os.makedirs(dir_name, exist_ok=True)
        if save_fig:
            pdf_name = os.path.join(dir_name , "fig_gp_examples.pdf")
            pdf = PdfPages(pdf_name)
        else:
            pdf = None

        df_all, df_liq, df_vapor = prepare_df_props(df_csv, data, ld_threshold)

        path_gps = os.path.join(dir_name, "gp_models.pkl")
        models_best, all_models, dir_train_test = get_prop_best_model(df_liq, data, path_gps, gp_shuffle_seed)
            
        models_molecs[mol_name] = models_best

    dir2 = f"analysis/all_mols/{iter_type}/iter-{str(iter_num)}"
    with open(dir2 + "/gp_models.pkl", "wb") as f:
        pickle.dump(models_molecs, f)

    return models_molecs

def build_classifier(df_iter1, root_dir, data, cl_shuffle_seed=1, verbose=True, save_fig=False):
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
        df_iter1, param_names, property_name, cl_shuffle_seed
    )

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
        plt.savefig(root_dir + "classifier.pdf")
    return classifier