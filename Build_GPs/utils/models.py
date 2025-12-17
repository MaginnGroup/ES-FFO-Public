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
from sklearn.linear_model import LinearRegression
import glob
from pathlib import Path
from gpflow.utilities import parameter_dict


# sys.path.append("../")
sys.path.append("../..")
from utils.prep_ms_data import prepare_df_props
from fffit.fffit.models import run_gpflow_scipy
from fffit.fffit.utils import shuffle_and_split, values_scaled_to_real, variances_scaled_to_real, values_real_to_scaled, variances_real_to_scaled
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


def get_prop_best_model(df_data, data, path_gps, gp_shuffle_seed=42, eotvos_scale=None):
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
        lj_param_names = list(data.param_names)
        # Get the property names from the data
        for prop_name in property_names:
            # Make GP models for each property that exists
            if prop_name in df_data.columns:
                #Remove groups of parameters that do not have at least 5 LD points with groupby
                if data.name == "DEC" and "vle_iters" in path_gps:  #DEC VLE special case where we need to filter an outlier
                    df_data_filt = df_data.groupby(lj_param_names).filter(lambda x: len(x) >= 5) 
                else:
                    df_data_filt = df_data

                models, x_train, y_train, x_test, y_test = fit_gp_models(
                    df_data_filt, data, prop_name, None, gp_shuffle_seed, False, data_path=dir_train_test, eotvos_scale=eotvos_scale
                )
                exp_data, property_bounds, name = get_exp_data(data, prop_name)
                # The best model is the one with the lowest MSE on the test set and not overfitting
                model_best, best_label = eval_model_performance(
                    models, x_test, y_test, property_bounds, xtrain=x_train, ytrain=y_train
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
    models_rq = {prop: models_props[prop]["RQ"] for prop in models_props.keys() if "RQ" in models_props[prop].keys()}

    if not os.path.exists(best_model_path):
        with open(best_model_path, "wb") as f:
            # Save the best models to a filewith open(gp_model_path, "wb") as f:
            pickle.dump(models_best, f)

    if not os.path.exists(best_model_path_act):
        with open(best_model_path_act, "wb") as f:
            # Save the best models to a filewith open(gp_model_path, "wb") as f:
            pickle.dump(models_rq, f)

    return models_best, models_rq, models_props, dir_train_test, best_labels


def fit_gp_models(df_data, data, property_name, pdf, gp_shuffle_seed=1, save_fig=False, data_path=None, eotvos_scale=None):
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
    # if (data.name != "Glycerol" and "liq_density" not in property_name) or (data.name == "Gly" and "liq_density" in property_name):
    # if data.name == "Glycerol":
    if "surf_tens" in property_name or (data.name == "Gly" and "liq_density" in property_name):
        gpConfig = {
            "useWhiteKernel": True, #Swap for Gly 
            "trainLikelihood": False, #Swap for Gly
            "anisotropic": True,
            "mean_function": "Linear",
        }
    else:
        gpConfig = {
            "useWhiteKernel": False,
            "trainLikelihood": True,
            "anisotropic": True,
            "mean_function": "Linear",
        }

    #For IFT set noise vriance
    if eotvos_scale != None and "surf_tens" in property_name:
        gpConfig["trainLikelihood"] = False
        gpConfig["noise_var"] = eotvos_scale #This is a scaled variance
        gpConfig["mean_function"] = "Custom"
    # print(gpConfig)
    ### Fit GP Model to liquid density
    param_names = list(data.param_names) + ["temperature"]

    #Check for existing data

    if data_path is not None and os.path.exists(os.path.join(data_path, f"{property_name}_x_train.csv")):
        x_train = np.loadtxt(os.path.join(data_path, f"{property_name}_x_train.csv"), delimiter=",", skiprows=1, usecols=range(1, len(param_names)+1))
        y_train = np.loadtxt(os.path.join(data_path, f"{property_name}_y_train.csv"), delimiter=",", skiprows=1, usecols=1)
        x_test = np.loadtxt(os.path.join(data_path, f"{property_name}_x_test.csv"), delimiter=",", skiprows=1, usecols=range(1, len(param_names)+1))
        y_test = np.loadtxt(os.path.join(data_path, f"{property_name}_y_test.csv"), delimiter=",", skiprows=1, usecols=1)
    else:
        x_train, y_train, x_test, y_test = shuffle_and_split(
            df_data,
            param_names,
            property_name,
            shuffle_seed=gp_shuffle_seed,
            fraction_train=0.8,
        )

    # Fit model
    models = {}
    kernels = ["RBF", "Matern32", "Matern52", "RQ"]
    for kernel in kernels:
        gpConfig["kernel"] = kernel
        models[kernel] = run_gpflow_scipy(x_train, y_train, gpConfig, restarts=3)

    # Plot model performance on train and test points
    exp_data, prop_bounds, prop_name = get_exp_data(data, property_name)
    if save_fig:
        pdf.savefig(plot_model_performance(models, x_train, y_train, prop_bounds))
        if len(x_test) > 0:
            pdf.savefig(plot_model_performance(models, x_test, y_test, prop_bounds))

    return models, x_train, y_train, x_test, y_test

def get_Eotvos_scale(df_csv, data_dict, mol_name, iter_type="vle_iters"):
    # Get all data
    #Get LD GPs
    files_ld = sorted(glob.glob(f"analysis/{mol_name}/ld_iters/iter-*/gp_models.pkl"))
    file_fin_ld = Path(files_ld[-1])
    #Ensure the file exists
    assert (file_fin_ld.exists()), f"{os.path.abspath(file_fin_ld)} does not exist. Check file path carefully."
    #Load the last file (most recent LD iter GPs) 
    with open(file_fin_ld, "rb") as pickle_file_ld:
        gp_models_ld, best_labels_ld = pickle.load(pickle_file_ld)
    LD_model = gp_models_ld["sim_liq_density"]["RQ"]
    data = data_dict[mol_name]
    #Get Tc, and ST values
    Tc = data.expt_Tc
    st_exp_data = data.expt_surf_tens
    if mol_name  == "MeOH" or mol_name == "EG":
        #Only get values where key < 430
        st_exp_data = {k: v for k, v in st_exp_data.items() if k < 430}
    st_data = st_exp_data.values()

    # Filter out rows with NaN values
    df_csv = df_csv.dropna().copy() 
    ld_threshold = (
        min(list(data.expt_liq_density.values()))
        + max(list(data.expt_vap_density.values()))
    ) / 2
    # df_csv = all_df_data[mol_name]
    iter_num = df_csv["iter"].max()

    dir_name = f"analysis/{mol_name}/{iter_type}/iter-{str(iter_num)}"
    os.makedirs(dir_name, exist_ok=True)
    #Get only liquid data, but do not scale
    df_all, df_liq, df_vapor = prepare_df_props(df_csv, data, ld_threshold, scale = False)
    df_all_scl, df_liq_scl, df_vapor_scl = prepare_df_props(df_csv, data, ld_threshold, scale = True)
    param_names = list(data.param_names)
    #Remove groups of parameters that do not have at least 5 LD points with groupby
    df_liq = df_liq.groupby(param_names).filter(lambda x: len(x) >= 5)
    df_liq_scl = df_liq_scl.groupby(param_names).filter(lambda x: len(x) >= 5)

    groups = df_liq.groupby(param_names)
    groups_scl = df_liq_scl.groupby(param_names)

    #Initialize eotvos mad list
    eotvos_mad_list = []
    for i, ((param_vals, group_df), (_, group_df_scl)) in enumerate(zip(groups, groups_scl)):
        #Organize group_df by temperature
        group_df = group_df.sort_values(by="temperature")
        #For each group, get sim_surf_tens and temperatures
        sim_surf_tens = group_df["sim_surf_tens"].values
        # sim_liq_dens = group_df["sim_liq_density"].values
        #Get GP-predicted LD
        GP_data = group_df_scl[param_names + ["temperature"]].values
        sim_liq_dens_scl, unc = LD_model.predict_f(GP_data)
        
        #Unscale GP-LD values
        y_bounds = data.liq_density_bounds
        y_bounds_st = data.surf_tens_bounds
        sim_liq_dens = values_scaled_to_real(sim_liq_dens_scl, y_bounds).flatten()

        #Fit Eotvos equation (ST = k*(M/rho_l)^(-2/3)*(Tc - 6 - T))
        temperatures = group_df["temperature"].values
        # X_eotvos = (Tc - 6 -temperatures).reshape(-1, 1)
        X_eotvos = temperatures.reshape(-1, 1)
        y_eotvos = sim_surf_tens*(((data.molecular_weight/1000) / (sim_liq_dens)) ** (2/3))
        reg = LinearRegression(fit_intercept=True).fit(X_eotvos, y_eotvos)
        #Predict ST using Eotvos
        eotvos_ST = reg.predict(X_eotvos)/(((data.molecular_weight/1000) / (sim_liq_dens)) ** (2/3))
        #Claculate MAD between Eotvos and sim_surf_tens
        mad_eotvos = np.mean(np.abs(eotvos_ST - sim_surf_tens))
        
        eotvos_mad_list.append(mad_eotvos)
    #Get the maximum Eotvos MAD for this molecule, this is the estimated "true" stdev
    #If glycerol use the mean, otherwise use max
    if mol_name == "Gly":
        avg_eotvos_mad = np.average(eotvos_mad_list)
    else:
        avg_eotvos_mad = np.max(eotvos_mad_list)
    #Save EotvosMAD list to a csv
    eotvos_df = pd.DataFrame(eotvos_mad_list, columns=["Eotvos_MAD"])
    eotvos_df.to_csv(f"{dir_name}/Eotvos_MAD_values.csv", index=False)

    eotvos_var = avg_eotvos_mad**2
    # print(f"Eotvos variance for {mol_name} is {eotvos_var:.4f} (MAD: {avg_eotvos_mad:.4f})")
    eotvos_var_scl = variances_real_to_scaled(np.array([[eotvos_var]]), data.surf_tens_bounds).flatten()[0]
    # print(f"Eotvos scaled variance for {mol_name} is {eotvos_var_scl:.6f}")
    return eotvos_var, eotvos_var_scl

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
        param_names = list(data.param_names)

        if iter_type == "vle_iters":
            #Get Eotvos scale for this molecule
            eotvos_var, eotvos_var_scl = get_Eotvos_scale(df_csv, data_dict, mol_name, iter_type=iter_type)
            #Set eotvos_var_scl to None to ignore Eotvos scaling and have the GP learn noise variance
            # eotvos_var_scl = None

        models_best, models_rq, all_models, dir_train_test, best_labels = get_prop_best_model(
            df_liq, data, dir_name, gp_shuffle_seed, eotvos_scale=eotvos_var_scl
        )

        models_molecs[mol_name] = models_best
        #Print hyperparameters of each best model
        #Print best kernels too
        # print(f"Best GP model hyperparameters for {mol_name}:") 
        for prop, model in models_best.items():
            # print(f" Property: {prop}, Kernel: {best_labels[prop]}")
            # gpflow.utilities.print_summary(model)
            #Send to a text file
            with open(f"{dir_name}/best_model_hypers.txt", "w") as f:
                params = parameter_dict(model)
                f.write(f"Property: {prop}, Kernel: {best_labels[prop]}\n")
                for k, v in params.items():
                    f.write(f"{k}: {v.numpy()}\n")

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


def eval_model_performance(models, x_data, y_data, property_bounds, xtrain=None, ytrain=None):
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
    if ytrain is not None and xtrain is not None:
        y_train_physical = values_scaled_to_real(ytrain, property_bounds)
        check_train = True
    else:
        check_train = False

    mse_min = np.inf
    mse_model = None
    mse_label = None
    for label, model in models.items():
        gp_mu, gp_var = model.predict_f(x_data)
        gp_mu_physical = values_scaled_to_real(gp_mu, property_bounds)
        meansqerr = np.mean((gp_mu_physical - y_data_physical.reshape(-1, 1)) ** 2)
        if meansqerr < mse_min:
            if check_train:
                # Also check that the model is not overfitting
                gp_mu_train, gp_var_train = model.predict_f(xtrain)
                gp_mu_train_physical = values_scaled_to_real(gp_mu_train, property_bounds)
                train_meansqerr = np.mean(
                    (gp_mu_train_physical - y_train_physical.reshape(-1, 1)) ** 2
                )
                # If the training MSE is less than 10x the test MSE, we consider it not overfitting
                print(
                    f"Model: {label}. Train MSE: {train_meansqerr:.2e}, Test MSE: {meansqerr:.2e}"
                )
                if train_meansqerr > 1e-7:
                    mse_min = meansqerr
                    mse_model = model
                    mse_label = label
            else:
                mse_min = meansqerr
                mse_model = model
                mse_label = label

        print("Model: {}. Mean squared err: {:.2e}".format(label, meansqerr))

    return mse_model, mse_label

def loo_model_perform(models, x_data, y_data, property_bounds):
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

    Returns
    -------
    matplotlib.Figure.figure
    """
    y_data_physical = values_scaled_to_real(y_data, property_bounds)
    # print("True Values", y_data_physical)

    for label, model in models.items():
        gp_mu, gp_var = model.predict_f(x_data)
        gp_mu_physical = values_scaled_to_real(gp_mu, property_bounds)
        # print("GP Pred",gp_mu_physical)
        meansqerr = np.nanmean((gp_mu_physical - y_data_physical.reshape(-1, 1)) ** 2)
        mapd = np.nanmean(np.abs((gp_mu_physical - y_data_physical.reshape(-1, 1)) / y_data_physical.reshape(-1, 1))) * 100.0

        # print("Model: {}. Mean squared err: {:.2e}".format(label, meansqerr))

    return meansqerr, mapd