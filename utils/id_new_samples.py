import numpy as np
import pandas as pd
import os
import gpflow
import matplotlib.pyplot as plt
import seaborn
from sklearn.metrics import ConfusionMatrixDisplay
from numpy.linalg import norm
from sklearn import svm

from fffit.fffit.models import run_gpflow_scipy

from fffit.fffit.utils import values_real_to_scaled, values_scaled_to_real, shuffle_and_split

def opt_dist(distance, top_samples, constants, target_num, rand_seed=None, eval=False):
    """
    Calculates the distance between points such that exactly a target number of points are chosen for the next iteration

    Parameters:
    -----------
        distance: float, The allowable minimum distance between points
        top_samples: pandas data frame, Collection of top liquid/vapor sampes
        constants: utils.r41.R41Constants, contains the infromation for a certain refrigerant
        target_num: int, the number of samples to choose next
        rand_seed: int, the seed number to use: None by default
        eval: bool, Determines whether error is calculated or new_points is returned

    Returns:
        error: float, The squared error between the target value and number of new_points
        OR
        new_points: pandas data frame, a pandas data frame containing the number of points to be used
    """
    if len(top_samples) <= target_num:
        print("Trying dist =", distance)

    top_samp0 = top_samples.copy()
    if rand_seed != None:
        np.random.seed(rand_seed)
    new_points = pd.DataFrame()
    discarded_points = pd.DataFrame(columns=top_samples.columns)
    while len(top_samples > 0):
        # Shuffle the pareto points
        top_samples = top_samples.sample(frac=1)
        new_samples_top = pd.DataFrame(top_samples.iloc[[0]])
        new_points = pd.concat([new_points, new_samples_top])
        # Remove anything within distance
        l1_norm = np.sum(
            np.abs(
                top_samples[list(constants.param_names)].values
                - new_points[list(constants.param_names)].iloc[[-1]].values
            ),
            axis=1,
        )
        points_to_remove = np.where(l1_norm <= distance)[
            0
        ]  # Changed to <= to get zero bc to work
        points_to_remove_df = pd.DataFrame(top_samples.iloc[points_to_remove])
        discarded_points = pd.concat([discarded_points, points_to_remove_df])
        # discarded_points = discarded_points.append(
        #     top_samples.iloc[points_to_remove]
        # )
        top_samples.drop(index=top_samples.index[points_to_remove], inplace=True)

    #     error = target_num - len(new_points)

    #     print("Error = ",error)
    #     return error
    if eval == True:
        if len(new_points) > target_num:
            # randomly remove extra points
            new_points = new_points.sample(n=target_num, random_state=rand_seed)
        return new_points
    else:
        #         return error
        return len(new_points)


def bisection(
    lower_bound,
    upper_bound,
    error_tol,
    top_samples,
    constants,
    target_num,
    rand_seed=None,
    verbose=False,
):
    """
    approximates a root of a function bounded by lower_bound and upper_bound to within a tolerance

    Parameters:
    -----------
        lower_bound: float, lower bound of the distance, must be > 0
        upper_bound: float, lower bound of the distance, must be > lower_bound
        error_tol: float, tolerance of error
        top_samples: pandas data frame, Collection of top liquid/vapor sampes
        constants: utils.r41.R41Constants, contains the infromation for a certain refrigerant
        target_num: int, the number of samples to choose next
        rand_seed: int, the seed number to use: None by default

    Returns:
    --------
        midpoint: The distance that satisfies the error criteria based on the target number

    """
    assert (
        len(top_samples) >= target_num
    ), "Ensure you have at least as many samples as the target number!"
    # Initialize Termination criteria and add assert statements
    assert lower_bound >= 0, "Lower bound must be greater than 0"
    assert lower_bound < upper_bound, "Lower bound must be less than the upper bound"

    # Set error of upper and lower bound
    # print("Low B", lower_bound)
    # print("High B", upper_bound)
    eval_lower_bound = opt_dist(
        lower_bound, top_samples, constants, target_num, rand_seed
    )
    eval_upper_bound = opt_dist(
        upper_bound, top_samples, constants, target_num, rand_seed
    )
    # print("Low Eval",eval_lower_bound )
    # print("High Eval",eval_upper_bound )

    # Throw Error if initial guesses are bad
    if not (eval_lower_bound >= target_num >= eval_upper_bound):
        print("Increase Length of Upper Bound. Given bounds do not include the root!")

    # While error > tolerance
    while (upper_bound - lower_bound) > error_tol:
        # Find the midpoint and evaluate it
        midpoint = (lower_bound + upper_bound) / 2
        #         print("Mid B", midpoint)
        eval_midpoint = opt_dist(
            midpoint, top_samples, constants, target_num, rand_seed
        )
        #         print("Mid Eval", eval_midpoint)
        error = target_num - eval_midpoint
        if verbose == True:
            print("distance = %0.6f and error = %0.6f" % (midpoint, error))

        # Set the upper or lower bound depending on sign
        if eval_midpoint == target_num:
            # Terminate loop if correct number of points is found
            break
        elif eval_midpoint < target_num:
            upper_bound = midpoint
        else:
            lower_bound = midpoint

    final_distance = lower_bound
    final_eval = opt_dist(final_distance, top_samples, constants, target_num, rand_seed)
    if final_eval < target_num:
        final_distance = upper_bound  # Just in case lower_bound fails, use upper_bound
        final_eval = opt_dist(
            final_distance, top_samples, constants, target_num, rand_seed
        )

    return final_distance, final_eval - target_num


def prep_df_density(mol_name, data, df_csv):
    """
    Prepare the density dataframe for a given molecule.

    Parameters
    ----------
    mol_name : str
        The name of the molecule
    data : object
        The data object containing the molecule information

    Returns
    -------
    df_iter1 : pd.DataFrame
        The dataframe for the first iteration
    root_dir : str
        The root directory for saving the results
    """
    #Prepare df_density
    ld_threshold = data.expt_rhoc
    
    print(df_csv.head())
    df_csv["dens-iter"] = df_csv["dens-iter"].astype(int)
    df_iter1_csv = df_csv[df_csv["dens-iter"] == 1].copy()
    
    df_all, df_liquid, df_vapor = prepare_df_density(
        df_csv, data, ld_threshold
    )
    df_iter1_all, df_iter1_l, df_iter1_v = prepare_df_density(
        df_iter1_csv, data, ld_threshold
    )
    root_dir = "density_iters/analysis/" + mol_name + "/"

    return df_iter1_all, df_liquid, root_dir

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

def fit_gp_model(df_liquid, data, gp_shuffle_seed=42):
    # Create training/test set
    param_names = list(data.param_names) + ["temperature"]
    models = {}

    for property_name in ["md_liq_density", "md_surf_tens"]:
        x_train, y_train, x_test, y_test = shuffle_and_split(
        df_liquid, param_names, property_name, shuffle_seed=gp_shuffle_seed
    )
        models[property_name] = run_gpflow_scipy(
            x_train,
            y_train,
            gpflow.kernels.RBF(lengthscales=np.ones(data.n_params + 1)),
        )
    return models

def rank_vl_samples(liquid_samples, vapor_samples, models, data, verbose=True):
    # Find the lowest MSE points from the GP in both sets
    ranked_liquid_samples = rank_samples(liquid_samples, models["md_liq_density"], data, "sim_liq_density")
    ranked_vapor_samples = rank_samples(
        vapor_samples, models["md_liq_density"], data, "sim_liq_density"
    )  # both l and g compared to liquid density

    # Make a set of the lowest MSE parameter sets
    top_liquid_samples = ranked_liquid_samples[ranked_liquid_samples["mse"] < 625.0]
    top_vapor_samples = ranked_vapor_samples[ranked_vapor_samples["mse"] < 625.0]

    if verbose:
        print(
            "There are:",
            top_liquid_samples.shape[0],
            "liquid parameter sets which produce densities within 25 kg/m$^3$ of experimental densities",
        )
        print(
            "There are:",
            top_vapor_samples.shape[0],
            " vapor parameter sets which produce densities within 25 kg/m$^3$ of experimental densities",
        )
    return top_liquid_samples, top_vapor_samples

def vis_top_samples(top_liquid_samples, top_vapor_samples, data, root_dir, iter_num, save_fig=False):
    column_names = list(data.param_names)
    objects = {}
    for i, sample in enumerate([top_liquid_samples, top_vapor_samples]):
        if i ==0:
            phase = "liq"
        else:
            phase = "vap"

        g = seaborn.pairplot(sample.drop(columns=["mse"]))
        g.set(xlim=(-0.1, 1.1), ylim=(-0.1, 1.1))

        if save_fig:
            g.savefig(root_dir + f"{phase}_mse_below625.pdf")

        #Drop the mse column for 
        new_sample_params = [sample.drop(columns=["mse"])]
        # Concatenate into a single dataframe and save to CSV
        new_sample_params = pd.concat(new_sample_params)
        dir_name = root_dir + "dens-iter-" + str(iter_num) + "/"
        os.makedirs(dir_name, exist_ok=True)
        samp_path = os.path.join(dir_name, f"{phase}-params.csv")
        new_sample_params.to_csv(samp_path)
        top_samp = new_sample_params.reset_index(drop=True)
        objects[phase] = top_samp
    top_liq = objects["liq"]
    top_vap = objects["vap"]

    return top_liq, top_vap

def get_next_iter_params(top_liq, top_vap, data, root_dir, iter_num, target_total=200, dist_seed=1, verbose=True):
    """
    Get the next set of parameters for MD simulations.
    Parameters
    ----------
    top_liq : pd.DataFrame
        The dataframe for the top liquid samples
    top_vap : pd.DataFrame
        The dataframe for the top vapor samples
    data : object
        The data object containing the molecule information
    root_dir : str
        The root directory for saving the results
    iter_num : int
        The current iteration number
    target_total : int, default 200
        The target number of samples to return
    dist_seed : int, default 1
        The seed for the random number generator
    verbose : bool, default True
        Whether to print the classifier accuracy

    Returns
    -------
    next_iter_params : pd.DataFrame
        The dataframe for the next iteration parameters
    """
    target_total = 200
    final_sample_file = root_dir + "params-iter-" + str(iter_num + 1) + ".csv"
    # We want to have as many liquid points as possible, but no more than 200 total and the rest vapor
    target_num_l = np.minimum(200, len(top_liq))
    target_num_v = target_total - target_num_l
    if verbose:
        print(target_num_l, target_num_v)

    zero_array = np.zeros(top_liq.shape[1])
    one_array = np.ones(top_liq.shape[1])
    ub_array = one_array - zero_array

    # lower_bound = 1e-8
    lower_bound = 0
    # IL norm between the highest high parameter space, and lowest low parameter space value
    upper_bound = norm(ub_array, 1)  # This number will be 10, the number of dimensions
    error_tol = 1e-8

    # If we have enough liquid samples, we want to find the distance that will give us the target number of liquid samples
    if len(top_liq) >= target_total:
        distance_opt_l, number_points_l = bisection(
            lower_bound, upper_bound, error_tol, top_liq, data, target_num_l, dist_seed
        )
        new_points_l = opt_dist(
            distance_opt_l, top_liq, data, target_num_l, rand_seed=dist_seed, eval=True
        )
        if verbose:
            print(
                "\nRequired Distance for liquid is : %0.8f and there are %0.1f points too many"
                % (distance_opt_l, number_points_l)
            )
            print(
                len(new_points_l),
                "top liquid density points are left after removing similar points using a distance of",
                np.round(distance_opt_l, 5),
            )
        
        next_iter_params = new_points_l
    # If we don't we want to find the vapor sets to add
    else:
        distance_opt_v, number_points_v = bisection(
            lower_bound, upper_bound, error_tol, top_vap, data, target_num_v, dist_seed
        )
        new_points_v = opt_dist(
            distance_opt_v, top_vap, data, target_num_v, rand_seed=dist_seed, eval=True
        )
        if verbose:
            print(
                "\nRequired Distance for vapor is : %0.8f and there are %0.1f points too many"
                % (distance_opt_v, number_points_v)
            )
            
            print(
                len(new_points_v),
                "top vapor density points are left after removing similar points using a distance of",
                np.round(distance_opt_v, 5),
            )
        next_iter_params = pd.concat([top_liq, new_points_v], axis=0)
    return next_iter_params, final_sample_file

def prepare_df_density(df_csv, molecule, liquid_density_threshold):
    """Prepare a pandas dataframe for fitting a GP model to density data

    Performs the following actions:
       - Renames "density" to "md_liq_density"
       - Adds "expt_density"
       - Adds "is_liquid"
       - Converts all values from physical values to scaled values

    Parameters
    ----------
    df_csv : pd.DataFrame
        The dataframe as loaded from a CSV file with the signac results
    molecule : RXXConstants
        An instance of a molecule constants class
    liquid_density_threshold : float
        Density threshold (kg/m^3) for distinguishing liquid and vapor

    Returns
    -------
    df_all : pd.DataFrame
        The dataframe with scaled parameters, temperature, density, and is_liquid
    df_liquid : pd.DataFrame
        `df_all` where `is_liquid` is True
    df_vapor : pd.DataFrame
        `df_all` where `is_liquid` is False
    """
    if "density" not in df_csv.columns or "surf_tens" not in df_csv.columns:
        raise ValueError("df_csv must contain column 'density' and 'surf_tens'")
    if "temperature" not in df_csv.columns:
        raise ValueError("df_csv must contain column 'temperature'")
    for param in list(molecule.param_names):
        if param not in df_csv.columns:
            raise ValueError(f"df_csv must contain a column for parameter: '{param}'")

    # Add expt density and is_liquid
    df_all = df_csv.rename(
        columns={"density": "md_liq_density", "surf_tens": "md_surf_tens"}
    )
    df_all["expt_density"] = df_all["temperature"].map(molecule.expt_liq_density)
    df_all["expt_surf_tens"] = df_all["temperature"].map(molecule.expt_surf_tens)
    df_all["is_liquid"] = df_all["md_liq_density"] > liquid_density_threshold

    # Scale all values
    scaling_info = {
        "temperature": molecule.temperature_bounds(),
        "md_liq_density": molecule.liq_density_bounds,
        "expt_density": molecule.liq_density_bounds,
        "md_surf_tens": molecule.surf_tens_bounds,
        "expt_surf_tens": molecule.surf_tens_bounds,
    }

    # Scale param values
    df_all[list(molecule.param_names)] = values_real_to_scaled(
        df_all[list(molecule.param_names)], molecule.param_bounds
    )

    # Scale other properties
    for col, bounds_func in scaling_info.items():
        df_all[col] = values_real_to_scaled(df_all[col], bounds_func)

    # Split out vapor and liquid samples
    df_liquid = df_all[df_all["is_liquid"] == True]
    df_vapor = df_all[df_all["is_liquid"] == False]

    return df_all, df_liquid, df_vapor


def classify_samples(samples, classifier):
    """Evaulate the classifer and return predicted liquid and vapor samples

    Parameters
    ----------
    samples : np.ndarray, shape=(n_samples, n_params)
        Samples to rank
    classifier : sklearn.svm.SVC
        Classifier to distinguish between liquid and vapor

    Returns
    -------
    liquid_samples : np.ndarray, shape=(n_liquid, n_params)
        Samples classified as liquid
    vapor_samples : np.ndarray, shape=(n_liquid, n_params)
        Samples classified as liquid

    """

    # Classification performed at the highest temperature
    # Append highest temperature (1.0) to LH samples
    samples_temperature = np.hstack((samples, np.tile(1.0, (samples.shape[0], 1))))

    # Apply clasifier
    pred = classifier.predict(samples_temperature)

    # Separate LH samples into predicted liquid and predicted vapor
    liquid_samples = samples[np.where(pred == 1)]
    vapor_samples = samples[np.where(pred == 0)]
    print("Shape of samples to classify:", samples.shape)
    print("Shape of the predicted liquid samples:", liquid_samples.shape)
    print("Shape of the predicted vapor samples:", vapor_samples.shape)

    return liquid_samples, vapor_samples


def rank_samples(samples, gp_model, molecule, property_name, property_offset=0.0):
    """Evalulate the GP model for a samples and return ranked results
    from lowest to highest MSE with experiment across the temperature range

    Parameters
    ----------
    samples : np.ndarray, shape=(n_samples, n_params)
        Samples to rank
    gp_model : gpflow.model
        GP model to predict the property_name of each sample
    molecule : RXXConstants
        An instance of a molecule constants class
    property_name : string
        The name of the property of interest. Valid options are
        "liq_density", "vap_density", "Pvap", "Hvap"
    property_offset : float
        Adjust the value predicted by the gp model by this amount.
        Quantity specified in physical units

    Returns
    -------
    ranked_samples : pd.DataFrame
        Samples sorted by MSE
    """

    valid_property_names = [
        "sim_liq_density",
        "sim_vap_density",
        "sim_Pvap",
        "sim_surf_tens",
    ]

    if property_name not in valid_property_names:
        raise ValueError(
            "Invalid property_name {}. Supported property_names are "
            "{}".format(property_name, valid_property_names)
        )
    print("Include properties:", property_name)
    temperature_bounds = molecule.temperature_bounds
    if property_name == "sim_liq_density":
        expt_property = molecule.expt_liq_density
        property_bounds = molecule.liq_density_bounds
    elif property_name == "sim_vap_density":
        expt_property = molecule.expt_vap_density
        property_bounds = molecule.vap_density_bounds
    elif property_name == "sim_Pvap":
        expt_property = molecule.expt_Pvap
        property_bounds = molecule.Pvap_bounds
    elif property_name == "sim_surf_tens":
        expt_property = molecule.expt_surf_tens
        property_bounds = molecule.surf_tens_bounds
    print("Property bounds are", property_bounds)
    # Apply GP model and calculate mean squared errors (MSE) between
    # GP model predictions and experimental data for all parameter samples
    mse = _calc_gp_mse(
        gp_model,
        samples,
        expt_property,
        property_bounds,
        temperature_bounds,
        property_offset,
    )
    print("MSE is", mse)
    # Make pandas dataframes, rank, and return
    samples_mse = np.hstack((samples, mse.reshape(-1, 1)))
    samples_mse = pd.DataFrame(
        samples_mse, columns=list(molecule.param_names) + ["mse"]
    )
    ranked_samples = samples_mse.sort_values("mse")

    return ranked_samples


def _calc_gp_mse(
    gp_model,
    samples,
    expt_property,
    property_bounds,
    temperature_bounds,
    property_offset=0.0,
):
    """Calculate the MSE between the GP model and experiment for samples"""

    all_errs = np.empty(shape=(samples.shape[0], len(expt_property.keys())))
    print("Initialize all errors!")
    col_idx = 0
    for temp, density in expt_property.items():
        print("Trying temp", temp, "and density", density)
        scaled_temp = values_real_to_scaled(temp, temperature_bounds)
        xx = np.hstack((samples, np.tile(scaled_temp, (samples.shape[0], 1))))
        means_scaled, vars_scaled = gp_model.predict_f(xx)
        means = values_scaled_to_real(means_scaled, property_bounds)
        means = means + property_offset
        err = means - density
        all_errs[:, col_idx] = err[:, 0]
        col_idx += 1

    return np.mean(all_errs**2, axis=1)
