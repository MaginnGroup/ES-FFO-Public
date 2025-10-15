import numpy as np
import pandas as pd
import os
import sys
import seaborn
from numpy.linalg import norm
from functools import reduce
import pickle

# Need import from the utils folder
sys.path.append("../..")
from utils.prep_ms_data import prepare_df_props, prepare_df_errors
from fffit.fffit.utils import values_real_to_scaled, values_scaled_to_real
from fffit.fffit.pareto import find_pareto_set, is_pareto_efficient

sys.path.remove("../..")

from .models import get_prop_best_model, build_classifier


def new_samples_vle(
    all_df_data, data_dict, verbose=True, gp_shuffle_seed=42, dist_seed=1
):
    """
    Find new samples for VLE simulations

    Parameters
    ----------
    all_df_data : dict
        Dictionary of all dataframes for each molecule
    data_dict : dict
        dictionary of data objects for each molecule
    verbose : bool, optional
        Whether to print progress messages. The default is True.
    save_fig : bool, optional
        Whether to save figures. The default is False.
    gp_shuffle_seed : int, optional
        Seed for the GP model. The default is 42.
    dist_seed : int, optional
        Seed for the distance calculation. The default is 1.

    Returns
    -------
    next_iter_params_all : dict
        Dictionary of new parameters for each molecule
    """
    max_mse = 25**2  # (kg/m^3)^2
    target_total = 25
    # Loop over all molecules:
    next_iter_params_all = {}
    for mol_name, df_csv in all_df_data.items():
        data = data_dict[mol_name]
        df_csv = df_csv.dropna().copy()
        iter_num = df_csv["iter"].max()
        iter_type = "vle_iters"
        ld_threshold = 0
        root_dir = f"analysis/{mol_name}/"

        ### Step 1: Prepare df_density
        df_all, df_liquid, df_vapor = prepare_df_props(df_csv, data, ld_threshold)

        # Get LD root directory
        root_dir_vle = os.path.join(root_dir, iter_type)

        ### Fit GP Model
        path_gps = f"{root_dir_vle}/iter-{str(iter_num)}"

        models_best, models_rq, all_models, dir_train_test, best_labels = get_prop_best_model(
            df_liquid, data, path_gps, gp_shuffle_seed
        )

        ### Step 3: Find new parameters for MD simulations
        # SVM to classify hypercube regions as liquid or vapor
        LHS_file = os.path.join(root_dir, "LHS_500000.csv")
        latin_hypercube = np.genfromtxt(
            LHS_file,
            delimiter=",",
            skip_header=1,
        )[:, 1:]

        #### Find Low MSE parameter sets
        vle_samples = rank_vle_samples(latin_hypercube, models_best, data, verbose)
        result, pareto_points, dominated_points = find_pareto_set(
            vle_samples.drop(columns=list(data.param_names)).values, is_pareto_efficient
        )
        vle_samples = vle_samples.join(pd.DataFrame(result, columns=["is_pareto"]))

        pareto_points = vle_samples[vle_samples["is_pareto"] == True]

        # Get the best row for each property from pareto_points
        mse_columns = [col for col in pareto_points.columns if "mse" in col]
        best_vals = [pareto_points.sort_values(col).iloc[[0]] for col in mse_columns]
        new_points = pd.concat(best_vals)

        # Drop the selected points from pareto_points to ensure new samples are selected
        pareto_points.drop(index=new_points.index, inplace=True)
        print(f"A total of {len(pareto_points)} pareto efficient points were found.")

        target_pts = np.minimum(len(pareto_points), target_total)

        next_iter_params, final_sample_file = get_next_vle_params(
            pareto_points,
            data,
            root_dir_vle,
            iter_num,
            target_pts,
            dist_seed,
            verbose,
        )
        next_iter_params.to_csv(final_sample_file)
        next_iter_params_all[mol_name] = next_iter_params
    return next_iter_params_all


def new_samples_ld(
    all_df_data,
    data_dict,
    verbose=True,
    save_fig=False,
    cl_shuffle_seed=1,
    gp_shuffle_seed=42,
    dist_seed=1,
):
    """
    Find new samples for LD simulations

    Parameters
    ----------
    all_df_data : dict
        Dictionary of all dataframes for each molecule
    data_dict : dict
        dictionary of data objects for each molecule
    verbose : bool, optional
        Whether to print progress messages. The default is True.
    save_fig : bool, optional
        Whether to save figures. The default is False.
    cl_shuffle_seed : int, optional
        Seed for the classifier. The default is 1.
    gp_shuffle_seed : int, optional
        Seed for the GP model. The default is 42.
    dist_seed : int, optional
        Seed for the distance calculation. The default is 1.

    Returns
    -------
    next_iter_params_all : dict
        Dictionary of new parameters for each molecule
    """
    # Loop over all molecules:
    next_iter_params_all = {}
    for mol_name, df_csv in all_df_data.items():
        data = data_dict[mol_name]
        # df_csv = all_df_data[mol_name]
        iter_num = df_csv["iter"].max()
        iter_type = "ld_iters"

        ### Step 1: Prepare df_density
        df_iter1, df_liquid, root_dir = prep_df_density(mol_name, data, df_csv)

        # Get LD root directory
        root_dir_ld = os.path.join(root_dir, iter_type)

        ### Fit GP Model
        path_gps = f"{root_dir_ld}/iter-{str(iter_num)}"
        df_liquid = df_liquid.dropna()
        models_best, models_rq, all_models, dir_train_test, best_labels = get_prop_best_model(
            df_liquid, data, path_gps, gp_shuffle_seed
        )

        ### Step 3: Find new parameters for MD simulations
        # SVM to classify hypercube regions as liquid or vapor
        LHS_file = os.path.join(root_dir, "LHS_500000.csv")
        latin_hypercube = np.genfromtxt(
            LHS_file,
            delimiter=",",
            skip_header=1,
        )[:, 1:]

        # Build classifier and indentify liquid samples
        try:
            # If there are enough samples that are liquid and vapor, build the classifier
            classifier = build_classifier(
                df_iter1, root_dir_ld, data, cl_shuffle_seed, verbose, save_fig
            )
            liquid_samples, vapor_samples = classify_samples(
                latin_hypercube, classifier, verbose
            )
        except:
            # Otherwise assume all samples are liquid
            liquid_samples = latin_hypercube
            vapor_samples = None

        top_liquid_samples, top_vapor_samples = rank_vl_samples(
            liquid_samples, vapor_samples, models_rq, data, verbose
        )

        #### Find and Visualize Low MSE parameter sets
        top_liq, top_vap = vis_top_samples(
            top_liquid_samples, top_vapor_samples, data, root_dir_ld, iter_num, save_fig
        )

        #### Get next set of 200 samples or the max number of samples w/ sufficiently low RMSE
        target_total = np.minimum(200, int(len(top_liq) + len(top_vap)))
        # target_total = 200
        next_iter_params, final_sample_file = get_next_iter_params(
            top_liq,
            top_vap,
            data,
            root_dir_ld,
            iter_num,
            target_total,
            dist_seed,
            verbose,
        )
        next_iter_params.to_csv(final_sample_file)
        next_iter_params_all[mol_name] = next_iter_params
    return next_iter_params_all


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
    # if len(top_samples) <= target_num:
    #     print("Trying dist =", distance)

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

        frames = [
            df
            for df in [discarded_points, points_to_remove_df]
            if not df.empty and not df.isna().all(axis=1).all()
        ]
        discarded_points = pd.concat(frames)
        # discarded_points = pd.concat([discarded_points, points_to_remove_df])
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
        verbose: bool, whether to print the distance and error at each iteration

    Returns:
    --------
        final_distance: float, the distance between points
        final_eval: float, the squared error between the target value and number of new_points

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
        final_distance = upper_bound # Just in case lower_bound fails, use upper_bound
        final_eval = opt_dist(
            final_distance, top_samples, constants, target_num, rand_seed
        )

    return final_distance, final_eval - target_num


def prep_df_density(mol_name, data, df_csv, iter_type="ld_iters"):
    """
    Prepare the density dataframe for a given molecule.

    Parameters
    ----------
    mol_name : str
        The name of the molecule
    data : object
        The data object containing the molecule information
    iter_type : str
        The type of iteration (ld_iters or vle_iters). Default is "ld_iters".

    Returns
    -------
    df_iter1 : pd.DataFrame
        The dataframe for the first iteration
    root_dir : str
        The root directory for saving the results
    """
    # Prepare df_density
    ld_threshold = (
        min(list(data.expt_liq_density.values()))
        + max(list(data.expt_vap_density.values()))
    ) / 2

    df_csv["iter"] = df_csv["iter"].astype(int)
    df_iter1_csv = df_csv[df_csv["iter"] == 1].copy()

    df_all, df_liquid, df_vapor = prepare_df_props(df_csv, data, ld_threshold)

    df_iter1_all, df_iter1_l, df_iter1_v = prepare_df_props(
        df_iter1_csv, data, ld_threshold
    )
    root_dir = f"analysis/{mol_name}"

    # Build classifier using the current iteration if iter 1 produced only 1 type of sample
    if df_iter1_all["is_liquid"].nunique() > 1:
        df_iter1_all = df_iter1_all
    elif df_all["is_liquid"].nunique() > 1:
        df_iter1_all = df_all

    # Set df_iter1_all to have is_liquid = False for all nan values
    df_iter1_all.loc[df_iter1_all.isna().any(axis=1), "is_liquid"] = False

    return df_iter1_all, df_liquid, root_dir


def rank_vle_samples(all_samples, models, data, verbose=True):
    """
    Rank the samples based on the GP predicted MSE of the properties.

    Parameters
    ----------
    all_samples : pd.DataFrame
        The dataframe containing all the samples
    models : dict
        The dictionary containing the models for each property
    data : object
        The data object containing the molecule information
    verbose : bool, optional
        Whether to print progress messages. The default is True.

    Returns
    -------
    vle_mses : pd.DataFrame
        The dataframe containing the MSE for each sample
    """
    top_liquid_samples, top_vapor_samples = rank_vl_samples(
        all_samples, None, models, data, verbose
    )
    viable_samples = top_liquid_samples.drop(columns="mse")

    # Calculate other properties
    vle_predicted_mses = {
        prop: rank_samples(viable_samples, model, data, prop)
        for prop, model in models.items()
    }
    for prop, df in vle_predicted_mses.items():
        vle_predicted_mses[prop] = df.rename(columns={"mse": f"mse_{prop}"})

    vle_mses = reduce(
        lambda left, right: left.merge(right, on=data.param_names),
        vle_predicted_mses.values(),
    )

    return vle_mses


def rank_vl_samples(liquid_samples, vapor_samples, models, data, verbose=True):
    """
    Find the top loquid and vapor samples based on molecular simulation data and the MSE

    Parameters
    ----------
    liquid_samples : pd.DataFrame
        The dataframe containing the liquid samples
    vapor_samples : pd.DataFrame
        The dataframe containing the vapor samples
    models : dict
        The dictionary containing the models for each property
    data : object
        The data object containing the molecule information
    verbose : bool, optional
        Whether to print progress messages. The default is True.

    Returns
    -------
    top_liquid_samples : pd.DataFrame
        The dataframe containing the top liquid samples
    top_vapor_samples : pd.DataFrame
        The dataframe containing the top vapor samples
    """
    # Find the lowest MSE points from the GP in both sets
    ranked_liquid_samples = rank_samples(
        liquid_samples, models["sim_liq_density"], data, "sim_liq_density"
    )
    # Make a set of the lowest MSE parameter sets
    top_liquid_samples = ranked_liquid_samples[ranked_liquid_samples["mse"] < 625.0]

    if vapor_samples is not None:
        ranked_vapor_samples = rank_samples(
            vapor_samples, models["sim_liq_density"], data, "sim_liq_density"
        )
        top_vapor_samples = ranked_vapor_samples[ranked_vapor_samples["mse"] < 625.0]
    else:
        top_vapor_samples = None

    if verbose:
        print(
            "There are:",
            top_liquid_samples.shape[0],
            "liquid parameter sets which produce densities within 25 kg/m$^3$ of experimental densities",
        )
        if vapor_samples is not None:
            print(
                "There are:",
                top_vapor_samples.shape[0],
                " vapor parameter sets which produce densities within 25 kg/m$^3$ of experimental densities",
            )
    return top_liquid_samples, top_vapor_samples


def vis_top_samples(
    top_liquid_samples, top_vapor_samples, data, root_dir, iter_num, save_fig=False
):
    """
    Visualize the top samples and save them to a CSV file.

    Parameters
    ----------
    top_liquid_samples : pd.DataFrame
        The dataframe containing the top liquid samples
    top_vapor_samples : pd.DataFrame
        The dataframe containing the top vapor samples
    data : object
        The data object containing the molecule information
    root_dir : str
        The root directory for saving the results
    iter_num : int
        The current iteration number
    save_fig : bool, optional
        Whether to save the figures. The default is False.

    Returns
    -------
    top_liq : pd.DataFrame
        The dataframe containing the top liquid samples
    top_vap : pd.DataFrame
        The dataframe containing the top vapor samples
    """
    column_names = list(data.param_names)
    objects = {}

    dir_name = f"{root_dir}/iter-{str(iter_num)}/"
    os.makedirs(dir_name, exist_ok=True)

    for i, sample in enumerate([top_liquid_samples, top_vapor_samples]):
        if i == 0:
            phase = "liq"
        else:
            phase = "vap"
        if sample is not None:
            g = seaborn.pairplot(sample.drop(columns=["mse"]))
            g.set(xlim=(-0.1, 1.1), ylim=(-0.1, 1.1))

            if save_fig:
                g.savefig(dir_name + f"{phase}_mse_below625.pdf")

            # Drop the mse column for
            new_sample_params = [sample.drop(columns=["mse"])]
            # Concatenate into a single dataframe and save to CSV
            new_sample_params = pd.concat(new_sample_params)

            samp_path = os.path.join(dir_name, f"{phase}-params.csv")
            new_sample_params.to_csv(samp_path)
            top_samp = new_sample_params.reset_index(drop=True)
            objects[phase] = top_samp
        else:
            # If there are no samples, create an empty DataFrame with the correct columns
            objects[phase] = None
    top_liq = objects["liq"]
    top_vap = objects["vap"]

    return top_liq, top_vap


def get_next_iter_params(
    top_liq,
    top_vap,
    data,
    root_dir,
    iter_num,
    target_total=200,
    dist_seed=1,
    verbose=True,
):
    """
    Get the next set of parameters for Liquid density iterations.
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
    final_sample_file = root_dir + "/params-iter-" + str(int(iter_num) + 1) + ".csv"
    # We want to have as many liquid points as possible, but no more than 200 total and the rest vapor
    target_num_l = np.minimum(target_total, len(top_liq))
    target_num_v = target_total - target_num_l
    if verbose:
        print("target liq: ", target_num_l, "target vap: ", target_num_v)

    zero_array = np.zeros(top_liq.shape[1]-1)
    one_array = np.ones(top_liq.shape[1]-1)
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


def get_next_vle_params(
    top_liq, data, root_dir, iter_num, target_total=25, dist_seed=1, verbose=True
):
    """
    Get the next set of parameters for VLE iterations.

    Parameters
    ----------
    top_liq : pd.DataFrame
        The dataframe for the top liquid samples
    data : object
        The data object containing the molecule information
    root_dir : str
        The root directory for saving the results
    iter_num : int
        The current iteration number
    target_total : int, default 25
        The target number of samples to return
    dist_seed : int, default 1
        The seed for the random number generator
    verbose : bool, default True
        Whether to print the sample distance

    Returns
    -------
    new_points_l : pd.DataFrame
        The dataframe for the next iteration parameters
    final_sample_file : str
        The path to the CSV file for the next iteration parameters
    """
    final_sample_file = root_dir + "/params-iter-" + str(iter_num + 1) + ".csv"
    # We want to have as many liquid points as possible, but no more than 200 total and the rest vapor
    target_num_l = target_total
    zero_array = np.zeros(top_liq.shape[1])
    one_array = np.ones(top_liq.shape[1])
    ub_array = one_array - zero_array

    # lower_bound = 1e-8
    lower_bound = 0
    # IL norm between the highest high parameter space, and lowest low parameter space value
    upper_bound = norm(ub_array, 1)  # This number will be 10, the number of dimensions
    error_tol = 1e-8

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

    props = ["sim_liq_density", "sim_vap_density", "sim_Pvap", "sim_Hvap", "sim_surf_tens"]
    col_del = ["is_pareto"]
    for prop in props:
        if "mse_" + prop in new_points_l.columns:
            col_del.append("mse_" + prop)
    new_points_l.drop(columns=col_del, inplace=True)

    return new_points_l, final_sample_file


def classify_samples(samples, classifier, verbose=True):
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

    if verbose:
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
        "sim_Hvap",
    ]

    if property_name not in valid_property_names:
        raise ValueError(
            "Invalid property_name {}. Supported property_names are "
            "{}".format(property_name, valid_property_names)
        )
    print("Include properties:", property_name)

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
    elif property_name == "sim_Hvap":
        expt_property = molecule.expt_Hvap
        property_bounds = molecule.Hvap_bounds
    print("Property bounds are", property_bounds)
    temperature_bounds = molecule.temperature_bounds(
        property_name.replace("sim", "expt")
    )
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
    # print("MSE is", mse)
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
    # print("Initialize all errors!")
    col_idx = 0
    for temp, density in expt_property.items():
        # print("Trying temp", temp, "and density", density)
        scaled_temp = values_real_to_scaled(temp, temperature_bounds)
        xx = np.hstack((samples, np.tile(scaled_temp, (samples.shape[0], 1))))
        means_scaled, vars_scaled = gp_model.predict_f(xx)
        means = values_scaled_to_real(means_scaled, property_bounds)
        means = means + property_offset
        err = means - density
        all_errs[:, col_idx] = err[:, 0]
        col_idx += 1

    return np.mean(all_errs**2, axis=1)


def check_mse_10(df_all_molec, data_dict, target_total=25, dist_seed=1, save_csv=True):
    """
    Check the MSE of the samples and return the samples with MSE < 10 kg/m^3

    Parameters
    ----------
    df_all_molec : dict
        Dictionary of all dataframes for each molecule
    data_dict : dict
        Dictionary of data objects for each molecule
    target_total : int, default 25
        The target number of samples to return
    dist_seed : int, default 1
        The seed for the random number generator
    save_csv : bool, default True
        Whether to save the results to a CSV file

    Returns
    -------
    num_mse_less10 : dict
        Dictionary of samples with MSE < 10 kg/m^3 for each molecule
    """
    num_mse_less10 = {}

    for mol_name, df_csv in df_all_molec.items():
        root_dir = f"analysis/{mol_name}/"
        iter_type = "ld_iters"
        molecule = data_dict[mol_name]
        # Pull the results from all iterations + calculate the MSE
        df_results = df_csv.dropna().copy()
        df_results["expt_liq_density"] = df_results["temperature"].apply(
            lambda x: molecule.expt_liq_density[x]
        )
        df_results["sq_err"] = (
            df_results["liq_density"] - df_results["expt_liq_density"]
        ) ** 2
        # df_mse2 = (df_results.groupby(list(molecule.param_names))["sq_err"].mean().reset_index(name="mse"))
        df_results["abs_pct_err"] = (
            (df_results["liq_density"] - df_results["expt_liq_density"]).abs()
            / df_results["expt_liq_density"]
        ) * 100
        df_results["pct_err"] = (
            (df_results["liq_density"] - df_results["expt_liq_density"])
            / df_results["expt_liq_density"]
        ) * 100

        df_mse = (
            df_results.groupby(list(molecule.param_names))
            .agg(mse=("sq_err", "mean"), mapd=("abs_pct_err", "mean"), mpd=("pct_err", "mean"))
            .reset_index()
        )

        # Scale the parameter values for the results to compare to the original parameter values
        scaled_param_values = values_real_to_scaled(
            df_mse[list(molecule.param_names)], molecule.param_bounds
        )
        # Pull all parameter values we have tested
        df_params = []
        for i in range(1, int(df_csv["iter"].max()) + 1):
            param_name_file = os.path.join(
                root_dir, iter_type, "params-iter-" + str(i) + ".csv"
            )
            df_params_sets = pd.read_csv(param_name_file, index_col=0)
            # Unscale the parameter values to the real values
            lhs_real = values_scaled_to_real(
                df_params_sets[list(molecule.param_names)], molecule.param_bounds
            )
            # For all rows if the value of a column is less than 1e-14 set it to 0
            lhs_real[np.abs(lhs_real) < 1e-14] = 0.0
            # Rescale the values to the scaled values so that scaled values will also be zero
            lhs_scl = values_real_to_scaled(lhs_real, molecule.param_bounds)
            lhs_scl_pd = pd.DataFrame(lhs_scl, columns=list(molecule.param_names))
            df_params.append(lhs_scl_pd)

        df_params = pd.concat(df_params).reset_index(drop=True)
        df_results = df_results.reset_index(drop=True)

        # Match indeces of the scaled values to the original parameter values
        param_idxs = []
        param_vals = []
        # Get the parameter values for the scaled values
        for params1 in scaled_param_values:
            for idx, params2 in enumerate(df_params[list(molecule.param_names)].values):
                if np.allclose(params1, params2, rtol=1e-8):
                    param_idxs.append(idx)
                    param_vals.append(params2)
                    break
        df_mse["param_idx"] = param_idxs
        df_mse[list(molecule.param_names)] = param_vals

        top_param_set = df_mse[df_mse["mse"] < 100]
        from numpy.linalg import norm

        target_num_l = target_total
        zero_array = np.zeros(top_param_set.shape[1]-1)
        one_array = np.ones(top_param_set.shape[1]-1)
        ub_array = one_array - zero_array

        lower_bound = 0
        # IL norm between the highest high parameter space, and lowest low parameter space value
        upper_bound = norm(
            ub_array, 1
        )  # This number will be 10, the number of dimensions
        error_tol = 1e-8

        new_points_vle = top_param_set

        # If we have enough liquid samples, we want to find the distance that will give us the target number of liquid samples
        if len(top_param_set) >= target_num_l:
            distance_opt_l, number_points_l = bisection(
                lower_bound,
                upper_bound,
                error_tol,
                top_param_set,
                molecule,
                target_num_l,
                dist_seed,
            )
            print(
                "\nRequired Distance for liquid is : %0.8f and there are %0.1f points too many"
                % (distance_opt_l, number_points_l)
            )
            new_points_vle = opt_dist(
                distance_opt_l,
                top_param_set,
                molecule,
                target_num_l,
                rand_seed=dist_seed,
                eval=True,
            )
            print(
                len(new_points_vle),
                "top liquid density points are left after removing similar points using a distance of",
                np.round(distance_opt_l, 5),
            )
            new_dir = os.path.join(root_dir, "vle_iters")
            os.makedirs(new_dir, exist_ok=True)
            out_csv = os.path.join(root_dir, "vle_iters", "params-iter-1.csv")
        else:
            # Print total number of top liquid samples
            print("Total number of top liquid samples is", len(top_param_set))
            print(top_param_set)
            # Add this analysis to the dens-iters folder
            out_csv = os.path.join(root_dir, iter_type, "mse-less10.csv")

        num_mse_less10[mol_name] = new_points_vle
        print(
            f"Number of {mol_name} samples with MSE < 10 kg/m^3 is", len(new_points_vle)
        )
        if save_csv:
            new_pts_copy = new_points_vle.copy()
            new_pts_copy.drop(columns=["mse", "mapd", "mpd", "param_idx"], inplace=True)
            new_pts_copy.to_csv(out_csv)

            # Save a copy of the real param sets too
            new_pts_real = values_scaled_to_real(new_pts_copy, molecule.param_bounds)
            new_pts_real = pd.DataFrame(
                new_pts_real, columns=list(molecule.param_names)
            )
            new_pts_real.to_csv(
                os.path.join(root_dir, iter_type, "mse-less10-real.csv")
            )

            # Save the full set of points with MSE < 10 w/ mse column
            new_pts_full = new_pts_real.copy()
            new_pts_full["mse"] = new_points_vle["mse"].values
            new_pts_full["mapd"] = new_points_vle["mapd"].values
            new_pts_full["mpd"] = new_points_vle["mpd"].values
            new_pts_full.to_csv(
                os.path.join(root_dir, iter_type, "mse-less10-full.csv")
            )
    return num_mse_less10


def find_pareto(all_df_data, data_dict, props_pareto, threshold = 10.0):
    """
    Find the pareto points for all molecules and save them to a CSV file.

    Parameters
    ----------
    all_df_data : dict
        Dictionary of all dataframes for each molecule
    data_dict : dict
        Dictionary of data objects for each molecule

    Returns
    -------
    all_final_params : dict
        Dictionary of final parameters for each molecule
    """
    # Loop over all molecules:
    all_final_params = {}
    for mol_name, df_csv in all_df_data.items():
        root_dir = f"analysis/{mol_name}/"
        root_dir_vle = os.path.join(root_dir, "vle_iters")
        # Get all data from last iteration
        df_csv = df_csv.dropna().copy()
        iter_num = df_csv["iter"].max()
        ld_threshold = 0
        data = data_dict[mol_name]

        # Get all result data
        df_all, df_liquid, df_vapor = prepare_df_props(
            df_csv, data, ld_threshold, scale=False
        )
        # Prepare error data to find pareto points
        df_paramsets = prepare_df_errors(df_all, data_dict, mol_name)
        # Save data to csv
        dir_name = root_dir_vle + "/iter-" + str(iter_num) + "/"
        os.makedirs(dir_name, exist_ok=True)
        csv_name = os.path.join(dir_name, "result_errors.csv")
        df_paramsets.to_csv(csv_name)

        mse_columns = [col for col in df_paramsets.columns if "mse" in col]
        result, pareto_points, dominated_points = find_pareto_set(
            df_paramsets.filter(mse_columns).values, is_pareto_efficient
        )

        df_paramsets = df_paramsets.join(pd.DataFrame(result, columns=["is_pareto"]))
        pareto_points = df_paramsets[df_paramsets["is_pareto"] == True]
        print(f"A total of {len(pareto_points)} pareto efficient points were found.")

        # Create the directory if it doesn't exist and store pareto points
        dir_name = os.path.join(root_dir_vle, "iter-" + str(iter_num))
        os.makedirs(dir_name, exist_ok=True)
        file_name = os.path.join(dir_name, "pareto-params.csv")
        pareto_points.to_csv(file_name)

        df_final = select_final_pareto(
            pareto_points, root_dir_vle, iter_num, props_pareto, threshold
        )
    all_final_params[mol_name] = df_final

    # Save all final parameters to a pkl file
    if len(list(all_df_data.keys())) > 1:
        names = "-".join(sorted(all_df_data.keys()))
        dir2 = f"analysis/{names}/"
        os.makedirs(dir2, exist_ok=True)
        with open(dir2 + "/final_params.pkl", "wb") as f:
            pickle.dump(all_final_params, f)

    return all_final_params


def select_final_pareto(df_pareto, root_dir, iter_num, props_pareto, threshold = 5.0):
    """
    Filter the pareto points and select the final parameters given the lowest error parameter set.

    Parameters
    ----------
    df_pareto : pd.DataFrame
        The dataframe containing the pareto points
    root_dir : str
        The root directory for saving the results
    iter_num : int
        The current iteration number

    Returns
    -------
    df_final : pd.DataFrame
        The dataframe containing the final parameters
    """
    # Filter for parameter sets with less than threshold (5) % error in all properties
    properties = [
        "liq_density",
        "vap_density",
        "Pvap",
        "Hvap",
        "surf_tens",
        "Tc",
        "rhoc",
    ]

    cols_to_drop = ["is_pareto"]
    for prop in properties:
        if "mse_" + prop in df_pareto.columns:
            # Remove the mpd, mse and mae columns
            cols_to_drop += ["mse_" + prop, "mae_" + prop, "mpd_" + prop]

    df_final = df_pareto.drop(columns=cols_to_drop)

    props_mse = ["mapd_" + prop for prop in props_pareto]
    df_final = df_final[df_final[props_mse].le(threshold).all(axis=1)]
    # Save CSV files
    dir_name = root_dir + "/iter-" + str(iter_num) + "/"
    os.makedirs(dir_name, exist_ok=True)
    csv_name = os.path.join(dir_name, "final-params.csv")
    df_final.to_csv(csv_name)

    # If more than one point is found, select the one with the lowest error
    if len(df_final) > 1:
        # Find the point with the lowest average error
        df_final["avg_MAPD"] = df_final[props_mse].mean(axis=1)
        # Sort the dataframe by the average MAPD
        df_final = df_final.sort_values("avg_MAPD")
        # Select the first row
        df_final = df_final.iloc[[0]]
        # Drop the avg_MAPD column
        df_final = df_final.drop(columns=["avg_MAPD"])

    return df_final
