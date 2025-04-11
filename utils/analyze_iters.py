import signac
import sys
import pandas as pd
import numpy as np
import os
import gpflow
import matplotlib.pyplot as plt
import seaborn
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import svm
from fffit.utils import (
    shuffle_and_split,
)
from fffit.models import run_gpflow_scipy
from utils.id_new_samples import (
    prepare_df_density,
    classify_samples,
    rank_samples,
)
from utils.molec_class_files import r41

sys.path.append("../")

def save_signac_results(projects, data_dict, prop_names, save_csv=True):
    """Save the signac results to a CSV file.

    Parameters
    ----------
    projects : list of signac.Project
        signac projects to load
    data_dict : dictionary
        dictionary of molecule names and data from esolvs.py
    prop_names : set
        set of property names
    save_csv : bool, default True
        Whether to save the results to a CSV file
    """
    if type(param_names) not in (list, tuple):
        raise TypeError("param_names must be a list or tuple")
    if type(property_names) not in (list, tuple):
        raise TypeError("property_names must be a list or tuple")

    # Group by project_name and molecules
    job_groupby = tuple(("mol_name", "dens-iter"))
    property_names = tuple(property_names)
    
    print(f"Extracting the following properties: {property_names}")  

    all_data_dict = {}

    # Loop over all jobs in project and group by mol name and density iter
    for (mol_name, dens_iter), job_group in project.groupby(job_groupby):
        data = [] # Store data here before converting to dataframe
        # Get the unique param sets for each molecule
        param_names = data_dict[mol_name].param_names
        #Loop over each parameter set in the group
        for param_vals, job_group_params in job_group.groupby(param_names):
            #Loop over all jobs (temperatures) in the group
            for job in job_group_params:
                # Extract the parameters into a dict
                new_row = {
                    name: param for (name, param) in zip(param_names, param_vals).
                }

                # Extract the temperature for each job.
                # Assumes temperature increments >= 1 K
                temperature = round(job.sp.T)
                new_row["temperature"] = temperature

                # Extract property values. Insert N/A if not found
                for property_name in property_names:
                    try:
                        property_ = job.doc[property_name]
                        new_row[property_name] = property_
                    except KeyError:
                        print(f"Job failed: {job.id}")
                        new_row[property_name] = np.nan
                
                data.append(new_row)

        #Create data from dict
        df = pd.DataFrame(data)

        #Add data to all_data_dict
        # If the molecule name is already in the dictionary, concatenate the dataframes
        if mol_name in all_data_dict:
            all_data_dict[mol_name] = pd.concat([all_data_dict[mol_name], df])
        # If the molecule name is not in the dictionary, add the dataframe
        else:
            all_data_dict[mol_name] = df

        # Save to csv file for record-keeping
        if save_csv:
            csv_name = "density_iters/analysis/" +  mol_name  + "/results-iter-" + str(dens_iter) + ".csv"
            df.to_csv(csv_name)
    
    # Save all data to a single CSV file
    for mol_name, data in all_data_dict.items():
        if save_csv:
            # Save each molecule data to a separate CSV file
            csv_name = "density_iters/analysis/" + mol_name + "/all_results.csv"
            data.to_csv(csv_name)

    return all_data_dict


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


# STOPPED HERE
def id_new_samples(all_df_data, mol_data_dict, verbose = True, save_fig=False):
    #Loop over all molecules:
    for mol_name, data in all_df_data.items():
        #Prepare df_density
        ld_threshold = mol_data_dict[mol_name].expt_rhoc
        df_csv = all_df_data[mol_name]
        df_all, df_liquid, df_vapor = prepare_df_density(
            df_csv, data, liquid_density_threshold
        )

        ### Step 2: Fit classifier and GP models
        # Create training/test set
        param_names = list(data.param_names) + ["temperature"]
        property_name = "is_liquid"
        x_train, y_train, x_test, y_test = shuffle_and_split(
            df_all, param_names, property_name, shuffle_seed
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
            plt.savefig("classifier.pdf")

        ### Fit GP Model
        # Create training/test set
        param_names = list(data.param_names) + ["temperature"]
        models = {}

        for property_name in ["md_density", "md_surf_tens"]:
            x_train, y_train, x_test, y_test = shuffle_and_split(
            df_liquid, param_names, property_name, shuffle_seed=gp_shuffle_seed
        )
            models[prop] = run_gpflow_scipy(
                x_train,
                y_train,
                gpflow.kernels.RBF(lengthscales=np.ones(data.n_params + 1)),
            )

        ### Step 3: Find new parameters for MD simulations

        # SVM to classify hypercube regions as liquid or vapor
        latin_hypercube = np.genfromtxt(
            "../../LHS_500000_x_6.csv",
            delimiter=",",
            skip_header=1,
        )[:, 1:]
        liquid_samples, vapor_samples = classify_samples(latin_hypercube, classifier)

        # Find the lowest MSE points from the GP in both sets
        ranked_liquid_samples = rank_samples(liquid_samples, model, R41, "sim_liq_density")
        ranked_vapor_samples = rank_samples(
            vapor_samples, model, R41, "sim_liq_density"
        )  # both l and g compared to liquid density

        # Make a set of the lowest MSE parameter sets
        top_liquid_samples = ranked_liquid_samples[ranked_liquid_samples["mse"] < 625.0]
        top_vapor_samples = ranked_vapor_samples[ranked_vapor_samples["mse"] < 625.0]

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

    #### Visualization: Low MSE parameter sets
    # Create a pairplot of the top "liquid" parameter values
    column_names = list(R41.param_names)
    g = seaborn.pairplot(top_liquid_samples.drop(columns=["mse"]))
    g.set(xlim=(-0.1, 1.1), ylim=(-0.1, 1.1))
    if save_fig:
        g.savefig("liq_mse_below625.pdf")

    # Create a pairplot of the top "vapor" parameter values
    column_names = list(R41.param_names)
    g = seaborn.pairplot(top_vapor_samples.drop(columns=["mse"]))
    g.set(xlim=(-0.1, 1.1), ylim=(-0.1, 1.1))
    if save_fig:
        g.savefig("vap_mse_below625.pdf")

    new_liquid_params = [top_liquid_samples.drop(columns=["mse"])]
    new_vapor_params = [top_vapor_samples.drop(columns=["mse"])]

    # Concatenate into a single dataframe and save to CSV
    new_liquid_params = pd.concat(new_liquid_params)
    new_vapor_params = pd.concat(new_vapor_params)
    if save_fig:
        new_liquid_params.to_csv(csv_path + out_top_liquid_csv_name)
        new_vapor_params.to_csv(csv_path + out_top_vapor_csv_name)
    top_liq = pd.read_csv(
        csv_path + out_top_liquid_csv_name, delimiter=",", index_col=0
    )
    top_vap = pd.read_csv(csv_path + out_top_vapor_csv_name, delimiter=",", index_col=0)

    top_liq = top_liq.reset_index(drop=True)
    top_vap = top_vap.reset_index(drop=True)

    from numpy.linalg import norm

    target_total = 200
    # We want to have as many liquid points as possible, but no more than 200 total and the rest vapor
    target_num_l = np.minimum(200, len(top_liq))
    target_num_v = target_total - target_num_l
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
            lower_bound, upper_bound, error_tol, top_liq, R41, target_num_l, dist_seed
        )
        print(
            "\nRequired Distance for liquid is : %0.8f and there are %0.1f points too many"
            % (distance_opt_l, number_points_l)
        )
        new_points_l = opt_dist(
            distance_opt_l, top_liq, R41, target_num_l, rand_seed=dist_seed, eval=True
        )
        print(
            len(new_points_l),
            "top liquid density points are left after removing similar points using a distance of",
            np.round(distance_opt_l, 5),
        )
        if save_fig:
            new_points_l.to_csv(csv_path + out_csv_name)
    # If we don't we want to find the vapor sets to add
    else:
        distance_opt_v, number_points_v = bisection(
            lower_bound, upper_bound, error_tol, top_vap, R41, target_num_v, dist_seed
        )
        print(
            "\nRequired Distance for vapor is : %0.8f and there are %0.1f points too many"
            % (distance_opt_v, number_points_v)
        )
        new_points_v = opt_dist(
            distance_opt_v, top_vap, R41, target_num_v, rand_seed=dist_seed, eval=True
        )
        print(
            len(new_points_v),
            "top vapor density points are left after removing similar points using a distance of",
            np.round(distance_opt_v, 5),
        )
        if save_fig:
            pd.concat([top_liq, new_points_v], axis=0).to_csv(csv_path + out_csv_name)

# def id_new_samples():
#     liquid_density_threshold = (
#         400  # kg/m^3  ##>500 is liquid; <500 is gas. used for classifier
#     )

#     csv_path = "../csv/"
#     in_csv_names = [
#         "r41-density-iter" + str(i) + "-results.csv" for i in range(1, iternum + 1)
#     ]
#     out_csv_name = "r41-density-iter" + str(iternum + 1) + "-params.csv"
#     out_top_liquid_csv_name = "r41-density-iter" + str(iternum) + "-liquid-params.csv"
#     out_top_vapor_csv_name = "r41-density-iter" + str(iternum) + "-vapor-params.csv"

#     # Read file
#     df_csvs = [
#         pd.read_csv(csv_path + in_csv_name, index_col=0) for in_csv_name in in_csv_names
#     ]
#     df_csv = pd.concat(df_csvs)
#     df_all, df_liquid, df_vapor = prepare_df_density(
#         df_csv, R41, liquid_density_threshold
#     )

#     ### Step 2: Fit classifier and GP models

#     # Create training/test set
#     param_names = list(R41.param_names) + ["temperature"]
#     property_name = "is_liquid"
#     x_train, y_train, x_test, y_test = shuffle_and_split(
#         df_all, param_names, property_name, shuffle_seed=cl_shuffle_seed
#     )

#     # Create and fit classifier
#     # class_weight "balanced" used because there are fewer liquid than vapor samples in the LHS sets
#     classifier = svm.SVC(kernel="rbf", class_weight="balanced")
#     classifier.fit(x_train, y_train)
#     test_score = classifier.score(x_test, y_test)
#     print(f"Classifer is {test_score*100.0}% accurate on the test set.")
#     ConfusionMatrixDisplay.from_estimator(
#         classifier, x_test, y_test, display_labels=["Vapor", "Liquid"]
#     )
#     if save_fig:
#         plt.savefig("classifier.pdf")

#     ### Fit GP Model
#     # Create training/test set
#     param_names = list(R41.param_names) + ["temperature"]
#     property_name = "md_density"
#     x_train, y_train, x_test, y_test = shuffle_and_split(
#         df_liquid, param_names, property_name, shuffle_seed=gp_shuffle_seed
#     )

#     # Fit model
#     model = run_gpflow_scipy(
#         x_train,
#         y_train,
#         gpflow.kernels.RBF(lengthscales=np.ones(R41.n_params + 1)),
#     )

#     ### Step 3: Find new parameters for MD simulations

#     # SVM to classify hypercube regions as liquid or vapor
#     latin_hypercube = np.genfromtxt(
#         "../../LHS_500000_x_6.csv",
#         delimiter=",",
#         skip_header=1,
#     )[:, 1:]
#     liquid_samples, vapor_samples = classify_samples(latin_hypercube, classifier)

#     # Find the lowest MSE points from the GP in both sets
#     ranked_liquid_samples = rank_samples(liquid_samples, model, R41, "sim_liq_density")
#     ranked_vapor_samples = rank_samples(
#         vapor_samples, model, R41, "sim_liq_density"
#     )  # both l and g compared to liquid density

#     # Make a set of the lowest MSE parameter sets
#     top_liquid_samples = ranked_liquid_samples[ranked_liquid_samples["mse"] < 625.0]
#     top_vapor_samples = ranked_vapor_samples[ranked_vapor_samples["mse"] < 625.0]

#     print(
#         "There are:",
#         top_liquid_samples.shape[0],
#         "liquid parameter sets which produce densities within 25 kg/m$^3$ of experimental densities",
#     )
#     print(
#         "There are:",
#         top_vapor_samples.shape[0],
#         " vapor parameter sets which produce densities within 25 kg/m$^3$ of experimental densities",
#     )

#     #### Visualization: Low MSE parameter sets
#     # Create a pairplot of the top "liquid" parameter values
#     column_names = list(R41.param_names)
#     g = seaborn.pairplot(top_liquid_samples.drop(columns=["mse"]))
#     g.set(xlim=(-0.1, 1.1), ylim=(-0.1, 1.1))
#     if save_fig:
#         g.savefig("liq_mse_below625.pdf")

#     # Create a pairplot of the top "vapor" parameter values
#     column_names = list(R41.param_names)
#     g = seaborn.pairplot(top_vapor_samples.drop(columns=["mse"]))
#     g.set(xlim=(-0.1, 1.1), ylim=(-0.1, 1.1))
#     if save_fig:
#         g.savefig("vap_mse_below625.pdf")

#     new_liquid_params = [top_liquid_samples.drop(columns=["mse"])]
#     new_vapor_params = [top_vapor_samples.drop(columns=["mse"])]

#     # Concatenate into a single dataframe and save to CSV
#     new_liquid_params = pd.concat(new_liquid_params)
#     new_vapor_params = pd.concat(new_vapor_params)
#     if save_fig:
#         new_liquid_params.to_csv(csv_path + out_top_liquid_csv_name)
#         new_vapor_params.to_csv(csv_path + out_top_vapor_csv_name)
#     top_liq = pd.read_csv(
#         csv_path + out_top_liquid_csv_name, delimiter=",", index_col=0
#     )
#     top_vap = pd.read_csv(csv_path + out_top_vapor_csv_name, delimiter=",", index_col=0)

#     top_liq = top_liq.reset_index(drop=True)
#     top_vap = top_vap.reset_index(drop=True)

#     from numpy.linalg import norm

#     target_total = 200
#     # We want to have as many liquid points as possible, but no more than 200 total and the rest vapor
#     target_num_l = np.minimum(200, len(top_liq))
#     target_num_v = target_total - target_num_l
#     print(target_num_l, target_num_v)

#     zero_array = np.zeros(top_liq.shape[1])
#     one_array = np.ones(top_liq.shape[1])
#     ub_array = one_array - zero_array

#     # lower_bound = 1e-8
#     lower_bound = 0
#     # IL norm between the highest high parameter space, and lowest low parameter space value
#     upper_bound = norm(ub_array, 1)  # This number will be 10, the number of dimensions
#     error_tol = 1e-8

#     # If we have enough liquid samples, we want to find the distance that will give us the target number of liquid samples
#     if len(top_liq) >= target_total:
#         distance_opt_l, number_points_l = bisection(
#             lower_bound, upper_bound, error_tol, top_liq, R41, target_num_l, dist_seed
#         )
#         print(
#             "\nRequired Distance for liquid is : %0.8f and there are %0.1f points too many"
#             % (distance_opt_l, number_points_l)
#         )
#         new_points_l = opt_dist(
#             distance_opt_l, top_liq, R41, target_num_l, rand_seed=dist_seed, eval=True
#         )
#         print(
#             len(new_points_l),
#             "top liquid density points are left after removing similar points using a distance of",
#             np.round(distance_opt_l, 5),
#         )
#         if save_fig:
#             new_points_l.to_csv(csv_path + out_csv_name)
#     # If we don't we want to find the vapor sets to add
#     else:
#         distance_opt_v, number_points_v = bisection(
#             lower_bound, upper_bound, error_tol, top_vap, R41, target_num_v, dist_seed
#         )
#         print(
#             "\nRequired Distance for vapor is : %0.8f and there are %0.1f points too many"
#             % (distance_opt_v, number_points_v)
#         )
#         new_points_v = opt_dist(
#             distance_opt_v, top_vap, R41, target_num_v, rand_seed=dist_seed, eval=True
#         )
#         print(
#             len(new_points_v),
#             "top vapor density points are left after removing similar points using a distance of",
#             np.round(distance_opt_v, 5),
#         )
#         if save_fig:
#             pd.concat([top_liq, new_points_v], axis=0).to_csv(csv_path + out_csv_name)
