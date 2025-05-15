import numpy as np
import scipy.optimize as optimize
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import MinMaxScaler
import warnings
import scipy
import os
import time
import pandas as pd
import pickle
import gpflow
import json

from fffit.fffit.utils import (
    values_real_to_scaled,
    values_scaled_to_real,
    variances_scaled_to_real,
    generate_lhs,
)
from fffit.fffit.plot import (
    plot_model_performance,
    plot_model_vs_test,
    plot_slices_temperature,
    plot_slices_params,
    plot_model_vs_exp,
    plot_obj_contour,
)
from fffit.fffit.pareto import find_pareto_set, is_pareto_efficient

import unyt as u
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import tensorflow as tf
from itertools import combinations
import numdifftools as nd
from sklearn.metrics import mean_absolute_percentage_error
from .molec_class_files import (
    r14,
    r32,
    r50,
    r125,
    r134a,
    r143a,
    r170,
    r41,
    r23,
    r161,
    r152a,
    r152,
    r134,
    r143,
    r116,
)
from .atom_type import make_atom_type_class
from pathlib import Path

mpl_is_inline = "inline" in matplotlib.get_backend()
# print(mpl_is_inline)


# Create a function for getting gp data from files
def get_gp_data_from_pkl(key_list):
    """
    Get gp data from .pkl files

    Parameters
    ----------
    key_list: list of keys to consider. Must be valid Keys: "R14", "R32", "R50", "R125", "R143a", "R134a", "R170"

    Returns:
    --------
    all_gp_dict: dict, dictionary of dictionary of training molecule gps for each property
    """
    valid_keys = ["R14", "R32", "R50", "R125", "R143a", "R134a", "R170", "R41"]
    assert isinstance(key_list, list), "at_names must be a list"
    assert (
        all(isinstance(name, str) for name in key_list) == True
    ), "all key in key_list must be string"
    assert (
        all(key in valid_keys for key in key_list) == True
    ), "all key in key_list must be valid keys"
    # Make a dict of the gp dictionaries for each molecule
    all_gp_dict = {}
    # Ensure we are working out of generalizedFF
    if Path(os.getcwd()).parent.name == "generalizedFF":
        path_use = Path(Path(os.getcwd()).parent)
    else:
        path_use = Path(os.getcwd())
    # loop over molecules
    for key in key_list:
        # Get dict of vle gps
        # OPTIONAL append the MD density gp to the VLE density gp dictionary w/ key "MD Density"
        file = path_use / "molec_gp_data" / (key + "-vlegp/vle-gps.pkl")
        assert (
            file.exists()
        ), f"{os.path.abspath(file)} does not exist. Check file path carefully."
        with open(file, "rb") as pickle_file:
            all_gp_dict[key] = pickle.load(pickle_file)

    return all_gp_dict


class Problem_Setup:
    """
    Gets GP/ Experimental Data For experiments

    Methods:
    --------
    __init__: Initializes the class
    make_results_dir: Makes a directory for results based on the scheme name, optimization method, and molecule names
    values_pref_to_real: Scales preferred units (Angstrom and Kelvin) to real units (nm, kJ/mol)
    values_real_to_pref: Scales real units (nm, kJ/mol) to preferred units (Angstrom and Kelvin)
    get_exp_data: Helper function for getting experimental data and bounds
    get_train_test_data: Get training and testing data from csv files
    eval_gp_new_theta: Evaluates the gpflow model
    calc_wt_res: Calculates the residuals for the objective function
    calc_obj: Calculates the objective function
    one_output_calc_obj: Helper function. Calls calc_obj and returns only the objective function value
    approx_jac: Builds Jacobian Approximation
    approx_hess: Builds Hessian Approximation
    get_best_results: Get the best optimization results. Pulls best parameter set from:
        1) Literature (stored in molecule objects)
        2) The algorithm trained for one training molecule (molec_ind)
        3) The algorithm trained for all training molecules (molec_data_dict.values())
    calc_MAPD_best: Calculate the mean absolute percentage deviation for each training data prediction
    """

    # Inherit objects from General_Analysis
    def __init__(self, molec_names, at_number, obj_choice, weight_sclr=None):
        """
        Parameters:
        -----------
        molec_data_dict: List of refrigerant names w/ capital R, values are class objects from r***.py
        at_class: at_number, int with which to create class for atom typing
        obj_choice: str, the objective choice. "SSE" (SSE) or "ExpVal" (Expected Value of SSE) or "ExpValPrior" (Expected Value of SSE with GAFF Prior)
        """
        # Load class properies for each molecule
        r14_class = r14.R14Constants()
        r32_class = r32.R32Constants()
        r50_class = r50.R50Constants()
        r125_class = r125.R125Constants()
        r134a_class = r134a.R134aConstants()
        r143a_class = r143a.R143aConstants()
        r170_class = r170.R170Constants()
        r41_class = r41.R41Constants()

        r23_class = r23.R23Constants()
        r161_class = r161.R161Constants()
        r152a_class = r152a.R152aConstants()
        r152_class = r152.R152Constants()
        r143_class = r143.R143Constants()
        r134_class = r134.R134Constants()
        r116_class = r116.R116Constants()
        # Set a dictionary of all molecule data
        self.all_train_molec_data = {
            "R14": r14_class,
            "R32": r32_class,
            "R50": r50_class,
            "R170": r170_class,
            "R125": r125_class,
            "R134a": r134a_class,
            "R143a": r143a_class,
            "R41": r41_class,
        }

        self.all_test_molec_data = {
            "R23": r23_class,
            "R161": r161_class,
            "R152a": r152a_class,
            "R152": r152_class,
            "R134": r134_class,
            "R143": r143_class,
            "R116": r116_class,
        }

        try:
            self.all_train_gp_dict = get_gp_data_from_pkl(
                list(self.all_train_molec_data.keys())
            )
        except:
            warnings.warn(
                "No gp data found. Many functions will not work without GP Data"
            )
        self.valid_mol_keys = [
            "R14",
            "R32",
            "R50",
            "R125",
            "R143a",
            "R134a",
            "R170",
            "R23",
            "R41",
            "R161",
            "R152a",
            "R152",
            "R134",
            "R143",
            "R116",
        ]
        self.valid_train_mol_keys = [
            "R14",
            "R32",
            "R50",
            "R125",
            "R143a",
            "R134a",
            "R170",
            "R41",
        ]
        self.valid_prop_keys = [
            "sim_vap_density",
            "sim_liq_density",
            "sim_Pvap",
            "sim_Hvap",
        ]
        assert isinstance(obj_choice, str), "obj_choice must be string"
        assert obj_choice in [
            "ExpVal",
            "SSE",
            "ExpValPrior",
        ], "obj_choice must be SSE, ExpVal, or ExpValPrior"

        # Set training and testing data dictionaries
        self.molec_data_dict = {}
        self.test_data_dict = {}

        for molec in molec_names:
            if molec in list(self.all_train_molec_data.keys()):
                self.molec_data_dict[molec] = self.all_train_molec_data[molec]
            else:
                self.test_data_dict[molec] = self.all_test_molec_data[molec]

        try:
            self.all_gp_dict = get_gp_data_from_pkl(list(self.molec_data_dict.keys()))
        except:
            warnings.warn(
                "No gp data found. Many functions will not work without GP Data"
            )
        self.at_class = make_atom_type_class(at_number)
        self.seed = 1

        # Ensure we are working out of generalizedFF
        if Path(os.getcwd()).parent.name == "generalizedFF":
            self.use_root = Path(Path(os.getcwd()).parent)
        else:
            self.use_root = Path(os.getcwd())

        # set obj_choice
        self.obj_choice = obj_choice
        self.use_dir_name = self.make_results_dir(
            list(self.molec_data_dict.keys()), obj_choice=obj_choice
        )

        # If using ExpValPrior, check that you can calculate the average value of the pareto set of ExpVal
        if obj_choice == "ExpValPrior":
            # If a weight value is not set
            if weight_sclr == None:
                calc_Esse_avg = True
                molecule_str = self.molec_names_to_str(molec_names)
                # See if a calculated weight exists
                weight_sclr_file = (
                    self.use_root
                    / ("Results")
                    / ("at_" + str(at_number))
                    / ("weight_sclrs.json")
                )
                if weight_sclr_file.exists():
                    # If the file exists, load the data
                    with open(weight_sclr_file, "r") as json_file:
                        weight_sclr_data = json.load(json_file)

                    if molecule_str in weight_sclr_data:
                        self.weight_sclr = weight_sclr_data[molecule_str]
                        calc_Esse_avg = False

                # If the data doesn't exist, create it using the results from ExpVal Data
                if calc_Esse_avg:
                    assert hasattr(
                        self.at_class, "gaff_params"
                    ), "at_class must have attribute gaff_params"
                    assert hasattr(
                        self.at_class, "at_weights"
                    ), "at_class must have attribute at_weights"
                    assert (
                        len(self.at_class.gaff_params)
                        == len(self.at_class.at_weights)
                        == len(self.at_class.at_names)
                    ), "at_class gaff_params must have same length as at_weights"
                    # Get g_avg from average ExpVal Data if it exists
                    best_info = self.__get_ExpVal_info()
                    # print("best_info: ", best_info)
                    if best_info is None:
                        warnings.warn(
                            "No best info found for ExpVal. Cannot calculate weight_sclr. Setting all weights to 0"
                        )
                        # If no data is available, set weight to 0, this is equivalent to the ExpVal method
                        Esse_avg = 0
                    else:
                        # Get average value of best sets for each parameter
                        Esse_avg = best_info["Min Obj"].mean()
                    # Set scaler to 2 significant figures
                    self.weight_sclr = float(
                        "{:g}".format(float("{:.2g}".format(Esse_avg)))
                    )
                    weight_sclr_data = {molecule_str: self.weight_sclr}

                # Add it to the Json dictionary
                if calc_Esse_avg and best_info is not None:
                    if weight_sclr_file.exists():
                        # load it
                        with open(weight_sclr_file, "r") as json_file:
                            file_data = json.loads(json_file.read())
                        # change it
                        file_data[molecule_str] = self.weight_sclr
                    else:
                        file_data = weight_sclr_data
                    # write it all back
                    with open(weight_sclr_file, "w") as json_file:
                        json_file.write(json.dumps(file_data))
            else:
                self.weight_sclr = weight_sclr

        # Make results directory if it doesn't exist
        os.makedirs(self.use_dir_name, exist_ok=True)

    def __get_ExpVal_info(self):
        """
        Get pareto info for ExpVal

        Returns
        -------
        best_run_info: pd.DataFrame, dataframe of info for the best runs of ExpVal
        """
        dir_name = self.make_results_dir(
            list(self.molec_data_dict.keys()), obj_choice="ExpVal"
        )
        file_end = "best_per_run.csv"  # pareto_info.csv
        prop_names = ["sim_liq_density", "sim_vap_density", "sim_Pvap", "sim_Hvap"]
        best_info = None
        file = dir_name / (file_end)
        # print(file)
        if file.exists():
            best_info = pd.read_csv(file, index_col=False, header=0)
            if not "Min Obj" in best_info.columns:
                best_info = None

        return best_info

    def molec_names_to_str(self, molecules):
        if isinstance(molecules, str):
            molecule_str = molecules
        elif len(molecules) > 1 and isinstance(molecules, (list, np.ndarray)):
            # Assure list in correct order
            desired_order = list(self.all_train_molec_data.keys())
            molec_sort = sorted(molecules, key=lambda x: desired_order.index(x))
            molecule_str = "-".join(molec_sort)
        else:
            molecule_str = molecules[0]

        return molecule_str

    def make_results_dir(self, molecules, obj_choice=None, at_choice=None):
        """
        Makes a directory for results based on the scheme name, optimization method, and molecule names

        Parameters:
        -----------
        molecules: str or list of str, names of molecules to make directory for

        Output:
        -------
        dir_name: str, directory name for results
        """
        if obj_choice == None:
            obj_choice = self.obj_choice
        if at_choice == None:
            scheme_name = self.at_class.scheme_name
        else:
            scheme_name = "at_" + str(at_choice)
        assert isinstance(obj_choice, str) and obj_choice in [
            "ExpVal",
            "SSE",
            "ExpValPrior",
        ], "obj_choice must be a string or None"
        assert isinstance(
            molecules, (str, list, np.ndarray)
        ), "molecules must be a string or list/np.ndarray of strings"
        molecule_str = self.molec_names_to_str(molecules)

        if Path(os.getcwd()).parent.name == "generalizedFF":
            use_dir_name = Path(Path(os.getcwd()).parent)
        else:
            use_dir_name = Path(os.getcwd())

        dir_name = (
            use_dir_name / "Results" / (scheme_name) / (molecule_str) / (obj_choice)
        )

        return dir_name

    def get_uncertainty_data(self):
        df = pd.DataFrame(
            columns=[
                "Molecule",
                "Property",
                "Value",
                "Uncertainty",
                "Uncertainty Used",
                "Weight",
            ]
        )
        for key in list(self.molec_data_dict.keys()):
            molec_object = self.molec_data_dict[key]
            for prop_key in list(molec_object.uncertainties.keys()):
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

                values = np.array(list(exp_data.values()))
                y_unc = molec_object.uncertainties[prop_key]
                unc_val = np.maximum(y_unc, 0.02)
                weight = 1 / (unc_val * values) ** 2
                df_vals = [key, prop_key, values, y_unc, unc_val, weight]
                iter_df = pd.DataFrame([df_vals], columns=df.columns)
                iter_df = (
                    iter_df.apply(lambda col: col.explode(), axis=0)
                    .reset_index(drop=True)
                    .copy(deep=True)
                )
                df = pd.concat([df, iter_df], ignore_index=True)
        save_dir = self.use_root / "Results" / "gp_val_figs"
        os.makedirs(save_dir, exist_ok=True)
        file_name = save_dir / ("molecule_exp_unc_data.csv")
        df.to_csv(file_name, index=False, header=True)
        return df

    def values_pref_to_real(self, theta_guess):
        """
        Scales preferred units (Angstrom and Kelvin) to real units (nm, kJ/mol)

        Parameters
        ----------
        theta_guess: np.ndarray, the atom type scheme parameter set to start optimization at (sigma in A, epsilon in K)

        Returns
        -------
        theta_guess: np.ndarray, the atom type scheme parameter set to start optimization at (sigma in nm, epsilon in kJ/mol)
        """
        assert isinstance(theta_guess, np.ndarray), "theta_guess must be an np.ndarray"

        if len(theta_guess.shape) == 1:
            theta_guess = theta_guess.reshape(1, -1)

        midpoint = theta_guess.shape[1] // 2
        sigmas = (theta_guess[:, :midpoint]) * float(
            (1.0 * u.Angstrom).in_units(u.nm).value
        )
        epsilons = (theta_guess[:, midpoint:]) * float(
            (1.0 * u.K * u.kb).in_units("kJ/mol")
        )
        theta_guess = np.hstack((sigmas, epsilons))
        # sigmas = [float((x * u.Angstrom).in_units(u.nm).value) for x in theta_guess[:midpoint]]
        # epsilons = [float(x * (u.K * u.kb).in_units("kJ/mol")) for x in theta_guess[midpoint:]]
        # theta_guess = np.array(sigmas + epsilons)

        if theta_guess.shape[0] == 1:
            theta_guess = theta_guess.flatten()

        return theta_guess

    def values_real_to_pref(self, theta_guess):
        """
        Scales real units (nm, kJ/mol) to preferred units (Angstrom and Kelvin)

        Parameters
        ----------
        theta_guess: np.ndarray, the atom type scheme parameter set to start optimization at (sigma in nm, epsilon in kJ/mol)

        Returns
        -------
        theta_guess: np.ndarray, the atom type scheme parameter set to start optimization at (sigma in A, epsilon in K)
        """
        assert isinstance(theta_guess, np.ndarray), "theta_guess must be an np.ndarray"

        if len(theta_guess.shape) == 1:
            theta_guess = theta_guess.reshape(1, -1)

        midpoint = theta_guess.shape[1] // 2
        sigmas = (theta_guess[:, :midpoint]) * float(
            (1.0 * u.nm).in_units(u.Angstrom).value
        )
        epsilons = (theta_guess[:, midpoint:]) / float(
            (1.0 * u.K * u.kb).in_units("kJ/mol")
        )
        theta_guess = np.hstack((sigmas, epsilons))
        # sigmas = [float((x * u.nm).in_units(u.Angstrom).value) for x in theta_guess[:midpoint]]
        # epsilons = [float(x / (u.K * u.kb).in_units("kJ/mol")) for x in theta_guess[midpoint:]]
        # theta_guess = np.array(sigmas + epsilons)

        if theta_guess.shape[0] == 1:
            theta_guess = theta_guess.flatten()

        return theta_guess

    def get_exp_data(self, molec_object, prop_key):
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
        if prop_key not in self.valid_prop_keys:
            raise ValueError(
                "Invalid prop_key {}. Supported prop_key names are "
                "{}".format(prop_key, self.valid_prop_keys)
            )

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
        else:
            raise (
                ValueError,
                "all_gp_dict must contain a dict with keys sim_vap_density, sim_liq_density, sim_Hvap, or, sim_Pvap",
            )
        return exp_data, property_bounds, property_name

    def get_train_test_data(self, molec_key, prop_keys):
        """
        Get training and testing data from csv files

        Parameters
        ----------
        molec_key: str, key to consider. Must be in valid Keys: "R14", "R32", "R50", "R125", "R143a", "R134a", "R170"
        prop_keys: list of str, keys to consider. Must be in valid Keys: "sim_vap_density", "sim_liq_density", "sim_Hvap", "sim_Pvap"

        Returns:
        --------
        train_data: dict, dictionary of training data
        test_data: dict, dictionary of testing data
        """
        assert isinstance(molec_key, str), "molec_key must be a string"
        assert isinstance(prop_keys, list), "prop_keys must be a list"
        assert (
            all(isinstance(name, str) for name in prop_keys) == True
        ), "all key in prop_keys must be string"

        # Check property keys
        for prop_key in prop_keys:
            if prop_key not in self.valid_prop_keys:
                raise ValueError(
                    "Invalid prop_key {}. Supported prop_key names are "
                    "{}".format(prop_key, self.valid_prop_keys)
                )
        # Check molecule key
        if molec_key not in self.valid_mol_keys:
            raise ValueError(
                "Invalid molec_key {}. Supported molec_key names are "
                "{}".format(molec_key, self.valid_train_mol_keys)
            )

        # Get dict of testing data
        test_data = {}
        train_data = {}
        for strng in ["train", "test"]:
            # OPTIONAL (but not implemented here) append the MD density gp to the VLE density gp dictionary w/ key "MD Density"
            file_x = (
                self.use_root
                / "molec_gp_data"
                / (molec_key + "-vlegp/x_" + strng + ".csv")
            )
            assert (
                file_x.exists()
            ), "molec_gp_data/key-vlegp/x_****.csv does not exist. Check key list carefully"
            x = np.loadtxt(file_x, delimiter=",", skiprows=1)
            dict = train_data if strng == "train" else test_data
            dict["x"] = x

            for prop_key in prop_keys:
                file_y = (
                    self.use_root
                    / "molec_gp_data"
                    / (molec_key + "-vlegp/" + prop_key + "_y_" + strng + ".csv")
                )
                prop_data = np.loadtxt(file_y, delimiter=",", skiprows=1)
                dict[prop_key] = prop_data

        return train_data, test_data

    # Create fxn for analyzing a single gp w/ gpflow
    def eval_gp_new_theta(self, gp_theta_guess, molec_object, gp_object, Xexp):
        """
        Evaluates the gpflow model

        Parameters
        ----------
        gp_theta_guess: np.ndarray, the initial gp parameter set to start optimization at (sigma in nm, epsilon in kJ/mol scaled)
        molec_object: Instance of RXXConstants(), The data associated with a refrigerant molecule
        gp_object: gpflow.models.GPR, GP Model for a specific property
        Xexp: np.ndarray, Experimental state point (Temperature (K)) data for the property estimated by gp_object

        Returns
        -------
        gp_mean: tf.tensor, The (flattened) mean of the gp prediction
        gp_covar: tf.tensor, The (squeezed) covariance of the gp prediction
        gp_var: tf.tensor, The (flattened) variance of the gp prediction
        """
        assert isinstance(Xexp, np.ndarray), "Xexp must be an np.ndarray"
        assert isinstance(
            gp_object, gpflow.models.GPR
        ), "gp_object must be a gpflow.models.GPR"
        assert isinstance(
            gp_theta_guess, np.ndarray
        ), "gp_theta_guess must be an np.ndarray"
        assert isinstance(
            molec_object,
            (
                r14.R14Constants,
                r32.R32Constants,
                r50.R50Constants,
                r125.R125Constants,
                r134a.R134aConstants,
                r143a.R143aConstants,
                r170.R170Constants,
                r41.R41Constants,
            ),
        ), "molec_object must be a class object from r***.py"

        # Scale X data
        gp_Xexp = values_real_to_scaled(Xexp, molec_object.temperature_bounds)
        # Repeat theta guess x number of times
        gp_theta = np.repeat(gp_theta_guess, len(Xexp), axis=0)
        # Concatenate theta and tem values to get a gp input
        gp_input = np.concatenate((gp_theta, gp_Xexp), axis=1)

        # Get mean and std from gp
        gp_mean, gp_covar = gp_object.predict_f(gp_input, full_cov=True)
        # get variance from covar
        gp_var = np.diag(np.squeeze(gp_covar))

        return np.squeeze(gp_mean), np.squeeze(gp_covar), gp_var

    def calc_fxn(self, theta_guess):
        """
        Calculates the sse objective function

        Parameters
        ----------
        theta_guess: np.ndarray, the atom type scheme parameter set to start optimization at (sigma in nm, epsilon in kJ/mol)

        Returns
        -------
        obj: float, the objective function value
        sse_pieces: dict, dictionary of sse values for each property
        mean_wt_pieces: dict, dictionary of mean weights for each property
        """
        assert isinstance(theta_guess, np.ndarray), "theta_guess must be an np.ndarray"
        # Initialize weight and squared error arrays
        mean_array = []
        # var_array = []
        # var_theta = []

        # Unscale data from 0 to 1 to get correct objective values
        at_bounds_pref = self.at_class.at_bounds_nm_kjmol
        theta_guess = values_scaled_to_real(theta_guess.reshape(1, -1), at_bounds_pref)

        # Loop over molecules
        for molec in list(self.molec_data_dict.keys()):
            # Get constants for molecule
            molec_object = self.molec_data_dict[molec]
            # Get theta associated with each gp
            param_matrix = self.at_class.get_transformation_matrix(
                {molec: molec_object}
            )
            # Transform the guess, and scale to bounds
            gp_theta = theta_guess.reshape(-1, 1).T @ param_matrix
            gp_theta_guess = values_real_to_scaled(gp_theta, molec_object.param_bounds)
            # Get GPs associated with each molecule
            molec_gps_dict = self.all_gp_dict[molec]

            # Loop over gps (1 per property)
            for key in list(molec_gps_dict.keys()):
                # Get GP associated with property
                gp_model = molec_gps_dict[key]
                # Get X and Y data and bounds associated with the GP
                exp_data, y_bounds, y_names = self.get_exp_data(molec_object, key)
                # Get x and y data
                x_exp = np.array(list(exp_data.keys())).reshape(-1, 1)
                y_exp = np.array(list(exp_data.values()))
                # #Evaluate GP
                gp_mean_scl, gp_covar_scl, gp_var_scl = self.eval_gp_new_theta(
                    gp_theta_guess, molec_object, gp_model, x_exp
                )
                # Scale gp output to real value
                # gp_mean = values_scaled_to_real(gp_mean_scl, y_bounds)
                # Get y data uncertainties
                # unc = molec_object.uncertainties[key.replace("sim", "expt")]
                # y_var_unc = (y_exp*unc)**2
                # y_var_2pct = (y_exp*0.02)**2
                # y_var = np.maximum(y_var_unc, y_var_2pct)
                # y_std =np.sqrt(y_var)
                # gp_mean_y_scl = gp_mean_scl.flatten()/gp_var_scl.flatten()
                # Scale gp_variances to real values
                # y_bounds_2D = np.asarray(y_bounds).reshape(-1,2)
                # gp_covar = gp_covar_scl * (y_bounds_2D[:, 1] - y_bounds_2D[:, 0]) ** 2
                # gp_var = variances_scaled_to_real(gp_var_scl, y_bounds)
                # mean_array += list(gp_mean_y_scl)
                # No scaling necessary since we are using the scaled values (between 0 and 1) for the objectives and parameters
                mean_array += list(gp_mean_scl)
                # var_array += list(gp_var.flatten())

        return np.array(mean_array)

    def calc_wt_res(self, theta_guess):
        """
        Calculates the sse objective function

        Parameters
        ----------
        theta_guess: np.ndarray, the atom type scheme parameter set to start optimization at (sigma in nm, epsilon in kJ/mol)

        Returns
        -------
        res_array: np.ndarray, the residuals array
        sse_pieces: dict, dictionary of sse values for each property
        sse_var_pieces: dict, dictionary of sse variance values for each property
        var_ratios: np.ndarray, the variance ratios array
        mean_wt_pieces: dict, dictionary of mean weights for each property
        """
        assert isinstance(theta_guess, np.ndarray), "theta_guess must be an np.ndarray"
        # Initialize weight and squared error arrays
        res_array = []
        var_ratios = []
        sse_var_pieces = {}
        sse_pieces = {}

        # print("theta guess: ", theta_guess)

        # Loop over molecules
        for molec in list(self.molec_data_dict.keys()):
            # print("moelc")
            # Get constants for molecule
            molec_object = self.molec_data_dict[molec]
            # Get theta associated with each gp
            param_matrix = self.at_class.get_transformation_matrix(
                {molec: molec_object}
            )
            # print("param_matrix: ", param_matrix)
            # Transform the guess, and scale to bounds
            gp_theta = theta_guess.reshape(-1, 1).T @ param_matrix
            # print("gp_theta: ", theta_guess)
            gp_theta_guess = values_real_to_scaled(gp_theta, molec_object.param_bounds)
            # print("gp_theta guess: ", theta_guess)
            # Get GPs associated with each molecule
            molec_gps_dict = self.all_gp_dict[molec]

            # Loop over gps (1 per property)
            for key in list(molec_gps_dict.keys()):
                # Get GP associated with property
                gp_model = molec_gps_dict[key]
                # Get X and Y data and bounds associated with the GP
                exp_data, y_bounds, y_names = self.get_exp_data(molec_object, key)
                # Get x and y data
                x_exp = np.array(list(exp_data.keys())).reshape(-1, 1)
                y_exp = np.array(list(exp_data.values()))
                # #Evaluate GP
                gp_mean_scl, gp_covar_scl, gp_var_scl = self.eval_gp_new_theta(
                    gp_theta_guess, molec_object, gp_model, x_exp
                )
                # Scale gp output to real value
                gp_mean = values_scaled_to_real(gp_mean_scl, y_bounds)
                # Scale gp_variances to real values
                y_bounds_2D = np.asarray(y_bounds).reshape(-1, 2)
                gp_covar = gp_covar_scl * (y_bounds_2D[:, 1] - y_bounds_2D[:, 0]) ** 2
                gp_var = variances_scaled_to_real(gp_var_scl, y_bounds)

                # Calculate weight from uncertainty
                # Get y data uncertainties
                unc = molec_object.uncertainties[key.replace("sim", "expt")]
                y_var_unc = (y_exp * unc) ** 2
                y_var_2pct = (y_exp * 0.02) ** 2
                y_var = np.maximum(y_var_unc, y_var_2pct)
                weight_mpi = 1 / y_var
                # Create weight matrix.
                weights = np.diag(weight_mpi)

                # Calculate residuals
                res_vals = y_exp.reshape(-1, 1) - gp_mean.reshape(-1, 1)
                residuals = (res_vals.flatten()).tolist()
                # Calculate SSE
                sse = res_vals.T @ weights @ res_vals
                var_ratios_all = weights @ gp_covar
                # Add variance ratios (GP_Variances/y_uncertainties) to list
                var_ratios.append(np.diag(var_ratios_all).flatten())

                # Calculate sse Variance
                sse_var = 4 * (
                    res_vals.T @ (weights @ gp_covar @ weights) @ res_vals
                ) + 2 * np.trace((var_ratios_all) ** 2)
                # Save pieces
                sse_pieces[molec + "-" + key + "-sse"] = float(sse)
                sse_var_pieces[molec + "-" + key + "-sse_var"] = float(sse_var)
                res_array += residuals

        # List to flattened array
        res_array = np.array(res_array).flatten()
        var_ratios_arr = np.array(var_ratios).flatten()
        return res_array, sse_pieces, sse_var_pieces, var_ratios_arr

    def calc_obj(self, theta_guess):
        """
        Calculates the sse objective function

        Parameters
        ----------
        theta_guess: np.ndarray, the atom type scheme parameter set to start optimization at (sigma in nm, epsilon in kJ/mol)

        Returns
        -------
        obj: float, the objective function value
        sse_pieces: dict, dictionary of sse values for each property
        mean_wt_pieces: dict, dictionary of mean weights for each property
        """
        assert isinstance(theta_guess, np.ndarray), "theta_guess must be an np.ndarray"

        res, sse_pieces, sse_var_pieces, var_ratios = self.calc_wt_res(theta_guess)
        sse = float(sum(sse_pieces.values()))

        # var_ratio_dict = {}
        # dummy_molec = list(self.molec_data_dict.keys())[0]
        # molec_object = self.molec_data_dict[dummy_molec]
        # #Get number of data per molecule
        # n_data = len(list(molec_object.expt_vap_density.keys()))
        # var_ratios_per_m_and_p = var_ratios.reshape(-1, n_data).sum(axis=1)
        # count = 0
        # for molec in list(self.molec_data_dict.keys()):
        #     molec_gps_dict = self.all_gp_dict[molec]
        #     for key in list(molec_gps_dict.keys()):
        #         var_ratio_dict[molec + "-" + key + "-var_ratio"] = var_ratios_per_m_and_p[count]
        #         count += 1
        # print("var_ratio_dict: ", var_ratio_dict)

        if self.obj_choice == "SSE":
            obj = sse
        else:
            sum_var_ratios = np.sum(var_ratios)
            # print("sum_var_ratios: ", sum_var_ratios)
            expected_sse_val = sse + sum_var_ratios
            obj = expected_sse_val

        # If we have pareto info saved, get the average value of the pareto set
        if self.obj_choice == "ExpValPrior":
            # Calculate weights for each parameter
            # A difference from GAFF parameters are weighted % increase of ExpVal best objective
            inv_w2 = self.weight_sclr * self.at_class.at_weights
            # Get scaled parameter values
            theta_scl = values_real_to_scaled(
                theta_guess.reshape(1, -1), self.at_class.at_bounds_nm_kjmol
            ).flatten()
            gaff_params_2D = self.at_class.gaff_params.flatten()
            gaff_real = self.values_pref_to_real(gaff_params_2D)
            gaff_scl = values_real_to_scaled(
                gaff_real.reshape(1, -1), self.at_class.at_bounds_nm_kjmol
            ).flatten()
            # Add objective value to difference
            gaff_penalty = np.sum(inv_w2 * (theta_scl - gaff_scl) ** 2)
            obj += gaff_penalty
        else:
            gaff_penalty = 0

        # print(obj)
        return float(obj), sse_pieces, var_ratios, gaff_penalty

    # define one output calc obj
    def one_output_calc_obj(self, theta_guess):
        """
        Helper function. Calls calc_obj and returns only the objective function value

        Parameters
        ----------
        theta_guess: np.ndarray, the atom type scheme parameter set to start optimization at (sigma in nm, epsilon in kJ/mol)

        Returns
        --------
        obj: float, the objective function from the formula defined in the paper
        """
        assert isinstance(theta_guess, np.ndarray), "theta_guess must be an np.ndarray"
        obj = self.calc_obj(theta_guess)[0]

        return obj

    def gen_pareto_sets(self, samples, bounds, save_data=False):
        """
        generate LHS samples, calculate objective values, and sort them based on non-dominated sorting
        """
        # Generate LHS samples
        samples = generate_lhs(samples, bounds, self.seed, labels=None)
        # Define cost matrix (n_samplesx4)
        molec_dict1 = next(iter(self.all_gp_dict.values()))
        num_props = len(list(molec_dict1.keys()))
        num_molecs = len(list(self.all_gp_dict.keys()))

        # Loop over samples
        for s in range(len(samples)):
            # Calculate objective values
            obj, obj_pieces, var_ratios, gaff_penalty = self.calc_obj(samples[s])
            # Get SSE values per property by summing sse_dicts based on prescence of keys for each molecule
            var_ratios_organized = var_ratios.reshape(num_molecs, num_props, -1)
            prop_var_ratios = np.sum(var_ratios_organized, axis=(0, 2))
            df_sums, prop_names = self.__sum_sse_keys(obj_pieces)
            # Add variance ratios for each property when not using SSE
            if self.obj_choice != "SSE":
                df_sums += prop_var_ratios
            # Add GAFF penalty equally among all properties if using GAFF Prior objective
            if self.obj_choice == "ExpValPrior":
                df_sums += gaff_penalty / num_props

            # Set columns for costs
            if s == 0:
                costs = pd.DataFrame(columns=prop_names)
            df_sums_reordered = df_sums[costs.columns]
            # Concatenate the DataFrames along the rows axis
            costs = pd.concat([costs, df_sums_reordered], ignore_index=True)

        # Sort based on non-dominated sorting (call fffit.find_pareto_set(data, is_pareto_efficient)
        idcs, pareto_cost, dom_cost = find_pareto_set(
            costs.to_numpy(), is_pareto_efficient
        )
        # Put samples and cost values in order
        df_samples = pd.DataFrame(samples, columns=self.at_class.at_names)
        costs["is_pareto"] = idcs
        pareto_info = pd.concat([df_samples, costs], axis=1)
        pareto_info["Min Obj"] = pareto_info[prop_names].sum(axis=1)

        # Save pareto info
        if save_data == True:
            os.makedirs(self.use_dir_name, exist_ok=True)
            save_csv_path1 = self.use_dir_name / "pareto_info.csv"
            pareto_info.to_csv(save_csv_path1, index=False, header=True)

        return pareto_info

    def __sum_sse_keys(self, obj_pieces):

        # Dictionary to store the sum of values for each key
        sums_by_prop = {}

        # Iterate over the dictionary
        for full_key, value in obj_pieces.items():
            # Split the key to get the part after "molecX-" and before "-sse"
            key_part = full_key.split("-")[1]  # This will give 'keyX'

            # Accumulate the sum for this key
            if key_part in sums_by_prop:
                sums_by_prop[key_part] += value
            else:
                sums_by_prop[key_part] = value

        df_sums = pd.DataFrame(list(sums_by_prop.items()), columns=["Key", "Sum"])
        df_needed = df_sums.set_index("Key").T

        return df_needed, list(sums_by_prop.keys())

    def rank_parameters(self, theta_guess, save_data=False, save_label=None):
        """
        Ranks parameter estimability according to the alogithm in Table 2 Yao, K.Z., et al. (2003), Polym. React. Eng.

        Parameters
        ----------
        theta_guess: np.ndarray, the atom type scheme parameter set to start optimization at (sigma in nm, epsilon in kJ/mol)

        Returns
        -------
        ranked_indices: np.ndarray, the ranked indices of the parameters
        n_data: int, the number of data points
        """
        at_bounds_real = self.at_class.at_bounds_nm_kjmol
        theta_scl = values_real_to_scaled(theta_guess.reshape(1, -1), at_bounds_real)

        # Get Sensitivity Matrix
        fun = lambda x: self.calc_fxn(x)
        dfun = nd.Gradient(fun)
        # Use standardized scaled values
        Z = dfun([theta_scl])

        # Step 1: Calculate the magnitude of each column in Z
        column_magnitudes = np.linalg.norm(Z, axis=0)

        # Initialize variables
        ranked_indices = []  # To keep track of ranked parameter indices
        ranked_norm_vals = [] # To keep track of ranked parameter magnitudes
        k = 1
        n_data, m = Z.shape

        while k <= m:
            if k == 1:
                # Step 2: Identify the most estimable parameter
                max_index = np.argmax(column_magnitudes)
                ranked_indices.append(max_index)
                ranked_norm_vals.append(column_magnitudes[max_index])
            else:
                # Step 3: Build X_k with the k most estimable columns
                X_k = Z[:, ranked_indices]

                # Predict Z using ordinary least-squares
                # Note that this value is reporting Z_hat(Xk)^-1
                ols_est, _, _, _ = scipy.linalg.lstsq(X_k, Z)
                Z_hat = X_k @ ols_est

                # Calculate the residual matrix R_k, using Z_hat
                R_k = Z - Z_hat

                # Step 4: Calculate the magnitude of each column in R_k
                residual_magnitudes = np.linalg.norm(R_k, axis=0)

                # Step 5: Determine the next most estimable parameter
                # Ensure we pick a column that hasn't been ranked yet
                for idx in np.argsort(-residual_magnitudes):
                    if idx not in ranked_indices:
                        ranked_indices.append(idx)
                        ranked_norm_vals.append(residual_magnitudes[idx])
                        break

            k += 1  # Step 6: Increase k and repeat

        at_names_ranked = [self.at_class.at_names[i] for i in np.array(ranked_indices)]
        if save_data:
            # Write ranked indices and n_data to a csv on separate lines
            save_label = "_" + save_label if save_label is not None else ""
            dir_name = self.make_results_dir(list(self.molec_data_dict.keys()))
            os.makedirs(dir_name / "ranked_indices", exist_ok=True)
            os.makedirs(dir_name / "n_data", exist_ok=True)
            os.makedirs(dir_name / "Z_matrix", exist_ok=True)
            save_csv_path1 = os.path.join(
                dir_name, "ranked_indices", save_label + ".csv"
            )
            save_csv_path2 = os.path.join(dir_name, "n_data", save_label + ".csv")
            save_csv_path3 = os.path.join(dir_name, "Z_matrix", save_label + ".csv")
            # Transform to Pandas df and save to csv
            df_ranked_indices = pd.DataFrame(
                {"Indices": ranked_indices, "Norm Values":ranked_norm_vals, "AT Names": at_names_ranked}
            )
            # df_ranked_indices = pd.DataFrame(ranked_indices)
            df_ndata = pd.DataFrame([n_data])
            df_Z = pd.DataFrame(Z)
            df_ranked_indices.to_csv(save_csv_path1, index=False, header=False)
            df_ndata.to_csv(save_csv_path2, index=False, header=False)
            df_Z.to_csv(save_csv_path3, index=False, header=False)

        return np.array(ranked_indices), n_data, at_names_ranked


class Analyze_opt_res(Problem_Setup):
    """
    Class for analyzing problem and optimization results. Child class of General_Analysis
    """

    # Inherit objects from General_Analysis
    def __init__(self, mol_name_train, at_number, seed, obj_choice):
        """
        Parameters:
        -----------
        mol_name_train: List of training molecules that were considered during optimization; training refrigerant names w/ capital R.
        at_number: int, class for atom typing
        seed: int, random seed for optimization
        obj_choice: str, the objective choice for optimization. "SSE" or "ExpVal"
        """

        # Asserts
        super().__init__(mol_name_train, at_number, obj_choice)
        assert isinstance(seed, int) or seed is None, "seed must be int or None"
        self.seed = seed

    # def get_best_results(self, molec_ind, theta_guess = None):
    #     """
    #     Get the best optimization results. Pulls best parameter set from:
    #         1) Literature (stored in molecule objects)
    #         2) The algorithm trained for one training molecule (molec_ind)
    #         3) The algorithm trained for all training molecules (molec_data_dict.values())

    #     Parameters
    #     ----------
    #     molec_ind: str, the molecule name to consider
    #     theta_guess: np.ndarray, the atom type scheme parameter set to start optimization at (sigma in nm, epsilon in kJ/mol)

    #     Returns
    #     -------
    #     param_dict: dict, dictionary of the best optimization results overall, for each molecule, and literature comparison
    #     """
    #     assert isinstance(molec_ind, str), "molec_ind must be a string"
    #     assert molec_ind in list(self.all_train_molec_data.keys()), "molec_ind must be a key in all_train_molec_data"

    #     #Initialize Dict
    #     param_dict = {}
    #     #Get names and transformation matrix
    #     all_molec_list = list(self.molec_data_dict.keys())
    #     param_matrix = self.at_class.get_transformation_matrix({molec_ind: self.all_train_molec_data[molec_ind]})

    #     #If no theta_guess is provided, get the best parameter set for the optimization scheme
    #     if theta_guess is None:
    #         #Get best_per_run.csv for all molecules
    #         all_molec_dir = self.make_results_dir(all_molec_list)
    #         if os.path.exists(all_molec_dir / "best_per_run.csv"):
    #             unsorted_df = pd.read_csv(all_molec_dir / "best_per_run.csv", header = 0)
    #             all_df = unsorted_df.sort_values(by = "Min Obj")
    #             first_param_name = self.at_class.at_names[0] + "_min"
    #             last_param_name = self.at_class.at_names[-1] + "_min"
    #             full_opt_best = all_df.loc[0, first_param_name:last_param_name].values
    #             theta_guess = self.values_pref_to_real(full_opt_best)

    #     #If we have a GP parameter set, get the values in the necessary form
    #     if theta_guess is not None:
    #         all_best_nec = theta_guess.reshape(-1,1).T@param_matrix
    #         all_best_gp = values_real_to_scaled(all_best_nec.reshape(1,-1), self.all_train_molec_data[molec_ind].param_bounds)
    #         all_best_gp = tf.convert_to_tensor(all_best_gp, dtype=tf.float64)
    #     else:
    #         all_best_gp = None

    #     if len(all_molec_list) > 1 and isinstance(all_molec_list, (list,np.ndarray)):
    #         molecule_str = '-'.join(all_molec_list)
    #     else:
    #         molecule_str = all_molec_list[0]
    #     param_dict["Opt " + molecule_str] = all_best_gp

    #     molec_dir = self.make_results_dir([molec_ind])
    #     if os.path.exists(molec_dir / "best_per_run.csv"):
    #         unsorted_molec_df = pd.read_csv(molec_dir / "best_per_run.csv", header = 0)
    #         molec_df = unsorted_molec_df.sort_values(by = "Min Obj")
    #         first_param_name = self.at_class.at_names[0] + "_min"
    #         last_param_name = self.at_class.at_names[-1] + "_min"
    #         molec_best = molec_df.loc[0, first_param_name:last_param_name].values
    #         ind_best_real = self.values_pref_to_real(molec_best)
    #         ind_best_nec = ind_best_real.reshape(-1,1).T@param_matrix
    #         ind_best_gp = values_real_to_scaled(ind_best_nec.reshape(1,-1), self.all_train_molec_data[molec_ind].param_bounds)
    #         ind_best_gp = tf.convert_to_tensor(ind_best_gp, dtype=tf.float64)
    #     else:
    #         ind_best_gp = None
    #     param_dict["Opt " + molec_ind] = ind_best_gp

    #     molec_paper = np.array(list(self.molec_data_dict[molec_ind].lit_param_set.values()))
    #     paper_real = self.values_pref_to_real(molec_paper)
    #     paper_best_gp = values_real_to_scaled(paper_real.reshape(1,-1), self.all_train_molec_data[molec_ind].param_bounds)
    #     paper_best_gp = tf.convert_to_tensor(paper_best_gp, dtype=tf.float64)

    #     param_dict["Literature"] = paper_best_gp

    #     return param_dict

    def get_best_results(self, molec_ind, theta_guess=None):
        """
        Get the best optimization results. Pulls best parameter set from:
            1) Literature (stored in molecule objects)
            2) The algorithm trained for one training molecule (molec_ind)
            3) The algorithm trained for all training molecules (molec_data_dict.values())

        Parameters
        ----------
        molec_ind: str, the molecule name to consider
        theta_guess: np.ndarray, the atom type scheme parameter set to start optimization at (sigma in nm, epsilon in kJ/mol)

        Returns
        -------
        param_dict: dict, dictionary of the best optimization results overall, for each molecule, and literature comparison
        """
        assert isinstance(molec_ind, str), "molec_ind must be a string"
        assert molec_ind in list(
            self.all_train_molec_data.keys()
        ), "molec_ind must be a key in all_train_molec_data"

        # Initialize Dict
        param_dict = {}
        # Get names and transformation matrix
        all_molec_list = list(self.molec_data_dict.keys())
        param_matrix = self.at_class.get_transformation_matrix(
            {molec_ind: self.all_train_molec_data[molec_ind]}
        )

        # If no theta_guess is provided, get the best parameter set for the optimization scheme
        if theta_guess is None:
            # Get best_per_run.csv for all molecules
            all_molec_dir = self.make_results_dir(all_molec_list)
            if os.path.exists(all_molec_dir / "best_per_run.csv"):
                unsorted_df = pd.read_csv(all_molec_dir / "best_per_run.csv", header=0)
                all_df = unsorted_df.sort_values(by="Min Obj")
                first_param_name = self.at_class.at_names[0] + "_min"
                last_param_name = self.at_class.at_names[-1] + "_min"
                full_opt_best = all_df.loc[0, first_param_name:last_param_name].values
                theta_guess = self.values_pref_to_real(full_opt_best)

        # If we have a GP parameter set, get the values in the necessary form
        if theta_guess is not None:
            all_best_nec = theta_guess.reshape(-1, 1).T @ param_matrix
            all_best_gp = values_real_to_scaled(
                all_best_nec.reshape(1, -1),
                self.all_train_molec_data[molec_ind].param_bounds,
            )
            all_best_gp = tf.convert_to_tensor(all_best_gp, dtype=tf.float64)
        else:
            all_best_gp = None

        param_dict["Opt_All"] = all_best_gp

        molec_dir = self.make_results_dir([molec_ind])
        if os.path.exists(molec_dir / "best_per_run.csv"):
            unsorted_molec_df = pd.read_csv(molec_dir / "best_per_run.csv", header=0)
            molec_df = unsorted_molec_df.sort_values(by="Min Obj")
            first_param_name = self.at_class.at_names[0] + "_min"
            last_param_name = self.at_class.at_names[-1] + "_min"
            molec_best = molec_df.loc[0, first_param_name:last_param_name].values
            ind_best_real = self.values_pref_to_real(molec_best)
            ind_best_nec = ind_best_real.reshape(-1, 1).T @ param_matrix
            ind_best_gp = values_real_to_scaled(
                ind_best_nec.reshape(1, -1),
                self.all_train_molec_data[molec_ind].param_bounds,
            )
            ind_best_gp = tf.convert_to_tensor(ind_best_gp, dtype=tf.float64)
        else:
            ind_best_gp = None
        param_dict["Opt_One"] = ind_best_gp

        molec_paper = np.array(
            list(self.molec_data_dict[molec_ind].lit_param_set.values())
        )
        paper_real = self.values_pref_to_real(molec_paper)
        paper_best_gp = values_real_to_scaled(
            paper_real.reshape(1, -1), self.all_train_molec_data[molec_ind].param_bounds
        )
        paper_best_gp = tf.convert_to_tensor(paper_best_gp, dtype=tf.float64)

        param_dict["Literature"] = paper_best_gp

        molec_gaff = np.array(
            list(self.molec_data_dict[molec_ind].gaff_param_set.values())
        )
        gaff_real = self.values_pref_to_real(molec_gaff)
        gaff_best_gp = values_real_to_scaled(
            gaff_real.reshape(1, -1), self.all_train_molec_data[molec_ind].param_bounds
        )
        gaff_best_gp = tf.convert_to_tensor(gaff_best_gp, dtype=tf.float64)
        param_dict["GAFF"] = gaff_best_gp

        return param_dict

    def approx_jac(self, x, scale_theta=True, save_data=False, x_label=None):
        """
        Builds Jacobian Approximation

        Parameters
        ----------
        x: np.ndarray, the atom type scheme parameter set to start optimization at (sigma in nm, epsilon in kJ/mol)

        Returns
        -------
        jac: np.ndarray, the jacobian approximation
        """
        assert isinstance(x, np.ndarray), "x must be an np.ndarray"
        assert isinstance(save_data, bool), "save_data must be a bool"
        assert isinstance(
            x_label, (str, type(None))
        ), "x_label must be a string or None"

        # If we want to scale parameters
        if scale_theta:
            # Scale x values between 0 and 1 to get Hessian scaled w.r.t parameter differences
            x = values_real_to_scaled(
                x.reshape(1, -1), self.at_class.at_bounds_nm_kjmol
            ).flatten()
            jac = nd.Jacobian(self.__unscl_theta_calc_obj)(x)
        # Otherwise use the unscaled x values
        else:
            jac = nd.Jacobian(self.one_output_calc_obj)(x)

        if save_data:
            x_label = x_label if x_label is not None else "param_guess"
            dir_name = self.make_results_dir(list(self.molec_data_dict.keys()))
            os.makedirs(dir_name / "jac_approx", exist_ok=True)
            save_path = os.path.join(dir_name / "jac_approx", x_label + ".npy")
            np.save(save_path, jac)

        return jac

    def __unscl_theta_calc_obj(self, theta_guess):
        """
        Wrapper function converting scaled x values from 0 to 1 to nm and kj/mol values before inputting to calc_obj
        """
        assert isinstance(theta_guess, np.ndarray), "theta_guess must be an np.ndarray"
        # Unscale data from 0 to 1 to get correct objective values
        at_bounds_pref = self.at_class.at_bounds_nm_kjmol
        theta_guess = values_scaled_to_real(theta_guess.reshape(1, -1), at_bounds_pref)
        obj = self.calc_obj(theta_guess.flatten())[0]

        return obj

    def approx_hess(self, x, scale_theta=True, save_data=False, x_label=None):
        """
        Calculate gradient of function my_f using central difference formula and my_grad

        Parameters
        ----------
        x: np.ndarray, the atom type scheme parameter set to start optimization at (sigma in nm, epsilon in kJ/mol)

        Returns
        -------
        H: np.ndarray, the hessian approximation
        """
        assert isinstance(x, np.ndarray), "x must be an np.ndarray"
        assert isinstance(save_data, bool), "save_data must be a bool"
        assert isinstance(
            x_label, (str, type(None))
        ), "x_label must be a string or None"

        # If we want to scale parameters
        if scale_theta:
            # Scale x values between 0 and 1 to get Hessian scaled w.r.t parameter differences
            x = values_real_to_scaled(
                x.reshape(1, -1), self.at_class.at_bounds_nm_kjmol
            ).flatten()
            H = nd.Hessian(self.__unscl_theta_calc_obj)(x)
            scl_str = "_scl"
        # Otherwise use the unscaled x values
        else:
            H = nd.Hessian(self.one_output_calc_obj)(x)
            scl_str = ""

        if save_data:
            x_label = x_label if x_label is not None else "param_guess"
            dir_name = self.make_results_dir(list(self.molec_data_dict.keys()))
            os.makedirs(dir_name / "hess_approx", exist_ok=True)
            save_path = os.path.join(
                dir_name / "hess_approx", x_label + scl_str + ".npy"
            )
            np.save(save_path, H)

        return H

    def calc_MAPD_any(
        self, all_molec_list, theta_guess, save_data=False, save_label=None
    ):
        """
        Calculate the mean absolute percentage deviation for each training data prediction
        """
        assert isinstance(save_data, bool), "save_data must be a bool"
        assert isinstance(
            save_label, (str, type(None))
        ), "save_label must be a string or None"
        assert all(
            item in list(self.molec_data_dict.keys()) for item in all_molec_list
        ), "all_molec_list must be a subset of the training molecules"

        new_data = []
        # Make pdf
        dir_name = self.make_results_dir(list(self.molec_data_dict.keys()))
        # Loop over all molecules of interest
        for molec in all_molec_list:
            # Get constants for molecule
            molec_object = self.all_train_molec_data[molec]
            # Get GPs associated with each molecule
            molec_gps_dict = self.all_gp_dict[molec]
            # Get param matrix
            param_matrix = self.at_class.get_transformation_matrix(
                {molec: self.all_train_molec_data[molec]}
            )

            prop_data = []
            # Loop over gps (1 per property)
            for key in list(molec_gps_dict.keys()):
                # Get GP associated with property
                gp_model = molec_gps_dict[key]
                # Get X and Y data and bounds associated with the GP
                exp_data, y_bounds, y_names = self.get_exp_data(molec_object, key)
                x_data = np.array(list(exp_data.keys()))
                y_data = np.array(list(exp_data.values()))

                # get theta guess into scaled units
                all_best_real = self.values_pref_to_real(theta_guess)
                all_best_nec = all_best_real.reshape(-1, 1).T @ param_matrix
                all_best_gp = values_real_to_scaled(
                    all_best_nec.reshape(1, -1),
                    self.all_train_molec_data[molec].param_bounds,
                )
                theta_guess_scl = tf.convert_to_tensor(all_best_gp, dtype=tf.float64)

                T_scaled = values_real_to_scaled(
                    x_data, molec_object.temperature_bounds
                )
                parm_set_repeat = np.tile(theta_guess_scl, (len(x_data), 1))
                gp_theta_guess = np.hstack((parm_set_repeat, T_scaled))
                mean_scaled, var_scaled = gp_model.predict_f(gp_theta_guess)
                mean = values_scaled_to_real(mean_scaled, y_bounds)
                mapd = mean_absolute_percentage_error(y_data, mean) * 100

                prop_data.append(mapd)
            data_to_append = [molec] + prop_data
            new_data.append(data_to_append)

        mapd_names = [s.replace("sim", "mapd") for s in list(molec_gps_dict.keys())]
        columns = list(["molecule"]) + mapd_names
        df = pd.DataFrame(new_data, columns=columns)

        if save_data == True:
            save_label = save_label if save_label is not None else "MAPD_ind_set"
            save_csv_path = os.path.join(dir_name, "MAPD_ind_" + save_label + ".csv")
            df.to_csv(save_csv_path, index=False, header=True)
        return df

    def calc_MAPD_best(
        self, all_molec_list, theta_guess=None, save_data=False, save_label=None
    ):
        """
        Calculate the mean absolute percentage deviation for each training data prediction
        """
        assert isinstance(save_data, bool), "save_data must be a bool"
        assert isinstance(
            save_label, (str, type(None))
        ), "save_label must be a string or None"
        # assert all(
        #     item in list(self.molec_data_dict.keys()) for item in all_molec_list
        # ), "all_molec_list must be a subset of the training molecules"
        df = pd.DataFrame(columns=["Molecule", "Property", "Model", "MAPD"])
        # Make pdf
        dir_name = self.make_results_dir(list(self.molec_data_dict.keys()))
        all_data_lists = {"Opt_All": [], "Opt_One": [], "Literature": [], "GAFF": []}
        # Loop over all molecules of interest
        for molec in all_molec_list:
            # Get constants for molecule
            molec_object = self.all_train_molec_data[molec]
            # Get GPs associated with each molecule
            molec_gps_dict = self.all_gp_dict[molec]
            test_params = self.get_best_results(molec, theta_guess)

            # Loop over gps (1 per property)
            mapd_vals = {"Opt_All": [], "Opt_One": [], "Literature": [], "GAFF": []}
            for key in list(molec_gps_dict.keys()):
                # Set label
                label = molec + "_" + key
                # Get GP associated with property
                gp_model = molec_gps_dict[key]
                # Get X and Y data and bounds associated with the GP
                exp_data, y_bounds, y_names = self.get_exp_data(molec_object, key)
                x_data = np.array(list(exp_data.keys()))
                y_data = np.array(list(exp_data.values()))

                T_scaled = values_real_to_scaled(
                    x_data, molec_object.temperature_bounds
                )
                for i, param_set_key in enumerate(list(test_params.keys())):
                    param_set = test_params[param_set_key]
                    if param_set is not None:
                        parm_set_repeat = np.tile(param_set, (len(x_data), 1))
                        gp_theta_guess = np.hstack((parm_set_repeat, T_scaled))
                        mean_scaled, var_scaled = gp_model.predict_f(gp_theta_guess)
                        mean = values_scaled_to_real(mean_scaled, y_bounds)
                        mapd = mean_absolute_percentage_error(y_data, mean) * 100
                        mapd_vals[param_set_key].append(mapd)
                    else:
                        mapd_vals[param_set_key].append(None)

            for param_set_key in list(test_params.keys()):
                all_data_lists[param_set_key].append([molec] + mapd_vals[param_set_key])

        mapd_names = [s.replace("sim", "mapd") for s in list(molec_gps_dict.keys())]
        columns = list(["molecule"]) + mapd_names

        dfs = {}
        for key in list(all_data_lists.keys()):
            df = pd.DataFrame(all_data_lists[key], columns=columns)
            df["Type"] = key
            dfs[key] = df

        combined_df = pd.concat(list(dfs.values()), ignore_index=True)
        if save_data == True:
            os.makedirs(dir_name / "MAPD", exist_ok=True)
            save_label = save_label if save_label is not None else "MAPD_set"
            save_csv_path = os.path.join(dir_name / "MAPD", save_label + ".csv")
            combined_df.to_csv(save_csv_path, index=False, header=True)

        return dfs

    def get_unique_sets(
        self, all_param_sets, tolerance=0.05, save_data=False, save_label=None
    ):
        """
        Gets unique sets of parameters from an array of parameters

        """
        assert isinstance(save_data, bool), "save_data must be a bool"
        assert isinstance(
            save_label, (str, type(None))
        ), "save_label must be a string or None"
        # Scale values from preferred to real units
        all_param_sets_real = self.values_pref_to_real(all_param_sets)
        # Scale values between 0 and 1 with minmax scaler
        scaler = MinMaxScaler()
        scaler.fit(self.at_class.at_bounds_nm_kjmol.T)
        all_param_sets_scaled = scaler.transform(all_param_sets_real)
        # Calculate the scaled euclidean distance between each pair of scaled points
        dist = pdist(all_param_sets_scaled) / np.sqrt(all_param_sets.shape[1])
        # Convert the condensed distance matrix to square form
        dist_sq = squareform(dist)

        # Initialize a boolean array to keep track of unique sets
        unique_mask = np.ones(all_param_sets.shape[0], dtype=bool)

        # Iterate over the upper triangle of the distance matrix
        for i in range(all_param_sets.shape[0]):
            # If the current set is already marked as non-unique, skip it
            if not unique_mask[i]:
                continue
            # Mark sets within the threshold distance as non-unique
            within_threshold = dist_sq[i] <= tolerance
            unique_mask[within_threshold] = False
            unique_mask[i] = True  # Keep the current set

        unique_param_sets = all_param_sets[unique_mask]

        # #Fill diagonals w/ infinity
        # np.fill_diagonal(dist_sq, np.inf)
        # # Find the indices of points where all distances to other points (excluding diagonal) are greater than 0.01
        # valid_indices = np.where(np.all(dist_sq > 0.01, axis=1))[0]
        # if len(valid_indices) > 0:
        #     unique_param_sets = all_param_sets[valid_indices]
        # else:
        #     unique_param_sets = all_param_sets[0].reshape(1,-1)

        data_df = pd.DataFrame(unique_param_sets, columns=self.at_class.at_names)

        if save_data == True:
            save_label = save_label if save_label is not None else "test_set"
            save_csv_path = os.path.join(
                self.use_dir_name, "unique_" + save_label + ".csv"
            )
            data_df.to_csv(save_csv_path, index=False, header=True)

        return data_df


class Opt_ATs(Problem_Setup):
    """
    The class for Least Squares regression analysis. Child class of General_Analysis

    Methods:
    --------
    __init__: Initializes the class
    __scipy_min_fxn: The scipy function for minimizing the data
    get_param_inits: Gets parameter guesses and sets up bounds for optimization
    __get_scipy_soln: Gets scipy solution
    __get_opt_iter_info: Runs Optimization, times progress, and makes iter_df
    optimize_ats: Optimizes the atom type parameters
    """

    # Inherit objects from General_Analysis
    def __init__(self, molec_names, at_number, repeats, seed, obj_choice):
        """
        Parameters:
        -----------
        molec_data_dict: dict, keys are training refrigerant names w/ capital R, values are class objects from r***.py
        all_gp_dict: dict of dict, keys are training refrigerant names w/ capital R, values are dictionaries of properties and GP objects
        at_class: Instance of Atom_Types, class for atom typing
        repeats: int, number of optimization runs to do
        seed: int, random seed for optimization
        obj_choice: str, the objective choice for optimization. "SSE" or "ExpVal"
        """

        # Asserts
        super().__init__(molec_names, at_number, obj_choice)
        assert isinstance(repeats, int) and repeats > 0, "repeats must be int > 0"
        assert isinstance(seed, int) or seed is None, "seed must be int or None"
        self.repeats = repeats
        self.seed = seed
        self.iter_param_data = []
        self.iter_obj_data = []
        self.iter_sse_pieces = []
        self.iter_count = 0

    # define the scipy function for minimizing
    def __scipy_min_fxn(self, theta_guess):
        """
        The scipy function for minimizing the data

        Parameters
        ----------
        theta_guess: np.ndarray, the atom type scheme parameter set to start optimization at (sigma in nm, epsilon in kJ/mol)

        Returns
        --------
        obj: float, the objective function from the formula defined in the paper
        """
        assert isinstance(theta_guess, np.ndarray), "theta_guess must be an np.ndarray"
        obj, sse_pieces, var_ratios, gaff_penalty = self.calc_obj(theta_guess)

        # Scale theta_guess to preferred units
        theta_guess_pref = self.values_real_to_pref(theta_guess)

        # if self.iter_count > 0:
        #     print(self.iter_param_data[-1])
        #     print(theta_guess_pref)
        #     print(np.allclose(self.iter_param_data[-1], theta_guess_pref))
        #     print(len(self.iter_param_data))
        # #Append intermediate values to list
        if len(self.iter_param_data) == 0 or (
            not np.allclose(self.iter_param_data[-1], theta_guess_pref)
        ):
            # print("Adding data to lists")
            self.iter_param_data.append(theta_guess_pref)
            self.iter_sse_pieces.append(sse_pieces)
            self.iter_obj_data.append(obj)
            self.iter_count += 1
            # print(self.iter_count, obj)

        return obj

    def estimate_opt(
        self, theta_guess, ranked_indices, n_data, save_data=False, save_label=None
    ):
        """
        Determines the optimal number of parameters to estimate based on algorithm in Wu, S., et al. (2011). Int. J. Advanced Mechatronic Systems

        Parameters
        ----------
        theta_guess: np.ndarray, the atom type scheme parameter set to start optimization at (sigma in nm, epsilon in kJ/mol)
        ranked_indices: np.ndarray, the ranked indices of the parameters
        n_data: int, the number of data points

        Returns
        -------
        opt_num_params: int, the optimal number of parameters to estimate
        rcc: np.ndarray, the relative critical cost values
        loss_k: np.ndarray, the loss values for each k
        loss_k_params: np.ndarray, the parameter values for each k
        """
        # Create arrays to store results
        loss_k_params = np.zeros((len(ranked_indices) + 1, len(theta_guess)))
        loss_k = np.zeros(len(ranked_indices) + 1)
        # Calculate the loss for the full parameter set with no guesses
        loss_k[0] = self.one_output_calc_obj(theta_guess)
        loss_k_params[0, :] = theta_guess

        # Set obj_choice to ExpVal if ExpValPrior for this operation
        # Cannot do ExpVal Prior because of scaled weights based on objective
        # obj_changed = False
        # if self.obj_choice == "ExpValPrior":
        #     self.obj_choice = "ExpVal"
        #     obj_changed = True

        for i in range(1, len(ranked_indices) + 1):
            # Create a mask to identify which parameters are being optimized
            # Note that ranked_indices is 0-based
            theta_estim = theta_guess[ranked_indices[:i]]
            mask = np.zeros(len(theta_guess), dtype=bool)
            mask[ranked_indices[:i]] = True

            def obj_wrapper(x_estim, *args):
                # Reconstruct the full parameter list
                theta_full = theta_guess.copy()
                theta_full[mask] = x_estim
                return self.__scipy_min_fxn(theta_full, *args)

            solution = optimize.minimize(
                obj_wrapper,
                theta_estim,
                bounds=self.at_class.at_bounds_nm_kjmol[ranked_indices[:i], :],
                method="L-BFGS-B",
                options={"disp": False, "eps": 1e-10, "ftol": 1e-10},
            )

            # Reconstruct the full parameter list with optimized values
            theta_opt = theta_guess
            theta_opt[mask] = solution.x
            loss_k_params[i, :] = theta_opt
            loss_k[i] = solution.fun

        # Reset obj_choice to original value
        # if obj_changed:
        #     self.obj_choice = "ExpValPrior"

        # Compute critical ratio.
        rcc = np.zeros(len(ranked_indices) + 1)
        # Set the 1st rcc as nan (if nothing is optimized rcc is undefined)
        rcc[0] = np.nan
        # Loop over subgroups. Note that the last rcc is 0 by definition
        for k in range(1, len(ranked_indices)):
            p = len(ranked_indices)
            rck = (loss_k[k] - loss_k[-1]) / (p - k)
            rc_kub = max(rck - 1, 2 * rck / (p - k + 2))
            rcc[k] = (p - k) * (rc_kub - 1) / n_data

        opt_num_params = np.nanargmin(rcc)

        # Make loss matrix
        # Get parameters from real to preferred units
        loss_k_params_pref = self.values_real_to_pref(loss_k_params)
        # Append loss and rcc columns
        loss_matrix = np.hstack(
            (loss_k_params_pref, loss_k[:, np.newaxis], rcc[:, np.newaxis])
        )
        col_names = self.at_class.at_names + [self.obj_choice, "rcc"]
        # Make dataframe
        loss_df = pd.DataFrame(loss_matrix, columns=col_names)
        # Add number of params optimized
        loss_df["num params opt"] = loss_df.index

        # Optimal params to df
        df_opt_k_params = loss_df.iloc[[opt_num_params]]

        # Save data
        if save_data:
            # Write ranked indices and n_data to a csv on separate lines
            save_label = "_" + save_label if save_label is not None else ""
            dir_name = self.make_results_dir(list(self.molec_data_dict.keys()))
            save_csv_path1 = os.path.join(dir_name, "loss_data" + save_label + ".csv")
            save_csv_path2 = os.path.join(
                dir_name, "opt_params_rcc" + save_label + ".csv"
            )

            # Save loss data to csv
            loss_df.to_csv(save_csv_path1, index=False, header=True)

            # Save optimal parameter set to csv
            df_opt_k_params.to_csv(save_csv_path2, index=False, header=True)

        col_locs = [df_opt_k_params.columns.get_loc(c) for c in self.at_class.at_names]
        opt_k_params = df_opt_k_params.iloc[0, col_locs].to_numpy()

        return opt_num_params, rcc, loss_matrix, opt_k_params

    def get_param_inits(self, method="pareto"):
        """
        Gets parameter guesses and sets up bounds for optimization

        Returns
        -------
        param_inits: np.ndarray, the initial atom type scheme parameter set to start optimization at (sigma in nm, epsilon in kJ/mol)
        """
        # set seed here
        if self.seed is not None:
            np.random.seed(self.seed)

        if method == "pareto":
            # Try to load from csv params if it exists
            save_csv_path1 = self.use_dir_name / "pareto_info.csv"
            if save_csv_path1.exists():
                all_pareto_info = pd.read_csv(save_csv_path1, header=0)
            # Otherwise generate 10**5 pareto sets, and save the data
            else:
                warnings.warn(
                    "No Pareto info found. Generating 10^5 LHS to find Pareto sets",
                    UserWarning,
                )
                all_pareto_info = self.gen_pareto_sets(
                    int(10**5), self.at_class.at_bounds_nm_kjmol, save_data=True
                )
            # Get dominated vs nondominated points
            pareto_data = all_pareto_info[all_pareto_info["is_pareto"] == True]
            dominated_data = all_pareto_info[all_pareto_info["is_pareto"] == False]
            pareto_points = pareto_data[self.at_class.at_names].copy()
            dom_points = dominated_data[self.at_class.at_names].copy()
            # Ensure we are using less repeats than pareto optimal points
            if len(pareto_points) < self.repeats:
                # Could opt to use more repeats than # of pareto sets
                warnings.warn(
                    f"More repeats ({self.repeats}) than Pareto optimal sets ({len(pareto_points)}). Generating repeats for number of Paerto sets",
                    UserWarning,
                )
                self.repeats = len(pareto_points)
                # num_false = self.repeats - len(pareto_points)
                # pareto_data_false = dom_points.sample(n=num_false, random_state=self.seed)
                # restart_data = pd.concat([pareto_points, pareto_data_false], ignore_index=True)
            # Randomly sample pareto points without replacement
            restart_data = pareto_points.sample(n=self.repeats, random_state=self.seed)
            param_inits = restart_data.to_numpy()
        elif method == "lhs":
            # Could also sample with LHS
            param_sets = generate_lhs(
                self.repeats, self.at_class.at_bounds_nm_kjmol, self.seed, labels=None
            )
            param_inits = param_sets.to_numpy()
        else:
            # Could also sample randomly
            lb = self.at_class.at_bounds_nm_kjmol[:, 0].T
            ub = self.at_class.at_bounds_nm_kjmol[:, 1].T
            param_inits = np.random.uniform(
                low=lb, high=ub, size=(self.repeats, len(lb))
            )

        return param_inits

    def __get_scipy_soln(self, run, param_inits):
        """
        Gets scipy solution

        Parameters
        ----------
        run: int, the current run number
        param_inits: np.ndarray, the initial atom type scheme parameter set to start optimization at (sigma in nm, epsilon in kJ/mol)

        Returns
        -------
        solution: scipy.optimize.OptimizeResult, the scipy optimization result
        time_per_run: float, the time taken for the run
        """
        # Start timer
        time_start = time.time()
        # Get guess and find scipy.optimize solution
        # bounds = (self.at_class.at_bounds_nm_kjmol[:,0], self.at_class.at_bounds_nm_kjmol[:,1])
        # solution = optimize.least_squares(self.__scipy_min_fxn, param_inits[run], bounds=bounds,
        #                                  method='trf', verbose = 0)
        # print("set: ", param_inits[run])
        solution = optimize.minimize(
            self.__scipy_min_fxn,
            param_inits[run],
            bounds=self.at_class.at_bounds_nm_kjmol,
            method="L-BFGS-B",
            options={"disp": False, "eps": 1e-10, "ftol": 1e-10},
        )
        # End timer and calculate total run time
        time_end = time.time()
        time_per_run = time_end - time_start

        return solution, time_per_run

    def __get_opt_iter_info(self, run, solution, time_per_run):
        """
        Runs Optimization, times progress, and makes iter_df

        Parameters
        ----------
        run: int, the current run number (indexed from 0)
        solution: scipy.optimize.OptimizeResult, the scipy optimization result
        time_per_run: float, the time taken for the run

        Returns
        -------
        iter_df: pd.DataFrame, the iteration dataframe
        """
        # Get list of iteration, sse, and parameter data
        iter_list = np.array(range(self.iter_count)) + 1
        obj_list = np.array(self.iter_obj_data)
        param_list = self.iter_param_data

        # Create df for each least squares run
        # Initialize results dataframe
        # Make list of column names
        org_names = ["Run", "Iter", "Min Obj", "Min Obj Cum."]
        at_names_min = [name + "_min" for name in self.at_class.at_names]
        at_names_cum = [name + "_cum" for name in self.at_class.at_names]
        self.col_names_iter = (
            org_names[0:3] + at_names_min + [org_names[3]] + at_names_cum
        )

        # Create a pd dataframe of all iteration information. Initialize cumulative columns as zero
        iter_df = pd.DataFrame(columns=self.col_names_iter)
        iter_df["Iter"] = iter_list
        iter_df["Min Obj"] = obj_list
        iter_df[at_names_min] = param_list
        iter_df = (
            iter_df.apply(lambda col: col.explode(), axis=0)
            .reset_index(drop=True)
            .copy(deep=True)
        )
        iter_df["Run"] = run + 1

        # Add Theta min obj and theta_sse obj
        # Loop over each iteration to create the min obj columns
        for j in range(len(iter_df)):
            min_sse = iter_df.loc[j, "Min Obj"].copy()
            if j == 0 or min_sse < iter_df["Min Obj Cum."].iloc[j - 1]:
                min_param = iter_df.loc[j, at_names_min].copy().to_numpy()
            else:
                min_sse = iter_df["Min Obj Cum."].iloc[j - 1].copy()
                min_param = iter_df.loc[j - 1, at_names_cum].copy().to_numpy()

            iter_df.loc[j, "Min Obj Cum."] = min_sse
            iter_df.loc[j, at_names_cum] = min_param

            # Add iteration info from sse and mean weight
            sse_names = list(self.iter_sse_pieces[j].keys())
            iter_df.loc[j, sse_names] = np.array(list(self.iter_sse_pieces[j].values()))

        # Find largest iter for each restart
        idx = iter_df.groupby("Run")["Iter"].idxmax()
        # Add following to only last iter runs
        iter_df.loc[idx, "Run Time"] = time_per_run
        iter_df.loc[idx, "jac evals"] = solution.njev
        iter_df.loc[idx, "func evals"] = solution.nfev
        iter_df.loc[idx, "Optimality"] = scipy.linalg.norm(solution.jac, ord=np.inf)
        iter_df.loc[idx, "Term Status"] = solution.status
        iter_df.loc[idx, "Message"] = solution.message
        # iter_df.loc[idx, "Optimality"] = solution.optimality

        return iter_df

    # Define fxn to optimize w/ restarts
    def optimize_ats(self, param_inits, repeat_num=None):
        """
        Optimizes New atom typing parameters

        Returns
        -------
        ls_results: pd.DataFrame, the results dataframe
        """
        ls_results = pd.DataFrame()

        # Generate param sets if one is not given
        if param_inits is None:
            param_inits = self.get_param_inits()

        # Optimize w/ retstarts
        for i in range(len(param_inits)):
            # Get Iteration results
            # print("restart: ", i)
            solution, time_per_run = self.__get_scipy_soln(i, param_inits)
            run_num = repeat_num if repeat_num != None else i
            iter_df = self.__get_opt_iter_info(run_num, solution, time_per_run)
            # print("iter_df: ", iter_df)
            # Append to results_df
            ls_results = pd.concat([ls_results, iter_df], ignore_index=True)

            # Reset iter lists and change seed
            self.seed += 1
            self.iter_param_data = []
            self.iter_obj_data = []
            self.iter_sse_pieces = []
            self.iter_count = 0

        # Reset the index of the pandas df
        ls_results = ls_results.reset_index(drop=True)

        # Sort by lowest obj first
        sort_ls_res = ls_results.sort_values(["Min Obj Cum."], ascending=True)

        # Back out best job for each run
        run_best = sort_ls_res.groupby("Run").first().reset_index()
        best_runs = run_best.sort_values(["Min Obj Cum."], ascending=True)

        # if self.save_data:
        #     dir_name = self.make_results_dir(list(self.molec_data_dict.keys()))
        #     os.makedirs(dir_name, exist_ok=True)
        #     #Save original results
        #     save_path1 = os.path.join(dir_name, "opt_at_results.csv")
        #     ls_results.to_csv(save_path1, index = False)
        #     #Save sorted results
        #     save_path2 = os.path.join(dir_name, "sorted_at_res.csv")
        #     sort_ls_res.to_csv(save_path2, index = False)
        #     #Save sorted results for each run
        #     save_path3 = os.path.join(dir_name, "best_per_run.csv")
        #     best_runs.to_csv(save_path3, index = False)

        return ls_results, sort_ls_res, best_runs


class Vis_Results(Analyze_opt_res):
    """
    Class For vizualizing GP and Optimization Results
    """

    # Inherit objects from General_Analysis
    def __init__(self, mol_name_train, at_number, seed, obj_choice):
        # Asserts
        super().__init__(mol_name_train, at_number, seed, obj_choice)

    # Define function to check GP Accuracy
    def check_GPs(self, get_slices=False, get_train_test=False):
        """
        Makes GP validation figures for each training molecule
        """

        dir_name = "Results/gp_val_figs/"
        os.makedirs(dir_name, exist_ok=True)

        mapd_df_rows = []

        # Loop over molecules
        for molec in list(self.all_train_molec_data.keys()):
            # Get constants for molecule
            molec_object = self.all_train_molec_data[molec]
            # Get GPs associated with each molecule
            molec_gps_dict = self.all_train_gp_dict[molec]
            # Get testing data for that molecule
            train_data, test_data = self.get_train_test_data(
                molec, list(molec_gps_dict.keys())
            )
            # Make pdf

            pdf = PdfPages(dir_name + "/" + molec + ".pdf")
            # Loop over gps (1 per property)
            for key in list(molec_gps_dict.keys()):
                # Set label
                label = molec + "_" + key
                # Get GP associated with property
                gp_model = molec_gps_dict[key]
                # Get X and Y data and bounds associated with the GP
                exp_data, y_bounds, y_names = self.get_exp_data(molec_object, key)
                # Plot model performance
                pdf.savefig(
                    plot_model_performance(
                        {label: gp_model}, test_data["x"], test_data[key], y_bounds
                    )
                )
                plt.close()

                #Calculate and save MAPD
                y_data_physical = values_scaled_to_real(test_data[key], y_bounds)
                for label, model in {label: gp_model}.items():
                    gp_mu, gp_var = model.predict_f(test_data["x"])
                    gp_mu_physical = values_scaled_to_real(gp_mu, y_bounds)
                    mapd = 100*np.mean(
                            np.abs(
                                (gp_mu_physical - y_data_physical.reshape(-1, 1))
                                / y_data_physical.reshape(-1, 1)
                            )
                        )
                    mapd_df_rows.append([molec, key, mapd])
                if get_slices:
                    # Plot temperature slices
                    figs = plot_slices_temperature(
                        {label: gp_model},
                        molec_object.n_params,
                        molec_object.temperature_bounds,
                        y_bounds,
                        plot_bounds=molec_object.temperature_bounds,
                        property_name=y_names,
                    )

                    for fig in figs:
                        pdf.savefig(fig)
                    del figs

                    # Plot Parameter slices
                    for param_name in molec_object.param_names:
                        figs = plot_slices_params(
                            {label: gp_model},
                            param_name,
                            molec_object.param_names,
                            list(exp_data.keys())[
                                2
                            ],  # Use the 3rd temp to plot param slices
                            molec_object.temperature_bounds,
                            y_bounds,
                            property_name=y_names,
                        )
                        plt.close()

                        for fig in figs:
                            pdf.savefig(fig)
                        del figs
                if get_train_test:
                    # #Plot test vs train for each parameter set
                    for test_params in test_data["x"][:, : molec_object.n_params]:
                        # Find points in test set with correct param value
                        # Locate rows where parameter set == test parameter set
                        match_test = np.unique(
                            np.where(
                                (
                                    test_data["x"][:, : molec_object.n_params]
                                    == test_params
                                ).all(axis=1)
                            )[0]
                        )
                        test_points = np.concatenate(
                            (
                                test_data["x"][match_test, -1].reshape(-1, 1),
                                test_data[key][match_test].reshape(-1, 1),
                            ),
                            axis=1,
                        )
                        # Find points in train set with correct param value
                        match_trn = np.unique(
                            np.where(
                                (
                                    train_data["x"][:, : molec_object.n_params]
                                    == test_params
                                ).all(axis=1)
                            )[0]
                        )
                        train_points = np.concatenate(
                            (
                                train_data["x"][match_trn, -1].reshape(-1, 1),
                                train_data[key][match_trn].reshape(-1, 1),
                            ),
                            axis=1,
                        )

                        pdf.savefig(
                            plot_model_vs_test(
                                {label: gp_model},
                                {"params": test_params},
                                train_points,
                                test_points,
                                molec_object.temperature_bounds,
                                y_bounds,
                                plot_bounds=molec_object.temperature_bounds,
                                property_name=y_names,
                            )
                        )
                        plt.close()
            pdf.close()
        # Make MAPD df
        mapd_df = pd.DataFrame(mapd_df_rows, columns=["Molecule", "Property", "MAPD"])
        mapd_df.to_csv(dir_name + "gp_mapd.csv", index=False, header=True)

        return

    def comp_paper_full_ind(self, all_molec_list, theta_guess=None, save_label=None):
        """
        Plots T vs Property for the best individual molecule param set, the best overall optimization param set,
        the experimental data, and the param set from the paper

        Parameters
        ----------
        all_molec_list: list, list of all molecules to generate predictions for
        theta_guess: np.ndarray, the atom type scheme parameter set of interest (sigma in nm, epsilon in kJ/mol)
        """
        assert all(
            item in list(self.molec_data_dict.keys()) for item in all_molec_list
        ), "all_molec_list must be a subset of the training molecules"
        assert isinstance(
            save_label, (str, type(None))
        ), "save_label must be a string or None"
        # Make pdf
        dir_name = self.make_results_dir(list(self.molec_data_dict.keys()))
        os.makedirs(dir_name / "prop_pred", exist_ok=True)
        save_label = save_label if save_label is not None else "best_set"
        pdf = PdfPages(dir_name / "prop_pred" / (save_label + ".pdf"))

        # Loop over molecules
        for molec in all_molec_list:
            # Get constants for molecule
            molec_object = self.all_train_molec_data[molec]
            # Get GPs associated with each molecule
            molec_gps_dict = self.all_train_gp_dict[molec]

            test_params = self.get_best_results(molec, theta_guess)

            # Loop over gps (1 per property)
            for key in list(molec_gps_dict.keys()):
                # Set label
                label = molec + "_" + key
                # Get GP associated with property
                gp_model = molec_gps_dict[key]
                # Get X and Y data and bounds associated with the GP
                exp_data, y_bounds, y_names = self.get_exp_data(molec_object, key)
                x_data = np.array(list(exp_data.keys()))
                y_data = np.array(list(exp_data.values()))

                # Plot test vs train for each parameter set
                pdf.savefig(
                    plot_model_vs_test(
                        {label: gp_model},
                        test_params,
                        np.array([]),
                        np.array([]),
                        molec_object.temperature_bounds,
                        y_bounds,
                        plot_bounds=molec_object.temperature_bounds,
                        property_name=y_names,
                        exp_x_data=x_data,
                        exp_y_data=y_data,
                        title=molec + " Prediction Data",
                    )
                )
                plt.close()
        pdf.close()
        return
    
    def compare_T_prop_best(self, theta_guess, all_molec_list):
        """
        Compares T vs Property for a given set

        Parameters
        ----------
        theta_guess: np.ndarray, the atom type scheme parameter set of interest (sigma in nm, epsilon in kJ/mol)
        """
        # assert isinstance(theta_guess, np.ndarray), "theta_guess must be an np.ndarray"
        assert all(
            item in list(self.molec_data_dict.keys()) for item in all_molec_list
        ), "all_molec_list must be a subset of the training molecules"

        # Make pdf
        dir_name = self.make_results_dir(list(self.molec_data_dict.keys()))
        pdf = PdfPages(os.path.join(dir_name, "prop_vs_T.pdf"))
        # Loop over molecules
        for molec in all_molec_list:
            # Get constants for molecule
            molec_object = self.all_train_molec_data[molec]
            # Get theta associated with each gp
            param_matrix = self.at_class.get_transformation_matrix(
                {molec: molec_object}
            )
            t_guesses = {}
            for t_key, t_guess in theta_guess.items():
                # Transform the guess, and scale to bounds
                gp_theta = t_guess.reshape(1, -1) @ param_matrix
                gp_theta_guess = values_real_to_scaled(
                    gp_theta, molec_object.param_bounds
                )
                t_guesses[t_key] = gp_theta_guess
            # Get GPs associated with each molecule
            molec_gps_dict = self.all_train_gp_dict[molec]
            # Loop over gps (1 per property)
            for key in list(molec_gps_dict.keys()):
                # Set label
                label = molec
                # Get GP associated with property
                gp_model = molec_gps_dict[key]
                # Get X and Y data and bounds associated with the GP
                exp_data, y_bounds, y_names = self.get_exp_data(molec_object, key)
                # Plot Exp Data vs Model Prediction
                pdf.savefig(
                    plot_model_vs_exp(
                        {label: gp_model},
                        t_guesses,
                        exp_data,
                        molec_object.temperature_bounds,
                        y_bounds,
                        plot_bounds=molec_object.temperature_bounds,
                        property_name=y_names,
                    )
                )
                plt.close()
        pdf.close()

    def __del_unphysical_hms(self, data_set_name):
        """
        Deletes unphysical heat maps
        """
        # Only create grids for legal combinations
        # Don't create grids for Cm and C parameters
        condition1 = "Cm" in data_set_name[0] and "C2" in data_set_name[1]
        condition2 = "C2" in data_set_name[0] and "Cm" in data_set_name[1]
        condition3 = "C1" in data_set_name[0] and "Cm" in data_set_name[1]
        condition4 = "Cm" in data_set_name[0] and "C1" in data_set_name[1]
        # Don't create grids for Cm and F parameters
        condition5 = "Cm" in data_set_name[0] and "F" in data_set_name[1]
        condition6 = "F" in data_set_name[0] and "Cm" in data_set_name[1]
        # Don't create grids for C1-C2 parameters
        condition7 = "C1" in data_set_name[0] and "C2" in data_set_name[1]
        condition8 = "C2" in data_set_name[0] and "C1" in data_set_name[1]
        # Don't create grids for F_4 and C2 parameters
        condition9 = "C2" in data_set_name[0] and "F_4" in data_set_name[1]
        condition10 = "F_4" in data_set_name[0] and "C2" in data_set_name[1]
        # Don't create grids for F_4 and H parameters
        condition11 = "H1" in data_set_name[0] and "F_4" in data_set_name[1]
        condition12 = "F_4" in data_set_name[0] and "H1" in data_set_name[1]
        # condition7 = 'C2_3' in data_set_name[0] and 'H' in data_set_name[1]
        # condition8 = 'H' in data_set_name[0] and 'C2_3' in data_set_name[1]
        # #Don't create grids for H or C2_0 parameters
        # condition9 = 'C2_0' in data_set_name[0] and 'F' in data_set_name[1]
        # condition10 = 'F' in data_set_name[0] and 'C2_0' in data_set_name[1]

        cond_list = [
            condition1,
            condition2,
            condition3,
            condition4,
            condition5,
            condition6,
            condition7,
            condition8,
            condition9,
            condition10,
            condition11,
            condition12,
        ]

        return cond_list

    def make_sse_sens_data(self, theta_guess):
        """
        Makes heat map data for obj predictions given a parameter set

        Parameters
        ----------
        theta_guess: np.ndarray, the atom type scheme parameter set to start optimization at (sigma in A, epsilon in K)

        Returns
        -------
        param_dict: dict, dictionary of heat map theta data
        obj_dict: dict, dictionary of heat map obj data
        """
        assert isinstance(theta_guess, np.ndarray), "theta_guess must be an np.ndarray"
        n_points = 15
        # Create dict of heat map theta data
        param_dict = {}

        # Create a linspace for the number of dimensions and define number of points
        dim_list = np.linspace(0, len(theta_guess) - 1, len(theta_guess) - 1)
        # Create a list of all combinations (without repeats e.g no (1,1), (2,2)) of dimensions of theta
        mesh_combos = np.array(list(combinations(dim_list, 2)), dtype=int)

        # Meshgrid set always defined by n_points**2
        theta_set = np.tile(np.array(theta_guess), (n_points**2, 1))

        # Loop over all possible theta combinations of 2
        for i in range(len(mesh_combos)):
            # Set the indeces of theta_set for evaluation as each row of mesh_combos
            idcs = mesh_combos[i]
            # define name of parameter set as tuple ("param_1,param_2")
            data_set_name = (
                self.at_class.at_names[idcs[0]],
                self.at_class.at_names[idcs[1]],
            )

            # Ensure unphysical parameters are filtered out
            cond_list = self.__del_unphysical_hms(data_set_name)

            # If the parameter combination is feasible, create heat map data for it
            if not any(cond_list):
                # Create a copy of the true values to change the mehsgrid valus on
                theta_set_copy = np.copy(theta_set)

                # Create a meshgrid of values of the 2 selected values of theta and reshape to the correct shape
                # Assume that theta1 and theta2 have equal number of points on the meshgrid
                theta1 = np.linspace(
                    self.at_class.at_bounds[idcs[0]][0],
                    self.at_class.at_bounds[idcs[0]][1],
                    n_points,
                )
                theta2 = np.linspace(
                    self.at_class.at_bounds[idcs[1]][0],
                    self.at_class.at_bounds[idcs[1]][1],
                    n_points,
                )
                theta12_mesh = np.array(np.meshgrid(theta1, theta2))
                theta12_vals = np.array(theta12_mesh).T.reshape(-1, 2)

                # Set initial values for evaluation (true values) to meshgrid values
                theta_set_copy[:, idcs] = theta12_vals

                # Append data set to dictionary with name
                param_dict[data_set_name] = theta_set_copy

        # Initialize obj dictionary
        obj_dict = {}
        # Loop over each heat map
        for key, value in param_dict.items():
            # Evaluate obj over data
            obj_arr = np.zeros(len(value))
            for i in range(len(value)):
                # Values pref to real
                val_real = self.values_pref_to_real(value[i])
                obj_arr[i] = self.calc_obj(val_real)[0]
            obj_dict[key] = obj_arr

        return param_dict, obj_dict

    def plot_avg_MAPD(self, mapd_dfs, df_labels, df_colors, title=None):
        fig, ax = plt.subplots()
        cols = ["mapd_liq_density", "mapd_vap_density", "mapd_Pvap", "mapd_Hvap"]
        bar_width = 0.1
        molec_names = list(mapd_dfs[0]["molecule"])
        indices = np.arange(len(mapd_dfs[0]))
        max_val_f = 0
        for i, df in enumerate(mapd_dfs):
            # Get mean max and min for MAPDs
            df["Avg MAPD"] = df[cols].mean(axis=1)
            df["Min MAPD"] = df[cols].min(axis=1)
            df["Max MAPD"] = df[cols].max(axis=1)
            average_df = df[["molecule", "Avg MAPD", "Min MAPD", "Max MAPD"]].copy()
            # Sort by molecule and set molecule as index
            average_df["molecule"] = pd.Categorical(
                average_df["molecule"],
                categories=self.valid_train_mol_keys,
                ordered=True,
            )
            average_df = average_df.sort_values(by="molecule")
            average_df.set_index("molecule")
            # Set max value so far for bounds
            max_val = float(average_df.max(numeric_only=True).max())
            max_val_f = max(max_val, max_val_f)
            # Plot min, max, and avg
            ax.bar(
                indices + i * bar_width,
                average_df["Avg MAPD"],
                bar_width,
                label=df_labels[i],
                color=df_colors[i],
            )
            maxy = average_df["Max MAPD"] - average_df["Avg MAPD"]
            miny = average_df["Avg MAPD"] - average_df["Min MAPD"]
            ax.errorbar(
                indices + i * bar_width,
                average_df["Avg MAPD"],
                yerr=[miny, maxy],
                fmt=".k",
            )

        ax.set_ylim(0, max_val_f * 1.05)
        ax.tick_params(axis="y", which="major", labelsize=16)
        ax.set_ylabel("Average MAPD", fontsize=18)
        ax.set_xticks(indices + bar_width)
        ax.set_xticklabels(molec_names, fontsize=14)
        ax.axhline(y=5, label="5% MAPD", color="black", linestyle="--")
        ax.grid()
        plt.legend(loc="upper left")
        if title:
            ax.set_title(title, fontsize=18)
        plt.tight_layout(rect=[0.01, 0.0, 1, 1])

        return fig

    def plot_MAPD(self, mapd_dfs, df_labels, df_colors, title):
        cols = ["mapd_liq_density", "mapd_vap_density", "mapd_Pvap", "mapd_Hvap"]
        names = [
            "Liquid Density",
            "Vapor Density",
            "Vapor Pressure",
            "Heat of Vaporization",
        ]
        molec_names = list(mapd_dfs[0]["molecule"])
        df_mse_list = []
        for mapd_df in mapd_dfs:
            df_mse_list.append(mapd_df.set_index("molecule"))

        fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        # Plot each column in a subplot
        for ax, column, name in zip(axs.flatten(), cols, names):
            bar_width = 0.1
            indices = np.arange(len(molec_names))
            max_val_f = 0
            for i in range(len(df_mse_list)):
                df = df_mse_list[i]
                if i < len(df_mse_list):
                    max_val = max(df[column])
                    max_val_f = max(max_val, max_val_f)
                ax.bar(
                    indices + i * bar_width,
                    df[column],
                    bar_width,
                    label=df_labels[i],
                    color=df_colors[i],
                )

            # ax.set_ylim(0, np.maximum(max_val_f * 1.05, 5))
            ax.set_ylim(0, max_val_f * 1.05)
            ax.tick_params(axis="y", which="major", labelsize=16)
            ax.set_title(name, fontsize=18)
            ax.set_xticks(indices + bar_width)
            ax.set_xticklabels(molec_names, fontsize=14)
            ax.axhline(y=5, label="5% MAPD", color="black", linestyle="--")
            if name == "Liquid Density":
                # ax.legend(loc="upper right", fontsize=18)
                handles, labels = ax.get_legend_handles_labels()
            ax.grid()

        # Adjust layout
        fig.supxlabel("Molecule", fontsize=20)
        fig.supylabel("MAPD", fontsize=20)
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=4, fontsize = 18)
        if title:
            fig.suptitle(title, fontsize=18)
        plt.tight_layout(rect=[0.01, 0.0, 1, 1])
        # Show the plot
        return fig

    def plot_obj_hms(self, theta_guess, set_label=None):
        """
        Plots objective contours given a set of data

        Parameters
        ----------
        theta_guess: np.ndarray, the atom type scheme parameter set to start optimization at (sigma in A, epsilon in K)
        """
        assert isinstance(theta_guess, np.ndarray), "theta_guess must be an np.ndarray"
        # Get HM Data
        param_dict, obj_dict = self.make_sse_sens_data(theta_guess)
        # Make pdf
        dir_name = self.make_results_dir(list(self.molec_data_dict.keys()))
        os.makedirs(dir_name / "obj_cont", exist_ok=True)
        set_label = set_label if set_label is not None else "best"
        pdf = PdfPages(dir_name / "obj_cont" / (set_label + ".pdf"))
        # Loop over keys
        for key in list(param_dict.keys()):
            # Get parameter and sse data
            theta_vals = param_dict[key]
            # Get index associated with params in key
            indcs = [
                self.at_class.at_names.index(key[0]),
                self.at_class.at_names.index(key[1]),
            ]
            n_points = int(np.sqrt(len(theta_vals)))
            obj_vals = obj_dict[key].reshape((n_points, n_points)).T
            # Remove unchanging columns
            theta_mesh = theta_vals[:, indcs].reshape((n_points, n_points, -1)).T
            pdf.savefig(plot_obj_contour(theta_mesh, theta_guess[indcs], obj_vals, key))
            plt.close()
        pdf.close()
