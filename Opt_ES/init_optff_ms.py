import signac
import numpy as np
import unyt as u
import sys
import pandas as pd

from utils.molec_class_files import (
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
from utils import atom_type, opt_atom_types

at_numbers = [1,2,6,8] #1,2,6,8
num_restarts = 3  # Number of restarts for replications
n_vap = 160  # number of molecules in vapor phase
n_liq = 640
obj_choice = "ExpVal"  # Objective to consider

R14 = r14.R14Constants()
R32 = r32.R32Constants()
R50 = r50.R50Constants()
R125 = r125.R125Constants()
R134a = r134a.R134aConstants()
R143a = r143a.R143aConstants()
R170 = r170.R170Constants()
R41 = r41.R41Constants()
R23 = r23.R23Constants()
R161 = r161.R161Constants()
R152a = r152a.R152aConstants()
R152 = r152.R152Constants()
R143 = r143.R143Constants()
R134 = r134.R134Constants()
R116 = r116.R116Constants()

molec_dict = {
    "R14": R14,
    "R32": R32,
    "R50": R50,
    "R125": R125,
    "R134a": R134a,
    "R143a": R143a,
    "R170": R170,
    "R41": R41,
    "R23": R23,
    "R161": R161,
    "R152a": R152a,
    "R152": R152,
    "R143": R143,
    "R134": R134,
    "R116": R116,
}

# Initialize project
project = signac.init_project("opt_ff_ms")


def unpack_molec_values(molec_name, at_class, sample, state_point):
    """
    Unpacks sckaled sample values given the molecule under study
    """
    # Unpack the sample according to atom typing scheme mapping dictionary
    molec_map_dict = at_class.molec_map_dicts[molec_name]
    param_names = molec_map_dict.keys()
    # Get param names in order of original mapping
    order = molec_data.param_names
    index_mapping = {elem: idx for idx, elem in enumerate(order)}
    # Add params based on the order they show up in given the mapping
    for param in param_names:
        state_point[param] = sample[index_mapping[param]].item()

    #For R23 AT Scheme 2, use the GAFF parameters for H since these parameters aren't identifiable
    if molec_name == "R23" and at_class.scheme_name == "at_02":
        state_point["epsilon_H1"] = float(7.901 * (u.K * u.kb).in_units("kJ/mol"))
        state_point["sigma_H1"] = float((2.115 * u.Angstrom).in_units(u.nm).value)
    # For R41 and R23 add the GAFF Parameter for epsilon C1 instead of the OptFF one
    # if molec_name in ["R41", "R23"]:
    #     state_point["epsilon_C1"] = float(55.052 * (u.K * u.kb).in_units("kJ/mol"))
    #     state_point["GAFF_epsilon_C1"] = True
    # print(state_point)
    return state_point


# Loop over all molecules
for molec_name, molec_data in molec_dict.items():
    # Define temps (from constants files)
    temps = list(molec_data.expt_Pvap.keys())

    # Run at vapor pressure (from constants file)
    press = molec_data.expt_Pvap

    # Load sample from best set using ExpVal and all training molecules
    save_data = False
    molec_names = [
        "R14",
        "R32",
        "R50",
        "R170",
        "R125",
        "R134a",
        "R143a",
        "R41",
    ]  # Training data to consider
    # molec_data_dict = {"R14":R14, "R32":R32, "R50":R50, "R170":R170, "R125":R125, "R134a":R134a, "R143a":R143a}
    # all_gp_dict = opt_atom_types.get_gp_data_from_pkl(list(molec_data_dict.keys()))
    for at_number in at_numbers:
        setup = opt_atom_types.Problem_Setup(molec_names, at_number, obj_choice)
        all_molec_dir = setup.use_dir_name
        all_df = pd.read_csv(all_molec_dir / "unique_best_set.csv", header=0)

        # Loop over best molecules
        for i in range(1):
            full_opt_best = all_df.iloc[i].values
            # Convert to units of nm and kJ/mol
            param_matrix = setup.at_class.get_transformation_matrix(
                {molec_name: molec_data}
            )
            all_best_real = setup.values_pref_to_real(full_opt_best)
            # Parameters in units nm and kJ/mol
            scaled_params = all_best_real.reshape(-1, 1).T @ param_matrix

            for restart in range(num_restarts):
                # Loop over temperatures
                for temp in temps:
                    # Theoretically, we could examine more than just the best
                    for sample in scaled_params:
                        # Define the initial state point
                        state_point = {
                            "atom_type": at_number,
                            "obj_choice": obj_choice,
                            "mol_name": molec_name,
                            "mol_weight": molec_data.molecular_weight,  # amu
                            "smiles": molec_data.smiles_str,
                            "N_atoms": molec_data.n_atoms,
                            "T": float(temp),  # K
                            "P": float(press[int(temp)]),  # bar
                            "N_vap": n_vap,
                            "N_liq": n_liq,
                            "expt_liq_density": molec_data.expt_liq_density[
                                int(temp)
                            ],  # kg/m^3
                            "nsteps_nvt": 2500,
                            "nsteps_npt": 5000,
                            "nsteps_gemc_eq": 10000, #We will actually be using sweeps and not steps as units here
                            "nsteps_gemc_prod": 100000, #Sweeps
                            "restart": restart + 1,
                            "param_set": i + 1,
                        }
                        state_point = unpack_molec_values(
                            molec_name, setup.at_class, sample, state_point
                        )                

                        # print(state_point)
                        job = project.open_job(state_point)
                        job.init()
                        # Add weights to job document
                        if obj_choice == "ExpValPrior":
                            job.doc.weights = setup.at_class.at_weights
                            job.doc.wt_params = setup.at_class.weighted_params
