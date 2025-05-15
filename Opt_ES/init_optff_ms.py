import signac
import numpy as np
import unyt as u
import sys
import pandas as pd

sys.path.append("..")
from utils.molec_class_files import esolvs
from utils import atom_type, opt_atom_types
sys.path.remove("..")

# Load class properies for each training molecule
mol_names = ["EG" , "Gly", "ACN", "MeOH", "DMSO", "THF", "DCM", "DEC", "DMF"]
molec_dict = esolvs.make_dict(mol_names)

at_numbers = [1,2] #1,2
num_restarts = 3  # Number of restarts for replications
n_vap = 160  # number of molecules in vapor phase
n_liq = 640
obj_choice = "ExpVal"  # Objective to consider

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
    return state_point


# Loop over all molecules
for molec_name, molec_data in molec_dict.items():
    # Define temps (from constants files)
    temps = list(molec_data.expt_Pvap.keys())

    # Run at vapor pressure (from constants file)
    press = molec_data.expt_Pvap

    # Load sample from best set using ExpVal and all training molecules
    save_data = False
    molec_names = mol_names  # Training data to consider

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
