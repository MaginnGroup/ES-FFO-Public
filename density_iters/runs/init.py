import signac
import numpy as np
import unyt as u
from scipy.stats import qmc
import pandas as pd
import csv
import glob

from fffit.utils import values_scaled_to_real
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

# Load class properies for each training and testing molecule
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


def _get_molec_dicts():
    # Load class properies for each molecule
    from utils.molec_class_files import r41  # import all the class files

    R41 = r41.R41Constants()

    # Create a dictionary with all of the data
    molec_dict = {
        "R41": R41,
    }
    return molec_dict


def _get_class_from_molecule(molecule_name):
    molec_dict = _get_molec_dicts()
    return {molecule_name: molec_dict[molecule_name]}


def unpack_molec_values(class_data, state_point, sample):
    """
    Unpacks sckaled sample values given the molecule under study
    """
    param_names = class_data.param_names
    for i, param in enumerate(param_names):
        if "sigma" in param:
            state_point[param] = float((sample[i] * u.Angstrom).in_units(u.nm).value)
        elif "epsilon" in param:
            state_point[param] = float(
                (sample[i] * u.K * u.kb).in_units("kJ/mol").value
            )
        state_point[param] = sample[i]

    return state_point


def determine_density_iter(molec_name):
    # Check the analysis folder for analysis/MolName/density-iter-X folders
    # Find the highest density-iter-X folder
    files = sorted(glob.glob("analysis/" + molec_name + "/density-iter-*"))
    if len(files) == 0:
        dens_iter = 1
    else:
        # Get the highest density-iter-X folder from the last character of the last file
        dens_iter = files[-1][-1]
    return dens_iter


nsteps_nvt1 = 100000  # 100ps
nsteps_npt = 500000  # 500ps (minimum)
nsteps_nvt2 = 100000  # 100ps
nsteps_intereq = 25000000  # 25 ns (minimum)
nsteps_interprod = 50000000  # 50 ns


def init_project():
    # Loop over all molecules
    for molec_name, molec_data in molec_dict.items():
        # Determine Density iter based off of the analysis folder
        # For now use dens-iter = 1
        dens_iter = determine_density_iter(molec_name)

        # Initialize project
        project = signac.init_project("dens-iter-" + str(dens_iter))

        # Use GenLHS samples to generate LHS samples in the analysis folder
        # Load the lhs_samples and bounds
        class_dict = _get_class_from_molecule(job.sp.mol_name)
        class_data = class_dict[job.sp.mol_name]
        bounds = class_data.param_bounds
        if dens_iter == 1:
            lhs_samples = pd.read_csv("analysis/" + molec_name + "/LHS_200.csv")
        else:
            lhs_samples = pd.read_csv(
                "analysis/" + molec_name + "/dens-iter-" + str(dens_iter) + ".csv"
            )

        # Convert scaled latin hypercube samples to physical values
        scaled_params = values_scaled_to_real(lhs_samples, bounds)

        # Define temps (from constants files)
        temps = list(molec_data.expt_Pvap.keys())
        for temp in temps:
            liq_density = molec_data.expt_liq_density[int(temp)]
            vap_density = molec_data.expt_vap_density[int(temp)]
            avg_density = (liq_density + vap_density) / 2
            for sample in scaled_params:
                # Define the state point w/ unchanging characteristics
                state_point = {
                    "mol_name": molec_name,
                    "density-iter": dens_iter,
                    "smiles": molec_data.smiles_str,
                    "T": float(temp.in_units(u.K).value),
                    "P": float(class_data.expt_Pvap[temp]),
                    "rho_avg": avg_density,  # kg/m^3
                    "nsteps_nvt1": nsteps_nvt1,
                    "nsteps_npt": nsteps_npt,
                    "nsteps_nvt2": nsteps_nvt2,
                    "nsteps_intereq": nsteps_intereq,
                    "nsteps_interprod": nsteps_interprod,
                }
                state_point = unpack_molec_values(class_data, state_point, sample)

                job = project.open_job(state_point)
                job.init()


if __name__ == "__main__":
    init_project()
