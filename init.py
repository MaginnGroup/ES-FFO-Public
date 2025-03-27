import signac
import numpy as np
import unyt as u
from scipy.stats import qmc
import pandas as pd
import csv

from fffit.fffit.utils import values_scaled_to_real
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


nsteps_nvt1 = 100000  # 100ps
nsteps_npt = 500000  # 500ps (minimum)
nsteps_nvt2 = 100000  # 100ps
nsteps_intereq = 25000000  # 25 ns (minimum)
nsteps_interprod = 50000000  # 50 ns


def init_project():
    # Initialize project
    project = signac.init_project("ES_FF")
    # Loop over all molecules
    for molec_name, molec_data in molec_dict.items():
        # Define temps (from constants files)
        temps = list(molec_data.expt_Pvap.keys())

        ##FIGURE OUT HOW I WANT TO DO THIS. Consider a script to make these before so they don't change
        # Get number of parameters from molecule class
        class_dict = _get_class_from_molecule(job.sp.mol_name)
        class_data = class_dict[job.sp.mol_name]
        d = class_data.num_params  # Number of dimensions
        seed = 7
        n = 200
        sampler = qmc.LatinHypercube(d, seed=seed)
        lh_samples = sampler.random(n)
        bounds = class_data.param_bounds

        # Save the samples to a csv file
        sample = pd.DataFrame(lh_samples)
        sample.columns = class_data.param_names
        filename = "LHS_" + str(n) + "_x_" + str(d) + ".csv"
        sample.to_csv(filename, index=True)

        # Convert scaled latin hypercube samples to physical values
        scaled_params = values_scaled_to_real(lh_samples, bounds)

        for temp in temps:
            liq_density = molec_data.expt_liq_density[int(temp)]
            vap_density = molec_data.expt_vap_density[int(temp)]
            avg_density = (liq_density + vap_density) / 2
            for sample in scaled_params:
                # Define the state point w/ unchanging characteristics
                state_point = {
                    "mol_name": molec_name,
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
