from scipy.stats import qmc
import pandas as pd
import csv


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


seed = 7  # Seed of data
n = 200  # Number of points to generate


for molec_name in molec_dict.keys():
    class_dict = _get_class_from_molecule(molec_name)
    class_data = class_dict[molec_name]
    d = class_data.n_params  # Number of dimensions
    sampler = qmc.LatinHypercube(d, seed=seed)
    sample = sampler.random(n)
    lhs_samples = pd.DataFrame(sample)
    lhs_samples.columns = list(class_data.param_names)
    filename = "analysis/" + molec_name + "/dens-iter-1.csv"
    lhs_samples.to_csv(filename, index=True)
