from scipy.stats import qmc
import pandas as pd
import os
import sys

sys.path.append("..")
from utils.molec_class_files import esolvs
sys.path.remove("..")

# Load class properies for each training molecule
mol_names = ["EG", "Gly", "ACN", "MeOH", "DMSO", "THF", "DCM", "DEC", "DMF", "R125"]
molec_dict = esolvs.make_dict()

seed = 7  # Seed of data
for n in [200, 500000]:  # Number of points to generate
    for molec_name in molec_dict.keys():
        class_data = molec_dict[molec_name]
        d = class_data.n_params  # Number of dimensions
        sampler = qmc.LatinHypercube(d, seed=seed)
        sample = sampler.random(n)
        lhs_samples = pd.DataFrame(sample)
        lhs_samples.columns = list(class_data.param_names)
        os.makedirs("analysis/" + molec_name, exist_ok=True)
        if n == 200:
            os.makedirs("analysis/" + molec_name + "/ld_iters", exist_ok=True)
            filename = f"analysis/{molec_name}/ld_iters/params-iter-1.csv"
        else:
            filename = "analysis/" + molec_name + "/LHS_500000.csv"
        lhs_samples.to_csv(filename, index=True)
