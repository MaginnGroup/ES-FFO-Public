from flow import FlowProject, directives
import warnings
from pathlib import Path
import os
import glob
import sys
import unyt as u
import copy
from pymser import pymser
import numpy as np
import matplotlib.pyplot as plt
import re
import signac
from pathlib import Path
from collections import defaultdict
from pathlib import Path
import shutil

project_paths = ["gemc_val_opt", "gemc_val_no_opt", "ift_val_opt", "ift_val_no_opt"]
mol_names = ["EG" , "Gly", "MeOH", "DMSO", "DEC", "DMF"] #["EG" , "Gly", "MeOH", "DMSO", "DEC", "DMF"]

for project_path in project_paths:
    project_all = signac.get_project(project_path)
    for mol_name in mol_names:
        for job in project_all.find_jobs({"mol_name": mol_name}):
            #For the first job, get the ff.xml file path and create the ff_mcf folder
            dir_name = f"FF_params/{mol_name}"
            os.makedirs(dir_name, exist_ok=True)
            ff_xml_path = job.fn("ff.xml")
            #Add ff.xml to directory with the dir_name if it doesn't already exist
            if not os.path.exists(os.path.join(dir_name, "ff.xml")):
                shutil.copy(ff_xml_path, os.path.join(dir_name, "ff.xml"))

            if "ift" in project_path and not os.path.exists(os.path.join(dir_name, "system.top")):
                system_file = job.fn("system.top")
                shutil.copy(system_file, os.path.join(dir_name, "system.top"))
            else:
                mcf_file = job.fn("species1.mcf")
                pdb_file = job.fn("species1.pdb")
                if not os.path.exists(os.path.join(dir_name, "ff.xml")):
                    shutil.copy(ff_xml_path, os.path.join(dir_name, "ff.xml"))
                if not os.path.exists(os.path.join(dir_name, "species1.mcf")):
                    shutil.copy(mcf_file, os.path.join(dir_name, "species1.mcf"))
                if not os.path.exists(os.path.join(dir_name, "species1.pdb")):
                    shutil.copy(pdb_file, os.path.join(dir_name, "species1.pdb"))
            break           
            
