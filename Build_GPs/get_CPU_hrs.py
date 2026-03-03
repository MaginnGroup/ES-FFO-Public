import re
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
from tqdm import tqdm

import re
import glob

def extract_core_time(logfile):
    """Extract the last Core t (s) from a GROMACS log file, reading from the bottom."""
    pattern = re.compile(r"^\s*Time:\s+([0-9.]+)")
    
    with open(logfile, "r") as f:
        for line in reversed(list(f)):  # read file from bottom
            match = pattern.match(line)
            if match:
                return float(match.group(1))  # Core t in seconds
    return None

# Path to your log files (adjust the pattern)
total_core_time_s = 0.0

#Get all signac jobs
project_path = "vle_iters"
project_all = signac.get_project(project_path)
for job in tqdm(project_all):
    #Get all files with ".log"
    files = last_prod_file = sorted(glob.glob(job.fn("*/*.log")))
    for file in files:
        core_time = extract_core_time(file)
        if core_time is not None:
            total_core_time_s += core_time

# Convert seconds to years
seconds_per_year = 365 * 24 * 3600
total_years = total_core_time_s / seconds_per_year

print(f"Total Core time: {total_core_time_s:.2f} seconds")
print(f"Which is approximately {total_years:.4f} years")

