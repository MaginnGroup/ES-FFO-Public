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

import re
import glob

def extract_core_time(logfile):
    """Extract the Core t (s) from a GROMACS log file."""
    pattern = re.compile(r"^\s*Time:\s+([0-9.]+)")
    core_time = None
    with open(logfile, "r") as f:
        for line in f:
            match = pattern.match(line)
            if match:
                core_time = float(match.group(1))  # Core t in seconds
    return core_time

# Path to your log files (adjust the pattern)
total_core_time_s = 0.0

#Get all signac jobs
project_path = "ift_val_opt"
project_all = signac.get_project(project_path)
for job in project_all:
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

