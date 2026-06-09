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

def extract_time_from_log(filepath):
    time_dict = {}

    pattern = re.compile(r"\s*(\d+)\s+(Years|Months|Days|Hours|Minutes|Seconds|ms)")

    with open(filepath, "r") as f:
        for line in f:
            match = pattern.match(line)
            if match:
                value = int(match.group(1))
                unit = match.group(2)
                time_dict[unit] = value

    return time_dict

total_time = defaultdict(int)

#Get all signac jobs
project_path = "gemc_val_opt"
project_all = signac.get_project(project_path)
for job in project_all:
    #Get all files with ".log"
    files = last_prod_file = sorted(glob.glob(job.fn("*.out.log")))
    for file in files:
        t = extract_time_from_log(file)

        for key, value in t.items():
            total_time[key] += value

time_dict = dict(total_time)
print(time_dict)

def convert_to_years(t):
    seconds = (
        t.get("Years", 0) * 365 * 24 * 3600 +
        t.get("Months", 0) * 30 * 24 * 3600 +
        t.get("Days", 0) * 24 * 3600 +
        t.get("Hours", 0) * 3600 +
        t.get("Minutes", 0) * 60 +
        t.get("Seconds", 0) +
        t.get("ms", 0) / 1000
    )

    years = seconds / (365 * 24 * 3600)
    return years


years_total = convert_to_years(time_dict)

print("Total years:", years_total)