import signac
import glob
import os
import shutil

# Load the project
project = signac.get_project()

#Loop over all jobs in the project
for job in project.find_jobs():
    # Move old results to a subfolder
    with job:
        if job.sp.atom_type > 0:
            job.remove()
