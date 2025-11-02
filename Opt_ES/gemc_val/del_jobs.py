import signac
import glob
import os
import shutil

# Load the project
project = signac.get_project()
count = 0
#Loop over all jobs in the project
for job in project.find_jobs({"mol_name": "DEC"}):
    # Move old results to a subfolder
    count += 1
    job.remove()
    # with job:
    #     if job.sp.atom_type > 0:
    #         job.remove()
print(count)
