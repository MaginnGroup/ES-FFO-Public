import signac
import glob
import os
import shutil

# Load the project
project = signac.get_project()

#Loop over all jobs in the project
for job in project.find_jobs({"mol_name":"DMSO", "iter": 1}):
    job.remove()  # Remove the job from the project
# for job in project:
#     # Check if ld_fail exists in the job document
#     if "ld_fail" in job.document:
#         for fname in os.listdir(job.path):
#             if fname != "signac_statepoint.json":
#                 fpath = os.path.join(job.path, fname)
#                 if os.path.isfile(fpath) or os.path.islink(fpath):
#                     os.remove(fpath)
#                 elif os.path.isdir(fpath):
#                     shutil.rmtree(fpath)
#                     print(f"Skipping directory: {fpath}")