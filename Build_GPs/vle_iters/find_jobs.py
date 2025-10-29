import signac
import glob
import os
import shutil

# Load the project
project = signac.get_project()

count = 0
count_running = 0
# Loop over all jobs in the project
group = project.find_jobs({"mol_name": "DEC", "iter": 1})
for job in group:
    # Check if job document exists
    # print(f"Checking job {job.id}")
    if os.path.exists(job.fn("signac_job_document.json")):
        count_running += 1
        # if "eq_liq_dens" in job.document and job.doc["eq_liq_dens"] < 1.0:
        #     print(f"Job {job.id} has eq_liq_dens < 1.0: {job.doc['eq_liq_dens']}")
        op_name = "npzzat_eq"
        if f"{op_name}_fin" not in job.doc and "ld_fail" not in job.doc:
            count += 1
            print(job.id)
            
            #Print the last line of the run_npt_prod.out log file
            log_file = job.fn(f"run_{op_name}.out")
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    lines = f.readlines()
                    if lines:
                        last_line = lines[-1].strip()
                        if "Thu" in last_line:
                            print(last_line)
                            print(f"Job {job.id}")
            else:
                print(f"Job {job.id} Log file does not exist.")
print(f"Total unfinished jobs: {count}/{count_running}")
