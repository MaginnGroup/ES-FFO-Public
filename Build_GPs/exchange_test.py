import signac
import sys
import os
from pathlib import Path
from file_read_backwards import FileReadBackwards
import glob

root_path = Path(__file__).resolve().parents[1]  # ES-FFO directory (two levels up from this script)
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

# Now import using package structure relative to ES-FFO root

print(f"Current working dir: {os.getcwd()}")
print(f"Script location: {Path(__file__).parent}")

def count_steps(fpath):
    with open(fpath + ".inp", "r") as file:
        in_sim_length_info = False
        for line in file:
            # Search for the line starting with "run" in "Simulation_Length_Info"
            if "Simulation_Length_Info" in line:
                # Enter the relevant section
                in_sim_length_info = True

            if in_sim_length_info and line.strip().startswith("run"):
                # Extract the run value
                run_value = int(line.split()[1])

            if in_sim_length_info and line.strip().startswith("steps_per_sweep"):
                steps_per_sweep = int(line.split()[1])
                break
    return run_value, steps_per_sweep

project = signac.get_project("vle_val")

for job in project:
    insert_val = None
    delete_val = None
    
    #Initialize log file as production file
    filename = job.fn("prod.out.log")

    #Use the last eq restart instead if it exists
    last_gemc_eq_file = sorted(glob.glob(job.fn(f"gemc.eq.rst.*.out.log")))
    if len(last_gemc_eq_file) > 0:
        filename = last_gemc_eq_file[-1]

    #Find the last instance of the words insert and delete starting from the bottom of the file
    # Read file lines
    with FileReadBackwards(filename, encoding="utf-8") as frb:
        for l in frb:
            line = frb.readline()
            if "Delete" in l:
                # Split the last line and extract the first number
                delete_val = int(line.split()[2])
            elif "Insert" in l:
                # Split the last line and extract the first number
                insert_val = int(line.split()[2])
            if insert_val == None:
                break
    N_mols = job.sp.N_vap + job.sp.N_liq
    pct_diff = abs(insert_val - delete_val) / insert_val * 100
    job.doc["insert_val"] = insert_val
    job.doc["delete_val"] = delete_val
    job.doc["pct_diff"] = pct_diff
    #Check that the insert and delete values are within 5% of each other
    if pct_diff > 5:
        print(f"Warning: Large difference between insert and delete counts for job {job.id}")
        print(f"Insert: {insert_val}, Delete: {delete_val}, Percent Difference: {pct_diff:.2f}%")
    #Check that he number of insertions and deletions are at least equal to the number of molecules
    if insert_val < N_mols or delete_val < N_mols:
        print(f"Warning: Low number of insertions or deletions for job {job.id}")
        print(f"Insert: {insert_val}, Delete: {delete_val}, N_mols: {N_mols}")