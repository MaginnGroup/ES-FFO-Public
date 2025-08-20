import signac
import sys
import os
from pathlib import Path
from file_read_backwards import FileReadBackwards
import matplotlib
import matplotlib.pyplot as plt

root_path = Path(__file__).resolve().parents[1]  # ES-FFO directory (two levels up from this script)
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

# Now import using package structure relative to ES-FFO root
from utils.molec_class_files import esolvs
from Build_GPs.utils.signac import get_signac_results, save_signac_results
from Build_GPs.utils.id_new_samples import new_samples_vle, find_pareto
from Build_GPs.utils.models import get_best_models
from Build_GPs.utils.plot import plot_gp_examples

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
    have_sims = True
    count = 0
    sweeps = []
    Exchanges = []

    while have_sims is True:
        insert_val = None
        delete_val = None
        
        #Find the log file for each restart
        if count == 0:
            filename = job.fn("gemc.eq.out.log")
        else:
            filename = job.fn(f"gemc.eq.rst.{count:03d}.out.log")

        
        #If we're out of restarts break the loop
        if not os.path.exists(filename):
            have_sims = False
            break
   
        #Otherwise, find the last instance of the words insert and deleter starting from the bottom of the file
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
        N_exc = delete_val + insert_val

        #Input file is file_name before .out.log
        fn_input = filename.replace(".out.log", "")
        run_value, steps_per_sweep = count_steps(fn_input)
        sweeps.appen(run_value)
        Exchanges.append(N_exc)
    
    #Plot
    plt.plot(sweeps, Exchanges, label=job.id)
    plt.xlabel("Sweeps")
    plt.ylabel("Exchanges")
    plt.title("Exchanges vs Sweeps")
    plt.legend()
    #Save figure to job directory
    plt.savefig(job.fn("Exc_vs_Sweeps.png"))
    plt.close()

#Other analysis here