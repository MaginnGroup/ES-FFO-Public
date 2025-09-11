from flow import FlowProject, directives
import templates.ndcrc
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
import signac
from file_read_backwards import FileReadBackwards

# simulation_length must be consistent with the "units" field in custom args below
# For instance, if the "units" field is "sweeps" and simulation_length = 1000,
# This will run a total of 1000 sweeps
# (1 sweep = N steps, where N is the total number of molecules (job.sp.N_vap + job.sp.N_liq)
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),  "..", ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from utils.molec_class_files import esolvs

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Note - Must define Project class with a different name that other project.py files
class ProjectGEMC(FlowProject):
    pass
    # def __init__(self):
    #     current_path = Path(os.getcwd()).absolute()
    #     # Set Project Path to be that of the current working directory
    #     super().__init__(path=current_path)

nptnvt_group = ProjectGEMC.make_group(name="NPT_NVT")
eq_group = ProjectGEMC.make_group(name="EQ")
prod_group = ProjectGEMC.make_group(name="PROD")

@nptnvt_group
@ProjectGEMC.post.isfile("ff.xml")
@ProjectGEMC.operation
def create_forcefield(job):
    """Create the forcefield .xml file for the job"""
    # Generate content based on job sp molecule name
    molec_xml_function = _get_xml_from_molecule(job.sp.mol_name)
    content = molec_xml_function(job)

    with open(job.fn("ff.xml"), "w") as ff:
        ff.write(content)


def nvt_finished(job):
    "Confirm a given simulation is completed"
    import numpy as np
    import os

    with job:
        try:
            thermo_data = np.genfromtxt("nvt.eq.out.prp", skip_header=3)
            completed = (
                int(thermo_data[-1][0]) == job.sp.nsteps_nvt
            )  # job.sp.nsteps_liqeq
        except:
            completed = False
            pass

    return completed


def npt_finished(job):
    "Confirm a given simulation is completed"
    import numpy as np
    import os

    with job:
        try:
            thermo_data = np.genfromtxt("npt.eq.out.prp", skip_header=3)
            completed = (
                int(thermo_data[-1][0]) == job.sp.nsteps_npt
            )  # job.sp.nsteps_liqeq
        except:
            completed = False
            pass

    return completed

@ProjectGEMC.label
def gemc_prod_complete(job):
    "Confirm gemc production has completed"
    import numpy as np

    try:
        # Get the last production restart file
        last_prod_file = sorted(glob.glob(job.fn("prod.*out.box1.prp")))[-1]
        # with open(job.fn("prod.out.box1.prp"), "rb") as f:
        with open(last_prod_file, "rb") as f:
            # Move the pointer to the end of the file, but leave space to find the last line
            f.seek(-2, os.SEEK_END)
            # Read backward until a newline is found
            while f.read(1) != b"\n":
                f.seek(-2, os.SEEK_CUR)
            # Read the last line after finding the newline
            last_line = f.readline().decode()
        # Split the last line and extract the first number
        first_value = int(last_line.split()[0])
        completed = first_value == job.doc.nsteps_gemc_eq + job.sp.nsteps_gemc_prod
    except:
        completed = False
        pass

    return completed


def calc_box_helper(job):
    "Calculate the initial box length of the boxes"

    import unyt as u

    # Get reference data from constants file
    # Load class properies for each training and testing molecule
    class_dict = _get_class_from_molecule(job.sp.mol_name)
    class_data = class_dict[job.sp.mol_name]
    # Reference data to compare to (i.e. experiments or other simulation studies) (load from constants file in ProjectGEMC_gaff.py as needed)
    # Loop over the keys of the dictionaries
    ref = {}
    # What is the best way to automate this if exp data crashes simulation?
    for t in class_data.expt_Pvap.keys():
        # Initialize rho_liq and rho_vap as the experimental values
        rho_liq = class_data.expt_liq_density[t] * u.kilogram / (u.meter) ** 3
        rho_vap = class_data.expt_vap_density[t] * u.kilogram / (u.meter) ** 3
        # If the gemc simulation failed previously, use the critical values
        if job.doc.get("use_crit", False):
            rho_liq = class_data.expt_rhoc * u.kilogram / (u.meter) ** 3
            rho_vap = class_data.expt_rhoc * u.kilogram / (u.meter) ** 3
        p_vap = class_data.expt_Pvap[t] * u.bar
        # Create a tuple containing the values from each dictionary
        ref[t] = (rho_liq, rho_vap, p_vap)

    vap_density = ref[job.sp.T][1]
    mol_density = vap_density / (job.sp.mol_weight * u.amu)
    vol_vap = job.sp.N_vap / mol_density
    vapboxl = vol_vap ** (1.0 / 3.0)

    # Strip unyts and round to 0.1 angstrom
    vapboxl = round(float(vapboxl.in_units(u.nm).to_value()), 2)

    # Save to job document file
    job.doc.vapboxl = vapboxl  # nm, compatible with mbuild

    liq_density = ref[job.sp.T][0]
    mol_density = liq_density / (job.sp.mol_weight * u.amu)
    vol_liq = job.sp.N_liq / mol_density
    liqboxl = vol_liq ** (1.0 / 3.0)

    # Strip unyts and round to 0.1 angstrom
    liqboxl = round(float(liqboxl.in_units(u.nm).to_value()), 2)

    # Save to job document file
    job.doc.liqboxl = liqboxl  # nm, compatible with mbuild

    return job.doc.liqboxl, job.doc.vapboxl

@nptnvt_group
@ProjectGEMC.post(lambda job: "vapboxl" in job.doc)
@ProjectGEMC.post(lambda job: "liqboxl" in job.doc)
@ProjectGEMC.operation
def calc_boxes(job):
    "Calculate the initial box length of the boxes"
    liqbox, vapbox = calc_box_helper(job)

@nptnvt_group
@ProjectGEMC.pre.after(calc_boxes)
@ProjectGEMC.pre(lambda job: "gemc_failed" not in job.doc)
@ProjectGEMC.post(nvt_finished)
@ProjectGEMC.operation(directives={"omp_num_threads": 4})
def NVT_liqbox(job):
    "Equilibrate the liquid box using NVT simulation"

    import os
    import errno
    import mbuild
    import foyer
    import mosdef_cassandra as mc
    import unyt as u

    ff = foyer.Forcefield(job.fn("ff.xml"))

    # Load the compound and apply the ff
    compound = mbuild.load(job.sp.smiles, smiles=True)
    compound_ff = ff.apply(compound)

    # Create a new moves object and species list
    species_list = [compound_ff]
    moves = mc.MoveSet("nvt", species_list)

    # Property outputs relevant for NPT simulations
    thermo_props = ["energy_total", "pressure"]

    custom_args, custom_args_gemc = _get_custom_args(job)
    custom_args["run_name"] = "nvt.eq"
    custom_args["properties"] = thermo_props
    mols_to_add = [[job.sp.N_liq]]

    # Create box list
    boxl = job.doc.liqboxl
    box = mbuild.Box(lengths=[boxl, boxl, boxl])
    box_list = [box]
    system = mc.System(box_list, species_list, mols_to_add=mols_to_add)

    # Try to run the NVT simulation with experimental starting conditions
    try:
        with job:
            # Run equilibration
            mc.run(
                system=system,
                moveset=moves,
                run_type="equilibration",
                run_length=job.sp.nsteps_nvt,
                temperature=job.sp.T * u.K,
                **custom_args,
            )

            if "use_crit" not in job.doc:
                job.doc.use_crit = False

            job.doc["nvt_fin"] = True
    #If it doesn't work, try with critical point starting conditions
    except:
        # Note this overwrites liquid and vapor box lengths in job.doc
        liqbox, vapbox = calc_box_helper(job)
        # Create system with box lengths based on critical points
        boxl = job.doc.liqboxl
        box = mbuild.Box(lengths=[boxl, boxl, boxl])
        box_list = [box]
        system = mc.System(box_list, species_list, mols_to_add=mols_to_add)

        try:
            with job:
                job.doc.use_crit = True
                # Run equilibration
                mc.run(
                    system=system,
                    moveset=moves,
                    run_type="equilibration",
                    run_length=job.sp.nsteps_nvt,
                    temperature=job.sp.T * u.K,
                    **custom_args,
                )
                
                job.doc["nvt_fin"] = True
        #Otherwise this job has failed
        except:
            job.doc.nvt_failed == True
            raise Exception(
                "NVT failed with critical and experimental starting conditions and the molecule is "
                + job.sp.mol_name
            )

@nptnvt_group
@ProjectGEMC.pre.after(NVT_liqbox)
@ProjectGEMC.post.isfile("nvt.final.xyz")
@ProjectGEMC.post(lambda job: "nvt_liqbox_final_dim" in job.doc)
@ProjectGEMC.operation
def extract_final_NVT_config(job):
    "Extract final coords and box dims from the liquid box simulation"

    import subprocess

    lines = job.sp.N_liq * job.sp.N_atoms
    cmd = [
        "tail",
        "-n",
        str(lines + 2),
        job.fn("nvt.eq.out.xyz"),
    ]

    # Save final liuqid box xyz file
    xyz = subprocess.check_output(cmd).decode("utf-8")
    with open(job.fn("nvt.final.xyz"), "w") as xyzfile:
        xyzfile.write(xyz)

    # Save final box dims to job.doc
    box_data = []
    with open(job.fn("nvt.eq.out.H")) as f:
        for line in f:
            box_data.append(line.strip().split())
    job.doc.nvt_liqbox_final_dim = float(box_data[-6][0]) / 10.0  # nm

@nptnvt_group
@ProjectGEMC.pre.after(extract_final_NVT_config)
@ProjectGEMC.pre(lambda job: "gemc_failed" not in job.doc)
@ProjectGEMC.post(npt_finished)
@ProjectGEMC.operation(directives={"omp_num_threads": 4})
def NPT_liqbox(job):
    "Equilibrate the liquid box"

    import os
    import errno
    import mbuild
    import foyer
    import mosdef_cassandra as mc
    import unyt as u

    ff = foyer.Forcefield(job.fn("ff.xml"))

    # Load the compound and apply the ff
    compound = mbuild.load(job.sp.smiles, smiles=True)
    compound_ff = ff.apply(compound)
    # Load the liquid box final configuration from NVT
    with job:
        # liq_box = mbuild.formats.xyz.read_xyz(job.fn("nvt.final.xyz"))
        liq_box = mbuild.load(job.fn("nvt.final.xyz"))
    boxl = job.doc.nvt_liqbox_final_dim
    liq_box.box = mbuild.Box(lengths=[boxl, boxl, boxl], angles=[90.0, 90.0, 90.0])
    liq_box.periodicity = [True, True, True]
    box_list = [liq_box]
    species_list = [compound_ff]
    mols_in_boxes = [[job.sp.N_liq]]

    #Create system
    system = mc.System(box_list, species_list, mols_in_boxes=mols_in_boxes)

    # Create a new moves object
    moves = mc.MoveSet("npt", species_list)

    # Edit the volume move probability to be more reasonable
    orig_prob_volume = moves.prob_volume
    new_prob_volume = 1.0 / job.sp.N_liq
    moves.prob_volume = new_prob_volume
    moves.prob_translate = moves.prob_translate + orig_prob_volume - new_prob_volume

    # Define thermo output props
    thermo_props = [
        "energy_total",
        "pressure",
        "mass_density",
    ]

    # Define custom args
    custom_args, custom_args_gemc = _get_custom_args(job)
    custom_args["run_name"] = "npt.eq"
    custom_args["properties"] = thermo_props

    # Move into the job dir and start doing things
    with job:
        # Run equilibration
        # Load class properies for each training and testing molecule
        class_dict = _get_class_from_molecule(job.sp.mol_name)
        class_data = class_dict[job.sp.mol_name]
        # Reference data to compare to (i.e. experiments or other simulation studies) (load from constants file in project_gaff.py as needed)
        # Loop over the keys of the dictionaries
        for t, pvap in class_data.expt_Pvap.items():
            if t == job.sp.T:
                pressure = pvap * u.bar

        # Try running NPT with experimental starting conditions
        try:
            # Run equilibration
            mc.run(
                system=system,
                moveset=moves,
                run_type="equilibration",
                run_length=job.sp.nsteps_npt,
                temperature=job.sp.T * u.K,
                pressure=pressure,
                **custom_args,
            )
            job.doc["npt_fin"] = True
        #If it fails, try restarting from ciritcal point conditions
        except:
            # if job failed with critical conditions as intial conditions, terminate with error
            if job.doc.get("use_crit", False):
                # If so, terminate with error and log failure in job document
                job.doc.gemc_failed = True
                raise Exception(
                    "NPT failed with critical and experimental starting conditions and the molecule is "
                    + job.sp.mol_name
                    + " at temperature "
                    + str(job.sp.T)
                )
            # Otherwise, try with critical conditions
            else:  
                job.doc.use_crit = True
                #Delete variables from previous failed run
                del job.doc["vapboxl"]  # calc_boxes
                del job.doc["liqboxl"]  # calc_boxes
                if "nvt_liqbox_final_dim" in job.doc.keys():
                    del job.doc["nvt_liqbox_final_dim"]
                if "nvt_fin" in job.doc.keys():
                    del job.doc["nvt_fin"]
                #Delete previous data files
                with job:
                    for file_path in glob.glob("nvt.*"):
                        os.remove(file_path)
                    for file_path in glob.glob("npt.*"):
                        os.remove(file_path)

@nptnvt_group
@ProjectGEMC.pre.after(NPT_liqbox)
@ProjectGEMC.post.isfile("npt.final.xyz")
@ProjectGEMC.post(lambda job: "npt_liqbox_final_dim" in job.doc)
@ProjectGEMC.operation
def extract_final_NPT_config(job):
    "Extract final coords and box dims from the liquid box simulation"

    import subprocess

    lines = job.sp.N_liq * job.sp.N_atoms
    cmd = ["tail","-n",str(lines + 2),job.fn("npt.eq.out.xyz"),]

    # Save final liquid box xyz file
    xyz = subprocess.check_output(cmd).decode("utf-8")
    with open(job.fn("npt.final.xyz"), "w") as xyzfile:
        xyzfile.write(xyz)

    # Save final box dims to job.doc
    box_data = []
    with open(job.fn("npt.eq.out.H")) as f:
        for line in f:
            box_data.append(line.strip().split())
    job.doc.npt_liqbox_final_dim = float(box_data[-6][0]) / 10.0  # nm


@ProjectGEMC.label
def gemc_equil_complete(job):
    "Confirm gemc equilibration has completed"
    import numpy as np

    # Get last restart file
    try:
        last_file = get_last_checkpoint(job.fn("gemc.eq"))
        selected_file = job.fn(last_file + ".out.box1.prp")
    except:
        selected_file = job.fn("gemc.eq.out.box1.prp")

    # Check that the last step was completed
    try:
        with open(selected_file, "rb") as f:
            # Move the pointer to the end of the file, but leave space to find the last line
            f.seek(-2, os.SEEK_END)
            # Read backward until a newline is found
            while f.read(1) != b"\n":
                f.seek(-2, os.SEEK_CUR)
            # Read the last line after finding the newline
            last_line = f.readline().decode()
        # Split the last line and extract the first number
        first_value = int(last_line.split()[0])
        # This line will fail until job.doc.nsteps_gemc_eq is defined
        if hasattr(job.doc, "nsteps_gemc_eq"):
            completed = first_value == job.doc.nsteps_gemc_eq
        else:
            completed = False
    except:
        completed = False

    #Regardless of whether equilibration is complete, if the job doc has gemc_eq_fin = True, we will return True
    #This allows us to check simulations before adding more steps
    if job.doc.get("gemc_eq_fin", False):
        completed = True

    return completed

def delete_data_gemc(job, run_name, mv=True, subfolder="results_old"):
    "Delete data from previous operations"
    import os
    import glob
    import shutil

    # List of files which must be moved or deleted when the job fails
    glob_args = [
        "MSER*",
        "*_eq_col_*",
        "box*.in.xyz",
        run_name + ".*",
        "mosdef_cassandra_*.log",
        "prod.*",
        "Equil_Output.txt",
    ]
    # If move is true, move the files to a subfolder instead of deleting them
    with job:
        if mv == True:
            if not os.path.exists(subfolder):
                os.makedirs(subfolder)
            for glob_arg in glob_args:
                for file_path in glob.glob(glob_arg):
                    shutil.move(
                        file_path, os.path.join(subfolder, os.path.basename(file_path))
                    )
            shutil.copy(
                "signac_job_document.json",
                os.path.join(subfolder, "signac_job_document.json"),
            )
        else:
            for glob_arg in glob_args:
                for file_path in glob.glob(glob_arg):
                    os.remove(file_path)

    # Regardless of whether we remove or move the files, we want to delete the job document keys
    if "nsteps_gemc_eq" in job.doc.keys():
        del job.doc["nsteps_gemc_eq"]
        del job.doc["gemc_vapboxl"]
    if "gemc_eq_fin" in job.doc.keys():
        del job.doc["gemc_eq_fin"]
    if "prod_ready" in job.doc.keys():
        del job.doc["prod_ready"]
    if "no_overlap" in job.doc.keys():
        del job.doc["no_overlap"]
    if "Nexc_good" in job.doc.keys():
        del job.doc["Nexc_good"]
    if "pct_diff" in job.doc.keys():
        del job.doc["pct_diff"]
        del job.doc["insert_val"]
        del job.doc["delete_val"]

def delete_data(job, run_name, mv=True, subfolder="results_old"):
    "Delete data from previous operations"
    import os
    import glob
    import shutil

    # List of files which must be moved or deleted when the job fails
    glob_args = [
        "MSER*",
        "*_eq_col_*",
        "box*.in.xyz",
        run_name + ".*",
        "mosdef_cassandra_*.log",
        "prod.*",
        "nvt.*",
        "npt.*",
        "Equil_Output.txt",
    ]
    # If move is true, move the files to a subfolder instead of deleting them
    with job:
        if mv == True:
            if not os.path.exists(subfolder):
                os.makedirs(subfolder)
            for glob_arg in glob_args:
                for file_path in glob.glob(glob_arg):
                    shutil.move(
                        file_path, os.path.join(subfolder, os.path.basename(file_path))
                    )
            shutil.copy(
                "signac_job_document.json",
                os.path.join(subfolder, "signac_job_document.json"),
            )
        else:
            for glob_arg in glob_args:
                for file_path in glob.glob(glob_arg):
                    os.remove(file_path)

    # Regardless of whether we remove or move the files, we want to delete the job document keys
    del job.doc["vapboxl"]  # calc_boxes
    del job.doc["liqboxl"]  # calc_boxes
    if "nsteps_gemc_eq" in job.doc.keys():
        del job.doc["nsteps_gemc_eq"]  # run_gemc
        del job.doc["gemc_vapboxl"]
    if "nvt_fin" in job.doc.keys():
        del job.doc["nvt_fin"]  # nvt_liqbox
    if "npt_fin" in job.doc.keys():
        del job.doc["npt_fin"]  # NPT_liqbox
    if "gemc_eq_fin" in job.doc.keys():
        del job.doc["gemc_eq_fin"]
    if "prod_ready" in job.doc.keys():
        del job.doc["prod_ready"]
    if "no_overlap" in job.doc.keys():
        del job.doc["no_overlap"]
    if "Nexc_good" in job.doc.keys():
        del job.doc["Nexc_good"]
    if "pct_diff" in job.doc.keys():
        del job.doc["pct_diff"]
        del job.doc["insert_val"]
        del job.doc["delete_val"]


def make_usable_xyz(job, filename, box):
    "Make the xyz file usable for mbuild"
    import subprocess

    H_file = job.fn(filename + ".out.box" + str(box) + ".H")
    filename_in = job.fn(filename + ".out.box" + str(box) + ".xyz")
    filename_out = job.fn(filename + ".box" + str(box) + ".final.xyz")
    num_molec = int(grab_last_value(job.fn(H_file)))
    # If the file doesn't already exist, generate it, otherwise point to it
    if not os.path.exists(filename_out):
        lines = num_molec * job.sp.N_atoms
        print(lines)
        cmd = ["tail","-n",str(lines + 2),filename_in,]

        # Save final liuqid box xyz file
        xyz = subprocess.check_output(cmd).decode("utf-8")
        with open(filename_out, "w") as xyzfile:
            xyzfile.write(xyz)

    return filename_out, num_molec


def grab_last_value(filename):
    last_value = None
    with open(filename, "r") as file:
        for line in reversed(file.readlines()):  # Read lines backwards
            stripped_line = line.strip()
            if stripped_line:  # Skip empty lines
                # Split the line into values
                values = stripped_line.split()
                if values:
                    last_value = values[-1]  # Get the last value
                    break
    return last_value


def get_gemc_boxes(job, eq_data_name):
    "Get the box from the final xyz file"
    import mbuild
    import signac
    import json

    # If we are restarting from a previous (better performing job)
    if "restart_from" in job.doc.keys():
        project = signac.get_project()
        job_init = project.open_job(id=job.doc.restart_from)
        job_init_doc = job_init.fn("signac_job_document.json")
        # Check if the file exists
        if os.path.exists(job_init_doc):

            if job.sp.T == job_init.sp.T:
                job.doc.T_rst_match = True
            # with open(job_init_doc, "r") as f:
            #     statepoint = json.load(f)

            # Extract the values for the specified keys
            # boxl_liq = statepoint.get("npt_liqbox_final_dim", None)
            # boxl_vap = statepoint.get("vapboxl", None) #nm
            last_file = get_last_checkpoint(job_init.fn(eq_data_name))
            boxl_liq, boxl_vap = extract_cubic_values(job_init, last_file)

            # Build liquid and vapor boxes from previous simulations
            file_liq_out, N_liq_use = make_usable_xyz(job_init, last_file, 1)
            file_vap_out, N_vap_use = make_usable_xyz(job_init, last_file, 2)
            # vap_box = mbuild.formats.xyz.read_xyz(file_vap_out)
            vap_box = mbuild.load(file_vap_out)
            vap_box.box = mbuild.Box(
                lengths=[boxl_vap, boxl_vap, boxl_vap], angles=[90.0, 90.0, 90.0]
            )
            vap_box.periodicity = [True, True, True]

            # Save a text file indicating that other job's points are used
            job.doc.liq_boxl_use = boxl_liq
            job.doc.vap_box1_use = boxl_vap
            mols_in_boxes = [[N_liq_use], [N_vap_use]]
            mols_to_add = [[0], [0]]
        else:
            raise Exception(
                "The file signac_statepoint.json does not exist in the job directory. Check job ID in restart_from"
            )
    # Otherwise create them from this job
    else:
        # Create box list and species list
        boxl_liq = job.doc.npt_liqbox_final_dim  # saved in nm
        N_liq_use = job.sp.N_liq
        file_liq_out = job.fn("npt.final.xyz")
        # When creating GEMC boxes, multiply the vapor box by a factor if necessary
        vap_box_mult = (
            1.0 if "vap_box_mult" not in job.doc.keys() else job.doc.vap_box_mult
        )
        boxl_vap = job.doc.vapboxl * vap_box_mult  # nm

        N_vap_use = job.sp.N_vap
        vap_box = mbuild.Box(lengths=[boxl_vap, boxl_vap, boxl_vap])
        mols_in_boxes = [[N_liq_use], [0]]
        mols_to_add = [[0], [N_vap_use]]

    # liq_box = mbuild.formats.xyz.read_xyz(file_liq_out)
    liq_box = mbuild.load(file_liq_out)
    liq_box.box = mbuild.Box(
        lengths=[boxl_liq, boxl_liq, boxl_liq], angles=[90.0, 90.0, 90.0]
    )
    liq_box.periodicity = [True, True, True]

    job.doc["gemc_vapboxl"] = boxl_vap

    return liq_box, vap_box, boxl_liq, boxl_vap, mols_in_boxes, mols_to_add

# @eq_group
# @ProjectGEMC.pre.after(extract_final_NPT_config)
# @ProjectGEMC.pre(lambda job: "gemc_failed" not in job.doc)
# @ProjectGEMC.post(gemc_equil_complete)
# @ProjectGEMC.operation(directives={"omp_num_threads": 4})
def run_gemc_eq(job):
    "Equilibrate GEMC"

    import os
    import errno
    import mbuild
    import foyer
    import mosdef_cassandra as mc
    import unyt as u
    import glob

    ff = foyer.Forcefield(job.fn("ff.xml"))

    # Load the compound and apply the ff
    compound = mbuild.load(job.sp.smiles, smiles=True)
    compound_ff = ff.apply(compound)
    run_name_eq = "gemc.eq"
    
    #Create box and system
    liq_box, vap_box, boxl_liq, boxl_vap, mols_in_boxes, mols_to_add = get_gemc_boxes(
        job, run_name_eq
    )
    box_list = [liq_box, vap_box]
    species_list = [compound_ff]
    system = mc.System(
        box_list, species_list, mols_in_boxes=mols_in_boxes, mols_to_add=mols_to_add
    )

    # Create a new moves object
    moves = mc.MoveSet("gemc", species_list)

    # Edit the volume and swap move probability to be more reasonable
    orig_prob_volume = moves.prob_volume
    orig_prob_swap = moves.prob_swap
    new_prob_volume = 1.0 / (job.sp.N_vap + job.sp.N_liq)
    
    if job.id == "5cffb08f9d07bdb3fe0601ba4896d72c":
        # Higher swap probability to test if Glycerol can be simulated with GEMC
        new_prob_swap = 8.0 / 0.05 / (job.sp.N_vap + job.sp.N_liq)
    else:
        new_prob_swap = 4.0 / 0.05 / (job.sp.N_vap + job.sp.N_liq)

    moves.prob_volume = new_prob_volume
    moves.prob_swap = new_prob_swap
    moves.prob_translate = moves.prob_translate + orig_prob_volume - new_prob_volume
    
    if job.id == "5cffb08f9d07bdb3fe0601ba4896d72c":
        # Higher swap probability to test if Glycerol can be simulated with GEMC
        moves.prob_rotate = moves.prob_rotate + (orig_prob_swap - new_prob_swap)/2
        moves.prob_translate = moves.prob_translate + (orig_prob_swap - new_prob_swap)/2
    else:
        moves.prob_translate = moves.prob_translate + orig_prob_swap - new_prob_swap

    # Define thermo output props
    thermo_props = [
        "energy_total",
        "pressure",
        "volume",
        "nmols",
        "mass_density",
        "enthalpy",
    ]

    # Define custom args
    custom_args, custom_args_gemc = _get_custom_args(job)
    custom_args_gemc["run_name"] = run_name_eq
    custom_args_gemc["properties"] = thermo_props
    custom_args_gemc["cbmc_n_insert"] = 20
    custom_args_gemc["cbmc_n_dihed"] = 20

    #Set vapor cutoff to 95% of half the box length to avoid k vectors issue
    # cutoff_vap = np.minimum(round(0.95*boxl_vap/2,5), round(6 * job.sp.max_sigma, 5))
    cutoff_vap = round(0.95*boxl_vap/2,5)
    custom_args_gemc["charge_cutoff_box2"] = (cutoff_vap * u.nanometer).to("angstrom")
    custom_args_gemc["vdw_cutoff_box2"] = (cutoff_vap * u.nanometer).to("angstrom")
    job.doc["cutoff_vap"] = cutoff_vap  # Save the cutoff value to the job document

    # Try to run GEMC
    try:
        with job:
            first_run = custom_args_gemc["run_name"]  # gemc.eq
            # Run initial equilibration if it does not exxist
            if not has_checkpoint(first_run):
                mc.run(
                    system=system,
                    moveset=moves,
                    run_type="equilibration",
                    run_length=job.sp.nsteps_gemc_eq,
                    temperature=job.sp.T * u.K,
                    **custom_args_gemc,
                )
            elif not check_complete(first_run):
                mc.restart(
                    restart_from=get_last_checkpoint(first_run),
                )

            init_gemc_liq = job.fn(first_run + ".out.box1.prp")
            init_gemc_vap = job.fn(first_run + ".out.box2.prp")
            prop_cols = [5]  # Use number of moles to decide equilibrium
            # Load initial eq data from both boxes
            df_box1 = np.genfromtxt(init_gemc_liq)
            df_box2 = np.genfromtxt(init_gemc_vap)

            # Process both boxes in one loop
            eq_data_dict = {}
            for b, box in enumerate([df_box1, df_box2]):
                box_name = "Liquid" if b == 0 else "Vapor"
                for prop_index in prop_cols:
                    eq_col = box[:, prop_index - 1]
                    # Save eq_col as a csv for later analysis
                    key = f"{box_name}_{prop_index}"
                    eq_col_file = job.fn(f"{box_name}_eq_col_{prop_index}.csv")
                    np.savetxt(eq_col_file, eq_col, delimiter=",")
                    # Save the eq_col and file to a dictionary for later use
                    eq_data_dict[key] = {"data": eq_col, "file": eq_col_file}

            if os.path.exists("Equil_Output.txt"):  # Remove the file if it exists
                os.remove("Equil_Output.txt")

            # Set number of iterations per extension and intitialize counter and total number of steps
            eq_extend = int(job.sp.nsteps_gemc_eq/4)  # int(job.sp.nsteps_gemc_eq/4)
            total_eq_steps = job.sp.nsteps_gemc_eq
            count = 1

            # Get the total number of equilibration restarts and steps so far
            existing_eq_steps = count_steps(
                get_last_checkpoint(custom_args_gemc["run_name"])
            )

            # Inititalize max number of eq_steps
            if "max_eq_steps" not in job.doc:
                # If no value exists, set it as 4 times the original number of eq steps
                job.doc.max_eq_steps = job.sp.nsteps_gemc_eq * 4
            # The max number of steps is the larger of the number of steps + 1-2*org number of steps or the current max
            max_eq_steps = np.maximum(
                job.doc.max_eq_steps, existing_eq_steps + job.sp.nsteps_gemc_eq
            )
            # Originally set the document eq_steps to the max number, it will be overwritten later
            job.doc.nsteps_gemc_eq = int(max_eq_steps)

            # While the max number of eq steps has not been reached
            while total_eq_steps <= max_eq_steps:
                # Set production start tolerance as at least 25% of the total number of data points
                prod_tol_eq = int(total_eq_steps / 4) / custom_args_gemc["prop_freq"]

                # Set this run and last last run
                this_run = custom_args_gemc["run_name"] + f".rst.{count:03d}"
                prior_run = get_last_checkpoint(custom_args_gemc["run_name"])

                # Check if equilibration is reached via the pymser algorithms
                is_equil = check_equil_converge(job, eq_data_dict, prod_tol_eq)

                # Check if this simulation restarts from an equilibrated one at the same temperature
                if "restart_from" in job.doc.keys():
                    if job.doc.get("T_rst_match", False):
                        # If at least 100k steps have been run
                        if (
                            total_eq_steps >= existing_eq_steps
                            and total_eq_steps >= job.sp.nsteps_gemc_prod
                        ):
                            # Start production run from the last piece of the equilibration run
                            is_equil = True

                # If equilibrium is reached, break the loop
                if is_equil:
                    break
                # Otherwise, extend the simulation
                else:
                    # Check if this simulation exists
                    sim_exists = has_checkpoint(this_run)

                    # If the simulation exists
                    if sim_exists:
                        # Get the number of total steps in the simulation
                        this_run_input = this_run
                        total_eq_steps = int(count_steps(this_run_input))
                    else:
                        # Set the number of total steps given eq_extend
                        total_eq_steps += int(eq_extend)

                    # If you have enough steps, run the simulation
                    if total_eq_steps <= max_eq_steps:
                        # If the simulation doesn't exist, run it
                        if not sim_exists:
                            mc.restart(
                                restart_from=prior_run,
                                run_type="equilibration",
                                total_run_length=total_eq_steps,
                                run_name=this_run,
                            )
                        # If the simulation exists but is not complete, restart it
                        elif sim_exists and not check_complete(this_run):
                            # Finish the simulation
                            mc.restart(
                                restart_from=get_last_checkpoint(this_run),
                            )
                    # Otherwise report an error
                    else:
                        job.doc.equil_fail = True
                        raise Exception(
                            f"GEMC equilibration failed to converge after {max_eq_steps} steps"
                        )

                    # Add restart data to eq_col
                    # After each restart, load the updated properties data for both boxes
                    sim_box1 = this_run + ".out.box1.prp"
                    sim_box2 = this_run + ".out.box2.prp"
                    df_box1r = np.genfromtxt(job.fn(sim_box1))
                    df_box2r = np.genfromtxt(job.fn(sim_box2))

                    # Process and add the restart data to eq_col for each property in each box
                    for b, box in enumerate([df_box1r, df_box2r]):
                        box_name = "Liquid" if b == 0 else "Vapor"
                        for i, prop_index in enumerate(prop_cols):
                            # Get the key from the property and box name
                            key = f"{box_name}_{prop_index}"
                            # Extract the column data for this restart and append to accumulated data
                            eq_col_restart = box[:, prop_index - 1]
                            all_eq_data = np.concatenate(
                                (eq_data_dict[key]["data"], eq_col_restart)
                            )
                            # Save the new data to the eq_col file
                            np.savetxt(
                                eq_data_dict[key]["file"], all_eq_data, delimiter=","
                            )
                            # Overwite the current data in the eq_data_dict with restart data
                            eq_data_dict[key]["data"] = all_eq_data
                # Increase the counter
                count += 1

            # Set the step counter to whatever the final number of equilibration steps was
            job.doc.nsteps_gemc_eq = total_eq_steps
            job.doc.equil_fail = False
            total_sim_steps = int(job.sp.nsteps_gemc_prod + job.doc.nsteps_gemc_eq)
            job.doc["total_gemc_steps"] = total_sim_steps
            job.doc["gemc_eq_fin"] = True

    except:
        vap_box_mult_str = (
            "_vbx_" + str(job.doc.vap_box_mult)
            if "vap_box_mult" in job.doc.keys()
            else ""
        )
        rst_str = "_rest_{:.4s}".format(job.doc.restart_from) if "restart_from" in job.doc.keys() else ""
        crit_str = "_crit" if job.doc.get("use_crit", False) else "_no_crit"
        results_folder = "results" + crit_str + vap_box_mult_str + rst_str
        # If equilibration wasn't long enough
        if job.doc.get("equil_fail", False):
            # Extend the simulation
            job.doc.max_eq_steps = max_eq_steps + job.sp.nsteps_gemc_eq
            job.doc.nsteps_gemc_eq = job.doc.max_eq_steps
            del job.doc["equil_fail"]
            #Make a flag to check this job before starting production or restarting gemc equilibration
            job.doc["check_me"] = True
            job.doc["gemc_eq_fin"] = True
        # if GEMC failed with critical conditions as intial conditions, terminate with error
        elif job.doc.get("use_crit", False):
            job.doc.gemc_failed = True
            delete_data(
                job,
                custom_args_gemc["run_name"],
                mv=True,
                subfolder=results_folder,
            )
            raise Exception(
                "GEMC failed with critical and experimental starting conditions and the molecule is "
                + job.sp.mol_name
                + " at temperature "
                + str(job.sp.T)
            )
        # If GEMC fails, remove files in post conditions of previous operations
        else:
            delete_data(
                job,
                custom_args_gemc["run_name"],
                mv=True,
                subfolder=results_folder,
            )
            # If the simulation failed for another reason, try with critical conditions
            job.doc.use_crit = True
            if "equil_fail" in job.doc:
                del job.doc["equil_fail"]

def check_eq(job):
    from scipy.signal import savgol_filter
    import numpy as np
    statement = ""
    
    vap_box_mult_str = (
            "_vbx_" + str(job.doc.vap_box_mult)
            if "vap_box_mult" in job.doc.keys()
            else ""
        )
    rst_str = "_rest_{:.4s}".format(job.doc.restart_from) if "restart_from" in job.doc.keys() else ""
    crit_str = "_crit" if job.doc.get("use_crit", False) else "_no_crit"
    # If the job failed, move files to a separate folder
    folder_name = "results" + crit_str + vap_box_mult_str + rst_str

    first_shrink = False
    prod_ready = {"rst_data": True, "nmol_under_30": True, "box_size": True}
    #Get box data
    # Process and add the restart data to eq_col for each property in each box
    eq_data_dict = {}
    prop_cols = [5]  # Use number of moles to decide equilibrium
    for b in range(2):
        box_name = "Liquid" if b == 0 else "Vapor"
        for i, prop_index in enumerate(prop_cols):
            # Get the key from the property and box name
            key = f"{box_name}_{prop_index}"
            # Extract the column data for this restart and append to accumulated data
            eq_col_file = job.fn(f"{box_name}_eq_col_{prop_index}.csv")
            eq_col = np.genfromtxt(eq_col_file, delimiter=",")
            # Save the eq_col and file to a dictionary for later use
            eq_data_dict[key] = {"data": eq_col, "file": eq_col_file}
            if i ==0 and b==0:
                eq_col_liq = eq_data_dict[key]["data"]
    # Get the eq data for liquid box n_mols
    results, adf_test = get_pymser_results(eq_col_liq)

    # Only count this run as ready for production if at least 30 molecules are in the liquid box
    if np.mean(eq_col_liq[results["t0"]:]) < 30:  # 30
        # Otherwise add to the job document that the production failed
        job.doc["nmol_under_30"] = True
        job.doc["check_me"] = True
        prod_ready["nmol_under_30"] = False
    
    #If check_me is true 
    if job.doc.get("check_me", False):
        if job.doc.get("gemc_eq_fin", False):
            del job.doc["gemc_eq_fin"]
        #First check if another simulation with the same or more steps at this temperature has completed
        project = signac.get_project()
        for job_new in project.find_jobs({"mol_name":job.sp.mol_name, "atom_type":job.sp.atom_type, "T":job.sp.T}):
            #Check if another simulation has completed and passed the check
            if job_new.doc.get("gemc_eq_fin", False) and job_new.doc.get("prod_ready", False):
                #Restart from this job
                job.doc["restart_from"] = job_new.id
                prod_ready["rst_data"] = False
                break

    # Estimate the slope of the number of molecules in the liquid box vs step number
    steps = np.arange(0, len(eq_col_liq))
    win_len = max(3, int(len(eq_col_liq) * 0.1) | 1)
    dydx = savgol_filter(eq_col_liq, window_length=win_len, polyorder=2, deriv=1)
    eq_col_est = savgol_filter(eq_col_liq, window_length=win_len, polyorder=2)

    #Plot the number of molecules in the liquid box vs step number and the slope
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(steps, eq_col_liq, label="Original Data", alpha=0.5)
    ax1.plot(steps, eq_col_est, color='red', label="Smoothed Data")
    ax1.set_xlabel("Sweeps")
    ax1.set_ylabel("Number of Molecules in Liquid Box")
    ax1.legend()
    ax1.set_title("Liquid Box Molecules vs Sweeps")
    ax2.plot(steps, dydx, color='green', label="Slope (dN/dx)")
    ax2.axhline(0, color='black', linestyle='--')
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Slope")
    ax2.legend()
    ax2.set_title("Slope of Liquid Box Molecules vs Sweeps")
    plt.tight_layout()
    plt.savefig(job.fn("liq_box_slope.png"))

    #If no such simulation exists, check if the simulation seems likely to vaporize/liquidate. 
    if ("restart_from" not in job.doc and job.doc.get("check_me", False) and job.doc.get("nsteps_gemc_eq", 0) >= 200000):
        # When this step happens, at least 200K steps should have been run

        #Check if the number of molecules in the liquid box is decreasing on average
        #Count percentage of points with positive slope
        pos_slope = np.count_nonzero(dydx > 0)
        pct_pos = pos_slope/len(dydx)*100
        
        #if more than 85% of the points have a positive slope, the liquid box is likely to condense (increase vapor box size)
        if pct_pos > 85:
             #If we've already increased the vapor box once, double the volume
            if "vap_box_mult" in job.doc.keys():
                job.doc.vap_box_mult = round(((job.doc["vap_box_mult"]**3)*2))**(1/3)
            #Shrink vapor box volume by factor of 3
            else:
                job.doc.vap_box_mult = round(2.5**(1/3),3)
            prod_ready["box_size"] = False
            statement += f"increased vapor box size to {job.doc.vap_box_mult}"
        elif pct_pos < 15:
            #If more than 85% of the points have a negative slope, the liquid box is likely to evaporate (decrease vapor box size)
            if "vap_box_mult" in job.doc.keys():
                #Shrink the vapor box, by half the volume
                job.doc.vap_box_mult = round(((job.doc["vap_box_mult"]**3)*0.5))**(1/3)
            #Try with critical conditions if not already done (vapo will be smaller and liquid larger)
            elif not job.doc.get("use_crit", False):
                job.doc["use_crit"] = True
                first_shrink = True
                statement += f"decreased vapor box size to critical conditions"
            #Shrink vapor box volume by factor of 2
            else:
                job.doc.vap_box_mult = round(0.5**(1/3),3)
                statement += f"decreased vapor box size to {job.doc.vap_box_mult}"
            prod_ready["box_size"] = False

    if np.all(list(prod_ready.values())):
        job.doc["prod_ready"] = True
        #If this job was being checked, it means it's not actually ready for production
        if "check_me" in job.doc.keys():
            del job.doc["prod_ready"]
    else:
        job.doc["prod_ready"] = False
        #Delete previous data files
        with job:
            if first_shrink == False:
                #Delete all gemc data if changing the vapor box size
                delete_data_gemc(job, "gemc.eq", mv=True, subfolder=folder_name)
            #Delete all data if switching to critical conditions
            else:
                #Delete only the gemc equilibration data if trying with critical conditions
                delete_data(job, "gemc.eq", mv=True, subfolder=folder_name)
    #Delete the check_me flag
    if "check_me" in job.doc.keys():
        del job.doc["check_me"]

    if statement != "":
        with open("Equil_Output.txt", "a") as f:
            print(statement, file=f)

    return job.doc["prod_ready"]

@eq_group
@ProjectGEMC.pre.after(extract_final_NPT_config)
@ProjectGEMC.pre(lambda job: "gemc_failed" not in job.doc) #Job has not failed
@ProjectGEMC.post(gemc_equil_complete) #Equilibration is complete
@ProjectGEMC.post(lambda job: "prod_ready" in job.doc) #Production is ready to start
@ProjectGEMC.operation(directives={"omp_num_threads": 4})
def gemc_eq_restart(job):
    "Restart GEMC equilibration if needed"
    prod_completed = False
    while not prod_completed:
        #If the job is ready for production, skip this step
        if job.doc.get("prod_ready", False):
            prod_completed = True
        else:
            run_gemc_eq(job)
            prod_completed = check_eq(job)
# @eq_group            
# @ProjectGEMC.pre.after(run_gemc_eq)
# @ProjectGEMC.pre(lambda job: "gemc_failed" not in job.doc)
# @ProjectGEMC.post(lambda job: "prod_ready" in job.doc)
# @ProjectGEMC.operation(directives={"omp_num_threads": 4})
            
@prod_group
@ProjectGEMC.pre.after(gemc_eq_restart)
@ProjectGEMC.pre(lambda job: job.doc.get("prod_ready", False))
@ProjectGEMC.post(gemc_prod_complete)
@ProjectGEMC.operation(directives={"omp_num_threads": 4})
def run_gemc_prod(job):
    "Production Phase GEMC"

    import os
    import errno
    import mbuild
    import foyer
    import mosdef_cassandra as mc
    import unyt as u
    import glob


    # Define thermo output props
    thermo_props = [
        "energy_total",
        "pressure",
        "volume",
        "nmols",
        "mass_density",
        "enthalpy",
    ]

    # Define custom args
    custom_args, custom_args_gemc = _get_custom_args(job)
    run_name_eq = "gemc.eq"
    custom_args_gemc["run_name"] = run_name_eq
    custom_args_gemc["properties"] = thermo_props

    #Set vapor cutoff to 95% of half the box length to avoid k vectors issue
    boxl_vap = job.doc["gemc_vapboxl"]
    cutoff_vap = round(0.95*boxl_vap/2,5)
    custom_args_gemc["charge_cutoff_box2"] = (cutoff_vap * u.nanometer).to("angstrom")
    custom_args_gemc["vdw_cutoff_box2"] = (cutoff_vap * u.nanometer).to("angstrom")
    job.doc["cutoff_vap"] = cutoff_vap  # Save the cutoff value to the job document

    #Add extra sweeps to production data if necessary
    ##DO LATER

    # Try to run GEMC
    try:
        with job:
            #Get last checkpoint from equilibration
            prior_run = get_last_checkpoint(custom_args_gemc["run_name"])
            # Get final number of equilibration steps
            total_sim_steps = job.doc.total_gemc_steps
            # Run production
            if not has_checkpoint("prod"):
                mc.restart(
                    restart_from=prior_run,
                    run_type="production",
                    total_run_length=total_sim_steps,
                    run_name="prod",
                )
            elif not check_complete("prod"):
                mc.restart(
                    restart_from=get_last_checkpoint("prod"),
                )

    except:
        # if GEMC fails in production, terminate with error
        job.doc.gemc_failed = True
        raise Exception(
            "GEMC failed in production phase and the molecule is "
            + job.sp.mol_name
            + " at temperature "
            + str(job.sp.T)
        )

@prod_group
@ProjectGEMC.pre.after(run_gemc_prod)
@ProjectGEMC.pre(gemc_prod_complete)
@ProjectGEMC.post(
    lambda job: ("no_overlap" in job.doc and "Nexc_good" in job.doc)
    or (job.doc.get("gemc_failed", False))
)
@ProjectGEMC.operation
def check_prod_data(job):
    "Check if the production files overlap"
    import numpy as np
    import os
    import sys
    import subprocess

    density_col = 6
    statement = ""

    check_dict = {"no_overlap": True,
                  "Nexc_good": True}
    if job.doc.get("gemc_failed", False):
        pass
    else:
        ###First, Check to make sure there is no overlap between boxes
        with job:
            # Get all production files
            prod_files1 = sorted(glob.glob("prod.*.box1.prp"))
            prod_files2 = sorted(glob.glob("prod.*.box2.prp"))
            # Concatenate all production files using genfromtxt into one
            df_box1 = np.vstack([np.genfromtxt(f) for f in prod_files1])
            df_box2 = np.vstack([np.genfromtxt(f) for f in prod_files2])

        density_liq = df_box1[:, density_col - 1]
        density_vap = df_box2[:, density_col - 1]

        # Compare each line of nmols_liq and nmols_vap, if the row is ever bigger for the vapor box, print the job id
        mask = density_vap > density_liq
        # print(mask)
        if np.any(mask):
            print(
                f"Job {job.id} has a vapor box with more molecules than the liquid box"
            )
            # Print mol_name, T, and restart from statepont
            print(
                f"Molecule: {job.sp.mol_name}, T: {job.sp.T}, Restart from: {job.sp.restart}"
            )
            check_dict["no_overlap"] = False

        ###Next, check that the number of insertions and deletions are reasonable
        insert_val = None
        delete_val = None
        
        #Initialize log file as production file
        filename = job.fn("prod.out.log")

        #Find the last instance of the words insert and delete starting from the bottom of the file
        # Read file lines
        with FileReadBackwards(filename, encoding="utf-8") as frb:
            for line in frb:
                if "Delete" in line:
                    # Split the last line and extract the first number
                    delete_val = int(line.split()[2])
                elif "Insert" in line:
                    # Split the last line and extract the first number
                    insert_val = int(line.split()[2])
                    break
        N_mols = job.sp.N_vap + job.sp.N_liq
        pct_diff = abs(insert_val - delete_val) / insert_val * 100
        job.doc["insert_val"] = insert_val
        job.doc["delete_val"] = delete_val
        job.doc["pct_diff"] = pct_diff
        #Check that the insert and delete values are within 5% of each other
        if pct_diff > 5:
            statement += f"Job {job.id} production has a large difference between insert and delete counts" + "\n"
            statement += f"Insert: {insert_val}, Delete: {delete_val}, Percent Difference: {pct_diff:.2f}%"
            check_dict["Nexc_good"] = False
        #Check that the number of insertions and deletions are at least equal to the number of molecules
        if insert_val < N_mols or delete_val < N_mols:
            print(f"Job {job.id}")
            statement += f"Job {job.id} production has a low number of insertions or deletions"  + "\n"
            statement += f"Insert: {insert_val}, Delete: {delete_val}, N_mols: {N_mols}"
            check_dict["Nexc_good"] = False

        # Add gemc_failed to job doc and add no_overlap to job doc
        prod_good = np.all(list(check_dict.values()))

        job.doc.no_overlap = check_dict["no_overlap"]
        job.doc.Nexc_good = check_dict["Nexc_good"]

        ##Add more production sweeps if Nexc is too low
        #TO DO
        
        if not prod_good:
            if statement != "":
                with open("Equil_Output.txt", "a") as f:
                    print(statement, file=f)
            if job.doc.get("use_crit", False):
                job.doc.gemc_failed = True
            else:
                vap_box_mult_str = (
                    "_vbx_" + str(job.doc.vap_box_mult)
                    if "vap_box_mult" in job.doc.keys()
                    else ""
                )
                rst_str = "_rest_{:.4s}".format(job.doc.restart_from) if "restart_from" in job.doc.keys() else ""
                crit_str = "_crit" if job.doc.get("use_crit", False) else "_no_crit"
                results_folder = "results" + crit_str + vap_box_mult_str + rst_str
                delete_data(
                    job,
                    "gemc.eq",
                    mv=True,
                    subfolder=results_folder,
                )
                job.doc.use_crit = True

        


# @Project.post(lambda job: "liq_density_unc" in job.doc)
# @Project.post(lambda job: "vap_density_unc" in job.doc)
# @Project.post(lambda job: "Pvap_unc" in job.doc)
# @Project.post(lambda job: "Hvap_unc" in job.doc)
# @Project.post(lambda job: "liq_enthalpy_unc" in job.doc)
# @Project.post(lambda job: "vap_enthalpy_unc" in job.doc)
# Create operation to delete failed jobs
@ProjectGEMC.label
def gemc_failed(job):
    "Confirm gemc failed"
    return "gemc_failed" in job.doc


@ProjectGEMC.pre(gemc_failed)
@ProjectGEMC.operation
def del_job(job):
    "Delete job if gemc failed"
    job.remove()

@prod_group
@ProjectGEMC.pre.after(run_gemc_prod)
@ProjectGEMC.pre.after(check_prod_data)
@ProjectGEMC.post.isfile("energy.png")
@ProjectGEMC.post(lambda job: "liq_density" in job.doc)
@ProjectGEMC.post(lambda job: "vap_density" in job.doc)
@ProjectGEMC.post(lambda job: "Pvap" in job.doc)
@ProjectGEMC.post(lambda job: "Hvap" in job.doc)
@ProjectGEMC.post(lambda job: "liq_enthalpy" in job.doc)
@ProjectGEMC.post(lambda job: "vap_enthalpy" in job.doc)
@ProjectGEMC.post(lambda job: "nmols_liq" in job.doc)
@ProjectGEMC.post(lambda job: "nmols_vap" in job.doc)
@ProjectGEMC.operation
def calculate_props(job):
    """Calculate the density"""

    import numpy as np
    import pylab as plt

    sys.path.append("../../")
    from block_average.block_average import block_average
    sys.path.remove("../../")

    thermo_props = [
        "energy_total",
        "pressure",
        "volume",
        "nmols",
        "mass_density",
        "enthalpy",
    ]

    with job:
        prod_files1 = sorted(glob.glob("prod.*.box1.prp"))
        prod_files2 = sorted(glob.glob("prod.*.box2.prp"))
        # Concatenate all production files using genfromtxt into one
        df_box1 = np.vstack([np.genfromtxt(f) for f in prod_files1])
        df_box2 = np.vstack([np.genfromtxt(f) for f in prod_files2])
        # df_box1 = np.genfromtxt("prod.out.box1.prp")
        # df_box2 = np.genfromtxt("prod.out.box2.prp")

    energy_col = 1
    density_col = 5
    pressure_col = 2
    enth_col = 6
    n_mols_col = 4

    # pull steps
    steps = df_box1[:, 0]

    # pull energy
    liq_energy = df_box1[:, energy_col]
    vap_energy = df_box2[:, energy_col]

    # pull density and take average
    liq_density = df_box1[:, density_col]
    liq_density_ave = np.mean(liq_density)
    vap_density = df_box2[:, density_col]
    vap_density_ave = np.mean(vap_density)

    # pull vapor pressure and take average
    Pvap = df_box2[:, pressure_col]
    Pvap_ave = np.mean(Pvap)

    # pull enthalpy and take average
    liq_enthalpy = df_box1[:, enth_col]
    liq_enthalpy_ave = np.mean(liq_enthalpy)
    vap_enthalpy = df_box2[:, enth_col]
    vap_enthalpy_ave = np.mean(vap_enthalpy)

    # pull number of moles and take average
    nmols_liq = df_box1[:, n_mols_col]
    nmols_liq_ave = np.mean(nmols_liq)
    nmols_vap = df_box2[:, n_mols_col]
    nmols_vap_ave = np.mean(nmols_vap)

    # calculate enthalpy of vaporization
    Hvap = (vap_enthalpy / nmols_vap) - (liq_enthalpy / nmols_liq)
    Hvap_ave = np.mean(Hvap)

    # save average density
    job.doc.liq_density = liq_density_ave
    job.doc.vap_density = vap_density_ave
    job.doc.Pvap = Pvap_ave
    job.doc.Hvap = Hvap_ave*1000/job.sp.mol_weight # kJ/kg from kJ/mol
    job.doc.liq_enthalpy = liq_enthalpy_ave*1000/job.sp.mol_weight # kJ/kg from kJ/mol
    job.doc.vap_enthalpy = vap_enthalpy_ave*1000/job.sp.mol_weight # kJ/kg from kJ/mol
    job.doc.nmols_liq = nmols_liq_ave
    job.doc.nmols_vap = nmols_vap_ave

    font = {"weight": "normal", "size": 12}

    fig, ax = plt.subplots(1, 1)

    ax.spines["bottom"].set_linewidth(3)
    ax.spines["left"].set_linewidth(3)
    ax.spines["right"].set_linewidth(3)
    ax.spines["top"].set_linewidth(3)

    ax.set_xlabel(r"MC Sweeps")
    ax.set_ylabel("Energy")
    ax.yaxis.tick_left()
    ax.yaxis.set_label_position("left")

    ax.title.set_text(f"Energy vs MC Sweeps @ {job.sp.T} K")
    ax.plot(steps, liq_energy, label="Liquid Energy")
    ax.plot(steps, vap_energy, label="Vapor Energy")
    ax.legend(loc="best")

    with job:
        plt.savefig("energy.png")
        plt.close(fig)

    Props = {
        "liq_density": liq_density,
        "vap_density": vap_density,
        "Pvap": Pvap,
        "Hvap": Hvap,
        "liq_enthalpy": liq_enthalpy,
        "vap_enthalpy": vap_enthalpy,
        "nmols_liq": nmols_liq,
        "nmols_vap": nmols_vap,
    }

    for name, prop in Props.items():
        (means_est, vars_est, vars_err) = block_average(prop)

        with open(job.fn(name + "_blk_avg.txt"), "w") as ferr:
            ferr.write("# nblk_ops, mean, vars, vars_err\n")
            for nblk_ops, (mean_est, var_est, var_err) in enumerate(
                zip(means_est, vars_est, vars_err)
            ):
                ferr.write(
                    "{}\t{}\t{}\t{}\n".format(nblk_ops, mean_est, var_est, var_err)
                )

        job.doc[name + "_unc"] = np.max(np.sqrt(vars_est))


@ProjectGEMC.label
def plot_finished(job):
    "Confirm plots have been made"
    import numpy as np
    import os

    last_plot = job.fn(f"all-energy-{job.sp.T}.png")
    if os.path.exists(last_plot):
        completed = True
    else:
        completed = False

    return completed


# @ProjectGEMC.pre.after(run_gemc)
# @ProjectGEMC.post(plot_finished)
# @ProjectGEMC.operation
# def plot(job):
#     import pandas as pd
#     import pylab as plt

#     with job:

#         nvt_box1 = pd.read_table("nvt.eq.out.prp", sep="\s+", names=["step", "energy", "pressure"], skiprows=3)
#         npt_box1 = pd.read_table("npt.eq.out.prp", sep="\s+", names=["step", "energy", "pressure", "density"], skiprows=3)
#         gemc_eq_box1 = pd.read_table("prod.out.box1.prp", sep="\s+", names=["step", "energy", "pressure", "volume", "nmols", "density", "enthalpy"], skiprows=3)
#         gemc_eq_box2 = pd.read_table("prod.out.box2.prp", sep="\s+", names=["step", "energy", "pressure", "volume", "nmols", "density", "enthalpy"], skiprows=3)
#         gemc_prod_box1 = pd.read_table("prod.out.box1.prp", sep="\s+", names=["step", "energy", "pressure", "volume", "nmols", "density", "enthalpy"], skiprows=3)
#         gemc_prod_box2 = pd.read_table("prod.out.box2.prp", sep="\s+", names=["step", "energy", "pressure", "volume", "nmols", "density", "enthalpy"], skiprows=3)

#     font = {'weight' : 'normal',
#                     'size'   : 12}


#     #####################
#     # GEMC Vapor Pressure
#     #####################

#     fig, ax = plt.subplots(1, 1)

#     ax.spines["bottom"].set_linewidth(3)
#     ax.spines["left"].set_linewidth(3)
#     ax.spines["right"].set_linewidth(3)
#     ax.spines["top"].set_linewidth(3)

#     ax.set_xlabel(r'MC steps or sweeps')
#     ax.set_ylabel('Pressure (bar)')
#     ax.yaxis.tick_left()
#     ax.yaxis.set_label_position('left')

#     ax.title.set_text(f"Vapor pressure vs MC Steps or Sweeps @ {job.sp.T} K")
#     ax.plot(gemc_eq_box2["step"][20:], gemc_eq_box2["pressure"][20:], label='GEMC-eq', color='red')
#     ax.plot(gemc_prod_box2["step"], gemc_prod_box2["pressure"], label='GEMC-prod', color='indianred')

#     ax.legend(loc="best")
#     with job:
#         plt.savefig(f"gemc-pvap-{job.sp.T}.png")
#         plt.close(fig)

#     #####################
#     # GEMC nmols
#     #####################

#     fig, ax = plt.subplots(1, 1)

#     ax.spines["bottom"].set_linewidth(3)
#     ax.spines["left"].set_linewidth(3)
#     ax.spines["right"].set_linewidth(3)
#     ax.spines["top"].set_linewidth(3)

#     ax.set_xlabel(r'MC steps or sweeps')
#     ax.set_ylabel('Number of molecules')
#     ax.yaxis.tick_left()
#     ax.yaxis.set_label_position('left')

#     ax.title.set_text(f"Number of molecules vs MC Steps or Sweeps @ {job.sp.T} K")
#     ax.plot(gemc_eq_box1["step"], gemc_eq_box1["nmols"], label='GEMC-eq-box1', color='blue')
#     ax.plot(gemc_eq_box2["step"], gemc_eq_box2["nmols"], label='GEMC-eq-box2', color='red')
#     ax.plot(gemc_prod_box1["step"], gemc_prod_box1["nmols"], label='GEMC-prod-box1', color='royalblue')
#     ax.plot(gemc_prod_box2["step"], gemc_prod_box2["nmols"], label='GEMC-prod-box2', color='indianred')

#     ax.legend(loc="best")
#     with job:
#         plt.savefig(f"gemc-nmols-{job.sp.T}.png")
#         plt.close(fig)

#     #####################
#     # GEMC volume
#     #####################

#     fig, ax = plt.subplots(1, 1)

#     ax.spines["bottom"].set_linewidth(3)
#     ax.spines["left"].set_linewidth(3)
#     ax.spines["right"].set_linewidth(3)
#     ax.spines["top"].set_linewidth(3)

#     ax.set_xlabel(r'MC steps or sweeps')
#     ax.set_ylabel('Volume $\AA^3$')
#     ax.yaxis.tick_left()
#     ax.yaxis.set_label_position('left')

#     ax.title.set_text(f"Volume vs MC Steps or Sweeps @ {job.sp.T} K")
#     ax.plot(gemc_eq_box1["step"], gemc_eq_box1["volume"], label='GEMC-eq-box1', color='blue')
#     ax.plot(gemc_eq_box2["step"], gemc_eq_box2["volume"], label='GEMC-eq-box2', color='red')
#     ax.plot(gemc_prod_box1["step"], gemc_prod_box1["volume"], label='GEMC-prod-box1', color='royalblue')
#     ax.plot(gemc_prod_box2["step"], gemc_prod_box2["volume"], label='GEMC-prod-box2', color='indianred')

#     ax.legend(loc="best")
#     with job:
#         plt.savefig(f"gemc-volume-{job.sp.T}.png")
#         plt.close(fig)

#     #####################
#     # GEMC density
#     #####################

#     fig, ax = plt.subplots(1, 1)

#     ax.spines["bottom"].set_linewidth(3)
#     ax.spines["left"].set_linewidth(3)
#     ax.spines["right"].set_linewidth(3)
#     ax.spines["top"].set_linewidth(3)

#     ax.set_xlabel(r'MC steps or sweeps')
#     ax.set_ylabel('Density $(kg / m^3)$')
#     ax.yaxis.tick_left()
#     ax.yaxis.set_label_position('left')

#     ax.title.set_text(f"Density vs MC Steps or Sweeps @ {job.sp.T} K")
#     ax.plot(gemc_eq_box1["step"], gemc_eq_box1["density"], label='GEMC-eq-box1', color='blue')
#     ax.plot(gemc_eq_box2["step"], gemc_eq_box2["density"], label='GEMC-eq-box2', color='red')
#     ax.plot(gemc_prod_box1["step"], gemc_prod_box1["density"], label='GEMC-prod-box1', color='royalblue')
#     ax.plot(gemc_prod_box2["step"], gemc_prod_box2["density"], label='GEMC-prod-box2', color='indianred')

#     ax.legend(loc="best")
#     with job:
#         plt.savefig(f"gemc-density-{job.sp.T}.png")
#         plt.close(fig)

#     #####################
#     # GEMC enthalpy
#     #####################

#     fig, ax = plt.subplots(1, 1)

#     ax.spines["bottom"].set_linewidth(3)
#     ax.spines["left"].set_linewidth(3)
#     ax.spines["right"].set_linewidth(3)
#     ax.spines["top"].set_linewidth(3)

#     ax.set_xlabel(r'MC steps or sweeps')
#     ax.set_ylabel('Enthalpy (kJ/mol-ext)')
#     ax.yaxis.tick_left()
#     ax.yaxis.set_label_position('left')

#     ax.title.set_text(f"Enthalpy vs MC Steps or Sweeps @ {job.sp.T} K")
#     ax.plot(gemc_eq_box1["step"], gemc_eq_box1["enthalpy"], label='GEMC-eq-box1', color='blue')
#     ax.plot(gemc_eq_box2["step"], gemc_eq_box2["enthalpy"], label='GEMC-eq-box2', color='red')
#     ax.plot(gemc_prod_box1["step"], gemc_prod_box1["enthalpy"], label='GEMC-prod-box1', color='royalblue')
#     ax.plot(gemc_prod_box2["step"], gemc_prod_box2["enthalpy"], label='GEMC-prod-box2', color='indianred')

#     ax.legend(loc="best")
#     with job:
#         plt.savefig(f"gemc-enthalpy-{job.sp.T}.png")
#         plt.close(fig)


#     #############
#     # NPT-Density
#     #############

#     fig, ax = plt.subplots(1, 1)

#     ax.spines["bottom"].set_linewidth(3)
#     ax.spines["left"].set_linewidth(3)
#     ax.spines["right"].set_linewidth(3)
#     ax.spines["top"].set_linewidth(3)

#     ax.set_xlabel(r'MC steps or sweeps')
#     ax.set_ylabel('Density $(kg / m^3)$')
#     ax.yaxis.tick_left()
#     ax.yaxis.set_label_position('left')

#     ax.title.set_text(f"NPT Density vs MC Steps or Sweeps @ {job.sp.T} K")
#     ax.plot(npt_box1["step"], npt_box1["density"], label='NpT')

#     ax.legend(loc="best")
#     with job:
#         plt.savefig(f"npt-density-{job.sp.T}.png")
#         plt.close(fig)

#     # Shift steps so that we get an overall plot of energy across
#     # different workflow steps

#     npt_box1["step"] += nvt_box1["step"].iloc[-1]
#     gemc_eq_box1["step"] += npt_box1["step"].iloc[-1]
#     gemc_eq_box2["step"] += npt_box1["step"].iloc[-1]
#     gemc_prod_box1["step"] += npt_box1["step"].iloc[-1]
#     gemc_prod_box2["step"] += npt_box1["step"].iloc[-1]

#     #############
#     # Energy
#     #############

#     fig, ax = plt.subplots(1, 1)

#     ax.spines["bottom"].set_linewidth(3)
#     ax.spines["left"].set_linewidth(3)
#     ax.spines["right"].set_linewidth(3)
#     ax.spines["top"].set_linewidth(3)

#     ax.set_xlabel(r'MC steps or sweeps')
#     ax.set_ylabel('Energy (kJ/mol-ext)')
#     ax.yaxis.tick_left()
#     ax.yaxis.set_label_position('left')

#     ax.title.set_text(f"Liquid Energy vs MC Steps or Sweeps @ {job.sp.T} K")
#     ax.plot(nvt_box1["step"][20:], nvt_box1["energy"][20:], label='NVT', color="black")

#     ax.plot(npt_box1["step"], npt_box1["energy"], label='NpT', color="gray")
#     ax.plot(gemc_eq_box1["step"], gemc_eq_box1["energy"], label='GEMC-eq-box1', color="blue")
#     ax.plot(gemc_prod_box1["step"], gemc_prod_box1["energy"], label='GEMC-prod-box1', color="royalblue")
#     ax.plot(gemc_eq_box2["step"], gemc_eq_box2["energy"], label='GEMC-eq-box2', color="red")
#     ax.plot(gemc_prod_box2["step"], gemc_prod_box2["energy"], label='GEMC-prod-box2', color="indianred")

#     ax.legend(loc="best")
#     with job:
#         plt.savefig(f"all-energy-{job.sp.T}.png")
#         plt.close(fig)


#####################################################################
################# HELPER FUNCTIONS BEYOND THIS POINT ################
#####################################################################
def _get_custom_args(job):
    # Define custom args
    # See page below for all options
    # https://mosdef-cassandra.readthedocs.io/en/latest/guides/kwargs.html
    custom_args = {
        "vdw_style": "lj",
        "cutoff_style": "cut_tail",
        "vdw_cutoff": 12.0 * u.angstrom,
        "charge_style": "ewald",
        "charge_cutoff": 12.0 * u.angstrom,
        "ewald_accuracy": 1.0e-5,
        "mixing_rule": "lb",
        "units": "sweeps",
        "steps_per_sweep": job.sp.N_liq,
        "coord_freq": 500,
        "prop_freq": 10,
    }

    custom_args_gemc = copy.deepcopy(custom_args)
    custom_args_gemc["steps_per_sweep"] = job.sp.N_liq + job.sp.N_vap
    custom_args_gemc["vdw_cutoff_box1"] = custom_args["vdw_cutoff"]
    custom_args_gemc["charge_cutoff_box1"] = custom_args["charge_cutoff"]

    return custom_args, custom_args_gemc


def _get_molec_dicts():
    # Load class properies for each training and testing molecule
    mol_names = ["EG" , "Gly", "ACN", "MeOH", "DMSO", "THF", "DCM", "DEC", "DMF"]
    molec_dict = esolvs.make_dict(mol_names)
    return molec_dict


def _get_class_from_molecule(molecule_name):
    molec_dict = _get_molec_dicts()
    return {molecule_name: molec_dict[molecule_name]}


def _get_xml_from_molecule(molecule_name):
    if molecule_name == "EG":
        molec_xml_function = __generate_EG_xml
    elif molecule_name == "Gly":
        molec_xml_function = __generate_Gly_xml
    elif molecule_name == "ACN":
        molec_xml_function = __generate_ACN_xml
    elif molecule_name == "MeOH":
        molec_xml_function = __generate_MeOH_xml
    elif molecule_name == "DMSO":
        molec_xml_function = __generate_DMSO_xml
    elif molecule_name == "THF":
        molec_xml_function = __generate_THF_xml
    elif molecule_name == "DCM":
        molec_xml_function = __generate_DCM_xml
    elif molecule_name == "DEC":
        molec_xml_function = __generate_DEC_xml
    elif molecule_name == "DMF":
        molec_xml_function = __generate_DMF_xml
    elif molecule_name == "R125":
        molec_xml_function = __generate_R125_xml
    else:
        raise ValueError("Molecule name not recognized")
    return molec_xml_function


def __generate_EG_xml(job):
    content = """<ForceField>
 <AtomTypes>
   <Type name="C1" class="c3" element="C" mass="12.01" def="[C;X4]" desc="Sp3 C"/>
   <Type name="H1" class="h1" element="H" mass="1.008" def="H[C;X4]([N,O,F,Cl,Br,I,S])" desc="H bonded to aliphatic carbon with 1 d. group"/>
   <Type name="O1" class="oh" element="O" mass="16.0" def="[O;X2]H" desc="Oxygen in hydroxyl group"/>
   <Type name="H2" class="ho" element="H" mass="1.008" def="HO" desc="Hydroxyl group"/>
 </AtomTypes>
 <HarmonicBondForce>
  <Bond class1="c3" class2="h1" length="0.1093" k="281080.370"/>
  <Bond class1="c3" class2="c3" length="0.1535" k="253634.331"/>
  <Bond class1="c3" class2="oh" length="0.1426" k="262838.440"/>
  <Bond class1="ho" class2="oh" length="0.0974" k="309281.363"/>
 </HarmonicBondForce>
 <HarmonicAngleForce>
  <Angle class1="c3" class2="c3" class3="h1" angle="1.921" k="387.936"/>
  <Angle class1="c3" class2="c3" class3="oh" angle="1.910" k="566.680"/>
  <Angle class1="c3" class2="oh" class3="ho" angle="1.888" k="394.056"/>
  <Angle class1="h1" class2="c3" class3="h1" angle="1.912" k="327.856"/>
  <Angle class1="h1" class2="c3" class3="oh" angle="1.918" k="426.515"/>
 </HarmonicAngleForce>
 <PeriodicTorsionForce>
  <Proper class1="c3" class2="c3" class3="oh" class4="ho" periodicity1="3" phase1="0.0" k1="0.669" periodicity2="1" phase2="0.0" k2="1.046"/>
  <Proper class1="h1" class2="c3" class3="c3" class4="h1" periodicity1="3" phase1="0.0" k1="0.653"/>
  <Proper class1="h1" class2="c3" class3="c3" class4="oh" periodicity1="3" phase1="0.0" k1="0.000" periodicity2="1" phase2="0.0" k2="1.046"/>
  <Proper class1="h1" class2="c3" class3="oh" class4="ho" periodicity1="3" phase1="0.0" k1="0.699"/>
  <Proper class1="oh" class2="c3" class3="c3" class4="h1" periodicity1="3" phase1="0.0" k1="0.000" periodicity2="1" phase2="0.0" k2="1.046"/>
  <Proper class1="oh" class2="c3" class3="c3" class4="oh" periodicity1="3" phase1="0.0" k1="0.602" periodicity2="2" phase2="0.0" k2="4.916"/>
 </PeriodicTorsionForce>
 <NonbondedForce coulomb14scale="0.8333" lj14scale="0.5">
  <Atom type="C1" charge="0.299206" sigma="{sigma_C1:0.6f}" epsilon="{epsilon_C1:0.6f}"/>
  <Atom type="H1" charge="0.002766" sigma="{sigma_H1:0.6f}" epsilon="{epsilon_H1:0.6f}"/>
  <Atom type="O1" charge="-0.731599" sigma="{sigma_O1:0.6f}" epsilon="{epsilon_O1:0.6f}"/>
  <Atom type="H2" charge="0.426861" sigma="{sigma_H2:0.6f}" epsilon="{epsilon_H2:0.6f}"/>
 </NonbondedForce>
</ForceField>""".format(
        sigma_C1=job.sp.sigma_C1,
        sigma_H1=job.sp.sigma_H1,
        sigma_O1=job.sp.sigma_O1,
        sigma_H2=job.sp.sigma_H2,
        epsilon_C1=job.sp.epsilon_C1,
        epsilon_H1=job.sp.epsilon_H1,
        epsilon_O1=job.sp.epsilon_O1,
        epsilon_H2=job.sp.epsilon_H2,
    )
    return content

def __generate_R125_xml(job):
    content = """<ForceField>
 <AtomTypes>
  <Type name="C1" class="c3" element="C" mass="12.011" def="C(C)(H)(F)(F)" desc="carbon bonded to 2 Fs, an H, and another carbon"/>
  <Type name="C2" class="c3" element="C" mass="12.011" def="C(C)(F)(F)(F)" desc="carbon bonded to 3 Fs and another carbon"/>
  <Type name="F1" class="f" element="F" mass="18.998" def="FC(C)(F)H" desc="F bonded to C1"/>
  <Type name="H1" class="h2" element="H" mass="1.008" def="H[C;X4]([N,O,F,Cl,Br,I,S])[N,O,F,Cl,Br,I,S]" desc="H bonded to aliphatic carbon with 2 d. group"/>
  <Type name="F2" class="f" element="F" mass="18.998" def="FC(C)(F)F" desc="F bonded to C2"/>
 </AtomTypes>
 <HarmonicBondForce>
  <Bond class1="c3" class2="c3" length="0.15375" k="251793.143"/>
  <Bond class1="c3" class2="f" length="0.13479" k="298653.950"/>
  <Bond class1="c3" class2="h2" length="0.10961" k="277566.579"/>
 </HarmonicBondForce>
 <HarmonicAngleForce>
  <Angle class1="c3" class2="c3" class3="f" angle="1.90660" k="553.125"/>
  <Angle class1="c3" class2="c3" class3="h2" angle="1.92370" k="386.602"/>
  <Angle class1="f" class2="c3" class3="h2" angle="1.89874" k="427.605"/>
  <Angle class1="f" class2="c3" class3="f" angle="1.87379" k="593.291"/>
 </HarmonicAngleForce>
 <PeriodicTorsionForce>
  <Proper class1="f" class2="c3" class3="c3" class4="f" periodicity1="3" phase1="0.0" k1="0.0" periodicity2="1" phase2="3.141592654" k2="5.02077111"/>
  <Proper class1="h2" class2="c3" class3="c3" class4="f" periodicity1="3" phase1="0.0" k1="0.65085610"/>
 </PeriodicTorsionForce>
 <NonbondedForce coulomb14scale="0.8333" lj14scale="0.5">
  <Atom type="C1" charge="0.224067" sigma="{sigma_C1:0.6f}" epsilon="{epsilon_C1:0.6f}"/>
  <Atom type="C2" charge="0.500886" sigma="{sigma_C2:0.6f}" epsilon="{epsilon_C2:0.6f}"/>
  <Atom type="F1" charge="-0.167131" sigma="{sigma_F1:0.6f}" epsilon="{epsilon_F1:0.6f}"/>
  <Atom type="H1" charge="0.121583" sigma="{sigma_H1:0.6f}" epsilon="{epsilon_H1:0.6f}"/>
  <Atom type="F2" charge="-0.170758" sigma="{sigma_F2:0.6f}" epsilon="{epsilon_F2:0.6f}"/>
 </NonbondedForce>
</ForceField>""".format(
        sigma_C1=job.sp.sigma_C1,
        sigma_C2=job.sp.sigma_C2,
        sigma_H1=job.sp.sigma_H1,
        sigma_F1=job.sp.sigma_F1,
        sigma_F2=job.sp.sigma_F2,
        epsilon_C1=job.sp.epsilon_C1,
        epsilon_C2=job.sp.epsilon_C2,
        epsilon_H1=job.sp.epsilon_H1,
        epsilon_F1=job.sp.epsilon_F1,
        epsilon_F2=job.sp.epsilon_F2,
    )
    return content

def __generate_Gly_xml(job):
    content = """<ForceField>
 <AtomTypes>
   <Type name="C1" class="c3" element="C" mass="12.01" def="[C;X4](C)(O)(H)H" desc="Sp3 C"/>
   <Type name="H1" class="h1" element="H" mass="1.008" def="[H][C;%C1]" desc="H bonded to aliphatic carbon with 1 d. group"/>
   <Type name="C2" class="c3" element="C" mass="12.01" def="[C;X4](C)(C)(O)H" desc="Sp3 C"/>
   <Type name="H2" class="h1" element="H" mass="1.008" def="[H][C;%C2]" desc="H bonded to aliphatic carbon with 1 d. group"/>
   <Type name="O1" class="oh" element="O" mass="16.0" def="[O;X2][C;%C1]" desc="Oxygen in hydroxyl group"/>
   <Type name="H3" class="ho" element="H" mass="1.008" def="[H][O;%O1]" desc="Hydroxyl group"/>
   <Type name="O2" class="oh" element="O" mass="16.0" def="[O;X2][C;%C2]" desc="Oxygen in hydroxyl group"/>
   <Type name="H4" class="ho" element="H" mass="1.008" def="[H][O;%O2]" desc="Hydroxyl group"/>
 </AtomTypes>
 <HarmonicBondForce>
  <Bond class1="c3" class2="h1" length="0.1093" k="281080.370"/>
  <Bond class1="c3" class2="c3" length="0.1535" k="253634.331"/>
  <Bond class1="c3" class2="oh" length="0.1426" k="262838.440"/>
  <Bond class1="ho" class2="oh" length="0.0974" k="309281.363"/>
 </HarmonicBondForce>
 <HarmonicAngleForce>
  <Angle class1="c3" class2="c3" class3="h1" angle="1.921" k="387.936"/>
  <Angle class1="c3" class2="c3" class3="c3" angle="1.931" k="528.933"/>
  <Angle class1="c3" class2="c3" class3="oh" angle="1.910" k="566.680"/>
  <Angle class1="c3" class2="oh" class3="ho" angle="1.888" k="394.056"/>
  <Angle class1="h1" class2="c3" class3="h1" angle="1.912" k="327.856"/>
  <Angle class1="h1" class2="c3" class3="oh" angle="1.918" k="426.515"/>
 </HarmonicAngleForce>
 <PeriodicTorsionForce>
  <Proper class1="c3" class2="c3" class3="c3" class4="h1" periodicity1="3" phase1="0.0" k1="0.653"/>
  <Proper class1="c3" class2="c3" class3="c3" class4="oh" periodicity1="3" phase1="0.0" k1="0.653"/>
  <Proper class1="c3" class2="c3" class3="oh" class4="ho" periodicity1="3" phase1="0.0" k1="0.669" periodicity2="1" phase2="0.0" k2="1.046"/>
  <Proper class1="h1" class2="c3" class3="c3" class4="c3" periodicity1="3" phase1="0.0" k1="0.653"/>
  <Proper class1="h1" class2="c3" class3="c3" class4="h1" periodicity1="3" phase1="0.0" k1="0.653"/>
  <Proper class1="h1" class2="c3" class3="c3" class4="oh" periodicity1="3" phase1="0.0" k1="0.000" periodicity2="1" phase2="0.0" k2="1.046"/>
  <Proper class1="h1" class2="c3" class3="oh" class4="ho" periodicity1="3" phase1="0.0" k1="0.699"/>
  <Proper class1="oh" class2="c3" class3="c3" class4="c3" periodicity1="3" phase1="0.0" k1="0.653"/>
  <Proper class1="oh" class2="c3" class3="c3" class4="h1" periodicity1="3" phase1="0.0" k1="0.000" periodicity2="1" phase2="0.0" k2="1.046"/>
  <Proper class1="oh" class2="c3" class3="c3" class4="oh" periodicity1="3" phase1="0.0" k1="0.602" periodicity2="2" phase2="0.0" k2="4.916"/>
 </PeriodicTorsionForce>
 <NonbondedForce coulomb14scale="0.8333" lj14scale="0.5">
  <Atom type="C1" charge="0.071924" sigma="{sigma_C1:0.6f}" epsilon="{epsilon_C1:0.6f}"/>
  <Atom type="H1" charge="0.049478" sigma="{sigma_H1:0.6f}" epsilon="{epsilon_H1:0.6f}"/>
  <Atom type="C2" charge="0.489580" sigma="{sigma_C2:0.6f}" epsilon="{epsilon_C2:0.6f}"/>
  <Atom type="H2" charge="0.021557" sigma="{sigma_H2:0.6f}" epsilon="{epsilon_H2:0.6f}"/>
  <Atom type="O1" charge="-0.718944" sigma="{sigma_O1:0.6f}" epsilon="{epsilon_O1:0.6f}"/>
  <Atom type="H3" charge="0.448091" sigma="{sigma_H3:0.6f}" epsilon="{epsilon_H3:0.6f}"/>
  <Atom type="O2" charge="-0.769141" sigma="{sigma_O2:0.6f}" epsilon="{epsilon_O2:0.6f}"/>
  <Atom type="H4" charge="0.457950" sigma="{sigma_H4:0.6f}" epsilon="{epsilon_H4:0.6f}"/>
 </NonbondedForce>
</ForceField>""".format(
        sigma_C1=job.sp.sigma_C1,
        sigma_C2=job.sp.sigma_C2,
        sigma_H1=job.sp.sigma_H1,
        sigma_H2=job.sp.sigma_H2,
        sigma_H3=job.sp.sigma_H3,
        sigma_H4=job.sp.sigma_H4,
        sigma_O1=job.sp.sigma_O1,
        sigma_O2=job.sp.sigma_O2,
        epsilon_C1=job.sp.epsilon_C1,
        epsilon_C2=job.sp.epsilon_C2,
        epsilon_H1=job.sp.epsilon_H1,
        epsilon_H2=job.sp.epsilon_H2,
        epsilon_H3=job.sp.epsilon_H3,
        epsilon_H4=job.sp.epsilon_H4,
        epsilon_O1=job.sp.epsilon_O1,
        epsilon_O2=job.sp.epsilon_O2,
    )
    return content


def __generate_MeOH_xml(job):
    content = """<ForceField>
 <AtomTypes>
   <Type name="C1" class="c3" element="C" mass="12.01" def="[C;X4]" desc="Sp3 C"/>
   <Type name="H1" class="h1" element="H" mass="1.008" def="H[C;X4]([N,O,F,Cl,Br,I,S])" desc="H bonded to aliphatic carbon with 1 d. group"/>
   <Type name="O1" class="oh" element="O" mass="16.0" def="[O;X2]H" desc="Oxygen in hydroxyl group"/>
   <Type name="H2" class="ho" element="H" mass="1.008" def="HO" desc="Hydroxyl group"/>
 </AtomTypes>
 <HarmonicBondForce>
  <Bond class1="c3" class2="h1" length="0.1093" k="281080.370"/>
  <Bond class1="c3" class2="oh" length="0.1426" k="262838.440"/>
  <Bond class1="ho" class2="oh" length="0.0974" k="309281.363"/>
 </HarmonicBondForce>
 <HarmonicAngleForce>
  <Angle class1="c3" class2="oh" class3="ho" angle="1.888" k="394.056"/>
  <Angle class1="h1" class2="c3" class3="h1" angle="1.912" k="327.856"/>
  <Angle class1="h1" class2="c3" class3="oh" angle="1.918" k="426.515"/>
 </HarmonicAngleForce>
 <PeriodicTorsionForce>
  <Proper class1="h1" class2="c3" class3="oh" class4="ho" periodicity1="3" phase1="0.0" k1="0.699"/>
 </PeriodicTorsionForce>
 <NonbondedForce coulomb14scale="0.8333" lj14scale="0.5">
  <Atom type="C1" charge="0.248643" sigma="{sigma_C1:0.6f}" epsilon="{epsilon_C1:0.6f}"/>
  <Atom type="H1" charge="0.002748" sigma="{sigma_H1:0.6f}" epsilon="{epsilon_H1:0.6f}"/>
  <Atom type="O1" charge="-0.672287" sigma="{sigma_O1:0.6f}" epsilon="{epsilon_O1:0.6f}"/>
  <Atom type="H2" charge="0.415400" sigma="{sigma_H2:0.6f}" epsilon="{epsilon_H2:0.6f}"/>
 </NonbondedForce>
</ForceField>""".format(
        sigma_C1=job.sp.sigma_C1,
        sigma_O1=job.sp.sigma_O1,
        sigma_H1=job.sp.sigma_H1,
        sigma_H2=job.sp.sigma_H2,
        epsilon_C1=job.sp.epsilon_C1,
        epsilon_O1=job.sp.epsilon_O1,
        epsilon_H1=job.sp.epsilon_H1,
        epsilon_H2=job.sp.epsilon_H2,
    )
    return content


def __generate_DCM_xml(job):
    content = """<ForceField>
  <AtomTypes>
   <Type name="C1" class="c3" element="C" mass="12.01" def="[C;X4]" desc="Sp3 C"/>
   <Type name="H1" class="h2" element="H" mass="1.008" def="H[C;X4]([N,O,F,Cl,Br,I,S])[N,O,F,Cl,Br,I,S]" desc="H bonded to aliphatic carbon with 2 d. group"/>
   <Type name="Cl1" class="cl" element="Cl" mass="35.45" def="Cl" desc="Chlorine"/>
  </AtomTypes>
  <HarmonicBondForce>
   <Bond class1="c3" class2="h2" length="0.1100" k="273131.744"/>
   <Bond class1="c3" class2="cl" length="0.1786" k="233466.771"/>
  </HarmonicBondForce>
  <HarmonicAngleForce>
   <Angle class1="h2" class2="c3" class3="h2" angle="1.906" k="326.359"/>
   <Angle class1="cl" class2="c3" class3="h2" angle="1.870" k="363.924"/>
   <Angle class1="cl" class2="c3" class3="cl" angle="1.938" k="524.676"/>
  </HarmonicAngleForce>
  <NonbondedForce coulomb14scale="0.833333" lj14scale="0.5">
    <Atom type="C1" charge="-0.375336" sigma="{sigma_C1:0.6f}" epsilon="{epsilon_C1:0.6f}"/>
    <Atom type="H1" charge="0.243662" sigma="{sigma_H1:0.6f}" epsilon="{epsilon_H1:0.6f}"/>
    <Atom type="Cl1" charge="-0.055994" sigma="{sigma_Cl1:0.6f}" epsilon="{epsilon_Cl1:0.6f}"/>
  </NonbondedForce>
</ForceField>
""".format(
        sigma_C1=job.sp.sigma_C1,
        sigma_H1=job.sp.sigma_H1,
        sigma_Cl1=job.sp.sigma_Cl1,
        epsilon_C1=job.sp.epsilon_C1,
        epsilon_H1=job.sp.epsilon_H1,
        epsilon_Cl1=job.sp.epsilon_Cl1,
    )
    return content


def __generate_DMSO_xml(job):
    content = """<ForceField>
 <AtomTypes>
   <Type name="S1" class="s4" element="S" mass="32.06" def="[S;X3]" desc="S with three connected atoms"/>
   <Type name="O1" class="o" element="O" mass="16.0" def="[O;X1]" desc="Oxygen with one connected atom"/>
   <Type name="C1" class="c3" element="C" mass="12.01" def="[C;X4]" desc="Sp3 C"/>
   <Type name="H1" class="h1" element="H" mass="1.008" def="H[C;X4]([N,O,F,Cl,Br,I,S])" desc="H bonded to aliphatic carbon with 1 d. group"/>
 </AtomTypes>
 <HarmonicBondForce>
  <Bond class1="o" class2="s4" length="0.1497" k="375471.133"/>
  <Bond class1="c3" class2="s4" length="0.1807" k="195644.283"/>
  <Bond class1="c3" class2="h1" length="0.1093" k="281080.370"/>
 </HarmonicBondForce>
 <HarmonicAngleForce>
  <Angle class1="h1" class2="c3" class3="s4" angle="1.896" k="454.219"/>
  <Angle class1="c3" class2="s4" class3="o" angle="1.854" k="343.587"/>
  <Angle class1="c3" class2="s4" class3="c3" angle="1.690" k="325.012"/>
  <Angle class1="h1" class2="c3" class3="h1" angle="1.912" k="327.856"/>
 </HarmonicAngleForce>
 <PeriodicTorsionForce>
  <Proper class1="h1" class2="c3" class3="s4" class4="c3" periodicity1="3" phase1="0.0" k1="0.837"/>
  <Proper class1="h1" class2="c3" class3="s4" class4="o" periodicity1="3" phase1="0.0" k1="0.837"/>
 </PeriodicTorsionForce>
 <NonbondedForce coulomb14scale="0.833333" lj14scale="0.5">
  <Atom type="S1" charge="0.357892" sigma="{sigma_S1:0.6f}" epsilon="{epsilon_S1:0.6f}"/>
  <Atom type="O1" charge="-0.485458" sigma="{sigma_O1:0.6f}" epsilon="{epsilon_O1:0.6f}"/>
  <Atom type="C1" charge="-0.492849" sigma="{sigma_C1:0.6f}" epsilon="{epsilon_C1:0.6f}"/>
  <Atom type="H1" charge="0.185544" sigma="{sigma_H1:0.6f}" epsilon="{epsilon_H1:0.6f}"/>
 </NonbondedForce>
</ForceField>""".format(
        sigma_C1=job.sp.sigma_C1,
        sigma_H1=job.sp.sigma_H1,
        sigma_O1=job.sp.sigma_O1,
        sigma_S1=job.sp.sigma_S1,
        epsilon_C1=job.sp.epsilon_C1,
        epsilon_H1=job.sp.epsilon_H1,
        epsilon_O1=job.sp.epsilon_O1,
        epsilon_S1=job.sp.epsilon_S1,
    )
    return content


def __generate_ACN_xml(job):
    content = """<ForceField>
 <AtomTypes>
  <Type name="N1" class="n1" element="N" mass="14.01" def="[N;X1]" desc="Sp N"/>
  <Type name="C1" class="c1" element="C" mass="12.01" def="[C;X2]" desc="Sp C"/>
  <Type name="C2" class="c3" element="C" mass="12.01" def="[C;X4]" desc="Sp3 C"/>
  <Type name="H1" class="hc" element="H" mass="1.008" def="H[C;X4]" desc="H bonded to aliphatic carbon without d. group"/>
 </AtomTypes>
 <HarmonicBondForce>
  <Bond class1="c1" class2="n1" length="0.1138" k="848933.191"/>
  <Bond class1="c1" class2="c3" length="0.1470" k="308193.831"/>
  <Bond class1="c3" class2="hc" length="0.1092" k="282252.709"/>
 </HarmonicBondForce>
 <HarmonicAngleForce>
  <Angle class1="c3" class2="c1" class3="n1" angle="3.11541" k="486.180"/>
  <Angle class1="c1" class2="c3" class3="hc" angle="1.91550" k="403.750"/>
  <Angle class1="hc" class2="c3" class3="hc" angle="1.89106" k="329.951"/>
 </HarmonicAngleForce>
 <PeriodicTorsionForce>
  <Proper class1="n1" class2="c1" class3="c3" class4="hc" periodicity1="2" phase1="3.141592654" k1="0.0"/>
 </PeriodicTorsionForce>
 <NonbondedForce coulomb14scale="0.8333" lj14scale="0.5">
  <Atom type="N1" charge="-0.505798" sigma="{sigma_N1:0.6f}" epsilon="{epsilon_N1:0.6f}"/>
  <Atom type="C1" charge="0.461146" sigma="{sigma_C1:0.6f}" epsilon="{epsilon_C1:0.6f}"/>
  <Atom type="C2" charge="-0.474882" sigma="{sigma_C2:0.6f}" epsilon="{epsilon_C2:0.6f}"/>
  <Atom type="H1" charge="0.173178" sigma="{sigma_H1:0.6f}" epsilon="{epsilon_H1:0.6f}"/>
 </NonbondedForce>
</ForceField>""".format(
        sigma_N1=job.sp.sigma_N1,
        sigma_C1=job.sp.sigma_C1,
        sigma_C2=job.sp.sigma_C2,
        sigma_H1=job.sp.sigma_H1,
        epsilon_N1=job.sp.epsilon_N1,
        epsilon_C1=job.sp.epsilon_C1,
        epsilon_C2=job.sp.epsilon_C2,
        epsilon_H1=job.sp.epsilon_H1,
    )
    return content


def __generate_DMF_xml(job):
    content = """<ForceField>
 <AtomTypes>
  <Type name="N1" class="n" element="N" mass="14.01" def="[N;X3][C;X3][O&X1,S&X1]" desc="Sp2 nitrogen in amide groups"/>
  <Type name="C1" class="c3" element="C" mass="12.01" def="[C;X4]" desc="Sp3 C"/>
  <Type name="H1" class="h1" element="H" mass="1.008" def="H[C;X4]([N,O,F,Cl,Br,I,S])" desc="H bonded to aliphatic carbon with 1 d. group"/>
  <Type name="C2" class="c" element="C" mass="12.01" def="[C;X3][O&X1,S&X1]" desc="Sp2 C carbonyl group"/>
  <Type name="H2" class="h5" element="H" mass="1.008" def="H[C;!X4]([N,O,F,Cl,Br,I,S])([N,O,F,Cl,Br,I,S])" desc="H bonded to non-sp3 carbon with 2 d. group"/>
  <Type name="O1" class="o" element="O" mass="16.0" def="[O;X1]" desc="Oxygen with one connected atom"/>
 </AtomTypes>
 <HarmonicBondForce>
  <Bond class1="c3" class2="n" length="0.1460" k="276645.436"/>
  <Bond class1="c" class2="n" length="0.1345" k="400156.771"/>
  <Bond class1="c3" class2="h1" length="0.1093" k="281080.370"/>
  <Bond class1="c" class2="h5" length="0.1105" k="267273.374"/>
  <Bond class1="c" class2="o" length="0.1214" k="542245.941"/>
 </HarmonicBondForce>
 <HarmonicAngleForce>
  <Angle class1="h1" class2="c3" class3="n" angle="1.90799" k="416.887"/>
  <Angle class1="h5" class2="c" class3="n" angle="1.95808" k="438.405"/>
  <Angle class1="n" class2="c" class3="o" angle="2.12983" k="634.543"/>
  <Angle class1="c3" class2="n" class3="c3" angle="2.01690" k="528.268"/>
  <Angle class1="c3" class2="n" class3="c" angle="2.11796" k="534.886"/>
  <Angle class1="h1" class2="c3" class3="h1" angle="1.91201" k="327.856"/>
  <Angle class1="h5" class2="c" class3="o" angle="2.15129" k="450.943"/>
 </HarmonicAngleForce>
 <PeriodicTorsionForce>
  <Proper class1="h1" class2="c3" class3="n" class4="c" periodicity1="3" phase1="0.0" k1="69.02990737"/>
  <Proper class1="h1" class2="c3" class3="n" class4="c3" periodicity1="3" phase1="0.0" k1="57.15386222"/>
  <Proper class1="h5" class2="c" class3="n" class4="c3" periodicity1="2" phase1="3.141592654" k1="10.46000910"/>
  <Proper class1="o" class2="c" class3="n" class4="c3" periodicity1="2" phase1="3.141592654" k1="10.46000910"/>
  <Improper class1="c" class2="c3" class3="n" class4="c3" periodicity1="2" phase1="3.141592654" k1="4.60238738"/>
  <Improper class1="h5" class2="n" class3="c" class4="o" periodicity1="2" phase1="3.141592654" k1="43.93195509"/>
 </PeriodicTorsionForce>
 <NonbondedForce coulomb14scale="0.8333" lj14scale="0.5">
  <Atom type="N1" charge="0.086913" sigma="{sigma_N1:0.6f}" epsilon="{epsilon_N1:0.6f}"/>
  <Atom type="C1" charge="-0.310731" sigma="{sigma_C1:0.6f}" epsilon="{epsilon_C1:0.6f}"/>
  <Atom type="H1" charge="0.115127" sigma="{sigma_H1:0.6f}" epsilon="{epsilon_H1:0.6f}"/>
  <Atom type="C2" charge="0.321534" sigma="{sigma_C2:0.6f}" epsilon="{epsilon_C2:0.6f}"/>
  <Atom type="H2" charge="0.044705" sigma="{sigma_H2:0.6f}" epsilon="{epsilon_H2:0.6f}"/>
  <Atom type="O1" charge="-0.522452" sigma="{sigma_O1:0.6f}" epsilon="{epsilon_O1:0.6f}"/>
 </NonbondedForce>
</ForceField>""".format(
        sigma_N1=job.sp.sigma_N1,
        sigma_C1=job.sp.sigma_C1,
        sigma_C2=job.sp.sigma_C2,
        sigma_H1=job.sp.sigma_H1,
        sigma_H2=job.sp.sigma_H2,
        sigma_O1=job.sp.sigma_O1,
        epsilon_N1=job.sp.epsilon_N1,
        epsilon_C1=job.sp.epsilon_C1,
        epsilon_C2=job.sp.epsilon_C2,
        epsilon_H1=job.sp.epsilon_H1,
        epsilon_H2=job.sp.epsilon_H2,
        epsilon_O1=job.sp.epsilon_O1,
    )
    return content


def __generate_DEC_xml(job):
    content = """<ForceField>
 <AtomTypes>
   <Type name="C1" class="c3" element="C" mass="12.01" def="[C;X4](C)(H)(H)H" desc="Sp3 C"/>
   <Type name="H1" class="hc" element="H" mass="1.008" def="[H][C;%C1]" desc="H bonded to aliphatic carbon without d. group"/>
   <Type name="C2" class="c3" element="C" mass="12.01" def="[C;X4](C)(O)(H)H" desc="Sp3 C"/>
   <Type name="H2" class="h1" element="H" mass="1.008" def="[H][C;%C2]" desc="H bonded to aliphatic carbon with 1 d. group"/>
   <Type name="O1" class="os" element="O" mass="16.0" def="[O;X2]([!H])[!H]" desc="Ether and ester oxygen"/>
   <Type name="C3" class="c" element="C" mass="12.01" def="[C;X3][O&X1,S&X1]" desc="Sp2 C carbonyl group"/>
   <Type name="O2" class="o" element="O" mass="16.0" def="[O;X1]" desc="Oxygen with one connected atom"/>
 </AtomTypes>
 <HarmonicBondForce>
  <Bond class1="c3" class2="hc" length="0.1092" k="282252.709"/>
  <Bond class1="c3" class2="c3" length="0.1535" k="253634.331"/>
  <Bond class1="c3" class2="h1" length="0.1093" k="281080.370"/>
  <Bond class1="c3" class2="os" length="0.1439" k="252295.702"/>
  <Bond class1="c" class2="os" length="0.1343" k="344175.498"/>
  <Bond class1="c" class2="o" length="0.1214" k="542245.941"/>
 </HarmonicBondForce>
 <HarmonicAngleForce>
  <Angle class1="c3" class2="c3" class3="h1" angle="1.921" k="387.936"/>
  <Angle class1="c3" class2="c3" class3="os" angle="1.892" k="567.179"/>
  <Angle class1="hc" class2="c3" class3="hc" angle="1.891" k="329.951"/>
  <Angle class1="c3" class2="c3" class3="hc" angle="1.921" k="388.019"/>
  <Angle class1="c3" class2="os" class3="c" angle="2.010" k="532.458"/>
  <Angle class1="h1" class2="c3" class3="h1" angle="1.912" k="327.856"/>
  <Angle class1="h1" class2="c3" class3="os" angle="1.899" k="425.434"/>
  <Angle class1="os" class2="c" class3="o" angle="2.153" k="635.375"/>
  <Angle class1="os" class2="c" class3="os" angle="1.944" k="639.731"/>
  <Angle class1="c" class2="os" class3="c3" angle="2.010" k="532.458"/>
  <Angle class1="o" class2="c" class3="os" angle="2.153" k="635.375"/>
 </HarmonicAngleForce>
 <PeriodicTorsionForce>
  <Proper class1="c3" class2="c3" class3="os" class4="c" periodicity1="3" phase1="0.0" k1="1.602" periodicity2="1" phase2="3.1" k2="3.347"/>
  <Proper class1="h1" class2="c3" class3="c3" class4="hc" periodicity1="3" phase1="0.0" k1="0.653"/>
  <Proper class1="h1" class2="c3" class3="os" class4="c" periodicity1="3" phase1="0.0" k1="1.602"/>
  <Proper class1="hc" class2="c3" class3="c3" class4="h1" periodicity1="3" phase1="0.0" k1="0.653"/>
  <Proper class1="hc" class2="c3" class3="c3" class4="os" periodicity1="3" phase1="0.0" k1="0.000" periodicity2="1" phase2="0.0" k2="1.046"/>
  <Proper class1="o" class2="c" class3="os" class4="c3" periodicity1="2" phase1="3.1" k1="11.297" periodicity2="1" phase2="3.1" k2="5.858"/>
  <Proper class1="o" class2="os" class3="c" class4="os" periodicity1="2" phase1="180.0" k1="43.932"/>
  <Proper class1="os" class2="c" class3="os" class4="c3" periodicity1="2" phase1="3.1" k1="11.297"/>
  <Proper class1="os" class2="c3" class3="c3" class4="hc" periodicity1="3" phase1="0.0" k1="0.000" periodicity2="1" phase2="0.0" k2="1.046"/>
 </PeriodicTorsionForce>
 <NonbondedForce coulomb14scale="0.8333" lj14scale="0.5">
  <Atom type="C1" charge="-0.453302" sigma="{sigma_C1:0.6f}" epsilon="{epsilon_C1:0.6f}"/>
  <Atom type="H1" charge="0.119642" sigma="{sigma_H1:0.6f}" epsilon="{epsilon_H1:0.6f}"/>
  <Atom type="C2" charge="0.496179" sigma="{sigma_C2:0.6f}" epsilon="{epsilon_C2:0.6f}"/>
  <Atom type="H2" charge="-0.031568" sigma="{sigma_H2:0.6f}" epsilon="{epsilon_H2:0.6f}"/>
  <Atom type="O1" charge="-0.541363" sigma="{sigma_O1:0.6f}" epsilon="{epsilon_O1:0.6f}"/>
  <Atom type="C3" charge="1.055588" sigma="{sigma_C3:0.6f}" epsilon="{epsilon_C3:0.6f}"/>
  <Atom type="O2" charge="-0.650193" sigma="{sigma_O2:0.6f}" epsilon="{epsilon_O2:0.6f}"/>
 </NonbondedForce>
</ForceField>""".format(
        sigma_C1=job.sp.sigma_C1,
        sigma_H1=job.sp.sigma_H1,
        sigma_C2=job.sp.sigma_C2,
        sigma_H2=job.sp.sigma_H2,
        sigma_O1=job.sp.sigma_O1,
        sigma_C3=job.sp.sigma_C3,
        sigma_O2=job.sp.sigma_O2,
        epsilon_C1=job.sp.epsilon_C1,
        epsilon_H1=job.sp.epsilon_H1,
        epsilon_C2=job.sp.epsilon_C2,
        epsilon_H2=job.sp.epsilon_H2,
        epsilon_O1=job.sp.epsilon_O1,
        epsilon_C3=job.sp.epsilon_C3,
        epsilon_O2=job.sp.epsilon_O2,
    )
    return content


def __generate_THF_xml(job):
    content = """<ForceField>
 <AtomTypes>
   <Type name="C1" class="c3" element="C" mass="12.01" def="[C;r5][O]" desc="Sp3 C"/>
   <Type name="O1" class="os" element="O" mass="16.0" def="[O;X2]([!H])[!H]" desc="Ether and ester oxygen"/>
   <Type name="C2" class="c3" element="C" mass="12.01" def="[C;r5][C;r5][O]" desc="Sp3 C"/>
   <Type name="H1" class="h1" element="H" mass="1.008" def="[H][C;r5][O]" desc="H bonded to aliphatic carbon with 1 d. group"/>
   <Type name="H2" class="hc" element="H" mass="1.008" def="[H][C;r5][C;r5][O]" desc="H bonded to aliphatic carbon without d. group"/>
 </AtomTypes>
 <HarmonicBondForce>
  <Bond class1="c3" class2="os" length="0.1439" k="252295.702"/>
  <Bond class1="c3" class2="c3" length="0.1535" k="253634.331"/>
  <Bond class1="c3" class2="h1" length="0.1093" k="281080.370"/>
  <Bond class1="c3" class2="hc" length="0.1092" k="282252.709"/>
 </HarmonicBondForce>
 <HarmonicAngleForce>
  <Angle class1="c3" class2="os" class3="c3" angle="1.963" k="522.082"/>
  <Angle class1="c3" class2="c3" class3="c3" angle="1.931" k="528.933"/>
  <Angle class1="c3" class2="c3" class3="hc" angle="1.921" k="388.019"/>
  <Angle class1="c3" class2="c3" class3="os" angle="1.892" k="567.179"/>
  <Angle class1="h1" class2="c3" class3="os" angle="1.899" k="425.434"/>
  <Angle class1="c3" class2="c3" class3="h1" angle="1.921" k="387.936"/>
  <Angle class1="h1" class2="c3" class3="h1" angle="1.912" k="327.856"/>
  <Angle class1="hc" class2="c3" class3="hc" angle="1.891" k="329.951"/>
 </HarmonicAngleForce>
 <PeriodicTorsionForce>
  <Proper class1="c3" class2="c3" class3="c3" class4="c3" periodicity1="3" phase1="0.0" k1="0.753" periodicity2="2" phase2="3.1" k2="1.046"/>
  <Proper class1="c3" class2="c3" class3="c3" class4="hc" periodicity1="3" phase1="0.0" k1="0.669"/>
  <Proper class1="c3" class2="c3" class3="os" class4="c3" periodicity1="3" phase1="0.0" k1="1.602" periodicity2="2" phase2="3.1" k2="0.418"/>
  <Proper class1="h1" class2="c3" class3="c3" class4="c3" periodicity1="3" phase1="0.0" k1="0.653"/>
  <Proper class1="h1" class2="c3" class3="c3" class4="hc" periodicity1="3" phase1="0.0" k1="0.653"/>
  <Proper class1="h1" class2="c3" class3="os" class4="c3" periodicity1="3" phase1="0.0" k1="1.602"/>
  <Proper class1="hc" class2="c3" class3="c3" class4="c3" periodicity1="3" phase1="0.0" k1="0.669"/>
  <Proper class1="hc" class2="c3" class3="c3" class4="hc" periodicity1="3" phase1="0.0" k1="0.628"/>
  <Proper class1="os" class2="c3" class3="c3" class4="c3" periodicity1="3" phase1="0.0" k1="0.653"/>
  <Proper class1="os" class2="c3" class3="c3" class4="hc" periodicity1="3" phase1="0.0" k1="0.000" periodicity2="1" phase2="0.0" k2="1.046"/>
 </PeriodicTorsionForce>
 <NonbondedForce coulomb14scale="0.8333" lj14scale="0.5">
  <Atom type="C1" charge="0.235221" sigma="{sigma_C1:0.6f}" epsilon="{epsilon_C1:0.6f}"/>
  <Atom type="O1" charge="-0.442876" sigma="{sigma_O1:0.6f}" epsilon="{epsilon_O1:0.6f}"/>
  <Atom type="C2" charge="-0.049525" sigma="{sigma_C2:0.6f}" epsilon="{epsilon_C2:0.6f}"/>
  <Atom type="H1" charge="-0.003069" sigma="{sigma_H1:0.6f}" epsilon="{epsilon_H1:0.6f}"/>
  <Atom type="H2" charge="0.020940" sigma="{sigma_H2:0.6f}" epsilon="{epsilon_H2:0.6f}"/>
 </NonbondedForce>
</ForceField>""".format(
        sigma_C1=job.sp.sigma_C1,
        sigma_H1=job.sp.sigma_H1,
        sigma_C2=job.sp.sigma_C2,
        sigma_H2=job.sp.sigma_H2,
        sigma_O1=job.sp.sigma_O1,
        epsilon_C1=job.sp.epsilon_C1,
        epsilon_H1=job.sp.epsilon_H1,
        epsilon_C2=job.sp.epsilon_C2,
        epsilon_H2=job.sp.epsilon_H2,
        epsilon_O1=job.sp.epsilon_O1,
    )
    return content


def has_checkpoint(run_name):
    """Check whether there is a checkpoint for run_name."""
    fname = run_name + ".out.chk"
    return os.path.exists(fname)


def check_complete(run_name):
    """Check whether MoSDeF Cassandra simulation with run_name or its last restart has completed."""
    complete = False
    fname = run_name + ".out.log"
    loglist = list_with_restarts(fname)
    if not loglist:
        return complete
    with loglist[-1].open() as f:
        for line in f:
            if "Cassandra simulation complete" in line:
                complete = True
                break
    return complete


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
                break
    return run_value


def extract_cubic_values(job_init, file_path):
    cubic_values = []
    with open(job_init.fn(file_path + ".out.chk"), "r") as file:
        lines = file.readlines()
        count_cubic = 0
        for i, line in enumerate(lines):
            if "CUBIC" in line:
                # The value is on the next line, split and take the first number
                next_line = lines[i + 1].strip()
                value = float(next_line.split()[0])  # Extract the first number
                value_units = float((value * u.angstrom).to("nanometer"))
                cubic_values.append(value_units)
                count_cubic += 1
            if count_cubic == 2:
                break
    return cubic_values


def list_with_restarts(fpath):
    """List fpath and its restart versions in order as pathlib Path objects."""
    fpath = Path(fpath)
    if not fpath.exists():
        return []
    parent = fpath.parent
    fname = fpath.name
    fnamesplit = fname.split(".out.")
    run_name = fnamesplit[0]
    suffix = fnamesplit[1]
    restarts = [
        Path(parent, f)
        for f in sorted(list(parent.glob(run_name + ".rst.*.out." + suffix)))
    ]
    restarts.insert(0, fpath)  # prepend fpath to list of restarts
    return restarts


def get_last_checkpoint(run_name):
    """Get name of last restart based on run_name."""
    fname = run_name + ".out.chk"
    return list_with_restarts(fname)[-1].name.split(".out.")[0]


def plot_res_pymser(job, eq_col, results, name, box_name):
    fig, [ax1, ax2] = plt.subplots(
        1, 2, gridspec_kw={"width_ratios": [2, 1]}, sharey=True
    )

    ax1.set_ylabel(name, color="black", fontsize=14, fontweight="bold")
    ax1.set_xlabel("GEMC Sweeps", fontsize=14, fontweight="bold")

    ax1.plot(range(0, len(eq_col) * 10, 10), eq_col, label="Raw data", color="blue")

    ax1.plot(
        range(0, len(eq_col) * 10, 10)[results["t0"] :],
        results["equilibrated"],
        label="Equilibrated data",
        color="red",
    )

    ax1.plot(
        [0, len(eq_col) * 10],
        [results["average"], results["average"]],
        color="green",
        zorder=4,
        label="Equilibrated average",
    )

    ax1.fill_between(
        range(0, len(eq_col) * 10, 10),
        results["average"] - results["uncertainty"],
        results["average"] + results["uncertainty"],
        color="lightgreen",
        alpha=0.3,
        zorder=4,
    )

    ax1.set_yticks(np.arange(0, eq_col.max() * 1.1, eq_col.max() / 10))
    ax1.set_xlim(-len(eq_col) * 10 * 0.02, len(eq_col) * 10 * 1.02)
    ax1.tick_params(axis="y", labelcolor="black")

    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.hist(
        eq_col,
        orientation="horizontal",
        bins=30,
        edgecolor="blue",
        lw=1.5,
        facecolor="white",
        zorder=3,
    )

    bin_red = 10
    ax2.hist(
        results["equilibrated"],
        orientation="horizontal",
        bins=bin_red,
        edgecolor="red",
        lw=1.5,
        facecolor="white",
        zorder=3,
    )

    ymax = int(ax2.get_xlim()[-1])

    ax2.plot(
        [0, ymax],
        [results["average"], results["average"]],
        color="green",
        zorder=4,
        label="Equilibrated average",
    )

    ax2.fill_between(
        range(ymax),
        results["average"] - results["uncertainty"],
        results["average"] + results["uncertainty"],
        color="lightgreen",
        alpha=0.3,
        zorder=4,
    )

    ax2.set_xlim(0, ymax)

    ax2.grid(alpha=0.5, zorder=1)

    fig.set_size_inches(9, 5)
    fig.set_dpi(100)
    fig.tight_layout()
    save_name = "MSER_eq_" + box_name + ".png"
    fig.savefig(job.fn(save_name), dpi=300, facecolor="white")
    plt.close(fig)

def get_pymser_results(eq_col):
    """
    This function determines the stationary region of a time series using pymser's equilibrate function.

    Parameters:
    -----------
    eq_col : array-like, array of time series data

    Returns:
    --------
    results : dict, results from pymser's equilibrate function
    adf_test_failed : bool, whether the ADF test failed
    """
    batch_size = max(1, int(len(eq_col) * 0.0005))

    # Try with ADF test enabled, fallback without it if it fails
    try:
        results = pymser.equilibrate(
            eq_col,
            LLM=False,
            batch_size=batch_size,
            ADF_test=True,
            uncertainty="uSD",
            print_results=False,
        )
        adf_test_failed = results["critical_values"]["1%"] <= results["adf"]
    except:
        results = pymser.equilibrate(
            eq_col,
            LLM=False,
            batch_size=batch_size,
            ADF_test=False,
            uncertainty="uSD",
            print_results=False,
        )
        results["adf"], results["critical_values"], adf_test_failed = (
            None,
            None,
            False,
        )
    return results, adf_test_failed

def check_equil_converge(job, eq_data_dict, prod_tol):
    equil_matrix = []
    res_matrix = []
    prop_cols = [5]
    prop_names = ["Number of Moles"]
    try:
        # Load data for both boxes
        for key in list(eq_data_dict.keys()):
            eq_col = eq_data_dict[key]["data"]
            results, adf_test_failed = get_pymser_results(eq_col)
            # Check if equilibrated and meets production tolerance
            equilibrium = len(eq_col) - results["t0"] >= prod_tol
            equil_matrix.append(equilibrium and not adf_test_failed)
            res_matrix.append(results)

        # Log results
        # print("ID", job.id, "AT", job.sp.atom_type, "T", job.sp.T)
        # print(equil_matrix)
        # log_text = '==============================================================================\n'

        for i, is_equilibrated in enumerate(equil_matrix):
            # box = df_box1 if i < len(prop_cols) else df_box2
            # box_name = "Liquid" if i < len(prop_cols) else "Vapor"
            # col_vals = box[:, prop_cols[i % len(prop_cols)] - 1]
            key_name = list(eq_data_dict.keys())[i]
            box_name = key_name.rsplit("_", 1)[0]
            col_vals = eq_data_dict[key_name]["data"]
            # plot all

            # if not all(equil_matrix):
            plot_res_pymser(
                job, col_vals, res_matrix[i], prop_names[i % len(prop_cols)], box_name
            )

            # Display outcome
            prod_cycles = len(col_vals) - res_matrix[i]["t0"]
            if is_equilibrated:
                # Plot successful equilibration
                statement = f"       > Success! Found {prod_cycles} production cycles."
            else:
                # Plot failed equilibration
                statement = f"       > {box_name} Box Failure! "
                if res_matrix[i]["adf"] is None:
                    # Note: ADF test failed to complete
                    statement += f"ADF test failed to complete! "
                elif res_matrix[i]["adf"] > res_matrix[i]["critical_values"]["1%"]:
                    adf, one_pct = (
                        res_matrix[i]["adf"],
                        res_matrix[i]["critical_values"]["1%"],
                    )
                    statement += f"ADF value: {adf}, 99% confidence value: {one_pct}! "
                if len(col_vals) - res_matrix[i]["t0"] < prod_tol:
                    statement += f"Only {prod_cycles} production cycles found."

            with open("Equil_Output.txt", "a") as f:
                print(statement, file=f)

    except Exception as e:
        # This will cause an error in the GEMC operation which lets us know that the job failed
        raise Exception(f"Error processing job {job.id}: {e}")

    return all(equil_matrix)


if __name__ == "__main__":
    ProjectGEMC().main()
