import flow
import numpy as np
import pandas as pd
import os
import sys
import subprocess
import shutil
import matplotlib.pyplot as plt
from flow import FlowProject, directives
import templates.ndcrc
from pymser import pymser
import warnings
import unyt as u

warnings.filterwarnings("ignore", category=DeprecationWarning)


class Project(FlowProject):
    pass


@Project.post.isfile("ff.xml")
@Project.operation
def create_forcefield(job):
    """Create the forcefield .xml file for the job"""
    molec_xml_function = _get_xml_from_molecule(job.sp.mol_name)
    content = molec_xml_function(job)

    with open(job.fn("ff.xml"), "w") as ff:
        ff.write(content)


@Project.pre.after(create_forcefield)
@Project.post.isfile("system.gro")
@Project.post.isfile("unedited.top")
@Project.operation
def create_system(job):
    """Construct the system in mbuild and apply the forcefield"""

    import mbuild
    import foyer
    import shutil
    import unyt as u

    compound = mbuild.load(job.sp.smiles, smiles=True)
    system = mbuild.fill_box(compound, n_compounds=750, density=job.sp.rho_avg)

    ff = foyer.Forcefield(job.fn("ff.xml"))

    system_ff = ff.apply(system)
    system_ff.combining_rule = "lorentz"

    system_ff.save("system.gro")
    system_ff.save("unedited.top")


@Project.pre.after(create_system)
@Project.post.isfile("system.top")
@Project.operation
def fix_topology(job):
    """Fix the LJ14 section of the topology file

    Parmed is writing the lj14 scaling factor as 1.0.
    GAFF uses 0.5. This function edits the topology
    file accordingly.
    """

    top_contents = []
    with open(job.fn("unedited.top")) as fin:
        for line_number, line in enumerate(fin):
            top_contents.append(line)
            if line.strip() == "[ defaults ]":
                defaults_line = line_number

    top_contents[defaults_line + 2] = (
        "1               2               yes              0.5       0.8333333\n"  # changed no to yes
    )

    with open(job.fn("system.top"), "w") as fout:
        for line in top_contents:
            fout.write(line)


@Project.post.isfile("em.mdp")
@Project.post.isfile("nvt1_eq.mdp")
@Project.post.isfile("npt_eq.mdp")
@Project.post.isfile("nvt2_eq.mdp")
@Project.post.isfile("inter_eq.mdp")
@Project.post.isfile("inter_prod.mdp")
@Project.operation
def generate_inputs(job):
    """Generate mdp files for energy minimization, equilibration, production"""

    content = _generate_em_mdp(job)

    with open(job.fn("em.mdp"), "w") as inp:
        inp.write(content)

    content = _generate_nvt1_eq_mdp(job)

    with open(job.fn("nvt1_eq.mdp"), "w") as inp:
        inp.write(content)

    content = _generate_npt_eq_mdp(job)

    with open(job.fn("npt_eq.mdp"), "w") as inp:
        inp.write(content)

    content = _generate_nvt2_eq_mdp(job)

    with open(job.fn("nvt2_eq.mdp"), "w") as inp:
        inp.write(content)

    content = _generate_inter_eq_mdp(job)

    with open(job.fn("inter_eq.mdp"), "w") as inp:
        inp.write(content)

    content = _generate_inter_prod_mdp(job)

    with open(job.fn("inter_prod.mdp"), "w") as inp:
        inp.write(content)


# @Project.pre.after(create_system)
# @Project.pre.after(fix_topology)
# @Project.pre.after(generate_inputs)
# @Project.post(em_complete)
# @Project.post(eq_complete)
# @Project.post(prod_complete)
# @Project.operation(with_job = True, cmd=True)
# def simulate(job):
#     """Run the minimization, equilibration, and production simulations"""

#     command = (
#         "gmx_d grompp -f em.mdp -c system.gro -p system.top -o em && "
#         "gmx_d mdrun -v -deffnm em -ntmpi 1 -ntomp 1 && "
#         "gmx_d grompp -f eq.mdp -c em.gro -p system.top -o eq && "
#         "gmx_d mdrun -v -deffnm eq -ntmpi 1 -ntomp 1 && "
#         "gmx_d grompp -f prod.mdp -c eq.gro -p system.top -o prod && "
#         "gmx_d mdrun -v -deffnm prod -ntmpi 1 -ntomp 1"
#     )

#     return command


# Energy Minimization
@Project.label
def em_complete(job):
    if "emin_fin" in job.doc:
        return True
    else:
        return False


@Project.pre.after(create_system)
@Project.pre.after(fix_topology)
@Project.pre.after(generate_inputs)
@Project.post(em_complete)
@Project.operation(with_job=True, cmd=False)
def em_sim(job):
    """Run the minimization simulations"""
    sim_name = "em"
    last_sim_name = "system"
    run_md_wo_eqcheck(job, sim_name, last_sim_name)
    job.doc.emin_fin = True


# Short NVT Equilibration
@Project.label
def nvt1_eq_comp(job):
    if "nvt1_eq_fin" in job.doc:
        return True
    else:
        return False


@Project.pre.after(em_sim)
@Project.post(nvt1_eq_comp)
@Project.operation(with_job=True, cmd=False)
def nvt_eq1_sim(job):
    """Run the 1st short NVT simulation"""
    sim_name = "nvt_eq1"
    last_sim_name = "em"
    run_md_wo_eqcheck(job, sim_name, last_sim_name)
    job.doc.nvt1_eq_fin = True


# Long Equilibration NPT
@Project.label
def npt_eq_comp(job):
    if "npt_eq_fin" in job.doc:
        return True
    else:
        return False


@Project.pre.after(nvt_eq1_sim)
@Project.post(npt_eq_comp)
@Project.operation(with_job=True, cmd=False)
def npt_eq_sim(job):
    import panedr

    """Run the equilibration simulations"""
    # Generate the first run
    sim_name = "npt_eq"
    last_sim_name = "nvt_eq1"
    property = "Density"
    run_md_w_eqcheck(job, sim_name, last_sim_name, property)
    job.doc.npt_eq_fin = True


# Run short NVT pre-equilibration
@Project.label
def nvt2_eq_comp(job):
    if "xy_box_len" in job.doc and "aspect_ratio" in job.doc:
        return True
    else:
        return False


@Project.pre.after(npt_eq_sim)
@Project.post(nvt2_eq_comp)
@Project.operation(with_job=True, cmd=False)
def nvt_eq2_sim(job):
    """Run the minimization simulations"""
    sim_name = "nvt_eq2"
    last_sim_name = "npt_eq"
    run_md_wo_eqcheck(job, sim_name, last_sim_name)

    # Get final box lengths
    # Extract the last line of the .gro file
    with open(sim_name + ".gro", "rb") as f:
        # Move the pointer to the end of the file, but leave space to find the last line
        f.seek(-2, os.SEEK_END)
        # Read backward until a newline is found
        while f.read(1) != b"\n":
            f.seek(-2, os.SEEK_CUR)
        # Read the last line after finding the newline
        last_line = f.readline().decode()
    # Extract the box length from the last line
    last_line.strip()
    ave_length = list(map(float, last_line.split()))[0]
    job.doc.xy_box_len = ave_length
    job.doc.aspect_ratio = 3.0


# Make Interface for simulation
@Project.pre.after(nvt_eq2_sim)
@Project.post.isfile("init_inter_eq.gro")
@Project.operation(with_job=True, cmd=False)
def init_inter_eq_sim(job):
    box_len = job.doc.xy_box_len
    xy_cen = job.doc.xy_box_len / 2
    z_cen = job.doc.xy_box_len * job.doc.aspect_ratio / 2
    new_z_len = z_cen * 2
    job.doc.z_box_len = new_z_len

    command = f"gmx_d editconf -f init_inter_eq.gro -center {xy_cen} {xy_cen} {z_cen} -bt triclinic -box {box_len} {box_len} {new_z_len} -angles 90 90 90 -o init_inter_eq.gro"
    subprocess.run(command, shell=True, check=True)


# Run Interface NVT equilibration
@Project.label
def inter_eq_comp(job):
    if "inter_eq_fin" in job.doc:
        return True
    else:
        return False


@Project.pre.after(nvt_eq2_sim)
@Project.post(inter_eq_comp)
@Project.operation(with_job=True, cmd=False)
def inter_eq_sim(job):
    """Run the interface equilibration simulations"""
    import panedr

    # Generate the first run
    last_sim_name = (
        "init_inter_eq"  # Use the same one since the -gro file is created beforehand
    )
    sim_name = "inter_eq"
    property = "Total-Energy"  # Total Energy Stable = Equilibrated
    run_md_w_eqcheck(job, sim_name, last_sim_name, property)
    job.doc.inter_eq_fin = True


# Run Interface NVT Production
@Project.label
def inter_prod_comp(job):
    if "inter_prod_fin" in job.doc:
        return True
    else:
        return False


@Project.pre.after(nvt_eq2_sim)
@Project.post(inter_prod_comp)
@Project.operation(with_job=True, cmd=False)
def inter_prod_sim(job):
    """Run the production simulations"""

    # Generate the first run
    last_sim_name = (
        "inter_eq"  # Use the same one since the -gro file is created beforehand
    )
    sim_name = "inter_prod"
    property = "#Surf*SurfTen"  # Surface tension
    run_md_wo_eqcheck(job, sim_name, last_sim_name)
    job.doc.inter_prod_fin = True


@Project.pre.after(inter_prod_sim)
@Project.post(lambda job: "surf_ten" in job.doc)
@Project.post(lambda job: "surf_ten_unc" in job.doc)
@Project.operation
def calculate_properties(job):
    """Calculate the density"""

    import panedr
    import numpy as np
    from block_average import block_average

    # Load the thermo data
    df = panedr.edr_to_df(job.fn("inter_prod.edr"))

    get_props = ["Density", "#Surf*SurfTen"]
    names = ["density", "surf_ten"]
    for prop, name in zip(get_props, names):
        property = df[prop].values
        ave = np.mean(property)

        # save average density
        job.doc[name] = ave

        (means_est, vars_est, vars_err) = block_average(property)

        with open(job.fn(name + "_blk_avg.txt"), "w") as ferr:
            ferr.write("# nblk_ops, mean, vars, vars_err\n")
            for nblk_ops, (mean_est, var_est, var_err) in enumerate(
                zip(means_est, vars_est, vars_err)
            ):
                ferr.write(
                    "{}\t{}\t{}\t{}\n".format(nblk_ops, mean_est, var_est, var_err)
                )

        job.doc[name + "_unc"] = np.max(np.sqrt(vars_est))


#####################################################################
################# HELPER FUNCTIONS BEYOND THIS POINT ################
#####################################################################
# Calculation Functions
def check_equil_converge(job, eq_data_dict, prod_tol):
    equil_matrix = []
    res_matrix = []
    prop_names = list(eq_data_dict.keys())
    num_cols = len(prop_names)

    try:
        # Load data for both boxes
        for key in list(eq_data_dict.keys()):
            eq_col = eq_data_dict[key]["data"]
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

            equilibrium = len(eq_col) - results["t0"] >= prod_tol
            equil_matrix.append(equilibrium and not adf_test_failed)
            res_matrix.append(results)

        for i, is_equilibrated in enumerate(equil_matrix):
            key_name = list(eq_data_dict.keys())[i]
            col_vals = eq_data_dict[key_name]["data"]
            t_vals = eq_data_dict[key_name]["time_data"]
            # plot all

            # if not all(equil_matrix):
            plot_res_pymser(
                job, t_vals, col_vals, res_matrix[i], prop_names[i % num_cols]
            )

            # Display outcome
            prod_cycles = len(col_vals) - res_matrix[i]["t0"]
            if is_equilibrated:
                # Plot successful equilibration
                statement = f"       > Success! Found {prod_cycles} production cycles."
            else:
                # Plot failed equilibration
                statement = f"       > Equil Failure! "
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


def plot_res_pymser(job, t_col, eq_col, results, name):
    fig, [ax1, ax2] = plt.subplots(
        1, 2, gridspec_kw={"width_ratios": [2, 1]}, sharey=True
    )

    ax1.set_ylabel(name, color="black", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Time (ns)", fontsize=14, fontweight="bold")

    ax1.plot(t_col, eq_col, label="Raw data", color="blue")

    ax1.plot(
        t_col[results["t0"] :],
        results["equilibrated"],
        label="Equilibrated data",
        color="red",
    )

    ax1.plot(
        [0, t_col[-1]],
        [results["average"], results["average"]],
        color="green",
        zorder=4,
        label="Equilibrated average",
    )

    ax1.fill_between(
        t_col,
        results["average"] - results["uncertainty"],
        results["average"] + results["uncertainty"],
        color="lightgreen",
        alpha=0.3,
        zorder=4,
    )

    # ax1.set_yticks(np.arange(eq_col.min(), eq_col.max(), eq_col.max() / 15))
    ax1.set_xlim(t_col.min(), t_col.max())
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
    save_name = "MSER_eq_" + name + ".png"
    fig.savefig(job.fn(save_name), dpi=300, facecolor="white")
    plt.close(fig)


# HELPER FUNCTIONS
def run_md_wo_eqcheck(job, sim_name, last_sim_name):
    with job:
        if os.path.exists(sim_name + ".cpt"):
            command = f"gmx_d mdrun -cpi {sim_name}.cpt -v -deffnm {sim_name} -ntmpi 1 -ntomp 8 -nb gpu -pme gpu -bonded gpu"
        else:
            command = (
                f"gmx_d grompp -maxwarn 5 -f {sim_name}.mdp -c {last_sim_name}.gro -p system.top -o {sim_name} && "
                f"gmx_d mdrun -v -deffnm {sim_name} -ntmpi 1 -ntomp 8 -nb gpu -pme gpu -bonded gpu"
            )
        subprocess.run(command, shell=True, check=True)


def run_md_w_eqcheck(job, sim_name, last_sim_name, property):
    with job:
        try:
            if sim_name == "npt_eq":
                nsteps_eq = job.sp.nsteps_npt
            elif sim_name == "inter_eq":
                nsteps_eq = job.sp.nsteps_intereq

            # Set number of iterations per extension and intitialize counter and total number of steps
            eq_extend = int(nsteps_eq / 4)  # In femtoseconds

            # Get the total number of equilibration restarts and steps so far
            existing_eq_steps = (
                count_steps(sim_name) * 1000
            )  # Convert to femtoseconds (step timescale is 1 fs)
            total_eq_steps = existing_eq_steps  # In femtoseconds

            # Set the maximum number of steps
            if max_eq_steps not in job.doc:
                job.doc.max_eq_steps = total_eq_steps * 2

            # The max number of steps is the larger of the number of steps + the org number of steps or the current max
            max_eq_steps = np.maximum(
                job.doc.max_eq_steps, existing_eq_steps + 2 * nsteps_eq
            )
            # Originally set the document eq_steps to the max number, it will be overwritten later
            job.doc.nsteps_gemc_eq = int(max_eq_steps)

            # Continue running while you have not exceeded the max number of steps
            while total_eq_steps < job.doc.max_eq_steps:
                # If you have enough steps, run the simulation, continue the simulation with more points
                if total_eq_steps + eq_extend <= max_eq_steps:
                    # If we have no steps, start the simulation
                    if existing_eq_steps == 0:
                        command = (
                            f"gmx_d grompp -maxwarn 5 -f {sim_name}.mdp -c {last_sim_name}.gro -p system.top -o {sim_name} && "
                            f"gmx_d mdrun -v -deffnm {sim_name} -ntmpi 1 -ntomp 8 -nb gpu -pme gpu -bonded gpu"
                        )
                    # Otherwise, check log file for whether previous simulation finished correctly
                    elif check_norm_term(job, sim_name):
                        # If it finished, extend the simulation
                        command = (
                            f"gmx_d convert-tpr -s {sim_name}.tpr -extend "
                            + str(eq_extend)
                            + f" -o {sim_name}.tpr &&"
                            f"gmx_d mdrun -s {sim_name}.tpr -cpi {sim_name}.cpt -v -deffnm {sim_name} -ntmpi 1 -ntomp 8 -nb gpu -pme gpu -bonded gpu "
                        )
                    # Otherwise restart the simulation from the checkpoint file
                    else:
                        command = f"gmx_d mdrun -cpi {sim_name}.cpt -v -deffnm {sim_name} -ntmpi 1 -ntomp 8 -nb gpu -pme gpu -bonded gpu"
                    subprocess.run(command, shell=True, check=True)

                    # Update equilibration data dictionary/files
                    eq_data_dict = get_eq_data_dict(
                        job, eq_data_dict, sim_name, property
                    )

                    # Track the number of added steps
                    total_eq_steps += eq_extend

                    # Set tolerance for determining equilibrium and check for convergence
                    prod_tol_eq = (
                        count_steps(sim_name) / 4
                    )  # In picoseconds (same units as the data)
                    is_equil = check_equil_converge(job, eq_data_dict, prod_tol_eq)

                    # If the simulation has converged, break
                    if is_equil:
                        break

                # Otherwise report an error
                else:
                    job.doc.equil_fail = True
                    raise Exception(
                        f"{sim_name} equilibration failed to converge after {max_eq_steps} steps"
                    )

            # Set the step counter to whatever the final number of equilibration steps was
            job.doc.nsteps_gemc_eq = total_eq_steps
            job.doc.equil_fail = False
        except:
            # If the simulation fails, extend the simulation
            if "equil_fail" in job.doc.keys() and job.doc.equil_fail:
                job.doc.max_eq_steps = total_eq_steps * 2
                del job.doc.nsteps_gemc_eq
                del job.doc.equil_fail
            # If another error occurs, set the equilibration failure flag
            else:
                job.doc["eq_fail"] = True


def check_norm_term(job, sim_name):
    selected_file = job.fn(sim_name + ".log")
    with open(selected_file, "rb") as f:
        # Move the pointer to the end of the file, but leave space to find the last line
        f.seek(-2, os.SEEK_END)
        # Read backward until a newline is found
        while f.read(1) != b"\n":
            f.seek(-2, os.SEEK_CUR)
        # Read the last line after finding the newline
        last_line = f.readline().decode()
    if "Finished mdrun on rank" in last_line:
        return True
    else:
        return False


def get_eq_data_dict(job, eq_data_dict, sim_name, property):
    import panedr

    with job:
        # Get the density and volume data
        df_all = panedr.edr_to_df(job.fn(sim_name + ".edr"))
        if property in df_all.columns:
            df = df[["Time", property]].copy()

        elif property in ["Volume", "Density"]:
            command = f"gmx_d energy -f {sim_name}.edr -s {sim_name}.tpr -o {sim_name}_{property}.xvg"
            subprocess.run(
                command, input=f"{property}", text=True, check=True, shell=True
            )
            prop_data = np.loadtxt(
                sim_name + "_" + property + ".xvg", comments=["#", "@"]
            )
            df = pd.DataFrame(prop_data)

        property_data = df.iloc[:, 1].values
        time_data = df.iloc[:, 0].values
        eq_col_file = job.fn(sim_name + "_" + property + ".csv")
        eq_data_dict[property] = {
            "data": property_data,
            "time_data": time_data,
            "file": eq_col_file,
        }
        np.savetxt(eq_col_file, property_data, delimiter=",")
        return eq_data_dict


def count_steps(job, sim_name):
    import panedr

    if os.path.exists(job.fn(sim_name + ".edr")):
        # Extract the maximum time recorded
        df = panedr.edr_to_df(job.fn(sim_name + ".edr"))
        time_total = df["Time"].max()  # in picoseconds
    else:
        time_total = 0

    return time_total


# Build FFs
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
  <Angle class1="c3" class2="c3" class3="h1" angle="1.92108" k="387.936"/>
  <Angle class1="c3" class2="c3" class3="oh" angle="1.90991" k="566.680"/>
  <Angle class1="c3" class2="oh" class3="ho" angle="1.88775" k="394.056"/>
  <Angle class1="h1" class2="c3" class3="h1" angle="1.91201" k="327.856"/>
  <Angle class1="h1" class2="c3" class3="oh" angle="1.91777" k="426.515"/>
 </HarmonicAngleForce>
 <PeriodicTorsionForce>
  <Proper class1="c3" class2="c3" class3="oh" class4="ho" periodicity1="3" phase1="0.0" k1="0.66948049" periodicity2="1" phase2="0.0" k2="1.04595934"/>
  <Proper class1="h1" class2="c3" class3="c3" class4="h1" periodicity1="3" phase1="0.0" k1="0.65268528"/>
  <Proper class1="h1" class2="c3" class3="c3" class4="oh" periodicity1="3" phase1="0.0" k1="0.0" periodicity2="1" phase2="0.0" k2="1.04595934"/>
  <Proper class1="h1" class2="c3" class3="oh" class4="ho" periodicity1="3" phase1="0.0" k1="0.69874740"/>
  <Proper class1="oh" class2="c3" class3="c3" class4="h1" periodicity1="3" phase1="0.0" k1="0.0" periodicity2="1" phase2="0.0" k2="1.04595934"/>
  <Proper class1="oh" class2="c3" class3="c3" class4="oh" periodicity1="3" phase1="0.0" k1="0.60246593" periodicity2="2" phase2="0.0" k2="4.91617518"/>
 </PeriodicTorsionForce>
 <NonbondedForce coulomb14scale="0.833333" lj14scale="0.5">
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


def __generate_Gly_xml(job):
    content = """<ForceField>
 <AtomTypes>
  <Type name="C1" class="c3" element="C" mass="12.01" def="[C;X4]" desc="Sp3 C"/>
  <Type name="H1" class="h1" element="H" mass="1.008" def="H[C;X4]([N,O,F,Cl,Br,I,S])" desc="H bonded to aliphatic carbon with 1 d. group"/>
  <Type name="C2" class="c3" element="C" mass="12.01" def="[C;X4]" desc="Sp3 C"/>
  <Type name="H2" class="h1" element="H" mass="1.008" def="H[C;X4]([N,O,F,Cl,Br,I,S])" desc="H bonded to aliphatic carbon with 1 d. group"/>
  <Type name="O1" class="oh" element="O" mass="16.0" def="[O;X2]H" desc="Oxygen in hydroxyl group"/>
  <Type name="H3" class="ho" element="H" mass="1.008" def="HO" desc="Hydroxyl group"/>
  <Type name="O2" class="oh" element="O" mass="16.0" def="[O;X2]H" desc="Oxygen in hydroxyl group"/>
  <Type name="H4" class="ho" element="H" mass="1.008" def="HO" desc="Hydroxyl group"/>
 </AtomTypes>
 <HarmonicBondForce>
  <Bond class1="c3" class2="h1" length="0.1093" k="281080.370"/>
  <Bond class1="c3" class2="c3" length="0.1535" k="253634.331"/>
  <Bond class1="c3" class2="oh" length="0.1426" k="262838.440"/>
  <Bond class1="ho" class2="oh" length="0.0974" k="309281.363"/>
 </HarmonicBondForce>
 <HarmonicAngleForce>
  <Angle class1="c3" class2="c3" class3="h1" angle="1.92108" k="387.936"/>
  <Angle class1="c3" class2="c3" class3="c3" angle="1.93086" k="528.933"/>
  <Angle class1="c3" class2="c3" class3="oh" angle="1.90991" k="566.680"/>
  <Angle class1="c3" class2="oh" class3="ho" angle="1.88775" k="394.056"/>
  <Angle class1="h1" class2="c3" class3="h1" angle="1.91201" k="327.856"/>
  <Angle class1="h1" class2="c3" class3="oh" angle="1.91777" k="426.515"/>
 </HarmonicAngleForce>
 <PeriodicTorsionForce>
  <Proper class1="c3" class2="c3" class3="c3" class4="h1" periodicity1="3" phase1="0.0" k1="0.65268528"/>
  <Proper class1="c3" class2="c3" class3="c3" class4="oh" periodicity1="3" phase1="0.0" k1="0.65268528"/>
  <Proper class1="c3" class2="c3" class3="oh" class4="ho" periodicity1="3" phase1="0.0" k1="0.66948049" periodicity2="1" phase2="0.0" k2="1.04595934"/>
  <Proper class1="h1" class2="c3" class3="c3" class4="c3" periodicity1="3" phase1="0.0" k1="0.65268528"/>
  <Proper class1="h1" class2="c3" class3="c3" class4="h1" periodicity1="3" phase1="0.0" k1="0.65268528"/>
  <Proper class1="h1" class2="c3" class3="c3" class4="oh" periodicity1="3" phase1="0.0" k1="0.0" periodicity2="1" phase2="0.0" k2="1.04595934"/>
  <Proper class1="h1" class2="c3" class3="oh" class4="ho" periodicity1="3" phase1="0.0" k1="0.69874740"/>
  <Proper class1="oh" class2="c3" class3="c3" class4="c3" periodicity1="3" phase1="0.0" k1="0.65268528"/>
  <Proper class1="oh" class2="c3" class3="c3" class4="h1" periodicity1="3" phase1="0.0" k1="0.0" periodicity2="1" phase2="0.0" k2="1.04595934"/>
  <Proper class1="oh" class2="c3" class3="c3" class4="oh" periodicity1="3" phase1="0.0" k1="0.60246593" periodicity2="2" phase2="0.0" k2="4.91617518"/>
 </PeriodicTorsionForce>
 <NonbondedForce coulomb14scale="0.833333" lj14scale="0.5">
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
  <Angle class1="c3" class2="oh" class3="ho" angle="1.88775" k="394.056"/>
  <Angle class1="h1" class2="c3" class3="h1" angle="1.91201" k="327.856"/>
  <Angle class1="h1" class2="c3" class3="oh" angle="1.91777" k="426.515"/>
 </HarmonicAngleForce>
 <PeriodicTorsionForce>
  <Proper class1="h1" class2="c3" class3="oh" class4="ho" periodicity1="3" phase1="0.0" k1="0.69874740"/>
 </PeriodicTorsionForce>
 <NonbondedForce coulomb14scale="0.833333" lj14scale="0.5">
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
  <Angle class1="h2" class2="c3" class3="h2" angle="1.90573" k="326.359"/>
  <Angle class1="cl" class2="c3" class3="h2" angle="1.86995" k="363.924"/>
  <Angle class1="cl" class2="c3" class3="cl" angle="1.93784" k="524.676"/>
 </HarmonicAngleForce>
 <NonbondedForce coulomb14scale="0.833333" lj14scale="0.5">
  <Atom type="C1" charge="-0.375336" sigma="{sigma_C1:0.6f}" epsilon="{epsilon_C1:0.6f}"/>
  <Atom type="H1" charge="0.243662" sigma="{sigma_H1:0.6f}" epsilon="{epsilon_H1:0.6f}"/>
  <Atom type="Cl1" charge="-0.055994" sigma="{sigma_Cl1:0.6f}" epsilon="{epsilon_Cl1:0.6f}"/>
 </NonbondedForce>
</ForceField>""".format(
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
  <Angle class1="h1" class2="c3" class3="s4" angle="1.89647" k="454.219"/>
  <Angle class1="c3" class2="s4" class3="o" angle="1.85371" k="343.587"/>
  <Angle class1="c3" class2="s4" class3="c3" angle="1.68983" k="325.012"/>
  <Angle class1="h1" class2="c3" class3="h1" angle="1.91201" k="327.856"/>
 </HarmonicAngleForce>
 <PeriodicTorsionForce>
  <Proper class1="h1" class2="c3" class3="s4" class4="c3" periodicity1="3" phase1="0.0" k1="0.83676747"/>
  <Proper class1="h1" class2="c3" class3="s4" class4="o" periodicity1="3" phase1="0.0" k1="0.83676747"/>
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
 <NonbondedForce coulomb14scale="0.833333" lj14scale="0.5">
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


def __generate_DEC_xml(job):
    content = """<ForceField>
 <AtomTypes>
  <Type name="C1" class="c3" element="C" mass="12.01" def="[C;X4]" desc="Sp3 C"/>
  <Type name="H1" class="hc" element="H" mass="1.008" def="H[C;X4]" desc="H bonded to aliphatic carbon without d. group"/>
  <Type name="C2" class="c3" element="C" mass="12.01" def="[C;X4]" desc="Sp3 C"/>
  <Type name="H2" class="h1" element="H" mass="1.008" def="H[C;X4]([N,O,F,Cl,Br,I,S])" desc="H bonded to aliphatic carbon with 1 d. group"/>
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
  <Angle class1="c3" class2="c3" class3="h1" angle="1.92108" k="387.936"/>
  <Angle class1="c3" class2="c3" class3="os" angle="1.89229" k="567.179"/>
  <Angle class1="hc" class2="c3" class3="hc" angle="1.89106" k="329.951"/>
  <Angle class1="c3" class2="c3" class3="hc" angle="1.92073" k="388.019"/>
  <Angle class1="c3" class2="os" class3="c" angle="2.00957" k="532.458"/>
  <Angle class1="h1" class2="c3" class3="h1" angle="1.91201" k="327.856"/>
  <Angle class1="h1" class2="c3" class3="os" angle="1.89927" k="425.434"/>
  <Angle class1="os" class2="c" class3="o" angle="2.15251" k="635.375"/>
  <Angle class1="os" class2="c" class3="os" angle="1.94395" k="639.731"/>
  <Angle class1="c" class2="os" class3="c3" angle="2.00957" k="532.458"/>
  <Angle class1="o" class2="c" class3="os" angle="2.15251" k="635.375"/>
 </HarmonicAngleForce>
 <PeriodicTorsionForce>
  <Proper class1="c3" class2="c3" class3="os" class4="c" periodicity1="3" phase1="0.0" k1="1.60244629" periodicity2="1" phase2="3.141592654" k2="3.34723617"/>
  <Proper class1="h1" class2="c3" class3="c3" class4="hc" periodicity1="3" phase1="0.0" k1="0.65268528"/>
  <Proper class1="h1" class2="c3" class3="os" class4="c" periodicity1="3" phase1="0.0" k1="1.60244629"/>
  <Proper class1="hc" class2="c3" class3="c3" class4="h1" periodicity1="3" phase1="0.0" k1="0.65268528"/>
  <Proper class1="hc" class2="c3" class3="c3" class4="os" periodicity1="3" phase1="0.0" k1="0.0" periodicity2="1" phase2="0.0" k2="1.04595934"/>
  <Proper class1="o" class2="c" class3="os" class4="c3" periodicity1="2" phase1="3.141592654" k1="11.29677657" periodicity2="1" phase2="3.141592654" k2="5.85762173"/>
  <Proper class1="os" class2="c" class3="os" class4="c3" periodicity1="2" phase1="3.141592654" k1="11.29677657"/>
  <Proper class1="os" class2="c3" class3="c3" class4="hc" periodicity1="3" phase1="0.0" k1="0.0" periodicity2="1" phase2="0.0" k2="1.04595934"/>
  <Improper class1="o" class2="os" class3="c" class4="os" periodicity1="2" phase1="3.141592654" k1="43.93195509"/>
 </PeriodicTorsionForce>
 <NonbondedForce coulomb14scale="0.833333" lj14scale="0.5">
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
  <Type name="C1" class="c3" element="C" mass="12.01" def="[C;X4]" desc="Sp3 C"/>
  <Type name="O1" class="os" element="O" mass="16.0" def="[O;X2]([!H])[!H]" desc="Ether and ester oxygen"/>
  <Type name="C2" class="c3" element="C" mass="12.01" def="[C;X4]" desc="Sp3 C"/>
  <Type name="H1" class="h1" element="H" mass="1.008" def="H[C;X4]([N,O,F,Cl,Br,I,S])" desc="H bonded to aliphatic carbon with 1 d. group"/>
  <Type name="H2" class="hc" element="H" mass="1.008" def="H[C;X4]" desc="H bonded to aliphatic carbon without d. group"/>
 </AtomTypes>
 <HarmonicBondForce>
  <Bond class1="c3" class2="os" length="0.1439" k="252295.702"/>
  <Bond class1="c3" class2="c3" length="0.1535" k="253634.331"/>
  <Bond class1="c3" class2="h1" length="0.1093" k="281080.370"/>
  <Bond class1="c3" class2="hc" length="0.1092" k="282252.709"/>
 </HarmonicBondForce>
 <HarmonicAngleForce>
  <Angle class1="c3" class2="os" class3="c3" angle="1.96262" k="522.082"/>
  <Angle class1="c3" class2="c3" class3="c3" angle="1.93086" k="528.933"/>
  <Angle class1="c3" class2="c3" class3="hc" angle="1.92073" k="388.019"/>
  <Angle class1="c3" class2="c3" class3="os" angle="1.89229" k="567.179"/>
  <Angle class1="h1" class2="c3" class3="os" angle="1.89927" k="425.434"/>
  <Angle class1="c3" class2="c3" class3="h1" angle="1.92108" k="387.936"/>
  <Angle class1="h1" class2="c3" class3="h1" angle="1.91201" k="327.856"/>
  <Angle class1="hc" class2="c3" class3="hc" angle="1.89106" k="329.951"/>
 </HarmonicAngleForce>
 <PeriodicTorsionForce>
  <Proper class1="c3" class2="c3" class3="c3" class4="c3" periodicity1="3" phase1="0.0" k1="0.75312398" periodicity2="2" phase2="3.141592654" k2="1.04595934"/>
  <Proper class1="c3" class2="c3" class3="c3" class4="hc" periodicity1="3" phase1="0.0" k1="0.66948049"/>
  <Proper class1="c3" class2="c3" class3="os" class4="c3" periodicity1="3" phase1="0.0" k1="1.60244629" periodicity2="2" phase2="3.141592654" k2="0.41838374"/>
  <Proper class1="h1" class2="c3" class3="c3" class4="c3" periodicity1="3" phase1="0.0" k1="0.65268528"/>
  <Proper class1="h1" class2="c3" class3="c3" class4="hc" periodicity1="3" phase1="0.0" k1="0.65268528"/>
  <Proper class1="h1" class2="c3" class3="os" class4="c3" periodicity1="3" phase1="0.0" k1="1.60244629"/>
  <Proper class1="hc" class2="c3" class3="c3" class4="c3" periodicity1="3" phase1="0.0" k1="0.66948049"/>
  <Proper class1="hc" class2="c3" class3="c3" class4="hc" periodicity1="3" phase1="0.0" k1="0.62757560"/>
  <Proper class1="os" class2="c3" class3="c3" class4="c3" periodicity1="3" phase1="0.0" k1="0.65268528"/>
  <Proper class1="os" class2="c3" class3="c3" class4="hc" periodicity1="3" phase1="0.0" k1="0.0" periodicity2="1" phase2="0.0" k2="1.04595934"/>
 </PeriodicTorsionForce>
 <NonbondedForce coulomb14scale="0.833333" lj14scale="0.5">
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


# Build mdp files
def _generate_em_mdp(job):

    contents = """
; MDP file for energy minimization

integrator	    = steep		    ; Algorithm (steep = steepest descent minimization)
emtol		    = 100.0  	    ; Stop minimization when the maximum force < 100.0 kJ/mol/nm
emstep          = 0.01          ; Energy step size
nsteps		    = 50000	  	    ; Maximum number of (minimization) steps to perform

nstenergy                = 1000
nstlog                   = 1000
nstxout-compressed       = 1000

nstlist		    = 1		    ; Frequency to update the neighbor list and long range forces
cutoff-scheme   = Verlet
ns-type		    = grid		; Method to determine neighbor list (simple, grid)
verlet-buffer-tolerance = 1e-5          ; kJ/mol/ps
coulombtype	    = PME		; Treatment of long range electrostatic interactions
rcoulomb	    = 1.2		; Short-range electrostatic cut-off
rvdw		    = 1.2		; Short-range Van der Waals cut-off
pbc		        = xyz 		; Periodic Boundary Conditions (yes/no)
constraints     = all-bonds
lincs-order     = 8
lincs-iter      = 4
"""

    return contents


def _generate_nvt1_eq_mdp(job):
    # Use 100000 (100 ps) for the first equilibration
    contents = """
; MDP file for NVT simulation

; Run parameters
integrator	            = md		    ; leap-frog integrator
nsteps		            = {nsteps}	    ;
dt		                = 0.001		    ; 1 fs

; Output control
nstxout		            = 10000		    ; save coordinates every 10.0 ps
nstvout		            = 0		        ; don't save velocities
nstenergy	            = 10000		    ; save energies every 10.0 ps
nstlog		            = 10000		    ; update log file every 10.0 ps

; Neighborsearching
cutoff-scheme           = Verlet
ns-type		            = grid		    ; search neighboring grid cells
nstlist		            = 10		    ; 10 fs, largely irrelevant with Verlet
verlet-buffer-tolerance = 1e-5          ; kJ/mol/ps

; VDW
vdwtype                 = Cut-off
rvdw		            = 2.5		    ; short-range van der Waals cutoff (in nm)
vdw-modifier            = None

; Electrostatics
rcoulomb	            = 2.5		    ; short-range electrostatic cutoff (in nm)
coulombtype	            = PME	        ; Particle Mesh Ewald for long-range electrostatics
pme-order	            = 4		        ; cubic interpolation
fourier-spacing         = 0.12          ; effects accuracy of pme
ewald-rtol              = 1e-5

; Temperature coupling is on
tcoupl		            = v-rescale     ; modified Berendsen thermostat
tc-grps		            = System 	    ; Single coupling group
tau-t		            = 0.1	  		; time constant, in ps
ref-t		            = {temp}        ; reference temperature, one for each group, in K

; Pressure coupling is off
pcoupl		            = berendsen
pcoupltype              = isotropic
ref-p                   = {press}
tau-p                   = 0.5
compressibility         = 4.5e-5

; Periodic boundary conditions
pbc		                = xyz		    ; 3-D PBC

; Dispersion correction
DispCorr	            = EnerPres	    ; apply analytical tail corrections

; Velocity generation
gen-vel		            = yes		    ; assign velocities from Maxwell distribution
gen-temp	            = {temp}        ; temperature for Maxwell distribution
gen-seed	            = -1		    ; generate a random seed

constraints             = all-bonds
lincs-order             = 8
lincs-iter              = 4
""".format(
        temp=job.sp.T, press=job.sp.P, nsteps=job.sp.nsteps_nvt1
    )

    return contents


def _generate_npt_eq_mdp(job):
    # Use 500000 (500 ps) for the first equilibration
    contents = """
; MDP file for NPT simulation

; Run parameters
integrator	            = md		    ; leap-frog integrator
nsteps		            = {nsteps}	    ;
dt		                = 0.001		    ; 1 fs

; Output control
nstxout		            = 10000		    ; save coordinates every 10.0 ps
nstvout		            = 0		        ; don't save velocities
nstenergy	            = 10000		    ; save energies every 10.0 ps
nstlog		            = 10000		    ; update log file every 10.0 ps

; Neighborsearching
cutoff-scheme           = Verlet
ns-type		            = grid		    ; search neighboring grid cells
nstlist		            = 10		    ; 10 fs, largely irrelevant with Verlet
verlet-buffer-tolerance = 1e-5          ; kJ/mol/ps

; VDW
vdwtype                 = Cut-off
rvdw		            = 2.5		    ; short-range van der Waals cutoff (in nm)
vdw-modifier            = None

; Electrostatics
rcoulomb	            = 2.5		    ; short-range electrostatic cutoff (in nm)
coulombtype	            = PME	        ; Particle Mesh Ewald for long-range electrostatics
pme-order	            = 4		        ; cubic interpolation
fourier-spacing         = 0.12          ; effects accuracy of pme
ewald-rtol              = 1e-5

; Temperature coupling is on
tcoupl		            = v-rescale     ; modified Berendsen thermostat
tc-grps		            = System 	    ; Single coupling group
tau-t		            = 0.1	  		; time constant, in ps
ref-t		            = {temp}        ; reference temperature, one for each group, in K

; Pressure coupling is on
pcoupl                  = Parrinello-Rahman     ; Pressure coupling on in NPT
pcoupltype              = isotropic             ; uniform scaling of box vectors
tau_p                   = 2.0                   ; time constant, in ps
ref-p                   = {press}               ; reference pressure, in bar (from the system defined pressure)
compressibility         = 4.5e-5
nstpcouple              = 1
;refcoord_scaling       = com

; Periodic boundary conditions
pbc		                = xyz		    ; 3-D PBC

; Dispersion correction
DispCorr	            = EnerPres	    ; apply analytical tail corrections

; Velocity generation
gen_vel                 = no        ; Velocity generation is off 

constraints             = all-bonds
lincs-order             = 8
lincs-iter              = 4
""".format(
        temp=job.sp.T, press=job.sp.P, nsteps=job.sp.nsteps_npt
    )

    return contents


def _generate_nvt2_eq_mdp(job):
    # Use 100000 (100 ps) minimum for the second equilibration
    contents = """
; MDP file for NVT simulation

; Run parameters
integrator	            = md		    ; leap-frog integrator
nsteps		            = {nsteps}	    ;
dt		                = 0.001		    ; 1 fs

; Output control
nstxout		            = 1000		    ; save coordinates every 1.0 ps
nstvout		            = 0		        ; don't save velocities
nstenergy	            = 1000		    ; save energies every 1.0 ps
nstlog		            = 1000		    ; update log file every 1.0 ps

; Neighborsearching
cutoff-scheme           = Verlet
ns-type		            = grid		    ; search neighboring grid cells
nstlist		            = 10		    ; 10 fs, largely irrelevant with Verlet
verlet-buffer-tolerance = 1e-5          ; kJ/mol/ps

; VDW
vdwtype                 = Cut-off
rvdw		            = 2.5		    ; short-range van der Waals cutoff (in nm)
vdw-modifier            = None

; Electrostatics
rcoulomb	            = 2.5		    ; short-range electrostatic cutoff (in nm)
coulombtype	            = PME	        ; Particle Mesh Ewald for long-range electrostatics
pme-order	            = 4		        ; cubic interpolation
fourier-spacing         = 0.12          ; effects accuracy of pme
ewald-rtol              = 1e-5

; Temperature coupling is on
tcoupl		            = v-rescale     ; modified Berendsen thermostat
tc-grps		            = System 	    ; Single coupling group
tau-t		            = 0.1	  		; time constant, in ps
ref-t		            = {temp}        ; reference temperature, one for each group, in K

; Pressure coupling is off
pcoupl		            = berendsen
pcoupltype              = isotropic
ref-p                   = {press}
tau-p                   = 0.5
compressibility         = 4.5e-5

; Periodic boundary conditions
pbc		                = xyz		    ; 3-D PBC

; Dispersion correction
DispCorr	            = EnerPres	    ; apply analytical tail corrections

; Velocity generation
gen-vel		            = yes		    ; assign velocities from Maxwell distribution
gen-temp	            = {temp}        ; temperature for Maxwell distribution
gen-seed	            = -1		    ; generate a random seed

constraints             = all-bonds
lincs-order             = 8
lincs-iter              = 4
""".format(
        temp=job.sp.T, press=job.sp.P, nsteps=job.sp.nsteps_nvt2
    )

    return contents


def _generate_inter_eq_mdp(job):
    # Use 25000000 (25 ns) minimum for the interfacial equilibration
    contents = """
; MDP file for NVT simulation

; Run parameters
integrator	            = md		    ; leap-frog integrator
nsteps		            = {nsteps}	    ;
dt		                = 0.001		    ; 1 fs

; Output control
nstxout		            = 1000		    ; save coordinates every 1.0 ps
nstvout		            = 0		        ; don't save velocities
nstenergy	            = 1000		    ; save energies every 1.0 ps
nstlog		            = 1000		    ; update log file every 1.0 ps

; Neighborsearching
cutoff-scheme           = Verlet
ns-type		            = grid		    ; search neighboring grid cells
nstlist		            = 10		    ; 10 fs, largely irrelevant with Verlet
verlet-buffer-tolerance = 1e-5          ; kJ/mol/ps

; VDW
vdwtype                 = Cut-off
rvdw		            = 2.5		    ; short-range van der Waals cutoff (in nm)
vdw-modifier            = None

; Electrostatics
rcoulomb	            = 2.5		    ; short-range electrostatic cutoff (in nm)
coulombtype	            = PME	        ; Particle Mesh Ewald for long-range electrostatics
pme-order	            = 4		        ; cubic interpolation
fourier-spacing         = 0.12          ; effects accuracy of pme
ewald-rtol              = 1e-5

; Temperature coupling is on
tcoupl		            = v-rescale     ; modified Berendsen thermostat
tc-grps		            = System 	    ; Single coupling group
tau-t		            = 0.1	  		; time constant, in ps
ref-t		            = {temp}        ; reference temperature, one for each group, in K

; Pressure coupling is off
pcoupl		            = berendsen
pcoupltype              = isotropic
ref-p                   = {press}
tau-p                   = 0.5
compressibility         = 4.5e-5

; Periodic boundary conditions
pbc		                = xyz		    ; 3-D PBC

; Dispersion correction
DispCorr	            = EnerPres	    ; apply analytical tail corrections

; Velocity generation
gen-vel		            = yes		    ; assign velocities from Maxwell distribution
gen-temp	            = {temp}        ; temperature for Maxwell distribution
gen-seed	            = -1		    ; generate a random seed

constraints             = all-bonds
lincs-order             = 8
lincs-iter              = 4
""".format(
        temp=job.sp.T, press=job.sp.P, nsteps=job.sp.nsteps_intereq
    )

    return contents


def _generate_inter_prod_mdp(job):
    # Use 50000000 (50 ns) for the interfacial production
    contents = """
; MDP file for NVT simulation

; Run parameters
integrator	            = md		    ; leap-frog integrator
nsteps		            = {nsteps}	    ;
dt		                = 0.001		    ; 1 fs

; Output control
nstxout		            = 1000		    ; save coordinates every 1.0 ps
nstvout		            = 0		        ; don't save velocities
nstenergy	            = 1000		    ; save energies every 1.0 ps
nstlog		            = 1000		    ; update log file every 1.0 ps

; Neighborsearching
cutoff-scheme           = Verlet
ns-type		            = grid		    ; search neighboring grid cells
nstlist		            = 10		    ; 10 fs, largely irrelevant with Verlet
verlet-buffer-tolerance = 1e-5          ; kJ/mol/ps

; VDW
vdwtype                 = Cut-off
rvdw		            = 2.5		    ; short-range van der Waals cutoff (in nm)
vdw-modifier            = None

; Electrostatics
rcoulomb	            = 2.5		    ; short-range electrostatic cutoff (in nm)
coulombtype	            = PME	        ; Particle Mesh Ewald for long-range electrostatics
pme-order	            = 4		        ; cubic interpolation
fourier-spacing         = 0.12          ; effects accuracy of pme
ewald-rtol              = 1e-5

; Temperature coupling is on
tcoupl		            = v-rescale     ; modified Berendsen thermostat
tc-grps		            = System 	    ; Single coupling group
tau-t		            = 0.1	  		; time constant, in ps
ref-t		            = {temp}        ; reference temperature, one for each group, in K

; Pressure coupling is off
pcoupl		            = berendsen
pcoupltype              = isotropic
ref-p                   = {press}
tau-p                   = 0.5
compressibility         = 4.5e-5

; Periodic boundary conditions
pbc		                = xyz		    ; 3-D PBC

; Dispersion correction
DispCorr                 = no        ; account for cut-off vdW scheme

; Velocity generation
gen-vel		            = yes		    ; assign velocities from Maxwell distribution
gen-temp	            = {temp}        ; temperature for Maxwell distribution
gen-seed	            = -1		    ; generate a random seed

constraints             = all-bonds
lincs-order             = 8
lincs-iter              = 4
""".format(
        temp=job.sp.T, press=job.sp.P, nsteps=job.sp.nsteps_interprod
    )

    return contents


if __name__ == "__main__":
    Project().main()
