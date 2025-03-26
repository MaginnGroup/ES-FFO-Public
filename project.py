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

warnings.filterwarnings("ignore", category=DeprecationWarning)


class Project(FlowProject):
    pass


@Project.post.isfile("ff.xml")
@Project.operation
def create_forcefield(job):
    """Create the forcefield .xml file for the job"""

    content = _generate_r41_xml(job)

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

    r41 = mbuild.load("CF", smiles=True)
    system = mbuild.fill_box(r41, n_compounds=300, density=700)

    ff = foyer.Forcefield(job.fn("ff.xml"))

    system_ff = ff.apply(system)
    system_ff.combining_rule = "lorentz"

    system_ff.save(job.fn("unedited.top"))

    # Get pre-minimized gro file
    shutil.copy("data/initial_config/system_em.gro", job.fn("system.gro"))


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
        for (line_number, line) in enumerate(fin):
            top_contents.append(line)
            if line.strip() == "[ defaults ]":
                defaults_line = line_number

    top_contents[
        defaults_line + 2
    ] = "1               2               yes              0.5       0.8333333\n" #changed no to yes

    with open(job.fn("system.top"), "w") as fout:
        for line in top_contents:
            fout.write(line)


@Project.post.isfile("em.mdp")
@Project.post.isfile("eq.mdp")
@Project.post.isfile("prod.mdp")
@Project.operation
def generate_inputs(job):
    """Generate mdp files for energy minimization, equilibration, production"""

    content = _generate_em_mdp(job)

    with open(job.fn("em.mdp"), "w") as inp:
        inp.write(content)

    content = _generate_eq_mdp(job)

    with open(job.fn("eq.mdp"), "w") as inp:
        inp.write(content)

    content = _generate_prod_mdp(job)

    with open(job.fn("prod.mdp"), "w") as inp:
        inp.write(content)


@Project.label
def em_complete(job):
    if job.isfile("em.gro"):
        return True
    else:
        return False


@Project.label
def eq_complete(job):
    if job.isfile("eq.gro"):
        return True
    else:
        return False


@Project.label
def prod_complete(job):
    if job.isfile("prod.gro"):
        return True
    else:
        return False


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

#Energy Minimization
@Project.pre.after(create_system)
@Project.pre.after(fix_topology)
@Project.pre.after(generate_inputs)
@Project.post(em_complete)
@Project.operation(with_job = True, cmd=False)
def em_sim(job):
    """Run the minimization simulations"""
    sim_name = "em"
    last_sim_name = "system"
    return run_md_wo_eqcheck(job, sim_name, last_sim_name)

#Short NVT Equilibration
@Project.pre.after(create_system)
@Project.pre.after(fix_topology)
@Project.pre.after(generate_inputs)
@Project.post(em_complete)
@Project.operation(with_job = True, cmd=False)
def nvt_eq1_sim(job):
    """Run the 1st short NVT simulation"""
    sim_name = "nvt_eq1"
    last_sim_name = "em"
    return run_md_wo_eqcheck(job, sim_name, last_sim_name)


#Long Equilibration NPT
@Project.pre.after(nvt_eq1_sim)
@Project.post(npt_eq_complete)
@Project.operation(with_job = True, cmd=False)
def npt_eq_sim(job):
    import panedr
    """Run the equilibration simulations"""
    #Generate the first run
    sim_name = "npt_eq"   
    last_sim_name = "nvt_eq1"
    property = "Pressure"
    run_md_w_eqcheck(job, sim_name, last_sim_name, property)

#Run short NVT pre-equilibration   
@Project.pre.after(npt_eq_sim)
@Project.post(prod_complete)
@Project.operation(with_job = True, cmd=False)
def nvt_eq2_sim(job):
    """Run the minimization simulations"""
    sim_name = "nvt_eq2"
    last_sim_name = "npt_eq"
    run_md_wo_eqcheck(job, sim_name, last_sim_name)
    #Get final box volume
    
    property = "Volume"
    eq_data_dict = {}
    eq_data_dict = get_eq_data_dict(job, eq_data_dict, sim_name, property)
    vol_equib_data = np.array([eq_data_dict["Volume"]["data"]])
    ave_vol = np.mean(vol_equib_data)
    ave_length = ave_vol**(1/3)
    job.doc.xy_box_len = ave_length
    job.doc.aspect_ratio = 3.0

#Make Interface for simulation
@Project.operation(with_job = True, cmd=False)
def inter_eq_sim(job):
    box_len = job.doc.xy_box_len
    xy_cen = job.doc.xy_box_len/2
    z_cen = job.doc.xy_box_len * job.doc.aspect_ratio/2
    new_z_len = z_cen*2
    job.doc.z_box_len = new_z_len

    command = (
        f"gmx_d editconf -f $GRO_file -center {xy_cen} {xy_cen} {z_cen} -bt triclinic -box {box_len} {box_len} {new_z_len} -angles 90 90 90 -o init_inter_eq.gro"
    )
    subprocess.run(command, shell=True, check=True)

#Run Interface NVT equilibration
@Project.pre.after(nvt_eq2_sim)
@Project.post(inter_eq_complete)
@Project.operation(with_job = True, cmd=False)
def inter_eq_sim(job):
    """Run the interface equilibration simulations"""
    import panedr
    #Generate the first run
    last_sim_name = "init_inter_eq" #Use the same one since the -gro file is created beforehand
    sim_name = "inter_eq"   
    property = "#Surf*SurfTen" #Surface tension
    run_md_w_eqcheck(job, sim_name, last_sim_name, property)



#Run Interface NVT Production
@Project.pre.after(nvt_eq2_sim)
@Project.post(inter_eq_complete)
@Project.operation(with_job = True, cmd=False)
def inter_eq_sim(job):
    """Run the production simulations"""
    
    #Generate the first run
    last_sim_name = "inter_eq" #Use the same one since the -gro file is created beforehand
    sim_name = "inter_prod"   
    property = "#Surf*SurfTen" #Surface tension
    run_md_wo_eqcheck(job, sim_name, last_sim_name)


@Project.pre.after(simulate_prod)
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
                ferr.write("{}\t{}\t{}\t{}\n".format(nblk_ops, mean_est, var_est, var_err))

        job.doc[name + "_unc"] = np.max(np.sqrt(vars_est))


#####################################################################
################# HELPER FUNCTIONS BEYOND THIS POINT ################
#####################################################################
def _generate_r41_xml(job):

    content = """<ForceField>
 <AtomTypes>
  <Type name="C1" class="c3" element="C" mass="12.010" def="C(F)" desc="carbon"/>
  <Type name="F1" class="f" element="F" mass="19.000" def="F(C)" desc="F bonded to C1"/>
  <Type name="H1" class="h1" element="H" mass="1.008" def="H(C)" desc="H bonded to C1"/>
 </AtomTypes>
 <HarmonicBondForce>
  <Bond class1="c3" class2="f" length="0.1344" k="304427.36"/>
  <Bond class1="c3" class2="h1" length="0.1093" k="281080.35"/>
 </HarmonicBondForce>
 <HarmonicAngleForce>
  <Angle class1="f" class2="c3" class3="h1" angle="1.8823376" k="431.53717916"/>
  <Angle class1="h1" class2="c3" class3="h1" angle="1.9120082" k="327.85584464"/>
 </HarmonicAngleForce>
 <NonbondedForce coulomb14scale="0.833333" lj14scale="0.5">
  <Atom type="C1" charge="0.119281"  sigma="{sigma_C1:0.6f}" epsilon="{epsilon_C1:0.6f}"/>
  <Atom type="F1" charge="-0.274252" sigma="{sigma_F1:0.6f}" epsilon="{epsilon_F1:0.6f}"/>
  <Atom type="H1" charge="0.051657"  sigma="{sigma_H1:0.6f}" epsilon="{epsilon_H1:0.6f}"/>
 </NonbondedForce>
</ForceField>
""".format(
        sigma_C1=job.sp.sigma_C1,
        sigma_F1=job.sp.sigma_F1,
        sigma_H1=job.sp.sigma_H1,
        epsilon_C1=job.sp.epsilon_C1,
        epsilon_F1=job.sp.epsilon_F1,
        epsilon_H1=job.sp.epsilon_H1,
        
    )


    return content


def _generate_em_mdp(job):

    contents = """
; MDP file for energy minimization

integrator	    = steep		    ; Algorithm (steep = steepest descent minimization)
emtol		    = 100.0  	    ; Stop minimization when the maximum force < 100.0 kJ/mol/nm
emstep          = 0.01          ; Energy step size
nsteps		    = 50000	  	    ; Maximum number of (minimization) steps to perform

nstlist		    = 1		    ; Frequency to update the neighbor list and long range forces
cutoff-scheme   = Verlet
ns-type		    = grid		; Method to determine neighbor list (simple, grid)
verlet-buffer-tolerance = 1e-5          ; kJ/mol/ps
coulombtype	    = PME		; Treatment of long range electrostatic interactions
rcoulomb	    = 1.0		; Short-range electrostatic cut-off
rvdw		    = 1.0		; Short-range Van der Waals cut-off
pbc		        = xyz 		; Periodic Boundary Conditions (yes/no)
constraints     = all-bonds
lincs-order     = 8
lincs-iter      = 4
"""

    return contents


def _generate_eq_mdp(job):

    contents = """
; MDP file for NVT simulation

; Run parameters
integrator	            = md		    ; leap-frog integrator
nsteps		            = {nsteps}	    ;
dt		                = 0.001		    ; 1 fs

; Output control
nstxout		            = 10000		    ; save coordinates every 10.0 ps
nstvout		            = 0		        ; don't save velocities
nstenergy	            = 100		    ; save energies every 0.1 ps
nstlog		            = 100		    ; update log file every 0.1 ps

; Neighborsearching
cutoff-scheme           = Verlet
ns-type		            = grid		    ; search neighboring grid cells
nstlist		            = 10		    ; 10 fs, largely irrelevant with Verlet
verlet-buffer-tolerance = 1e-5          ; kJ/mol/ps

; VDW
vdwtype                 = Cut-off
rvdw		            = 1.0		    ; short-range van der Waals cutoff (in nm)
vdw-modifier            = None

; Electrostatics
rcoulomb	            = 1.0		    ; short-range electrostatic cutoff (in nm)
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
        temp=job.sp.T, press=job.sp.P, nsteps=job.sp.nstepseq
    )

    return contents


def _generate_prod_mdp(job):

    contents = """
; MDP file for NVT simulation

; Run parameters
integrator	            = md		    ; leap-frog integrator
nsteps		            = {nsteps}	    ;
dt		                = 0.001		    ; 1 fs

; Output control
nstxout		            = 10000		    ; save coordinates every 10.0 ps
nstvout		            = 0		        ; don't save velocities
nstenergy	            = 100		    ; save energies every 0.1 ps
nstlog		            = 100		    ; update log file every 0.1 ps

; Neighborsearching
cutoff-scheme           = Verlet
ns-type		            = grid		    ; search neighboring grid cells
nstlist		            = 10		    ; 10 fs, largely irrelevant with Verlet
verlet-buffer-tolerance = 1e-5          ; kJ/mol/ps

; VDW
vdwtype                 = Cut-off
rvdw		            = 1.0		    ; short-range van der Waals cutoff (in nm)
vdw-modifier            = None          ; standard LJ potential

; Electrostatics
rcoulomb	            = 1.0		    ; short-range electrostatic cutoff (in nm)
coulombtype	            = PME	        ; Particle Mesh Ewald for long-range electrostatics
pme-order	            = 4		        ; cubic interpolation
fourier-spacing         = 0.12          ; effects accuracy of pme
ewald-rtol              = 1e-5

; Temperature coupling is on
tcoupl		            = v-rescale     ; Bussi thermostat
tc-grps		            = System 	    ; Single coupling group
tau-t		            = 0.5	  		; time constant, in ps
ref-t		            = {temp}        ; reference temperature, one for each group, in K

; Pressure coupling is off
pcoupl		            = parrinello-rahman
pcoupltype              = isotropic
ref-p                   = {press}
tau-p                   = 1.0
compressibility         = 4.5e-5

; Periodic boundary conditions
pbc		                = xyz		    ; 3-D PBC

; Dispersion correction
DispCorr	            = EnerPres	    ; apply analytical tail corrections

; Velocity generation
gen-vel		            = no		    ; assign velocities from Maxwell distribution
gen-temp	            = {temp}        ; temperature for Maxwell distribution
gen-seed	            = -1		    ; generate a random seed

constraints             = all-bonds
lincs-order             = 8
lincs-iter              = 4
""".format(
        temp=job.sp.T, press=job.sp.P, nsteps=job.sp.nstepsprod
    )

    return contents

def check_equil_converge(job, eq_data_dict, prod_tol):
    equil_matrix = []
    res_matrix = []
    prop_cols = [0]
    prop_names = ["Volume"]
    try:
        # Load data for both boxes
        for key in list(eq_data_dict.keys()):
            eq_col = eq_data_dict[key]["data"]
            batch_size = max(1, int(len(eq_col) * 0.0005))

            # Try with ADF test enabled, fallback without it if it fails
            try:
                results = pymser.equilibrate(eq_col, LLM=False, batch_size=batch_size, ADF_test=True, uncertainty='uSD', print_results=False)
                adf_test_failed = results["critical_values"]["1%"] <= results["adf"]
            except:
                results = pymser.equilibrate(eq_col, LLM=False, batch_size=batch_size, ADF_test=False, uncertainty='uSD', print_results=False)
                results["adf"], results["critical_values"], adf_test_failed = None, None, False

            equilibrium = len(eq_col) - results['t0'] >= prod_tol
            equil_matrix.append(equilibrium and not adf_test_failed)
            res_matrix.append(results)
        
        for i, is_equilibrated in enumerate(equil_matrix):
            key_name = list(eq_data_dict.keys())[i]
            col_vals = eq_data_dict[key_name]["data"]
            #plot all

            # if not all(equil_matrix):
            plot_res_pymser(job, col_vals, res_matrix[i], prop_names[i % len(prop_cols)])

            # Display outcome
            prod_cycles = len(col_vals) - res_matrix[i]['t0']
            if is_equilibrated:
                #Plot successful equilibration
                statement = f"       > Success! Found {prod_cycles} production cycles."
            else:
                #Plot failed equilibration
                statement = f"       > Equil Failure! "
                if res_matrix[i]["adf"] is None:
                    # Note: ADF test failed to complete
                    statement += f"ADF test failed to complete! "
                elif res_matrix[i]['adf'] > res_matrix[i]['critical_values']['1%']:
                    adf, one_pct = res_matrix[i]['adf'], res_matrix[i]['critical_values']['1%']
                    statement += f"ADF value: {adf}, 99% confidence value: {one_pct}! "
                if len(col_vals) - res_matrix[i]['t0'] < prod_tol:
                   statement += f"Only {prod_cycles} production cycles found."
                
            with open("Equil_Output.txt", "a") as f:
                print(statement, file=f)

    except Exception as e:
        #This will cause an error in the GEMC operation which lets us know that the job failed
        raise Exception(f"Error processing job {job.id}: {e}")

def plot_res_pymser(job, eq_col, results, name):
    fig, [ax1, ax2] = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2, 1]}, sharey=True)

    ax1.set_ylabel(name, color="black", fontsize=14, fontweight='bold')
    ax1.set_xlabel("GEMC Steps", fontsize=14, fontweight='bold')

    ax1.plot(range(0, len(eq_col)*10, 10), 
            eq_col, 
            label = 'Raw data', 
            color='blue')

    ax1.plot(range(0, len(eq_col)*10, 10)[results['t0']:], 
            results['equilibrated'], 
            label = 'Equilibrated data', 
            color='red')

    ax1.plot([0, len(eq_col)*10], 
            [results['average'], results['average']], 
            color='green', zorder=4, 
            label='Equilibrated average')

    ax1.fill_between(range(0, len(eq_col)*10, 10), 
                    results['average'] - results['uncertainty'], 
                    results['average'] + results['uncertainty'], 
                    color='lightgreen', alpha=0.3, zorder=4)

    ax1.set_yticks(np.arange(0, eq_col.max()*1.1, eq_col.max()/10))
    ax1.set_xlim(-len(eq_col)*10*0.02, len(eq_col)*10*1.02)
    ax1.tick_params(axis="y", labelcolor="black")

    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.hist(eq_col, 
            orientation=u'horizontal', 
            bins=30, 
            edgecolor='blue', 
            lw=1.5, 
            facecolor='white', 
            zorder=3)

    bin_red = 10
    ax2.hist(results['equilibrated'], 
            orientation=u'horizontal', 
            bins=bin_red, 
            edgecolor='red', 
            lw=1.5, 
            facecolor='white', 
            zorder=3)

    ymax = int(ax2.get_xlim()[-1])

    ax2.plot([0, ymax], 
            [results['average'], results['average']],
            color='green', zorder=4, label='Equilibrated average')

    ax2.fill_between(range(ymax), 
                    results['average'] - results['uncertainty'],
                    results['average'] + results['uncertainty'],
                    color='lightgreen', alpha=0.3, zorder=4)

    ax2.set_xlim(0, ymax)

    ax2.grid(alpha=0.5, zorder=1)

    fig.set_size_inches(9,5)
    fig.set_dpi(100)
    fig.tight_layout()
    save_name = 'MSER_eq_vol.png'
    fig.savefig(job.fn(save_name), dpi=300, facecolor='white')
    plt.close(fig)

#HELPER FUNCTIONS
def run_md_wo_eqcheck(job, sim_name, last_sim_name):
    with job:
        if os.path.exists(sim_name+".cpt"):
            command = (
                f"gmx_d mdrun -cpi {sim_name}.cpt -v -deffnm {sim_name} -ntmpi 1 -ntomp 8 -nb gpu -pme gpu -bonded gpu"
            )
        else:
            command = (
            f"gmx_d grompp -maxwarn 5 -f {sim_name}.mdp -c {last_sim_name}.gro -p system.top -o {sim_name} && "
            f"gmx_d mdrun -v -deffnm {sim_name} -ntmpi 1 -ntomp 8 -nb gpu -pme gpu -bonded gpu"
        )
        subprocess.run(command, shell=True, check=True)
        
def run_md_w_eqcheck(job, sim_name, last_sim_name, property):
    with job:
        #Set number of iterations per extension and intitialize counter and total number of steps
        eq_extend = int(job.sp.nsteps_gemc_eq/4) #In femtoseconds
        total_eq_steps = job.sp.nsteps_gemc_eq #In femtoseconds
        existing_eq_steps = 0
        

        if max_eq_steps not in job.doc:
            job.doc.max_eq_steps = total_eq_steps*2
            #Get the total number of equilibration restarts and steps so far
            existing_eq_steps = count_steps(sim_name)*1000 #Convert to femtoseconds
            #The max number of steps is the larger of the number of steps + the org number of steps or the current max
            max_eq_steps = np.maximum(job.doc.max_eq_steps )#, existing_eq_steps + 2*job.sp.nsteps_gemc_eq)
            #Originally set the document eq_steps to the max number, it will be overwritten later
            job.doc.nsteps_gemc_eq = int(max_eq_steps)
            
        eq_data_dict = {}
        eq_data_dict = get_eq_data_dict(job, eq_data_dict, sim_name, property)
        
        while total_eq_steps < job.doc.max_eq_steps:
            #Set tolerance for determining equilibrium and check for convergence
            prod_tol_eq = count_steps(sim_name)/4 #In picoseconds (same units as the data)
            is_equil = check_equil_converge(job, eq_data_dict, prod_tol_eq)

            if is_equil:
                break
            else:
                #If you have enough steps, run the simulation, continue the simulation with more points
                if total_eq_steps <= max_eq_steps:
                    #If we have no steps, start the simulation
                    if total_eq_steps == 0:
                        command = (
                            f"gmx_d grompp -maxwarn 5 -f {sim_name}.mdp -c {last_sim_name}.gro -p system.top -o {sim_name} && "
                            f"gmx_d mdrun -v -deffnm {sim_name} -ntmpi 1 -ntomp 8 -nb gpu -pme gpu -bonded gpu"
                        )
                    #Otherwise, check log file for whether previous simulation finished correctly
                    elif check_norm_term(job, sim_name):
                        #If it finished, extend the simulation
                        command = (
                            f"gmx convert-tpr -s {sim_name}.tpr -extend " + eq_extend + f" -o {sim_name}.tpr &&"
                            f"gmx_d mdrun -s {sim_name}.tpr -cpi {sim_name}.cpt -v -deffnm {sim_name} -ntmpi 1 -ntomp 8 -nb gpu -pme gpu -bonded gpu "
                        )
                    #Otherwise restart the simulation from the checkpoint file
                    else:
                        command = (
                            f"gmx_d mdrun -cpi {sim_name}.cpt -v -deffnm eq -ntmpi 1 -ntomp 8 -nb gpu -pme gpu -bonded gpu"
                        )
                    subprocess.run(command, shell=True, check=True)
                    #Track the number of added steps
                    total_eq_steps += eq_extend

                    #Resave volume steps as needed
                    eq_data_dict = get_eq_data_dict(job, eq_data_dict, sim_name, property)

                #Otherwise report an error
                else:
                    job.doc.equil_fail = True
                    raise Exception(f"GEMC equilibration failed to converge after {max_eq_steps} steps")

        #Set the step counter to whatever the final number of equilibration steps was
        job.doc.nsteps_gemc_eq = total_eq_steps
        job.doc.equil_fail = False

def check_norm_term(job, sim_name):
    selected_file = job.fn(sim_name + ".log")
    with open(selected_file, "rb") as f:
        # Move the pointer to the end of the file, but leave space to find the last line
        f.seek(-2, os.SEEK_END)
        # Read backward until a newline is found
        while f.read(1) != b'\n':
            f.seek(-2, os.SEEK_CUR)
        # Read the last line after finding the newline
        last_line = f.readline().decode()
    if "Finished mdrun on rank" in last_line:
        return True
    else:
        return False
        
def get_eq_data_dict(job, eq_data_dict, sim_name, property):
    import panedr
    #Get the density and volume data
    df_all = panedr.edr_to_df(job.fn(sim_name+".edr"))
    with job:
        if property in df_all.columns:
            df = df[["Time", property]].copy()

        elif property in ["Volume", "Density"]:
            command = (
                f"gmx energy -f prd.edr -s prd.tpr -o {sim_name}_{property}.xvg << EOF &&"
                f"{property}"
                f"EOF"
            )
            subprocess.run(command, shell=True, check=True)
            prop_data = np.loadtxt(sim_name+"_"+property+".xvg", comments=["#" , "@"])
            df = pd.DataFrame(prop_data)

        property_data = df[:, 1]
        eq_col_file = job.fn(sim_name + "_" + property + ".csv")
        eq_data_dict[property] = {"data": property_data, "file": eq_col_file}  
        np.savetxt(eq_col_file, property_data, delimiter=",")  
        return eq_data_dict

def count_steps(job, sim_name):
    import panedr
    if os.path.exists(job.fn(sim_name+".edr")):
        # Extract the maximum time recorded
        df = panedr.edr_to_df(job.fn(sim_name+".edr"))
        time_total = df["Time"].max() #in picoseconds
    else:
        time_total = 0

    return time_total

if __name__ == "__main__":
    Project().main()
