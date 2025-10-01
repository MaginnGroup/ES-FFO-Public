# Machine Learning-Enabled Development of Accurate Force Fields for Refrigerants
Authors: Ning Wang, Montana Carlozo, Eliseo Marin-Rimoldi, Bridgette Belfort, Alexander W. Dowling, and Edward J. Maginn
<!-- Introduction: Provide a brief introduction to the project, including its purpose, goals, and any key features or benefits. -->
## Introduction
**ES-FFO** is a repository used to rapidly calibrate the LJ parameters of (generalized) electrolyte solvent forcefields given experimental data. The key feature of this work is using machine learning tools in the form of Gaussian processes (GPs) which allow us to cheaply estimate the results of a molecular simulation given temperature state points and thermophysical property data. This workflow combines the approaches of Wang et. al., 2023, J. Chem. Theory Comput. and GENFFPaperCitation to create one repository to build GP models and optimizae parameters for electrolyte solvent FFs.

**NOTE**: We use Signac and signac flow (`<https://signac.io/>`) to manage the setup and execution of each workflow. These instructions assume a working knowledge of that software.

## Citation
This work has been published on TBD, whose link is TBD. Please cite as:

CITATION HERE
   
## Available Data

### Repository Organization
The repository is organized as follows: <br />
ES-FFO/ is the top level directory. It contains: <br />
1. .gitignore prevents large files from the signac workflow and plots from being tracked by git and prevents  tracking of other unimportant files. <br />
2. utils/ contains functions used by all pieces of this workflow including molecule experimental data and reference parameters <br />
3. block_averages/ contains a code for block averaging via the methods in H. Flyvbjerg and H.G. Peterson. Error estimates on averages of correlated data. J. Chem. Phys. 91:461-466, 1989.<br />
4. fffit/ contains another set of utility functions used by multiple aspects of this workflow. <br />
5. Build_GPs/ contains the workflow runs and analyzed results from applying the methods in Wang et al. 2023 <br />
6. Opt_ES/ contains the workflow runs and analyzed results from applying the methods in genFF Paper HERE which create the generalized FF parameter set(s).<br />
7. Opt_ESFF_MS/ contains the molecular simulation results and analysis of the FF(s) developed in step 6 and GAFF benchmarks.  <br />
13. hfcs-fffit.yml is the environment for running this workflow. <br />


### utils/
This directory contains:
1. molec_class_files/esolvs.py: A .py file which loads all electrolyte solvent data into a class
2. prep_ms_data.py: Contains scripts used to prepare data for GP training and get error data


### fffit/fffit/
This directory contains functions used by both workflows:
1. models.py: Functions related to building GP models
2. pareto.py: Functions for determining pareto efficient points
3. plot.py: Functions for plotting data
4. signac.py: Functions for extracting signac data
5. utils.py: Functions for scaling and unscaling data and shuffling/splitting GP data

### Build_GPs/
This directory contains all data related to running the workflow of Wang et al. including:
1. Analysis/mol/ld_iters: Results for liquid density (LD) iterations for each molecule "mol"
2. Analysis/mol/vle_iters: Results for vapor-liquid-equilibrium (VLE) iterations
3. utils/*.py: Functions for analyzing the results of LD and VLE iterations
4. ld_iters/ : The signac project directory for running the LD iterations including project.py
5. vle_iters/ : The signac project directory for running the VLE iterations including project.py
6. gen_lhs_samples.py: A script for generating the LHS data necessary for LD and VLE iterations
7. init_LD.py: A script to generate the signac project for LD iterations
8. init_IFT.py: A script to generate the signac project for VLE iterations
9. post_ld.py: A script to analyze LD iteration data
10. post_vle.py: A script to analyze VLE iteration data

### Opt_ES/
This directory contains all data related to running the optimization workflow of genFFpaper HERE including:
1. utils/atom_types.py: Class definitions for atom typing schemes
2. utils/opt_atom_types.py: Other functions necessary for optimizing FF parameters and analyzing workflow results
3. init_opt_at.py: Script for initializing signac project for generalized FF optimization
4. post_analysis_opt.py: Script for analyzing optimized FF results
5. opt_ff/: Signac project directory for running the FF optimization including project.py
6. rcc_opt_at_analysis: Script for ranking parameters accoring to Yao 2003

### Opt_ESFF_MS/
This directory contains all data related to validating the generalized FF parameters:
WRITE ME LATER

## Installation
To run this software, you must have access to all packages in the hfcs-fffit environment (hfcs-fffit.yml) which can be installed using the instructions in the next section.
<!-- Installation: Provide instructions on how to install and set up the project, including any dependencies that need to be installed. -->
This package has a number of requirements that can be installed in
different ways. We recommend using a conda environment to manage
most of the installation and dependencies. However, some items will
need to be installed from source or pip.

Running the simulations will also require an installation of GROMACS.
This can be installed separately (see installation instructions
`here <https://manual.gromacs.org/documentation/2021.2/install-guide/index.html>`).

<!-- Usage: Provide instructions on how to use the project, including any configuration or customization options. Examples of usage scenarios can also be added. -->
## Usage

### Liquid Density Optimization
To run liquid density iterations, follow the following steps:
1. Initialize Signac workflow
   ```
     conda activate hfcs-fffit
     cd Build_GPs/
     python init_LD.py
   ```    
2. Run LD iterations
   - **Note: rm -r workspace/ signac_project_document.json signac.rc will remove everything and allow you to start fresh if you mess up**
  ```
     cd Build_GPs/ld_iters/
     python project.py submit -o LD --bundle=12 --parallel
   ```  
3. Check status a few times throughout the process
   ```  
     python project.py status
   ```   
Note: Step 2 operation LD runs multiple operations in series. Alternatively, the following can be run one at a time:
4. Alternative submission
   ```
     python project.py run -o create_forcefield
     python project.py submit -o create_system
     python project.py run -o fix_topology
     python project.py submit -o em_sim --bundle=12 --parallel
     python project.py submit -o nvt_eq_sim --bundle=12 --parallel
     python project.py submit -o npt_eq_sim --bundle=12 --parallel
     python project.py submit -o npt_prod_sim --bundle=12 --parallel
     python project.py submit -o calculate_props --bundle=12 --parallel
   ```
     
5. Extract densities, run GP optimization and get samples for the next iteration in the Build_GPs/ directory
   - **Note: vle_iters/ will be populated with parameter sets automatically once the termination criteria for LD iters is satisfied**
   ```
     qsub -N postLD submit_job_long post_ld.py
   ```       

### VLE Optimization

To run vapor-liquid-equilibrium iterations, follow the following steps:
1. Initialize Signac workflow
   ```
     conda activate hfcs-fffit
     cd Build_GPs/
     python init_IFT.py
   ```    
2. Run LD iterations
   - **Note: rm -r workspace/ signac_project_document.json signac.rc will remove everything and allow you to start fresh if you mess up**
  ```
     cd Build_GPs/vle_iters/
     python project.py submit -o IFT #--bundle=2 --parallel
   ```  
3. Check status a few times throughout the process
   ```  
     python project.py status
   ```   
Note: Step 2 operation IFT runs multiple operations in series. Alternatively, the following can be run one at a time:
4. Alternative submission
   ```
     python project.py run -o create_forcefield
     python project.py submit -o create_system
     python project.py run -o fix_topology
     python project.py submit -o em_sim #--bundle=2 --parallel
     python project.py submit -o nvt_eq_sim #--bundle=2 --parallel
     python project.py submit -o npzzat_eq_sim #--bundle=2 --parallel
     python project.py submit -o npzzat_prod_sim #--bundle=2 --parallel
     python project.py submit -o init_inter_eq_sim #--bundle=2 --parallel
     python project.py submit -o inter_eq_sim #--bundle=2 --parallel
     python project.py submit -o inter_prod_sim #--bundle=2 --parallel
     python project.py submit -o calculate_props #--bundle=2 --parallel
   ```
     
5. Extract properties, run GP optimization and get samples for the next iteration in the Build_GPs/ directory
   ```
     qsub -N postVLE submit_job_long post_vle.py
   ```  

### Generalized Atom Type and FF Optimization
To run generalized FF calibration, follow the following steps:
1. Initialize Signac workflow
   ```
     conda activate hfcs-fffit
     cd Opt_ES/
     python init_opt_at.py
   ```    
2. Run LD iterations
   - **Note: rm -r workspace/ signac_project_document.json signac.rc will remove everything and allow you to start fresh if you mess up**
  ```
     cd Opt_ES/opt_ff/
     python project.py submit -o OptGenFF --bundle=12 --parallel
   ```  
3. Check status a few times throughout the process
   ```  
     python project.py status
   ```   
Note: Step 2 operation OptGenFF multiple operations in series. Alternatively, the following can be run one at a time:
4. Alternative submission
   ```
     python project.py submit -o gen_pareto_sets --bundle=12 --parallel
     python project.py submit -o run_obj_alg --bundle=12 --parallel
   ```
     
5. Extract and analyze data:
   ```
     qsub -N postOptAnaly ../submit_job post_analysis_opt.py
     qsub -N postOptRank ../submit_job rcc_opt_at_analysis.py (optional)
   ``` 

### Generalized FF Validation
To validate the generalized FF from above, follow the following steps:
FINISH LATER
1. Initialize Signac workflow
   ```
     conda activate hfcs-fffit
     cd Opt_ESFF_MS/
   ```    
2. Run GEMC simulations (Pvap, rho_v, Hvap) and MD interface simulations (rho_l and surface tension)
   - **Note: rm -r workspace/ signac_project_document.json signac.rc will remove everything and allow you to start fresh if you mess up**
  ```
     cd Opt_ES/gemc_val/
     python project.py submit -o NPT_NVT --bundle=6 --parallel
     python project.py submit -o EQ --bundle=6 --parallel
     python project.py submit -o PROD --bundle=6 --parallel
     
   ```  
3. Check status a few times throughout the process
   ```  
     python project.py status
   ```   
Note: Step 2 operation XXXX multiple operations in series. Alternatively, the following can be run one at a time:
4. Alternative submission
   ```
   ```
     
5. Extract and analyze data:
   ```
     qsub -N postOptAnaly submit_job_long post_analysis_ms.py
   ``` 

### Known Issues
The instructions outlined above seem to be system-dependent. In some cases, users have the following error:
```
ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.29' not found
```
If you observe this, please try the following in the terminal
```
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```
which should fix the problem. This is not an optimal solution and is something we would like to address. We found that related projects [1](https://github.com/openmm/openmm/issues/3943), [2](https://github.com/conda/conda/issues/12410) have similar issues.
If you are aware of a robust solution to this issue, please let us know by raising an issue or sending an email!

## Credits
The authors thank the financial support from the National Science Foundation via two grants: EFRI DChem: Next-generation Low Global Warming Refrigerants, Award no. 2029354 and Collaborative Research: Development and Application of a Molecular and Process Design Framework for the Separation of Hydrofluorocarbon Mixtures, Award no. CBET-1917474. This research is based upon work supported by the National Science Foundation under award number ERC-2330175 for the Engineering Research Center EARTH. Computing resources were provided by the Center for Research Computing (CRC) at the University of Notre Dame. We also thank the Shiflett group from The University of Kansas for their collaboration. MC acknowledges support from the Graduate Assistance in Areas of National Need fellowship from the Department of Education, grant number P200A210048. 

## Contact
Please contact Montana Carlozo (mcarlozo@nd.edu) with any questions, suggestions, or issues.
