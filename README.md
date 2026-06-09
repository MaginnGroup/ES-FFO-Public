# Transferable Force Fields for Six Common Solvents from Liquid Density and Surface Tension Data: A Gaussian Process Optimization Framework
Authors: Montana N. Carlozo, Alexander W. Dowling, and Edward J. Maginn
<!-- Introduction: Provide a brief introduction to the project, including its purpose, goals, and any key features or benefits. -->
## Introduction
**ES-FFO** is a repository used to rapidly calibrate the LJ parameters of electrolyte solvent forcefields given experimental data. The key feature of this work is using machine learning tools in the form of Gaussian processes (GPs) which allow us to cheaply estimate the results of a molecular simulation given temperature state points and thermophysical property data. This workflow combines the approaches of Wang et al., 2023, J. Chem. Theory Comput. and Carlozo et al., 2025, Digital Discovery to create one repository to build GP models and optimize parameters for electrolyte solvent FFs.

**NOTE**: We use Signac and signac flow (`<https://signac.io/>`) to manage the setup and execution of each workflow. These instructions assume a working knowledge of that software.

## Citation
This work is currently in preparation and has not yet been published. Please cite the following when referring to this work:

Carlozo, M. N., Dowling, A. W., \& Maginn, E. J.. Transferable Force Fields for Six Common Solvents from Liquid Density and Surface Tension Data: A Gaussian Process Optimization Framework. In preparation.
   
## Available Data

### Repository Organization
The repository is organized as follows: <br />
ES-FFO/ is the top level directory. It contains: <br />
1. .gitignore prevents large files from the signac workflow and plots from being tracked by git and prevents  tracking of other unimportant files. <br />
2. utils/ contains functions used by all pieces of this workflow including molecule experimental data and reference parameters <br />
3. block_averages/ contains a code for block averaging via the methods in H. Flyvbjerg and H.G. Peterson. Error estimates on averages of correlated data. J. Chem. Phys. 91:461-466, 1989.<br />
4. fffit/ contains another set of utility functions used by multiple aspects of this workflow. <br />
5. Build_GPs/ contains the workflow runs and analyzes results from applying the methods modified from those in Wang et al. 2023 to create Base FFs <br />
6. Opt_ES/ contains the workflow runs and analyzes results from applying the methods in Carlozo et al. 2025. to create GP-optimized FFs <br />
7. Opt_ES/ Also contains the molecular simulation results and analysis of the FF(s) developed in steps 5 and 6.  <br />
8. hfcs-fffit.yml is the environment for running this workflow. <br />
9. submit_jobs is a shell script for submitting jobs to the cluster. <br />
10. create_analysis_figs.ipynb is a jupyter notebook for creating the final analysis figures including all figures from running the scripts in 11-17. <br />
11. make_corr_figs_all_molec.py is a python script for creating correlation figures for all molecules. Figure 12, Figure 14, and SI Figure S12 in the main text. <br />
12. make_corr_figs_one_molec.py is a python script for creating correlation figures for individual molecules. SI Figures S1-S6 and SI Figures S13-S24 in the main text. <br />
13. make_GP_vs_sim_and_sens.py is a python script for creating figures comparing GP and actual predictions and generating the results for individual molecule sensitivity analyses. Figure 9, SI Figure S11, and SI Tables S2-S13 in the main text. <br />
14. make_ift_val_figs.py is a python script for creating visualizations of the interfactial tension boxes for all ST simulations. <br />
15. make_param_comp_figs.py is a python script for creating parameter comparison figures. Figures 10-11 and SI Figures S7-S10 in the main text. <br />
16. make_pareto_comp_figs.py is a python script for visualizing differences between pareto-optimal LJ parameter sets in ST (and LD) iterations. Figure 13 in the main text. <br />
Note that all workspace/ subdirectories below include one example signac workspace with some example input data and key simulation validation figures where appropriate.


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
This directory contains all data related to running the GP building workflow, including:
1. analysis/mol/ld_iters: Results for liquid density (LD) iterations for each molecule "mol"
2. analysis/mol/vle_iters: Results for surface tension (ST) iterations
3. utils/*.py: Functions for analyzing the results of LD and ST iterations
4. ld_iters/ : The signac project directory for running the LD iterations including project.py, templates/, and workspace/ 
5. vle_iters/ : The signac project directory for running the ST iterations including project.py, templates/, and workspace/
6. gen_lhs_samples.py: A script for generating the LHS data necessary for LD and ST iterations
7. init_LD.py: A script to generate the signac project for LD iterations
8. init_IFT.py: A script to generate the signac project for ST iterations
9. post_ld.py: A script to analyze LD iteration data
10. post_vle.py: A script to analyze ST iteration data
11. get_CPU_hrs.py : A script to calculate the total CPU hours used by the workflow

### Opt_ES/
This directory contains all data related to running the optimization workflow of Carlozo et al., 2025, Digital Discovery including:
1. analysis/: Data from analyzing the results of the optimization workflow
2. utilsOpt/: Functions for running the optimization workflow
3. gemc_val_opt/: The signac project directory for running the gemc validation for GP-optimized FF including project.py, templates/, and workspace/ 
4. gemc_val_no_opt/: The signac project directory for running the gemc validation for the Base FF including project.py, templates/, and workspace/
5. ift_val_opt/: The signac project directory for running the interfacial tension validation for GP-optimized FF including project.py, templates/, and workspace/
6. ift_val_no_opt/: The signac project directory for running the interfacial tension validation for the Base FF including project.py, templates/, and workspace/
7. opt_at_params_new/: The signac project directory for running the optimization of FF parameters with updated GP models including project.py, templates/, and workspace/
8. compress_gemc.py: A script to compress the gemc validation data
9. compress_ift.py: A script to compress the interfacial tension validation data
10. get_CPU_hrs.py: A script to calculate the total CPU hours used by the optimization and validation workflows
11. get_GPU_hrs.py: A script to calculate the total GPU hours used by the optimization and validation workflows
12. init_gemc_val.py: A script to initialize the gemc validation workflow
13. init_ift_val.py: A script to initialize the interfacial tension validation workflow
14. init_opt_at.py: A script to initialize the optimization workflow
15. post_analysis_opt.py: A script to analyze the results of the optimization workflow
16. post_analysis_val.py: A script to analyze the results of the validation workflows. Generates Figures 3-8 and the data in Tables 4-5 of the main text

#### Opt_ES/analysis/
This directory contains data from analyzing the results of the optimization workflow for each molecule "mol".:
1. at_00/mol/ExpVal/opt_res/jac_approx:  Includes jacobian approximations
2. at_00/mol/ExpVal/opt_res/hess_approx:  Includes hessian approximations
3. at_00/mol/ExpVal/opt_res/MAPD:  Includes Mean Absolute Percentage Deviation values
4. at_00/mol/ExpVal/opt_res/ms_val_opt:  Includes the error data and validation results for the GP-Optimized FF
5. at_00/mol/ExpVal/opt_res/ms_val_no_opt:  Includes the error data and validation results for the Base FF
6. at_00/mol/ExpVal/opt_res/prop_pred/:  Includes comparisons of GP, simulated, and experimental values for the GP-optimized and Base FFs
7. at_00/mol/ExpVal/opt_res/best_per_run.csv:  The best FF optimization results for each optimization run
8. at_00/mol/ExpVal/opt_res/pareto_info.csv:  The pareto-optimal FF optimization results for each molecule
9. at_00/mol/ExpVal/opt_res/unique_best_set.csv:  The unique LJ parameter sets identified for each molecule during optimization
10. AT-0/ms_val_no_opt/: Base FF property prediction plots for heat of vaporization and vapor pressure (h_p_vap.pdf), surface tension (surf_tens.pdf), and density (vle.pdf)
11. AT-0/ms_val_opt/: GP-optimized FF property prediction plots for heat of vaporization and vapor pressure (h_p_vap.pdf), surface tension (surf_tens.pdf), and density (vle.pdf)
12. AT-0/ms_val_opt_comp/: GP-optimized FF and Base FF property prediction plots. (h_p_vap.pdf, surf_tens.pdf, vle.pdf)
13. comp_err_data.csv: Data comparing the errors of different FFs
14. lit_err_data.csv: Data comparing the errors of the literature FFs
15. lit_ff_data.csv: Data comparing the properties of the literature FFs
16. lit_ff_data_w_no_opt.csv: Data comparing the properties of the literature FFs with the Base FFs
17. lit_ff_data_w_opt.csv: Data comparing the properties of the literature FFs with the GP-optimized FFs
18. lit_Hvap_est_w_no_opt.csv: Data comparing the heat of vaporization of the literature FFs and Base FFs
19. lit_Hvap_est_w_opt.csv: Data comparing the heat of vaporization of the literature FFs and GP-optimized FFs

#### Opt_ES/utilsOpt/
This directory contains all functions related to running the optimization workflow of Carlozo et al., 2025, Digital Discovery including:
1. utilsOpt/atom_type.py: Class definitions for atom typing schemes
2. utilsOpt/opt_atom_types.py: Other functions necessary for optimizing FF parameters and analyzing workflow results
3. utilsOpt/plot.py: Plotting functions for analyzing optimization results
4. utilsOpt/signac.py: Functions for extracting signac data


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

### Liquid Density Optimization (LD Iterations)
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
     qsub -N postLD submit_job post_ld.py
   ```       

### Surface Tension Optimization (ST Iterations)

To run surface tension iterations, follow the following steps:
1. Initialize Signac workflow
   ```
     conda activate hfcs-fffit
     cd Build_GPs/
     python init_IFT.py
   ```    
2. Run ST iterations
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
     qsub -N postVLE submit_job post_vle.py
   ```  

### Distinct Atom Type Optimization
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
   ``` 

### GP-optimized and Base FF Validation
To validate the generalized FF from above, follow the following steps:
1. Initialize Signac workflow
   ```
     conda activate hfcs-fffit
     cd Opt_ES/
   ```    
2. Run GEMC validation simulations (Pvap, rho_v, Hvap) - Simulatenous with Steps 5-7
   - **Note: rm -r workspace/ signac_project_document.json signac.rc will remove everything and allow you to start fresh if you mess up**
  ```
     cd Opt_ES/gemc_val_xxx/
     python project.py submit -o NPT_NVT --bundle=6 --parallel
     python project.py submit -o EQ --bundle=6 --parallel
     python project.py submit -o PROD --bundle=6 --parallel
     
   ``` 
Note: In step 2, "xxx" represents which FF is being validated. use "no_opt" for the Base FF and  "opt" for the GP-optimized FF.
3. Check status a few times throughout the process
   ```  
     python project.py status
   ```   
Note: Step 2 operation is multiple operations in series. Alternatively, the following can be run one at a time:
4. Alternative submission
   ```
     python project.py run -o create_forcefield
     python project.py run -o calc_boxes
     python project.py submit -o NVT_liqbox --bundle=4 --parallel
     python project.py run -o extract_final_NVT_config
     python project.py submit -o NPT_liqbox --bundle=4 --parallel
     python project.py run -o extract_final_NPT_config
     python project.py submit -o run_gemc_eq --bundle=4 --parallel
     python project.py submit -o gemc_eq_restart --bundle=4 --parallel (optional)
     python project.py submit -o run_gemc_prod --bundle=4 --parallel
     python project.py submit -o check_prod_data --bundle=4 --parallel
     python project.py submit -o calculate_props --bundle=4 --parallel
     python project.py run -o del_job (optional)
   ```
     
5. Run validation MD interface simulations (rho_l and surface tension) - Simultaneously with Steps 2-4
   - **Note: rm -r workspace/ signac_project_document.json signac.rc will remove everything and allow you to start fresh if you mess up**
  ```
     cd Opt_ES/ift_val_xxx/
     python project.py submit -o IFT #--bundle=2 --parallel
   ```  
Note: In step 5, "xxx" represents which FF is being validated. use "no_opt" for the Base FF and  "opt" for the GP-optimized FF.
6. Check status a few times throughout the process
   ```  
     python project.py status
   ```   
Note: Step 2 operation IFT runs multiple operations in series. Alternatively, the following can be run one at a time:
7. Alternative submission
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
     
8. Extract and analyze data:
   ```
     cd Opt_ES/
     qsub -N postOptAnaly ../submit_job post_analysis_val.py

     cd ../
     qsub -N makeCorrAll submit_job make_corr_figs_all_molec.py
     qsub -N makeCorrInd submit_job make_corr_figs_one_molec.py
     qsub -N makeGPSimSense submit_job make_GP_vs_sim_and_sens.py
     qsub -N makeIFTValfigs submit_job make_ift_val_figs.py
     qsub -N makeParamComp submit_job make_param_comp_figs.py
     qsub -N makeParetoComp submit_job make_pareto_comp_figs.py
   ``` 
Note: The second set of jobs create final figures for the SI and main text. Alternatively, these can be created by running the jupyter notebook `create_analysis_figs.ipynb`.

### Known Issues
The instructions outlined above seem to be system-dependent. In some cases, users have the following error:
```
ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.29' not found
```
If you observe this, please try the following in the terminal
```
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
```
which should fix the problem. This is not an optimal solution and is something we would like to address. We found that related projects [1](https://github.com/openmm/openmm/issues/3943), [2](https://github.com/conda/conda/issues/12410) have similar issues.
If you are aware of a robust solution to this issue, please let us know by raising an issue or sending an email!

## Credits
This research is based upon work supported by the National Science Foundation under award number EEC-2330175 for the Engineering Research Center EARTH. Computing resources were provided by the Center for Research Computing (CRC) at the University of Notre Dame. MC acknowledges support from the Graduate Assistance in Areas of National Need fellowship from the Department of Education, grant number P200A210048.

## Contact
Please contact Montana Carlozo (mcarlozo@nd.edu) with any questions, suggestions, or issues.