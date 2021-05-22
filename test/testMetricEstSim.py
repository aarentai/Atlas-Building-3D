import pickle
from sims import metricEstSim as mes

# to avoid unittest, run from top level metpy directory as
# PYTHONPATH=. python3 test/testMetricEstSim.py

if __name__ == "__main__":
  # TODO make a script to read these values in from command line
  #sim_name = "Xiang_annulus"
  #input_tensor = "SimDTI_alpha10_method2_2D.nhdr"
  #input_mask = "SimDTI_alpha10_method2_2D_mask.nhdr"
  #start_coords = [[20, 31]]
  #atols = [0.001, 0.0005, 0.0004, 0.0003, 0.0002, 0.0001, 0.00005]
  #atols = [5e-5,4e-5,3e-5,2e-5,1e-5,8e-6,6e-6,4e-6]
  #atols = [2.9e-5,2.7e-5,2.5e-5,2.3e-5,2.1e-5,1.9e-5,1.7e-5,1.5e-5,1.3e-5,1.1e-5]
  sim_name = "Kris_annulus"
  #input_tensor = "half_annulus_2D_no_blur.nhdr"
  #input_mask = "half_annulus_2D_mask.nhdr"
  input_tensor = "metpy_annulus_antialiased.nhdr"
  input_mask = "metpy_annulus_antialiased_mask.nhdr"
  start_coords = [[15, 53]]
  init_velocities = [None]
  #sim_name = "cubic_1"
  #input_tensor = "cubic_1_2D_no_blur.nhdr"
  #input_mask = "cubic_1_2D_mask.nhdr"
  #start_coords = [[20, 72]]
  #init_velocities = [[0.97288936, -0.23127102]] # using default sends in the wrong direction
  #sim_name = "cubic_2"
  #input_tensor = "cubic_2_2D_no_blur.nhdr"
  #input_mask = "cubic_2_2D_mask.nhdr"
  #start_coords = [[26, 80]]
  #init_velocities = [[0.97239358, -0.23334678]] # changing sign from default to head in other direction
  bin_dir = "/home/sci/kris/Software/AdaptaMetric/build/metric"
  input_dir = "/usr/sci/projects/HCP/Kris/NSFCRCNS/TestData"
  #output_root_dir = f"/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/test_metpy/{sim_name}"
  #output_root_dir = f"/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/test_SolveAlpha2D_filt_symm_analytic_rhs/{sim_name}"
  #output_root_dir = f"/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/test_SolveAlpha2D_symm_analytic_rhs/{sim_name}"
  #output_root_dir = f"/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/test_SolveAlpha2D_analytic_rhs/{sim_name}"
  #output_root_dir = f"/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/test_SolveAlpha2D_analytic_rhs_useMinEnergy/{sim_name}"
  #output_root_dir = f"/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/test_SolveAlpha2D_np_rhs/{sim_name}"
  #output_root_dir = f"/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/test_SolveAlpha2D_np_rhs_useMinEnergy/{sim_name}"
  #output_root_dir = f"/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/test_SolveAlpha2D_np_thresh_rhs/{sim_name}"
  #output_root_dir = f"/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/test_SolveAlpha2D_div_U2/{sim_name}"
  #output_root_dir = f"/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/test_SolveAlpha2D_noMinEnergy/{sim_name}"
  #output_root_dir = f"/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/test_SolveAlpha2D_useMinEnergy/{sim_name}"
  #output_root_dir = f"/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/test_SolveAlpha2D_noMinEnergy_bdry/{sim_name}"
  #output_root_dir = f"/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/test_SolveAlpha2D_useMinEnergy_bdry/{sim_name}"
  output_root_dir = f"/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/test_SolveAlpha2D_noMinEnergy_ripples/{sim_name}"
  #output_root_dir = f"/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/test_SolveAlpha2D_useMinEnergy_ripples/{sim_name}"
  atols = [0.7, 0.5, 0.4, 0.35, 0.3, 0.2, 0.1, 0.01, 1e-3, 1e-5, 1e-9, 1e-15]
  #atols = [0.4, 0.35, 0.1, 0.01, 0.00001]
  #atols = [0.4, 0.39, 0.38, 0.37, 0.36, 0.35]
  #atols = [0.35]
  #atols = [0.4]
  #atols = [1e-15]
  #atols = [230, 7.5, 4.0, 0.4, 3.5, 3, 2.5, 2, 1.5, 1] # for sigma = 1.25 case
  #atols = [4.0]
  #atols = [100, 32, 25, 20, 15, 10, 5, 1] # for sigma = 1.25 case after fixing bdry stuff
  #atols = [25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15] # for sigma = 1.25 case after fixing bdry stuff
  #atols = [5, 4.5, 4, 3.5, 3, 2.5, 2, 1.5, 1] # for sigma = 1.25 case with analytic rhs 
  #atols = [2.4, 2.3, 2.2, 2.1, 2.0, 1.9, 1.8, 1.7, 1.6, 1.5] # for sigma = 1.25 case with analytic rhs
  #atols = [5e-5, 4e-5, 3.5e-5, 3e-5, 2e-5, 1e-5] # for  sigma = 1.25 case with filtered analytic rhs
  #atols = [330, 200, 151, 150.1] # for sigma = 1.25 case with np rhs
  #atols = [9,2,1.5,1.0,0.95,-.946] # for sigma = 1.25 case with np rhs after scaling
  #atols = [20, 15, 12, 11.87, 11.82, 11.81] # for sigma = 1.25 case with np thresh rhs 
  #atols = [5, 4.5, 4, 3.5, 3, 2.5, 2, 1.5, 1] # for sigma = 1.25 case with div U2
  #atols = [20, 15, 10, 9, 8, 7, 6, 5] # for sigma = 1.25 case with div U2
  #atols = [25] # for sigma = 1.25 case after fixing bdry stuff
  alpha_inits = [True,False]
  #alpha_inits = [True]
  #alpha_inits = [False]
  #sigmas = [0, 1.25]
  sigmas = [1.25]
  #sigmas = [0]
  use_minimum_energy = False
  #use_minimum_energy = True
  #atols = [0.3]
  #alpha_inits = [False]
  rhs_file = None
  
  # analytic solution for rhs
  #rhs_file = '/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/masked_analytic_div_alpha.nhdr'
  #rhs_file = '/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/symmetric_analytic_div_alpha.nhdr'
  #rhs_file = '/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/div_U2.nhdr'
  #rhs_file = '/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/filtered_symmetric_analytic_div_alpha.nhdr'
  #rhs_file = '/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/np_rhs.nhdr'
  #rhs_file = '/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/np_thresh_rhs.nhdr'
  results_from_file = False
  #results_from_file = True

  if results_from_file:
    print(f"Loading results for {sim_name}")
    with open(f'{output_root_dir}/{sim_name}_results.pkl', 'rb') as f:
      results = pickle.load(f)  
  else:
    print(f"Running simulation for {sim_name}")
    results = mes.run_simulation(sim_name, input_tensor, input_mask, bin_dir, input_dir, output_root_dir, use_minimum_energy, atols, alpha_inits, sigmas, start_coords, init_velocities, rhs_file)
    with open(f'{output_root_dir}/{sim_name}_results.pkl', 'wb') as f:
      pickle.dump(results, f)  

  mes.summarize_results(results, sim_name, output_root_dir, 1, 1, use_minimum_energy)

  
