# Simulation to run metric estimation, followed by geodesic shooting for verification
from apps import metricEstimation as me
from apps import geodesicShooting as gs
from util import YAMLcfg
from disp.vis import vis_tensors, vis_path, disp_scalar_to_file
from disp.vis import  disp_vector_to_file, disp_tensor_to_file
from disp.vis import disp_gradG_to_file, disp_gradA_to_file
from lazy_imports import plt, np
from util.parsers import parseMetricEstStdout

# for each parameter set
#   run metric estimation
#   extract number of iterations and final energy from output, save and plot it
#   run geodesic shooting on results
# save off plots

# Note that having to rerun multiple times for multiple iteration settings is done because the metric estimation solver doesn't write out these results after every iteration.  If it did, we could just run once and inspect results.  Instead we have to run multiple times and aggregate results.

#TODO better to call these experiments?

def run_simulation(sim_name, input_tensor, input_mask, bin_dir, input_dir, output_root_dir, use_minimum_energy, atols, alpha_inits, sigmas, start_coords, init_velocities, rhs_file=None):
  me_cfg = me.default_config()
  gs_cfg = gs.default_config()

  me_cfg.paths.commandPrefix = bin_dir
  me_cfg.paths.inputPrefix = input_dir
  me_cfg.options.inputTensorSuffix = input_tensor
  me_cfg.options.inputMaskSuffix = input_mask
  me_cfg.options.outputTensorSuffix = 'gmres_scaled_tensors.nhdr'
  me_cfg.options.euc_init.outputTensorSuffix = 'euc_scaled_tensors.nhdr'
  me_cfg.options.dim = 2
  me_cfg.options.use_minimum_energy = use_minimum_energy
  me_cfg.options.rtol = 1e-20 # make small so this isn't the reason for stopping

  gs_cfg.options.inputTensorSuffix = me_cfg.options.outputTensorSuffix
  gs_cfg.options.inputMaskFile = f'{me_cfg.paths.inputPrefix}/{me_cfg.options.inputMaskSuffix}'
  gs_cfg.options.doDataTranspose = True
  gs_cfg.options.transpose = (2, 0, 1)

  if rhs_file is not None:
    me_cfg.options.read_rhs_from_file = True
    me_cfg.options.rhsFilename = rhs_file
  
  # TODO do we want to pass these in instead?
  geo_delta_t = 0.005
  geo_iters = 44000 # 22000 for Kris annulus
  euler_delta_t = 0.01
  euler_iters = 10600

  results = []

  for init_alpha in alpha_inits:
    for atol in atols:
      for sigma in sigmas:
        test_case = f"alpha_init_{init_alpha}_atol_{atol}_sigma_{sigma}"
        test_dir = f"{output_root_dir}/{test_case}/"
        me_cfg.paths.outputPrefix = test_dir
        me_cfg.options.do_Euclidean_initialization = init_alpha
        me_cfg.options.atol = atol
        me_cfg.options.sigma = sigma
        
        result = {}
        result["test_case"] = test_case
        result["dir"] = test_dir
        result["tensor_file"] = f"{test_dir}/{me_cfg.options.outputTensorSuffix}"
        result["metricEst"] = {}
        result["metricEst"]["config"] = me_cfg
        
        met_est = me.MetricEstimator(me_cfg)
        try:
          print(f"Running metric estimation for {test_case}")
          (success, res) = met_est.run()
        
          # saving off config, commands, time to run is good provenance tracking
          # TODO figure out how to automate/standardize this to make it easier
          #    for other sim writing later
          result["metricEst"]["commands"] = met_est.commands
          result["metricEst"]["time"] = met_est.get_run_time()
          result["metricEst"]["stdout"] = res
        except:
          print(f"Error while running {test_case} with config {YAMLcfg.ConfigToYAML(me.metricEstimatorConfigSpec, me_cfg)}")
          success = False
          
        result["metricEst"]["status"] = success
        
        #TODO log commands
        #TODO parse results
        gs_cfg.paths.outputPrefix = me_cfg.paths.outputPrefix
        gs_cfg.paths.inputPrefix = me_cfg.paths.outputPrefix
        
        coord_results = []
        for coords in start_coords:
          for init_velocity in init_velocities:
            coord_res = {}
            coord_res["coords"] = coords
            coord_res["init_velocity"] = init_velocity
            coord_res["shooting"] = {}
            coord_res["euler"] = {}
            gs_cfg.options.start_coordinate = coords
            gs_cfg.options.initial_velocity = init_velocity
            gs_cfg.options.outputPathSuffix = f'{sim_name}_{coords}_{init_velocity}_geo_path.npy'
            gs_cfg.options.delta_t = geo_delta_t
            gs_cfg.options.num_iterations = geo_iters
        
            # TODO consider whether we want to construct new class instance for each test case
            # or reuse same existing class
            coord_res["shooting"]["config"] = gs_cfg
            shooter = gs.GeodesicShooter(gs_cfg)
            try:
              print(f"Running geodesic shooting for {test_case} and starting point {coords}")
              points_x, points_y = shooter.run()
              success = True
            except:
              print(f"Error while geodesic shooting for {test_case} with config {YAMLcfg.ConfigToYAML(gs.geodesicShootingConfigSpec, gs_cfg)}")
              success = False
        
            coord_res["shooting"]["status"] = success
            if success:
              coord_res["shooting"]["time"] = shooter.get_run_time()
              #TODO is it ok to save off all these x, y coords?  Not too expensive?
              coord_res["shooting"]["x"] = points_x
              coord_res["shooting"]["y"] = points_y
        
          
            gs_cfg.options.outputPathSuffix = f'{sim_name}_{coords}_{init_velocity}_euler_path.npy'
            gs_cfg.options.delta_t = euler_delta_t
            gs_cfg.options.num_iterations = euler_iters
        
            coord_res["euler"] = {}
            coord_res["euler"]["config"] = gs_cfg
            integrator = gs.EulerIntegrator(gs_cfg)
            try:
              print(f"Running Euler integration for {test_case} and starting point {coords}")
              points_x, points_y = integrator.run()
              success = True
            except:
              print(f"Error during Euler integration for {test_case} with config {YAMLcfg.ConfigToYAML(gs.geodesicShootingConfigSpec, gs_cfg)}")
              success = False
        
            coord_res["euler"]["status"] = success
            if success:
              coord_res["euler"]["time"] = integrator.get_run_time()
        
              #TODO is it ok to save off all these x, y coords?  Not too expensive?
              coord_res["euler"]["x"] = points_x
              coord_res["euler"]["y"] = points_y
        
            coord_results.append(coord_res)
          # end for each initial velocity
        # end for each start coordinate
        result["paths"] = coord_results
        results.append(result)
      # end for each sigma
    # end for each atol
  # end for each init_alpha
        
  print(f"Finished metricEstSim {sim_name}")
  return(results)

def summarize_results(results, sim_name, output_root_dir, subplot_nrows, subplot_ncols,
                      show_energy_plot):
  num_test_cases = len(results)
  cases_per_fig = subplot_nrows * subplot_ncols
  new_fig = True
  tens_fig = None
  energy_fig = None
  energy_resid_fig = None
  resid_fig = None
  fig_count = 1
  #fig_file_root = f"{sim_name}_paths_{fig_count}.png"
  # TODO decide whether want to keep using inputTensorSuffix from first result as background
  # tensor field to plot
  show_euler = True
  #colors = ['k', 'r', 'g', 'purple', 'b', 'orange']
  colors = ['k', 'tab:blue', 'tab:orange', 'tab:purple', 'tab:green', 'tab:red', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
  coloridx = 1
  case_count = 1

  data_cfg = gs.default_config()
  gs_cfg = results[0]["paths"][0]["shooting"]["config"]
  me_cfg = results[0]["metricEst"]["config"]
  data_cfg.paths.inputPrefix = me_cfg.paths.inputPrefix
  data_cfg.options.inputTensorSuffix = me_cfg.options.inputTensorSuffix
  data_cfg.options.inputMaskFile = gs_cfg.options.inputMaskFile
  data_cfg.options.doDataTranspose = gs_cfg.options.doDataTranspose
  data_cfg.options.transpose = gs_cfg.options.transpose
  
  tensors, mask = gs.load_data(data_cfg)

  scalar_files = ['divergence_rhs', 'rhs', 'output_result_rhs', 'output_alpha_final', 'output_result_lhs', 'usermult2_lhs']
  vector_files = ['cross_x_first_nablaTT', 'cross_x_second_nablaTT', 'cross_x_nablaTT',
                  'cross_y_first_nablaTT', 'cross_y_second_nablaTT', 'cross_y_nablaTT',
                  'cross_x_T','cross_y_T', 'output_dot_T',
                  'dot_T_after_cross_x_T', 'dot_T_after_cross_y_T',
                  'cross_x_sqrt_det_nablaTT', 'cross_x_sqrt_det_nablaTT_after_expend',
                  'cross_y_sqrt_det_nablaTT', 'cross_y_sqrt_det_nablaTT_after_expend']
  tensor_files = ['output_dot_g', 'output_dot_g_inv']
  gradG_files = ['cross_x_gradG', 'cross_y_gradG']
  gradA_files = ['cross_x_gradA', 'cross_y_gradA']
  
  for test in results:
    if test["metricEst"]["status"] == False:
      continue

    if new_fig:
      tens_fig = vis_tensors(tensors, f"{sim_name}", False)
      energy_fig = plt.figure()
      plt.title(f"{sim_name} energy")
      plt.xlabel('Iteration')
      plt.ylabel('Energy')
      energy_resid_fig = plt.figure()
      plt.title(f"{sim_name} residual norm vs energy")
      plt.xlabel('-log(Residual Norm)')
      plt.ylabel('Energy')
      resid_fig = plt.figure()
      plt.title(f"{sim_name} residual norm")
      plt.xlabel('Iteration')
      plt.ylabel('Residual Norm')


    label =  f"{test['test_case']}"

    # plot energies and residual
    if show_energy_plot:
      energies = parseMetricEstStdout(test["metricEst"]["stdout"][-1])
      vis_path(energies['KSP Residual norm']['iter'], energies['KSP Residual norm']['value'], resid_fig, label, colors[coloridx], 5, 1, False, yscale='log')
      vis_path(energies['current energy']['iter'], energies['current energy']['value'], energy_fig, label, colors[coloridx], 5, 1, False, yscale='log')
      vis_path(-np.log(energies['KSP Residual norm']['value']), energies['current energy']['value'], energy_resid_fig, label, colors[coloridx], 5, 1, False, yscale='log')
    
    # plot paths  
    # TODO still fit everything on one plot?  
    for path in test["paths"]:
      coords = path["coords"]
      if path["shooting"]["status"]:
        vis_path(path["shooting"]["x"], path["shooting"]["y"], tens_fig, label+"(Geodesic)", colors[coloridx], 20, 1, False)

      if show_euler and path["euler"]["status"]:
        vis_path(path["euler"]["x"], path["euler"]["y"], tens_fig, label+"(Euler)", colors[0], 20, 1, False)

    # plot images
    for sf in scalar_files:
      fname = f"{output_root_dir}/{label}/{sf}"
      try:
        disp_scalar_to_file(fname, fname + f'_{label}.png', label)
      except:
        pass
    for vf in vector_files:
      fname = f"{output_root_dir}/{label}/{vf}"
      outfiles = [fname + '_{}_{}.png'.format(label, i) for i in range(2)]
      try:
        disp_vector_to_file(fname, outfiles, vf)
      except:
        pass
    for tf in tensor_files:
      fname = f"{output_root_dir}/{label}/{tf}"
      outfiles = [fname + '_{}_{}.png'.format(label, i) for i in range(3)]
      try:
        disp_tensor_to_file(fname, outfiles, tf)
      except:
        pass
    for gf in gradG_files:
      fname = f"{output_root_dir}/{label}/{gf}"
      outfiles = [fname + '_{}_{}.png'.format(label, i) for i in range(6)]
      try:
        disp_gradG_to_file(fname, outfiles, gf)
      except:
        pass
    for af in gradA_files:
      fname = f"{output_root_dir}/{label}/{af}"
      outfiles = [fname + '_{}_{}.png'.format(label, i) for i in range(4)]
      try:
        disp_gradA_to_file(fname, outfiles, af) 
      except:
        pass
        
    # After first data set no more Euler   
    show_euler = False
    coloridx += 1
    if coloridx == len(colors):
      coloridx = 1
    new_fig = False
    case_count += 1
    if case_count > num_test_cases / 2.0:
      new_fig = True
      case_count = 1
      tens_fig.savefig(f"{output_root_dir}/{sim_name}_paths_{fig_count}.png")
      plt.close(tens_fig)
      resid_fig.savefig(f"{output_root_dir}/{sim_name}_residual_{fig_count}.png")
      plt.close(resid_fig)
      energy_fig.savefig(f"{output_root_dir}/{sim_name}_energy_{fig_count}.png")
      plt.close(energy_fig)
      energy_resid_fig.savefig(f"{output_root_dir}/{sim_name}_residual_vs_energy_{fig_count}.png")
      plt.close(energy_resid_fig)
      fig_count += 1
      show_euler = True


               

  
