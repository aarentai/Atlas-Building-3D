# an app to run SolveAlpha2D and SolveAlpha_GMRES2D
# options include to use or skip Euclidean initialization
# directory and file names to use
# tolerances and other options
from . import appTypes
from util import YAMLcfg
from data import fileManager as fm

MetricEstimatorConfigSpec = {
  'paths': appTypes.CmdLineAppConfigSpec,
  'options': { 'dim':
               YAMLcfg.Param(default = 3,
                             comment = 'dimension of data, used to select between 2D and 3D versions of underlying code.  Valid values are either 2 or 3.'),
               'inputTensorSuffix':
               YAMLcfg.Param(default = 'input_tensors.nhdr',
                             comment = 'name of input tensor file, not including path'),
               'inputMaskSuffix':
               YAMLcfg.Param(default = 'input_mask.nhdr',
                             comment = 'name of input mask file, not including path'),
               'outputTensorSuffix':
               YAMLcfg.Param(default = 'output_gmres_tensors.nhdr',
                             comment = 'name of output tensor file scaled by the Riemannian alpha, not including path'),
               'use_minimum_energy':
               YAMLcfg.Param(default = False,
                             comment = 'use the result with minimum energy instead of the result from the final iteration of GMRES'),
               'read_rhs_from_file':
               YAMLcfg.Param(default = False,
                             comment = 'Read RHS from file'),
                'rhsFilename':
               YAMLcfg.Param(default = '',
                             comment = 'name of rhs file, including path'),
               'do_Euclidean_initialization':
               YAMLcfg.Param(default = False,
                             comment = 'Run SolveAlpha to do Euclidean initialization'),
               'euc_init': {'outputTensorSuffix':
                            YAMLcfg.Param(default = 'output_euc_tensors.nhdr',
                                          comment = 'name of output tensor file scaled by the Euclidean alpha, not including path'),
                            'tol':
                            YAMLcfg.Param(default = 0.1,
                                          comment = 'Euclidean initialization stops when error is below tol.'),
                            'cf':
                            YAMLcfg.Param(default = 0.1,
                                       comment = 'Euclidean initialization stops if the CG iterations have converged or not changed by more than cf from previous iteration')
                           },
               'atol': YAMLcfg.Param(default = 0.1,
                                     comment = 'GMRES stops when the absolute size of the residual norm is below atol.'),
               'rtol': YAMLcfg.Param(default = 0.01,
                                     comment = 'GMRES stops when the  decrease  of  the  residual norm relative to the norm of the right hand side is less than rtol'),
               'sigma': YAMLcfg.Param(default = 0,
                                     comment = 'Sigma for blurring the input tensors prior to GMRES solver.  sigma = 0 skips blurring')
             },
  '_resource': 'AdaptaMetric_metricEstimation'
}

def default_config():
  return YAMLcfg.SpecToConfig(MetricEstimatorConfigSpec)

class MetricEstimator(appTypes.CmdLineApp):
  def __init__(self, config=None):
    cfg = config
    if config is None:
      cfg = default_config()
    super().__init__(None, None, cfg)
    self.name = "Metric Estimator"
    # TODO auto-detect 2D vs 3D
    # Note that 2D vs 3D also refers to whether the tensors themselves are 2D or 3D embedded in a 2D or 3D image.
    self.desc = """ Runs the metric estimators SolveAlpha followed by SolveAlpha_GMRES.
Set the config param options.dim to run the 2D versions SolveAlpha2D and SolveAlpha_GMRES2D"""

  def construct_commands(self):
    self.commands = []
    if self.cfg.options.do_Euclidean_initialization:
      cmd = 'SolveAlpha'
      if self.cfg.options.dim == 2:
        cmd = 'SolveAlpha2D'
      self.commands.append(f"{self.cfg.paths.commandPrefix}/{cmd} -it {self.cfg.paths.inputPrefix}/{self.cfg.options.inputTensorSuffix} -im {self.cfg.paths.inputPrefix}/{self.cfg.options.inputMaskSuffix} -o {self.cfg.options.euc_init.outputTensorSuffix} -method 3 -tol {self.cfg.options.euc_init.tol} -cf {self.cfg.options.euc_init.cf} -sir 1")

    cmd = 'SolveAlpha_GMRES'
    if self.cfg.options.dim == 2:
        cmd = 'SolveAlpha_GMRES2D'

    energy_flag = ''
    if self.cfg.options.use_minimum_energy:
      energy_flag = '-my_energy_ksp_monitor'

    if self.cfg.options.do_Euclidean_initialization:
      cmd = cmd + " -ii Alpha.nhdr -it PreprocessedTensorImage.nhdr"
    else:
      cmd = cmd + f" -it {self.cfg.paths.inputPrefix}/{self.cfg.options.inputTensorSuffix}"

    if self.cfg.options.read_rhs_from_file:
      cmd = cmd + f" -rhs {self.cfg.options.rhsFilename}"
      
    self.commands.append(f"{self.cfg.paths.commandPrefix}/{cmd} -im {self.cfg.paths.inputPrefix}/{self.cfg.options.inputMaskSuffix} -o {self.cfg.options.outputTensorSuffix} -iter 1 -method 3 -ksp_monitor -ksp_type gmres -ksp_gmres_restart 10000 -ksp_atol {self.cfg.options.atol} -ksp_max_it 500 -ksp_rtol {self.cfg.options.rtol} {energy_flag} -sigma {self.cfg.options.sigma}")
    
  def run(self):
    self.construct_commands()
    fm.create_dir_hierarchy(self.cfg.paths.outputPrefix)
    self.cwd = self.cfg.paths.outputPrefix
    return(super().run())
    
