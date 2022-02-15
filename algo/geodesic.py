import math
from util.tensors import circ_shift, tens_interp, direction
from util.tensors import circ_shift_3d, tens_interp_3d, direction_3d
from util.tensors import circ_shift_torch, tens_interp_torch, direction_torch
from util.tensors import circ_shift_3d_torch, tens_interp_3d_torch, direction_3d_torch
from util.tensors import batch_tens_interp_3d, batch_direction_3d
from util.tensors import batch_tens_interp_3d_torch, batch_direction_3d_torch
from lazy_imports import np
from lazy_imports import torch
from util import diff
from util import maskops
from data import io

from numba import jit
# uncomment this for so that @profile is defined when not using kernprof
#def profile(blah):                
#  return blah

def get_gamma_ddot_at_point(x, y, Gamma_field, gamma_dot):
  tens = tens_interp(x,y,Gamma_field)
  Gamma11 = tens[0,0]
  Gamma12 = tens[0,1]
  Gamma22 = tens[1,1]

  gamma_ddot = -(Gamma11*gamma_dot[0]*gamma_dot[0]
                 +Gamma12*gamma_dot[0]*gamma_dot[1]
                 +Gamma12*gamma_dot[1]*gamma_dot[0]
                 +Gamma22*gamma_dot[1]*gamma_dot[1])

  return(gamma_ddot)

def get_gamma_ddot_at_point_torch(x, y, Gamma_field, gamma_dot):
  tens = tens_interp_torch(x,y,Gamma_field).clone()
  Gamma11 = tens[0,0]
  Gamma12 = tens[0,1]
  Gamma22 = tens[1,1]

  gamma_ddot = -(Gamma11*gamma_dot[0]*gamma_dot[0]
                 +Gamma12*gamma_dot[0]*gamma_dot[1]
                 +Gamma12*gamma_dot[1]*gamma_dot[0]
                 +Gamma22*gamma_dot[1]*gamma_dot[1])

  return(gamma_ddot)

def get_gamma_ddot_at_point_3d(x, y, z, Gamma_field, gamma_dot):
  tens = tens_interp_3d(x,y,z,Gamma_field)
  Gamma11 = tens[0,0]
  Gamma12 = tens[0,1]
  Gamma13 = tens[0,2]
  Gamma22 = tens[1,1]
  Gamma23 = tens[1,2]
  Gamma33 = tens[2,2]

  gamma_ddot = -(Gamma11*gamma_dot[0]*gamma_dot[0]
                 +Gamma12*gamma_dot[0]*gamma_dot[1]
                 +Gamma12*gamma_dot[1]*gamma_dot[0]
                 +Gamma13*gamma_dot[0]*gamma_dot[2]
                 +Gamma13*gamma_dot[2]*gamma_dot[0]
                 +Gamma22*gamma_dot[1]*gamma_dot[1]
                 +Gamma23*gamma_dot[1]*gamma_dot[2]
                 +Gamma23*gamma_dot[2]*gamma_dot[1]
                 +Gamma33*gamma_dot[2]*gamma_dot[2])

  return(gamma_ddot)

def get_gamma_ddot_at_point_3d_torch(x, y, z, Gamma_field, gamma_dot):
  tens = tens_interp_3d_torch(x,y,z,Gamma_field)
  Gamma11 = tens[0,0]
  Gamma12 = tens[0,1]
  Gamma13 = tens[0,2]
  Gamma22 = tens[1,1]
  Gamma23 = tens[1,2]
  Gamma33 = tens[2,2]

  gamma_ddot = -(Gamma11*gamma_dot[0]*gamma_dot[0]
                 +Gamma12*gamma_dot[0]*gamma_dot[1]
                 +Gamma12*gamma_dot[1]*gamma_dot[0]
                 +Gamma13*gamma_dot[0]*gamma_dot[2]
                 +Gamma13*gamma_dot[2]*gamma_dot[0]
                 +Gamma22*gamma_dot[1]*gamma_dot[1]
                 +Gamma23*gamma_dot[1]*gamma_dot[2]
                 +Gamma23*gamma_dot[2]*gamma_dot[1]
                 +Gamma33*gamma_dot[2]*gamma_dot[2])

  return(gamma_ddot)


#@jit(nopython=True)
def batch_get_gamma_ddot_at_point_3d(x, y, z, Gamma_field, gamma_dot):
  tens = batch_tens_interp_3d(x,y,z,Gamma_field)
  Gamma11 = tens[:,0,0]
  Gamma12 = tens[:,0,1]
  Gamma13 = tens[:,0,2]
  Gamma22 = tens[:,1,1]
  Gamma23 = tens[:,1,2]
  Gamma33 = tens[:,2,2]

  gamma_ddot = -(Gamma11*gamma_dot[:,0]*gamma_dot[:,0]
                 +Gamma12*gamma_dot[:,0]*gamma_dot[:,1]
                 +Gamma12*gamma_dot[:,1]*gamma_dot[:,0]
                 +Gamma13*gamma_dot[:,0]*gamma_dot[:,2]
                 +Gamma13*gamma_dot[:,2]*gamma_dot[:,0]
                 +Gamma22*gamma_dot[:,1]*gamma_dot[:,1]
                 +Gamma23*gamma_dot[:,1]*gamma_dot[:,2]
                 +Gamma23*gamma_dot[:,2]*gamma_dot[:,1]
                 +Gamma33*gamma_dot[:,2]*gamma_dot[:,2])

  return(gamma_ddot)

def batch_get_gamma_ddot_at_point_3d_torch(x, y, z, Gamma_field, gamma_dot):
  tens = batch_tens_interp_3d_torch(x,y,z,Gamma_field)
  Gamma11 = tens[:,0,0]
  Gamma12 = tens[:,0,1]
  Gamma13 = tens[:,0,2]
  Gamma22 = tens[:,1,1]
  Gamma23 = tens[:,1,2]
  Gamma33 = tens[:,2,2]

  gamma_ddot = -(Gamma11*gamma_dot[:,0]*gamma_dot[:,0]
                 +Gamma12*gamma_dot[:,0]*gamma_dot[:,1]
                 +Gamma12*gamma_dot[:,1]*gamma_dot[:,0]
                 +Gamma13*gamma_dot[:,0]*gamma_dot[:,2]
                 +Gamma13*gamma_dot[:,2]*gamma_dot[:,0]
                 +Gamma22*gamma_dot[:,1]*gamma_dot[:,1]
                 +Gamma23*gamma_dot[:,1]*gamma_dot[:,2]
                 +Gamma23*gamma_dot[:,2]*gamma_dot[:,1]
                 +Gamma33*gamma_dot[:,2]*gamma_dot[:,2])

  return(gamma_ddot)

#@jit(nopython=True)
def compute_gammas_3d(tensor_field, mask_image):
  # Compute the Christoffel symbols Gamma1, Gamma2, Gamma3
  eps11 = tensor_field[0, :, :, :]
  eps12 = tensor_field[1, :, :, :]
  eps13 = tensor_field[2, :, :, :]
  eps22 = tensor_field[3, :, :, :]
  eps23 = tensor_field[4, :, :, :]
  eps33 = tensor_field[5, :, :, :]

  # Compute inverse of g
  eps_11 = eps22 * eps33 - eps23 * eps23
  eps_12 = eps13 * eps23 - eps12 * eps33
  eps_13 = eps12 * eps23 - eps13 * eps22
  eps_22 = eps11 * eps33 - eps13 * eps13
  eps_23 = eps13 * eps12 - eps11 * eps23
  eps_33 = eps11 * eps22 - eps12 * eps12
  det_eps = eps11 * eps_11 + eps12 * eps_12 + eps13 * eps_13
  eps_11 = eps_11 / det_eps
  eps_12 = eps_12 / det_eps
  eps_13 = eps_13 / det_eps
  eps_22 = eps_22 / det_eps
  eps_23 = eps_23 / det_eps
  eps_33 = eps_33 / det_eps
    
  bdry_type, bdry_idx, bdry_map = maskops.determine_boundary_3d(mask_image, False)
  d1_eps_11, d2_eps_11, d3_eps_11 = diff.gradient_bdry_3d(eps_11, bdry_idx, bdry_map)
  d1_eps_12, d2_eps_12, d3_eps_12 = diff.gradient_bdry_3d(eps_12, bdry_idx, bdry_map)
  d1_eps_13, d2_eps_13, d3_eps_13 = diff.gradient_bdry_3d(eps_13, bdry_idx, bdry_map)
  d1_eps_22, d2_eps_22, d3_eps_22 = diff.gradient_bdry_3d(eps_22, bdry_idx, bdry_map)
  d1_eps_23, d2_eps_23, d3_eps_23 = diff.gradient_bdry_3d(eps_23, bdry_idx, bdry_map)
  d1_eps_33, d2_eps_33, d3_eps_33 = diff.gradient_bdry_3d(eps_33, bdry_idx, bdry_map)
    
  Gamma1_11 = (eps11 * d1_eps_11 + eps12 * (2 * d1_eps_12 - d2_eps_11) + eps13 * (2 * d1_eps_13 - d3_eps_11)) / 2
  Gamma1_12 = (eps11 * d2_eps_11 + eps12 * d1_eps_22 + eps13 * (d1_eps_23 + d2_eps_13 - d3_eps_11)) / 2
  Gamma1_13 = (eps11 * d3_eps_11 + eps12 * (d1_eps_23 + d3_eps_12 - d2_eps_13) + eps13 * (d1_eps_33)) / 2
  Gamma1_22 = (eps11 * (2 * d2_eps_12 - d1_eps_22) + eps12 * d2_eps_22 + eps13 * (2 * d2_eps_23 - d3_eps_22)) / 2
  Gamma1_23 = (eps11 * (d3_eps_12 + d2_eps_13 - d1_eps_23) + eps12 * d3_eps_22 + eps13 * (d2_eps_33)) / 2
  Gamma1_33 = (eps11 * (2 * d3_eps_13 - d1_eps_33) + eps12 * (2 * d3_eps_23 - d2_eps_33) + eps13 * d3_eps_33) / 2
  Gamma1 = np.stack((Gamma1_11, Gamma1_12, Gamma1_13, Gamma1_22, Gamma1_23, Gamma1_33))
  Gamma2_11 = (eps12 * d1_eps_11 + eps22 * (2 * d1_eps_12 - d2_eps_11) + eps23 * (2 * d1_eps_13 - d3_eps_11)) / 2
  Gamma2_12 = (eps12 * d2_eps_11 + eps22 * d1_eps_22 + eps23 * (d1_eps_23 + d2_eps_13 - d3_eps_11)) / 2
  Gamma2_13 = (eps12 * d3_eps_11 + eps22 * (d1_eps_23 + d3_eps_12 - d2_eps_13) + eps23 * (d1_eps_33)) / 2
  Gamma2_22 = (eps12 * (2 * d2_eps_12 - d1_eps_22) + eps22 * d2_eps_22 + eps23 * (2 * d2_eps_23 - d3_eps_22)) / 2
  Gamma2_23 = (eps12 * (d3_eps_12 + d2_eps_13 - d1_eps_23) + eps22 * d3_eps_22 + eps23 * (d2_eps_33)) / 2
  Gamma2_33 = (eps12 * (2 * d3_eps_13 - d1_eps_33) + eps22 * (2 * d3_eps_23 - d2_eps_33) + eps23 * d3_eps_33) / 2
  Gamma2 = np.stack((Gamma2_11, Gamma2_12, Gamma2_13, Gamma2_22, Gamma2_23, Gamma2_33))
  Gamma3_11 = (eps13 * d1_eps_11 + eps23 * (2 * d1_eps_12 - d2_eps_11) + eps33 * (2 * d1_eps_13 - d3_eps_11)) / 2
  Gamma3_12 = (eps13 * d2_eps_11 + eps23 * d1_eps_22 + eps33 * (d1_eps_23 + d2_eps_13 - d3_eps_11)) / 2
  Gamma3_13 = (eps13 * d3_eps_11 + eps23 * (d1_eps_23 + d3_eps_12 - d2_eps_13) + eps33 * (d1_eps_33)) / 2
  Gamma3_22 = (eps13 * (2 * d2_eps_12 - d1_eps_22) + eps23 * d2_eps_22 + eps33 * (2 * d2_eps_23 - d3_eps_22)) / 2
  Gamma3_23 = (eps13 * (d3_eps_12 + d2_eps_13 - d1_eps_23) + eps23 * d3_eps_22 + eps33 * (d2_eps_33)) / 2
  Gamma3_33 = (eps13 * (2 * d3_eps_13 - d1_eps_33) + eps23 * (2 * d3_eps_23 - d2_eps_33) + eps33 * d3_eps_33) / 2
  Gamma3 = np.stack((Gamma3_11, Gamma3_12, Gamma3_13, Gamma3_22, Gamma3_23, Gamma3_33))
  return(Gamma1, Gamma2, Gamma3)
# end compute_gammas_3d


def geodesicpath(tensor_field, mask_image, start_coordinate, initial_velocity, delta_t=0.15, metric='withoutscaling', iter_num=18000, filename = '', both_directions=False):
  geodesicpath_points_x = np.zeros((iter_num))
  geodesicpath_points_y = np.zeros((iter_num))
  eps11 = tensor_field[0, :, :]
  eps12 = tensor_field[1, :, :]
  eps22 = tensor_field[2, :, :]

  init_v = initial_velocity
  if initial_velocity is None:
    init_v = direction(start_coordinate, tensor_field)

  if both_directions:
    back_x, back_y = geodesicpath(tensor_field, mask_image, start_coordinate, [-init_v[0], -init_v[1]], delta_t, metric, iter_num, filename, both_directions=False)

  print(f"Finding geodesic path from {start_coordinate} with initial velocity {init_v}")

  # adaptive riemannian metric without scaling
  if metric=='withoutscaling':
    eps_11 = eps22 / (eps11 * eps22 - eps12 ** 2)
    eps_12 = -eps12 / (eps11 * eps22 - eps12 ** 2)
    eps_22 = eps11 / (eps11 * eps22 - eps12 ** 2)

    d1_eps_11, d2_eps_11 = diff.gradient_mask_2d(eps_11, mask_image)
    d1_eps_12, d2_eps_12 = diff.gradient_mask_2d(eps_12, mask_image)
    d1_eps_22, d2_eps_22 = diff.gradient_mask_2d(eps_22, mask_image)

    Gamma1_11 = (eps11 * d1_eps_11 + eps12 * (2 * d1_eps_12 - d2_eps_11)) / 2
    Gamma1_12 = (eps11 * d2_eps_11 + eps12 * d1_eps_22) / 2
    Gamma1_22 = (eps11 * (2 * d2_eps_12 - d1_eps_22) + eps12 * d2_eps_22) / 2
    Gamma1 = np.stack((Gamma1_11, Gamma1_12, Gamma1_22))
    Gamma2_11 = (eps12 * d1_eps_11 + eps22 * (2 * d1_eps_12 - d2_eps_11)) / 2
    Gamma2_12 = (eps12 * d2_eps_11 + eps22 * d1_eps_22) / 2
    Gamma2_22 = (eps12 * (2 * d2_eps_12 - d1_eps_22) + eps22 * d2_eps_22) / 2
    Gamma2 = np.stack((Gamma2_11, Gamma2_12, Gamma2_22))

  # adaptive riemannian metric with scaling
  elif metric == 'withscaling':
    scaling_field = np.loadtxt(open("input/e_alpha_kris.csv", "rb"), delimiter=",")
    eps11 = eps11 * scaling_field
    eps12 = eps12 * scaling_field
    eps22 = eps22 * scaling_field
    eps_11 = eps22 / (eps11 * eps22 - eps12 ** 2)
    eps_12 = -eps12 / (eps11 * eps22 - eps12 ** 2)
    eps_22 = eps11 / (eps11 * eps22 - eps12 ** 2)

    d1_eps_11, d2_eps_11 = diff.gradient_mask_2d(eps_11, mask_image)
    d1_eps_12, d2_eps_12 = diff.gradient_mask_2d(eps_12, mask_image)
    d1_eps_22, d2_eps_22 = diff.gradient_mask_2d(eps_22, mask_image)

    Gamma1_11 = (eps11 * d1_eps_11 + eps12 * (2 * d1_eps_12 - d2_eps_11)) / 2
    Gamma1_12 = (eps11 * d2_eps_11 + eps12 * d1_eps_22) / 2
    Gamma1_22 = (eps11 * (2 * d2_eps_12 - d1_eps_22) + eps12 * d2_eps_22) / 2
    Gamma1 = np.stack((Gamma1_11, Gamma1_12, Gamma1_22))
    Gamma2_11 = (eps12 * d1_eps_11 + eps22 * (2 * d1_eps_12 - d2_eps_11)) / 2
    Gamma2_12 = (eps12 * d2_eps_11 + eps22 * d1_eps_22) / 2
    Gamma2_22 = (eps12 * (2 * d2_eps_12 - d1_eps_22) + eps22 * d2_eps_22) / 2
    Gamma2 = np.stack((Gamma2_11, Gamma2_12, Gamma2_22))
  # simple conformal metric
  elif metric == 'simple':
    scaling_field = np.loadtxt(open("input/e_alpha_euclidean.csv", "rb"), delimiter=",")
    d1_alpha, d2_alpha = diff.gradient_mask_2d(scaling_field, mask_image)

    Gamma1_11 = d1_alpha
    Gamma1_12 = d2_alpha
    Gamma1_22 = -d1_alpha
    Gamma1 = np.stack((Gamma1_11, Gamma1_12, Gamma1_22))
    Gamma2_11 = -d2_alpha
    Gamma2_12 = d1_alpha
    Gamma2_22 = d2_alpha
    Gamma2 = np.stack((Gamma2_11, Gamma2_12, Gamma2_22))
  else:
    Gamma1_11 = 0
    Gamma1_12 = 0
    Gamma1_22 = 0
    Gamma1 = np.stack((Gamma1_11, Gamma1_12, Gamma1_22))
    Gamma2_11 = 0
    Gamma2_12 = 0
    Gamma2_22 = 0
    Gamma2 = np.stack((Gamma2_11, Gamma2_12, Gamma2_22))

  gamma = np.zeros((iter_num,2))
  gamma_dot = np.zeros((iter_num,2))
  gamma_ddot = np.zeros((iter_num,2))
  gamma[0, :] = start_coordinate
  gamma_dot[0, :] = init_v
  gamma_ddot[0, 0] = get_gamma_ddot_at_point(gamma[0,0], gamma[0,1], Gamma1, gamma_dot[0])
  gamma_ddot[0, 1] = get_gamma_ddot_at_point(gamma[0,0], gamma[0,1], Gamma2, gamma_dot[0])
  #print(eps_11[13,13], d1_eps_11[13,13])
  #print(Gamma1_11[13,13])
  #print(Gamma1[:,13,13], Gamma2[:,13,13])
  #print('tens_interp 0', tens_interp(gamma[0,0], gamma[0,1], Gamma1))
  #print('tens_interp 1', tens_interp(gamma[0,0], gamma[0,1], Gamma2))
  #print(gamma[0,0], gamma[0,1], gamma_dot[0], gamma_ddot[0,0], gamma_ddot[0,1])

  g_ddot_k1 = np.zeros((2))
  g_ddot_k2 = np.zeros((2))
  g_ddot_k3 = np.zeros((2))
  g_ddot_k4 = np.zeros((2))
  
  for i in range(1,iter_num):

    # Do Fourth-Order Runge Kutta
    g_ddot_k1 = gamma_ddot[i-1]
    g_dot_k1 = gamma_dot[i-1] + delta_t * g_ddot_k1
    g_k1 = gamma[i-1] + delta_t * g_dot_k1
    
    g_ddot_k2[0] = get_gamma_ddot_at_point(g_k1[0], g_k1[1], Gamma1, g_dot_k1)
    g_ddot_k2[1] = get_gamma_ddot_at_point(g_k1[0], g_k1[1], Gamma2, g_dot_k1)

    g_dot_k2 = gamma_dot[i-1] + 0.5 * delta_t * g_ddot_k2
    g_k2 = gamma[i-1] + 0.5 * delta_t * g_dot_k2

    g_ddot_k3[0] = get_gamma_ddot_at_point(g_k2[0], g_k2[1], Gamma1, g_dot_k2)
    g_ddot_k3[1] = get_gamma_ddot_at_point(g_k2[0], g_k2[1], Gamma2, g_dot_k2)

    g_dot_k3 = gamma_dot[i-1] + 0.5 * delta_t * g_ddot_k3
    g_k3 = gamma[i-1] + 0.5 * delta_t * g_dot_k3

    g_ddot_k4[0] = get_gamma_ddot_at_point(g_k3[0], g_k3[1], Gamma1, g_dot_k3)
    g_ddot_k4[1] = get_gamma_ddot_at_point(g_k3[0], g_k3[1], Gamma2, g_dot_k3)

    g_dot_k4 = gamma_dot[i-1] + delta_t * g_ddot_k4
    g_k4 = gamma[i-1] + delta_t * g_dot_k4

    
    gamma_dot[i, :] = gamma_dot[i-1, :] + delta_t/6.0*(g_ddot_k1 + 2.0*g_ddot_k2 + 2.0*g_ddot_k3 + g_ddot_k4)
    gamma[i, :] = gamma[i-1, :]+delta_t/6.0*(g_dot_k1 + 2.0*g_dot_k2 + 2.0*g_dot_k3 + g_dot_k4)
    gamma_ddot[i, 0] = get_gamma_ddot_at_point(gamma[i,0], gamma[i,1], Gamma1, gamma_dot[i])
    gamma_ddot[i, 1] = get_gamma_ddot_at_point(gamma[i,0], gamma[i,1], Gamma2, gamma_dot[i])
    ## print(gamma_dot[i-1, :])
    ## print('gamma_ddot', i-2, ': ', gamma_ddot[i-2, :], 'gamma_dot', i-1, ': ', gamma_dot[i-1, :], 'gamma', i, ': ', gamma[i, :])
    if (math.ceil(gamma[i, 0]) >= 0 and math.ceil(gamma[i, 0]) < np.size(eps11, 0)
       and math.ceil(gamma[i, 1]) >= 0 and math.ceil(gamma[i, 1])  <np.size(eps11, 1)
       and mask_image[int(math.ceil(gamma[i, 0])), int(math.ceil(gamma[i, 1]))] > 0):
      geodesicpath_points_x[i-1] = gamma[i, 0]
      geodesicpath_points_y[i-1] = gamma[i, 1]
    else:
      # truncate and stop
      geodesicpath_points_x = geodesicpath_points_x[:i-1]
      geodesicpath_points_y = geodesicpath_points_y[:i-1]
      break

  if both_directions:
    geodesicpath_points_x = np.concatenate((geodesicpath_points_x[::-1], back_x))
    geodesicpath_points_y = np.concatenate((geodesicpath_points_y[::-1], back_y))
    
  if filename:
    io.writePath(geodesicpath_points_x, geodesicpath_points_y, filename)

  return geodesicpath_points_x, geodesicpath_points_y
# end geodesicpath


def geodesicpath_torch(tensor_field, mask_image, start_coordinate, initial_velocity, delta_t=0.15, metric='withoutscaling', iter_num=18000, filename = ''):

  # Expect tensor_field and mask_image to by torch tensors
  
  geodesicpath_points_x = torch.zeros((iter_num))
  geodesicpath_points_y = torch.zeros((iter_num))
  eps11 = tensor_field[0, :, :]
  eps12 = tensor_field[1, :, :]
  eps22 = tensor_field[2, :, :]

  init_v = initial_velocity
  if initial_velocity is None:
    init_v = direction_torch(start_coordinate, tensor_field)

  print(f"Finding geodesic path from {start_coordinate} with initial velocity {init_v}")

  # adaptive riemannian metric without scaling
  if metric=='withoutscaling':
    eps_11 = mask_image * eps22 / (eps11 * eps22 - eps12 ** 2)
    eps_12 = mask_image * -eps12 / (eps11 * eps22 - eps12 ** 2)
    eps_22 = mask_image * eps11 / (eps11 * eps22 - eps12 ** 2)

    d1_eps_11 = (circ_shift_torch(eps_11, [-1, 0]) - circ_shift_torch(eps_11, [1, 0])) / 2
    d1_eps_12 = (circ_shift_torch(eps_12, [-1, 0]) - circ_shift_torch(eps_12, [1, 0])) / 2
    d1_eps_22 = (circ_shift_torch(eps_22, [-1, 0]) - circ_shift_torch(eps_22, [1, 0])) / 2
    d2_eps_11 = (circ_shift_torch(eps_11, [0, -1]) - circ_shift_torch(eps_11, [0, 1])) / 2
    d2_eps_12 = (circ_shift_torch(eps_12, [0, -1]) - circ_shift_torch(eps_12, [0, 1])) / 2
    d2_eps_22 = (circ_shift_torch(eps_22, [0, -1]) - circ_shift_torch(eps_22, [0, 1])) / 2

    Gamma1_11 = (eps11 * d1_eps_11 + eps12 * (2 * d1_eps_12 - d2_eps_11)) / 2
    Gamma1_12 = (eps11 * d2_eps_11 + eps12 * d1_eps_22) / 2
    Gamma1_22 = (eps11 * (2 * d2_eps_12 - d1_eps_22) + eps12 * d2_eps_22) / 2
    Gamma1 = torch.stack((Gamma1_11, Gamma1_12, Gamma1_22))
    Gamma2_11 = (eps12 * d1_eps_11 + eps22 * (2 * d1_eps_12 - d2_eps_11)) / 2
    Gamma2_12 = (eps12 * d2_eps_11 + eps22 * d1_eps_22) / 2
    Gamma2_22 = (eps12 * (2 * d2_eps_12 - d1_eps_22) + eps22 * d2_eps_22) / 2
    Gamma2 = torch.stack((Gamma2_11, Gamma2_12, Gamma2_22))

  # adaptive riemannian metric with scaling
  elif metric == 'withscaling':
    scaling_field = torch.from_numpy(np.loadtxt(open("input/e_alpha_kris.csv", "rb"), delimiter=","))
    eps11 = eps11 * scaling_field
    eps12 = eps12 * scaling_field
    eps22 = eps22 * scaling_field
    eps_11 = mask_image * eps22 / (eps11 * eps22 - eps12 ** 2)
    eps_12 = mask_image * -eps12 / (eps11 * eps22 - eps12 ** 2)
    eps_22 = mask_image * eps11 / (eps11 * eps22 - eps12 ** 2)

    d1_eps_11 = (circ_shift_torch(eps_11, [-1, 0]) - circ_shift_torch(eps_11, [1, 0])) / 2
    d1_eps_12 = (circ_shift_torch(eps_12, [-1, 0]) - circ_shift_torch(eps_12, [1, 0])) / 2
    d1_eps_22 = (circ_shift_torch(eps_22, [-1, 0]) - circ_shift_torch(eps_22, [1, 0])) / 2
    d2_eps_11 = (circ_shift_torch(eps_11, [0, -1]) - circ_shift_torch(eps_11, [0, 1])) / 2
    d2_eps_12 = (circ_shift_torch(eps_12, [0, -1]) - circ_shift_torch(eps_12, [0, 1])) / 2
    d2_eps_22 = (circ_shift_torch(eps_22, [0, -1]) - circ_shift_torch(eps_22, [0, 1])) / 2

    Gamma1_11 = (eps11 * d1_eps_11 + eps12 * (2 * d1_eps_12 - d2_eps_11)) / 2
    Gamma1_12 = (eps11 * d2_eps_11 + eps12 * d1_eps_22) / 2
    Gamma1_22 = (eps11 * (2 * d2_eps_12 - d1_eps_22) + eps12 * d2_eps_22) / 2
    Gamma1 = torch.stack((Gamma1_11, Gamma1_12, Gamma1_22))
    Gamma2_11 = (eps12 * d1_eps_11 + eps22 * (2 * d1_eps_12 - d2_eps_11)) / 2
    Gamma2_12 = (eps12 * d2_eps_11 + eps22 * d1_eps_22) / 2
    Gamma2_22 = (eps12 * (2 * d2_eps_12 - d1_eps_22) + eps22 * d2_eps_22) / 2
    Gamma2 = torch.stack((Gamma2_11, Gamma2_12, Gamma2_22))
  # simple conformal metric
  elif metric == 'simple':
    scaling_field = torch.from_numpy(np.loadtxt(open("input/e_alpha_euclidean.csv", "rb"), delimiter=","))
    d1_alpha = (circ_shift_torch(scaling_field, [-1, 0]) - circ_shift_torch(scaling_field, [1, 0])) / 2
    d2_alpha = (circ_shift_torch(scaling_field, [0, -1]) - circ_shift_torch(scaling_field, [0, 1])) / 2

    Gamma1_11 = d1_alpha
    Gamma1_12 = d2_alpha
    Gamma1_22 = -d1_alpha
    Gamma1 = torch.stack((Gamma1_11, Gamma1_12, Gamma1_22))
    Gamma2_11 = -d2_alpha
    Gamma2_12 = d1_alpha
    Gamma2_22 = d2_alpha
    Gamma2 = torch.stack((Gamma2_11, Gamma2_12, Gamma2_22))
  else:
    Gamma1_11 = 0
    Gamma1_12 = 0
    Gamma1_22 = 0
    Gamma1 = torch.stack((Gamma1_11, Gamma1_12, Gamma1_22))
    Gamma2_11 = 0
    Gamma2_12 = 0
    Gamma2_22 = 0
    Gamma2 = torch.stack((Gamma2_11, Gamma2_12, Gamma2_22))

  gamma = start_coordinate
  gamma_dot = init_v
  gamma_ddot_x = get_gamma_ddot_at_point_torch(gamma[0], gamma[1], Gamma1, gamma_dot)
  gamma_ddot_y = get_gamma_ddot_at_point_torch(gamma[0], gamma[1], Gamma2, gamma_dot)

  prev_gamma = gamma
  prev_gamma_dot = gamma_dot
  prev_gamma_ddot = torch.stack((gamma_ddot_x, gamma_ddot_y))
  #print(eps_11[13,13], d1_eps_11[13,13])
  #print(Gamma1_11[13,13])
  #print(Gamma1[:,13,13], Gamma2[:,13,13])
  #print('tens_interp 0', tens_interp_torch(gamma[0,0], gamma[0,1], Gamma1))
  #print('tens_interp 1', tens_interp_torch(gamma[0,0], gamma[0,1], Gamma2))
  #print(gamma[0,0], gamma[0,1], gamma_dot[0], gamma_ddot[0,0], gamma_ddot[0,1])

  g_ddot_k1 = torch.zeros((2))
  g_ddot_k2 = torch.zeros((2))
  g_ddot_k3 = torch.zeros((2))
  g_ddot_k4 = torch.zeros((2))
  
  #for i in range(2, iter_num):
  for i in range(1,iter_num):

    # Do Fourth-Order Runge Kutta
    g_ddot_k1 = prev_gamma_ddot
    g_dot_k1 = prev_gamma_dot + delta_t * g_ddot_k1
    g_k1 = prev_gamma + delta_t * g_dot_k1

    g_ddot_k2_x = get_gamma_ddot_at_point_torch(g_k1[0], g_k1[1], Gamma1, g_dot_k1)
    g_ddot_k2_y = get_gamma_ddot_at_point_torch(g_k1[0], g_k1[1], Gamma2, g_dot_k1)
    g_ddot_k2 = torch.stack((g_ddot_k2_x, g_ddot_k2_y))

    g_dot_k2 = prev_gamma_dot + 0.5 * delta_t * g_ddot_k2
    g_k2 = prev_gamma + 0.5 * delta_t * g_dot_k2

    g_ddot_k3_x = get_gamma_ddot_at_point_torch(g_k2[0], g_k2[1], Gamma1, g_dot_k2)
    g_ddot_k3_y = get_gamma_ddot_at_point_torch(g_k2[0], g_k2[1], Gamma2, g_dot_k2)
    g_ddot_k3 = torch.stack((g_ddot_k3_x, g_ddot_k3_y))

    g_dot_k3 = prev_gamma_dot + 0.5 * delta_t * g_ddot_k3
    g_k3 = prev_gamma + 0.5 * delta_t * g_dot_k3

    g_ddot_k4_x = get_gamma_ddot_at_point_torch(g_k3[0], g_k3[1], Gamma1, g_dot_k3)
    g_ddot_k4_y = get_gamma_ddot_at_point_torch(g_k3[0], g_k3[1], Gamma2, g_dot_k3)
    g_ddot_k4 = torch.stack((g_ddot_k4_x, g_ddot_k4_y))
    
    g_dot_k4 = prev_gamma_dot + delta_t * g_ddot_k4
    g_k4 = prev_gamma + delta_t * g_dot_k4

    
    gamma_dot = prev_gamma_dot + delta_t/6.0*(g_ddot_k1 + 2.0*g_ddot_k2 + 2.0*g_ddot_k3 + g_ddot_k4)
    gamma = prev_gamma + delta_t/6.0*(g_dot_k1 + 2.0*g_dot_k2 + 2.0*g_dot_k3 + g_dot_k4)
    gamma_ddot_x = get_gamma_ddot_at_point_torch(gamma[0], gamma[1], Gamma1, gamma_dot)
    gamma_ddot_y = get_gamma_ddot_at_point_torch(gamma[0], gamma[1], Gamma2, gamma_dot)
    gamma_ddot = torch.stack((gamma_ddot_x, gamma_ddot_y))

    prev_gamma = gamma
    prev_gamma_dot = gamma_dot
    prev_gamma_ddot = gamma_ddot

    ## print(gamma_dot[i-1, :])
    ## print('gamma_ddot', i-2, ': ', gamma_ddot[i-2, :], 'gamma_dot', i-1, ': ', gamma_dot[i-1, :], 'gamma', i, ': ', gamma[i, :])
    if (math.ceil(gamma[0]) >= 0 and math.ceil(gamma[0]) < np.size(eps11, 0)
       and math.ceil(gamma[1]) >= 0 and math.ceil(gamma[1])  <np.size(eps11, 1)
       and mask_image[int(math.ceil(gamma[0])), int(math.ceil(gamma[1]))] > 0):
      geodesicpath_points_x[i-1] = gamma[0]
      geodesicpath_points_y[i-1] = gamma[1]
    else:
      # truncate and stop
      geodesicpath_points_x = geodesicpath_points_x[:i-1]
      geodesicpath_points_y = geodesicpath_points_y[:i-1]
      break

  final_point = gamma
    
  if filename:
    io.writePath(geodesicpath_points_x, geodesicpath_points_y, filename)

  return final_point, geodesicpath_points_x, geodesicpath_points_y, 
# end geodesicpath_torch

def geodesicpath_3d(tensor_field, mask_image, start_coordinate, initial_velocity, delta_t=0.15, metric='withoutscaling', iter_num=18000, filename = '', both_directions=False):
  # Compute 3d geodesic path
  # Assumes that mask_image is already a differentiable mask
  geodesicpath_points_x = np.zeros((iter_num))
  geodesicpath_points_y = np.zeros((iter_num))
  geodesicpath_points_z = np.zeros((iter_num))
  eps11 = tensor_field[0, :, :]
  eps12 = tensor_field[1, :, :]
  eps13 = tensor_field[2, :, :]
  eps22 = tensor_field[3, :, :]
  eps23 = tensor_field[4, :, :]
  eps33 = tensor_field[5, :, :]

  init_v = initial_velocity
  if initial_velocity is None:
    init_v = direction_3d(start_coordinate, tensor_field)

  if both_directions:
    back_x, back_y, back_z = geodesicpath_3d(tensor_field, mask_image, start_coordinate,
                                             [-init_v[0], -init_v[1], -init_v[2]], delta_t,
                                             metric, iter_num, filename, both_directions=False)

  print(f"Finding geodesic path from {start_coordinate} with initial velocity {init_v}")

  # adaptive riemannian metric without scaling
  if metric=='withoutscaling':

    # Compute inverse of g
    eps_11 = eps22 * eps33 - eps23 * eps23
    eps_12 = eps13 * eps23 - eps12 * eps33
    eps_13 = eps12 * eps23 - eps13 * eps22
    eps_22 = eps11 * eps33 - eps13 * eps13
    eps_23 = eps13 * eps12 - eps11 * eps23
    eps_33 = eps11 * eps22 - eps12 * eps12
    det_eps = eps11 * eps_11 + eps12 * eps_12 + eps13 * eps_13
    eps_11 = eps_11 / det_eps
    eps_12 = eps_12 / det_eps
    eps_13 = eps_13 / det_eps
    eps_22 = eps_22 / det_eps
    eps_23 = eps_23 / det_eps
    eps_33 = eps_33 / det_eps

    bdry_type, bdry_idx, bdry_map = maskops.determine_boundary_3d(mask_image, False)
    d1_eps_11, d2_eps_11, d3_eps_11 = diff.gradient_bdry_3d(eps_11, bdry_idx, bdry_map)
    d1_eps_12, d2_eps_12, d3_eps_12 = diff.gradient_bdry_3d(eps_12, bdry_idx, bdry_map)
    d1_eps_13, d2_eps_13, d3_eps_13 = diff.gradient_bdry_3d(eps_13, bdry_idx, bdry_map)
    d1_eps_22, d2_eps_22, d3_eps_22 = diff.gradient_bdry_3d(eps_22, bdry_idx, bdry_map)
    d1_eps_23, d2_eps_23, d3_eps_23 = diff.gradient_bdry_3d(eps_23, bdry_idx, bdry_map)
    d1_eps_33, d2_eps_33, d3_eps_33 = diff.gradient_bdry_3d(eps_33, bdry_idx, bdry_map)

    Gamma1_11 = (eps11 * d1_eps_11 + eps12 * (2 * d1_eps_12 - d2_eps_11) + eps13 * (2 * d1_eps_13 - d3_eps_11)) / 2
    Gamma1_12 = (eps11 * d2_eps_11 + eps12 * d1_eps_22 + eps13 * (d1_eps_23 + d2_eps_13 - d3_eps_11)) / 2
    Gamma1_13 = (eps11 * d3_eps_11 + eps12 * (d1_eps_23 + d3_eps_12 - d2_eps_13) + eps13 * (d1_eps_33)) / 2
    Gamma1_22 = (eps11 * (2 * d2_eps_12 - d1_eps_22) + eps12 * d2_eps_22 + eps13 * (2 * d2_eps_23 - d3_eps_22)) / 2
    Gamma1_23 = (eps11 * (d3_eps_12 + d2_eps_13 - d1_eps_23) + eps12 * d3_eps_22 + eps13 * (d2_eps_33)) / 2
    Gamma1_33 = (eps11 * (2 * d3_eps_13 - d1_eps_33) + eps12 * (2 * d3_eps_23 - d2_eps_33) + eps13 * d3_eps_33) / 2
    Gamma1 = np.stack((Gamma1_11, Gamma1_12, Gamma1_13, Gamma1_22, Gamma1_23, Gamma1_33))
    Gamma2_11 = (eps12 * d1_eps_11 + eps22 * (2 * d1_eps_12 - d2_eps_11) + eps23 * (2 * d1_eps_13 - d3_eps_11)) / 2
    Gamma2_12 = (eps12 * d2_eps_11 + eps22 * d1_eps_22 + eps23 * (d1_eps_23 + d2_eps_13 - d3_eps_11)) / 2
    Gamma2_13 = (eps12 * d3_eps_11 + eps22 * (d1_eps_23 + d3_eps_12 - d2_eps_13) + eps23 * (d1_eps_33)) / 2
    Gamma2_22 = (eps12 * (2 * d2_eps_12 - d1_eps_22) + eps22 * d2_eps_22 + eps23 * (2 * d2_eps_23 - d3_eps_22)) / 2
    Gamma2_23 = (eps12 * (d3_eps_12 + d2_eps_13 - d1_eps_23) + eps22 * d3_eps_22 + eps23 * (d2_eps_33)) / 2
    Gamma2_33 = (eps12 * (2 * d3_eps_13 - d1_eps_33) + eps22 * (2 * d3_eps_23 - d2_eps_33) + eps23 * d3_eps_33) / 2
    Gamma2 = np.stack((Gamma2_11, Gamma2_12, Gamma2_13, Gamma2_22, Gamma2_23, Gamma2_33))
    Gamma3_11 = (eps13 * d1_eps_11 + eps23 * (2 * d1_eps_12 - d2_eps_11) + eps33 * (2 * d1_eps_13 - d3_eps_11)) / 2
    Gamma3_12 = (eps13 * d2_eps_11 + eps23 * d1_eps_22 + eps33 * (d1_eps_23 + d2_eps_13 - d3_eps_11)) / 2
    Gamma3_13 = (eps13 * d3_eps_11 + eps23 * (d1_eps_23 + d3_eps_12 - d2_eps_13) + eps33 * (d1_eps_33)) / 2
    Gamma3_22 = (eps13 * (2 * d2_eps_12 - d1_eps_22) + eps23 * d2_eps_22 + eps33 * (2 * d2_eps_23 - d3_eps_22)) / 2
    Gamma3_23 = (eps13 * (d3_eps_12 + d2_eps_13 - d1_eps_23) + eps23 * d3_eps_22 + eps33 * (d2_eps_33)) / 2
    Gamma3_33 = (eps13 * (2 * d3_eps_13 - d1_eps_33) + eps23 * (2 * d3_eps_23 - d2_eps_33) + eps33 * d3_eps_33) / 2
    Gamma3 = np.stack((Gamma3_11, Gamma3_12, Gamma3_13, Gamma3_22, Gamma3_23, Gamma3_33))

  else:
    print(metric, 'not supported for 3d currently')

  gamma = np.zeros((iter_num,3))
  gamma_dot = np.zeros((iter_num,3))
  gamma_ddot = np.zeros((iter_num,3))
  gamma[0, :] = start_coordinate
  gamma_dot[0, :] = init_v
  gamma_ddot[0, 0] = get_gamma_ddot_at_point_3d(gamma[0,0], gamma[0,1], gamma[0,2], Gamma1, gamma_dot[0])
  gamma_ddot[0, 1] = get_gamma_ddot_at_point_3d(gamma[0,0], gamma[0,1], gamma[0,2], Gamma2, gamma_dot[0])
  gamma_ddot[0, 2] = get_gamma_ddot_at_point_3d(gamma[0,0], gamma[0,1], gamma[0,2], Gamma3, gamma_dot[0])
  #print(eps_11[13,13], d1_eps_11[13,13])
  #print(Gamma1_11[13,13])
  #print(Gamma1[:,13,13], Gamma2[:,13,13])
  #print('tens_interp 0', tens_interp(gamma[0,0], gamma[0,1], Gamma1))
  #print('tens_interp 1', tens_interp(gamma[0,0], gamma[0,1], Gamma2))
  #print(gamma[0,0], gamma[0,1], gamma_dot[0], gamma_ddot[0,0], gamma_ddot[0,1])
  #gamma[1,:] = gamma[0,:] + delta_t * gamma_dot[0,:]

  g_ddot_k1 = np.zeros((3))
  g_ddot_k2 = np.zeros((3))
  g_ddot_k3 = np.zeros((3))
  g_ddot_k4 = np.zeros((3))
  
  for i in range(1,iter_num):

    # Do Fourth-Order Runge Kutta
    g_ddot_k1 = gamma_ddot[i-1]
    g_dot_k1 = gamma_dot[i-1] + delta_t * g_ddot_k1
    g_k1 = gamma[i-1] + delta_t * g_dot_k1
    
    g_ddot_k2[0] = get_gamma_ddot_at_point_3d(g_k1[0], g_k1[1], g_k1[2], Gamma1, g_dot_k1)
    g_ddot_k2[1] = get_gamma_ddot_at_point_3d(g_k1[0], g_k1[1], g_k1[2], Gamma2, g_dot_k1)
    g_ddot_k2[2] = get_gamma_ddot_at_point_3d(g_k1[0], g_k1[1], g_k1[2], Gamma3, g_dot_k1)

    g_dot_k2 = gamma_dot[i-1] + 0.5 * delta_t * g_ddot_k2
    g_k2 = gamma[i-1] + 0.5 * delta_t * g_dot_k2

    g_ddot_k3[0] = get_gamma_ddot_at_point_3d(g_k2[0], g_k2[1], g_k2[2], Gamma1, g_dot_k2)
    g_ddot_k3[1] = get_gamma_ddot_at_point_3d(g_k2[0], g_k2[1], g_k2[2], Gamma2, g_dot_k2)
    g_ddot_k3[2] = get_gamma_ddot_at_point_3d(g_k2[0], g_k2[1], g_k2[2], Gamma3, g_dot_k2)

    g_dot_k3 = gamma_dot[i-1] + 0.5 * delta_t * g_ddot_k3
    g_k3 = gamma[i-1] + 0.5 * delta_t * g_dot_k3

    g_ddot_k4[0] = get_gamma_ddot_at_point_3d(g_k3[0], g_k3[1], g_k3[2], Gamma1, g_dot_k3)
    g_ddot_k4[1] = get_gamma_ddot_at_point_3d(g_k3[0], g_k3[1], g_k3[2], Gamma2, g_dot_k3)
    g_ddot_k4[2] = get_gamma_ddot_at_point_3d(g_k3[0], g_k3[1], g_k3[2], Gamma3, g_dot_k3)

    g_dot_k4 = gamma_dot[i-1] + delta_t * g_ddot_k4
    g_k4 = gamma[i-1] + delta_t * g_dot_k4

    
    gamma_dot[i, :] = gamma_dot[i-1, :] + delta_t/6.0*(g_ddot_k1 + 2.0*g_ddot_k2 + 2.0*g_ddot_k3 + g_ddot_k4)
    gamma[i, :] = gamma[i-1, :]+delta_t/6.0*(g_dot_k1 + 2.0*g_dot_k2 + 2.0*g_dot_k3 + g_dot_k4)
    gamma_ddot[i, 0] = get_gamma_ddot_at_point_3d(gamma[i,0], gamma[i,1], gamma[i,2], Gamma1, gamma_dot[i])
    gamma_ddot[i, 1] = get_gamma_ddot_at_point_3d(gamma[i,0], gamma[i,1], gamma[i,2], Gamma2, gamma_dot[i])
    gamma_ddot[i, 2] = get_gamma_ddot_at_point_3d(gamma[i,0], gamma[i,1], gamma[i,2], Gamma3, gamma_dot[i])
    
    ## print(gamma_dot[i-1, :])
    ## print('gamma_ddot', i-2, ': ', gamma_ddot[i-2, :], 'gamma_dot', i-1, ': ', gamma_dot[i-1, :], 'gamma', i, ': ', gamma[i, :])
    if (math.ceil(gamma[i, 0]) >= 0 and math.ceil(gamma[i, 0]) < np.size(eps11, 0)
       and math.ceil(gamma[i, 1]) >= 0 and math.ceil(gamma[i, 1])  <np.size(eps11, 1)
       and math.ceil(gamma[i, 2]) >= 0 and math.ceil(gamma[i, 2])  <np.size(eps11, 2)
        and (mask_image[int(math.ceil(gamma[i, 0])), int(math.ceil(gamma[i, 1])), int(math.ceil(gamma[i, 2]))] > 0)):
      geodesicpath_points_x[i-1] = gamma[i, 0]
      geodesicpath_points_y[i-1] = gamma[i, 1]
      geodesicpath_points_z[i-1] = gamma[i, 2]
    else:
      # truncate and stop
      geodesicpath_points_x = geodesicpath_points_x[:i-1]
      geodesicpath_points_y = geodesicpath_points_y[:i-1]
      geodesicpath_points_z = geodesicpath_points_z[:i-1]
      break

  if both_directions:
    geodesicpath_points_x = np.concatenate((geodesicpath_points_x[::-1], back_x))
    geodesicpath_points_y = np.concatenate((geodesicpath_points_y[::-1], back_y))
    geodesicpath_points_z = np.concatenate((geodesicpath_points_z[::-1], back_z))
    
  if filename:
    io.writePath3D(geodesicpath_points_x, geodesicpath_points_y, geodesicpath_points_z, filename)

  return geodesicpath_points_x, geodesicpath_points_y, geodesicpath_points_z
# end geodesicpath_3d

def geodesicpath_3d_torch(tensor_field, mask_image, start_coordinate, initial_velocity, delta_t=0.15, metric='withoutscaling', iter_num=18000, filename = '', both_directions=False):
  # Compute 3d geodesic path
  # Assumes that mask_image is already a differentiable mask
  geodesicpath_points_x = torch.zeros((iter_num),device=tensor_field.device)
  geodesicpath_points_y = torch.zeros((iter_num),device=tensor_field.device)
  geodesicpath_points_z = torch.zeros((iter_num),device=tensor_field.device)
  eps11 = tensor_field[0, :, :]
  eps12 = tensor_field[1, :, :]
  eps13 = tensor_field[2, :, :]
  eps22 = tensor_field[3, :, :]
  eps23 = tensor_field[4, :, :]
  eps33 = tensor_field[5, :, :]

  init_v = initial_velocity
  if initial_velocity is None:
    init_v = direction_3d_torch(start_coordinate, tensor_field)

  if both_directions:
    back_x, back_y, back_z = geodesicpath_3d_torch(tensor_field, mask_image, start_coordinate,
                                             -init_v, delta_t,
                                             metric, iter_num, filename, both_directions=False)

  print(f"Finding geodesic path from {start_coordinate} with initial velocity {init_v}")

  # adaptive riemannian metric without scaling
  if metric=='withoutscaling':

    # Compute inverse of g
    eps_11 = eps22 * eps33 - eps23 * eps23
    eps_12 = eps13 * eps23 - eps12 * eps33
    eps_13 = eps12 * eps23 - eps13 * eps22
    eps_22 = eps11 * eps33 - eps13 * eps13
    eps_23 = eps13 * eps12 - eps11 * eps23
    eps_33 = eps11 * eps22 - eps12 * eps12
    det_eps = eps11 * eps_11 + eps12 * eps_12 + eps13 * eps_13
    eps_11 = eps_11 / det_eps
    eps_12 = eps_12 / det_eps
    eps_13 = eps_13 / det_eps
    eps_22 = eps_22 / det_eps
    eps_23 = eps_23 / det_eps
    eps_33 = eps_33 / det_eps

    # TODO is circ_shift good enough for derivative calculations?
    bdry_type, bdry_idx, bdry_map = maskops.determine_boundary_3d(mask_image, False)
    d1_eps_11, d2_eps_11, d3_eps_11 = diff.gradient_bdry_3d(eps_11, bdry_idx, bdry_map)
    d1_eps_12, d2_eps_12, d3_eps_12 = diff.gradient_bdry_3d(eps_12, bdry_idx, bdry_map)
    d1_eps_13, d2_eps_13, d3_eps_13 = diff.gradient_bdry_3d(eps_13, bdry_idx, bdry_map)
    d1_eps_22, d2_eps_22, d3_eps_22 = diff.gradient_bdry_3d(eps_22, bdry_idx, bdry_map)
    d1_eps_23, d2_eps_23, d3_eps_23 = diff.gradient_bdry_3d(eps_23, bdry_idx, bdry_map)
    d1_eps_33, d2_eps_33, d3_eps_33 = diff.gradient_bdry_3d(eps_33, bdry_idx, bdry_map)

    Gamma1_11 = (eps11 * d1_eps_11 + eps12 * (2 * d1_eps_12 - d2_eps_11) + eps13 * (2 * d1_eps_13 - d3_eps_11)) / 2
    Gamma1_12 = (eps11 * d2_eps_11 + eps12 * d1_eps_22 + eps13 * (d1_eps_23 + d2_eps_13 - d3_eps_11)) / 2
    Gamma1_13 = (eps11 * d3_eps_11 + eps12 * (d1_eps_23 + d3_eps_12 - d2_eps_13) + eps13 * (d1_eps_33)) / 2
    Gamma1_22 = (eps11 * (2 * d2_eps_12 - d1_eps_22) + eps12 * d2_eps_22 + eps13 * (2 * d2_eps_23 - d3_eps_22)) / 2
    Gamma1_23 = (eps11 * (d3_eps_12 + d2_eps_13 - d1_eps_23) + eps12 * d3_eps_22 + eps13 * (d2_eps_33)) / 2
    Gamma1_33 = (eps11 * (2 * d3_eps_13 - d1_eps_33) + eps12 * (2 * d3_eps_23 - d2_eps_33) + eps13 * d3_eps_33) / 2
    Gamma1 = torch.stack((Gamma1_11, Gamma1_12, Gamma1_13, Gamma1_22, Gamma1_23, Gamma1_33))
    Gamma2_11 = (eps12 * d1_eps_11 + eps22 * (2 * d1_eps_12 - d2_eps_11) + eps23 * (2 * d1_eps_13 - d3_eps_11)) / 2
    Gamma2_12 = (eps12 * d2_eps_11 + eps22 * d1_eps_22 + eps23 * (d1_eps_23 + d2_eps_13 - d3_eps_11)) / 2
    Gamma2_13 = (eps12 * d3_eps_11 + eps22 * (d1_eps_23 + d3_eps_12 - d2_eps_13) + eps23 * (d1_eps_33)) / 2
    Gamma2_22 = (eps12 * (2 * d2_eps_12 - d1_eps_22) + eps22 * d2_eps_22 + eps23 * (2 * d2_eps_23 - d3_eps_22)) / 2
    Gamma2_23 = (eps12 * (d3_eps_12 + d2_eps_13 - d1_eps_23) + eps22 * d3_eps_22 + eps23 * (d2_eps_33)) / 2
    Gamma2_33 = (eps12 * (2 * d3_eps_13 - d1_eps_33) + eps22 * (2 * d3_eps_23 - d2_eps_33) + eps23 * d3_eps_33) / 2
    Gamma2 = torch.stack((Gamma2_11, Gamma2_12, Gamma2_13, Gamma2_22, Gamma2_23, Gamma2_33))
    Gamma3_11 = (eps13 * d1_eps_11 + eps23 * (2 * d1_eps_12 - d2_eps_11) + eps33 * (2 * d1_eps_13 - d3_eps_11)) / 2
    Gamma3_12 = (eps13 * d2_eps_11 + eps23 * d1_eps_22 + eps33 * (d1_eps_23 + d2_eps_13 - d3_eps_11)) / 2
    Gamma3_13 = (eps13 * d3_eps_11 + eps23 * (d1_eps_23 + d3_eps_12 - d2_eps_13) + eps33 * (d1_eps_33)) / 2
    Gamma3_22 = (eps13 * (2 * d2_eps_12 - d1_eps_22) + eps23 * d2_eps_22 + eps33 * (2 * d2_eps_23 - d3_eps_22)) / 2
    Gamma3_23 = (eps13 * (d3_eps_12 + d2_eps_13 - d1_eps_23) + eps23 * d3_eps_22 + eps33 * (d2_eps_33)) / 2
    Gamma3_33 = (eps13 * (2 * d3_eps_13 - d1_eps_33) + eps23 * (2 * d3_eps_23 - d2_eps_33) + eps33 * d3_eps_33) / 2
    Gamma3 = torch.stack((Gamma3_11, Gamma3_12, Gamma3_13, Gamma3_22, Gamma3_23, Gamma3_33))

  else:
    print(metric, 'not supported for 3d currently')

  gamma = torch.zeros((iter_num,3),device=tensor_field.device)
  gamma_dot = torch.zeros((iter_num,3),device=tensor_field.device)
  gamma_ddot = torch.zeros((iter_num,3),device=tensor_field.device)
  gamma[0, :] = start_coordinate
  gamma_dot[0, :] = init_v
  gamma_ddot[0, 0] = get_gamma_ddot_at_point_3d_torch(gamma[0,0], gamma[0,1], gamma[0,2], Gamma1, gamma_dot[0])
  gamma_ddot[0, 1] = get_gamma_ddot_at_point_3d_torch(gamma[0,0], gamma[0,1], gamma[0,2], Gamma2, gamma_dot[0])
  gamma_ddot[0, 2] = get_gamma_ddot_at_point_3d_torch(gamma[0,0], gamma[0,1], gamma[0,2], Gamma3, gamma_dot[0])
  #print(eps_11[13,13], d1_eps_11[13,13])
  #print(Gamma1_11[13,13])
  #print(Gamma1[:,13,13], Gamma2[:,13,13])
  #print('tens_interp 0', tens_interp(gamma[0,0], gamma[0,1], Gamma1))
  #print('tens_interp 1', tens_interp(gamma[0,0], gamma[0,1], Gamma2))
  #print(gamma[0,0], gamma[0,1], gamma_dot[0], gamma_ddot[0,0], gamma_ddot[0,1])
  #gamma[1,:] = gamma[0,:] + delta_t * gamma_dot[0,:]

  g_ddot_k1 = torch.zeros((3))
  g_ddot_k2 = torch.zeros((3))
  g_ddot_k3 = torch.zeros((3))
  g_ddot_k4 = torch.zeros((3))
  
  for i in range(1,iter_num):

    # Do Fourth-Order Runge Kutta
    g_ddot_k1 = gamma_ddot[i-1]
    g_dot_k1 = gamma_dot[i-1] + delta_t * g_ddot_k1
    g_k1 = gamma[i-1] + delta_t * g_dot_k1
    
    g_ddot_k2[0] = get_gamma_ddot_at_point_3d_torch(g_k1[0], g_k1[1], g_k1[2], Gamma1, g_dot_k1)
    g_ddot_k2[1] = get_gamma_ddot_at_point_3d_torch(g_k1[0], g_k1[1], g_k1[2], Gamma2, g_dot_k1)
    g_ddot_k2[2] = get_gamma_ddot_at_point_3d_torch(g_k1[0], g_k1[1], g_k1[2], Gamma3, g_dot_k1)

    g_dot_k2 = gamma_dot[i-1] + 0.5 * delta_t * g_ddot_k2
    g_k2 = gamma[i-1] + 0.5 * delta_t * g_dot_k2

    g_ddot_k3[0] = get_gamma_ddot_at_point_3d_torch(g_k2[0], g_k2[1], g_k2[2], Gamma1, g_dot_k2)
    g_ddot_k3[1] = get_gamma_ddot_at_point_3d_torch(g_k2[0], g_k2[1], g_k2[2], Gamma2, g_dot_k2)
    g_ddot_k3[2] = get_gamma_ddot_at_point_3d_torch(g_k2[0], g_k2[1], g_k2[2], Gamma3, g_dot_k2)

    g_dot_k3 = gamma_dot[i-1] + 0.5 * delta_t * g_ddot_k3
    g_k3 = gamma[i-1] + 0.5 * delta_t * g_dot_k3

    g_ddot_k4[0] = get_gamma_ddot_at_point_3d_torch(g_k3[0], g_k3[1], g_k3[2], Gamma1, g_dot_k3)
    g_ddot_k4[1] = get_gamma_ddot_at_point_3d_torch(g_k3[0], g_k3[1], g_k3[2], Gamma2, g_dot_k3)
    g_ddot_k4[2] = get_gamma_ddot_at_point_3d_torch(g_k3[0], g_k3[1], g_k3[2], Gamma3, g_dot_k3)

    g_dot_k4 = gamma_dot[i-1] + delta_t * g_ddot_k4
    g_k4 = gamma[i-1] + delta_t * g_dot_k4

    
    gamma_dot[i, :] = gamma_dot[i-1, :] + delta_t/6.0*(g_ddot_k1 + 2.0*g_ddot_k2 + 2.0*g_ddot_k3 + g_ddot_k4)
    gamma[i, :] = gamma[i-1, :]+delta_t/6.0*(g_dot_k1 + 2.0*g_dot_k2 + 2.0*g_dot_k3 + g_dot_k4)
    gamma_ddot[i, 0] = get_gamma_ddot_at_point_3d_torch(gamma[i,0], gamma[i,1], gamma[i,2], Gamma1, gamma_dot[i])
    gamma_ddot[i, 1] = get_gamma_ddot_at_point_3d_torch(gamma[i,0], gamma[i,1], gamma[i,2], Gamma2, gamma_dot[i])
    gamma_ddot[i, 2] = get_gamma_ddot_at_point_3d_torch(gamma[i,0], gamma[i,1], gamma[i,2], Gamma3, gamma_dot[i])
    
    ## print(gamma_dot[i-1, :])
    ## print('gamma_ddot', i-2, ': ', gamma_ddot[i-2, :], 'gamma_dot', i-1, ': ', gamma_dot[i-1, :], 'gamma', i, ': ', gamma[i, :])
    if (math.ceil(gamma[i, 0]) >= 0 and math.ceil(gamma[i, 0]) < np.size(eps11, 0)
       and math.ceil(gamma[i, 1]) >= 0 and math.ceil(gamma[i, 1])  <np.size(eps11, 1)
       and math.ceil(gamma[i, 2]) >= 0 and math.ceil(gamma[i, 2])  <np.size(eps11, 2)
        and (mask_image[int(math.ceil(gamma[i, 0])), int(math.ceil(gamma[i, 1])), int(math.ceil(gamma[i, 2]))] > 0)):
      geodesicpath_points_x[i-1] = gamma[i, 0]
      geodesicpath_points_y[i-1] = gamma[i, 1]
      geodesicpath_points_z[i-1] = gamma[i, 2]
    else:
      # truncate and stop
      geodesicpath_points_x = geodesicpath_points_x[:i-1]
      geodesicpath_points_y = geodesicpath_points_y[:i-1]
      geodesicpath_points_z = geodesicpath_points_z[:i-1]
      break

  if both_directions:
    # torch can't do negative indexing
    inv_idx = torch.arange(geodesicpath_points_x.size(0)-1, -1, -1).long()
    geodesicpath_points_x = torch.cat((geodesicpath_points_x[inv_idx], back_x))
    geodesicpath_points_y = torch.cat((geodesicpath_points_y[inv_idx], back_y))
    geodesicpath_points_z = torch.cat((geodesicpath_points_z[inv_idx], back_z))
    
  if filename:
    io.writePath3D(geodesicpath_points_x, geodesicpath_points_y, geodesicpath_points_z, filename)

  return geodesicpath_points_x, geodesicpath_points_y, geodesicpath_points_z
# end geodesicpath_3d_torch

#@profile # uncomment for kernprof to profile this method
#@jit(nopython=True)
def batch_geodesicpath_3d(tensor_field, mask_image, start_coordinates, initial_velocities, delta_t=0.15, iter_num=18000,
                          both_directions=False, Gamma1=None, Gamma2=None, Gamma3=None):
  # Compute 3d geodesic path of metric, where the input tensor_field is the inverse of the metric
  # Assumes that mask_image is already a differentiable mask
  # To speed up when calling multiple times, precompute gammas by calling
  #  Gamma1, Gamma2, Gamma3 = compute_gammas_3d(tensor_field, mask_image)
  # and pass in to method

  num_paths = len(start_coordinates)
  xsz,ysz,zsz = tensor_field.shape[1:4]
  
  geodesicpath_points_x = [np.zeros((iter_num-1)) for p in range(num_paths)]
  geodesicpath_points_y = [np.zeros((iter_num-1)) for p in range(num_paths)]
  geodesicpath_points_z = [np.zeros((iter_num-1)) for p in range(num_paths)]
  continue_path = np.ones((num_paths))
  if ( (Gamma1 is None) or (Gamma2 is None) or (Gamma3 is None) ):
    Gamma1, Gamma2, Gamma3 = compute_gammas_3d(tensor_field, mask_image)
  # end if Gamma1 is None...

  init_v = initial_velocities
  if initial_velocities is None:
    init_v = batch_direction_3d(np.array(start_coordinates), tensor_field)

  if both_directions:
    back_x, back_y, back_z = batch_geodesicpath_3d(tensor_field, mask_image, start_coordinates,
                                                   -init_v, delta_t, iter_num, both_directions=False,
                                                   Gamma1=Gamma1, Gamma2=Gamma2, Gamma3=Gamma3)

  #print(f"Finding geodesic path from {start_coordinate} with initial velocity {init_v}")

  gamma = np.zeros((num_paths,iter_num,3))
  gamma_dot = np.zeros((num_paths,iter_num,3))
  gamma_ddot = np.zeros((num_paths,iter_num,3))
  gamma[:, 0, :] = start_coordinates
  gamma_dot[:, 0, :] = init_v
  gamma_ddot[:, 0, 0] = batch_get_gamma_ddot_at_point_3d(gamma[:,0,0], gamma[:,0,1], gamma[:,0,2], Gamma1, gamma_dot[:,0])
  gamma_ddot[:, 0, 1] = batch_get_gamma_ddot_at_point_3d(gamma[:,0,0], gamma[:,0,1], gamma[:,0,2], Gamma2, gamma_dot[:,0])
  gamma_ddot[:, 0, 2] = batch_get_gamma_ddot_at_point_3d(gamma[:,0,0], gamma[:,0,1], gamma[:,0,2], Gamma3, gamma_dot[:,0])
  
  #print(eps_11[13,13], d1_eps_11[13,13])
  #print(Gamma1_11[13,13])
  #print(Gamma1[:,13,13], Gamma2[:,13,13])
  #print('tens_interp 0', tens_interp(gamma[0,0], gamma[0,1], Gamma1))
  #print('tens_interp 1', tens_interp(gamma[0,0], gamma[0,1], Gamma2))
  #print(gamma[0,0], gamma[0,1], gamma_dot[0], gamma_ddot[0,0], gamma_ddot[0,1])
  #gamma[1,:] = gamma[0,:] + delta_t * gamma_dot[0,:]

  g_ddot_k1 = np.zeros((num_paths,3))
  g_ddot_k2 = np.zeros((num_paths,3))
  g_ddot_k3 = np.zeros((num_paths,3))
  g_ddot_k4 = np.zeros((num_paths,3))
  
  for i in range(1,iter_num):

    # Do Fourth-Order Runge Kutta
    g_ddot_k1 = gamma_ddot[:,i-1]
    g_dot_k1 = gamma_dot[:,i-1] + delta_t * g_ddot_k1
    g_k1 = gamma[:,i-1] + delta_t * g_dot_k1
    
    g_ddot_k2[:,0] = batch_get_gamma_ddot_at_point_3d(g_k1[:,0], g_k1[:,1], g_k1[:,2], Gamma1, g_dot_k1)
    g_ddot_k2[:,1] = batch_get_gamma_ddot_at_point_3d(g_k1[:,0], g_k1[:,1], g_k1[:,2], Gamma2, g_dot_k1)
    g_ddot_k2[:,2] = batch_get_gamma_ddot_at_point_3d(g_k1[:,0], g_k1[:,1], g_k1[:,2], Gamma3, g_dot_k1)

    g_dot_k2 = gamma_dot[:,i-1] + 0.5 * delta_t * g_ddot_k2
    g_k2 = gamma[:,i-1] + 0.5 * delta_t * g_dot_k2

    g_ddot_k3[:,0] = batch_get_gamma_ddot_at_point_3d(g_k2[:,0], g_k2[:,1], g_k2[:,2], Gamma1, g_dot_k2)
    g_ddot_k3[:,1] = batch_get_gamma_ddot_at_point_3d(g_k2[:,0], g_k2[:,1], g_k2[:,2], Gamma2, g_dot_k2)
    g_ddot_k3[:,2] = batch_get_gamma_ddot_at_point_3d(g_k2[:,0], g_k2[:,1], g_k2[:,2], Gamma3, g_dot_k2)

    g_dot_k3 = gamma_dot[:,i-1] + 0.5 * delta_t * g_ddot_k3
    g_k3 = gamma[:,i-1] + 0.5 * delta_t * g_dot_k3

    g_ddot_k4[:,0] = batch_get_gamma_ddot_at_point_3d(g_k3[:,0], g_k3[:,1], g_k3[:,2], Gamma1, g_dot_k3)
    g_ddot_k4[:,1] = batch_get_gamma_ddot_at_point_3d(g_k3[:,0], g_k3[:,1], g_k3[:,2], Gamma2, g_dot_k3)
    g_ddot_k4[:,2] = batch_get_gamma_ddot_at_point_3d(g_k3[:,0], g_k3[:,1], g_k3[:,2], Gamma3, g_dot_k3)

    g_dot_k4 = gamma_dot[:,i-1] + delta_t * g_ddot_k4
    g_k4 = gamma[:,i-1] + delta_t * g_dot_k4

    
    gamma_dot[:, i, :] = gamma_dot[:, i-1, :] + delta_t/6.0*(g_ddot_k1 + 2.0*g_ddot_k2 + 2.0*g_ddot_k3 + g_ddot_k4)
    gamma[:, i, :] = gamma[:, i-1, :]+delta_t/6.0*(g_dot_k1 + 2.0*g_dot_k2 + 2.0*g_dot_k3 + g_dot_k4)
    gamma_ddot[:, i, 0] = batch_get_gamma_ddot_at_point_3d(gamma[:,i,0], gamma[:,i,1], gamma[:,i,2], Gamma1, gamma_dot[:,i])
    gamma_ddot[:, i, 1] = batch_get_gamma_ddot_at_point_3d(gamma[:,i,0], gamma[:,i,1], gamma[:,i,2], Gamma2, gamma_dot[:,i])
    gamma_ddot[:, i, 2] = batch_get_gamma_ddot_at_point_3d(gamma[:,i,0], gamma[:,i,1], gamma[:,i,2], Gamma3, gamma_dot[:,i])

    active_path=False
    for p in range(num_paths):
      if ( continue_path[p]
       and np.ceil(gamma[p, i, 0]) >= 0 and np.ceil(gamma[p, i, 0]) < xsz
       and np.ceil(gamma[p, i, 1]) >= 0 and np.ceil(gamma[p, i, 1]) < ysz
       and np.ceil(gamma[p, i, 2]) >= 0 and np.ceil(gamma[p, i, 2]) < zsz
       and (mask_image[int(np.ceil(gamma[p, i, 0])), int(np.ceil(gamma[p, i, 1])), int(np.ceil(gamma[p, i, 2]))] > 0)):
        active_path = True
        geodesicpath_points_x[p][i-1] = gamma[p, i, 0]
        geodesicpath_points_y[p][i-1] = gamma[p, i, 1]
        geodesicpath_points_z[p][i-1] = gamma[p, i, 2]
      else:
        # truncate and stop
        geodesicpath_points_x[p] = geodesicpath_points_x[p][:i-1]
        geodesicpath_points_y[p] = geodesicpath_points_y[p][:i-1]
        geodesicpath_points_z[p] = geodesicpath_points_z[p][:i-1]
        continue_path[p] = 0
    if not active_path:
      break
  # End for each time point i

  if both_directions:
    for p in range(num_paths):
      geodesicpath_points_x[p] = np.concatenate((geodesicpath_points_x[p][::-1], back_x[p]))
      geodesicpath_points_y[p] = np.concatenate((geodesicpath_points_y[p][::-1], back_y[p]))
      geodesicpath_points_z[p] = np.concatenate((geodesicpath_points_z[p][::-1], back_z[p]))
    
  return geodesicpath_points_x, geodesicpath_points_y, geodesicpath_points_z
# end batch_geodesicpath_3d

def batch_geodesicpath_3d_torch(tensor_field, mask_image, start_coordinates, initial_velocities, delta_t=0.15, iter_num=18000,
                                both_directions=False, Gamma1=None, Gamma2=None, Gamma3=None):
  # Compute 3d geodesic path of metric, where the input tensor_field is the inverse of the metric
  # Assumes that mask_image is already a differentiable mask
  num_paths = start_coordinates.shape[0]
  xsz,ysz,zsz = tensor_field.shape[1:4]
  
  geodesicpath_points_x = [torch.zeros((iter_num-1),device=tensor_field.device) for p in range(num_paths)]
  geodesicpath_points_y = [torch.zeros((iter_num-1),device=tensor_field.device) for p in range(num_paths)]
  geodesicpath_points_z = [torch.zeros((iter_num-1),device=tensor_field.device) for p in range(num_paths)]
  continue_path = np.ones((num_paths))
  if ( (Gamma1 is None)
    or (Gamma2 is None)
    or (Gamma3 is None) ):
    eps11 = tensor_field[0, :, :, :]
    eps12 = tensor_field[1, :, :, :]
    eps13 = tensor_field[2, :, :, :]
    eps22 = tensor_field[3, :, :, :]
    eps23 = tensor_field[4, :, :, :]
    eps33 = tensor_field[5, :, :, :]

    # Compute inverse of g
    eps_11 = eps22 * eps33 - eps23 * eps23
    eps_12 = eps13 * eps23 - eps12 * eps33
    eps_13 = eps12 * eps23 - eps13 * eps22
    eps_22 = eps11 * eps33 - eps13 * eps13
    eps_23 = eps13 * eps12 - eps11 * eps23
    eps_33 = eps11 * eps22 - eps12 * eps12
    det_eps = eps11 * eps_11 + eps12 * eps_12 + eps13 * eps_13
    eps_11 = eps_11 / det_eps
    eps_12 = eps_12 / det_eps
    eps_13 = eps_13 / det_eps
    eps_22 = eps_22 / det_eps
    eps_23 = eps_23 / det_eps
    eps_33 = eps_33 / det_eps

    bdry_type, bdry_idx, bdry_map = maskops.determine_boundary_3d(mask_image, False)
    d1_eps_11, d2_eps_11, d3_eps_11 = diff.gradient_bdry_3d(eps_11, bdry_idx, bdry_map)
    d1_eps_12, d2_eps_12, d3_eps_12 = diff.gradient_bdry_3d(eps_12, bdry_idx, bdry_map)
    d1_eps_13, d2_eps_13, d3_eps_13 = diff.gradient_bdry_3d(eps_13, bdry_idx, bdry_map)
    d1_eps_22, d2_eps_22, d3_eps_22 = diff.gradient_bdry_3d(eps_22, bdry_idx, bdry_map)
    d1_eps_23, d2_eps_23, d3_eps_23 = diff.gradient_bdry_3d(eps_23, bdry_idx, bdry_map)
    d1_eps_33, d2_eps_33, d3_eps_33 = diff.gradient_bdry_3d(eps_33, bdry_idx, bdry_map)

    Gamma1_11 = (eps11 * d1_eps_11 + eps12 * (2 * d1_eps_12 - d2_eps_11) + eps13 * (2 * d1_eps_13 - d3_eps_11)) / 2
    Gamma1_12 = (eps11 * d2_eps_11 + eps12 * d1_eps_22 + eps13 * (d1_eps_23 + d2_eps_13 - d3_eps_11)) / 2
    Gamma1_13 = (eps11 * d3_eps_11 + eps12 * (d1_eps_23 + d3_eps_12 - d2_eps_13) + eps13 * (d1_eps_33)) / 2
    Gamma1_22 = (eps11 * (2 * d2_eps_12 - d1_eps_22) + eps12 * d2_eps_22 + eps13 * (2 * d2_eps_23 - d3_eps_22)) / 2
    Gamma1_23 = (eps11 * (d3_eps_12 + d2_eps_13 - d1_eps_23) + eps12 * d3_eps_22 + eps13 * (d2_eps_33)) / 2
    Gamma1_33 = (eps11 * (2 * d3_eps_13 - d1_eps_33) + eps12 * (2 * d3_eps_23 - d2_eps_33) + eps13 * d3_eps_33) / 2
    Gamma1 = torch.stack((Gamma1_11, Gamma1_12, Gamma1_13, Gamma1_22, Gamma1_23, Gamma1_33))
    Gamma2_11 = (eps12 * d1_eps_11 + eps22 * (2 * d1_eps_12 - d2_eps_11) + eps23 * (2 * d1_eps_13 - d3_eps_11)) / 2
    Gamma2_12 = (eps12 * d2_eps_11 + eps22 * d1_eps_22 + eps23 * (d1_eps_23 + d2_eps_13 - d3_eps_11)) / 2
    Gamma2_13 = (eps12 * d3_eps_11 + eps22 * (d1_eps_23 + d3_eps_12 - d2_eps_13) + eps23 * (d1_eps_33)) / 2
    Gamma2_22 = (eps12 * (2 * d2_eps_12 - d1_eps_22) + eps22 * d2_eps_22 + eps23 * (2 * d2_eps_23 - d3_eps_22)) / 2
    Gamma2_23 = (eps12 * (d3_eps_12 + d2_eps_13 - d1_eps_23) + eps22 * d3_eps_22 + eps23 * (d2_eps_33)) / 2
    Gamma2_33 = (eps12 * (2 * d3_eps_13 - d1_eps_33) + eps22 * (2 * d3_eps_23 - d2_eps_33) + eps23 * d3_eps_33) / 2
    Gamma2 = torch.stack((Gamma2_11, Gamma2_12, Gamma2_13, Gamma2_22, Gamma2_23, Gamma2_33))
    Gamma3_11 = (eps13 * d1_eps_11 + eps23 * (2 * d1_eps_12 - d2_eps_11) + eps33 * (2 * d1_eps_13 - d3_eps_11)) / 2
    Gamma3_12 = (eps13 * d2_eps_11 + eps23 * d1_eps_22 + eps33 * (d1_eps_23 + d2_eps_13 - d3_eps_11)) / 2
    Gamma3_13 = (eps13 * d3_eps_11 + eps23 * (d1_eps_23 + d3_eps_12 - d2_eps_13) + eps33 * (d1_eps_33)) / 2
    Gamma3_22 = (eps13 * (2 * d2_eps_12 - d1_eps_22) + eps23 * d2_eps_22 + eps33 * (2 * d2_eps_23 - d3_eps_22)) / 2
    Gamma3_23 = (eps13 * (d3_eps_12 + d2_eps_13 - d1_eps_23) + eps23 * d3_eps_22 + eps33 * (d2_eps_33)) / 2
    Gamma3_33 = (eps13 * (2 * d3_eps_13 - d1_eps_33) + eps23 * (2 * d3_eps_23 - d2_eps_33) + eps33 * d3_eps_33) / 2
    Gamma3 = torch.stack((Gamma3_11, Gamma3_12, Gamma3_13, Gamma3_22, Gamma3_23, Gamma3_33))
  # end if Gamma1 is None...

  init_v = initial_velocities
  if initial_velocities is None:
    init_v = batch_direction_3d_torch(start_coordinates, tensor_field)

  if both_directions:
    back_x, back_y, back_z = batch_geodesicpath_3d_torch(tensor_field, mask_image, start_coordinates,
                                                         -init_v, delta_t, iter_num, both_directions=False,
                                                         Gamma1=Gamma1, Gamma2=Gamma2, Gamma3=Gamma3)

  gamma = torch.zeros((num_paths,iter_num,3),device=tensor_field.device)
  gamma_dot = torch.zeros((num_paths,iter_num,3),device=tensor_field.device)
  gamma_ddot = torch.zeros((num_paths,iter_num,3),device=tensor_field.device)
  gamma[:, 0, :] = start_coordinates.double()
  gamma_dot[:, 0, :] = init_v
  gamma_ddot[:, 0, 0] = batch_get_gamma_ddot_at_point_3d_torch(gamma[:,0,0], gamma[:,0,1], gamma[:,0,2], Gamma1, gamma_dot[:,0])
  gamma_ddot[:, 0, 1] = batch_get_gamma_ddot_at_point_3d_torch(gamma[:,0,0], gamma[:,0,1], gamma[:,0,2], Gamma2, gamma_dot[:,0])
  gamma_ddot[:, 0, 2] = batch_get_gamma_ddot_at_point_3d_torch(gamma[:,0,0], gamma[:,0,1], gamma[:,0,2], Gamma3, gamma_dot[:,0])
  #print(eps_11[13,13], d1_eps_11[13,13])
  #print(Gamma1_11[13,13])
  #print(Gamma1[:,13,13], Gamma2[:,13,13])
  #print('tens_interp 0', tens_interp(gamma[0,0], gamma[0,1], Gamma1))
  #print('tens_interp 1', tens_interp(gamma[0,0], gamma[0,1], Gamma2))
  #print(gamma[0,0], gamma[0,1], gamma_dot[0], gamma_ddot[0,0], gamma_ddot[0,1])
  #gamma[1,:] = gamma[0,:] + delta_t * gamma_dot[0,:]

  g_ddot_k1 = torch.zeros((num_paths,3))
  g_ddot_k2 = torch.zeros((num_paths,3))
  g_ddot_k3 = torch.zeros((num_paths,3))
  g_ddot_k4 = torch.zeros((num_paths,3))
  
  for i in range(1,iter_num):

    # Do Fourth-Order Runge Kutta
    g_ddot_k1 = gamma_ddot[:,i-1]
    g_dot_k1 = gamma_dot[:,i-1] + delta_t * g_ddot_k1
    g_k1 = gamma[:,i-1] + delta_t * g_dot_k1
    
    g_ddot_k2[:,0] = batch_get_gamma_ddot_at_point_3d_torch(g_k1[:,0], g_k1[:,1], g_k1[:,2], Gamma1, g_dot_k1)
    g_ddot_k2[:,1] = batch_get_gamma_ddot_at_point_3d_torch(g_k1[:,0], g_k1[:,1], g_k1[:,2], Gamma2, g_dot_k1)
    g_ddot_k2[:,2] = batch_get_gamma_ddot_at_point_3d_torch(g_k1[:,0], g_k1[:,1], g_k1[:,2], Gamma3, g_dot_k1)

    g_dot_k2 = gamma_dot[:,i-1] + 0.5 * delta_t * g_ddot_k2
    g_k2 = gamma[:,i-1] + 0.5 * delta_t * g_dot_k2

    g_ddot_k3[:,0] = batch_get_gamma_ddot_at_point_3d_torch(g_k2[:,0], g_k2[:,1], g_k2[:,2], Gamma1, g_dot_k2)
    g_ddot_k3[:,1] = batch_get_gamma_ddot_at_point_3d_torch(g_k2[:,0], g_k2[:,1], g_k2[:,2], Gamma2, g_dot_k2)
    g_ddot_k3[:,2] = batch_get_gamma_ddot_at_point_3d_torch(g_k2[:,0], g_k2[:,1], g_k2[:,2], Gamma3, g_dot_k2)

    g_dot_k3 = gamma_dot[:,i-1] + 0.5 * delta_t * g_ddot_k3
    g_k3 = gamma[:,i-1] + 0.5 * delta_t * g_dot_k3

    g_ddot_k4[:,0] = batch_get_gamma_ddot_at_point_3d_torch(g_k3[:,0], g_k3[:,1], g_k3[:,2], Gamma1, g_dot_k3)
    g_ddot_k4[:,1] = batch_get_gamma_ddot_at_point_3d_torch(g_k3[:,0], g_k3[:,1], g_k3[:,2], Gamma2, g_dot_k3)
    g_ddot_k4[:,2] = batch_get_gamma_ddot_at_point_3d_torch(g_k3[:,0], g_k3[:,1], g_k3[:,2], Gamma3, g_dot_k3)

    g_dot_k4 = gamma_dot[:,i-1] + delta_t * g_ddot_k4
    g_k4 = gamma[:,i-1] + delta_t * g_dot_k4

    
    gamma_dot[:, i, :] = gamma_dot[:, i-1, :] + delta_t/6.0*(g_ddot_k1 + 2.0*g_ddot_k2 + 2.0*g_ddot_k3 + g_ddot_k4)
    gamma[:, i, :] = gamma[:, i-1, :]+delta_t/6.0*(g_dot_k1 + 2.0*g_dot_k2 + 2.0*g_dot_k3 + g_dot_k4)
    gamma_ddot[:, i, 0] = batch_get_gamma_ddot_at_point_3d_torch(gamma[:,i,0], gamma[:,i,1], gamma[:,i,2], Gamma1, gamma_dot[:,i])
    gamma_ddot[:, i, 1] = batch_get_gamma_ddot_at_point_3d_torch(gamma[:,i,0], gamma[:,i,1], gamma[:,i,2], Gamma2, gamma_dot[:,i])
    gamma_ddot[:, i, 2] = batch_get_gamma_ddot_at_point_3d_torch(gamma[:,i,0], gamma[:,i,1], gamma[:,i,2], Gamma3, gamma_dot[:,i])
    
    ## print(gamma_dot[i-1, :])
    ## print('gamma_ddot', i-2, ': ', gamma_ddot[i-2, :], 'gamma_dot', i-1, ': ', gamma_dot[i-1, :], 'gamma', i, ': ', gamma[i, :])
    active_path=False
    for p in range(num_paths):
      if ( continue_path[p]
       and torch.ceil(gamma[p, i, 0]) >= 0 and torch.ceil(gamma[p, i, 0]) < xsz
       and torch.ceil(gamma[p, i, 1]) >= 0 and torch.ceil(gamma[p, i, 1]) < ysz
       and torch.ceil(gamma[p, i, 2]) >= 0 and torch.ceil(gamma[p, i, 2]) < zsz
       and (mask_image[int(torch.ceil(gamma[p, i, 0])), int(torch.ceil(gamma[p, i, 1])), int(torch.ceil(gamma[p, i, 2]))] > 0)):
        active_path = True
        geodesicpath_points_x[p][i-1] = gamma[p, i, 0]
        geodesicpath_points_y[p][i-1] = gamma[p, i, 1]
        geodesicpath_points_z[p][i-1] = gamma[p, i, 2]
      else:
        # truncate and stop
        geodesicpath_points_x[p] = geodesicpath_points_x[p][:i-1]
        geodesicpath_points_y[p] = geodesicpath_points_y[p][:i-1]
        geodesicpath_points_z[p] = geodesicpath_points_z[p][:i-1]
        continue_path[p] = 0
    if not active_path:
      break
  # End for each time point i

  if both_directions:
    # torch can't do negative indexing
    for p in range(num_paths):
      inv_idx = torch.arange(geodesicpath_points_x[p].size(0)-1, -1, -1).long()
      geodesicpath_points_x[p] = torch.cat((geodesicpath_points_x[p][inv_idx], back_x[p]))
      geodesicpath_points_y[p] = torch.cat((geodesicpath_points_y[p][inv_idx], back_y[p]))
      geodesicpath_points_z[p] = torch.cat((geodesicpath_points_z[p][inv_idx], back_z[p]))
    
  return geodesicpath_points_x, geodesicpath_points_y, geodesicpath_points_z
# end batch_geodesicpath_3d_torch


def geodesic_between_points_torch(tensor_field, mask_image, start_coordinate, end_coordinate, init_velocity=None, step_size=0.0001, num_iters=18000, filename = ''):
  # assumes tensor_field and mask_image are np arrays, converts to torch here
  torch_field = torch.from_numpy(tensor_field)
  mask = torch.from_numpy(mask_image)
  start_coords = torch.tensor(start_coordinate)
  end_coords = torch.tensor(end_coordinate)

  # TODO Is there a way to use pytorch batching to compute many geodesics at once?
  energy = torch.zeros((num_iters))
  init_v = torch.zeros((num_iters, 2), requires_grad=True)
  
  if init_velocity is None:
    init_v[0] = direction_torch(start_coordinate, tensor_field)
  else:
    init_v[0] = torch.tensor(init_velocity)


  all_points_x = []
  all_points_y = []
  
  for it in range(0,num_iters-1):
    end_point, points_x, points_y = geodesicpath_torch(torch_field, mask, start_coords, init_v[it], delta_t=0.15, iter_num=18000, filename = '')
    all_points_x.append(points_x)
    all_points_y.append(points_y)
    energy[it] = ((end_point[0] - end_coords[0])**2 + (end_point[1] - end_coords[1])**2)
    energy.backward()
    init_v[it+1] = init_v[it] - step_size * init_v.grad

  return(all_points_x, all_points_y, init_v, energy)
