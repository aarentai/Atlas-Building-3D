from lazy_imports import np
import math
from util.tensors import tens_interp, tens_interp_3d, eigen_vec, eigen_vec_3d, direction_3d
from data import io


def get_eigenvec_at_point(x, y, tensor_field, prev_angle):
  # return first and second component of eigenvector at a point, and associated angle
  tens = tens_interp(x, y, tensor_field)

  # important!!!!!!!!!!!!
  '''
  When using the eigenvector calculate by myself:
  Because when the principal eigenvector is almost horizontal, say at the top of the annulus,
  the eigenvector becomes extremely small, like [0.009986493070520448 1.9950060037743356e-05]
  so we have to adjust it to [1 0] manually.
  When the tensor is very vertical or horizontal, it's typically [6 0; 0 1] or [1 0; 0 6]
  '''
  # if abs(tens[0,1]) < 0.01 and abs(tens[1,0]) < 0.01:
  #   print("small b and c")
  #   u, v = eigen_vec(tens)
  #   print(u, v)
  #   # print(x, y)
  #   u = 1
  #   v = 0
  # else:
  #   u, v = eigen_vec(tens)
  u, v = eigen_vec(tens)

  # important too!!!!!!!!
  angle1 = math.atan2(v, u)
  angle2 = math.atan2(-v, -u)
  if abs(angle1 - prev_angle) < abs(angle2 - prev_angle):
    # keep the sign of eigenvector
    new_angle = angle1
  else:
    u = -u
    v = -v
    new_angle = angle2
  return(u, v, new_angle)

def eulerpath(tensor_field, mask_image, start_coordinate, initial_velocity=None, delta_t=0.25, metric='withoutscaling', iter_num=700, filename = '', both_directions=False):

  eps11 = tensor_field[0, :, :]
  eps12 = tensor_field[1, :, :]
  eps22 = tensor_field[2, :, :]
  # print(tensor_field[:, 50, 83])
  # print(tensor_field[:, 18, 50])

  if metric=='withscaling':
    scaling_field = np.loadtxt(open("input/e_alpha_kris.csv", "rb"), delimiter=",")
    eps11 = eps11 * scaling_field
    eps12 = eps12 * scaling_field
    eps22 = eps22 * scaling_field

  # initialing
  tens = np.zeros((2, 2), dtype=float)
  tens[0, 0] = eps11[int(start_coordinate[0]), int(start_coordinate[1])]
  tens[0, 1] = eps12[int(start_coordinate[0]), int(start_coordinate[1])]
  tens[1, 0] = eps12[int(start_coordinate[0]), int(start_coordinate[1])]
  tens[1, 1] = eps22[int(start_coordinate[0]), int(start_coordinate[1])]

  # calculating first eigenvector
  (x, y) = start_coordinate

  u, v = eigen_vec(tens)
  if initial_velocity is None:
    init_velocity = direction(start_coordinate, tensor_field)
    u = init_velocity[0]
    v = init_velocity[1]
  else:
    u = initial_velocity[0]
    v = initial_velocity[1]

  if both_directions:
    back_x, back_y = eulerpath(tensor_field, mask_image, start_coordinate, [-u, -v], delta_t, metric, iter_num, filename, both_directions=False)
  print("Euler starting eigenvector:", [u,v])
  prev_angle = math.atan2(v, u)

  points_x = []
  points_y = []

  # calculating following eigenvectors
  for i in range(iter_num):
    '''
    The reason why x should -v*delta_t instead of +v*delta_t is that: in calculation, we regard upper left
    namely the cell[0,0] as the origin. However, the vector field derived by tensor field regards down left 
    as the origin, namely the cell[size[0]-1,0], only by this can the the value in vector field make sense.
    '''
    # original
    #x = x + u * delta_t
    #y = y + v * delta_t
    ## points_x.append(x)
    ## points_y.append(y)
    #if (math.ceil(x) >= 0 and math.ceil(x) < np.size(eps11, 0)
    #    and math.ceil(y) >= 0 and math.ceil(x) < np.size(eps11, 1)
    #    and mask_image[int(math.ceil(x)), int(math.ceil(y))] > 0):
    #  points_x.append(x)
    #  points_y.append(y)
    #else:
    #  break

    #(u, v, new_angle) = get_eigenvec_at_point(x, y, tensor_field, prev_angle)

    # new 
    #(u,v) = f(tn,yn)
    uk1 = u
    vk1 = v
    xk1 = x + uk1 * delta_t
    yk1 = y + vk1 * delta_t

    (uk2, vk2, new_angle1) = get_eigenvec_at_point(xk1, yk1, tensor_field, prev_angle)
    xk2 = x + 0.5 * uk2 * delta_t
    yk2 = y + 0.5 * vk2 * delta_t

    (uk3, vk3, new_angle2) = get_eigenvec_at_point(xk2, yk2, tensor_field, prev_angle)
    xk3 = x + 0.5 * uk3 * delta_t
    yk3 = y + 0.5 * vk3 * delta_t

    (uk4, vk4, new_angle3) = get_eigenvec_at_point(xk3, yk3, tensor_field, prev_angle)
    xk4 = x + uk4 * delta_t
    yk4 = y + vk4 * delta_t

    x = x + (uk1 + 2.0 * uk2 + 2.0 * uk3 + uk4) * delta_t / 6.0
    y = y + (vk1 + 2.0 * vk2 + 2.0 * vk3 + vk4) * delta_t / 6.0

    if (math.ceil(x) >= 0 and math.ceil(x) < np.size(eps11, 0)
        and math.ceil(y) >= 0 and math.ceil(y) < np.size(eps11, 1)
        and mask_image[int(math.ceil(x)), int(math.ceil(y))] > 0):
      points_x.append(x)
      points_y.append(y)
    else:
      break

    (u, v, prev_angle) = get_eigenvec_at_point(x, y, tensor_field, prev_angle)

  if both_directions:
    points_x = points_x + back_x
    points_y = points_y + back_y
    
  if filename:
    io.writePath(points_x, points_y, filename)

  return points_x, points_y

def get_eigenvec_at_point_3d(x, y, z, tensor_field, prev_angle):
  # return first and second component of eigenvector at a point, and associated angle
  tens = tens_interp_3d(x, y, z, tensor_field)

  # important!!!!!!!!!!!!
  '''
  When using the eigenvector calculate by myself:
  Because when the principal eigenvector is almost horizontal, say at the top of the annulus,
  the eigenvector becomes extremely small, like [0.009986493070520448 1.9950060037743356e-05]
  so we have to adjust it to [1 0] manually.
  When the tensor is very vertical or horizontal, it's typically [6 0; 0 1] or [1 0; 0 6]
  '''
  # if abs(tens[0,1]) < 0.01 and abs(tens[1,0]) < 0.01:
  #   print("small b and c")
  #   u, v = eigen_vec(tens)
  #   print(u, v)
  #   # print(x, y)
  #   u = 1
  #   v = 0
  # else:
  #   u, v = eigen_vec(tens)
  u, v, w = eigen_vec_3d(tens)

  # important too!!!!!!!!
  angle1 = math.atan2(v, u)
  angle2 = math.atan2(-v, -u)
  if abs(angle1 - prev_angle) < abs(angle2 - prev_angle):
    # keep the sign of eigenvector
    new_angle = angle1
  else:
    u = -u
    v = -v
    w = -w
    new_angle = angle2
  return(u, v, w, new_angle)

def eulerpath_3d(tensor_field, mask_image, start_coordinate, initial_velocity=None, delta_t=0.25, metric='withoutscaling', iter_num=700, filename = '', both_directions=False):

  eps11 = tensor_field[0, :, :]
  eps12 = tensor_field[1, :, :]
  eps13 = tensor_field[2, :, :]
  eps22 = tensor_field[3, :, :]
  eps23 = tensor_field[4, :, :]
  eps33 = tensor_field[5, :, :]
  # print(tensor_field[:, 50, 83])
  # print(tensor_field[:, 18, 50])

  if metric=='withscaling':
    scaling_field = np.loadtxt(open("input/e_alpha_kris.csv", "rb"), delimiter=",")
    eps11 = eps11 * scaling_field
    eps12 = eps12 * scaling_field
    eps13 = eps13 * scaling_field
    eps22 = eps22 * scaling_field
    eps23 = eps23 * scaling_field
    eps33 = eps33 * scaling_field

  # initialing
  tens = np.zeros((3, 3), dtype=float)
  tens[0, 0] = eps11[int(start_coordinate[0]), int(start_coordinate[1]), int(start_coordinate[2])]
  tens[0, 1] = eps12[int(start_coordinate[0]), int(start_coordinate[1]), int(start_coordinate[2])]
  tens[1, 0] = tens[0,1]
  tens[0, 2] = eps23[int(start_coordinate[0]), int(start_coordinate[1]), int(start_coordinate[2])]
  tens[2, 0] = tens[0,2]
  tens[1, 1] = eps22[int(start_coordinate[0]), int(start_coordinate[1]), int(start_coordinate[2])]
  tens[1, 2] = eps23[int(start_coordinate[0]), int(start_coordinate[1]), int(start_coordinate[2])]
  tens[2, 1] = tens[1,2]
  tens[2, 2] = eps33[int(start_coordinate[0]), int(start_coordinate[1]), int(start_coordinate[2])]

  # calculating first eigenvector
  (x, y, z) = start_coordinate

  u, v, w = eigen_vec_3d(tens)
  if initial_velocity is None:
    init_velocity = direction_3d(start_coordinate, tensor_field)
    u = init_velocity[0]
    v = init_velocity[1]
    w = init_velocity[2]
  else:
    u = initial_velocity[0]
    v = initial_velocity[1]
    w = initial_velocity[2]

  if both_directions:
    back_x, back_y, back_z = eulerpath_3d(tensor_field, mask_image, start_coordinate, [-u, -v, -w], delta_t, metric, iter_num, filename, both_directions=False)
  print("Euler starting eigenvector:", [u,v,w])
  prev_angle = math.atan2(v, u)

  points_x = []
  points_y = []
  points_z = []

  # calculating following eigenvectors
  for i in range(iter_num):
    '''
    The reason why x should -v*delta_t instead of +v*delta_t is that: in calculation, we regard upper left
    namely the cell[0,0] as the origin. However, the vector field derived by tensor field regards down left 
    as the origin, namely the cell[size[0]-1,0], only by this can the the value in vector field make sense.
    '''
    # original
    #x = x + u * delta_t
    #y = y + v * delta_t
    ## points_x.append(x)
    ## points_y.append(y)
    #if (math.ceil(x) >= 0 and math.ceil(x) < np.size(eps11, 0)
    #    and math.ceil(y) >= 0 and math.ceil(x) < np.size(eps11, 1)
    #    and mask_image[int(math.ceil(x)), int(math.ceil(y))] > 0):
    #  points_x.append(x)
    #  points_y.append(y)
    #else:
    #  break

    #(u, v, new_angle) = get_eigenvec_at_point(x, y, tensor_field, prev_angle)

    # new 
    #(u,v) = f(tn,yn)
    uk1 = u
    vk1 = v
    wk1 = w
    xk1 = x + uk1 * delta_t
    yk1 = y + vk1 * delta_t
    zk1 = z + wk1 * delta_t

    (uk2, vk2, wk2, new_angle1) = get_eigenvec_at_point_3d(xk1, yk1, zk1, tensor_field, prev_angle)
    xk2 = x + 0.5 * uk2 * delta_t
    yk2 = y + 0.5 * vk2 * delta_t
    zk2 = z + 0.5 * wk2 * delta_t

    (uk3, vk3, wk3, new_angle2) = get_eigenvec_at_point_3d(xk2, yk2, zk2, tensor_field, prev_angle)
    xk3 = x + 0.5 * uk3 * delta_t
    yk3 = y + 0.5 * vk3 * delta_t
    zk3 = z + 0.5 * wk3 * delta_t

    (uk4, vk4, wk4, new_angle3) = get_eigenvec_at_point_3d(xk3, yk3, zk3, tensor_field, prev_angle)
    xk4 = x + uk4 * delta_t
    yk4 = y + vk4 * delta_t
    zk4 = z + wk4 * delta_t

    x = x + (uk1 + 2.0 * uk2 + 2.0 * uk3 + uk4) * delta_t / 6.0
    y = y + (vk1 + 2.0 * vk2 + 2.0 * vk3 + vk4) * delta_t / 6.0
    z = z + (wk1 + 2.0 * wk2 + 2.0 * wk3 + wk4) * delta_t / 6.0

    if (math.ceil(x) >= 0 and math.ceil(x) < np.size(eps11, 0)
        and math.ceil(y) >= 0 and math.ceil(y) < np.size(eps11, 1)
        and math.ceil(z) >= 0 and math.ceil(z) < np.size(eps11, 2)
        and mask_image[int(math.ceil(x)), int(math.ceil(y)), int(math.ceil(z))] > 0):
      points_x.append(x)
      points_y.append(y)
      points_z.append(z)
    else:
      break

    (u, v, w, prev_angle) = get_eigenvec_at_point_3d(x, y, z, tensor_field, prev_angle)

  if both_directions:
    points_x = points_x + back_x
    points_y = points_y + back_y
    points_z = points_z + back_z
    
  if filename:
    io.writePath3D(points_x, points_y, points_z, filename)

  return points_x, points_y, points_z
