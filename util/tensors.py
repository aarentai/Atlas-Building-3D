import math
from lazy_imports import np
from lazy_imports import torch

def direction(coordinate, tensor_field):
  tens = tens_interp(coordinate[0], coordinate[1], tensor_field)
  u, v = eigen_vec(tens)
  return (np.array([u, v]))

def direction_torch(coordinate, tensor_field):
  tens = tens_interp_torch(coordinate[0], coordinate[1], tensor_field)
  u, v = eigen_vec_torch(tens)
  return (torch.tensor([u, v]))

def direction_3d(coordinate, tensor_field):
  tens = tens_interp_3d(coordinate[0], coordinate[1], coordinate[2], tensor_field)
  u, v, w = eigen_vec_3d(tens)
  return (np.array([u, v, w]))


def eigen_vec(tens):
  evals, evecs = np.linalg.eigh(tens)
  [u, v] = evecs[:, evals.argmax()]
  return (u, v)

def eigen_vec_torch(tens):
  evals, evecs =torch.symeig(tens, eigenvectors=True)
  [u, v] = evecs[:, evals.argmax()]
  return (u, v)

def eigen_vec_3d(tens):
  evals, evecs = np.linalg.eigh(tens)
  [u, v, w] = evecs[:, evals.argmax()]
  return (u, v, w)

def circ_shift(I, shift):
  I = np.roll(I, shift[0], axis=0)
  I = np.roll(I, shift[1], axis=1)
  return (I)

def circ_shift_torch(I, shift):
  I = torch.roll(I, shift[0], dims=0)
  I = torch.roll(I, shift[1], dims=1)
  return (I)

def circ_shift_3d(I, shift):
  I = np.roll(I, shift[0], axis=0)
  I = np.roll(I, shift[1], axis=1)
  I = np.roll(I, shift[2], axis=2)
  return (I)

def tens_interp(x, y, tensor_field):
  tens = np.zeros((2, 2))
  eps11 = tensor_field[0, :, :]
  eps12 = tensor_field[1, :, :]
  eps22 = tensor_field[2, :, :]
  if (math.floor(x) < 0) or (math.ceil(x) >= eps11.shape[0]) or (math.floor(y) < 0) or (math.ceil(y) >= eps11.shape[1]):
     # data is out of bounds, return identity tensor
     tens[0,0] = 1
     tens[1,1] = 1
     return(tens)
  if x == math.floor(x) and y == math.floor(y):
    tens[0, 0] = eps11[int(x), int(y)]
    tens[0, 1] = eps12[int(x), int(y)]
    tens[1, 0] = eps12[int(x), int(y)]
    tens[1, 1] = eps22[int(x), int(y)]
  elif x == math.floor(x) and y != math.floor(y):
    tens[0, 0] = abs(y - math.floor(y)) * eps11[int(x), math.ceil(y)] + \
           abs(y - math.ceil(y)) * eps11[int(x), math.floor(y)]
    tens[0, 1] = abs(y - math.floor(y)) * eps12[int(x), math.ceil(y)] + \
           abs(y - math.ceil(y)) * eps12[int(x), math.floor(y)]
    tens[1, 0] = abs(y - math.floor(y)) * eps12[int(x), math.ceil(y)] + \
           abs(y - math.ceil(y)) * eps12[int(x), math.floor(y)]
    tens[1, 1] = abs(y - math.floor(y)) * eps22[int(x), math.ceil(y)] + \
           abs(y - math.ceil(y)) * eps22[int(x), math.floor(y)]
  elif x != math.floor(x) and y == math.floor(y):
    tens[0, 0] = abs(x - math.floor(x)) * eps11[math.ceil(x), int(y)] + \
           abs(x - math.ceil(x)) * eps11[math.floor(x), int(y)]
    tens[0, 1] = abs(x - math.floor(x)) * eps12[math.ceil(x), int(y)] + \
           abs(x - math.ceil(x)) * eps12[math.floor(x), int(y)]
    tens[1, 0] = abs(x - math.floor(x)) * eps12[math.ceil(x), int(y)] + \
           abs(x - math.ceil(x)) * eps12[math.floor(x), int(y)]
    tens[1, 1] = abs(x - math.floor(x)) * eps22[math.ceil(x), int(y)] + \
           abs(x - math.ceil(x)) * eps22[math.floor(x), int(y)]
  else:
    tens[0, 0] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * eps11[math.ceil(x), math.ceil(y)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * eps11[math.ceil(x), math.floor(y)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * eps11[math.floor(x), math.ceil(y)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * eps11[math.floor(x), math.floor(y)]
    tens[0, 1] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * eps12[math.ceil(x), math.ceil(y)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * eps12[math.ceil(x), math.floor(y)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * eps12[math.floor(x), math.ceil(y)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * eps12[math.floor(x), math.floor(y)]
    tens[1, 0] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * eps12[math.ceil(x), math.ceil(y)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * eps12[math.ceil(x), math.floor(y)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * eps12[math.floor(x), math.ceil(y)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * eps12[math.floor(x), math.floor(y)]
    tens[1, 1] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * eps22[math.ceil(x), math.ceil(y)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * eps22[math.ceil(x), math.floor(y)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * eps22[math.floor(x), math.ceil(y)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * eps22[math.floor(x), math.floor(y)]

  return (tens)
# end tens_interp

def tens_interp_torch(x, y, tensor_field):
  tens = torch.zeros((2, 2))
  eps11 = tensor_field[0, :, :]
  eps12 = tensor_field[1, :, :]
  eps22 = tensor_field[2, :, :]
  if x == math.floor(x) and y == math.floor(y):
    tens[0, 0] = eps11[int(x), int(y)]
    tens[0, 1] = eps12[int(x), int(y)]
    tens[1, 0] = eps12[int(x), int(y)]
    tens[1, 1] = eps22[int(x), int(y)]
  elif x == math.floor(x) and y != math.floor(y):
    tens[0, 0] = abs(y - math.floor(y)) * eps11[int(x), math.ceil(y)] + \
           abs(y - math.ceil(y)) * eps11[int(x), math.floor(y)]
    tens[0, 1] = abs(y - math.floor(y)) * eps12[int(x), math.ceil(y)] + \
           abs(y - math.ceil(y)) * eps12[int(x), math.floor(y)]
    tens[1, 0] = abs(y - math.floor(y)) * eps12[int(x), math.ceil(y)] + \
           abs(y - math.ceil(y)) * eps12[int(x), math.floor(y)]
    tens[1, 1] = abs(y - math.floor(y)) * eps22[int(x), math.ceil(y)] + \
           abs(y - math.ceil(y)) * eps22[int(x), math.floor(y)]
  elif x != math.floor(x) and y == math.floor(y):
    tens[0, 0] = abs(x - math.floor(x)) * eps11[math.ceil(x), int(y)] + \
           abs(x - math.ceil(x)) * eps11[math.floor(x), int(y)]
    tens[0, 1] = abs(x - math.floor(x)) * eps12[math.ceil(x), int(y)] + \
           abs(x - math.ceil(x)) * eps12[math.floor(x), int(y)]
    tens[1, 0] = abs(x - math.floor(x)) * eps12[math.ceil(x), int(y)] + \
           abs(x - math.ceil(x)) * eps12[math.floor(x), int(y)]
    tens[1, 1] = abs(x - math.floor(x)) * eps22[math.ceil(x), int(y)] + \
           abs(x - math.ceil(x)) * eps22[math.floor(x), int(y)]
  else:
    tens[0, 0] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * eps11[math.ceil(x), math.ceil(y)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * eps11[math.ceil(x), math.floor(y)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * eps11[math.floor(x), math.ceil(y)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * eps11[math.floor(x), math.floor(y)]
    tens[0, 1] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * eps12[math.ceil(x), math.ceil(y)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * eps12[math.ceil(x), math.floor(y)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * eps12[math.floor(x), math.ceil(y)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * eps12[math.floor(x), math.floor(y)]
    tens[1, 0] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * eps12[math.ceil(x), math.ceil(y)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * eps12[math.ceil(x), math.floor(y)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * eps12[math.floor(x), math.ceil(y)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * eps12[math.floor(x), math.floor(y)]
    tens[1, 1] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * eps22[math.ceil(x), math.ceil(y)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * eps22[math.ceil(x), math.floor(y)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * eps22[math.floor(x), math.ceil(y)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * eps22[math.floor(x), math.floor(y)]

  return (tens)

def tens_interp_3d(x, y, z, tensor_field):
  tens = np.zeros((3, 3))
  eps11 = tensor_field[0, :, :]
  eps12 = tensor_field[1, :, :]
  eps13 = tensor_field[2, :, :]
  eps22 = tensor_field[3, :, :]
  eps23 = tensor_field[4, :, :]
  eps33 = tensor_field[5, :, :]
  if ((math.floor(x) < 0) or (math.ceil(x) >= eps11.shape[0])
      or (math.floor(y) < 0) or (math.ceil(y) >= eps11.shape[1])
      or (math.floor(z) < 0) or (math.ceil(z) >= eps11.shape[2])):
     # data is out of bounds, return identity tensor
     tens[0,0] = 1
     tens[1,1] = 1
     tens[2,2] = 1
     return(tens)

   
  if x == math.floor(x) and y == math.floor(y) and z == math.floor(z):
    tens[0, 0] = eps11[int(x), int(y), int(z)]
    tens[0, 1] = eps12[int(x), int(y), int(z)]
    tens[1, 0] = eps12[int(x), int(y), int(z)]
    tens[0, 2] = eps13[int(x), int(y), int(z)]
    tens[2, 0] = eps13[int(x), int(y), int(z)]
    tens[1, 1] = eps22[int(x), int(y), int(z)]
    tens[1, 2] = eps23[int(x), int(y), int(z)]
    tens[2, 1] = eps23[int(x), int(y), int(z)]
    tens[2, 2] = eps33[int(x), int(y), int(z)]
  elif x == math.floor(x) and y != math.floor(y) and z != math.floor(z):
    tens[0, 0] = abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps11[int(x), math.ceil(y), math.ceil(z)] \
                 + abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps11[int(x), math.floor(y), math.ceil(z)] \
                 + abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps11[int(x), math.ceil(y), math.floor(z)] \
                 + abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps11[int(x), math.floor(y), math.floor(z)] 
    tens[0, 1] = abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps12[int(x), math.ceil(y), math.ceil(z)] \
                 + abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps12[int(x), math.floor(y), math.ceil(z)] \
                 + abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps12[int(x), math.ceil(y), math.floor(z)] \
                 + abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps12[int(x), math.floor(y), math.floor(z)] 
    tens[1, 0] = tens[0,1]
    tens[0, 2] = abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps13[int(x), math.ceil(y), math.ceil(z)] \
                 + abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps13[int(x), math.floor(y), math.ceil(z)] \
                 + abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps13[int(x), math.ceil(y), math.floor(z)] \
                 + abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps13[int(x), math.floor(y), math.floor(z)] 
    tens[2, 0] = tens[0,2]
    tens[1, 1] = abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps22[int(x), math.ceil(y), math.ceil(z)] \
                 + abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps22[int(x), math.floor(y), math.ceil(z)] \
                 + abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps22[int(x), math.ceil(y), math.floor(z)] \
                 + abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps22[int(x), math.floor(y), math.floor(z)] 
    tens[1, 2] = abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps23[int(x), math.ceil(y), math.ceil(z)] \
                 + abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps23[int(x), math.floor(y), math.ceil(z)] \
                 + abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps23[int(x), math.ceil(y), math.floor(z)] \
                 + abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps23[int(x), math.floor(y), math.floor(z)] 
    tens[2, 1] = tens[1,2]
    tens[2, 2] = abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps33[int(x), math.ceil(y), math.ceil(z)] \
                 + abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps33[int(x), math.floor(y), math.ceil(z)] \
                 + abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps33[int(x), math.ceil(y), math.floor(z)] \
                 + abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps33[int(x), math.floor(y), math.floor(z)]
  elif x == math.floor(x) and y == math.floor(y) and z != math.floor(z):
    tens[0, 0] = abs(z - math.floor(z)) * eps11[int(x), int(y), math.ceil(z)] \
                 + abs(z - math.ceil(z)) * eps11[int(x), int(y), math.floor(z)] 
    tens[0, 1] = abs(z - math.floor(z)) * eps12[int(x), int(y), math.ceil(z)] \
                 + abs(z - math.ceil(z)) * eps12[int(x), int(y), math.floor(z)] 
    tens[1, 0] = tens[0,1]
    tens[0, 2] = abs(z - math.floor(z)) * eps13[int(x), int(y), math.ceil(z)] \
                 + abs(z - math.ceil(z)) * eps13[int(x), int(y), math.floor(z)] 
    tens[2, 0] = tens[0,2]
    tens[1, 1] = abs(z - math.floor(z)) * eps22[int(x), int(y), math.ceil(z)] \
                 + abs(z - math.ceil(z)) * eps22[int(x), int(y), math.floor(z)] 
    tens[1, 2] = abs(z - math.floor(z)) * eps23[int(x), int(y), math.ceil(z)] \
                 + abs(z - math.ceil(z)) * eps23[int(x), int(y), math.floor(z)] 
    tens[2, 1] = tens[1,2]
    tens[2, 2] = abs(z - math.floor(z)) * eps33[int(x), int(y), math.ceil(z)] \
                 + abs(z - math.ceil(z)) * eps33[int(x), int(y), math.floor(z)]   
  elif x != math.floor(x) and y == math.floor(y) and z != math.floor(z):
    tens[0, 0] = abs(x - math.floor(x)) * abs(z - math.floor(z)) * eps11[math.ceil(x), int(y), math.ceil(z)] \
                 + abs(x - math.ceil(x)) * abs(z - math.floor(z)) * eps11[math.floor(x), int(y), math.ceil(z)] \
                 + abs(x - math.floor(x)) * abs(z - math.ceil(z)) * eps11[math.ceil(x), int(y), math.floor(z)] \
                 + abs(x - math.ceil(x)) * abs(z - math.ceil(z)) * eps11[math.floor(x), int(y), math.floor(z)]
    tens[0, 1] = abs(x - math.floor(x)) * abs(z - math.floor(z)) * eps12[math.ceil(x), int(y), math.ceil(z)] \
                 + abs(x - math.ceil(x)) * abs(z - math.floor(z)) * eps12[math.floor(x), int(y), math.ceil(z)] \
                 + abs(x - math.floor(x)) * abs(z - math.ceil(z)) * eps12[math.ceil(x), int(y), math.floor(z)] \
                 + abs(x - math.ceil(x)) * abs(z - math.ceil(z)) * eps12[math.floor(x), int(y), math.floor(z)]
    tens[1, 0] = tens[0,1]
    tens[0, 2] = abs(x - math.floor(x)) * abs(z - math.floor(z)) * eps13[math.ceil(x), int(y), math.ceil(z)] \
                 + abs(x - math.ceil(x)) * abs(z - math.floor(z)) * eps13[math.floor(x), int(y), math.ceil(z)] \
                 + abs(x - math.floor(x)) * abs(z - math.ceil(z)) * eps13[math.ceil(x), int(y), math.floor(z)] \
                 + abs(x - math.ceil(x)) * abs(z - math.ceil(z)) * eps13[math.floor(x), int(y), math.floor(z)]
    tens[2, 0] = tens[0,2]
    tens[1, 1] = abs(x - math.floor(x)) * abs(z - math.floor(z)) * eps22[math.ceil(x), int(y), math.ceil(z)] \
                 + abs(x - math.ceil(x)) * abs(z - math.floor(z)) * eps22[math.floor(x), int(y), math.ceil(z)] \
                 + abs(x - math.floor(x)) * abs(z - math.ceil(z)) * eps22[math.ceil(x), int(y), math.floor(z)] \
                 + abs(x - math.ceil(x)) * abs(z - math.ceil(z)) * eps22[math.floor(x), int(y), math.floor(z)]
    tens[1, 2] = abs(x - math.floor(x)) * abs(z - math.floor(z)) * eps23[math.ceil(x), int(y), math.ceil(z)] \
                 + abs(x - math.ceil(x)) * abs(z - math.floor(z)) * eps23[math.floor(x), int(y), math.ceil(z)] \
                 + abs(x - math.floor(x)) * abs(z - math.ceil(z)) * eps23[math.ceil(x), int(y), math.floor(z)] \
                 + abs(x - math.ceil(x)) * abs(z - math.ceil(z)) * eps23[math.floor(x), int(y), math.floor(z)]
    tens[2, 1] = tens[1,2]
    tens[2, 2] = abs(x - math.floor(x)) * abs(z - math.floor(z)) * eps33[math.ceil(x), int(y), math.ceil(z)] \
                 + abs(x - math.ceil(x)) * abs(z - math.floor(z)) * eps33[math.floor(x), int(y), math.ceil(z)] \
                 + abs(x - math.floor(x)) * abs(z - math.ceil(z)) * eps33[math.ceil(x), int(y), math.floor(z)] \
                 + abs(x - math.ceil(x)) * abs(z - math.ceil(z)) * eps33[math.floor(x), int(y), math.floor(z)]
  elif x != math.floor(x) and y == math.floor(y) and z == math.floor(z):
    tens[0, 0] = abs(x - math.floor(x)) * eps11[math.ceil(x), int(y),int(z)] \
                 + abs(x - math.ceil(x)) * eps11[math.floor(x), int(y), int(z)]
    tens[0, 1] = abs(x - math.floor(x)) * eps12[math.ceil(x), int(y),int(z)] \
                 + abs(x - math.ceil(x)) * eps12[math.floor(x), int(y), int(z)]
    tens[1, 0] = tens[0,1]
    tens[0, 2] = abs(x - math.floor(x)) * eps13[math.ceil(x), int(y),int(z)] \
                 + abs(x - math.ceil(x)) * eps13[math.floor(x), int(y), int(z)]
    tens[2, 0] = tens[0,2]
    tens[1, 1] = abs(x - math.floor(x)) * eps22[math.ceil(x), int(y),int(z)] \
                 + abs(x - math.ceil(x)) * eps22[math.floor(x), int(y), int(z)]
    tens[1, 2] = abs(x - math.floor(x)) * eps23[math.ceil(x), int(y),int(z)] \
                 + abs(x - math.ceil(x)) * eps23[math.floor(x), int(y), int(z)]
    tens[2, 1] = tens[1,2]
    tens[2, 2] = abs(x - math.floor(x)) * eps33[math.ceil(x), int(y),int(z)] \
                 + abs(x - math.ceil(x)) * eps33[math.floor(x), int(y), int(z)]
  elif x != math.floor(x) and y != math.floor(y) and z == math.floor(z):
    tens[0, 0] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * eps11[math.ceil(x), math.ceil(y), int(z)] \
                 + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * eps11[math.floor(x), math.ceil(y), int(z)] \
                 + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * eps11[math.ceil(x), math.floor(y), int(z)] \
                 + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * eps11[math.floor(x), math.floor(y), int(z)]  
    tens[0, 1] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * eps12[math.ceil(x), math.ceil(y), int(z)] \
                 + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * eps12[math.floor(x), math.ceil(y), int(z)] \
                 + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * eps12[math.ceil(x), math.floor(y), int(z)] \
                 + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * eps12[math.floor(x), math.floor(y), int(z)]  
    tens[1, 0] = tens[0, 1]
    tens[0, 2] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * eps13[math.ceil(x), math.ceil(y), int(z)] \
                 + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * eps13[math.floor(x), math.ceil(y), int(z)] \
                 + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * eps13[math.ceil(x), math.floor(y), int(z)] \
                 + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * eps13[math.floor(x), math.floor(y), int(z)]  
    tens[2, 0] = tens[0, 2]
    tens[1, 1] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * eps22[math.ceil(x), math.ceil(y), int(z)] \
                 + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * eps22[math.floor(x), math.ceil(y), int(z)] \
                 + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * eps22[math.ceil(x), math.floor(y), int(z)] \
                 + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * eps22[math.floor(x), math.floor(y), int(z)]  
    tens[1, 2] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * eps23[math.ceil(x), math.ceil(y), int(z)] \
                 + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * eps23[math.floor(x), math.ceil(y), int(z)] \
                 + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * eps23[math.ceil(x), math.floor(y), int(z)] \
                 + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * eps23[math.floor(x), math.floor(y), int(z)]  
    tens[2, 1] = tens[1, 2]
    tens[2, 2] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * eps33[math.ceil(x), math.ceil(y), int(z)] \
                 + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * eps33[math.floor(x), math.ceil(y), int(z)] \
                 + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * eps33[math.ceil(x), math.floor(y), int(z)] \
                 + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * eps33[math.floor(x), math.floor(y), int(z)]  
  elif x == math.floor(x) and y != math.floor(y) and z == math.floor(z):
    tens[0, 0] = abs(y - math.floor(y)) * eps11[int(x), math.ceil(y), int(z)] \
                 + abs(y - math.ceil(y)) * eps11[int(x), math.floor(y), int(z)]
    tens[0, 1] = abs(y - math.floor(y)) * eps12[int(x), math.ceil(y), int(z)] \
                 + abs(y - math.ceil(y)) * eps12[int(x), math.floor(y), int(z)]
    tens[1, 0] = tens[0,1]
    tens[0, 2] = abs(y - math.floor(y)) * eps13[int(x), math.ceil(y), int(z)] \
                 + abs(y - math.ceil(y)) * eps13[int(x), math.floor(y), int(z)]
    tens[2, 0] = tens[0,2]
    tens[1, 1] = abs(y - math.floor(y)) * eps22[int(x), math.ceil(y), int(z)] \
                 + abs(y - math.ceil(y)) * eps22[int(x), math.floor(y), int(z)]
    tens[1, 2] = abs(y - math.floor(y)) * eps23[int(x), math.ceil(y), int(z)] \
                 + abs(y - math.ceil(y)) * eps23[int(x), math.floor(y), int(z)]
    tens[2, 1] = tens[1,2]
    tens[2, 2] = abs(y - math.floor(y)) * eps33[int(x), math.ceil(y), int(z)] \
                 + abs(y - math.ceil(y)) * eps33[int(x), math.floor(y), int(z)]
  else:
    tens[0, 0] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps11[math.ceil(x), math.ceil(y), math.ceil(z)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps11[math.ceil(x), math.floor(y), math.ceil(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps11[math.floor(x), math.ceil(y), math.ceil(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps11[math.floor(x), math.floor(y), math.ceil(z)] \
           + abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps11[math.ceil(x), math.ceil(y), math.floor(z)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps11[math.ceil(x), math.floor(y), math.floor(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps11[math.floor(x), math.ceil(y), math.floor(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps11[math.floor(x), math.floor(y), math.floor(z)]
    tens[0, 1] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps12[math.ceil(x), math.ceil(y), math.ceil(z)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps12[math.ceil(x), math.floor(y), math.ceil(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps12[math.floor(x), math.ceil(y), math.ceil(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps12[math.floor(x), math.floor(y), math.ceil(z)] \
           + abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps12[math.ceil(x), math.ceil(y), math.floor(z)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps12[math.ceil(x), math.floor(y), math.floor(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps12[math.floor(x), math.ceil(y), math.floor(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps12[math.floor(x), math.floor(y), math.floor(z)]
    tens[1, 0] = tens[0,1]
    tens[0, 2] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps13[math.ceil(x), math.ceil(y), math.ceil(z)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps13[math.ceil(x), math.floor(y), math.ceil(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps13[math.floor(x), math.ceil(y), math.ceil(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps13[math.floor(x), math.floor(y), math.ceil(z)] \
           + abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps13[math.ceil(x), math.ceil(y), math.floor(z)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps13[math.ceil(x), math.floor(y), math.floor(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps13[math.floor(x), math.ceil(y), math.floor(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps13[math.floor(x), math.floor(y), math.floor(z)]
    tens[2, 0] = tens[0,2]
    tens[1, 1] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps22[math.ceil(x), math.ceil(y), math.ceil(z)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps22[math.ceil(x), math.floor(y), math.ceil(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps22[math.floor(x), math.ceil(y), math.ceil(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps22[math.floor(x), math.floor(y), math.ceil(z)] \
           + abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps22[math.ceil(x), math.ceil(y), math.floor(z)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps22[math.ceil(x), math.floor(y), math.floor(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps22[math.floor(x), math.ceil(y), math.floor(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps22[math.floor(x), math.floor(y), math.floor(z)]
    tens[1, 2] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps23[math.ceil(x), math.ceil(y), math.ceil(z)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps23[math.ceil(x), math.floor(y), math.ceil(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps23[math.floor(x), math.ceil(y), math.ceil(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps23[math.floor(x), math.floor(y), math.ceil(z)] \
           + abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps23[math.ceil(x), math.ceil(y), math.floor(z)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps23[math.ceil(x), math.floor(y), math.floor(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps23[math.floor(x), math.ceil(y), math.floor(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps23[math.floor(x), math.floor(y), math.floor(z)]
    tens[2,1] = tens[1,2]
    tens[2, 2] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps33[math.ceil(x), math.ceil(y), math.ceil(z)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps33[math.ceil(x), math.floor(y), math.ceil(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps33[math.floor(x), math.ceil(y), math.ceil(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps33[math.floor(x), math.floor(y), math.ceil(z)] \
           + abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps33[math.ceil(x), math.ceil(y), math.floor(z)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps33[math.ceil(x), math.floor(y), math.floor(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps33[math.floor(x), math.ceil(y), math.floor(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps33[math.floor(x), math.floor(y), math.floor(z)]

  return (tens)

# compute eigenvectors according to A Method for Fast Diagonalization of a 2x2 or 3x3 Real Symmetric Matrix
# M.J. Kronenburg
# https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwj6zeiLut3qAhUPac0KHcyjDn4QFjAGegQIAxAB&url=https%3A%2F%2Farxiv.org%2Fpdf%2F1306.6291&usg=AOvVaw0BbaDECw-ghHGxek-LaB33
def eigv(tens):
    # TODO check dimensions, for now assuming 2D
    # The hope is that this implementation fixes the sign issue
    phi = 0.5 * np.arctan2(2 * tens[:,:,0,1] , (tens[:,:,0,0] - tens[:,:,1,1]))
    vs = np.zeros_like(tens)
    # want vs [:,:,0] to contain the eigenvector corresponding to the smallest
    # eigenvalue, which is \lambda_2 from the link above ie [-sin(\phi) cos(\phi)]
    vs[:,:,1,0] = np.cos(phi)
    vs[:,:,1,1] = np.sin(phi)
    vs[:,:,0,1] = vs[:,:,1,0] # cos(phi)
    vs[:,:,0,0] = -vs[:,:,1,1] # -sin(phi)
    return (vs)

def eigv_up(tens):
    # Compute eigenvectors for 2D tensors stored in upper triangular format
    # TODO check dimensions, for now assuming 2D
    # The hope is that this implementation fixes the sign issue
    phi = 0.5 * np.arctan2(2 * tens[:,:,1] , (tens[:,:,0] - tens[:,:,2]))
    vs = np.zeros_like(tens)
    # want vs [:,:,0] to contain the eigenvector corresponding to the smallest
    # eigenvalue, which is \lambda_2 from the link above ie [-sin(\phi) cos(\phi)]
    vs[:,:,1,0] = np.cos(phi)
    vs[:,:,1,1] = np.sin(phi)
    vs[:,:,0,1] = vs[:,:,1,0] # cos(phi)
    vs[:,:,0,0] = -vs[:,:,1,1] # -sin(phi)
    return (vs)

def eigv_3d(tens):
    # TODO check dimensions, for now assuming 3D
    # The hope is that this implementation fixes the sign issue
#     phi = 0.5 * np.arctan2(2 * tens[:,:,:,0,1] , (tens[:,:,:,0,0] - tens[:,:,:,1,1]))
#     vs = np.zeros_like(tens)
#     # want vs [:,:,0] to contain the eigenvector corresponding to the smallest
#     # eigenvalue, which is \lambda_2 from the link above ie [-sin(\phi) cos(\phi)]
#     vs[:,:,:,1,0] = np.cos(phi)
#     vs[:,:,:,1,1] = np.sin(phi)
#     vs[:,:,:,0,1] = vs[:,:,:,1,0] # cos(phi)
#     vs[:,:,:,0,0] = -vs[:,:,:,1,1] # -sin(phi)
  # instead of using Kronenburg method, we will call numpy's linalg.eigh
  # and simply correct the sign for the case where we get a positive eigenvector [1 0 0]
  # in between two eigenvectors [-0.99967193 -0.02561302 -0] and [-0.99967193  0.02561302  0]
  # in this case, we really want the positive eigenvector to be [-1 -0 -0]
  eigenvals, eigenvecs = np.linalg.eigh(tens)
  # want center pixel eigenvector to have same sign as both neighbors, when both neighbors sign matches
  # lr_dot, bt_dot, rf_dot > 0 ==> neighbors sign matches
  # lp_dot, bp_dot, rp_dot < 0 ==> pixel sign does not match neighbor
  lp_dot = np.einsum('...j,...j',eigenvecs[:-1,:,:,:,2],eigenvecs[1:,:,:,:,2])
  lr_dot = np.einsum('...j,...j',eigenvecs[:-2,:,:,:,2],eigenvecs[2:,:,:,:,2])
  bp_dot = np.einsum('...j,...j',eigenvecs[:,:-1,:,:,2],eigenvecs[:,1:,:,:,2])
  bt_dot = np.einsum('...j,...j',eigenvecs[:,:-2,:,:,2],eigenvecs[:,2:,:,:,2])
  rp_dot = np.einsum('...j,...j',eigenvecs[:,:,:-1,:,2],eigenvecs[:,:,1:,:,2])
  rf_dot = np.einsum('...j,...j',eigenvecs[:,:,:-2,:,2],eigenvecs[:,:,2:,:,2])
  for xx in range(tens.shape[0]):
    for yy in range(tens.shape[1]):
      for zz in range(tens.shape[2]):
        if xx < lr_dot.shape[0]:
          if lr_dot[xx,yy,zz] > 0 and lp_dot[xx,yy,zz] < 0:
            eigenvecs[xx+1,yy,zz,:,2] = -eigenvecs[xx+1,yy,zz,:,2]
        if yy < bt_dot.shape[1]:
          if bt_dot[xx,yy,zz] > 0 and bp_dot[xx,yy,zz] < 0:
            print('y direction need to correct:',xx,yy,zz)
            #eigenvecs[xx,yy+1,zz,:,2] = -eigenvecs[xx,yy+1,zz,:,2]
        if zz < rf_dot.shape[2]:
          if rf_dot[xx,yy,zz] > 0 and rp_dot[xx,yy,zz] < 0:
            print('z direction need to correct:',xx,yy,zz)
            #eigenvecs[xx,yy,zz+1,:,2] = -eigenvecs[xx,yy,zz+1,:,2]
  return (eigenvecs)

# def eigv_up_3d(tens):
#     # Compute eigenvectors for 3D tensors stored in upper triangular format
#     # TODO check dimensions, for now assuming 3D
#     # The hope is that this implementation fixes the sign issue
# NOT IMPLEMENTED YET
#     phi = 0.5 * np.arctan2(2 * tens[:,:,:,1] , (tens[:,:,:,0] - tens[:,:,:,2]))
#     vs = np.zeros_like(tens)
#     # want vs [:,:,0] to contain the eigenvector corresponding to the smallest
#     # eigenvalue, which is \lambda_2 from the link above ie [-sin(\phi) cos(\phi)]
#     vs[:,:,:,1,0] = np.cos(phi)
#     vs[:,:,:,1,1] = np.sin(phi)
#     vs[:,:,:,0,1] = vs[:,:,:,1,0] # cos(phi)
#     vs[:,:,:,0,0] = -vs[:,:,:,1,1] # -sin(phi)
#     return (vs)
  
 
def scale_by_alpha(tensors, alpha):
  # This scaling function assumes that the input provided for scaling are diffusion tensors
  # and hence scales by 1/e^{\alpha}.
  # If the inverse-tensor metric is provided instead, we would need to scale by e^\alpha
  out_tensors = np.copy(tensors)

  if tensors.shape[2] == 3:
    for kk in range(3): 
      out_tensors[:,:,kk] /= np.exp(alpha)
  elif tensors.shape[2:] == (2, 2):
    for jj in range(2):
      for kk in range(2):
        out_tensors[:,:,jj,kk] /= np.exp(alpha)
  elif tensors.shape[3] == 6:
    for kk in range(6): 
      out_tensors[:,:,:,kk] /= np.exp(alpha)
  elif tensors.shape[3:] == (3, 3):
    for jj in range(3):
      for kk in range(3):
        out_tensors[:,:,:,jj,kk] /= np.exp(alpha)
  else:
    print(tensors.shape, "unexpected tensor shape")
  return(out_tensors)

def threshold_to_input(tens_to_thresh, input_tens, mask, ratio=1.0):
  # scale the tens_to_thresh by the ratio * norm^2 of the largest tensor in input_tens
  # assumes input tens are full 2x2 tensors
  # TODO confirm that ratio is between 0 and 1
  if input_tens.shape[2] == 3:
    norm_in_tens = np.linalg.norm(input_tens,axis=(2))
  elif input_tens.shape[2:] == (2,2):
    norm_in_tens = np.linalg.norm(input_tens,axis=(2,3))
  elif input_tens.shape[3] == 6:
    norm_in_tens = np.linalg.norm(input_tens,axis=(3))
  elif input_tens.shape[3:] == (3,3):
    norm_in_tens = np.linalg.norm(input_tens,axis=(3,4))
  else:
    print(input_tens.shape, "unexpected tensor shape")
  if tens_to_thresh.shape[2] == 3:
    norm_sq = np.linalg.norm(tens_to_thresh,axis=(2))
  elif tens_to_thresh.shape[2:] == (2,2):
    norm_sq = np.linalg.norm(tens_to_thresh,axis=(2,3))
  elif tens_to_thresh.shape[3] == 6:
    norm_sq = np.linalg.norm(tens_to_thresh,axis=(3))    
  elif tens_to_thresh.shape[3:] == (3,3):
    norm_sq = np.linalg.norm(tens_to_thresh,axis=(3,4))    
  else:
    print(tens_to_thresh.shape, "unexpected tensor shape")  
  norm_sq = norm_sq * norm_sq # norm squared of each tensor in tens_to_thresh

  # just square the threshold, no need to element-wise square the entire norm_in_tens matrix
  thresh = np.max(norm_in_tens)
  thresh = ratio * thresh * thresh
  
  thresh_tens = np.copy(tens_to_thresh)
  scale_factor = np.ones_like(norm_sq)
  scale_factor[norm_sq > thresh] = thresh / norm_sq[norm_sq > thresh]
  scale_factor[mask == 0] = 1

  if tens_to_thresh.shape[2] == 3:
    for kk in range(3): 
      thresh_tens[:,:,kk] *= scale_factor
  elif tens_to_thresh.shape[2:] == (2, 2):
    for jj in range(2):
      for kk in range(2):
        thresh_tens[:,:,jj,kk] *= scale_factor
  elif tens_to_thresh.shape[3] == 6:
    for kk in range(6): 
      thresh_tens[:,:,:,kk] *= scale_factor
  elif tens_to_thresh.shape[3:] == (3, 3):
    for jj in range(3):
      for kk in range(3):
        thresh_tens[:,:,:,jj,kk] *= scale_factor
  else:
    print(tens_to_thresh.shape, "unexpected tensor shape")

  return(thresh_tens)
# end threshold_to_input

  
