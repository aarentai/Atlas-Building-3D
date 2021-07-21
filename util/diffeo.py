# Some useful operations for diffeomorphisms

import math
from lazy_imports import sio
from lazy_imports import torch
from util.tensors import direction
 
def coord_register(point_x, point_y, diffeo):
  # TODO work out which is y and which is x, maintain consistency.
  # For now, pass in y for point_x, x for point_y
  new_point_x, new_point_y = [], []
  for i in range(len(point_x)):
    D = point_y[i] - math.floor(point_y[i])
    C = point_x[i] - math.floor(point_x[i])
    new_point_x.append((1. - D) * (1. - C) * diffeo[
            0, math.floor(point_y[i]) % diffeo.shape[1], math.floor(point_x[i]) % diffeo.shape[2]]
                           + C * (1. - D) * diffeo[
                               0, math.floor(point_y[i]) % diffeo.shape[1], math.ceil(point_x[i]) % diffeo.shape[2]]
                           + D * (1. - C) * diffeo[
                               0, math.ceil(point_y[i]) % diffeo.shape[1], math.floor(point_x[i]) % diffeo.shape[2]]
                           + C * D * diffeo[
                               0, math.ceil(point_y[i]) % diffeo.shape[1], math.ceil(point_x[i]) % diffeo.shape[2]])
 
    new_point_y.append((1. - D) * (1. - C) * diffeo[
            1, math.floor(point_y[i]) % diffeo.shape[1], math.floor(point_x[i]) % diffeo.shape[2]]
                           + C * (1. - D) * diffeo[
                               1, math.floor(point_y[i]) % diffeo.shape[1], math.ceil(point_x[i]) % diffeo.shape[2]]
                           + D * (1. - C) * diffeo[
                               1, math.ceil(point_y[i]) % diffeo.shape[1], math.floor(point_x[i]) % diffeo.shape[2]]
                           + C * D * diffeo[
                               1, math.ceil(point_y[i]) % diffeo.shape[1], math.ceil(point_x[i]) % diffeo.shape[2]])
 
  return(new_point_x, new_point_y)

def coord_velocity_register(point_x, point_y, tensor_field, delta_t, diffeo):
  # TODO work out which is y and which is x, maintain consistency.
  # For now, pass in y for point_x, x for point_y
  # returns new y, new x as new_point_x, new_point_y
  # keeping y-x convention from coord_register for now for points,
  # but fixing it to x-y for velocity
  # tensor_field is in [tensor, x, y] order 
  # velocity will be returned as [new_vel_x, new_vel_y]
  new_point_x, new_point_y = coord_register(point_x, point_y, diffeo)
  new_velocity = []

  print("WARNING WARNING WARNING!!! Treat the following velocity code as highly suspect!!!")
  
  for i in range(len(point_x)):
    v = direction([new_point_y[i], new_point_x[i]], tensor_field)
    end_x = point_x[i] + v[1] * delta_t
    end_y = point_y[i] + v[0] * delta_t
    new_end_x, new_end_y = coord_register([end_x], [end_y], diffeo)

    new_velocity.append([(new_end_y[0] - new_point_y[i]) / delta_t, (new_end_x[0] - new_point_x[i]) / delta_t])

  return(new_point_x, new_point_y, new_velocity)

# define the pullback action of phi
def phi_pullback(phi, g):
    idty = get_idty(*g.shape[-2:])
#     four layers of scalar field, of all 1, all 0, all 1, all 0, where the shape of each layer is g.shape[-2:]?
    d_phi = get_jacobian_matrix(phi - idty) + torch.einsum("ij,mn->ijmn", [torch.eye(2,dtype=torch.double),
                                                                           torch.ones(g.shape[-2:],dtype=torch.double)])
    g_phi = compose_function(g, phi)
#     matrix multiplication
# the last two dimension stays the same means point-wise multiplication, ijmn instead of jimn means the first d_phi need to be transposed
    return torch.einsum("ijmn,ikmn,klmn->jlmn",[d_phi, g_phi, d_phi])

def phi_pullback_3d(phi, g):
#     input: phi.shape = [3, h, w, d]; g.shape = [h, w, d, 3, 3]
#     output: shape = [h, w, d, 3, 3]
#     torch.set_default_tensor_type('torch.cuda.DoubleTensor')
    g = g.permute(3, 4, 0, 1, 2)
    idty = get_idty_3d(*g.shape[-3:]).to(g.device)
    #     four layers of scalar field, of all 1, all 0, all 1, all 0, where the shape of each layer is g.shape[-2:]?
    eye = torch.eye(3, device=phi.device)
    ones = torch.ones(*g.shape[-3:], device=g.device)
    d_phi = get_jacobian_matrix_3d(phi - idty) + torch.einsum("ij,mno->ijmno", eye, ones)
    g_phi = compose_function_3d(g, phi)
    return torch.einsum("ij...,ik...,kl...->...jl", d_phi, g_phi, d_phi)

def get_jacobian_matrix(diffeo): # diffeo: 2 x size_h x size_w
#     return torch.stack((get_gradient(diffeo[1]), get_gradient(diffeo[0])))
    return torch.stack((get_gradient(diffeo[0]), get_gradient(diffeo[1])))

def get_jacobian_matrix_3d(diffeo):  # diffeo: 3 x size_h x size_w x size_d
    return torch.stack((get_gradient_3d(diffeo[0]), get_gradient_3d(diffeo[1]), get_gradient_3d(diffeo[2])))#.to(device=torch.device('cuda'))

def get_gradient_3d(F):  # 3D F: size_h x size_w x size_d
    F_padded = torch.zeros((F.shape[0] + 2, F.shape[1] + 2, F.shape[2] + 2), device=F.device)#, dtype=torch.double)
    F_padded[1:-1, 1:-1, 1:-1] = F
    F_padded[0, :, :] = F_padded[1, :, :]
    F_padded[-1, :, :] = F_padded[-2, :, :]
    F_padded[:, 0, :] = F_padded[:, 1, :]
    F_padded[:, -1, :] = F_padded[:, -2, :]
    F_padded[:, :, 0] = F_padded[:, :, 1]
    F_padded[:, :, -1] = F_padded[:, :, -2]
#     original
#     F_x = (torch.roll(F_padded, shifts=(0, 0, -1), dims=(0, 1, 2))
#            - torch.roll(F_padded, shifts=(0, 0, 1), dims=(0, 1, 2))) / 2
#     F_y = (torch.roll(F_padded, shifts=(0, -1, 0), dims=(0, 1, 2))
#            - torch.roll(F_padded, shifts=(0, 1, 0), dims=(0, 1, 2))) / 2
#     F_z = (torch.roll(F_padded, shifts=(-1, 0, 0), dims=(0, 1, 2))
#            - torch.roll(F_padded, shifts=(1, 0, 0), dims=(0, 1, 2))) / 2
# 4.3 version
    F_x = (torch.roll(F_padded, shifts=(-1, 0, 0), dims=(0, 1, 2))
           - torch.roll(F_padded, shifts=(1, 0, 0), dims=(0, 1, 2))) / 2
    F_y = (torch.roll(F_padded, shifts=(0, -1, 0), dims=(0, 1, 2))
           - torch.roll(F_padded, shifts=(0, 1, 0), dims=(0, 1, 2))) / 2
    F_z = (torch.roll(F_padded, shifts=(0, 0, -1), dims=(0, 1, 2))
           - torch.roll(F_padded, shifts=(0, 0, 1), dims=(0, 1, 2))) / 2
    return torch.stack((F_x[1:-1, 1:-1, 1:-1],
                        F_y[1:-1, 1:-1, 1:-1],
                        F_z[1:-1, 1:-1, 1:-1]))



def get_gradient(F):  # 2D F: size_h x size_w
    F_padded = torch.zeros((F.shape[0]+2,F.shape[1]+2))
    F_padded[1:-1,1:-1] = F
    F_padded[0,:] = F_padded[1,:]
    F_padded[-1,:] = F_padded[-2,:]
    F_padded[:,0] = F_padded[:,1]
    F_padded[:,-1] = F_padded[:,-2]
    F_x = (torch.roll(F_padded, shifts=(0, -1), dims=(0, 1)) - torch.roll(F_padded, shifts=(0, 1), dims=(0, 1)))/2
    F_y = (torch.roll(F_padded, shifts=(-1, 0), dims=(0, 1)) - torch.roll(F_padded, shifts=(1, 0), dims=(0, 1)))/2
    return torch.stack((F_x[1:-1,1:-1].type(torch.DoubleTensor), F_y[1:-1,1:-1].type(torch.DoubleTensor)))
#     F_x = (torch.roll(F, shifts=(0, -1), dims=(0, 1)) - torch.roll(F, shifts=(0, 1), dims=(0, 1)))/2
#     F_y = (torch.roll(F, shifts=(-1, 0), dims=(0, 1)) - torch.roll(F, shifts=(1, 0), dims=(0, 1)))/2
#     return torch.stack((F_x, F_y))


# get the identity mapping
def get_idty(size_h, size_w): 
    HH, WW = torch.meshgrid([torch.arange(size_h, dtype=torch.double), torch.arange(size_w, dtype=torch.double)])
#     return torch.stack((HH, WW))
    return torch.stack((WW, HH))

def get_idty_3d(size_h, size_w, size_d):
    HH, WW, DD = torch.meshgrid([torch.arange(size_h),#, dtype=torch.double
                                 torch.arange(size_w),#, dtype=torch.double
                                 torch.arange(size_d)])#, dtype=torch.double
# original and 4.3
    return torch.stack((HH, WW, DD)).float()#.double() #.half()
# 4.7
#     return torch.stack((DD, WW, HH)).double() #.half()
  

# my interpolation function
def compose_function(f, diffeo, mode='periodic'):  # f: N x m x n  diffeo: 2 x m x n
    
    f = f.permute(f.dim()-2, f.dim()-1, *range(f.dim()-2))  # change the size of f to m x n x ...
    
    size_h, size_w = f.shape[:2]
#     Ind_diffeo = torch.stack((torch.floor(diffeo[0]).long()%size_h, torch.floor(diffeo[1]).long()%size_w))
    Ind_diffeo = torch.stack((torch.floor(diffeo[1]).long()%size_h, torch.floor(diffeo[0]).long()%size_w))

    F = torch.zeros(size_h+1, size_w+1, *f.shape[2:], dtype=torch.double)
    
    if mode=='border':
        F[:size_h,:size_w] = f
        F[-1, :size_w] = f[-1]
        F[:size_h, -1] = f[:, -1]
        F[-1, -1] = f[-1,-1]
    elif mode =='periodic':
        # extend the function values periodically (1234 1)
        F[:size_h,:size_w] = f
        F[-1, :size_w] = f[0]
        F[:size_h, -1] = f[:, 0]
        F[-1, -1] = f[0,0]
    
    # use the bilinear interpolation method
    F00 = F[Ind_diffeo[0], Ind_diffeo[1]].permute(*range(2, f.dim()), 0, 1)  # change the size to ...*m*n
    F01 = F[Ind_diffeo[0], Ind_diffeo[1]+1].permute(*range(2, f.dim()), 0, 1)
    F10 = F[Ind_diffeo[0]+1, Ind_diffeo[1]].permute(*range(2, f.dim()), 0, 1)
    F11 = F[Ind_diffeo[0]+1, Ind_diffeo[1]+1].permute(*range(2, f.dim()), 0, 1)

#     C = diffeo[0] - Ind_diffeo[0].type(torch.DoubleTensor)
#     D = diffeo[1] - Ind_diffeo[1].type(torch.DoubleTensor)
    C = diffeo[0] - Ind_diffeo[1].type(torch.DoubleTensor)
    D = diffeo[1] - Ind_diffeo[0].type(torch.DoubleTensor)

    F0 = F00 + (F01 - F00)*C
    F1 = F10 + (F11 - F10)*C
    return F0 + (F1 - F0)*D
#     return (1-D)*(1-C)*F00+C*(1-D)*F01+D*(1-C)*F10+C*D*F11

def compose_function_3d(f, diffeo, mode='periodic'):  # f: N x h x w x d  diffeo: 3 x h x w x d
    f = f.permute(f.dim() - 3, f.dim() - 2, f.dim() - 1, *range(f.dim() - 3))  # change the size of f to m x n x ...
    size_h, size_w, size_d = f.shape[:3]
#     original and 4.3
    Ind_diffeo = torch.stack((torch.floor(diffeo[0]).long() % size_h,
                              torch.floor(diffeo[1]).long() % size_w,
                              torch.floor(diffeo[2]).long() % size_d))#.to(device=torch.device('cuda'))
#     4.7
#     Ind_diffeo = torch.stack((torch.floor(diffeo[2]).long() % size_h,
#                               torch.floor(diffeo[1]).long() % size_w,
#                               torch.floor(diffeo[0]).long() % size_d))#.to(device=torch.device('cuda'))

    F = torch.zeros(size_h + 1, size_w + 1, size_d + 1, *f.shape[3:], device=f.device)#, dtype=torch.double

    if mode == 'border':
        F[:size_h, :size_w, :size_d] = f
        F[-1, :size_w, :size_d] = f[-1]
        F[:size_h, -1, :size_d] = f[:, -1]
        F[:size_h, :size_w, -1] = f[:, :, -1]
        F[-1, -1, -1] = f[-1, -1, -1]
    elif mode == 'periodic':
        # extend the function values periodically (1234 1)
        F[:size_h, :size_w, :size_d] = f
        F[-1, :size_w, :size_d] = f[0]
        F[:size_h, -1, :size_d] = f[:, 0]
        F[:size_h, :size_w, -1] = f[:, :, 0]
        F[-1, -1, -1] = f[0, 0, 0]

    # Break up following into pieces to reduce memory usage:
    # But do so in a way that allows back-propagation to work...
    # # use the bilinear interpolation method
#     F000 = F[Ind_diffeo[0], Ind_diffeo[1], Ind_diffeo[2]].permute(*range(3, f.dim()), 0, 1, 2)
#     F010 = F[Ind_diffeo[0], Ind_diffeo[1] + 1, Ind_diffeo[2]].permute(*range(3, f.dim()), 0, 1, 2)
#     F100 = F[Ind_diffeo[0] + 1, Ind_diffeo[1], Ind_diffeo[2]].permute(*range(3, f.dim()), 0, 1, 2)
#     F110 = F[Ind_diffeo[0] + 1, Ind_diffeo[1] + 1, Ind_diffeo[2]].permute(*range(3, f.dim()), 0, 1, 2)
#     F001 = F[Ind_diffeo[0], Ind_diffeo[1], Ind_diffeo[2] + 1].permute(*range(3, f.dim()), 0, 1, 2)
#     F011 = F[Ind_diffeo[0], Ind_diffeo[1] + 1, Ind_diffeo[2] + 1].permute(*range(3, f.dim()), 0, 1, 2)
#     F101 = F[Ind_diffeo[0] + 1, Ind_diffeo[1], Ind_diffeo[2] + 1].permute(*range(3, f.dim()), 0, 1, 2)
#     F111 = F[Ind_diffeo[0] + 1, Ind_diffeo[1] + 1, Ind_diffeo[2] + 1].permute(*range(3, f.dim()), 0, 1, 2)

# #     original and 4.3
#     C = diffeo[0] - Ind_diffeo[0]#.type(torch.DoubleTensor)
#     D = diffeo[1] - Ind_diffeo[1]#.type(torch.DoubleTensor)
#     E = diffeo[2] - Ind_diffeo[2]#.type(torch.DoubleTensor)
# # 4.7
# #     C = diffeo[0] - Ind_diffeo[2]#.type(torch.DoubleTensor)
# #     D = diffeo[1] - Ind_diffeo[1]#.type(torch.DoubleTensor)
# #     E = diffeo[2] - Ind_diffeo[0]#.type(torch.DoubleTensor)

#     interp_f = (1 - C) * (1 - D) * (1 - E) * F000 \
#                + (1 - C) * D * (1 - E) * F010 \
#                + C * (1 - D) * (1 - E) * F100 \
#                + C * D * (1 - E) * F110 \
#                + (1 - C) * (1 - D) * E * F001 \
#                + (1 - C) * D * E * F011 \
#                + C * (1 - D) * E * F101 \
#                + C * D * E * F111

    # Reduced memory usage version below.  Issues with back propagation?

#     original and 4.3
    C = diffeo[0] - Ind_diffeo[0]#.type(torch.DoubleTensor)
    D = diffeo[1] - Ind_diffeo[1]#.type(torch.DoubleTensor)
    E = diffeo[2] - Ind_diffeo[2]#.type(torch.DoubleTensor)

# 4.7
#     C = diffeo[0] - Ind_diffeo[2]#.type(torch.DoubleTensor)
#     D = diffeo[1] - Ind_diffeo[1]#.type(torch.DoubleTensor)
#     E = diffeo[2] - Ind_diffeo[0]#.type(torch.DoubleTensor)

   # use the bilinear interpolation method
    #F000 = F[Ind_diffeo[0], Ind_diffeo[1], Ind_diffeo[2]].permute(*range(3, f.dim()), 0, 1, 2)
    interp_f = (1 - C) * (1 - D) * (1 - E) * F[Ind_diffeo[0], Ind_diffeo[1], Ind_diffeo[2]].permute(*range(3, f.dim()), 0, 1, 2)

    #F010 = F[Ind_diffeo[0], Ind_diffeo[1] + 1, Ind_diffeo[2]].permute(*range(3, f.dim()), 0, 1, 2)
    interp_f += (1 - C) * D * (1 - E) * F[Ind_diffeo[0], Ind_diffeo[1] + 1, Ind_diffeo[2]].permute(*range(3, f.dim()), 0, 1, 2)
    
    #F100 = F[Ind_diffeo[0] + 1, Ind_diffeo[1], Ind_diffeo[2]].permute(*range(3, f.dim()), 0, 1, 2)
    interp_f += C * (1 - D) * (1 - E) * F[Ind_diffeo[0] + 1, Ind_diffeo[1], Ind_diffeo[2]].permute(*range(3, f.dim()), 0, 1, 2) 
    
    #F110 = F[Ind_diffeo[0] + 1, Ind_diffeo[1] + 1, Ind_diffeo[2]].permute(*range(3, f.dim()), 0, 1, 2)
    interp_f += C * D * (1 - E) * F[Ind_diffeo[0] + 1, Ind_diffeo[1] + 1, Ind_diffeo[2]].permute(*range(3, f.dim()), 0, 1, 2)
    
    #F001 = F[Ind_diffeo[0], Ind_diffeo[1], Ind_diffeo[2] + 1].permute(*range(3, f.dim()), 0, 1, 2)
    interp_f += (1 - C) * (1 - D) * E * F[Ind_diffeo[0], Ind_diffeo[1], Ind_diffeo[2] + 1].permute(*range(3, f.dim()), 0, 1, 2)
    
    #F011 = F[Ind_diffeo[0], Ind_diffeo[1] + 1, Ind_diffeo[2] + 1].permute(*range(3, f.dim()), 0, 1, 2)
    interp_f += (1 - C) * D * E * F[Ind_diffeo[0], Ind_diffeo[1] + 1, Ind_diffeo[2] + 1].permute(*range(3, f.dim()), 0, 1, 2)
    
    #F101 = F[Ind_diffeo[0] + 1, Ind_diffeo[1], Ind_diffeo[2] + 1].permute(*range(3, f.dim()), 0, 1, 2)
    interp_f += C * (1 - D) * E * F[Ind_diffeo[0] + 1, Ind_diffeo[1], Ind_diffeo[2] + 1].permute(*range(3, f.dim()), 0, 1, 2)
    
    #F111 = F[Ind_diffeo[0] + 1, Ind_diffeo[1] + 1, Ind_diffeo[2] + 1].permute(*range(3, f.dim()), 0, 1, 2)
    interp_f += C * D * E * F[Ind_diffeo[0] + 1, Ind_diffeo[1] + 1, Ind_diffeo[2] + 1].permute(*range(3, f.dim()), 0, 1, 2)

#     del F000, F010, F100, F110, F001, F011, F101, F111, C, D, E
#     torch.cuda.empty_cache()
    return interp_f

  

if __name__ == "__main__":
  phi = sio.loadmat('103818toTemp_phi.mat')
  phi_inv = sio.loadmat('103818toTemp_phi_inv.mat')
  phi = phi['diffeo']
  phi_inv = phi_inv['diffeo']
  new_points_x, new_points_y = coord_register(points_x, points_y, phi)
