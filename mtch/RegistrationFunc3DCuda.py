import torch


def get_div(v):
#     original
#     v_x = (torch.roll(v[0], shifts=(0, 0, -1), dims=(0, 1, 2))
#            - torch.roll(v[0], shifts=(0, 0, -1), dims=(0, 1, 2))) / 2
#     v_y = (torch.roll(v[1], shifts=(0, -1, 0), dims=(0, 1, 2))
#            - torch.roll(v[1], shifts=(0, -1, 0), dims=(0, 1, 2))) / 2
#     v_z = (torch.roll(v[2], shifts=(-1, 0, 0), dims=(0, 1, 2))
#            - torch.roll(v[2], shifts=(-1, 0, 0), dims=(0, 1, 2))) / 2
# 4.3 version
#     v_z = (torch.roll(v[0], shifts=(0, 0, -1), dims=(0, 1, 2))
#            - torch.roll(v[0], shifts=(0, 0, -1), dims=(0, 1, 2))) / 2
#     v_y = (torch.roll(v[1], shifts=(0, -1, 0), dims=(0, 1, 2))
#            - torch.roll(v[1], shifts=(0, -1, 0), dims=(0, 1, 2))) / 2
#     v_x = (torch.roll(v[2], shifts=(-1, 0, 0), dims=(0, 1, 2))
#            - torch.roll(v[2], shifts=(-1, 0, 0), dims=(0, 1, 2))) / 2
# 4.7 version
#     print('div')
    v_x = (torch.roll(v[0], shifts=(-1, 0, 0), dims=(0, 1, 2))
           - torch.roll(v[0], shifts=(-1, 0, 0), dims=(0, 1, 2))) / 2
    v_y = (torch.roll(v[1], shifts=(0, -1, 0), dims=(0, 1, 2))
           - torch.roll(v[1], shifts=(0, -1, 0), dims=(0, 1, 2))) / 2
    v_z = (torch.roll(v[2], shifts=(0, 0, -1), dims=(0, 1, 2))
           - torch.roll(v[2], shifts=(0, 0, -1), dims=(0, 1, 2))) / 2
    return v_x + v_y + v_z


def get_jacobian_determinant(diffeo):  # diffeo: 3 x size_h x size_w x size_d
    M = get_jacobian_matrix(diffeo)  # jac_m: 3 x 3 x size_h x size_w x size_d
    det = M[0, 0] * (M[1, 1] * M[2, 2] - M[1, 2] * M[2, 1]) \
          - M[0, 1] * (M[1, 0] * M[2, 2] - M[1, 2] * M[2, 0]) \
          + M[0, 2] * (M[1, 0] * M[2, 1] - M[1, 1] * M[2, 0])
    return det  # size_h x size_w x size_d


def get_jacobian_matrix(diffeo):  # diffeo: 3 x size_h x size_w x size_d
    return torch.stack((get_gradient(diffeo[0]), get_gradient(diffeo[1]), get_gradient(diffeo[2]))).to(device=torch.device('cuda'))


def get_gradient(F):  # 3D F: size_h x size_w x size_d
    F_padded = torch.zeros((F.shape[0] + 2, F.shape[1] + 2, F.shape[2] + 2))
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
                        F_z[1:-1, 1:-1, 1:-1])).type(torch.DoubleTensor).to(device=torch.device('cuda'))


# get the identity mapping
def get_idty(size_h, size_w, size_d):
    HH, WW, DD = torch.meshgrid([torch.arange(size_h),#, dtype=torch.double
                                 torch.arange(size_w),#, dtype=torch.double
                                 torch.arange(size_d)])#, dtype=torch.double
# original and 4.3
    return torch.stack((HH, WW, DD)).double().to(device=torch.device('cuda')) #.half()
# 4.7
#     return torch.stack((DD, WW, HH)).double() #.half()


# my interpolation function
def compose_function(f, diffeo, mode='periodic'):  # f: N x h x w x d  diffeo: 3 x h x w x d
    f = f.permute(f.dim() - 3, f.dim() - 2, f.dim() - 1, *range(f.dim() - 3))  # change the size of f to m x n x ...
    size_h, size_w, size_d = f.shape[:3]
#     original and 4.3
    Ind_diffeo = torch.stack((torch.floor(diffeo[0]).long() % size_h,
                              torch.floor(diffeo[1]).long() % size_w,
                              torch.floor(diffeo[2]).long() % size_d)).to(device=torch.device('cuda'))
#     4.7
#     Ind_diffeo = torch.stack((torch.floor(diffeo[2]).long() % size_h,
#                               torch.floor(diffeo[1]).long() % size_w,
#                               torch.floor(diffeo[0]).long() % size_d))#.to(device=torch.device('cuda'))

    F = torch.zeros(size_h + 1, size_w + 1, size_d + 1, *f.shape[3:])#, dtype=torch.double

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

    # use the bilinear interpolation method
    F000 = F[Ind_diffeo[0], Ind_diffeo[1], Ind_diffeo[2]].permute(*range(3, f.dim()), 0, 1, 2)
    F010 = F[Ind_diffeo[0], Ind_diffeo[1] + 1, Ind_diffeo[2]].permute(*range(3, f.dim()), 0, 1, 2)
    F100 = F[Ind_diffeo[0] + 1, Ind_diffeo[1], Ind_diffeo[2]].permute(*range(3, f.dim()), 0, 1, 2)
    F110 = F[Ind_diffeo[0] + 1, Ind_diffeo[1] + 1, Ind_diffeo[2]].permute(*range(3, f.dim()), 0, 1, 2)
    F001 = F[Ind_diffeo[0], Ind_diffeo[1], Ind_diffeo[2] + 1].permute(*range(3, f.dim()), 0, 1, 2)
    F011 = F[Ind_diffeo[0], Ind_diffeo[1] + 1, Ind_diffeo[2] + 1].permute(*range(3, f.dim()), 0, 1, 2)
    F101 = F[Ind_diffeo[0] + 1, Ind_diffeo[1], Ind_diffeo[2] + 1].permute(*range(3, f.dim()), 0, 1, 2)
    F111 = F[Ind_diffeo[0] + 1, Ind_diffeo[1] + 1, Ind_diffeo[2] + 1].permute(*range(3, f.dim()), 0, 1, 2)

#     original and 4.3
    C = diffeo[0] - Ind_diffeo[0]#.type(torch.DoubleTensor)
    D = diffeo[1] - Ind_diffeo[1]#.type(torch.DoubleTensor)
    E = diffeo[2] - Ind_diffeo[2]#.type(torch.DoubleTensor)
# 4.7
#     C = diffeo[0] - Ind_diffeo[2]#.type(torch.DoubleTensor)
#     D = diffeo[1] - Ind_diffeo[1]#.type(torch.DoubleTensor)
#     E = diffeo[2] - Ind_diffeo[0]#.type(torch.DoubleTensor)

    interp_f = (1 - C) * (1 - D) * (1 - E) * F000 \
               + (1 - C) * D * (1 - E) * F010 \
               + C * (1 - D) * (1 - E) * F100 \
               + C * D * (1 - E) * F110 \
               + (1 - C) * (1 - D) * E * F001 \
               + (1 - C) * D * E * F011 \
               + C * (1 - D) * E * F101 \
               + C * D * E * F111

#     del F000, F010, F100, F110, F001, F011, F101, F111, C, D, E
#     torch.cuda.empty_cache()
    return interp_f.to(device=torch.device('cuda'))
