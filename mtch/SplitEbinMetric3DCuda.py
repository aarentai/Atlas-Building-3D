import numpy as np
import torch
from scipy.linalg import expm, logm
import warnings
import matplotlib.pyplot as plt
from IPython.core.debugger import set_trace
from math import pi
from torchvectorized import vlinalg as tv
from torch_sym3eig import Sym3Eig as se
'''
SplitEbinMetric.py stays the same from Atlas2D to Atlas3D
'''

def trKsquare(B, A):
    G = torch.linalg.cholesky(B)
    inv_G = torch.inverse(G)
    W = torch.einsum("...ij,...jk,...lk->...il", inv_G, A, inv_G)
    lamda , _ = se.apply(W.reshape((-1,3,3)))
    lamda = lamda.reshape((*W.shape[:-2],3))
    result = torch.sum(torch.log(lamda) ** 2, (-1))
    return result


def Squared_distance_Ebin(g0, g1, a, mask):
#     inputs: g0.shape, g1.shape = [hxwxd, 3, 3]
#     output: scalar
#     3.3.4 https://www.cs.utah.edu/~haocheng/notes/NoteonMatching.pdf
    inv_g0_g1 = torch.einsum("...ik,...kj->...ij", torch.inverse(g0), g1)
    trK0square = trKsquare(g0, g1) - torch.log(torch.det(inv_g0_g1)) ** 2 *a  # torch.log(torch.det(inv_g0_g1) + 1e-25)
    theta = torch.min((trK0square / a + 1e-7).sqrt() / 4., torch.tensor(np.pi, dtype=torch.double))  # change 1e-40 to 1e-13, because there is only one negative error of 1e-15 in UKF brain experiment 
    alpha, beta = torch.det(g0).pow(1. / 4.), torch.det(g1).pow(1. / 4.)
    E = 16 * a * (alpha ** 2 - 2 * alpha * beta * torch.cos(theta) + beta ** 2)
    return torch.einsum("hwd,hwd->", E, mask)


def logm_invB_A(B, A):
    import SimpleITK as sitk
#     inputs: A/B.shape = [hxwxd, 3, 3]
#     output: shape = [hxwxd, 3, 3]
    G = torch.linalg.cholesky(B)
    inv_G = torch.inverse(G)
    W = torch.einsum("...ij,...jk,...lk->...il", inv_G, A, inv_G)
    lamda, Q = se.apply(W)
    log_lamda = torch.zeros((*lamda.shape, lamda.shape[-1]),dtype=torch.double)
    log_lamda = torch.diag_embed(torch.log(lamda))
    V = torch.einsum('...ji,...jk->...ik', inv_G, Q)
    inv_V = torch.inverse(V)
    return torch.einsum('...ij,...jk,...kl->...il', V, log_lamda, inv_V)


# 2 without for loops using Kyle's method
def inv_RieExp(g0, g1, a):  # g0,g1: two tensors of size (s,t,...,3,3), where g0\neq 0
    '''this function is to calculate the inverse Riemannian exponential of g1 in the image of the maximal domain of the Riemannian exponential at g0
    '''
    n = g1.size(-1)
    #     matrix multiplication
    inv_g0_g1 = torch.einsum("...ik,...kj->...ij", torch.inverse(g0), g1)  # (s,t,...,3,3)

    def get_u_g0direction(g0, inv_g0_g1):  # (-1,3,3) first reshape g0,g1,inv_g..
        #         permute
        inv_g0_g1 = torch.einsum("...ij->ij...", inv_g0_g1)  # (3,3,-1)
        s = inv_g0_g1[0, 0]  # (-1)
        u = 4 / n * (s ** (n / 4) - 1) * torch.einsum("...ij->ij...", g0)  # (-1)@(3,3,-1) -> (3,3,-1)
        return u.permute(2, 0, 1)  # (-1,3,3)

    def get_u_ng0direction(g0, g1, inv_g0_g1, a):  # (-1,3,3) first reshape g0,g1,inv_g..
        K = logm_invB_A(g0, g1)
        KTrless = K - torch.einsum("...ii,kl->...kl", K, torch.eye(n, dtype=torch.double)) / n  # (-1,3,3)
        #         AA^T
        theta = (1 / a * torch.einsum("...ik,...ki->...", KTrless, KTrless)).sqrt() / 4  # (-1)
        gamma = torch.det(g1).pow(1 / 4) / (torch.det(g0).pow(1 / 4))  # (-1)

        A = 4 / n * (gamma * torch.cos(theta) - 1)  # (-1)
        B = 1 / theta * gamma * torch.sin(theta)
        u = A * torch.einsum("...ij->ij...", g0) + B * torch.einsum("...ik,...kj->ij...", g0, KTrless)  # (-1)@(3,3,-1) -> (3,3,-1)
        return u.permute(2, 0, 1)  # (-1,3,3)

    inv_g0_g1_trless = inv_g0_g1 - torch.einsum("...ii,kl->...kl", inv_g0_g1, torch.eye(n, dtype=torch.double)) / n  # (s,t,...,2,2)
    norm0 = torch.einsum("...ij,...ij->...", inv_g0_g1_trless, inv_g0_g1_trless).reshape(-1)  # (-1)

    # find the indices for which the entries are 0s and non0s
    Ind0 = (norm0 <= 1e-12).nonzero().reshape(-1)  # using squeeze results in [1,1]->[]
    Indn0 = (norm0 > 1e-12).nonzero().reshape(-1)

    u = torch.zeros(g0.reshape(-1, n, n).size(), dtype=torch.double)  # (-1,3,3)
    if len(Indn0) == 0:
        u = get_u_g0direction(g0.reshape(-1, n, n), inv_g0_g1.reshape(-1, n, n))
    elif len(Ind0) == 0:
        u = get_u_ng0direction(g0.reshape(-1, n, n), g1.reshape(-1, n, n), inv_g0_g1.reshape(-1, n, n), a)
    else:
        u[Ind0] = get_u_g0direction(g0.reshape(-1, n, n)[Ind0], inv_g0_g1.reshape(-1, n, n)[Ind0])
        u[Indn0] = get_u_ng0direction(g0.reshape(-1, n, n)[Indn0], g1.reshape(-1, n, n)[Indn0], inv_g0_g1.reshape(-1, n, n)[Indn0], a)

    return u.reshape(g1.size())


def Rie_Exp(g0, u, a):  # here g0 is of size (s,t,...,3,3) and u is of size (s,t,...,3,3), where g0\neq 0
    '''this function is to calculate the Riemannian exponential of u in the the maximal domain of the Riemannian exponential at g0
    '''
    n = g0.size(-1)

    U = torch.einsum("...ik,...kj->...ij", torch.inverse(g0), u)  # (s,t,...,3,3)
    trU = torch.einsum("...ii->...", U)  # (s,t,...)
    UTrless = U - torch.einsum("...,ij->...ij", trU, torch.eye(n, n, dtype=torch.double)) / n  # (s,t,...,3,3)

    #     in g0 direction:K_0=0
    def get_g1_g0direction(g0, trU):  # first reshape g0 (-1,3,3) and trU (-1)
        g1 = (trU / 4 + 1).pow(4 / n) * torch.einsum("...ij->ij...", g0)  # (3,3,-1)
        return g1.permute(2, 0, 1)  # (-1,3,3)

    #     not in g0 direction SplitEbinMetric.pdf Theorem 1 :K_0\not=0
    def get_g1_ng0direction(g0, trU, UTrless, a):  # first reshape g0,UTrless (-1,3,3) and trU (-1)
        if len((trU < -4).nonzero().reshape(-1)) != 0:
            warnings.warn('The tangent vector u is out of the maximal domain of the Riemannian exponential.', DeprecationWarning)

        q = trU / 4 + 1  # (-1)
        r = (1 / a * torch.einsum("...ik,...ki->...", UTrless, UTrless)).sqrt() / 4  # (-1)

        ArctanUtrless = torch.atan2(r, q) * torch.einsum("...ij->ij...", UTrless) / r  # use (2,2,-1) for computation
        ExpArctanUtrless = torch.matrix_exp(ArctanUtrless.permute(2, 0, 1)).permute(1, 2, 0)

        g1 = (q ** 2 + r ** 2).pow(2 / n) * torch.einsum("...ik,kj...->ij...", g0, ExpArctanUtrless)  # (2,2,-1)
        return g1.permute(2, 0, 1)  # (-1,2,2)

    #     pointwise multiplication Tr(U^TU)
    norm0 = torch.einsum("...ij,...ij->...", UTrless, UTrless).reshape(-1)  # (-1)

    # find the indices for which the entries are 0s and non0s
    #     k_0=0 or \not=0
    Ind0 = (norm0 <= 1e-12).nonzero().reshape(-1)
    Indn0 = (norm0 > 1e-12).nonzero().reshape(-1)

    g1 = torch.zeros(g0.reshape(-1, n, n).size(), dtype=torch.double)  # (-1,2,2)
    if len(Indn0) == 0:
        g1 = get_g1_g0direction(g0.reshape(-1, n, n), trU.reshape(-1))
    elif len(Ind0) == 0:
        g1 = get_g1_ng0direction(g0.reshape(-1, n, n), trU.reshape(-1), UTrless.reshape(-1, n, n), a)
    else:
        g1[Ind0] = get_g1_g0direction(g0.reshape(-1, n, n)[Ind0], trU.reshape(-1)[Ind0])
        g1[Indn0] = get_g1_ng0direction(g0.reshape(-1, n, n)[Indn0], trU.reshape(-1)[Indn0], UTrless.reshape(-1, n, n)[Indn0], a)

    return g1.reshape(g0.size())


''' 
The following Riemannian exponential and inverse Riemannian exponential are extended to the case g0=0 
'''
def Rie_Exp_extended(g0, u, a):  # here g0 is of size (s,t,...,3,3) and u is of size (s,t,...,3,3)
    size = g0.size()
    g0, u = g0.reshape(-1, *size[-2:]), u.reshape(-1, *size[-2:])  # (-1,3,3)
    detg0 = torch.det(g0)

    Ind_g0_is0 = (detg0 == 0).nonzero().reshape(-1)
    Ind_g0_isnot0 = (detg0 != 0).nonzero().reshape(-1)

    if len(Ind_g0_isnot0) == 0:  # g0x are 0s for all x
        g1 = u * g0.size(-1) / 4
    elif len(Ind_g0_is0) == 0:  # g0x are PD for all x
        g1 = Rie_Exp(g0, u, a)
    else:
        g1 = torch.zeros(g0.size(), dtype=torch.double)
        g1[Ind_g0_is0] = u[Ind_g0_is0] * g0.size(-1) / 4
        g1[Ind_g0_isnot0] = Rie_Exp(g0[Ind_g0_isnot0], u[Ind_g0_isnot0], a)
    return g1.reshape(size)


def inv_RieExp_extended(g0, g1, a):  # g0, g1: (s,t,...,3,3)
    size = g0.size()
    g0, g1 = g0.reshape(-1, *size[-2:]), g1.reshape(-1, *size[-2:])  # (-1,3,3)
    detg0 = torch.det(g0)

    Ind_g0_is0 = (detg0 == 0).nonzero().reshape(-1)
    Ind_g0_isnot0 = (detg0 != 0).nonzero().reshape(-1)

    if len(Ind_g0_isnot0) == 0:  # g0x are 0s for all x
        u = g1 * 4 / g0.size(-1)
    elif len(Ind_g0_is0) == 0:  # g0x are PD for all x
        u = inv_RieExp(g0, g1, a)
    else:
        u = torch.zeros(g0.size(), dtype=torch.double)
        u[Ind_g0_is0] = g1[Ind_g0_is0] * 4 / g0.size(-1)
        u[Ind_g0_isnot0] = inv_RieExp(g0[Ind_g0_isnot0], g1[Ind_g0_isnot0], a)
    return u.reshape(size)


def get_geo(g0, g1, a, Tpts):  # (s,t,...,,3,3)
    '''
    use odd number Tpts of time points since the geodesic may go
    though the zero matrix which will give the middle point of the geodesic
    '''
    size = g0.size()

    g0, g1 = g0.reshape(-1, *size[-2:]), g1.reshape(-1, *size[-2:])  # (-1,3,3)

    Time = torch.arange(Tpts, out=torch.DoubleTensor()) / (Tpts - 1)  # (Tpts)

    U = logm_invB_A(g0, g1)
    UTrless = U - torch.einsum("...ii,kl->...kl", U, torch.eye(g1.size(-1), dtype=torch.double)) / g1.size(
        -1)  # (...,3,3)
    theta = ((1 / a * torch.einsum("...ik,...ki->...", UTrless, UTrless)).sqrt() / 4 - np.pi)
    Ind_inRange = (theta < 0).nonzero().reshape(-1)
    Ind_notInRange = (theta >= 0).nonzero().reshape(-1)

    def geo_in_range(g0, g1, a, Tpts):
        u = inv_RieExp_extended(g0, g1, a)  # (-1,3,3)
        geo = torch.zeros(Tpts, *g0.size(), dtype=torch.double)  # (Tpts,-1,3,3)
        geo[0], geo[-1] = g0, g1
        for i in range(1, Tpts - 1):
            geo[i] = Rie_Exp_extended(g0, u * Time[i], a)
        return geo  # (Tpts,-1,2,2)

    def geo_not_in_range(g0, g1, a, Tpts):  # (-1,3,3)
        m0 = torch.zeros(g0.size(), dtype=torch.double)
        u0 = inv_RieExp_extended(g0, m0, a)
        u1 = inv_RieExp_extended(g1, m0, a)

        geo = torch.zeros(Tpts, *g0.size(), dtype=torch.double)  # (Tpts,-1,3,3)
        geo[0], geo[-1] = g0, g1

        for i in range(1, int((Tpts - 1) / 2)):
            geo[i] = Rie_Exp_extended(g0, u0 * Time[i], a)
        for j in range(-int((Tpts - 1) / 2), -1):
            geo[j] = Rie_Exp_extended(g1, u1 * (1 - Time[j]), a)
        return geo  # (Tpts,-1,2,2)

    # If g1 = 0, len(Ind_notInRange) and len(Ind_inRange) are both zero. In this case we say that g1 is in the range
    if len(Ind_notInRange) == 0:  # all in the range
        geo = geo_in_range(g0, g1, a, Tpts)
    elif len(Ind_inRange) == 0:  # all not in range
        geo = geo_not_in_range(g0, g1, a, Tpts)
    else:
        geo = torch.zeros(Tpts, *g0.size(), dtype=torch.double)  # (Tpts,-1,3,3)
        geo[:, Ind_inRange] = geo_in_range(g0[Ind_inRange], g1[Ind_inRange], a, Tpts)
        geo[:, Ind_notInRange] = geo_not_in_range(g0[Ind_notInRange], g1[Ind_notInRange], a, Tpts)
    return geo.reshape(Tpts, *size)


def ptPick_notInRange(g0, g1, i):  # (-1,3,3)
    alpha = torch.det(g1).pow(1 / 4) / torch.det(g0).pow(1 / 4)  # (-1)
    Ind_close_to_g0 = (alpha <= i).nonzero().reshape(-1)
    Ind_close_to_g1 = (alpha > i).nonzero().reshape(-1)

    def get_gm_inLine_0g0(alpha, g0, i):
        kn_over4 = -(1 + alpha) / (i + 1)  # (-1)
        gm = (1 + kn_over4) ** (4 / g0.size(-1)) * torch.einsum("...ij->ij...", g0)  # (3,3,-1)
        return gm.permute(2, 0, 1)  # (-1,3,3)

    def get_gm_inLine_0g1(alpha, g1, i):
        kn_over4 = -i * (1 + 1 / alpha) / (i + 1)  # (-1)
        gm = (1 + kn_over4) ** (4 / g1.size(-1)) * torch.einsum("...ij->ij...", g1)  # (3,3,-1)
        return gm.permute(2, 0, 1)

    if len(Ind_close_to_g1) == 0:  # all are close to g0
        gm = get_gm_inLine_0g0(alpha, g0, i)
    elif len(Ind_close_to_g0) == 0:
        gm = get_gm_inLine_0g1(alpha, g1, i)
    else:
        gm = torch.zeros(g0.size(), dtype=torch.double)
        gm[Ind_close_to_g0] = get_gm_inLine_0g0(alpha[Ind_close_to_g0], g0[Ind_close_to_g0], i)
        gm[Ind_close_to_g1] = get_gm_inLine_0g1(alpha[Ind_close_to_g1], g1[Ind_close_to_g1], i)
    return gm



def get_karcher_mean(G, a):
    size = G.size()
    G = G.reshape(size[0], -1, *size[-2:])  # (T,-1,3,3)
    gm = G[0]
    for i in range(1, G.size(0)):
#         print('logm_invB_A')
        U = logm_invB_A(gm, G[i])
        UTrless = U - torch.einsum("...ii,kl->...kl", U, torch.eye(size[-1], dtype=torch.double)) / size[
            -1]  # (...,2,2)
        theta = ((torch.einsum("...ik,...ki->...", UTrless, UTrless) / a).sqrt() / 4 - np.pi)
        Ind_inRange = (theta < 0).nonzero().reshape(-1)  ## G[i] is in the range of the exponential map at gm
        Ind_notInRange = (theta >= 0).nonzero().reshape(-1)  ## G[i] is not in the range

        # when g1 = 0, len(Ind_notInRange) and len(Ind_inRange) are both zero. So check len(Ind_notInRange) first
        if len(Ind_notInRange) == 0:  # all in the range
#             print('Rie_Exp_extended')
            gm = Rie_Exp_extended(gm, inv_RieExp_extended(gm, G[i], a) / (i + 1), a)
        elif len(Ind_inRange) == 0:  # all not in range
#             print('ptPick_notInRange')
            gm = ptPick_notInRange(gm, G[i], i)
        else:
#             print('Rie_Exp_extended, ptPick_notInRange')
            gm[Ind_inRange] = Rie_Exp_extended(gm[Ind_inRange],
                                               inv_RieExp_extended(gm[Ind_inRange], G[i, Ind_inRange], a) / (i + 1),
                                               a)  # stop here
            gm[Ind_notInRange] = ptPick_notInRange(gm[Ind_notInRange], G[i, Ind_notInRange], i)
#             print('end')

    return gm.reshape(*size[1:])
