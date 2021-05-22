import torch
import numpy as np
from Packages.RegistrationFunc3D import *
from Packages.SplitEbinMetric import *
from Packages.GeoPlot import *
import scipy.io as sio
import matplotlib.pyplot as plt
import SimpleITK as sitk

# torch.set_default_tensor_type('torch.cuda.DoubleTensor')
# torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
l0 = torch.tril(torch.randint(1,10,(145, 174, 145, 3, 3),dtype=torch.double))
g0 = torch.einsum("...ij,...kj->...ik", l0, l0)
print(l0[43,54,32])

l1 = torch.tril(torch.randint(1,10,(145, 174, 145, 3, 3),dtype=torch.double))
g1 = torch.einsum("...ij,...kj->...ik", l1, l1)
G = torch.stack((g0, g1))
# Tpts = 3
geo_group = get_karcher_mean(G, 1./g0.shape[-1])