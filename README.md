## Environment
`python37` with `pytorch 1.9.0` and `sym3eig`

## Data I/O convention

### Read
Shape of input_tensor.nhdr should be `[d, w, h, 6]`, and shape of input_mask.nhdr should be `[d, w, h]`
```
input_tensor = np.transpose(sitk.GetArrayFromImage(sitk.ReadImage(path)),(3,2,1,0))
input_mask = np.transpose(sitk.GetArrayFromImage(sitk.ReadImage(path)),(2,1,0))
```
input_tensor.shape is `[3, h, w, d]`, and input_mask.shape is `[h, w, d]`
### Write
output_tensor.shape is `[3, h, w, d]`, and output_mask.shape is `[h, w, d]`
```
output_tensor = sitk.WriteImage(sitk.GetImageFromArray(np.transpose(output_tensor,(3,2,1,0)), path)
output_mask = sitk.WriteImage(sitk.GetImageFromArray(np.transpose(output_tensor,(2,1,0)), path)
```
Shape of output_tensor.nhdr is `[d, w, h, 3]`, and shape of output_mask.nhdr is `[d, w, h]`

### Note
`sitk.WriteImage(sitk.GetImageFromArray())` and `sitk.GetArrayFromImage(sitk.ReadImage(path))` is a pair of inverse operation.
```
output_tensor = np.zeros((12,34,56,78))
sitk.WriteImage(sitk.GetImageFromArray(output_tensor), path)
input_tensor = sitk.GetArrayFromImage(sitk.ReadImage(path))
print(input_tensor)
'(12,34,56,78)'
```

## Data dim convention

Make sure you follow the conventions below to make the algorithm consistent.
- Tensor fields: All the tensor fields variables by default are of size `[h, w, d, 3, 3]`, making the last two dimensions index metric matrix, to comply pytorch. In my code, arguments and the outputs of all functions meet this requirement. 
- Compressed tensor fields: `atlas_lin` is always of size `[6, h, w, d]`, when it comes to the argument of `view_3d_tensors()`, using `np.transpose(atlas_lin,(1,2,3,0))` to satisfy the requirement temporarily.
- Diffeomorphisms: All the diffeo variables by default are of size `[3, h, w, d]`.
- Masks: All the mask variables by default are of size `[h, w, d]`, when it comes to `torch.einsum()`, you can use `.unsqueeze(0)` temporarily.

## Data plotting convention

To avoid the `x` and `y` ambiguity in indexing and ploting, naming the first two dimension in `[h, w, 2, 2]` in the order of `x`, `y` is the best choice! 
- When indexing the array, `x` indexes row and `y` indexes column, the way I typically do and the way how matplotlib plot the 2d image. 
- When plotting the tensors, matplotlib would rotate the array counterclockwise by 90 degrees. So the vertical axis is `y` and horizontal axis is `x`, which is also consistent with our knowledge in drawing the Cartesian coordinate system. 


## Algorithm caveat
- In energy calculation, only use the binary mask, rather than a weighted map, which will change the alpha field applied to the tensor field previously and result in geodesic misgoing.
- Both metric matching and mean calculating should be implemented on the inverse of the original DTI tensor field, since the geodesics are running on the inverse of the tensor field.
- When accumulating the diffeomorphisms, always remember the order of accumulation of phi and its inverse is different.
```
phi_acc = compose_function(phi_acc, phi)
psi_inv_acc = compose_function(phi_inv, psi_inv_acc)
```
- When an error like below is raised:
```
cholesky_cpu: For batch 0: U(1,1) is zero, singular U.
```
1. It's probably caused by a large epsilon, so the composed tensor field is no longer positive definite everywhere.
2. For cholesky failure in `Squared_distance_Ebin`, it's resulted from `trK0square = trKsquare(g0, g1) - torch.log(torch.det(inv_g0_g1)) ** 2 *a` in `SplitEbinMetric3DCuda.py` gets negative value, may need to increase the constant in `theta`.
3. For cholesky failure in `get_karchar_mean`, it's resulted from dirty data. When dealing with real brain data, making sure all the tensors are positive definite and have positive determinants are not enough, extremely small positive determinants are also not allowed, say 1e-19.
4. For `sym3eig` package, there might be some problems when encounter the matrix that is similar but not exactly equals to identity matrix, we suggest using the `torch.linalg.eig()`.
- `a` in `Squared_distance_Ebin(g0, g1, a, mask)`, `get_karcher_mean(G, a)`, `get_geo(g0, g1, a, Tpts)`, `inv_RieExp_extended(g0, g1, a)`, `Rie_Exp_extended(g0, u, a)`, `Rie_Exp(g0, u, a)`, `inv_RieExp(g0, g1, a)` equals to the reciprocal of dimension, `1/dim`, namely the last entry of tensor field's shape.

- When an out of range error is raised, check if all the tensors get the right dimension order.
- Pay attention to the indexes when assigning the `atlas` to `atlas_lin`.