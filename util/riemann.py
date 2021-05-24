# Riemannian utilities

from lazy_imports import np

#def riem_norm_elem(vec, g):
#    ne = vec / np.sqrt(np.dot(vec.transpose(), np.dot(g, vec)))
#    return(ne)
  
#riem_vec_norm = np.vectorize(riem_norm_elem, signature='(n),(n,n)->(n)')

def riem_vec_norm(vec, g):
  # Compute the Riemannian norm of a vector based on a metric, g
  nrm = 1.0 / np.sqrt(np.einsum('...i,...ij,...j',vec,g,vec))
  return (np.einsum('...i,...->...i',vec,nrm))
