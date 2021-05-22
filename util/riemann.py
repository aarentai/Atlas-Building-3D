# Riemannian utilities

from lazy_imports import np

def riem_norm_elem(vec, g):
    ne = vec / np.sqrt(np.dot(vec.transpose(), np.dot(g, vec)))
    #ne = vec / np.dot(vec.transpose(), vec)
    return(ne)
  
riem_vec_norm = np.vectorize(riem_norm_elem, signature='(n),(n,n)->(n)')
