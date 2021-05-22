from lazy_imports import np

def make_symmetric(a):
    # make an array that is symmetric about the x center line
    is_even = True
    ctr = int(a.shape[0]/2)
    if ctr * 2 != a.shape[0]:
        is_even = False
    b = np.copy(a)
    if is_even:
        for ii in range(ctr):
            b[ctr-ii-1] = a[ctr+ii]
    else:
        for ii in range(ctr):
            b[ctr-ii-1] = a[ctr+ii+1]
    return(b)

def diff_from_symmetric(a):
    diff = make_symmetric(a) - a
    diff[np.abs(diff) < 1e-13] = 0
    return(diff)

def make_antisymmetric(a):
    # make an array that is antisymmetric about the x center line
    is_even = True
    ctr = int(a.shape[0]/2)
    if ctr * 2 != a.shape[0]:
        is_even = False
        #print('odd shape, 2*',ctr,'not equal to', a.shape[0])
        #return np.array([])
    b = np.copy(a)
    if is_even:
        for ii in range(ctr):
            b[ctr-ii-1] = -a[ctr+ii]
    else:
        for ii in range(ctr):
            b[ctr-ii-1] = -a[ctr+ii+1]
    return(b)

def diff_from_antisymmetric(a):
    diff = make_antisymmetric(a) - a
    diff[np.abs(diff) < 1e-13] = 0
    return(diff)

def run_test():
  test_even = np.array([0,1,2,3,4,5,6,7,8,9])
  test_odd = np.array([0,1,2,3,4,5,6,7,8])
  test_symm_even = make_symmetric(test_even)
  test_symm_odd = make_symmetric(test_odd)
  test_symm_anti_even = make_antisymmetric(test_even)
  test_symm_anti_odd = make_antisymmetric(test_odd)
  print("check even", test_even, test_symm_even, test_symm_anti_even)
  print("check odd" test_odd, test_symm_odd, test_symm_anti_odd)
  print("is odd symmetric?", diff_from_symmetric(test_odd))
  print("is symmetric odd symmetric?", diff_from_symmetric(test_symm_odd))
  print("is symmetric odd antisymmetric?", diff_from_antisymmetric(test_symm_odd))
  print("is antisymmetric odd antisymmetric?", diff_from_antisymmetric(test_symm_anti_odd))

if __name__ == "__main__":
  run_test()
