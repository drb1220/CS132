import numpy as np
import warnings

def swapRows(A, i, j):
    """
    interchange two rows of A
    operates on A in place
    """
    tmp = A[i].copy()
    A[i] = A[j]
    A[j] = tmp

def relError(a, b):
    """
    compute the relative error of a and b
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            return np.abs(a-b)/np.max(np.abs(np.array([a, b])))
        except:
            return 0.0

def rowReduce(A, i, j, pivot):
    """
    reduce row j using row i with pivot pivot, in matrix A
    operates on A in place
    """
    factor = A[j][pivot] / A[i][pivot]
    for k in range(len(A[j])):
        if np.isclose(A[j][k], factor * A[i][k]):
            A[j][k] = 0.0
        else:
            A[j][k] = A[j][k] - factor * A[i][k]


# stage 1 (forward elimination)
def forwardElimination(B):
    """
    Return the row echelon form of B
    """
    A = B.copy().astype(float)
    m, n = np.shape(A)
    for i in range(m-1):
        # Let lefmostNonZeroCol be the position of the leftmost nonzero value 
        # in row i or any row below it 
        leftmostNonZeroRow = m
        leftmostNonZeroCol = n
        ## for each row below row i (including row i)
        for h in range(i,m):
            ## search, starting from the left, for the first nonzero
            for k in range(i,n):
                if (A[h][k] != 0.0) and (k < leftmostNonZeroCol):
                    leftmostNonZeroRow = h
                    leftmostNonZeroCol = k
                    break
        # if there is no such position, stop
        if leftmostNonZeroRow == m:
            break
        # If the leftmostNonZeroCol in row i is zero, swap this row 
        # with a row below it
        # to make that position nonzero. This creates a pivot in that position.
        if (leftmostNonZeroRow > i):
            swapRows(A, leftmostNonZeroRow, i)
        # Use row reduction operations to create zeros in all positions 
        # below the pivot.
        for h in range(i+1,m):
            rowReduce(A, i, h, leftmostNonZeroCol)
    return A

#################### 

def rank(A):
    """
    finds the rank of a given matrix by counting the number of non-zero rows
    returns the rank as an integer
    """
    m, n = np.shape(A)
    nz = np.transpose(np.nonzero(A))
    count = 1
    for i in (range(1, int(nz.size/2))):
        if nz[i,0] != nz[i-1,0]:
            count += 1
    return count

def inconsistentSystem(B):
    """
    checks if B is inconsistent by comparing the ranks of the augmented matrix and the coefficient matrix
    returns True or False
    """
    A = B.copy().astype(float)
    ACo = A[:,:-1]
    nzA = np.transpose(np.nonzero(A))
    nzACount = np.size(nzA)/2
    nzACo = np.transpose(np.nonzero(ACo))
    nzACoCount = np.size(nzACo)/2
    return rank(A) != rank(ACo)
    

def backsubstitution(B):
    """
    returns the reduced row echelon form of B
    """
    A = B.copy().astype(float)
    m, n = np.shape(A)
    nz = np.transpose(np.nonzero(A))
    nzLen = int(np.size(nz)/2)
    for i in range(nzLen):
        if i != 0:
            if nz[i,0] == nz[i-1,0]:
                continue
        multiple = 1/A[nz[i,0],nz[i,1]]
        A[nz[i,0]] = multiple * A[nz[i,0]]
    for i in reversed(range(m)):
        leftmostNonZero = None
        for k in range(n):
            if A[i,k] != 0.0:
                leftmostNonZero = k
                break
        if leftmostNonZero == None:
            continue
        for j in range(i):
            rowReduce(A,i,j,k)
            
    return A

A = np.loadtxt('/Users/davidbunger/Desktop/CS132/hw2/h2m6.txt')
AEchelon = forwardElimination(A)
if (not inconsistentSystem(AEchelon)):
    AReducedEchelon = backsubstitution(AEchelon)
print(AReducedEchelon)
print(np.array_equal([0],[-0]))

