import numpy as np

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

def rank(A):
    m, n = np.shape(A)
    nz = np.transpose(np.nonzero(A))
    count = 1
    for i in (range(1, int(nz.size/2))):
        if nz[i,0] != nz[i-1,0]:
            count += 1
    return count

def inconsistentSystem(B):
    A = B.copy().astype(float)
    ACo = A[:,:-1]
    nzA = np.transpose(np.nonzero(A))
    nzACount = np.size(nzA)/2
    nzACo = np.transpose(np.nonzero(ACo))
    nzACoCount = np.size(nzACo)/2
    return rank(A) == rank(ACo)
    

def backSubstitution(B):
    A = B.copy().astype(float)
    m, n = np.shape(A)
    nz = np.transpose(np.nonzero(A))
    nzLen = int(np.size(nz)/2)
    for i in range(nzLen):
        if i != 0:
            if nz[i,0] == nz[i-1,0]:
                continue
        multiple = 1/A[nz[i,0],nz[i,1]]
        print(multiple)
        A[nz[i,0]] = multiple * A[nz[i,0]]

    for i in reversed(range(m)):
        for j in range(i):
           A[j] = A[j] - A[i]

    print(A)



        




# test = np.array([[1.0,1.0,2.0],[3.0,3.0,6.0]])
# test = np.array([[1.0,1.0,3.0],[4.0,4.0,10.0]])
test = np.array([[1.0,1.0,1.0,10.0],[1.0,2.0,1.0,12.0],[2.0,2.0,2.0,20.0]])
feTest = forwardElimination(test)
print(feTest)
# print(inconsistentSystem(feTest))
backSubstitution(feTest)