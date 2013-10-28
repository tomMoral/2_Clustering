import sys.stdout as out
import numpy as np

def print_matrix(mat):
    mat = np.array(mat)
    (R,C) = mat.shape
    out.write("\\begin{pmatrix}")
    for i in range(R):
        for j in range(C-1):
            out.write("{}&".format(mat[i,j])
        print "{}\\\\".format(mat[i, C-1])
    print "\\end{pmatrix}"


