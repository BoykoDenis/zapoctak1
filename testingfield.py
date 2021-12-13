import zapoctovyProgram as zp

matrix1 = zp.Matrix(matrix = [[1, 2, 3, 1, 3, 4],
                              [4, 6, 6, 3, 2, 1],
                              [7, 8, 9, 2, 3, 4],
                              [8, 4, 1, 6, 3, 0],
                              [1, 2, 4, 5, 0, 4],
                              [1, 3, 7, 8, 5, 4]])

matrix2 = zp.Matrix(matrix = [[1, 2, 3, 4],
                              [8, 5, 6, 3],
                              [2, 5, 1, 0]])

#matrix1.transpose()
#print(matrix1.transposed)

#output = matrix1.multiply_without_saving(matrix2)
#print(output)

print(matrix1.determinant())