import zapoctovyProgram as zp

matrix1 = zp.Matrix(matrix = [[1, 2, 3],
                              [4, 5, 6],
                              [7, 8, 9],
                              [1, 4, 8]])

matrix2 = zp.Matrix(matrix = [[1, 2, 3, 4],
                              [8, 5, 6, 3],
                              [2, 5, 1, 0]])

matrix1.transpose()
print(matrix1.transposed)

output = matrix1.multiply_without_saving(matrix2)
print(output)