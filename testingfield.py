import zapoctovyProgram as zp

matrix1 = zp.Matrix(matrix = [[1, 2, 3, 1, 3, 4],
                              [4, 6, 6, 3, 2, 1],
                              [0, 8, 9, 2, 3, 4],
                              [0, 4, 1, 6, 3, 0],
                              [0, 2, 4, 5, 0, 4],
                              [0, 3, 7, 8, 5, 4]])

matrix3 = zp.Matrix(matrix=[[2, 1, -1],
                            [1, 1, -1],
                            [0, 0, 1]])

matrix2 = zp.Matrix(matrix = [[1, 2, 3, 4],
                              [8, 5, 6, 3],
                              [2, 5, 1, 0]])
matrix4 = zp.Matrix(matrix=[[0.8, -0.6, 0.4, -0.2],
                            [-0.6, 1.2, -0.8, 0.4],
                            [0.4, -0.8, 1.2, -0.6],
                            [-0.2, 0.4, -0.6, 0.8]])
#print(matrix3.matrix)
inverse = matrix1.calculate_inverse()
##matrix3.print_matrix()
#print(inverse)
matrix1inv = zp.Matrix(matrix= inverse)
matrix1inv.print_matrix()
matrix1.print_matrix()
matrix1.multiply_with_saving(matrix1inv)
matrix1.print_matrix()
#matrix3inv.print_matrix()
#out = zp.Matrix.multiply_without_saving(matrix3, matrix3inv)
#print(out)