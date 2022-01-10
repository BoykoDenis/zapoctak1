import zapoctovyProgram as zp

matrix1 = zp.Matrix(matrix = [[1, 2, 3, 1, 3, 4],
                              [0, 0, 6, -3, 0, 0],
                              [0, 1, 9, 2, 3, 4],
                              [5, 0, 3, 0, 20, 0],
                              [0, 1, 0, 0, 0, 4],
                              [2, 0, 0, 8, 0, 4]])

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
zp.Matrix.add2(matrix1, matrix1).print_matrix()
#matrix1.print_matrix(which='matrix')
#inverse = matrix1.gaus_jordan_elimination()
#matrix1.print_matrix(1, which='REF')
#print()
#matrix1.print_matrix(1, which='RREF')
#matrix1.calculate_inverse()
#matrix1.print_matrix(1, which='inverse')
#print(matrix1.get_rank())
#print(inverse)
#matrix1inv = zp.Matrix(matrix= inverse)
#matrix1inv.print_matrix()
#matrix1.print_matrix()
#matrix1.multiply_with_saving(matrix1inv)
#matrix1.print_matrix()
#matrix3inv.print_matrix()
#out = zp.Matrix.multiply_without_saving(matrix3, matrix3inv)
#print(out)
#matrix1.calculate_inverse()