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

vector = zp.Matrix(matrix=[[1], [2]])

matrix2 = zp.Matrix(matrix = [[1, 2, 3, 4],
                              [8, 5, 6, 3],
                              [2, 5, 1, 0]])
matrix4 = zp.Matrix(matrix=[[0.8, -0.6, 0.4, -0.2],
                            [-0.6, 1.2, -0.8, 0.4],
                            [0.4, -0.8, 1.2, -0.6],
                            [-0.2, 0.4, -0.6, 0.8]])

print('multiplication:')
matrix1.multiply_with_saving(matrix1)
matrix1.print_matrix()
print()
print('determinant:', matrix1.determinant())
print()
print('rank:', matrix1.get_rank())
print()
print('transposition')
matrix1.transpose()
matrix1.print_matrix(which='transposed')
print()
print('addition:')
matrix1.add(matrix1)
matrix1.print_matrix()
print()
rot = 24
print('vector rotation: ', rot)
vector.rotate(rot)
vector.print_matrix()
