import zapoctovyProgram as zp

matrix1 = zp.Matrix(matrix = [[1, 2, 3, 1, 3, 4],
                              [4, 6, 6, 3, 2, 1],
                              [0, 8, 9, 2, 3, 4],
                              [0, 4, 1, 6, 3, 0],
                              [0, 2, 4, 5, 0, 4],
                              [0, 3, 7, 8, 5, 4]])

matrix3 = zp.Matrix(matrix=[[2, 1, 0, 0],
                            [1, 2, 1, 0],
                            [0, 1, 2, 1],
                            [0, 0, 1, 2]])

matrix2 = zp.Matrix(matrix = [[1, 2, 3, 4],
                              [8, 5, 6, 3],
                              [2, 5, 1, 0]])
matrix4 = zp.Matrix(matrix=[[0.8, -0.6, 0.4, -0.2],
                            [-0.6, 1.2, -0.8, 0.4],
                            [0.4, -0.8, 1.2, -0.6],
                            [-0.2, 0.4, -0.6, 0.8]])

matrix3.multiply_with_saving(matrix4)
matrix3.print_matrix()


matrix3.solve(extended=False, ext_matrix=[[1, 0, 0, 0],
                                          [0, 1, 0, 0],
                                          [0, 0, 1, 0],
                                          [0, 0, 0, 1]])
matrix3.print_matrix()