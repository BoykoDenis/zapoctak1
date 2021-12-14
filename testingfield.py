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


matrix3.solve()
matrix3.print_matrix()