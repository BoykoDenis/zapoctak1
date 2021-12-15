from math import sin, cos, radians


class MatrixException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class Matrix():

    def __init__(self, dims = [0, 0], matrix = [[]]):

        if dims:
            self.dims = None
        else:
            self.__set_dims()

        if matrix:
            self.matrix = matrix

        self.transposed = None
        self.rank = None
        self.inverse = None
        self.determinantVal = None
        self.RREF = None # Reduced Row Echelon Forms -> output of Gause-Jordan elimination
        self.REF = None # Row Echelon Forms -> output of Gause elimination


    def fill_with_ones(self):
        self.matrix = [[0 for _ in range(self.dims[1])] for _ in range(self.dims[0])]


    def fill_with_zeros(self):
        self.matrix = [[1 for _ in range(self.dims[1])] for _ in range(self.dims[0])]


    def id(self):
        if self.__is_square():
            self.fill_with_zeros
            for i in range(self.dims[0]):
                self.matrix[i][i] = 1

        else:
            print('Matica nie je stvorcova...')





    def multiply_with_saving(self, other: object):

        if other.get_dims()[1] == self.get_dims()[0]:
            output_matrix = []
            if not other.transposed:
                other.transpose()

            for idx, raw in enumerate(self.matrix):
                output_matrix.append([])

                for column in other.transposed:
                    output_matrix[idx].append(self.__vector_multiplication(raw, column))

        self.matrix = output_matrix
        self.__reset() # removes information about tranposed matrix, inverse matrix, rank, due to uselessness of these after multiplication with saving
        self.__set_dims() # calculates new size of the matrix




    def multiply_without_saving(self, other):
        if other.get_dims()[1] == self.get_dims[0]:
            output_matrix = []
            if not other.transposed:
                other.transpose()

            for idx, raw in enumerate(self.matrix):
                output_matrix.append([])

                for column in other.transposed:
                    output_matrix[idx].append(self.__vector_multiplication(raw, column))

        return output_matrix


    def calculate_inverse(self) -> list:
        pass


    def transpose(self):

        if self.transposed:
            return
        else:
            if self.is_matrix():
                transposed_matrix = [[] for i in range(len(self.matrix[0]))]
                for raw in self.matrix:
                    for col_idx, element in enumerate(raw):
                        transposed_matrix[col_idx].append(element)

                self.transposed = transposed_matrix


    def rank_calculate(self) -> int:
        pass


    def extend(self, matrix): # extendes matrix by vector or matrix from the right side

        if isinstance(matrix, Matrix):
            matrix = matrix.matrix
        if len(matrix) == len(self.matrix):
            for idx, row in enumerate(matrix):
                for element in row:
                    self.matrix[idx].append(element)

        else:
            raise MatrixException('Extention matrix sholud have the same amount of rows as the extended matrix')


    def solve(self, extended = True, ext_matrix = None) -> list: #returns vector

        if self.__get_determinant() == 0:
            print('Nonregular matrix... infinite solution set')
            return

        if not extended:
            if ext_matrix:
                self.extend(ext_matrix)
                self.__set_dims()

            else:
                raise MatrixException('No matrix extension provided')

        self.gaus_jordan_elimination()


        return [[self.RREF[row][col] for col in range(self.get_dims()[0], len(ext_matrix[0]))] for row in range(self.get_dims()[0])]

    def gaus_elimination(self):

        self.REF = self.matrix
        for col in range(len(self.REF[0])): # gause elimination
            pivot_pos = None
            for row in range(col, len(self.REF)):
                if self.REF[row][col] == 0:
                    continue
                else:
                    if pivot_pos == None:
                        pivot_pos = row
                        continue

                    self.REF = Matrix.add_scalar_mul2row(self.REF, pivot_pos, row, -(self.REF[row][col]/self.REF[pivot_pos][col]))


    def gaus_jordan_elimination(self):

        self.gaus_elimination()
        self.RREF = self.REF
        for row in range(len(self.RREF)): # setting all pivots to have value 1
            self.RREF = Matrix.multiply_row(self.RREF, row, 1/self.RREF[row][row])

        for col in range(self.get_dims()[0] - 1, -1, -1): # G-J eliomination
            for row in range(col - 1, -1, -1):
                self.RREF =Matrix.add_scalar_mul2row(self.RREF, col, row, -self.RREF[row][col])






    def multiply_row(matrix, index, scalar): #elementary row operation (multiplies i-th row with scalar)
        for col in range(Matrix.get_dims(matrix = matrix)[1]):
            matrix[index][col] *= scalar
        return matrix

    def add_scalar_mul2row(matrix, source_row, target_row, scalar): #elementary row operation (adds scalar multiplication of i-th row to j-th row)
        for col in range(Matrix.get_dims(matrix = matrix)[1]):
            matrix[target_row][col] += matrix[source_row][col] * scalar
        return matrix


    def row_pivot_position(self, row):
        for i in range(len(row)):
            if row[i] == 0:
                continue
            else:
                return i

        return None


    def determinant(self = None, matrix = None):

        if self:
            if self.matrix:
                matrix = self.matrix
            elif matrix:
                pass
            else:
                raise MatrixException('No matrix was providet for determinant calculation...')

        else:
            if matrix:
                pass
            else:
                raise MatrixException('No matrix was providet for determinant calculation...')


        if Matrix.is_square(matrix = matrix):

            if len(matrix) == 1:
                return matrix[0][0]

            else:
                output = 0
                for idx, elements in enumerate(matrix[0]):
                    output += matrix[0][idx] * Matrix.determinant(matrix = Matrix.submatrix(matrix, idx)) * (-1)**idx

                return output



    def submatrix(matrix, forbiden_col): # support function for determinant calculation (returns matrix of size n-1 x m-1) removes first raw and i-th column
        return [[i for col, i in enumerate(matrix[row]) if col != forbiden_col] for row in range(1, len(matrix))]


    def add(self, matrix):
        if isinstance(matrix, Matrix):
            matrix2 = matrix.matrix
        elif isinstance(list):
            pass
        else:
            raise MatrixException('Unsapported data format')

        matrix1 = self.matrix
        return [[matrix1[row][col] + matrix2[row][col] for col in range(len(matrix1[0]))] for row in range(len(matrix1))]



    def add2(matrix1, matrix2):
        if isinstance(matrix1, Matrix) and isinstance(matrix2, Matrix):
            if matrix1.get_dims() == matrix2.get_dims() and matrix1.get_dims() != [0, 0]:

                matrix1 = matrix1.matrix
                matrix2 = matrix2.matrix
                return Matrix(matrix = [[matrix1[row][col] + matrix2[row][col] for col in range(len(matrix1[0]))] for row in range(len(matrix1))])

            else:
                raise MatrixException('Additing matrices should be of the same size and not empty')

        elif isinstance(matrix1, list) and isinstance(matrix1, list):
            if len(matrix1) == len(matrix2) and len(matrix1[0]) == len(matrix2[0]) and len(matrix1[0] != 0):
                return [[matrix1[row][col] + matrix2[row][col] for col in range(len(matrix1[0]))] for row in range(len(matrix1))]

            else:
                raise MatrixException('Additing matrices should be of the same size and not empty')


        else:
            raise MatrixException('addition objects should be of the same instance (Matrix or list)')


    def get_dims(self = None, matrix = None) -> list:
        if self:
            if self.dims:
                return self.dims

            else:
                self.__set_dims()
                return self.dims

        elif matrix:
            return [len(matrix), len(matrix[0])]

        else:
            raise MatrixException('No object nor matrix were provided...')


    def get_matrix(self) -> list:
        return self.matrix


    def print_matrix(self) -> None:
        for row in self.matrix:
            for element in row:
                print(round(element, 2), end=' ')

            print(end='\n')


    def __is_vector(self) -> bool:
        return self.dims[1] == 1


    def __is2d(self) -> bool:
        return self.dims[0] == 2


    def __is3d(self) -> bool:
        return self.dims[1] == 3

    def __get_determinant(self):
        if self.determinantVal:
            return self.determinantVal
        else:
            self.determinantVal = self.determinant()
            return self.determinantVal



    def __vector_multiplication(self, vector1, vector2):
        scalar = 0
        for element_index in range(len(vector1)):
            scalar += vector1[element_index] * vector2[element_index]

        return scalar


    def is_matrix(self = None, other = None) -> bool:

        if other:
            return len(other.matrix[0]) != 0
        elif self:
            return len(self.matrix[0]) != 0
        else:
            raise MatrixException('No object nor matrix were provided...')


    def __set_dims(self):

        if self.is_matrix():
            if not self.dims:
                self.dims = [0, 0]

            self.dims[0] = len(self.matrix)
            self.dims[1] = len(self.matrix[0])



    def __reset(self):
        self.rank = None
        self.inverse = None
        self.transposed = None
        self.dims = None
        self.determinantVal = None


    def is_square(self = None, matrix = None):
        if matrix:
            return len(matrix) == len(matrix[0])

        else:
            if self.matrix:
                return self.get_dims()[0] == self.get_dims()[1]
            else:
                raise MatrixException('Unable to operate on empty matrix...')






    def rotate(self, angle):

        if self.__is_vector():
            if self.__is2d():
                self.rotation_matrix = Matrix(
                                              dims = [2, 2],
                                              matrix = [[cos(radians(angle), -sin(radians(angle)))],
                                                        [sin(radians(angle), cos(radians(angle)))]]
                                             )

                self.matrix = self.rotation_matrix.multiply_without_saving(self.matrix)



            elif self.__is3d():
                pass

            else:
                raise MatrixException('only 2d and 3d space rotation is supported')

