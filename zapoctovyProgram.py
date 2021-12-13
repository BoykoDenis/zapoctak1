from math import sin, cos, radians


class MatrixException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class Matrix():

    def __init__(self, dims = [0, 0], matrix = [[]]):

        if dims:
            self.dims = dims
        else:
            self.__set_dims()

        if matrix:
            self.matrix = matrix

        self.transposed = None
        self.rank = None
        self.inverse = None
        self.determinantVal = None


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





    def multiply_with_saving(self, other):
        if other.dims[1] == self.dims[0]:
            output_matrix = []
            if not other.transposed:
                other.transpose()

            for idx, raw in enumerate(self.matrix):
                output_matrix.append([])

                for column in self.transposed:
                    output_matrix[idx].append(self.__vector_multiplication(raw, column))

        self.matrix = output_matrix
        self.__reset() # removes information about tranposed matrix, inverse matrix, rank, due to uselessness of these after multiplication with saving
        self.__set_dims() # calculates new size of the matrix




    def multiply_without_saving(self, other):
        if other.dims[1] == self.dims[0]:
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
            self.__is_matrix()
            transposed_matrix = [[] for i in range(len(self.matrix[0]))]
            for raw in self.matrix:
                for col_idx, element in enumerate(raw):
                    transposed_matrix[col_idx].append(element)

            self.transposed = transposed_matrix


    def rank_calculate(self) -> int:
        pass


    def extend(self, vector: list):
        pass


    def solve(self, extended = True, vector = None) -> list: #returns vector

        if not extended:
            self.extend(vector)

        pass

    def determinant(self = None, matrix = None):

        if not self.determinantVal:
            if not matrix:
                matrix = self.matrix

            if Matrix.is_square(matrix = matrix):

                if len(matrix) == 1:
                    return matrix[0][0]

                else:
                    output = 0
                    for idx, elements in enumerate(matrix[0]):
                        output += matrix[0][idx] * Matrix.determinant(matrix = Matrix.submatrix(matrix, idx)) * (-1)**idx

                    return output

        else:
            return self.determinantVal

    def submatrix(matrix, forbiden_col):
        return [[i for col, i in enumerate(matrix[row]) if col != forbiden_col] for row in range(1, len(matrix))]


    def get_dims(self) -> list:
        return self.dims


    def get_matrix(self) -> list:
        return self.matrix


    def print_matrix(self):
        pass


    def __is_vector(self) -> bool:
        return self.dims[1] == 1


    def __is2d(self):
        return self.dims[0] == 2


    def __is3d(self):
        return self.dims[1] == 3


    def __vector_multiplication(self, vector1, vector2):
        scalar = 0
        for element_index in range(len(vector1)):
            scalar += vector1[element_index] * vector2[element_index]

        return scalar


    def __is_matrix(self, other = None) -> bool:

        if other:
            return len(other.matrix[0]) != 0
        else:
            return len(self.matrix[0]) != 0


    def __set_dims(self):

        if self.__is_matrix():
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
                return self.dims[0] == self.dims[1]
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
                print('not supported')

