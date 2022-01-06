from math import sin, cos, radians


class MatrixException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class Matrix():

    def __init__(self, dims = None, matrix = [[]]) -> None: # matrix initialization function

        self.dims = dims
        self.matrix = matrix

        if dims:
            self.dims = dims
        else:
            self.__set_dims()


        self.transposed = None
        self.rank = None
        self.inverse = None
        self.determinantVal = None
        self.RREF = None # Reduced Row Echelon Forms -> output of Gause-Jordan elimination
        self.REF = None # Row Echelon Forms -> output of Gause elimination


    def fill_with_ones(self) -> None:
        """
        fill_with_ones [fills the matrix object with zeros, basing o its dimmensions]

        [extended_summary]
        """
        self.matrix = [[1 for _ in range(self.get_dims()[1])] for _ in range(self.get_dims()[0])]



    def fill_with_zeros(self) -> None:
        """
        fill_with_zeros [fills the matrix object with zeros basing on its dimmensions]

        [extended_summary]
        """
        self.matrix = [[0 for _ in range(self.get_dims()[1])] for _ in range(self.get_dims()[0])]


    def id(self) -> None:
        """
        id [makes identity matrix from existing matrix basing on its dimensions (dims).
        Note: only if matrix is square matrix]

        [extended_summary]
        """
        if self.is_square():
            self.fill_with_zeros()
            for i in range(self.get_dims()[0]):
                self.matrix[i][i] = 1

        else:
            print('Matica nie je stvorcova...')





    def multiply_with_saving(self, other: object) -> None:
        """
        multiply_with_saving [multiply the matrix object by another matrix object, than repalce existing matrix in object with the result
        Note: dimensions must be fir for multiplications]

        Args:
            other (object): [matrix object to multiply by]
        """

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




    def multiply_without_saving(self, other) -> list:
        """
        multiply_without_saving [multiply the matrix object by another matrix object, than retur the result without affecting existing matrices
        Note: dimensions must be fir for multiplications]

        Args:
            other ([type]): [matrix to multiply by]

        Returns:
            list: [result in form of list of lists]
        """
        if other.get_dims()[1] == self.get_dims()[0]:
            output_matrix = []
            if not other.transposed:
                other.transpose()

            for idx, raw in enumerate(self.matrix):
                output_matrix.append([])

                for column in other.transposed:
                    output_matrix[idx].append(self.__vector_multiplication(raw, column))

        return output_matrix




    def calculate_inverse(self) -> list:
        """
        calculate_inverse [calculates inverse matrix for the given object matrix and saves it to the self.inverse variable of the given matrix object]

        Returns:
            list: [list of list containing inverse matrix]
        """
        idm = Matrix(dims=[len(self.matrix), len(self.matrix)])
        idm.id()
        self.inverse = self.solve(extended=False, ext_matrix = idm.get_matrix())
        return self.inverse


    def transpose(self):
        """
        transpose [transpose the matrix of the object and saves transposed matrix into self.transposed. Note: original matrix remains untouched]
        """
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
        """
        extend [extends the matrix of the object with provided matrix or vector(adds it to the right side of the matrix)]

        Args:
            matrix ([type]): [matrix or vector to extend matrix with]

        Raises:
            MatrixException: [extention matrix should have the same amount of rows as the extended matrix]
        """
        if isinstance(matrix, Matrix):
            matrix = matrix.matrix
        if len(matrix) == len(self.matrix):
            for idx, row in enumerate(matrix):
                for element in row:
                    self.matrix[idx].append(element)

        else:
            raise MatrixException('Extention matrix sholud have the same amount of rows as the extended matrix')


    def solve(self, extended = True, ext_matrix = None) -> list: #returns vector
        """
        solve [solves the system of equations(in matrix) using Gaus-Jordan elimination and returns the vector/matrix of solutions]

        Args:
            extended (bool, optional): [defines if the matrix of object was previously extended. If not the matrix will be extended with the matrix provided in ext_matrix]. Defaults to True.
            ext_matrix ([list], optional): [defines a matrix to extend the matrix of object with]. Defaults to None.

        Raises:
            MatrixException: [no extension matrix was provided/ no solutions for equations were provided]

        Returns:
            list: [vector/matrix of the solution (the extended part after G-J elimination)]
        """
        if self.__get_determinant() == 0:
            print('Nonregular matrix... infinite solution set')
            return

        if not extended:
            if ext_matrix:
                self.__set_dims()
                self.extend(ext_matrix)


            else:
                raise MatrixException('No matrix extension provided')

        self.gaus_jordan_elimination()

        self.matrix = [self.matrix[i][:self.get_dims()[1]] for i in range(self.get_dims()[0])]
        output = [[round(self.RREF[row][col], 5) for col in range(self.get_dims()[1], self.get_dims()[1] + len(ext_matrix[0]))] for row in range(self.get_dims()[0])]
        return output

    def gaus_elimination(self):
        """
        gaus_elimination [applies Gauss elimination on the matrix of the object and saves the resulti in self.REF. Note: original matrix remains untouched]
        """

        self.REF = [[self.matrix[row][col] for col in range(len(self.matrix[0]))] for row in range(len(self.matrix))]#self.matrix.copy()
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
        """
        gaus_jordan_elimination [applies Gauss-Jordan elimination on the matrix of the object and saves the resulti in self.RREF. Note: original matrix remains untouched. Note2: this function if called calls function gaus_elimination first]
        """
        self.gaus_elimination()
        self.RREF = [[self.REF[row][col] for col in range(len(self.REF[0]))] for row in range(len(self.REF))]#self.matrix.copy()
        for row in range(len(self.RREF)): # setting all pivots to have value 1
            self.RREF = Matrix.multiply_row(self.RREF, row, 1/self.RREF[row][row])

        for col in range(self.get_dims()[0] - 1, -1, -1): # G-J eliomination
            for row in range(col - 1, -1, -1):
                self.RREF = Matrix.add_scalar_mul2row(self.RREF, col, row, -self.RREF[row][col])






    def multiply_row(matrix, index, scalar): #elementary row operation (multiplies i-th row with scalar)
        """
        multiply_row [elementary row operation: multiplies row with provided scalar]

        Args:
            matrix ([type]): [matrix to operate on can be object or list]
            index ([type]): [index of the row to opperate on]
            scalar ([type]): [scalar to multiply the row with]

        Returns:
            [object or list]: [returns matrix object if the input was matrix object, returns list if the input was list]
        """
        if isinstance(matrix, Matrix):
            matrix = matrix.matrix
        for col in range(Matrix.get_dims(matrix = matrix)[1]):
            matrix[index][col] *= scalar
        if isinstance(matrix, Matrix):
            return Matrix(matrix = matrix)
        else:
            return matrix


    def add_scalar_mul2row(matrix, source_row, target_row, scalar): #elementary row operation (adds scalar multiplication of i-th row to j-th row)
        """
        add_scalar_mul2row [elementary row operation: adds scalar multiplication of source row to the target row]

        Args:
            matrix ([object or list]): [matrix to opperate on]
            source_row ([int]): [index of the source row]
            target_row ([int]): [index of the target row]
            scalar ([int]): [scalar to multiply the source row with]

        Returns:
            [object or list]: [returns matrix object if the input was matrix object, returns list if the input was list]
        """
        if isinstance(matrix, Matrix):
            matrix = matrix.matrix
        for col in range(Matrix.get_dims(matrix = matrix)[1]):
            matrix[target_row][col] += matrix[source_row][col] * scalar
        if isinstance(matrix, Matrix):
            return Matrix(matrix = matrix)
        else:
            return matrix



    def determinant(self = None, matrix = None):
        """
        determinant [calculates determinant of the given matrix. Note: the matrix should be square]

        Args:
            self ([object], optional): [if self is provided other parameters are ignored]. Defaults to None.
            matrix ([list or object], optional): [if self is None matrix parameter will be taken as the matrix to opperate on]. Defaults to None.

        Raises:
            MatrixException: [if self nor matrix parameters were provided]

        Returns:
            [float]: [the determinant of the matrix]
        """
        if self:
            if self.matrix != [[]]:
                matrix = self.matrix
            elif matrix:
                pass
            else:
                raise MatrixException('No matrix was provided for determinant calculation...')

        else:
            if matrix:
                pass
            else:
                raise MatrixException('No matrix was provided for determinant calculation...')


        if Matrix.is_square(matrix = matrix):

            if len(matrix) == 1:
                return matrix[0][0]

            else:
                output = 0
                for idx, elements in enumerate(matrix[0]):
                    output += matrix[0][idx] * Matrix.determinant(matrix = Matrix.submatrix(matrix, idx)) * (-1)**idx

                if self:
                    self.determinantVal = output
                return output



    def submatrix(matrix, forbiden_col): # support function for determinant calculation (returns matrix of size n-1 x m-1) removes first raw and i-th column
        """
        submatrix [this function is a support function for determinant calculation]


        Args:
            matrix ([list]): [matirix to opperate on]
            forbiden_col ([int]): [index of column to remove]

        Returns:
            [matrix]: [matrix without the first row and column with index of forbiden_column]
        """
        return [[i for col, i in enumerate(matrix[row]) if col != forbiden_col] for row in range(1, len(matrix))]


    def add(self, matrix):
        """
        add [adds given matrix to the matrix of object]

        Args:
            matrix ([object or list]): [matrix to add]

        Raises:
            MatrixException: [if the input is not matrix object or list]

        Returns:
            [list]: [output of the addition]
        """
        if isinstance(matrix, Matrix):
            matrix2 = matrix.matrix
        elif isinstance(list):
            pass
        else:
            raise MatrixException('Unsapported data format')

        matrix1 = self.matrix
        return [[matrix1[row][col] + matrix2[row][col] for col in range(len(matrix1[0]))] for row in range(len(matrix1))]



    def add2(matrix1, matrix2):
        """
        add [adds 2 matrices of the same size]

        Args:
            matrix1 ([object or list]): [matrix to add]
            matrix2 ([object or list]): [matrix to add. Note should be of the same type as matrix1]

        Raises:
            MatrixException: [if the matrices are of the different size or empty]

        Returns:
            [list]: [output of the addition]
        """
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
        """
        get_dims [returns dimmensions of the matrix [rows, columns]]


        Args:
            self ([object], optional): [description]. Defaults to None.
            matrix ([list], optional): [description]. Defaults to None.

        Raises:
            MatrixException: [description]

        Returns:
            list: [description]
        """
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
        """
        get_matrix [summary]

        [extended_summary]

        Returns:
            list: [description]
        """
        return self.matrix


    def print_matrix(self) -> None:
        """
        print_matrix [summary]

        [extended_summary]
        """
        for row in self.matrix:
            for element in row:
                print(round(element, 1), end=' ')

            print(end='\n')


    def __is_vector(self) -> bool:
        """
        __is_vector [summary]

        [extended_summary]

        Returns:
            bool: [description]
        """
        return self.dims[1] == 1


    def __is2d(self) -> bool:
        """
        __is2d [summary]

        [extended_summary]

        Returns:
            bool: [description]
        """
        return self.dims[0] == 2


    def __get_determinant(self):
        """
        __get_determinant [summary]

        [extended_summary]

        Returns:
            [type]: [description]
        """
        if self.determinantVal:
            return self.determinantVal
        else:
            self.determinantVal = self.determinant()
            return self.determinantVal



    def __vector_multiplication(self, vector1, vector2):
        """
        __vector_multiplication [summary]

        [extended_summary]

        Args:
            vector1 ([type]): [description]
            vector2 ([type]): [description]

        Returns:
            [type]: [description]
        """
        scalar = 0
        for element_index in range(len(vector1)):
            scalar += vector1[element_index] * vector2[element_index]

        return scalar


    def is_matrix(self = None, other = None) -> bool:
        """
        is_matrix [summary]

        [extended_summary]

        Args:
            self ([type], optional): [description]. Defaults to None.
            other ([type], optional): [description]. Defaults to None.

        Raises:
            MatrixException: [description]

        Returns:
            bool: [description]
        """
        if other:
            if isinstance(other, Matrix):
                return len(other.get_matrix()[0]) != 0
            else:
                return len(other[0]) != 0
        elif self:
            return len(self.get_matrix()[0]) != 0
        else:
            raise MatrixException('No object nor matrix were provided...')


    def __set_dims(self):
        """
        __set_dims [summary]

        [extended_summary]
        """
        if self.is_matrix():
            self.dims = [0, 0]
            self.dims[0] = len(self.matrix)
            self.dims[1] = len(self.matrix[0])





    def __reset(self):
        """
        __reset [summary]

        [extended_summary]
        """
        self.rank = None
        self.inverse = None
        self.transposed = None
        self.dims = None
        self.determinantVal = None


    def is_square(self = None, matrix = None):
        """
        is_square [summary]

        [extended_summary]

        Args:
            self ([type], optional): [description]. Defaults to None.
            matrix ([type], optional): [description]. Defaults to None.

        Raises:
            MatrixException: [description]

        Returns:
            [type]: [description]
        """
        if matrix:
            return len(matrix) == len(matrix[0])

        else:
            if self.matrix:
                return self.get_dims()[0] == self.get_dims()[1]
            else:
                raise MatrixException('Unable to operate on empty matrix...')






    def rotate(self, angle):
        """
        rotate [summary]

        [extended_summary]

        Args:
            angle ([type]): [description]

        Raises:
            MatrixException: [description]
        """

        if self.__is_vector():
            if self.__is2d():
                self.rotation_matrix = Matrix(
                                              dims = [2, 2],
                                              matrix = [[cos(radians(angle), -sin(radians(angle)))],
                                                        [sin(radians(angle), cos(radians(angle)))]]
                                             )

                self.matrix = self.rotation_matrix.multiply_without_saving(self.matrix)
                self.__reset()


            else:
                raise MatrixException('only 2d space rotation is supported')


