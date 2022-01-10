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
        if self.__get_determinant() == 0:
            print('unable to find inverse for a singular matrix')
        else:
            idm = Matrix(dims=[self.get_dims()[0], self.get_dims()[0]])
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

        if self.REF:
            pass
        else:
            self.gaus_elimination()
        for row in range(self.get_dims()[0]-1, -1, -1):
            for col in range(self.get_dims()[1]-1, -1, -1):
                if self.REF[row][col] != 0:
                    self.rank = row + 1
                    return

    def get_rank(self):
        """
        get_rank [returns rank if it was previously calculated, else it will calculate it and return]


        Returns:
            [int]: [rank of the matrix]
        """
        if self.rank:
            return self.rank
        else:
            self.rank_calculate()
            return self.rank




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
                #print(self.matrix)


            else:
                raise MatrixException('No matrix extension provided')

        self.gaus_jordan_elimination()
        #print(self.RREF)
        self.matrix = [self.matrix[i][:self.get_dims()[1]] for i in range(self.get_dims()[0])]
        #print(self.matrix)
        output = [[round(self.RREF[row][col], 5) for col in range(self.get_dims()[1], self.get_dims()[1] + len(ext_matrix[0]))] for row in range(self.get_dims()[0])]
        return output

    def gaus_elimination(self):
        """
        gaus_elimination [applies Gauss elimination on the matrix of the object and saves the resulti in self.REF. Note: original matrix remains untouched]
        """
        self.REF = [[self.matrix[row][col] for col in range(len(self.matrix[0]))] for row in range(len(self.matrix))]#self.matrix.copy()
        self.sort_by_leading_zeros()
        srow, col = self.quick_zero_sort(0, 0)
        pivot_pos = srow # describes the pivot row, plays role of a flag to determine the next step of gaus elimination
        # gause elimination
        while col != None:
            for row in range(pivot_pos + 1, self.get_dims()[0]):
                if self.REF[row][col] == 0:
                    continue

                self.REF = Matrix.add_scalar_mul2row(self.REF, pivot_pos, row, -(self.REF[row][col]/self.REF[pivot_pos][col]))
            if pivot_pos + 1 <= self.dims[0] - 1 and col + 1 <= self.dims[1] - 1:
                _, col = self.quick_zero_sort(pivot_pos + 1 ,col + 1 )
                pivot_pos += 1

            else:
                break


    def gaus_jordan_elimination(self):
        """
        gaus_jordan_elimination [applies Gauss-Jordan elimination on the matrix of the object and saves the resulti in self.RREF. Note: original matrix remains untouched. Note2: this function if called calls function gaus_elimination first]
        """
        self.gaus_elimination()
        self.RREF = [[self.REF[row][col] for col in range(len(self.REF[0]))] for row in range(len(self.REF))]#self.matrix.copy()

        for row in range(self.get_dims()[0]-1, -1, -1): # setting all pivots to have value 1
            for col in range(self.get_dims()[1]-1, -1, -1):
                if self.RREF[row][col] != 0:
                    self.RREF = Matrix.multiply_row(self.RREF, row, 1/self.RREF[row][col])

                    if row != 0:
                        for gjrow in range(row-1, -1, -1):
                            self.RREF = Matrix.add_scalar_mul2row(self.RREF, row, gjrow, -self.RREF[gjrow][col])





    def sort_by_leading_zeros(self):
        """
        sort_by_leading_zeros [support function for gaus elimination: sorts rows by amount of leading zeros]

        """
        vacation = 0
        for col in range(0, len(self.REF[0])):
            for row in range(vacation, len(self.REF)):
                if self.REF[row][col] != 0:
                    self.REF[row], self.REF[vacation] = self.REF[vacation], self.REF[row]
                    vacation += 1

    def quick_zero_sort(self, r, c):
        """
        quick_zero_sort [support function for gous elimination: finds next pivot and moves it up to its desired position]

        Args:
            r ([int]): [row to start looking from]
            c ([int]): [column to start looking from]

        Returns:
            [list]: [postion of the next pivot]
        """
        for col in range(c, self.dims[1]):
            for row in range(r, self.dims[0]):
                if self.REF[row][col] != 0:
                    self.REF[row], self.REF[r] = self.REF[r], self.REF[row]
                    return r, col
        return [None, None]


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
            matrix[index][col] = round(matrix[index][col]*scalar, 10)
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
            matrix[target_row][col] = round(matrix[target_row][col] + matrix[source_row][col] * scalar, 10)
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
        if isinstance(matrix, object):
            matrix = matrix.matrix

        matrix1 = self.matrix

        if len(matrix) == len(matrix1) and len(matrix[0]) == len(matrix1[0]):
            self.matrix = [[matrix1[row][col] + matrix[row][col] for col in range(len(matrix1[0]))] for row in range(len(matrix1))]
        else:
            raise MatrixException('Matrices should be of the same size...')



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
            self ([object], optional): [matrix object, if not provided the matrix parameter will be taken]. Defaults to None.
            matrix ([list], optional): [matrix if self is not provided]. Defaults to None.

        Raises:
            MatrixException: [if self nor matrix were provided]

        Returns:
            list: [dimmensions of matrix]
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
        get_matrix [returns the matrix of the object in list]


        Returns:
            list: [self.matrix]
        """
        return self.matrix


    def print_matrix(self, rnd = 1, which = "matrix") -> None:
        """
        print_matrix [prints matrix without list breakets]

        """
        if which == 'matrix':
            if self.matrix:
                pmatrix = self.matrix
            else:
                print('no matrix in object...')
                return

        elif which == 'REF':
            if self.REF:
                pmatrix = self.REF
            else:
                print('run gaus elimination first...')
                return

        elif which == 'RREF':
            if self.RREF:
                pmatrix = self.RREF
            else:
                print('run gaus-jordan elimination first...')
                return

        elif which == 'transposed':
            if self.transposed:
                pmatrix = self.transposed
            else:
                print('run gaus transpose first...')
                return

        elif which == 'inverse':
            if self.inverse:
                pmatrix = self.inverse
            else:
                print('run inverse calculation first')
                return

        for row in pmatrix:
            for element in row:
                if element == 0:
                    element = 0
                if isinstance(element, float):
                    print(round(element, rnd), end=' ')
                else:
                    print(element, end=' ')

            print(end='\n')


    def __is_vector(self) -> bool:
        """
        __is_vector [returns if the matrix of object is vector]


        Returns:
            bool: [is or is not vector]
        """
        return self.dims[1] == 1


    def __is2d(self) -> bool:
        """
        __is2d [returns if the matrix of object has dims 2x2]

        Returns:
            bool: [is square]
        """
        return self.dims[0] == 2


    def __get_determinant(self):
        """
        __get_determinant [returns tederminant value if it was previouslu calculated, if not it will calculate it and then return]


        Returns:
            float: [determinant value]
        """
        if self.determinantVal:
            return self.determinantVal
        else:
            self.determinantVal = self.determinant()
            return self.determinantVal



    def __vector_multiplication(self, vector1, vector2):
        """
        __vector_multiplication [scalar multiplication of 2 vectors]

        Args:
            vector1 ([list]):
            vector2 ([list]):

        Returns:
            [float]: [scalar multiplication of vector1 and vector2]
        """
        scalar = 0
        for element_index in range(len(vector1)):
            scalar += vector1[element_index] * vector2[element_index]

        return scalar


    def is_matrix(self = None, other = None) -> bool:
        """
        is_matrix [returns if the matrix is empty or not]

        [extended_summary]

        Args:
            self ([object], optional): [if self is provided other will be ignored]. Defaults to None.
            other ([type], optional): [must be provided if self wasnt]. Defaults to None.

        Raises:
            MatrixException: [self nor other were provided]

        Returns:
            bool: [is empty]
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
        __set_dims [calculates the dimensions of the martix and saves the result in object]

        """
        if self.is_matrix():
            self.dims = [0, 0]
            self.dims[0] = len(self.matrix)
            self.dims[1] = len(self.matrix[0])





    def __reset(self):
        """
        __reset [sets all variables (but matrix) of the object to None]

        [extended_summary]
        """
        self.rank = None
        self.inverse = None
        self.transposed = None
        self.dims = None
        self.determinantVal = None


    def is_square(self = None, matrix = None):
        """
        is_square [returns if the matrix has equal dimmensions or not]


        Args:
            self ([object], optional): [if self is provided matrix will be ignored]. Defaults to None.
            matrix ([list], optional): [will be used if self is not provided, must be provided if self is not provided]. Defaults to None.

        Raises:
            MatrixException: [if self nor matrix were provided]

        Returns:
            [bool]:
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
        rotate [rotate vector by the angle provided]


        Args:
            angle ([float]): [angle to rotate by]

        Raises:
            MatrixException: [not supported size]
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


