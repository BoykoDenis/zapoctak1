Zapoctovy program (kniznica operaci na maticach)

Majeme classu Matrix:

Pri vytvarane objektu berie nepovinne premenne:
    - dims -> list s 2 prvkami [pocet riadkov, pocet stlpcov],
              defaultna hodnota = None

    - matrix -> 2d list(zaznam zaznamov) - zaznam riadkov, defaultna hodnota = [[]]

Metody:

    - fill_with_ones(self)
        - input : Matrix object
        - Popis : naplni maticu jednickami podla premennej "dims"
        - Output : No

    - fill_with_zeros(self)
        - input : Matrix object
        - Popis : naplni maticu nulami podla premennej "dims"
        - Output : No

    - id(self)
        - input : Matrix object
        - Popis : Ak matica je stvorcova(podla premennej dims) spravi z nej jednotkovu maticu
        - Output : No

    - multiply_with_saving(self, other)
        - input : Matrix object(self), Matrix object(other)
        - Popis : vynasobi maticu self maticou other a vysledok ulozi do matice self
        - Note : po vynasobeni na maticu self bude pouzita privatna metoda __reset() (Popis dole)
        - Output : No

    - multiply_without_saving(self, other)
        - Note: sprava sa rovnako ako funkcia multiply_with_saving()
        - Neuklada vysledok do self
        - Output : vysledok nasobenia

    - calculate_inverse(self)
        - input : Matrix object
        - Popis : vypocita inverznu maticu(ak je to mozne(cili determinant != 0)) v ralnych cislach(zaokruhlene 2 desatinne miesta)
        - Inverzna matica sa ulozi do premennej self.inverse
        - Output : Matrix -> list

    - transpose(self)
        - input : Matrix object
        - Popis : spravi stransponovanu maticu a ulozi ju do premennej self.tranposed
        - Output : transponovana Matica -> list

    - rank_calculate(self)
        - input : Matrix object
        - Popis : vypocita rank matice a ulozi ho do promennej self.rank
        - Output : rank matice -> list

    - extend(self, matrix)
        - input : Matrix object, Matrix object alebo list
        - Popis : rozsili maticu o inu maticu alebo vektor
        - Output : No

    - solve(self, extended = True, ext_matrix = None)
        - input : Matrix object,
                  extended = True -> definuje ci matica je rozsirena True ak hej False ak nie
                  ext_matrix = None -> v pripade ak extended = False a ext_matrix je definovana potom sa matica rozsiri o tu maticu
        - Popis : vypocita sustavu pomocou G-J eliminacii (ak riesenie je jednoznacne)
        - Output : matica riesenia

    - gaus_elimination(self)
        - input : Matrix object
        - Popis : spravis G eliminaciu a ulozi ju do self.REF
        - output : No

    - gaus_jordan_elimination(self)
        - input : Matrix object
        - Popis : spravi G-J eliminaciu a ulozi vysledok do self.RREF
        - Output : No

    - multiply_row(matrix, index, scalar)
        - input : matrix -> Matrix object or list
                  index -> row to operate on
                  scalar -> scalar to multiply row with
        - Popis : Vynasobi i-ty riadok skalarem
        - Output : matica typu list alebo object podla toho akeho typu bol vstup

    - add_scalar_mul2row(matrix, source_row, target_row, scalar)
        - input : matrix -> Matrix object or list
                  source_row -> index riadku ktory pripocitavame
                  target_row -> index riadku ku ktoremu pripocitavame
                  scalar -> scalar to multiply row with
        - Popis : Vynasobi i-ty riadok skalarem a pripocita ho ku j-temu
        - Output : matica typu list alebo object podla toho akeho typu bol vstup

    - determinant(self = None, matrix = None)
        - input : Matrix object, default = None
                 matrix typu list, default = None
        - Popis : Vypocita a vrati determinant matice a ulozi ho do premennej self.determinantVal.
        - Note : ak self = None, sa zoberie hodnota matrix. Ak matrix == None
                 dostaneme No matrix provided error
        - output : hodnota determinantu

    - submatrix(matrix, forbiden_col)
        - input : matrix typu list
                  forbiden_col -> index stlpcu ktory odstranujeme
        - Popis : Pomocna funkcia pre pocitanie determinantu. Odstrihne horny riadok a i-ty stlpec
                  tym padom vrati maticu velkosti n-1 x n-1
        - output : list

    - add(self, matrix)
        - input : Matrix object
                  matrix -> list or Matrix object
        - Popis : pripocita k matici objektu maticu matrix a ulozi ju do matici objektu
        - Output : list

    - add2(matrix1, matrix2):
        - input : matrix1 a matrix2 typu Object alebo list(obidvaja)
        - Popis : pripocita matrix1 ku matrix2
        - Output : Matrix object

    - get_dims(self = None, matrix = None)
        - input : self -> Matrix object, default = None
                 matrix -> list, default = None
        - Popis : vrati velkost matice, ak self != None zoberie dimensie matice self.matrix
                  pre self == None, zoberie maticu matrix,
                  ak matrix == None, raisne error "No matrix provided"
        - output: list velkosti 2 [pocet riadkov, pocet stlbcov]

    - get_matrix(self)
        - input : Matrix object
        - Output : 2d list matice

    - print_matrix(self)
        - Popis : vypise maticu

    - is_matrix(self = None, other = None)
        - input : Matrix object, default = None
                  other -> Matrix object, default = None
        - Popis : vrati ci matica je neprazdna
        - Output : boolS

    - is_square(self = None, matrix = None)
        - Popis : zkontroluje ci matica je stvorcova
        - Output : bool

    - rotate(self, angle)
        - input : angle -> int/float
        - Popis : otoci maticu alebo vektor o dany uhol
        - Output : No

Privatne funkcie:

    -__is_vector(self)
        - Popis : vrati ci matica je vektor
        - Output : bool

    -__is2d(self)
        - Popis : vrati ci matica je 2 rozmerna
        - Output : bool

    -__get_determinant(self)
        - Popis : vrati determinant ak uz je vypocitany, ak nie zavola vypocet
                  determinant() a vypocita ho a pak vrati
        - Output : int/float

    -__vector_multiplication(self, vector1, vector2)
        - input : vector1 a vector2 typu list (rovnakej velkosti)
        - Popis : urobi skalarny sucin vector1 a vector2
        - output : int/float

    -__set_dims(self)
        - Popis : nastavi dimensie objektu Matrix ak neboli nastavene predtym

    -__reset(self)
        -Popis : vymaze hodnoty vsetkych premennych matice
                 pouziva sa po operaciach na matici ako napriklad nasobenie matic