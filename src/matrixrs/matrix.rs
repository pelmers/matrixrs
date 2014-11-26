use std::num;
use std::num::{Zero, NumCast, Num};

/// Matrix -- Generic 2D Matrix implementation in Rust.
#[deriving(Show)]
pub struct Matrix<T> {
    /// Number of rows
    m : uint,
    /// Number of columns
    n : uint,
    /// Table (vector of vector) of data values in the matrix
    data : Vec<Vec<T>>
}

fn approx_eq(a: f64, b: f64, threshold: f64) -> bool {
    //! Return whether two floats a and b are within threshold of each other.
    //! i.e. | a - b | <= threshold
    (if a > b { a-b } else { b-a }) <= threshold
}

fn abs_diff<T:NumCast>(a : T, b : T) -> f64 {
    //! Return the difference in the absolute values of a and b.
    //! i.e. |a| - |b|
    let a64 = a.to_f64().unwrap_or_else(|| 0.0);
    let b64 = b.to_f64().unwrap_or_else(|| 0.0);
    let abs_a = if a64 > 0.0 { a64 } else { -a64 };
    let abs_b = if b64 > 0.0 { b64 } else { -b64 };
    abs_a - abs_b
}

impl<T> Matrix<T> {
    pub fn new(data : Vec<Vec<T>>) -> Matrix<T> {
        //! Create a new matrix using the given data.
        //! Asserts that the number of columns is consistent.
        let rows = data.len();
        let cols = data[0].len();
        for row in data.iter() {
            assert_eq!(row.len(), cols);
        }
        Matrix{m: rows, n: cols, data: data}
    }
    pub fn from_fn(m : uint, n : uint, func : |uint, uint| -> T) -> Matrix<T> {
        //! Create an m-by-n matrix by using a function func
        //! that returns a number given row and column.
        let mut data = Vec::with_capacity(m);
        for i in range(0, m) {
            data.push(Vec::from_fn(n, |j:uint| -> T { func(i, j) }));
        }
        Matrix{m:m, n:n, data:data}
    }
    pub fn size(&self) -> (uint, uint) {
        //! Return the size of a Matrix as row, column.
        (self.m, self.n)
    }
}

impl<T:Clone> Matrix<T> {
    pub fn from_elem(m : uint, n : uint, val : T) -> Matrix<T> {
        //! Create an m-by-n matrix, where each element is a clone of elem.
        let mut data = Vec::with_capacity(m);
        for _ in range(0, m) {
            data.push(Vec::from_elem(n, val.clone()));
        }
        Matrix{m:m, n:n, data:data}
    }
    pub fn at(&self, row : uint, col : uint) -> T {
        //! Return a clone of the element at row, col.
        self.data[row][col].clone()
    }
    pub fn row(&self, row : uint) -> Matrix<T> {
        //! Return specified row from an MxN matrix as a 1xN matrix.
        Matrix{m: 1, n:self.n, data: vec![self.data[row].clone()]}
    }
    pub fn col(&self, col : uint) -> Matrix<T> {
        //! Return specified col from an MxN matrix as an Mx1 matrix.
        let mut c = Vec::with_capacity(self.m);
        for i in range(0, self.m) {
            c.push(vec![self.at(i, col)]);
        }
        Matrix{m: self.m, n: 1, data: c}
    }
    //pub fn diag(&self, k : int) -> Matrix<T>
    pub fn augment(&self, mat : &Matrix<T>) -> Matrix<T> {
        //! Return a new matrix, self augmented by matrix mat.
        //! An MxN matrix augmented with an MxC matrix produces an Mx(N+C) matrix.
        Matrix::from_fn(self.m, self.n+mat.n, |i,j| {
            if j < self.n { self.at(i, j) } else { mat.at(i, j - self.n) }
        })
    }
    pub fn transpose(&self) -> Matrix<T> {
        //! Return the transpose of the matrix.
        //! The transpose of a matrix MxN has dimensions NxM.
        Matrix::from_fn(self.n, self.m, |i,j| { self.at(j, i) })
    }
    pub fn apply(&self, applier : |uint, uint|) {
        //! Call an applier function with each index in self.
        //! Input to applier is two parameters: row, col.
        for i in range(0, self.m) {
            for j in range(0, self.n) {
                applier(i, j);
            }
        }
    }
    pub fn fold(&self, init : T, folder: |T,T| -> T) -> T {
        //! Call a folder function that acts as if it flattens the matrix
        //! onto one row and then folds across.
        let mut acc = init;
        self.apply(|i,j| { acc = folder(acc.clone(), self.at(i,j)); });
        acc
    }
}

impl<T:Clone, U> Matrix<T> {
    pub fn map(&self, mapper : |T| -> U) -> Matrix<U> {
        //! Return a copy of self where each value has been
        //! operated upon by mapper.
        Matrix::from_fn(self.m, self.n, |i,j| { mapper(self.at(i,j)) })
    }
}

// methods for Matrix of numbers
impl<T:Add<T,T>+Mul<T,T>+Zero+Clone> Matrix<T> {
    pub fn sum(&self) -> T {
        //! Return the summation of all elements in self.
        self.fold(num::zero(), |a,b| { a + b })
    }
    fn dot(&self, other: &Matrix<T>) -> T {
        //! Return the product of the first row in self with the first row in other.
        let mut sum : T = num::zero();
        for i in range(0, self.n) {
            sum = sum + self.at(0, i) * other.at(i, 0);
        }
        sum
    }
}

impl<T:NumCast+Clone> Matrix<T> {
    pub fn to_f64(&self) -> Matrix<f64> {
        //! Return a new Matrix with all of the elements of self cast to f64.
        self.map(|n| -> f64 { num::cast(n).unwrap_or_else(|| 0.0) })
    }
}

impl<T:NumCast+Clone, U:NumCast+Clone> Matrix<T> {
    pub fn approx_eq(&self, other: &Matrix<U>, threshold : f64) -> bool {
        //! Return whether all of the elements of self are within
        //! threshold of all of the corresponding elements of other.
        let other_f64 = other.to_f64();
        let self_f64 = self.to_f64();
        let mut equal = true;
        self.apply(|i,j| {
            equal = if approx_eq(self_f64.at(i,j), other_f64.at(i,j), threshold) {
                equal
            } else {
                false
            };
        });
        equal
    }
}

impl<T:Num+NumCast+Clone> Matrix<T> {
    fn doolittle_pivot(&self) -> Matrix<T> {
        //! Return the pivoting matrix for self (for Doolittle algorithm)
        //! Assume that self is a square matrix.
        // initialize with a type T identity matrix
        let mut pivot = Matrix::from_fn(self.m, self.n, |i, j| {
            if i == j { num::one() } else { num::zero() }
        });
        // rearrange pivot matrix so max of each column of self is on
        // the diagonal of self when multiplied by the pivot
        for j in range(0,self.n) {
            let mut row_max = j;
            for i in range(j,self.m) {
                if abs_diff(self.at(i,j), self.at(row_max, j)) > 0.0 {
                    row_max = i;
                }
            }
            // swap the maximum row with the current one
            let tmp = pivot.data[j].clone();
            pivot.data[j] = pivot.data[row_max].clone();
            pivot.data[row_max] = tmp;
        }
        pivot
    }
    pub fn lu(&self) -> (Matrix<T>, Matrix<f64>, Matrix<f64>) {
        //! Perform the LU decomposition of square matrix self, and return
        //! the tuple (P,L,U) where P*self = L*U, and L and U are triangular.
        assert_eq!(self.m, self.n);
        let P = self.doolittle_pivot();
        let PM = (P*(*self)).to_f64();
        let mut L = identity(self.m);
        let mut U = zeros(self.m, self.n);
        for j in range(0, self.n) {
            for i in range(0, j+1) {
                let mut uppersum = 0.0;
                for k in range(0,i) {
                    uppersum += U.at(k,j)*L.at(i,k);
                }
                U.data[i][j] = PM.at(i,j) - uppersum;
            }
            for i in range(j, self.m) {
                let mut lowersum = 0.0;
                for k in range(0,j) {
                    lowersum += U.at(k,j)*L.at(i,k);
                }
                L.data[i][j] = (PM.at(i,j) - lowersum) / U.at(j,j);
            }
        }
        (P, L, U)
    }
    pub fn det(&self) -> f64 {
        //! Return the determinant of square matrix self
        //! via LU decomposition.
        //! If not a square matrix, fail.
        match self.lu() {
            // |L|=1 because it L is unitriangular
            // |P|=1 or -1 because it's a permutation matrix
            // |U|=product of U's diagonal
            (P, _, U) => {
                // return the product of the diagonal
                let mut prod = 1.0;
                let mut swaps = 0i32;
                for i in range(0, self.m) {
                    prod *= U.at(i,i);
                    swaps += if P.at(i,i) == num::one() { 0 } else { 1 };
                }
                // flip the sign of the determinant based on swaps of P
                if (swaps/2) % 2 == 1 {
                    -prod
                } else {
                    prod
                }
            }
        }
    }
}

impl<T:PartialEq+Clone> PartialEq for Matrix<T> {
    fn eq(&self, rhs: &Matrix<T>) -> bool {
        //! Return whether the elements of self equal the elements of rhs.
        if self.size() == rhs.size() {
            let mut equal = true;
            self.apply(|i,j| {
                equal = if self.at(i,j) == rhs.at(i,j) { equal } else { false };
            });
            equal
        }
        else {
            false
        }
    }
}

// use + to add matrices
impl<T:Add<T,T>+Clone> Add<Matrix<T>,Matrix<T>> for Matrix<T> {
    fn add(&self, rhs: &Matrix<T>) -> Matrix<T> {
        //! Return the sum of two matrices with the same dimensions.
        //! If sizes don't match, fail.
        assert_eq!(self.size(), rhs.size());
        Matrix::from_fn(self.m, self.n, |i, j| {
            self.at(i,j) + rhs.at(i,j)
        })
    }
}

// use unary - to negate matrices
impl<T:Neg<T>+Clone> Neg<Matrix<T>> for Matrix<T> {
    fn neg(&self) -> Matrix<T> {
        //! Return a matrix of the negation of each value in self.
        self.map(|n| { -n })
    }
}

// use binary - to subtract matrices
impl<T:Neg<T>+Add<T,T>+Clone> Sub<Matrix<T>, Matrix<T>> for Matrix<T> {
    fn sub(&self, rhs: &Matrix<T>) -> Matrix<T> {
        //! Return the difference of two matrices with the same dimensions.
        //! If sizes don't match, fail.
        (*self) + (-(*rhs))
    }
}

// use * to multiply matrices
impl<T:Add<T,T>+Mul<T,T>+Zero+Clone> Mul<Matrix<T>, Matrix<T>> for Matrix<T> {
    fn mul(&self, rhs: &Matrix<T>) -> Matrix<T> {
        //! Return the product of multiplying two matrices.
        //! MxR matrix * RxN matrix = MxN matrix.
        //! If inner dimensions don't match, fail.
        assert_eq!(self.n, rhs.m);
        Matrix::from_fn(self.m, rhs.n, |i,j| {
            self.row(i).dot(&rhs.col(j))
        })
    }
}

// use [(x,y)] to index matrices
impl<T> Index<(uint, uint), T> for Matrix<T> {
    fn index(&self, index: &(uint, uint)) -> &T {
        //! Return the element at the location specified by a (row, column) tuple.
        match *index {
            (x,y) => &self.data[x][y]
        }
    }
}

// use [(x,y)] to do mutable indexing
impl<T> IndexMut<(uint, uint), T> for Matrix<T> {
    fn index_mut(&mut self, index: &(uint, uint)) -> &mut T {
        match *index {
            (x,y) => &mut self.data[x][y]
        }
    }
}

// use ! to transpose matrices
impl<T:Clone> Not<Matrix<T>> for Matrix<T> {
    fn not(&self) -> Matrix<T> {
        //! Return the transpose of self.
        self.transpose()
    }
}

// use | to augment matrices
impl<T:Clone> BitOr<Matrix<T>,Matrix<T>> for Matrix<T> {
    fn bitor(&self, rhs: &Matrix<T>) -> Matrix<T> {
        //! Return self augmented by matrix rhs.
        self.augment(rhs)
    }
}

// use ^ to exponentiate matrices
impl<T:Add<T,T>+Mul<T,T>+Zero+Clone> BitXor<uint, Matrix<T>> for Matrix<T> {
    fn bitxor(&self, rhs: &uint) -> Matrix<T> {
        //! Return a matrix of self raised to the power of rhs.
        //! Self must be a square matrix.
        assert_eq!(self.m, self.n);
        let mut ret = Matrix::from_fn(self.m, self.n, |i,j| {
            self.at(i, j)
        });
        for _ in range(1, *rhs) {
            ret = (*self)*ret;
        }
        ret
    }
}

// convenience constructors
pub fn zeros(m : uint, n : uint) -> Matrix<f64> {
    //! Create an MxN zero matrix of type f64.
    Matrix::from_elem(m, n, 0.0)
}

pub fn ones(m : uint, n : uint) -> Matrix<f64> {
    //! Create an MxN ones matrix of type f64.
    Matrix::from_elem(m, n, 1.0)
}

pub fn identity(dim : uint) -> Matrix<f64> {
    //! Create a dimxdim identity matrix of type f64.
    Matrix::from_fn(dim, dim, |i, j| { if i == j { 1.0 } else { 0.0 }})
}

