use std::cmp;
use std::num::{Zero, ToPrimitive};
use std::ops::{Not,Neg,Add,Sub,Mul,Index,IndexMut,BitXor,BitOr};
use std::f64::NAN;

/// Matrix -- Generic 2D Matrix implementation in Rust.
#[derive(Debug)]
pub struct Matrix<T> {
    /// Number of rows
    m : usize,
    /// Number of columns
    n : usize,
    /// Table (vector of vector) of data values in the matrix
    data : Vec<Vec<T>>
}

pub fn sum<T>(vec: &Vec<T>) -> T
    where T:Copy+Zero+Add<Output=T>
{
    //! Return sum of vec.
    vec.iter().fold(T::zero(), |b,&f| b + f)
}

pub fn dot<T>(a: &Vec<T>, b: &Vec<T>) -> T
    where T:Copy+Zero+Add<Output=T>+Mul<Output=T>
{
    //! Return dot product of a and b.
    a.iter().zip(b.iter()).map(|(x, y)| (*x)*(*y)).fold(T::zero(), |b,f| b + f)
}

fn approx_eq(a: f64, b: f64, threshold: f64) -> bool {
    //! Return whether two floats a and b are within threshold of each other.
    //! i.e. | a - b | <= threshold
    (if a > b { a-b } else { b-a }) <= threshold
}

fn abs_diff<T:ToPrimitive>(a : T, b : T) -> f64 {
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
        let cols = if rows > 0 { data[0].len() } else { 0 };
        for row in data.iter() {
            assert_eq!(row.len(), cols);
        }
        Matrix{m: rows, n: cols, data: data}
    }
    pub fn from_fn<F>(m : usize, n : usize, func : F) -> Matrix<T>
        where F:Fn(usize, usize) -> T
    {
        //! Create an m-by-n matrix by using a function func
        //! that returns a number given row and column.
        Matrix{
            m:m,
            n:n,
            data:(0..m).map(|i| (0..n).map(|j| func(i,j)).collect()).collect()
        }
    }
    pub fn size(&self) -> (usize, usize) {
        //! Return the size of a Matrix as row, column.
        (self.m, self.n)
    }
    pub fn row(&self, row : usize) -> &Vec<T> {
        //! Return specified row from an MxN matrix as vector.
        &self.data[row]
    }
    pub fn col(&self, col: usize) -> Vec<&T> {
        //! Return specified col from an MxN matrix as a vector.
        self.data.iter().map(|r| &r[col]).collect()
    }
    pub fn diag(&self) -> Vec<&T> {
        //! Return specified diagonal as a vector.
        (0..cmp::min(self.m,self.n)).map(|i| &self.data[i][i]).collect()
    }
    pub fn augment<'a>(&'a self, mat : &'a Matrix<T>) -> Matrix<&'a T> {
        //! Return a new matrix, self augmented by matrix mat.
        //! An MxN matrix augmented with an MxC matrix produces an Mx(N+C) matrix.
        Matrix::from_fn(self.m, self.n+mat.n, |i,j| {
            if j < self.n { &self.data[i][j] } else { &mat.data[i][j-self.n] }
        })
    }
    pub fn transpose(&self) -> Matrix<&T> {
        //! Return the transpose of the matrix.
        //! The transpose of a matrix MxN has dimensions NxM.
        Matrix::from_fn(self.n, self.m, |i,j| &self.data[j][i])
    }
    pub fn flatten(&self) -> Vec<&T> {
        //! Flatten matrix in row-major order as a vector.
        self.data.iter().flat_map(|r| r.iter()).collect()
    }
    // TODO: don't collect then iterate
    // (i.e. define flatten as iter.collect, rather than iter as flatten.iter)
    pub fn iter(&self) -> ::std::vec::IntoIter<&T> {
        //! Return iterator over matrix in row-major order.
        self.flatten().into_iter()
    }
    pub fn map<U,F>(&self, mapper : F) -> Matrix<U>
        where F: Fn(&T) -> U
    {
        //! Return a copy of self where each value has been
        //! operated upon by mapper.
        Matrix::from_fn(self.m, self.n, |i,j| mapper(&self.data[i][j]))
    }
}

impl<T:Clone> Matrix<T> {
    pub fn from_elem(m: usize, n: usize, elem: T) -> Matrix<T> {
        //! Create an m-by-n matrix, where each element is a clone of elem.
        Matrix{
            m: m,
            n: n,
            data: vec![vec![elem; n]; m]
        }
    }
}

impl<T:Copy> Matrix<T> {
    pub fn owned_col(&self, col: usize) -> Vec<T> {
        self.data.iter().map(|r| r[col]).collect()
    }
}

impl<T:ToPrimitive> Matrix<T> {
    pub fn to_f64(&self) -> Matrix<f64> {
        //! Return a new Matrix with all of the elements of self cast to f64.
        self.map(|n| n.to_f64().unwrap_or(NAN))
    }
    pub fn approx_eq<U:ToPrimitive>(&self, other: &Matrix<U>, threshold : f64) -> bool {
        //! Return whether all of the elements of self are within
        //! threshold of all of the corresponding elements of other.
        if self.size() != other.size() {
            false
        } else {
            self.iter().zip(other.iter())
                .all(|(a,b)| approx_eq(a.to_f64().unwrap_or(NAN),
                                       b.to_f64().unwrap_or(NAN),
                                       threshold))
        }
    }
}

impl<T:PartialEq> PartialEq for Matrix<T> {
    fn eq(&self, rhs: &Matrix<T>) -> bool {
        //! Return whether the elements of self equal the elements of rhs.
        if self.size() == rhs.size() {
            self.iter().zip(rhs.iter()).all(|(a,b)| a == b)
        }
        else {
            false
        }
    }
}

impl<'a,'b,T:Add+Copy> Add<&'b Matrix<T>> for &'a Matrix<T> {
    type Output=Matrix<<T as Add>::Output>;

    fn add(self, rhs: &'b Matrix<T>) -> Self::Output {
        //! Return the sum of two matrices with the same dimensions.
        //! If sizes don't match, fail.
        assert_eq!(self.size(), rhs.size());
        Matrix::from_fn(self.m, self.n, |i, j| {
            self.data[i][j] + rhs.data[i][j]
        })
    }
}

impl<'a,T:Neg+Copy> Neg for &'a Matrix<T> {
    type Output=Matrix<<T as Neg>::Output>;

    fn neg(self) -> Self::Output {
        //! Return a matrix of the negation of each value in self.
        self.map(|n| { -(*n) })
    }
}

impl<'a,'b,T:Sub+Copy> Sub<&'b Matrix<T>> for &'a Matrix<T> {
    type Output=Matrix<<T as Sub>::Output>;
    fn sub(self, rhs: &'b Matrix<T>) -> Self::Output {
        //! Return the difference of two matrices with the same dimensions.
        //! If sizes don't match, fail.
        assert_eq!(self.size(), rhs.size());
        Matrix::from_fn(self.m, self.n, |i, j| {
            self.data[i][j] - rhs.data[i][j]
        })
    }
}

// use * to multiply matrices
impl<'a,'b,T> Mul<&'b Matrix<T>> for &'a Matrix<T>
    where T:Zero+Copy+Mul<Output=T>+Add<Output=T>
{
    type Output=Matrix<T>;

    fn mul(self, rhs: &'b Matrix<T>) -> Self::Output {
        //! Return the product of multiplying two matrices.
        //! MxR matrix * RxN matrix = MxN matrix.
        //! If inner dimensions don't match, fail.
        assert_eq!(self.n, rhs.m);
        Matrix::from_fn(self.m, rhs.n, |i,j| {
            dot(self.row(i), &rhs.owned_col(j))
        })
    }
}

impl<T> Index<(usize, usize)> for Matrix<T> {
    type Output = T;
    fn index(&self, index: (usize, usize)) -> &T {
        //! Return the element at the location specified by a (row, column) tuple.
        match index {
            (x,y) => &self.data[x][y]
        }
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut T {
        match index {
            (x,y) => &mut self.data[x][y]
        }
    }
}

impl<'a, T> Not for &'a Matrix<T> {
    type Output = Matrix<&'a T>;
    fn not(self) -> Self::Output {
        //! Return the transpose of self.
        self.transpose()
    }
}

impl<'a, T> BitOr for &'a Matrix<T> {
    type Output = Matrix<&'a T>;
    fn bitor(self, rhs: Self) -> Matrix<&'a T> {
        //! Return self augmented by matrix rhs.
        self.augment(rhs)
    }
}

impl<'a,T:Zero+Mul<Output=T>+Add<Output=T>+Copy> BitXor<usize> for &'a Matrix<T> {
    type Output = Matrix<T>;
    fn bitxor(self, rhs: usize) -> Matrix<T> {
        //! Return a matrix of self raised to the power of rhs.
        //! Self must be a square matrix.
        assert_eq!(self.m, self.n);
        let mut ret = Matrix::from_fn(self.m, self.n, |i,j| {
            self.data[i][j]
        });
        for _ in 1..rhs {
            ret = self*&ret;
        }
        ret
    }
}

/*
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
        let pivot = self.doolittle_pivot();
        let pivot_m = (pivot*(*self)).to_f64();
        let mut lower = identity(self.m);
        let mut upper = zeros(self.m, self.n);
        for j in range(0, self.n) {
            for i in range(0, j+1) {
                let mut uppersum = 0.0;
                for k in range(0,i) {
                    uppersum += upper.at(k,j)*lower.at(i,k);
                }
                upper.data[i][j] = pivot_m.at(i,j) - uppersum;
            }
            for i in range(j, self.m) {
                let mut lowersum = 0.0;
                for k in range(0,j) {
                    lowersum += upper.at(k,j)*lower.at(i,k);
                }
                lower.data[i][j] = (pivot_m.at(i,j) - lowersum) / upper.at(j,j);
            }
        }
        (pivot, lower, upper)
    }
    pub fn det(&self) -> f64 {
        //! Return the determinant of square matrix self
        //! via LU decomposition.
        //! If not a square matrix, fail.
        match self.lu() {
            // |L|=1 because it is unitriangular
            // |P|=1 or -1 because it's a permutation matrix
            // |U|=product of U's diagonal
            (pivot, _, upper) => {
                // return the product of the diagonal
                let mut prod = 1.0;
                let mut swaps = 0i32;
                for i in range(0, self.m) {
                    prod *= upper.at(i,i);
                    swaps += if pivot.at(i,i) == num::one() { 0 } else { 1 };
                }
                // flip the sign of the determinant based on swaps of pivot
                if (swaps/2) % 2 == 1 {
                    -prod
                } else {
                    prod
                }
            }
        }
    }
}
*/

// convenience constructors
pub fn zeros(m : usize, n : usize) -> Matrix<f64> {
    //! Create an MxN zero matrix of type f64.
    Matrix::from_elem(m, n, 0.0)
}

pub fn ones(m : usize, n : usize) -> Matrix<f64> {
    //! Create an MxN ones matrix of type f64.
    Matrix::from_elem(m, n, 1.0)
}

pub fn identity(dim : usize) -> Matrix<f64> {
    //! Create a dimxdim identity matrix of type f64.
    Matrix::from_fn(dim, dim, |i, j| { if i == j { 1.0 } else { 0.0 }})
}

