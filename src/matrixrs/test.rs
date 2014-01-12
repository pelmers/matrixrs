extern mod matrixrs;

use matrixrs::Matrix;

#[test]
fn test_eq() {
	// test equality
	let m1 = Matrix{m:2, n:2, data: ~[~[1,1],~[2,2]]};
	let m2 = Matrix{m:2, n:2, data: ~[~[1,1],~[2,2]]};
	assert!(m1 == m2);
	// test inequality
	let m3 = Matrix{m:3, n:1, data: ~[~[2],~[3],~[1]]};
	let m4 = Matrix{m:3, n:1, data: ~[~[1],~[2],~[3]]};
	assert!(m3 != m4);
	// make sure it doesn't break if dimensions don't match
	let m5 = Matrix{m:2, n:1, data: ~[~[2],~[3]]};
	let m6 = Matrix{m:1, n:2, data: ~[~[2,3]]};
	assert!(m5 != m6)
}

#[test]
fn test_index() {
	let m1 = Matrix{m:2, n:2, data: ~[~[1,1],~[2,2]]};
	assert!(m1[(1,1)] == 2);
	assert!(m1[(0,1)] == 1);
}

#[test]
fn test_neg() {
	let m4 = Matrix{m:3, n:1, data: ~[~[1],~[2],~[3]]};
	let m7 = Matrix{m:3, n:1, data: ~[~[-1],~[-2],~[-3]]};
	assert!(-m4 == m7);
}

#[test]
fn test_add() {
	let m3 = Matrix{m:3, n:1, data: ~[~[2],~[3],~[1]]};
	let m4 = Matrix{m:3, n:1, data: ~[~[1],~[2],~[3]]};
	let expected = Matrix{m:3, n:1, data: ~[~[3],~[5],~[4]]};
	assert!(m3+m4 == expected);
}

#[test]
fn test_sub() {
	let m1 = Matrix{m:2, n:2, data: ~[~[1,1],~[2,2]]};
	let m2 = Matrix{m:2, n:2, data: ~[~[1,1],~[2,2]]};
	let expected = Matrix{m:2, n:2, data:~[~[0,0],~[0,0]]};
	assert!(m1 - m2 == expected);
}

#[test]
fn test_mul() {
	let m = Matrix{m:3,n:3,data:~[~[8,1,6],~[3,5,7],~[4,9,2]]};
	let m_sq = Matrix{m:3,n:3,data:~[~[91,67,67],~[67,91,67],~[67,67,91]]};
	assert!(m*m == m_sq);
	let m1 = Matrix{m:2, n:2, data: ~[~[1,1],~[2,2]]};
	let m5 = Matrix{m:2, n:1, data: ~[~[2],~[3]]};
	let m1m5 = Matrix{m:2,n:1, data: ~[~[5],~[10]]};
	assert!(m1*m5 == m1m5);
	let m4_row = Matrix{m:1, n:3, data:~[~[1,2,3]]};
	let m4_col = Matrix{m:3, n:1, data: ~[~[1],~[2],~[3]]};
	let m4colrow = Matrix{m:3,n:3,data:~[~[1,2,3],~[2,4,6],~[3,6,9]]};
	assert!(m4_col*m4_row == m4colrow);
	let m4rowcol = Matrix{m:1, n:1, data:~[~[14]]};
	assert!(m4_row*m4_col == m4rowcol);
}

#[test]
fn test_not() {
	let m1 = Matrix{m:2, n:2, data: ~[~[1,1],~[2,2]]};
	let m1_tpose = Matrix{m:2, n:2, data:~[~[1,2],~[1,2]]};
	assert!(!m1 == m1_tpose);
	let m4 = Matrix{m:3, n:1, data: ~[~[1],~[2],~[3]]};
	let m4_tpose = Matrix{m:1, n:3, data:~[~[1,2,3]]};
	assert!(!m4 == m4_tpose);
}

#[test]
fn test_bitor() {
	let m1 = Matrix{m:2, n:2, data: ~[~[1,1],~[2,2]]};
	let m2 = Matrix{m:2, n:2, data: ~[~[1,1],~[2,2]]};
	let m1_aug_m2 = Matrix{m:2, n:4, data: ~[~[1,1,1,1],~[2,2,2,2]]};
	assert!(m1|m2 == m1_aug_m2);
}

#[test]
fn test_sum() {
	let m1 = Matrix{m:2, n:2, data: ~[~[1,1],~[2,2]]};
	assert!(m1.sum() == 6);
	let m3 = Matrix{m:3, n:1, data: ~[~[2],~[3],~[1]]};
	let m4 = Matrix{m:3, n:1, data: ~[~[1],~[2],~[3]]};
	assert!(m3.sum() == m4.sum());
	let m8 = Matrix{m:1, n:1, data: ~[~[2]]};
	assert!(m8.sum() == 2);
}

#[test]
fn test_ones() {
	let m = matrixrs::ones(2,3);
	let m_exp = Matrix{m:2, n:3, data: ~[~[1.0,1.0,1.0],~[1.0,1.0,1.0]]};
	assert!(m == m_exp);
	let m2 = matrixrs::ones(3,3);
	let m2_exp = Matrix{m:3, n:3, data: ~[~[1.0,1.0,1.0],~[1.0,1.0,1.0],~[1.0,1.0,1.0]]};
	assert!(m2 == m2_exp);
}

#[test]
fn test_zeros() {
	let m = matrixrs::zeros(2,3);
	let m_exp = Matrix{m:2, n:3, data: ~[~[0.0,0.0,0.0],~[0.0,0.0,0.0]]};
	assert!(m == m_exp);
	let m2 = matrixrs::zeros(3,3);
	let m2_exp = Matrix{m:3, n:3, data: ~[~[0.0,0.0,0.0],~[0.0,0.0,0.0],~[0.0,0.0,0.0]]};
	assert!(m2 == m2_exp);
}

#[test]
fn test_identity() {
	let m1 = matrixrs::identity(1);
	assert!(m1 == Matrix{m:1,n:1, data:~[~[1.0]]});
	let m2 = matrixrs::identity(4);
	let m2_exp = Matrix{m:4, n:4, data: ~[~[1.0,0.0,0.0,0.0],
		~[0.0,1.0,0.0,0.0],~[0.0,0.0,1.0,0.0],~[0.0,0.0,0.0,1.0]]};
	assert!(m2 == m2_exp);
}

#[test]
fn test_col() {
	let m1 = Matrix{m:3,n:2,data:~[~[1,4],~[2,5],~[3,6]]};
	let m1_col0 = Matrix{m:3,n:1,data:~[~[1],~[2],~[3]]};
	assert!(m1.col(0) == m1_col0);
	let m1_col1 = Matrix{m:3,n:1,data:~[~[4],~[5],~[6]]};
	assert!(m1.col(1) == m1_col1);
}

#[test]
fn test_row() {
	let m = Matrix{m:3,n:3,data:~[~[8,1,6],~[3,5,7],~[4,9,2]]};
	let m_row0 = Matrix{m:1,n:3,data:~[~[8,1,6]]};
	assert!(m.row(0) == m_row0);
	let m_row1 = Matrix{m:1,n:3,data:~[~[3,5,7]]};
	assert!(m.row(1) == m_row1);
	let m_row2 = Matrix{m:1,n:3,data:~[~[4,9,2]]};
	assert!(m.row(2) == m_row2);
}
