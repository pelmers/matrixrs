use matrixrs;
use matrixrs::Matrix;

#[test]
fn test_from_fn() {
    assert_eq!(Matrix::<uint>::from_fn(2, 2, |i,j| { i+j }), Matrix::new(vec![vec![0,1],vec![1,2]]));
}

#[test]
fn test_identity() {
    let m1 = matrixrs::identity(1);
    assert_eq!(m1, Matrix::new(vec![vec![1.0]]));
    let m2 = matrixrs::identity(4);
    let m2_exp = Matrix::new(vec![vec![1.0,0.0,0.0,0.0], vec![0.0,1.0,0.0,0.0],vec![0.0,0.0,1.0,0.0],vec![0.0,0.0,0.0,1.0]]);
    assert_eq!(m2, m2_exp);
}

#[test]
fn test_zeros() {
    let m = matrixrs::zeros(2,3);
    let m_exp = Matrix::new(vec![vec![0.0,0.0,0.0],vec![0.0,0.0,0.0]]);
    assert_eq!(m, m_exp);
    let m2 = matrixrs::zeros(3,3);
    let m2_exp = Matrix::new(vec![vec![0.0,0.0,0.0],vec![0.0,0.0,0.0],vec![0.0,0.0,0.0]]);
    assert_eq!(m2, m2_exp);
}

#[test]
fn test_ones() {
    let m = matrixrs::ones(2,3);
    let m_exp = Matrix::new(vec![vec![1.0,1.0,1.0],vec![1.0,1.0,1.0]]);
    assert_eq!(m, m_exp);
    let m2 = matrixrs::ones(3,3);
    let m2_exp = Matrix::new(vec![vec![1.0,1.0,1.0],vec![1.0,1.0,1.0],vec![1.0,1.0,1.0]]);
    assert_eq!(m2, m2_exp);
}

