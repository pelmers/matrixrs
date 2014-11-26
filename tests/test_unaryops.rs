use matrixrs::Matrix;

#[test]
fn test_neg() {
    let m4 = Matrix::<int>::new(vec![vec![1],vec![2],vec![3]]);
    let m7 = Matrix::<int>::new(vec![vec![-1],vec![-2],vec![-3]]);
    assert_eq!(-m4, m7);
}

#[test]
fn test_not() {
    let m1 = Matrix::<int>::new(vec![vec![1,1],vec![2,2]]);
    let m1_tpose = Matrix::<int>::new(vec![vec![1,2],vec![1,2]]);
    assert_eq!(!m1, m1_tpose);
    let m4 = Matrix::<int>::new(vec![vec![1],vec![2],vec![3]]);
    let m4_tpose = Matrix::<int>::new(vec![vec![1,2,3]]);
    assert_eq!(!m4, m4_tpose);
}

#[test]
fn test_sum() {
    let m1 = Matrix::<int>::new(vec![vec![1,1],vec![2,2]]);
    assert_eq!(m1.sum(), 6);
    let m3 = Matrix::<int>::new(vec![vec![2],vec![3],vec![1]]);
    let m4 = Matrix::<int>::new(vec![vec![1],vec![2],vec![3]]);
    assert_eq!(m3.sum(), m4.sum());
    let m8 = Matrix::<int>::new(vec![vec![2]]);
    assert_eq!(m8.sum(), 2);
}
