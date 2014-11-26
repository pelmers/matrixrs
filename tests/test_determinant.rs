use matrixrs::Matrix;

#[test]
fn test_det() {
    let m = Matrix::<int>::new(vec![vec![8,1,6],vec![3,5,7],vec![4,9,2]]);
    assert_eq!(m.det(), -360.0);
    let m2 = Matrix::<int>::new(vec![vec![100]]);
    assert_eq!(m2.det(), 100.0);
    let m3 = Matrix::<int>::new(vec![vec![-4,6],vec![-10,3]]);
    assert_eq!(m3.det(), 48.0);
}

