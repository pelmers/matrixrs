use matrixrs::Matrix;

#[test]
fn test_eq() {
    // test equality
    let m1 = Matrix::<int>::new(vec![vec![1,1],vec![2,2]]);
    let m2 = Matrix::<int>::new(vec![vec![1,1],vec![2,2]]);
    assert_eq!(m1, m2);
    // test inequality
    let m3 = Matrix::<int>::new(vec![vec![2],vec![3],vec![1]]);
    let m4 = Matrix::<int>::new(vec![vec![1],vec![2],vec![3]]);
    assert!(m3 != m4);
    // make sure it doesn't break if dimensions don't match
    let m5 = Matrix::<int>::new(vec![vec![2],vec![3]]);
    let m6 = Matrix::<int>::new(vec![vec![2,3]]);
    assert!(m5 != m6)
}

