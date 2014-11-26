use matrixrs::Matrix;

#[test]
fn test_add() {
    let m3 = Matrix::<int>::new(vec![vec![2],vec![3],vec![1]]);
    let m4 = Matrix::<int>::new(vec![vec![1],vec![2],vec![3]]);
    let expected = Matrix::<int>::new(vec![vec![3],vec![5],vec![4]]);
    assert_eq!(m3+m4, expected);
}

#[test]
fn test_sub() {
    let m1 = Matrix::<int>::new(vec![vec![1,1],vec![2,2]]);
    let m2 = Matrix::<int>::new(vec![vec![1,1],vec![2,2]]);
    let expected = Matrix::<int>::new(vec![vec![0,0],vec![0,0]]);
    assert_eq!(m1 - m2, expected);
}

#[test]
fn test_mul() {
    let m = Matrix::<int>::new(vec![vec![8,1,6],vec![3,5,7],vec![4,9,2]]);
    let m_sq = Matrix::<int>::new(vec![vec![91,67,67],vec![67,91,67],vec![67,67,91]]);
    assert_eq!(m*m, m_sq);
    let m1 = Matrix::<int>::new(vec![vec![1,1],vec![2,2]]);
    let m5 = Matrix::<int>::new(vec![vec![2],vec![3]]);
    let m1m5 = Matrix::<int>::new(vec![vec![5],vec![10]]);
    assert_eq!(m1*m5, m1m5);
    let m4_row = Matrix::<int>::new(vec![vec![1,2,3]]);
    let m4_col = Matrix::<int>::new(vec![vec![1],vec![2],vec![3]]);
    let m4colrow = Matrix::<int>::new(vec![vec![1,2,3],vec![2,4,6],vec![3,6,9]]);
    assert_eq!(m4_col*m4_row, m4colrow);
    let m4rowcol = Matrix::<int>::new(vec![vec![14]]);
    assert_eq!(m4_row*m4_col, m4rowcol);
}

#[test]
fn test_exp() {
    let m = Matrix::<int>::new(vec![vec![8,1,6],vec![3,5,7],vec![4,9,2]]);
    let m_exp_3 = Matrix::<int>::new(vec![vec![1197,1029,1149], vec![1077,1125,1173],vec![1101,1221,1053]]);
    assert_eq!(m^3, m_exp_3);
}

#[test]
fn test_bitor() {
    let m1 = Matrix::<int>::new(vec![vec![1,1],vec![2,2]]);
    let m2 = Matrix::<int>::new(vec![vec![1,1],vec![2,2]]);
    let m1_aug_m2 = Matrix::<int>::new(vec![vec![1,1,1,1],vec![2,2,2,2]]);
    assert_eq!(m1|m2, m1_aug_m2);
}

