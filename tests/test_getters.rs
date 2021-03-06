use matrixrs::Matrix;

#[test]
fn test_index() {
    let m1 = Matrix::<i32>::new(vec![vec![1,1],vec![2,2]]);
    assert_eq!(m1[(1,1)], 2);
    assert_eq!(m1[(0,1)], 1);
}

#[test]
fn test_index_mut() {
    let mut m1 = Matrix::<i32>::new(vec![vec![1,1],vec![2,2]]);
    m1[(1, 1)] = 1;
    assert_eq!(m1[(1,1)], 1);
    assert_eq!(m1[(0,1)], 1);
}

#[test]
fn test_col() {
    let m1 = Matrix::<i32>::new(vec![vec![1,4],vec![2,5],vec![3,6]]);
    let m1_col0 = vec![1,2,3];
    assert_eq!(m1.owned_col(0), m1_col0);
    let m1_col1 = vec![4,5,6];
    assert_eq!(m1.owned_col(1), m1_col1);
}

#[test]
fn test_row() {
    let m = Matrix::<i32>::new(vec![vec![8,1,6],vec![3,5,7],vec![4,9,2]]);
    let m_row0 = &vec![8,1,6];
    assert_eq!(m.row(0), m_row0);
    let m_row1 = &vec![3,5,7];
    assert_eq!(m.row(1), m_row1);
    let m_row2 = &vec![4,9,2];
    assert_eq!(m.row(2), m_row2);
}


