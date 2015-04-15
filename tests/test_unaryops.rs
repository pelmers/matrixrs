use matrixrs::Matrix;
use matrixrs::sum;

#[test]
fn test_neg() {
    let m4 = &Matrix::<i32>::new(vec![vec![1],vec![2],vec![3]]);
    let m7 = Matrix::<i32>::new(vec![vec![-1],vec![-2],vec![-3]]);
    assert_eq!(-m4, m7);
}

#[test]
fn test_not() {
    let m1 = &Matrix::<i32>::new(vec![vec![1,1],vec![2,2]]);
    let m1_tpose = Matrix::<i32>::new(vec![vec![1,2],vec![1,2]]);
    assert_eq!((!m1).into_flatten(), m1_tpose.flatten());
    let m4 = &Matrix::<i32>::new(vec![vec![1],vec![2],vec![3]]);
    let m4_tpose = Matrix::<i32>::new(vec![vec![1,2,3]]);
    assert_eq!((!m4).into_flatten(), m4_tpose.flatten());
}

#[test]
fn test_sum() {
    let m1 = Matrix::<i32>::new(vec![vec![1,1],vec![2,2]]);
    assert_eq!(sum(&m1.into_flatten()), 6);
    let m3 = Matrix::<i32>::new(vec![vec![2],vec![3],vec![1]]);
    let m4 = Matrix::<i32>::new(vec![vec![1],vec![2],vec![3]]);
    assert_eq!(sum(&m3.into_flatten()), sum(&m4.into_flatten()));
    let m8 = Matrix::<i32>::new(vec![vec![2]]);
    assert_eq!(sum(&m8.into_flatten()), 2);
}
