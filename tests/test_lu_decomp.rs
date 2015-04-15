use matrixrs::Matrix;

#[test]
fn test_lu() {
    let m = &Matrix::<i32>::new(vec![vec![7,3,-1,2],vec![3,8,1,-4],vec![-1,1,4,-1],vec![2,-4,-1,6]]);
    let p_exp = &Matrix::<i32>::new(vec![vec![1,0,0,0],vec![0,1,0,0],vec![0,0,1,0],vec![0,0,0,1]]);
    match m.lu() {
        (ref p, ref l, ref u) => {
            assert_eq!(p, p_exp);
            // m*p should equal l*u, but we'll give a little room for rounding
            assert!((l*u).approx_eq(&(p*m), 0.01));
        }
    }
    let m2 = &Matrix::<i32>::new(vec![vec![8,1,6],vec![3,5,7],vec![4,9,2]]);
    match m2.lu() {
        (ref p, ref l, ref u) => {
            assert!((l*u).approx_eq(&(p*m2), 0.01));
        }
    }
    let m3 = &Matrix::<i32>::new(vec![vec![-4,6],vec![-10,3]]);
    match m3.lu() {
        (ref p, ref l, ref u) => {
            assert!((l*u).approx_eq(&(p*m3), 0.01));
        }
    }
}
