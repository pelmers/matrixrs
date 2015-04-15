use matrixrs::Matrix;

#[test]
fn test_to_string() {
    let m : Matrix<i32> = Matrix::new(vec![vec![8,1,6],vec![3,5,7],vec![4,9,2]]);
    println!("{:?}", m);
    println!("^Make sure this looks okay^");
}

