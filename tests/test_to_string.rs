use matrixrs::Matrix;

#[test]
fn test_to_string() {
    let m : Matrix<i32> = Matrix::new(vec![vec![8,1,6],vec![3,5,7],vec![4,9,2]]);
    let m_str = m.to_string();
    println!("{:s}", m_str);
    println!("^Make sure this looks okay^");
}

