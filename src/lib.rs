trait Float:
    core::ops::Neg<Output = Self>
    + core::ops::Div<Output = Self>
    + core::ops::Mul<Output = Self>
    + core::ops::Add<Output = Self>
    + core::ops::AddAssign
    + core::ops::Sub<Output = Self>
    + core::ops::SubAssign
    + core::iter::Sum<Self>
    + core::cmp::PartialOrd
    + core::fmt::Display
    + core::fmt::Debug
    + PartialEq
    + Copy
    + Sized
{
    /// abstracting square root from floating types
    fn sqrt(self) -> Self;

    /// abstracting powi from floating types
    fn powi(self, n: i32) -> Self;

    /// method to produce zero vector for the floating type
    fn zero() -> Self;

    /// method to produce unity vector for the floating type
    fn one() -> Self;

    /// method to round off the value
    fn round(self) -> Self;

    /// calculates the absolute of the value
    fn abs(self) -> Self;
}

impl Float for f32 {
    fn sqrt(self) -> Self {
        self.sqrt()
    }
    
    fn powi(self, n: i32) -> Self {
        self.powi(n)
    }
    
    fn zero() -> Self {
        0.
    }

    fn one() -> Self {
        1.
    }

    fn round(self) -> Self {
        self.round()
    }

    fn abs(self) -> Self {
        self.abs()
    }
}

impl Float for f64 {
    fn sqrt(self) -> Self {
        self.sqrt()
    }
    
    fn powi(self, n: i32) -> Self {
        self.powi(n)
    }
    
    fn zero() -> Self {
        0.
    }

    fn one() -> Self {
        1.
    }

    fn round(self) -> Self {
        self.round()
    }

    fn abs(self) -> Self {
        self.abs()
    }
}

#[derive(Debug, Clone, Copy)]
struct Complex<T: Float = f32>(T, T);

impl<T: Float> Complex<T> {
    fn real(&self) -> T {
        self.0
    }
    
    fn imaginary(&self) -> T {
        self.1
    }
    
    fn conjugate(&self) -> Self {
        Self(self.0, -self.1)
    }
    
    fn magnitude(&self) -> T {
        (self.0.powi(2) + self.1.powi(2)).sqrt()
    }

    fn round(self) -> Self {
        Self(self.0.round(), self.1.round())
    }
}

impl<T: Float> core::ops::Div for Complex<T> {
    type Output = Self;
    
    fn div(mut self, other: Complex<T>) -> Self::Output {
        self /= other;
        self
    }
}

impl<T: Float> std::ops::DivAssign for Complex<T> {
    fn div_assign(&mut self, rhs: Self) {
        let mag = rhs.0.powi(2) + rhs.1.powi(2); 
        *self = Self((self.0 * rhs.0 - self.1 * rhs.1) / mag, (self.0 * rhs.1 + self.1 * rhs.0) / mag);
    }
}

impl<T: Float> std::ops::Mul for Complex<T> {
    type Output = Self;
    
    fn mul(mut self, other: Complex<T>) -> Self::Output {
        self *= other;
        self
    }
}

impl<T: Float> std::ops::MulAssign for Complex<T> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = Self(self.0 * rhs.0 - self.1 * rhs.1, self.0 * rhs.1 + self.1 * rhs.0);
    }
}

impl<T: Float> std::ops::Add for Complex<T> {
    type Output = Self;
    
    fn add(mut self, other: Complex<T>) -> Self::Output {
        self += other;
        self
    }  
}

impl<T: Float> std::ops::AddAssign for Complex<T> {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
        self.1 += rhs.1;
    }
}

impl<T: Float> std::iter::Sum for Complex<T> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|a, b| a + b).unwrap()
    }
}

impl<T: Float> std::ops::Sub for Complex<T> {
    type Output = Self;
    
    fn sub(mut self, other: Complex<T>) -> Self::Output {
        self -= other;
        self
    }  
}

impl<T: Float> std::ops::SubAssign for Complex<T> {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
        self.1 -= rhs.1;
    }
}

impl<T: Float> std::ops::Neg for Complex<T> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(-self.0, -self.1)
    }
}

impl<T: Float> From<T> for Complex<T> {
    fn from(var: T) -> Self {
        Self(var, T::zero())
    }
}

impl<T: Float> From<&T> for Complex<T> {
    fn from(var: &T) -> Self {
        Self(var.to_owned(), T::zero())
    }
}

impl<T: Float> From<(T, T)> for Complex<T> {
    fn from(var: (T, T)) -> Self {
        Self(var.0, var.1)
    }
}


impl<T: Float> std::fmt::Display for Complex<T> {
   fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "{}", self.0)?;
        
        if self.1 != T::zero() {
            write!(f, " + {}j", self.1)?;
        }
        
        Ok(())
   }
}

impl<T: Float> PartialEq for Complex<T> {
    fn eq(&self, other: &Self) -> bool { 
        self.0 == other.0 && self.1 == other.1
    }
}

mod matrix {
    use super::{Float, Complex};

    #[derive(Debug)]
    enum MatrixError {
        Singular,
    }

    impl std::fmt::Display for MatrixError {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            match self {
                Self::Singular => write!(f, "Matrix is singular"),
            }
        }
    }

    impl std::error::Error for MatrixError {}

    /// Defines a matrix
    #[derive(Debug, PartialEq)]
    pub struct Matrix<const M: usize, const N: usize, T: Float = f32>([[Complex<T>; N]; M]);

    impl<T: Float, const M: usize, const N: usize> Matrix<M, N, T> {
        pub fn new_zero() -> Self {
            Self([[T::zero().into(); N]; M])
        }

        pub fn transpose(&self) -> Matrix<N, M, T> {
            (0..M).flat_map(|i| (0..N).map(move |j| (i, j))).fold(Matrix::new_zero(), |mut acc, (i, j)| {
                acc.0[j][i] = self.0[i][j];
                acc
            })
        }

        /// Returns the two norm. Also known as *Euclidean norm* or *L_2 norm*
        /// The norm of a matrix is given by **||A|| = sqrt(AA^T)**
        /// The matrix AA^T us symmetic, therefore its 
        pub fn norm(&self) -> T {
            let sum = self.0.iter().flat_map(|r| r.iter()).map(Complex::to_owned).sum::<Complex<T>>();
            sum.real().sqrt()
        }

        /// Calculates the inner product of two matrix.
        /// `x.inner_product(y)` is denoted as <x, y>.
        /// For matrices, the inner product is: *<x, y> = trace(x^T \* y)*
        pub fn inner_product(&self, mut other: Self) -> Complex<T> {
            other.0.iter_mut().for_each(|r| r.iter_mut().for_each(|c| *c = c.conjugate()));
            let mat = self.transpose() * other;
            mat.trace()
        }

        fn rank() -> usize {
            todo!()
        }

        /// rounds off the values of the matrix
        fn round(mut self) -> Self {
            self.0.iter_mut().for_each(|r| r.iter_mut().for_each(|c| *c = c.round()));
            self
        }
    }

    impl<T: Float, const N: usize> Matrix<N, N, T> {
        /// Creates a new identity matrix
        fn new_identity() -> Self {
            Self(std::array::from_fn(|i| 
                    std::array::from_fn(|j| match j == i {
                        true => T::one(),
                        false => T::zero(),
                    }.into())
            ))
        }

        /// Calculates the sum of the diagonal elements of the matrix
        fn trace(&self) -> Complex<T> {
            (0..N).map(|i| self.0[i][i]).sum()
        }

        /// Calculates the determinant of the given square matrix
        /// - Uses Gaussian elemination and transformations to reduce the matrix to upper triangular form.
        /// - The determinant is then the product of all diagonal elements.
        ///
        /// Complexity: O(n^3)
        fn determinant(&self) -> Complex<T> {
            let mut matrix = self.0;
            let zero = Complex::from(T::zero());
            let [mut det, mut total] = [Complex::from(T::one()); 2];

            for diag_row in 0..N {
                // swap row with non-zero element
                let Some(non_zero_row) = (diag_row..N).find(|&i| matrix[i][diag_row] != zero) else { continue; };
                if non_zero_row != diag_row {
                    // swap rows
                    (matrix[non_zero_row], matrix[diag_row]) = (matrix[diag_row], matrix[non_zero_row]);

                    // change sign if odd
                    if (non_zero_row - diag_row) & 1 == 1 {
                        det = -det;
                    }
                }

                // transform every row below diag_row
                let temp: [Complex<T>; N] = std::array::from_fn(|j| matrix[diag_row][j]);
                for row in matrix.iter_mut().skip(diag_row + 1) {
                    let num2 = row[diag_row];
                    row.iter_mut().zip(temp).for_each(|(k, ele)| *k = (temp[diag_row] * *k) - (num2 * ele));
                    total *= temp[diag_row];
                }
            }

            // multiply diagonal elements
            (0..N).for_each(|i| det *= matrix[i][i]);
            det / total
        }

        /// Determines the inverse of a matrix
        /// ### Procedure
        /// Gaussian Jordan Elemination Method
        fn inverse(&self) -> Result<Self, MatrixError> {
            let mut matrix = self.0;
            let mut inverse = Self::new_identity();

            for i in 0..N {
                let max_row = match (i..N).max_by(|&a, &b| {
                    let magnitude = |z: Complex<T>| z.real().abs() + z.imaginary().abs();
                    magnitude(matrix[a][i]).partial_cmp(&magnitude(matrix[b][i])).unwrap_or(core::cmp::Ordering::Equal)
                }) {
                    Some(max_row) if matrix[max_row][i] != T::zero().into() => max_row,
                    _ => return Err(MatrixError::Singular),
                };

                // swap row
                if max_row != i {
                    (matrix[max_row], matrix[i]) = (matrix[i], matrix[max_row]);
                    (inverse.0[max_row], inverse.0[i]) = (inverse.0[i], inverse.0[max_row]);
                }

                let pivot = matrix[i][i];
                (0..N).for_each(|c| {
                    matrix[i][c] /= pivot;
                    inverse.0[i][c] /= pivot;
                });

                (0..N).filter(|&r| r != i).for_each(|r| {
                    let factor = matrix[r][i];
                    (0..N).for_each(|k| {
                        matrix[r][k] -= factor * matrix[i][k];
                        inverse.0[r][k] -= factor * inverse.0[i][k];
                    });
                });
            }

            Ok(inverse)
        }
    }

    impl<T: Float, C: Into<Complex<T>>, const M: usize, const N: usize> From<[[C; N]; M]> for Matrix<M, N, T> {
        fn from(value: [[C; N]; M]) -> Self {
            Self(value.map(|r| r.map(|c| c.into())))
        }
    }

    impl<T: Float, C: Into<Complex<T>> + Clone, const M: usize, const N: usize> From<&[&[C; N]; M]> for Matrix<M, N, T> {
        fn from(value: &[&[C; N]; M]) -> Self {
            Self(core::array::from_fn(|r| core::array::from_fn(|c| value[r][c].to_owned().into())))
        }
    }

    impl<T: Float, const M: usize, const N: usize> core::fmt::Display for Matrix<M, N, T> {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            for i in 0..M {
                write!(f, "│\t")?;
                for j in 0..N {
                    write!(f, "{}\t", self.0[i][j])?;
                }
                writeln!(f, "│")?;
            }

            Ok(())
        }
    }

    impl<T: Float, C: Into<Complex<T>>, const M: usize, const N: usize> core::ops::Div<C> for Matrix<M, N, T> {
        type Output = Self;

        fn div(mut self, rhs: C) -> Self::Output {
            self /= rhs;
            self
        }
    }

    impl<T: Float, C: Into<Complex<T>>, const M: usize, const N: usize> core::ops::DivAssign<C> for Matrix<M, N, T> {
        fn div_assign(&mut self, rhs: C) {
            let val = rhs.into();
            self.0.iter_mut().for_each(|r| r.iter_mut().for_each(|c| *c /= val));
        }
    }

    impl<T: Float, const M: usize, const N: usize, const O: usize> core::ops::Mul<Matrix<N, O, T>> for Matrix<M, N, T> {
        type Output = Matrix<M, O, T>;

        fn mul(self, rhs: Matrix<N, O, T>) -> Self::Output {
            self * &rhs
        }
    }

    impl<T: Float, const M: usize, const N: usize, const O: usize> core::ops::Mul<&Matrix<N, O, T>> for Matrix<M, N, T> {
        type Output = Matrix<M, O, T>;

        fn mul(self, rhs: &Matrix<N, O, T>) -> Self::Output {
            let matrix = std::array::from_fn(|i| 
                std::array::from_fn(|j| 
                    (0..N).map(|k| self.0[i][k] * rhs.0[k][j]).sum::<Complex<T>>()
                )
            );

            Matrix(matrix)
        }
    }

    /// **Only for square matrices**
    impl<T: Float, const N: usize> core::ops::MulAssign for Matrix<N, N, T> {
        fn mul_assign(&mut self, rhs: Self) {
            let matrix = std::array::from_fn(|i| 
                std::array::from_fn(|j| 
                    (0..N).map(|k| self.0[i][k] * rhs.0[k][j]).sum::<Complex<T>>()
                )
            );

            *self = Self(matrix);
        }
    }

    impl<T: Float, C: Into<Complex<T>>, const M: usize, const N: usize> core::ops::Mul<C> for Matrix<M, N, T> {
        type Output = Self;

        fn mul(mut self, rhs: C) -> Self::Output {
            self *= rhs;
            self
        }
    }

    impl<T: Float, C: Into<Complex<T>>, const M: usize, const N: usize> core::ops::MulAssign<C> for Matrix<M, N, T> {
        fn mul_assign(&mut self, rhs: C) {
            let val = rhs.into();
            self.0.iter_mut().for_each(|r| r.iter_mut().for_each(|c| *c *= val));
        }
    }

    impl<T: Float, const M: usize, const N: usize> core::ops::Add for Matrix<M, N, T> {
        type Output = Self;

        fn add(mut self, rhs: Self) -> Self::Output {
            self += rhs;
            self
        }
    }

    impl<T: Float, const M: usize, const N: usize> core::ops::AddAssign for Matrix<M, N, T> {
        fn add_assign(&mut self, rhs: Self) {
            self.0.iter_mut().zip(rhs.0).for_each(|(s_arr, o_arr)| s_arr.iter_mut().zip(o_arr).for_each(|(s, o)| *s += o));
        }
    }

    impl<T: Float, const M: usize, const N: usize> core::ops::Sub for Matrix<M, N, T> {
        type Output = Self;

        fn sub(mut self, rhs: Self) -> Self::Output {
            self -= rhs;
            self
        }
    }

    impl<T: Float, const M: usize, const N: usize> core::ops::SubAssign for Matrix<M, N, T> {
        fn sub_assign(&mut self, rhs: Self) {
            self.0.iter_mut().zip(rhs.0).for_each(|(s_arr, o_arr)| s_arr.iter_mut().zip(o_arr).for_each(|(s, o)| *s -= o));
        }
    }

    impl<T: Float, const M: usize, const N: usize> core::ops::Neg for Matrix<M, N, T> {
        type Output = Self;

        fn neg(self) -> Self::Output {
            Self(self.0.map(|r| r.map(Complex::neg)))
        }
    }


    #[cfg(test)]
    mod tests {
        use super::Matrix;
        use crate::Complex;

        #[test]
        fn test_matrix_display() {
            let matrix = Matrix::from([[10., 0., 20.], [0., 30., 0.], [200., 0., 100.]]);
            assert_eq!("│\t10\t0\t20\t│\n│\t0\t30\t0\t│\n│\t200\t0\t100\t│\n", matrix.to_string());
        }

        #[test]
        fn test_zero_matrix() {
            let matrix = Matrix::new_zero();
            assert_eq!(Matrix::from([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]), matrix);
        }

        #[test]
        fn test_identity_matrix() {
            let matrix = Matrix::new_identity();
            assert_eq!(Matrix::from([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]), matrix);
        }

        #[test]
        fn test_matrix_mul() {
            let a = Matrix::from([[1., 8., 3.], [9., 4., 5.], [6., 2., 7.]]);
            let b = Matrix::from([[6., 7., 4.], [1., 3., 2.], [5., 9., 8.]]);

            assert_eq!(a * b, Matrix::from([[29., 58., 44.], [83., 120., 84.], [73., 111., 84.]]));
        }

        #[test]
        fn test_matrix_mul_assign() {
            let mut a = Matrix::from([[1., 8., 3.], [9., 4., 5.], [6., 2., 7.]]);
            let b = Matrix::from([[6., 7., 4.], [1., 3., 2.], [5., 9., 8.]]);

            a *= b;

            assert_eq!(a, Matrix::from([[29., 58., 44.], [83., 120., 84.], [73., 111., 84.]]));
        }

        #[test]
        fn test_matrix_mul_scalar() {
            let a = Matrix::from([[1., 8., 3.], [9., 4., 5.], [6., 2., 7.]]);
            assert_eq!(a * 2., Matrix::from([[2., 16., 6.], [18., 8., 10.], [12., 4., 14.]]));
        }

        #[test]
        fn test_matrix_mul_assign_scalar() {
            let mut a = Matrix::from([[1., 8., 3.], [9., 4., 5.], [6., 2., 7.]]);

            a *= 2.;
            assert_eq!(a, Matrix::from([[2., 16., 6.], [18., 8., 10.], [12., 4., 14.]]));
        }

        #[test]
        fn test_matrix_add() {
            let a = Matrix::from([[4., 3., 8.], [6., 2., 5.], [1., 5., 9.]]);
            let b = Matrix::from([[-7., 13., 1.], [-49., 28., 28.], [28., -17., -10.]]);

            assert_eq!(a + b, Matrix::from([[-3., 16., 9.], [-43., 30., 33.], [29., -12., -1.]]));
        }

        #[test]
        fn test_matrix_add_assign() {
            let mut a = Matrix::from([[4., 3., 8.], [6., 2., 5.], [1., 5., 9.]]);
            let b = Matrix::from([[-7., 13., 1.], [-49., 28., 28.], [28., -17., -10.]]);

            a += b;

            assert_eq!(a, Matrix::from([[-3., 16., 9.], [-43., 30., 33.], [29., -12., -1.]]));
        }

        #[test]
        fn test_matrix_sub() {
            let a = Matrix::from([[4., 3., 8.], [6., 2., 5.], [1., 5., 9.]]);
            let b = Matrix::from([[-7., 13., 1.], [-49., 28., 28.], [28., -17., -10.]]);

            assert_eq!(a - b, Matrix::from([[11., -10., 7.], [55., -26., -23.], [-27., 22., 19.]]));
        }

        #[test]
        fn test_matrix_sub_assign() {
            let mut a = Matrix::from([[4., 3., 8.], [6., 2., 5.], [1., 5., 9.]]);
            let b = Matrix::from([[-7., 13., 1.], [-49., 28., 28.], [28., -17., -10.]]);

            a -= b;

            assert_eq!(a, Matrix::from([[11., -10., 7.], [55., -26., -23.], [-27., 22., 19.]]));
        }

        #[test]
        fn test_matrix_neg() {
            let a = Matrix::from([[4., 3., 8.], [6., 2., 5.], [1., 5., 9.]]);
            assert_eq!(-a, Matrix::from([[-4., -3., -8.], [-6., -2., -5.], [-1., -5., -9.]]));
        }

        #[test]
        fn test_matrix_transpose() {
            let a = Matrix::from([[4., 4., -4.], [4., 8., 8.], [0., 12., 16.]]);
            let a_t = Matrix::from([[4., 4., 0.], [4., 8., 12.], [-4., 8., 16.]]);

            assert_eq!(a.transpose(), a_t);
        }

        #[test]
        fn test_matrix_determinant() {
            let a = Matrix::from([[0., 12., 16.], [4., 4., -4.], [4., 8., 8.]]);
            assert_eq!(a.determinant(), Complex::from(-320.))
        }

        #[test]
        fn test_matrix_inverse() {
            const PRECISION: f32 = 1e2;

            let a = Matrix::from([[2., -1., 0.], [-1., 2., -1.], [0., -1., 2.]]);
            let inverse = {
                let matrix = a.inverse().unwrap();
                (matrix * PRECISION).round() / PRECISION
            };

            assert_eq!(inverse, Matrix::from([[0.75, 0.5, 0.25], [0.5, 1., 0.5], [0.25, 0.5, 0.75]]))
        }

        #[test]
        #[should_panic]
        fn test_singular_matrix_inverse() {
            let a = Matrix::from([[1., -2., 2.], [-1., 2., -2.], [3., -2., -1.]]);
            a.inverse().unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Complex;
    
    #[test]
    fn test_complex_display() {
        let mut complex = Complex::from(1.0);
        assert_eq!(complex.to_string(), "1".to_string());
        
        complex = Complex::from(1.33);
        assert_eq!(complex.to_string(), "1.33".to_string());
        
        complex = Complex::from((1.33, 20.));
        assert_eq!(complex.to_string(), "1.33 + 20j".to_string());
        
        complex = Complex::from((1.33, 20.55));
        assert_eq!(complex.to_string(), "1.33 + 20.55j".to_string());
    }
    
    #[test]
    fn test_complex_div() {
        let a = Complex::from((1., 2.));
        let b = Complex::from((3., 4.));
        
        assert_eq!(a / b, Complex::from((-0.2, 0.4)))
    }
    
    #[test]
    fn test_complex_div_assign() {
        let mut a = Complex::from((1., 2.));
        let b = Complex::from((3., 4.));

        a /= b;
        
        assert_eq!(a, Complex::from((-0.2, 0.4)))
    }
    
    #[test]
    fn test_complex_mul() {
        let a = Complex::from((1., 2.));
        let b = Complex::from((3., 4.));
        
        assert_eq!(a * b, Complex::from((-5., 10.)))
    }

    #[test]
    fn test_complex_mul_assign() {
        let mut a = Complex::from((1., 2.));
        let b = Complex::from((3., 4.));

        a *= b;
        
        assert_eq!(a, Complex::from((-5., 10.)))
    }
    
    #[test]
    fn test_complex_add() {
        let a = Complex::from((1., 2.));
        let b = Complex::from((3., 4.));
        
        assert_eq!(a + b, Complex::from((4., 6.)))
    }
    
    #[test]
    fn test_complex_add_assign() {
        let mut a = Complex::from((1., 2.));
        let b = Complex::from((3., 4.));

        a += b;
        
        assert_eq!(a, Complex::from((4., 6.)))
    }
    
    #[test]
    fn test_complex_sub() {
        let a = Complex::from((1., 2.));
        let b = Complex::from((3., 4.));
        
        assert_eq!(a - b, Complex::from((-2., -2.)))
    }

    #[test]
    fn test_complex_sub_assign() {
        let mut a = Complex::from((1., 2.));
        let b = Complex::from((3., 4.));

        a -= b;
        
        assert_eq!(a, Complex::from((-2., -2.)))
    }

    #[test]
    fn test_complex_neg() {
        let complex = Complex::from((1., -2.));
        assert_eq!(-complex, Complex::from((-1., 2.)));
    }
    
    #[test]
    fn test_complex_magnitude() {
        let complex = Complex::from((3., 4.));
        assert_eq!(complex.magnitude(), 5.)
    }
    
    #[test]
    fn test_complex_conjugate() {
        let complex = Complex::from((3., 4.));
        assert_eq!(complex.conjugate(), Complex::from((3., -4.)))
    }
}
