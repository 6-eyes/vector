trait Float:
    core::ops::Neg<Output = Self>
    + core::ops::Div<Output = Self>
    + core::ops::Mul<Output = Self>
    + core::ops::Add<Output = Self>
    + core::ops::AddAssign
    + core::ops::Sub<Output = Self>
    + core::ops::SubAssign
    + core::iter::Sum<Self>
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
    
    fn norm(&self) -> T {
        (self.0.powi(2) + self.1.powi(2)).sqrt()
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
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::Singular => write!(f, "Matrix is singular"),
            }
        }
    }

    impl std::error::Error for MatrixError {}

    trait Matrix {
        fn transpose(&self) -> Self;
    }

    trait Square<T>: Sized + Matrix {
        fn determinant(&self) -> T;
        fn inverse(&self) -> Result<Self, MatrixError>;
    }

    #[derive(Debug)]
    pub struct Matrix3x3<T: Float>([[Complex<T>; 3]; 3]);

    impl<T: Float> Matrix for Matrix3x3<T> {
        fn transpose(&self) -> Self {
            todo!()
        }
    }

    impl<T: Float> Square<T> for Matrix3x3<T> {
        fn determinant(&self) -> T {
            todo!()
        }

        fn inverse(&self) -> Result<Self, MatrixError> {
            todo!()
        }
    }

    impl<T: Float> Matrix3x3<T> {
        /// Creates a new zero matrix
        pub fn new_zero() -> Self {
            Self(std::array::from_fn(|_| std::array::from_fn(|_| T::zero().into())))
        }

        /// Creates a new identity matrix
        pub fn new_identity() -> Self {
            Self(std::array::from_fn(|i| 
                    std::array::from_fn(|j| match j == i {
                        true => T::one(),
                        false => T::zero(),
                    }.into())
            ))
        }

        pub fn adjoint(&self) -> Self {
            todo!()
        }
        
        pub fn trace(&self) -> T {
            todo!()
        }
        
        pub fn norm(&self) -> T {
            todo!()
        }
    }

    impl<T: Float, C: Into<Complex<T>>> From<[[C; 3]; 3]> for Matrix3x3<T> {
        fn from(var: [[C; 3]; 3]) -> Self {
            Self(var.map(|r| r.map(C::into)))
        }
    }

    impl<T: Float, C: Into<Complex<T>> + Clone> From<&[&[C; 3]; 3]> for Matrix3x3<T> {
        fn from(var: &[&[C; 3]; 3]) -> Self {
            Self(var.map(|r| r.to_owned().map(C::into)))
        }
    }

    impl<T: Float> std::fmt::Display for Matrix3x3<T> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            for i in 0..3 {
                writeln!(f, "│\t{}\t{}\t{}\t│", self.0[i][0], self.0[i][1], self.0[i][2])?;
            }

            Ok(())
        }
    }

    impl<T: Float> PartialEq for Matrix3x3<T> {
        fn eq(&self, other: &Self) -> bool {
            self.0.eq(&other.0)
        }
    }

    impl<T: Float> std::ops::Mul for Matrix3x3<T> {
        type Output = Matrix3x3<T>;

        fn mul(mut self, rhs: Self) -> Self::Output {
            self *= rhs;
            self
        }
    }

    impl<T: Float> std::ops::MulAssign for Matrix3x3<T> {
        fn mul_assign(&mut self, rhs: Self) {
            let matrix = std::array::from_fn(|i| 
                std::array::from_fn(|j| 
                    (0..3).map(|k| self.0[i][k] * rhs.0[k][j]).sum::<Complex<T>>()
                )
            );

            *self = Self(matrix);
        }
    }

    impl<T: Float> std::ops::Add for Matrix3x3<T> {
        type Output = Matrix3x3<T>;

        fn add(mut self, rhs: Self) -> Self::Output {
            self += rhs;
            self
        }
    }

    impl<T: Float> std::ops::AddAssign for Matrix3x3<T> {
        fn add_assign(&mut self, rhs: Self) {
            self.0.iter_mut().zip(rhs.0).for_each(|(s_arr, o_arr)| s_arr.iter_mut().zip(o_arr).for_each(|(s, o)| *s += o));
        }
    }

    impl<T: Float> std::ops::Sub for Matrix3x3<T> {
        type Output = Matrix3x3<T>;

        fn sub(mut self, rhs: Self) -> Self::Output {
            self -= rhs;
            self
        }
    }

    impl<T: Float> std::ops::SubAssign for Matrix3x3<T> {
        fn sub_assign(&mut self, rhs: Self) {
            self.0.iter_mut().zip(rhs.0).for_each(|(s_arr, o_arr)| s_arr.iter_mut().zip(o_arr).for_each(|(s, o)| *s -= o));
        }
    }

    impl<T: Float> std::ops::Neg for Matrix3x3<T> {
        type Output = Self;

        fn neg(self) -> Self::Output {
            Self(self.0.map(|r| r.map(Complex::neg)))
        }
    }

    #[cfg(test)]
    mod tests {
        use super::Matrix3x3;

        #[test]
        fn test_matrix_display() {
            let matrix = Matrix3x3::from([[10., 0., 20.], [0., 30., 0.], [200., 0., 100.]]);
            assert_eq!("│\t10\t0\t20\t│\n│\t0\t30\t0\t│\n│\t200\t0\t100\t│\n", matrix.to_string());
        }

        #[test]
        fn test_zero_matrix() {
            let matrix = Matrix3x3::<f32>::new_zero();
            assert_eq!(Matrix3x3::from([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]), matrix);
        }

        #[test]
        fn test_identity_matrix() {
            let matrix = Matrix3x3::<f32>::new_identity();
            assert_eq!(Matrix3x3::from([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]), matrix);
        }

        #[test]
        fn test_matrix_mul() {
            let a = Matrix3x3::from([[1., 8., 3.], [9., 4., 5.], [6., 2., 7.]]);
            let b = Matrix3x3::from([[6., 7., 4.], [1., 3., 2.], [5., 9., 8.]]);

            assert_eq!(a * b, Matrix3x3::from([[29., 58., 44.], [83., 120., 84.], [73., 111., 84.]]));
        }

        #[test]
        fn test_matrix_mul_assign() {
            let mut a = Matrix3x3::from([[1., 8., 3.], [9., 4., 5.], [6., 2., 7.]]);
            let b = Matrix3x3::from([[6., 7., 4.], [1., 3., 2.], [5., 9., 8.]]);

            a *= b;

            assert_eq!(a, Matrix3x3::from([[29., 58., 44.], [83., 120., 84.], [73., 111., 84.]]));
        }

        #[test]
        fn test_matrix_add() {
            let a = Matrix3x3::from([[4., 3., 8.], [6., 2., 5.], [1., 5., 9.]]);
            let b = Matrix3x3::from([[-7., 13., 1.], [-49., 28., 28.], [28., -17., -10.]]);

            assert_eq!(a + b, Matrix3x3::from([[-3., 16., 9.], [-43., 30., 33.], [29., -12., -1.]]));
        }

        #[test]
        fn test_matrix_add_assign() {
            let mut a = Matrix3x3::from([[4., 3., 8.], [6., 2., 5.], [1., 5., 9.]]);
            let b = Matrix3x3::from([[-7., 13., 1.], [-49., 28., 28.], [28., -17., -10.]]);

            a += b;

            assert_eq!(a, Matrix3x3::from([[-3., 16., 9.], [-43., 30., 33.], [29., -12., -1.]]));
        }

        #[test]
        fn test_matrix_sub() {
            let a = Matrix3x3::from([[4., 3., 8.], [6., 2., 5.], [1., 5., 9.]]);
            let b = Matrix3x3::from([[-7., 13., 1.], [-49., 28., 28.], [28., -17., -10.]]);

            assert_eq!(a - b, Matrix3x3::from([[11., -10., 7.], [55., -26., -23.], [-27., 22., 19.]]));
        }

        #[test]
        fn test_matrix_sub_assign() {
            let mut a = Matrix3x3::from([[4., 3., 8.], [6., 2., 5.], [1., 5., 9.]]);
            let b = Matrix3x3::from([[-7., 13., 1.], [-49., 28., 28.], [28., -17., -10.]]);

            a -= b;

            assert_eq!(a, Matrix3x3::from([[11., -10., 7.], [55., -26., -23.], [-27., 22., 19.]]));
        }

        #[test]
        fn test_matrix_neg() {
            let a = Matrix3x3::from([[4., 3., 8.], [6., 2., 5.], [1., 5., 9.]]);
            assert_eq!(-a, Matrix3x3::from([[-4., -3., -8.], [-6., -2., -5.], [-1., -5., -9.]]));
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
    fn test_complex_norm() {
        let complex = Complex::from((3., 4.));
        assert_eq!(complex.norm(), 5.)
    }
    
    #[test]
    fn test_complex_conjugate() {
        let complex = Complex::from((3., 4.));
        assert_eq!(complex.conjugate(), Complex::from((3., -4.)))
    }
}
