pub trait Float:
    core::ops::Neg<Output = Self>
    + core::ops::Div<Output = Self>
    + core::ops::DivAssign
    + core::ops::Mul<Output = Self>
    + core::ops::MulAssign
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

    /// method to produce two
    fn two() -> Self;

    /// method to round off the value
    fn round(self) -> Self;

    /// calculates the absolute of the value
    fn abs(self) -> Self;

    /// calculates the arctan
    fn atan2(self, other: Self) -> Self;

    // returns the min of the two numbers
    fn min(self, other: Self) -> Self {
        match self > other {
            true => other,
            false => self,
        }
    }

    // returns the max of the two numbers
    fn max(self, other: Self) -> Self {
        match self < other {
            true => other,
            false => self,
        }
    }

    fn tolerence() -> Self;

    fn sin(self) -> Self;

    fn cos(self) -> Self;
}

impl Float for f32 {
    #[inline(always)]
    fn sqrt(self) -> Self {
        self.sqrt()
    }
    
    #[inline(always)]
    fn powi(self, n: i32) -> Self {
        self.powi(n)
    }
    
    #[inline(always)]
    fn zero() -> Self {
        0.
    }

    #[inline(always)]
    fn one() -> Self {
        1.
    }

    #[inline(always)]
    fn two() -> Self {
        2.
    }

    #[inline(always)]
    fn round(self) -> Self {
        self.round()
    }

    #[inline(always)]
    fn abs(self) -> Self {
        self.abs()
    }

    #[inline(always)]
    fn atan2(self, other: Self) -> Self {
        self.atan2(other)    
    }

    #[inline(always)]
    fn tolerence() -> Self {
        const TOLERENCE_F32: f32 = 1e-6;
        TOLERENCE_F32
    }

    #[inline(always)]
    fn sin(self) -> Self {
       self.sin() 
    }

    #[inline(always)]
    fn cos(self) -> Self {
        self.cos()
    }
}

impl Float for f64 {
    #[inline(always)]
    fn sqrt(self) -> Self {
        self.sqrt()
    }
    
    #[inline(always)]
    fn powi(self, n: i32) -> Self {
        self.powi(n)
    }
    
    #[inline(always)]
    fn zero() -> Self {
        0.
    }

    #[inline(always)]
    fn one() -> Self {
        1.
    }

    #[inline(always)]
    fn two() -> Self {
        2.
    }

    #[inline(always)]
    fn round(self) -> Self {
        self.round()
    }

    #[inline(always)]
    fn abs(self) -> Self {
        self.abs()
    }

    #[inline(always)]
    fn atan2(self, other: Self) -> Self {
        self.atan2(other)    
    }

    #[inline(always)]
    fn tolerence() -> Self {
        const TOLERENCE_F64: f64 = 1e-12;
        TOLERENCE_F64
    }

    #[inline(always)]
    fn sin(self) -> Self {
        self.sin()
    }

    #[inline(always)]
    fn cos(self) -> Self {
        self.cos()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Complex<T: Float = f32>(T, T);

impl<T: Float> Complex<T> {
    #[inline(always)]
    fn real(&self) -> T {
        self.0
    }
    
    #[inline(always)]
    fn imaginary(&self) -> T {
        self.1
    }
    
    /// ### Conjugate
    /// Returns the conjugate of the complex number
    ///
    /// #### Example
    /// ```rust
    /// use vector::Complex;
    ///
    /// let complex = Complex::from((3., 4.));
    /// assert_eq!(complex.conjugate(), Complex::from((3., -4.)))
    /// ```
    pub fn conjugate(mut self) -> Self {
        self.1 = -self.1;
        self
    }

    /// ### Argument
    /// Returns the argument of the complex number
    pub fn arg(&self) -> T {
        self.1.atan2(self.0)
    }
    
    /// ### Magnitude
    /// Calculates the magnitude of the complex number $a + ib$ given by:$$a^2 + b^2$$
    ///
    /// #### Example
    /// ```rust
    /// use vector::Complex;
    ///
    /// let complex = Complex::from((3., 4.));
    /// assert_eq!(complex.magnitude(), 5.)
    /// ```
    #[inline(always)]
    pub fn magnitude(&self) -> T {
        if self.is_real() {
            self.0.abs()
        }
        else if self.is_imaginary()  {
            self.1.abs()
        }
        else {
            self.norm_squared().sqrt()
        }
    }

    #[inline(always)]
    pub fn norm_squared(&self) -> T {
        self.0.powi(2) + self.1.powi(2)
    }

    fn round(self) -> Self {
        Self(self.0.round(), self.1.round())
    }

    /// normalizes the Complex number
    #[inline(always)]
    pub fn normalize(self) -> Self {
        self / self.magnitude()
    }

    /// ### Is real
    /// Returns `true` is the complex number has an imaginary value lesser than the tolerence.
    ///
    /// #### Example
    /// ```rust
    /// use vector::Complex;
    ///
    /// let complex = Complex::from(2.);
    /// assert_eq!(complex.is_real(), true);
    /// ```
    #[inline(always)]
    pub fn is_real(&self) -> bool {
        self.1.abs() <= T::tolerence()
    }

    /// ### Is imaginary
    /// Returns `true` is the complex number has an real value lesser than the tolerence.
    ///
    /// #### Example
    /// ```rust
    /// use vector::Complex;
    ///
    /// let complex = Complex::from((0., 2.));
    /// assert_eq!(complex.is_imaginary(), true);
    /// ```
    #[inline(always)]
    pub fn is_imaginary(&self) -> bool {
        self.0.abs() <= T::tolerence()
    }

    /// ### Is zero
    /// Returns `true` if both the real and imaginary parts have values lerrer than the tolerence
    ///
    /// #### Example
    /// ```rust
    /// use vector::Complex;
    ///
    /// let complex = Complex::from((0., 0.));
    /// assert_eq!(complex.is_zero(), true);
    /// ```
    #[inline(always)]
    pub fn is_zero(&self) -> bool {
        self.is_imaginary() && self.is_real()
    }
}

/// ## Divide
impl<T: Float> core::ops::Div for Complex<T> {
    type Output = Self;
    
    /// ```rust
    /// use vector::Complex;
    ///
    /// let a = Complex::from((1., 2.));
    /// let b = Complex::from((3., 4.));
    ///  
    /// assert_eq!(a / b, Complex::from((-0.2, 0.4)))
    /// ```
    fn div(mut self, other: Complex<T>) -> Self::Output {
        self /= other;
        self
    }
}

impl<T: Float> core::ops::Div<T> for Complex<T> {
    type Output = Self;

    /// ```rust
    /// use vector::Complex;
    ///
    /// let a = Complex::from((1., 2.));
    ///
    /// assert_eq!(a / 2., Complex::from((0.5, 1.)))
    /// ```
    fn div(mut self, rhs: T) -> Self::Output {
        self /= rhs;
        self
    }
}

/// ## Divide assign
impl<T: Float> core::ops::DivAssign for Complex<T> {
    /// ```rust
    /// use vector::Complex;
    ///
    /// let mut a = Complex::from((1., 2.));
    /// let b = Complex::from((3., 4.));
    ///  
    /// a /= b;
    /// assert_eq!(a, Complex::from((-0.2, 0.4)))
    /// ```
    fn div_assign(&mut self, rhs: Self) {
        let mag = rhs.0.powi(2) + rhs.1.powi(2); 
        *self = Self((self.0 * rhs.0 - self.1 * rhs.1) / mag, (self.0 * rhs.1 + self.1 * rhs.0) / mag);
    }
}

/// ## Divide assign
impl<T: Float> core::ops::DivAssign<T> for Complex<T> {
    /// ```rust
    /// use vector::Complex;
    ///
    /// let mut a = Complex::from((1., 2.));
    ///  
    /// a /= 2.;
    /// assert_eq!(a, Complex::from((0.5, 1.)))
    /// ```
    fn div_assign(&mut self, rhs: T) {
        self.0 /= rhs;
        self.1 /= rhs;
    }
}

impl<T: Float> core::ops::Mul for Complex<T> {
    type Output = Self;
    
    /// ```rust
    /// use vector::Complex;
    ///
    /// let a = Complex::from((1., 2.));
    /// let b = Complex::from((3., 4.));
    /// 
    /// assert_eq!(a * b, Complex::from((-5., 10.)))
    /// ```
    fn mul(mut self, other: Complex<T>) -> Self::Output {
        self *= other;
        self
    }
}

impl<T: Float> core::ops::Mul<T> for Complex<T> {
    type Output = Self;
    
    /// ```rust
    /// use vector::Complex;
    ///
    /// let a = Complex::from((1., 2.));
    /// 
    /// assert_eq!(a * 2., Complex::from((2., 4.)))
    /// ```
    fn mul(mut self, other: T) -> Self::Output {
       self *= other;
       self
    }
}

impl<T: Float> core::ops::Mul<&Complex<T>> for Complex<T> {
    type Output = Complex<T>;

    /// ```rust
    /// use vector::Complex;
    ///
    /// let a = Complex::from((1., 2.));
    /// let b = Complex::from((3., 4.));
    /// 
    /// assert_eq!(a * &b, Complex::from((-5., 10.)))
    /// ```
    fn mul(mut self, rhs: &Complex<T>) -> Self::Output {
        self *= rhs;
        self
    }
}

impl<T: Float> core::ops::Mul<&Complex<T>> for &Complex<T> {
    type Output = Complex<T>;

    /// ```rust
    /// use vector::Complex;
    ///
    /// let a = Complex::from((1., 2.));
    /// let b = Complex::from((3., 4.));
    /// 
    /// assert_eq!(&a * &b, Complex::from((-5., 10.)))
    /// ```
    fn mul(self, rhs: &Complex<T>) -> Self::Output {
        let mut complex = self.to_owned();
        complex *= rhs;
        complex
    }
}

impl<T: Float> core::ops::MulAssign for Complex<T> {
    /// ```rust
    /// use vector::Complex;
    ///
    /// let mut a = Complex::from((1., 2.));
    /// let b = Complex::from((3., 4.));
    /// 
    /// a *= b;
    ///
    /// assert_eq!(a, Complex::from((-5., 10.)))
    /// ```
    fn mul_assign(&mut self, rhs: Self) {
        *self = Self(self.0 * rhs.0 - self.1 * rhs.1, self.0 * rhs.1 + self.1 * rhs.0);
    }
}

impl<T: Float> core::ops::MulAssign<&Complex<T>> for Complex<T> {
    /// ```rust
    /// use vector::Complex;
    ///
    /// let mut a = Complex::from((1., 2.));
    /// let b = Complex::from((3., 4.));
    /// 
    /// a *= &b;
    ///
    /// assert_eq!(a, Complex::from((-5., 10.)))
    /// ```
    fn mul_assign(&mut self, rhs: &Complex<T>) {
        *self = Self(self.0 * rhs.0 - self.1 * rhs.1, self.0 * rhs.1 + self.1 * rhs.0);
    }
}

impl<T: Float> core::ops::MulAssign<T> for Complex<T> {
    /// ```rust
    /// use vector::Complex;
    ///
    /// let mut a = Complex::from((1., 2.));
    /// 
    /// a *= 2.;
    ///
    /// assert_eq!(a, Complex::from((2., 4.)))
    /// ```
    fn mul_assign(&mut self, rhs: T) {
        self.0 *= rhs;
        self.1 *= rhs;
    }
}

impl<T: Float> core::ops::Add for Complex<T> {
    type Output = Self;
    
    /// ```rust
    /// use vector::Complex;
    ///
    /// let a = Complex::from((1., 2.));
    /// let b = Complex::from((3., 4.));
    /// 
    /// assert_eq!(a + b, Complex::from((4., 6.)))
    /// ```
    fn add(mut self, other: Complex<T>) -> Self::Output {
        self += other;
        self
    }  
}

impl<T: Float> core::ops::AddAssign for Complex<T> {
    /// ```rust
    /// use vector::Complex;
    ///
    /// let mut a = Complex::from((1., 2.));
    /// let b = Complex::from((3., 4.));
    /// 
    /// a += b;
    ///
    /// assert_eq!(a, Complex::from((4., 6.)))
    /// ```
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
        self.1 += rhs.1;
    }
}

impl<T: Float> core::iter::Sum for Complex<T> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|a, b| a + b).unwrap()
    }
}

impl<T: Float> core::ops::Sub for Complex<T> {
    type Output = Self;
    
    /// ```rust
    /// use vector::Complex;
    ///
    /// let a = Complex::from((1., 2.));
    /// let b = Complex::from((3., 4.));
    /// 
    /// assert_eq!(a - b, Complex::from((-2., -2.)))
    /// ```
    fn sub(mut self, other: Complex<T>) -> Self::Output {
        self -= other;
        self
    }  
}

impl<T: Float> core::ops::SubAssign for Complex<T> {
    /// ```rust
    /// use vector::Complex;
    ///
    /// let mut a = Complex::from((1., 2.));
    /// let b = Complex::from((3., 4.));
    /// 
    /// a -= b;
    /// assert_eq!(a, Complex::from((-2., -2.)))
    /// ```
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
        self.1 -= rhs.1;
    }
}

/// ### Negate
/// Negates the real and imaginary part of the complex number
impl<T: Float> core::ops::Neg for Complex<T> {
    type Output = Self;

    /// #### Example
    ///
    /// ```rust
    /// use vector::Complex;
    ///
    /// let complex = Complex::from((1., -2.));
    /// assert_eq!(-complex, Complex::from((-1., 2.)));
    /// ```
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


/// ### Display
impl<T: Float> std::fmt::Display for Complex<T> {
    /// ```rust
    /// use vector::Complex;
    ///
    /// let mut complex = Complex::from(1.0);
    /// assert_eq!(complex.to_string(), "1".to_string());
    /// 
    /// complex = Complex::from(1.33);
    /// assert_eq!(complex.to_string(), "1.33".to_string());
    ///      
    /// complex = Complex::from((1.33, 20.));
    /// assert_eq!(complex.to_string(), "1.33 + 20j".to_string());
    ///      
    /// complex = Complex::from((1.33, 20.55));
    /// assert_eq!(complex.to_string(), "1.33 + 20.55j".to_string());
    /// ```
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

pub mod matrix {
    use super::{Float, Complex};

    #[derive(Debug)]
    pub enum MatrixError {
        Singular,
        Dependent,
    }

    impl std::fmt::Display for MatrixError {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            match self {
                Self::Singular => write!(f, "Matrix is singular"),
                Self::Dependent => write!(f, "Matrix has dependent columns"),
            }
        }
    }

    impl std::error::Error for MatrixError {}

    /// Defines a matrix
    #[derive(Debug, PartialEq, Clone)]
    pub struct Matrix<const R: usize, const C: usize, T: Float = f32>([[Complex<T>; C]; R]);

    impl<T: Float, const R: usize, const C: usize> Matrix<R, C, T> {
        /// Creates a new matrix with zero entries
        /// ### Example
        /// ```rust
        /// use vector::matrix::Matrix;
        ///
        /// let matrix = Matrix::new_zero();
        /// assert_eq!(Matrix::from([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]), matrix);
        /// ```
        pub fn new_zero() -> Self {
            assert_ne!((R, C), (0, 0), "Cannot create a matrix with no dimensions");
            Self([[T::zero().into(); C]; R])
        }

        /// returns true if the entries of the matrix are zero within tolerence
        #[inline]
        pub fn is_zero(&self) -> bool {
            self.0.iter().flatten().all(Complex::is_zero)
        }

        /// Returns a new transposed matrix.
        ///
        /// ### Example
        /// ```rust
        /// use vector::matrix::Matrix;
        ///
        /// let a = Matrix::from([[(4., 0.55), (4., -20.1901), (-4., 3.696969)], [(4., -1.4000006), (8., 15.919191), (8., 22.2)], [(0., 40.00001), (12., -92.99999), (16., 8.867193)]]);
        /// let a_c_t = Matrix::from([[(4., 0.55), (4., -1.4000006), (0., 40.00001)], [(4., -20.1901), (8., 15.919191), (12., -92.99999)], [(-4., 3.696969), (8., 22.2), (16., 8.867193)]]);
        ///
        /// assert_eq!(a.transpose(), a_c_t);
        /// ```
        pub fn transpose(&self) -> Matrix<C, R, T> {
            (0..R).flat_map(|i| (0..C).map(move |j| (i, j))).fold(Matrix::new_zero(), |mut acc, (i, j)| {
                acc.0[j][i] = self.0[i][j];
                acc
            })
        }

        /// Returns a new conjugate transposed matrix.
        /// Also known as **Hermitian transpose**.
        ///
        /// ### Example
        /// ```rust
        /// use vector::matrix::Matrix;
        ///
        /// let a = Matrix::from([[(4., 0.55), (4., -20.1901), (-4., 3.696969)], [(4., -1.4000006), (8., 15.919191), (8., 22.2)], [(0., 40.00001), (12., -92.99999), (16., 8.867193)]]);
        /// let a_c_t = Matrix::from([[(4., -0.55), (4., 1.4000006), (0., -40.00001)], [(4., 20.1901), (8., -15.919191), (12., 92.99999)], [(-4., -3.696969), (8., -22.2), (16., -8.867193)]]);
        ///
        /// assert_eq!(a.conjugate_transpose(), a_c_t);
        /// ```
        pub fn conjugate_transpose(&self) -> Matrix<C, R, T> {
            (0..R).flat_map(|i| (0..C).map(move |j| (i, j))).fold(Matrix::new_zero(), |mut acc, (i, j)| {
                acc.0[j][i] = self.0[i][j].conjugate();
                acc
            })
        }

        /// ## Inner product
        /// Calculates the inner product of two matrix.
        /// $$<A, B> = \mbox{trace}(A^HB)$$
        /// where, $A^H$ is the *conjugate transpose* or the *Hermitian transpose*.
        pub fn inner_product(&self, mut other: Self) -> Complex<T> {
            other.0.iter_mut().for_each(|r| r.iter_mut().for_each(|c| *c = c.conjugate()));
            let mat = self.conjugate_transpose() * other;
            mat.trace()
        }

        /// ## Rank
        /// Calculates the **RANK** of the given matrix. It the number of independent columns/rows.
        /// ### Complexity
        /// $O(R \times C \times \mbox{rank})$
        ///
        /// ### Example
        /// ```rust
        /// use vector::matrix::Matrix;
        ///
        /// let a = Matrix::from([[10., 20., 10.], [-20., -30., 10.], [30., 50., 0.]]);
        ///
        /// assert_eq!(a.rank(), 2);
        /// ```
        pub fn rank(&self) -> usize {
            let (mut matrix, mut rank) = (self.0, C);

            let mut i = 0;
            let zero = T::zero().into();
            while i < rank {
                if matrix[i][i] != zero {
                    (0..R).filter(|&r| r != i).for_each(|r| {
                        let mult = matrix[r][i] / matrix[i][i];
                        (0..rank).for_each(|c| matrix[r][c] -= mult * matrix[i][c]);
                    });

                    i += 1;
                }
                else {
                    // find non-zero row
                    match (i + 1..C).find(|&r| matrix[r][i] != zero) {
                        Some(r) => (0..rank).for_each(|c| (matrix[r][c], matrix[i][c]) = (matrix[i][c], matrix[r][c])),
                        None => {
                            // reduce rank
                            rank -= 1;
                            // copy the last column (rank) here
                            (0..R).for_each(|r| matrix[r][i] = matrix[r][rank]);
                        },
                    }
                }
            }
            
            rank
        }

        /// rounds off the values of the matrix
        pub fn round(mut self) -> Self {
            self.0.iter_mut().for_each(|r| r.iter_mut().for_each(|c| *c = c.round()));
            self
        }

        /// ## QR Decomposition
        /// Decomposes the given matrix into two matrices using QR decomposition.
        /// Let there be a matrix $A_{R \times C}$.
        /// $$A_{R \times C} = Q_{R \times C} R_{C \times C}$$
        /// Error is returned if any of the columns are dependent.
        ///
        /// **NOTE:** The matrix Q represents the Gram Schmidt matrix.
        ///
        /// ### Uses
        /// - Solving least squares problem
        /// - Eigen value computation
        ///
        /// ### Example
        /// ```rust
        /// use vector::matrix::Matrix;
        ///
        /// let a = Matrix::from([[12., -51., 4.], [6., 167., -68.], [-4., 24., -41.]]);
        /// let (q, r) = a.qr().unwrap();
        ///
        /// assert_eq!(q, Matrix::from([[-0.857142857142857, 0.394285714285714, -0.3314285714285714], [-0.42857142857142855, -0.9028571428571422, 0.03428571428571427], [0.2857142857142857, -0.17142857142857137, -0.942857142857143]]));
        /// assert_eq!(r, Matrix::from([[-13.999999999999998, -21.000000000000004, 14.000000000000002], [-0.0000000000000007783517420333592, -174.9999999999999, 69.99999999999996], [0.0000000000000005335902659890752, -0.000000000000007105427357601002, 35.]]));
        /// ```
        ///
        /// ```rust, should_panic
        /// use vector::matrix::Matrix;
        ///
        /// // Dependent columns
        /// let a = Matrix::from([[3., 3.], [1., 1.]]);
        /// a.qr().unwrap();
        /// ```
        pub fn qr(&self) -> Result<(Matrix<R, R, T>, Matrix<R, C, T>), MatrixError> {
            let mut r = self.to_owned();
            let mut q = Matrix::<R, R, T>::new_identity();

            let min = C.min(R);
            for i in 0..min {
                let norm_x_squared = (i..R).map(|k| r.0[k][i].norm_squared()).sum::<T>();
                let norm = norm_x_squared.sqrt();
                if norm < T::tolerence() {
                    return Err(MatrixError::Dependent);
                }

                let alpha = r.0[i][i].normalize() * norm;
                let u_norm = (norm_x_squared + alpha.norm_squared() + T::two() * r.0[i][i].real() * alpha.real() + T::two() * r.0[i][i].imaginary() * alpha.imaginary()).sqrt();

                let mut u_cache: [Option<Complex<T>>; R] = core::array::from_fn(|_| None);
                let mut u = |k: usize| match u_cache[k] { // k ranges from i..R
                    Some(res) => res,
                    None => {
                        let mut u_k = r.0[k][i];
                        if k == i {
                            u_k += alpha;
                        }

                        let res = u_k / u_norm;
                        u_cache[k] = Some(res);
                        res
                    },
                };

                // create q
                let mut q_loc = Matrix::<R, R, T>::new_identity();
                (i..R).flat_map(|row| (i..C).map(move |col| (row, col))).for_each(|(row, col)| q_loc.0[row][col] = match row == col {
                    true => {
                        let u = u(row); 
                        (T::one() - T::two() * u.norm_squared()).into()
                    },
                    false => -u(row) * u(col).conjugate() * T::two(),
                });

                r = &q_loc * r;
                q *= q_loc.conjugate_transpose();
            }
            
            Ok((q, r))
        }

        /// ## Column normalization
        /// Method to normalize the columns of a given matrix
        /// ### Example
        /// ```rust
        /// use vector::matrix::Matrix;
        ///
        /// let mut a = Matrix::from([[12., -51., 4.], [6., 167., -68.], [-4., 24., -41.]]);
        /// a.normalize_columns();
        /// let normalized_a = Matrix::from([[0.8571428571428571, -0.2893526786447601, 0.05031148036146422], [0.42857142857142855, 0.9474881830132341, -0.8552951661448918], [-0.2857142857142857, 0.1361659664210636, -0.5156926737050083]]);
        ///
        /// assert_eq!(normalized_a, a);
        /// ```
        pub fn normalize_columns(&mut self) {
            for c in 0..C {
                let sum = (0..R).map(|r| self.0[r][c].norm_squared()).sum::<T>().sqrt();
                (0..R).for_each(|r| self.0[r][c] /= sum);
            }
        }

        /// ## Normal columns check
        /// Returns true if the columns of the matrix are normalized
        pub fn has_normal_columns(&self) -> bool {
            (0..C).map(|c| (0..R).map(|r| self.0[r][c].norm_squared()).sum::<T>().sqrt()).all(|sum| (sum - T::one()).abs() <= T::tolerence())
        }
    }

    impl<T: Float, const C: usize> Matrix<C, C, T> {
        /// Creates a new **identity matrix**.
        /// ```rust
        /// use vector::matrix::Matrix;
        ///
        /// let matrix = Matrix::new_identity();
        /// assert_eq!(Matrix::from([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]), matrix);
        /// ```
        pub fn new_identity() -> Self {
            assert_ne!(C, 0, "Cannot create a matrix with no dimensions");
            Self(std::array::from_fn(|i| 
                    std::array::from_fn(|j| match j == i {
                        true => T::one(),
                        false => T::zero(),
                    }.into())
            ))
        }

        /// Calculates the sum of the diagonal elements of the matrix
        fn trace(&self) -> Complex<T> {
            (0..C).map(|i| self.0[i][i]).sum()
        }

        /// Calculates the determinant of the given square matrix
        ///
        /// ### Procedure
        /// - Uses Gaussian elemination and transformations to reduce the matrix to upper triangular form.
        /// - The determinant is then the product of all diagonal elements.
        ///
        /// ### Complexity
        /// $O(C^3)$
        ///
        /// ### Example
        /// ```rust
        /// use vector::{matrix::Matrix, Complex};
        ///
        /// let a = Matrix::from([[0., 12., 16.], [4., 4., -4.], [4., 8., 8.]]);
        /// assert_eq!(a.determinant(), Complex::from(-320.))
        /// ```
        pub fn determinant(&self) -> Complex<T> {
            let mut matrix = self.0;
            let zero = Complex::from(T::zero());
            let [mut det, mut total] = [Complex::from(T::one()); 2];

            for diag_row in 0..C {
                // swap row with non-zero element
                let Some(non_zero_row) = (diag_row..C).find(|&i| matrix[i][diag_row] != zero) else { continue; };
                if non_zero_row != diag_row {
                    // swap rows
                    (matrix[non_zero_row], matrix[diag_row]) = (matrix[diag_row], matrix[non_zero_row]);

                    // change sign if odd
                    if (non_zero_row - diag_row) & 1 == 1 {
                        det = -det;
                    }
                }

                // transform every row below diag_row
                let temp: [Complex<T>; C] = std::array::from_fn(|j| matrix[diag_row][j]);
                for row in matrix.iter_mut().skip(diag_row + 1) {
                    let num2 = row[diag_row];
                    row.iter_mut().zip(temp).for_each(|(k, ele)| *k = (temp[diag_row] * *k) - (num2 * ele));
                    total *= temp[diag_row];
                }
            }

            // multiply diagonal elements
            (0..C).for_each(|i| det *= matrix[i][i]);
            det / total
        }

        /// Determines the inverse of a matrix
        /// ### Procedure
        /// Gaussian Jordan Elemination Method
        ///
        /// ### Complexity
        /// $O(C^3)$
        ///
        /// ### Note
        /// Since we are not using rationals, **floating point inacuracy** might be encountered.
        /// This means, the soulution might output 
        ///
        /// ### Example
        /// 1. Non-singular matrix
        ///
        /// ```rust
        /// use vector::matrix::Matrix;
        ///
        /// const PRECISION: f32 = 1e2;
        ///
        /// let a = Matrix::from([[2., -1., 0.], [-1., 2., -1.], [0., -1., 2.]]);
        /// let inverse = {
        ///     let matrix = a.inverse().unwrap();
        ///     (matrix * PRECISION).round() / PRECISION
        /// };
        ///
        /// assert_eq!(inverse, Matrix::from([[0.75, 0.5, 0.25], [0.5, 1., 0.5], [0.25, 0.5, 0.75]]))
        /// ```
        /// 2. Singular matrix
        ///
        /// ```rust, should_panic
        /// use vector::matrix::Matrix;
        ///
        /// let a = Matrix::from([[1., -2., 2.], [-1., 2., -2.], [3., -2., -1.]]);
        /// a.inverse().unwrap();
        /// ```
        pub fn inverse(&self) -> Result<Self, MatrixError> {
            let mut matrix = self.0;
            let mut inverse = Self::new_identity();

            for i in 0..C {
                let max_row = match (i..C).max_by(|&a, &b| {
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
                (0..C).for_each(|c| {
                    matrix[i][c] /= pivot;
                    inverse.0[i][c] /= pivot;
                });

                (0..C).filter(|&r| r != i).for_each(|r| {
                    let factor = matrix[r][i];
                    (0..C).for_each(|k| {
                        matrix[r][k] -= factor * matrix[i][k];
                        inverse.0[r][k] -= factor * inverse.0[i][k];
                    });
                });
            }

            Ok(inverse)
        }
    }

    impl<T: Float, const R: usize, const C: usize> core::ops::Index<usize> for Matrix<R, C, T> {
        type Output = [Complex<T>; C];

        /// Method to enable matrix indexing
        fn index(&self, index: usize) -> &Self::Output {
            &self.0[index]
        }
    }

    impl<T: Float, const R: usize, const C: usize> core::ops::IndexMut<usize> for Matrix<R, C, T> {
        /// Method to enable mutable matrix indexing
        fn index_mut(&mut self, index: usize) -> &mut Self::Output {
            &mut self.0[index]
        }
    }

    /// Create a new Matrix by taking ownership of the 2 dimensional array
    impl<T: Float, Z: Into<Complex<T>>, const R: usize, const C: usize> From<[[Z; C]; R]> for Matrix<R, C, T> {
        fn from(value: [[Z; C]; R]) -> Self {
            assert_ne!((R, C), (0, 0), "Cannot create a matrix with no dimensions");
            Self(value.map(|r| r.map(|c| c.into())))
        }
    }

    /// Create a new Matrix from a 2 dimensional slice
    impl<T: Float, Z: Into<Complex<T>> + Clone, const R: usize, const C: usize> From<&[&[Z; C]; R]> for Matrix<R, C, T> {
        fn from(value: &[&[Z; C]; R]) -> Self {
            assert_ne!((R, C), (0, 0), "Cannot create a matrix with no dimensions");
            Self(core::array::from_fn(|r| core::array::from_fn(|c| value[r][c].to_owned().into())))
        }
    }

    /// Create a new Matrix from a 2 dimensional complex
    impl<T: Float, const R: usize, const C: usize> From<[[&Complex<T>; C]; R]> for Matrix<R, C, T> {
        fn from(value: [[&Complex<T>; C]; R]) -> Self {
            assert_ne!((R, C), (0, 0), "Cannot create a matrix with no dimensions");
            Self(value.map(|row| row.map(|&v| v)))
        }
    }

    /// ## Matrix display
    impl<T: Float, const R: usize, const C: usize> core::fmt::Display for Matrix<R, C, T> {
        /// ### Example
        ///
        /// Consider the matrix:$$\begin{bmatrix} 10 & 0 & 20 \\\ 0 & 30 & 0 \\\ 200 & 0 & 100 \end{bmatrix}$$
        /// ```rust
        /// use vector::matrix::Matrix;
        ///
        /// let matrix = Matrix::from([[10., 0., 20.], [0., 30., 0.], [200., 0., 100.]]);
        /// assert_eq!("│\t10\t0\t20\t│\n│\t0\t30\t0\t│\n│\t200\t0\t100\t│\n", matrix.to_string());
        /// ```
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            for i in 0..R {
                write!(f, "│\t")?;
                for j in 0..C {
                    write!(f, "{}\t", self.0[i][j])?;
                }
                writeln!(f, "│")?;
            }

            Ok(())
        }
    }

    /// ## Dividing matrx by Complex
    /// This divides each entry by the given RHS value.
    impl<T: Float, Z: Into<Complex<T>>, const R: usize, const C: usize> core::ops::Div<Z> for Matrix<R, C, T> {
        type Output = Self;

        fn div(mut self, rhs: Z) -> Self::Output {
            self /= rhs;
            self
        }
    }

    impl<T: Float, Z: Into<Complex<T>>, const R: usize, const C: usize> core::ops::DivAssign<Z> for Matrix<R, C, T> {
        fn div_assign(&mut self, rhs: Z) {
            let val = rhs.into();
            self.0.iter_mut().for_each(|r| r.iter_mut().for_each(|c| *c /= val));
        }
    }

    /// ## Matrix multiplication
    impl<T: Float, const R: usize, const C: usize, const O: usize> core::ops::Mul<Matrix<C, O, T>> for Matrix<R, C, T> {
        type Output = Matrix<R, O, T>;

        /// ### Example
        ///
        /// ```rust
        /// use vector::matrix::Matrix;
        ///
        /// let a = Matrix::from([[4., 1., -4., 5.], [-2., 0., 6., 3.], [2., 7., 8., 9.], [10., -1., -3., 11.]]);
        /// let b = Matrix::from([[4., 1.], [-2., 0.], [2., 7.], [10., 12.]]);
        ///
        /// assert_eq!(a * b, Matrix::from([[56., 36.], [34., 76.], [100., 166.], [146., 121.]]));
        /// ```
        fn mul(self, rhs: Matrix<C, O, T>) -> Self::Output {
            self * &rhs
        }
    }

    /// ## Matrix multiplication
    impl<T: Float, const R: usize, const C: usize, const O: usize> core::ops::Mul<&Matrix<C, O, T>> for Matrix<R, C, T> {
        type Output = Matrix<R, O, T>;

        /// ### Example
        ///
        /// ```rust
        /// use vector::matrix::Matrix;
        ///
        /// let a = Matrix::from([[4., 1., -4., 5.], [-2., 0., 6., 3.], [2., 7., 8., 9.], [10., -1., -3., 11.]]);
        /// let b = Matrix::from([[4., 1.], [-2., 0.], [2., 7.], [10., 12.]]);
        ///
        /// assert_eq!(a * &b, Matrix::from([[56., 36.], [34., 76.], [100., 166.], [146., 121.]]));
        /// ```
        fn mul(self, rhs: &Matrix<C, O, T>) -> Self::Output {
            let matrix = std::array::from_fn(|i| 
                std::array::from_fn(|j| 
                    (0..C).map(|k| self.0[i][k] * rhs.0[k][j]).sum::<Complex<T>>()
                )
            );

            Matrix(matrix)
        }
    }

    /// ## Matrix multiplication
    impl<T: Float, const R: usize, const C: usize, const O: usize> core::ops::Mul<Matrix<C, O, T>> for &Matrix<R, C, T> {
        type Output = Matrix<R, O, T>;

        /// ### Example
        ///
        /// ```rust
        /// use vector::matrix::Matrix;
        ///
        /// let a = Matrix::from([[4., 1., -4., 5.], [-2., 0., 6., 3.], [2., 7., 8., 9.], [10., -1., -3., 11.]]);
        /// let b = Matrix::from([[4., 1.], [-2., 0.], [2., 7.], [10., 12.]]);
        ///
        /// assert_eq!(&a * b, Matrix::from([[56., 36.], [34., 76.], [100., 166.], [146., 121.]]));
        /// ```
        fn mul(self, rhs: Matrix<C, O, T>) -> Self::Output {
            let matrix = std::array::from_fn(|i| 
                std::array::from_fn(|j| 
                    (0..C).map(|k| self.0[i][k] * rhs.0[k][j]).sum::<Complex<T>>()
                )
            );

            Matrix(matrix)
        }
    }

    /// ## Matrix multiplication
    impl<T: Float, const R: usize, const C: usize, const O: usize> core::ops::Mul<&Matrix<C, O, T>> for &Matrix<R, C, T> {
        type Output = Matrix<R, O, T>;

        /// ### Example
        ///
        /// ```rust
        /// use vector::matrix::Matrix;
        ///
        /// let a = Matrix::from([[4., 1., -4., 5.], [-2., 0., 6., 3.], [2., 7., 8., 9.], [10., -1., -3., 11.]]);
        /// let b = Matrix::from([[4., 1.], [-2., 0.], [2., 7.], [10., 12.]]);
        ///
        /// assert_eq!(&a * &b, Matrix::from([[56., 36.], [34., 76.], [100., 166.], [146., 121.]]));
        /// ```
        fn mul(self, rhs: &Matrix<C, O, T>) -> Self::Output {
            let matrix = std::array::from_fn(|i| 
                std::array::from_fn(|j| 
                    (0..C).map(|k| self.0[i][k] * rhs.0[k][j]).sum::<Complex<T>>()
                )
            );

            Matrix(matrix)
        }
    }

    impl <T: Float, const R: usize, const C: usize> core::ops::Mul<&Matrix<R, C, T>> for Complex<T> {
        type Output = Matrix<R, C, T>;

        /// ### Example
        ///
        /// ```rust
        /// use vector::{matrix::Matrix, Complex};
        ///
        /// let a = Matrix::from([[1., 8., 3.], [9., 4., 5.], [6., 2., 7.]]);
        /// assert_eq!(Complex::from(2.) * &a, Matrix::from([[2., 16., 6.], [18., 8., 10.], [12., 4., 14.]]));
        /// ```
        fn mul(self, rhs: &Matrix<R, C, T>) -> Self::Output {
            rhs.clone() * self
        }
    }

    /// ## Matrix multiplication by a scalar at lhs
    impl<T: Float, Z: Into<Complex<T>>, const R: usize, const C: usize> core::ops::Mul<Z> for Matrix<R, C, T> {
        type Output = Self;

        /// ### Example
        ///
        /// ```rust
        /// use vector::matrix::Matrix;
        ///
        /// let a = Matrix::from([[1., 8., 3.], [9., 4., 5.], [6., 2., 7.]]);
        /// assert_eq!(a * 2., Matrix::from([[2., 16., 6.], [18., 8., 10.], [12., 4., 14.]]));
        /// ```
        fn mul(mut self, rhs: Z) -> Self::Output {
            self *= rhs;
            self
        }
    }

    /// ## Matrix multiplication by a complex at rhs
    impl<T: Float, const R: usize, const C: usize> core::ops::Mul<Matrix<R, C, T>> for Complex<T> {
        type Output = Matrix<R, C, T>;

        fn mul(self, mut rhs: Matrix<R, C, T>) -> Self::Output {
            rhs *= self;
            rhs
        } 
    }

    /// ## Matrix multiplication and assignment
    /// **Only applicable for square matrices**
    impl<T: Float, const C: usize> core::ops::MulAssign for Matrix<C, C, T> {
        /// ### Example
        /// 
        /// ```rust
        /// use vector::matrix::Matrix;
        ///
        /// let a = Matrix::from([[1., 8., 3.], [9., 4., 5.], [6., 2., 7.]]);
        /// let b = Matrix::from([[6., 7., 4.], [1., 3., 2.], [5., 9., 8.]]);
        /// 
        /// assert_eq!(a * b, Matrix::from([[29., 58., 44.], [83., 120., 84.], [73., 111., 84.]]));
        /// ```
        fn mul_assign(&mut self, rhs: Self) {
            let matrix = std::array::from_fn(|i| std::array::from_fn(|j| (0..C).map(|k| self.0[i][k] * rhs.0[k][j]).sum::<Complex<T>>()));
            *self = Self(matrix);
        }
    }

    impl<T: Float, Z: Into<Complex<T>>, const R: usize, const C: usize> core::ops::MulAssign<Z> for Matrix<R, C, T> {
        fn mul_assign(&mut self, rhs: Z) {
            let val = rhs.into();
            self.0.iter_mut().for_each(|r| r.iter_mut().for_each(|c| *c *= val));
        }
    }

    /// ## Matrix addition
    impl<T: Float, const R: usize, const C: usize> core::ops::Add for Matrix<R, C, T> {
        type Output = Self;

        /// ### Example
        ///
        /// ```rust
        /// use vector::matrix::Matrix;
        ///
        /// let a = Matrix::from([[4., 3., 8.], [6., 2., 5.], [1., 5., 9.]]);
        /// let b = Matrix::from([[-7., 13., 1.], [-49., 28., 28.], [28., -17., -10.]]);
        ///
        /// assert_eq!(a + b, Matrix::from([[-3., 16., 9.], [-43., 30., 33.], [29., -12., -1.]]));
        /// ```
        fn add(mut self, rhs: Self) -> Self::Output {
            self += rhs;
            self
        }
    }

    /// ## Matrix addition assign
    impl<T: Float, const R: usize, const C: usize> core::ops::AddAssign for Matrix<R, C, T> {
        /// ### Example
        ///
        /// ```rust
        /// use vector::matrix::Matrix;
        ///
        /// let mut a = Matrix::from([[4., 3., 8.], [6., 2., 5.], [1., 5., 9.]]);
        /// let b = Matrix::from([[-7., 13., 1.], [-49., 28., 28.], [28., -17., -10.]]);
        ///
        /// a += b;
        ///
        /// assert_eq!(a, Matrix::from([[-3., 16., 9.], [-43., 30., 33.], [29., -12., -1.]]));
        /// ```
        fn add_assign(&mut self, rhs: Self) {
            self.0.iter_mut().zip(rhs.0).for_each(|(s_arr, o_arr)| s_arr.iter_mut().zip(o_arr).for_each(|(s, o)| *s += o));
        }
    }

    /// ## Matrix subtraction
    impl<T: Float, const R: usize, const C: usize> core::ops::Sub for Matrix<R, C, T> {
        type Output = Self;

        /// ### Example
        ///
        /// ```rust
        /// use vector::matrix::Matrix;
        ///
        /// let a = Matrix::from([[4., 3., 8.], [6., 2., 5.], [1., 5., 9.]]);
        /// let b = Matrix::from([[-7., 13., 1.], [-49., 28., 28.], [28., -17., -10.]]);
        ///
        /// assert_eq!(a - b, Matrix::from([[11., -10., 7.], [55., -26., -23.], [-27., 22., 19.]]));
        /// ```
        fn sub(mut self, rhs: Self) -> Self::Output {
            self -= rhs;
            self
        }
    }

    /// ## Matrix subtraction assign
    impl<T: Float, const R: usize, const C: usize> core::ops::SubAssign for Matrix<R, C, T> {
        /// ### Example
        ///
        /// ```rust
        /// use vector::matrix::Matrix;
        ///
        /// let mut a = Matrix::from([[4., 3., 8.], [6., 2., 5.], [1., 5., 9.]]);
        /// let b = Matrix::from([[-7., 13., 1.], [-49., 28., 28.], [28., -17., -10.]]);
        ///
        /// a -= b;
        /// assert_eq!(a, Matrix::from([[11., -10., 7.], [55., -26., -23.], [-27., 22., 19.]]));
        /// ```
        fn sub_assign(&mut self, rhs: Self) {
            self.0.iter_mut().zip(rhs.0).for_each(|(s_arr, o_arr)| s_arr.iter_mut().zip(o_arr).for_each(|(s, o)| *s -= o));
        }
    }

    /// ## Matrix negation
    impl<T: Float, const R: usize, const C: usize> core::ops::Neg for Matrix<R, C, T> {
        type Output = Self;

        /// ### Example
        ///
        /// ```rust
        /// use vector::matrix::Matrix;
        ///
        /// let a = Matrix::from([[4., 3., 8.], [6., 2., 5.], [1., 5., 9.]]);
        /// assert_eq!(-a, Matrix::from([[-4., -3., -8.], [-6., -2., -5.], [-1., -5., -9.]]));
        /// ```
        fn neg(self) -> Self::Output {
            Self(self.0.map(|r| r.map(Complex::neg)))
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        #[should_panic]
        fn new_dimensionless_zero_matrix() {
            Matrix::<0, 0, f32>::new_zero();
        }

        #[test]
        #[should_panic]
        fn new_dimensionless_identity_matrix() {
            Matrix::<0, 0, f32>::new_identity();
        }
    }
}

pub mod transformation {
    use core::fmt::Display;
    use crate::{matrix::Matrix, Complex, Float};

    type Result<T> = core::result::Result<T, Error>;

    // Method to make skew symmetric of a 3x1 matrix
    fn skew_symmetric<T: Float>(matrix: &Matrix<3, 1, T>) -> Matrix<3, 3, T> {
        let zero = T::zero().into();
        Matrix::from([
            [zero, -matrix[2][0], matrix[1][0]],
            [matrix[2][0], zero, -matrix[0][0]],
            [-matrix[1][0], matrix[0][0], zero],
        ])
    }

    #[derive(Debug)]
    pub struct Transformation<T: Float> {
        rotation: Matrix<3, 3, T>,
        translation: Matrix<3, 1, T>,
    }

    impl<T: Float> Default for Transformation<T> {
        /// - Gives a transformation matrix which performs no transformation.
        /// - It can also be thought of to be representing origin.
        /// - Identuty 4x4 matrix.
        ///
        /// ### Example
        /// ```rust
        /// use vector::{transformation::Transformation, matrix::Matrix};
        ///
        /// let default_transformation = Transformation::default();
        /// let matrix = Matrix::from([
        ///     [1., 0., 0., 0.],
        ///     [0., 1., 0., 0.],
        ///     [0., 0., 1., 0.],
        ///     [0., 0., 0., 1.],
        /// ]);
        ///
        /// assert_eq!(default_transformation, matrix);
        /// ```
        fn default() -> Self {
            Self {
                rotation: Matrix::new_identity(),
                translation: Matrix::new_zero(),
            }
        }
    }

    impl<T: Float> Transformation<T> {
        /// Method to find inverse of the rotation matrix
        /// $$T^{-1} = \begin{bmatrix} R & p \\ 0 & 1 \end{bmatrix}^{-1} = \begin{bmatrix} R^T & -R^Tp \\ 0 & 1\end{bmatrix}$$
        /// Also,$$R^TR = I$$
        pub fn inverse(&self) -> Self {
            let rotation = self.rotation.transpose();
            let translation = -(rotation.clone()) * &self.translation;

            Self {
                rotation,
                translation,
            }
        }

        /// ## New transformation matrix from screw representation
        /// The screw axis has a pitch which signifies the translation performed after rotating a certain angle along the screw axis.
        /// The pitch is denoted as:$$h = \hat{\omega}^T s / \theta$$
        /// ### Case 1: The pitch of the screw axis is finite
        /// $\Vert \omega \Vert = 1$ and $\theta$ corresponds to the angle of rotation along the screw axis.
        /// ### Case 2: The pitch of the screw axis is infinite
        /// In this case $\omega = 0$ and $\Vert s \Vert = 1$ and $\theta$ corresponds to the vertical distance travelled along the screw axis.
        ///
        /// ### Rodrigues' formula
        /// $$\mbox{Rot}(\hat\omega, \theta) = e^{\left[\hat\omega\right] \theta} = I + \sim\theta\left[\hat\omega\right] + (1 - \cos \theta) \left[\hat\omega\right]^2 $$
        pub fn from_screw(axis: Matrix<3, 1, T>, translation: Matrix<3, 1, T>, theta: T) -> Result<Self> {
            // no movement
            let mut rotation = Matrix::new_identity();
            if theta.abs() < T::tolerence() {
                Ok(Self {
                    rotation,
                    translation,
                })
            }
            // infinite pitch
            else if axis.is_zero() {
                match translation.has_normal_columns() {
                    true => Ok(Self {
                        rotation,
                        translation: translation * theta,
                    }),
                    false => Err(Error::Screw(FromScrewError::RotationAxisNotNormal)),
                }
            }
            // finite pitch
            else {
                if !axis.has_normal_columns() {
                    return Err(Error::Screw(FromScrewError::RotationAxisNotNormal));
                }

                // the rotation matrix is obtained from rodrigurs formula
                // 1. calculate skew symmetric matrix
                let skew_symmetric = skew_symmetric(&axis);

                let skew_symmetric_squared = &skew_symmetric * &skew_symmetric;

                let complex_one_minus_cos = Complex::from(T::one() - theta.cos());
                let sin = theta.sin();

                // 2. Rodrigues' formula
                rotation += Complex::from(sin) * skew_symmetric.clone() + complex_one_minus_cos * skew_symmetric_squared.clone();

                // Chasles-Mozzi theorem
                let translation = (Matrix::new_identity() + complex_one_minus_cos * skew_symmetric + Complex::from(theta - sin) * skew_symmetric_squared) * translation;

                Ok(Self {
                    rotation,
                    translation,
                })
            }
        }

        /// Method to validate the rotation matrix
        /// Normally only 3 entries (out of 9) can be selcted independently in a rotation matrix. These correspond to the angles to be rotated by each axis.
        ///
        /// The rotation matrix has 9 elements. Hence 6 constraints need to be applied.
	    /// 1) The unit norm condition: $\hat{x}_b$, $\hat{y}_b$, $\hat{z}_b$ are all unit vectors. $$\begin{aligned} r_{11}^2 + r_{21}^2 + r_{31}^2 &= 1 \\ r_{12}^2 + r_{22}^2 + r_{32}^2 &= 1 \\ r_{13}^2 + r_{23}^2 + r_{33}^2 &= 1 \end{aligned}$$
	    /// 2) The orthogonality condition (dot product of columns is zero): $$\begin{aligned} \hat{x}_b . \hat{y}_b &= r_{11}r_{12} + r_{21}r_{22} + r_{31}r_{32} &= 0 \\ \hat{y}_b . \hat{z}_b &= r_{12}r_{13} + r_{22}r_{23} + r_{32}r_{32} &= 0 \\ \hat{x}_b . \hat{z}_b &= r_{11}r_{13} + r{21}r_{23} + r_{31}r_{33} &= 0 \end{aligned}$$
        ///
        /// Also for the frame to be right handed, an additional condition should be that the determinant should be `+1`.
        /// If the determinant is `-1`, it would lead to reflections or inversions.
        fn validate_rotation(rotation: &mut Matrix<3, 3, T>) -> Result<()> {
            if rotation.determinant().magnitude() <= T::tolerence() {
                return Err(Error::Singular);
            }

            // columns should be orthogonal
            if [(0, 1), (0, 2), (1, 2)].iter().any(|&(c1, c2)| !(0..3).map(|i| rotation[i][c1] * rotation[i][c2]).sum::<Complex<T>>().is_zero()) {
                return Err(Error::Dependent);
            }

            // normalize
            rotation.normalize_columns();

            // check orientation
            let determinant = rotation.determinant();
            println!("rotation:\n{rotation}\ndeterminant: {determinant}");
            if determinant.is_imaginary() || (determinant.real() - T::one()).abs() > T::tolerence() {
                return Err(Error::Orientation);
            }

            Ok(())
        }

        /// Returns thn 6x6 adjoint matrix for the transformation matrix
        pub fn adjoint(&self) -> Matrix<6, 6, T> {
            let one_zero = skew_symmetric(&self.translation) * &self.rotation;
            let zero = T::zero().into();

            Matrix::from([
                [self.rotation[0][0], self.rotation[0][1], self.rotation[0][2], zero, zero, zero],
                [self.rotation[1][0], self.rotation[1][1], self.rotation[1][2], zero, zero, zero],
                [self.rotation[2][0], self.rotation[2][1], self.rotation[2][2], zero, zero, zero],
                [one_zero[0][0], one_zero[0][1], one_zero[0][2], self.rotation[0][0], self.rotation[0][1], self.rotation[0][2]],
                [one_zero[1][0], one_zero[1][1], one_zero[1][2], self.rotation[1][0], self.rotation[1][1], self.rotation[1][2]],
                [one_zero[2][0], one_zero[2][1], one_zero[2][2], self.rotation[2][0], self.rotation[2][1], self.rotation[2][2]],
            ])
        }
    }

    impl<T: Float> From<Screw<T>> for Transformation<T> {
        fn from(Screw { angular, linear, theta }: Screw<T>) -> Self {
            // no movement
            let mut rotation = Matrix::new_identity();
            // infinite pitch
            let translation = if angular.is_zero() {
                linear * theta
            }
            // finite pitch
            else {
                // the rotation matrix is obtained from rodrigurs formula while the complete matrix translation matrix is created using Chasles-Mozzi theorem.
                // 1. calculate skew symmetric matrix
                let skew_symmetric = skew_symmetric(&angular);
                let skew_symmetric_squared = &skew_symmetric * &skew_symmetric;

                let complex_one_minus_cos = Complex::from(T::one() - theta.cos());
                let sin = theta.sin();

                // 2. Rodrigues' formula
                rotation += Complex::from(sin) * &skew_symmetric + complex_one_minus_cos * &skew_symmetric_squared;

                // 3. Translation matrix
                (Matrix::new_identity() + complex_one_minus_cos * skew_symmetric + Complex::from(theta - sin) * skew_symmetric_squared) * linear
            };

            Self {
                rotation,
                translation,
            }
        }
    }

    impl<T: Float> TryFrom<Matrix<3, 3, T>> for Transformation<T> {
        type Error = Error;

        /// ## New transformation matrix from a rotation matrix
        /// ```rust
        /// todo!("add test cases")
        /// ```
        fn try_from(mut rotation: Matrix<3, 3, T>) -> core::result::Result<Self, Self::Error> {
            Self::validate_rotation(&mut rotation)?;

            Ok(Self {
                rotation,
                ..Default::default()
            })
        }
    }

    impl<T: Float> TryFrom<Matrix<4, 4, T>> for Transformation<T> {
        type Error = Error;

        /// ```rust
        /// todo!("add test cases")
        /// ```
        fn try_from(value: Matrix<4, 4, T>) -> core::result::Result<Self, Self::Error> {
            // rotation
            let rotation = Matrix::from([
                [value[0][0], value[0][1], value[0][2]],
                [value[1][0], value[1][1], value[1][2]],
                [value[2][0], value[2][1], value[2][2]],
            ]);

            let translation = Matrix::from([
                [value[0][3]],
                [value[1][3]],
                [value[2][3]],
            ]);

            Self::try_from((rotation, translation))
        }
    }

    impl<T: Float> TryFrom<(Matrix<3, 3, T>, Matrix<3, 1, T>)> for Transformation<T> {
        type Error = Error;

        /// ```rust
        /// todo!("add test cases")
        /// ```
        fn try_from((mut rotation, translation): (Matrix<3, 3, T>, Matrix<3, 1, T>)) -> core::result::Result<Self, Self::Error> {
            Self::validate_rotation(&mut rotation)?;

            Ok(Self {
                rotation,
                translation,
            })
        }
    }

    impl<T: Float> From<Matrix<3, 1, T>> for Transformation<T> {
        /// ## New transformation matrix from a translation matrix
        /// ```rust
        /// use vector::{transformation::Transformation, matrix::Matrix};
        ///
        /// let translation = Matrix::from([
        ///     [1.],
        ///     [2.],
        ///     [3.],
        /// ]);
        ///
        /// let matrix = Matrix::from([
        ///     [1., 0., 0., 1.],
        ///     [0., 1., 0., 2.],
        ///     [0., 0., 1., 3.],
        ///     [0., 0., 0., 1.],
        /// ]);
        ///
        /// assert_eq!(Transformation::from(translation), matrix);
        /// ```
        fn from(translation: Matrix<3, 1, T>) -> Self {
            Self {
                translation,
                ..Default::default()
            }
        }
    }

    impl<T: Float> PartialEq<Matrix<4, 4, T>> for Transformation<T> {
        /// Method compares transformation matrix to a 4x4 matrix
        fn eq(&self, other: &Matrix<4, 4, T>) -> bool {
            (0..3).flat_map(|r| (0..3).map(move |c| (r, c))).all(|(r, c)| self.rotation[r][c] == other[r][c]) // rotation
            &&
            (0..3).all(|c| self.translation[c][0] == other[c][3]) // translation
        }
    }

    impl<T: Float> PartialEq for Transformation<T> {
        fn eq(&self, other: &Self) -> bool {
            self.rotation == other.rotation && self.translation == other.translation
        }
    }

    impl<T: Float> core::ops::Mul for Transformation<T> {
        type Output = Self;

        /// Multiplication of transformation matrices is a transformation matrix
        /// ```rust
        /// use vector::{transformation::Transformation, matrix::Matrix};
        ///
        /// // body b with respect to frame s
        /// let sb = Matrix::from([
        ///     [0., 0., 1., 400.],
        ///     [0., 1., 0., 50.],
        ///     [1., 0., 0., 300.],
        ///     [0., 0., 0., 1.],
        /// ]);
        ///
        /// // body c with respect to frame b
        /// let bc = Matrix::from([
        ///     [0., 0., 1., 300.],
        ///     [0., 1., 0., 100.],
        ///     [1., 0., 0., 120.],
        ///     [0., 0., 0., 1.],
        /// ]);
        ///
        /// let sc = Matrix::from([
        ///     [1., 0., 0., 520.],
        ///     [0., 1., 0., 150.],
        ///     [0., 0., 1., 600.],
        ///     [0., 0., 0., 1.],
        /// ]);
        ///
        /// assert_eq!(Transformation::try_from(sb).unwrap() * Transformation::try_from(bc).unwrap(), Transformation::try_from(sc).unwrap());
        /// ```
        fn mul(self, rhs: Self) -> Self::Output {
            let rotation = &self.rotation * rhs.rotation;
            let translation = self.rotation * rhs.translation + self.translation;

            Self {
                rotation,
                translation,
            }
        }
    }

    impl<T: Float> Display for Transformation<T> {
        /// ### Example
        ///
        /// Consider the matrix:$$\begin{bmatrix} 10 & 0 & 20 \\\ 0 & 30 & 0 \\\ 200 & 0 & 100 \end{bmatrix}$$
        /// ```rust
        /// use vector::{transformation::Transformation, matrix::Matrix};
        ///
        /// let matrix = Matrix::from([[10., 0., 20.], [0., 30., 0.], [200., 0., 100.]]);
        /// assert_eq!("│\t10\t0\t20\t│\n│\t0\t30\t0\t│\n│\t200\t0\t100\t│\n", matrix.to_string());
        /// ```
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            for i in 0..3 {
                write!(f, "│\t")?;
                for j in 0..3 {
                    write!(f, "{}\t", self.rotation[i][j])?;
                }
                writeln!(f, "{}\t│", self.translation[i][0])?;
            }

            writeln!(f, "│\t{0}\t{0}\t{0}\t{1}\t│", 0., 1.)
        }
    }

    /// Denote velocity
    #[derive(Debug)]
    pub struct Twist<T: Float> {
        angular: Matrix<3, 1, T>,
        linear: Matrix<3, 1, T>,
    }

    impl<T: Float> From<Matrix<6, 1, T>> for Twist<T> {
        /// Converts a column vector with 6 elements to a [Twist]
        /// ```rust
        /// use vector::{matrix::Matrix, transformation::Twist};
        ///
        /// let mat = Matrix::from([
        ///     [12.],
        ///     [1.],
        ///     [1.],
        ///     [0.898],
        ///     [-0.2222],
        ///     [0.898],
        /// ]);
        ///
        /// let _twist: Twist<f32> = mat.into();
        /// ```
        fn from(value: Matrix<6, 1, T>) -> Self {
            Self {
                angular: Matrix::from([[value[0][0]], [value[1][0]], [value[2][0]]]),
                linear: Matrix::from([[value[3][0]], [value[4][0]], [value[5][0]]]),
            }
        }
    }

    impl<T: Float> From<(Matrix<3, 1, T>, Matrix<3, 1, T>)> for Twist<T> {
        /// Converts a tuple of 2 elements with type [Matrix<3, 3>] to Twist
        /// ```rust
        /// use vector::{matrix::Matrix, transformation::Twist};
        ///
        /// let angular = Matrix::from([
        ///     [12.],
        ///     [1.],
        ///     [1.],
        /// ]);
        /// let linear = Matrix::from([
        ///     [0.898],
        ///     [-0.2222],
        ///     [0.898],
        /// ]);
        ///
        /// let _twist: Twist<f32> = (angular, linear).into();
        /// ```
        fn from((angular, linear): (Matrix<3, 1, T>, Matrix<3, 1, T>)) -> Self {
            Self {
                angular,
                linear,
            }
        }
    }

    impl<T: Float> Twist<T> {
        pub fn new(angular: Matrix<3, 1, T>, linear: Matrix<3, 1, T>) -> Self {
            Self {
                angular,
                linear,
            }
        }

        /// creates the adjoint 4x4 matrix for [Twist].
        /// ```rust
        /// use vector::{matrix::Matrix, transformation::Twist};
        ///
        /// let matrix = Matrix::from([
        ///     [0.],
        ///     [0.],
        ///     [-2.],
        ///     [2.8],
        ///     [4.],
        ///     [0.],
        /// ]);
        ///
        /// let res = Matrix::from([
        ///     [0., ]
        /// ]);
        /// ```
        pub fn adjoint(&self) -> Matrix<4, 4, T> {
            let adjoint = skew_symmetric(&self.angular);
            let zero = T::zero().into();

            Matrix::from([
                [adjoint[0][0], adjoint[0][1], adjoint[0][2], self.linear[0][0]],
                [adjoint[1][0], adjoint[1][1], adjoint[1][2], self.linear[1][0]],
                [adjoint[2][0], adjoint[2][1], adjoint[2][2], self.linear[2][0]],
                [zero; 4],
            ]) 
        }

        /// creates [Twist] from [Screw] if screw represents transformation for a unit time
        pub fn from_screw_for_unit_time(Screw { mut angular, mut linear, theta }: Screw<T>) -> Self {
            angular *= theta;
            linear *= theta;
            
            Self {
                angular,
                linear,
            }
        }
    }

    impl<T: Float> Display for Twist<T> {
        /// Display for Twist
        /// ```rust
        /// use vector::{matrix::Matrix, transformation::Twist};
        ///
        /// let matrix = Matrix::from([
        ///     [0.],
        ///     [0.],
        ///     [-2.],
        ///     [2.8],
        ///     [4.],
        ///     [0.],
        /// ]);
        ///
        /// assert_eq!(Twist::from(matrix).to_string(), "│\t0\t│\n│\t0\t│\n│\t-2\t│\n│\t2.8\t│\n│\t4\t│\n│\t0\t│\n");
        /// ```
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            for i in 0..3 {
                writeln!(f, "│\t{}\t│", self.angular[i][0])?;
            }
            for i in 0..3 {
                writeln!(f, "│\t{}\t│", self.linear[i][0])?;
            }

            Ok(())
        }
    }

    /// Denote screw transformation
    /// Here $\theta$ is analogous to time.
    /// If [Twist] denote velocity, and if time is known, one can identify the [Transformation] from initial to final position.
    /// [Screw] can be thought of another way of representing [Transformation].
    #[derive(Debug)]
    pub struct Screw<T: Float> {
        // one of angular and linear is normal
        angular: Matrix<3, 1, T>,
        linear: Matrix<3, 1, T>,
        theta: T,
    }

    impl<T: Float> Screw<T> {
        /// Creates a new screw from angular and linear matrices and the given $\theta$.
        pub fn new(angular: Matrix<3, 1, T>, linear: Matrix<3, 1, T>, theta: T) -> Result<Self> {
            if !linear.has_normal_columns() {
                Err(Error::Screw(FromScrewError::TranslationAxisNotNormal))
            }
            else if !angular.is_zero() && !angular.has_normal_columns() {
                Err(Error::Screw(FromScrewError::RotationAxisNotNormal))
            }
            else {
                Ok(Self {
                    angular,
                    linear,
                    theta,
                })
            }
        }

        /// Creates [Screw] from [Twist] if twist is applied for unit time
        pub fn from_twist_for_unit_time(Twist { mut angular, mut linear }: Twist<T>) -> Self {
            if !linear.is_zero() {
                let theta = match angular.is_zero() {
                    true => (0..3).map(|r| linear[r][0].norm_squared()).sum::<T>().sqrt(),
                    false => {
                        let theta = (0..3).map(|r| angular[r][0].norm_squared()).sum::<T>().sqrt();
                        angular /= theta;
                        theta
                    },
                };

                linear /= theta;
                
                Self {
                    angular,
                    linear,
                    theta,
                }
            }
            else {
                Self {
                    angular,
                    linear,
                    theta: T::zero(),
                }
            }
        }
    }

    pub fn forward_kinematice_in_body_frame<T: Float>(m: Transformation<T>, b_list: &[Screw<T>]) -> Transformation<T> {
        todo!()
    }

    pub fn forward_kinematice_in_space_frame<T: Float>(m: Transformation<T>, s_list: &[Screw<T>]) -> Transformation<T> {
        todo!()
    }

    #[derive(Debug)]
    pub enum Error {
        Dependent,
        Singular,
        Orientation,
        Screw(FromScrewError),
    }

    impl Display for Error {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            match self {
                Self::Dependent => write!(f, "columns of rotation matrix not orthogonal"),
                Self::Singular => write!(f, "singular matrix"),
                Self::Orientation => write!(f, "rotation matrix not right handed. this represents reflection matrix."),
                Self::Screw(e) => write!(f, "error while making transformation matrix. {e}"),
            }
        }
    }

    #[derive(Debug)]
    pub enum FromScrewError {
        RotationAxisNotNormal,
        TranslationAxisNotNormal,
    }

    impl Display for FromScrewError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::RotationAxisNotNormal => write!(f, "screw axis should be normalized"),
                Self::TranslationAxisNotNormal => write!(f, "translation vector should be normal if there is no rotation (theta defines the magnitude of translation).")
            }
        }
    }
}
