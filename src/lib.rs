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

        /// Getter
        /// Gets the value present at the index (r, c).
        pub fn get(&self, r: usize, c: usize) -> &Complex<T> {
            self.0.get(r).expect("row index out of bounds").get(c).expect("column index out of bounds")
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

        /// Method to check if the square matrix is identity or not.
        /// todo!()
        pub fn is_identity(&self) -> bool {
            self.0.iter().enumerate().all(|(r, row)| {
                row.iter().enumerate().all(|(c, &val)| {
                    let expected = if r == c { T::one().into() } else { T::zero().into() };
                    (val - expected).magnitude().abs() <= T::tolerence()
                })
            })
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
            let matrix = std::array::from_fn(|i| 
                std::array::from_fn(|j| 
                    (0..C).map(|k| self.0[i][k] * rhs.0[k][j]).sum::<Complex<T>>()
                )
            );

            *self = Self(matrix);
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

    impl<T: Float, Z: Into<Complex<T>>, const R: usize, const C: usize> core::ops::MulAssign<Z> for Matrix<R, C, T> {
        fn mul_assign(&mut self, rhs: Z) {
            let val = rhs.into();
            self.0.iter_mut().for_each(|r| r.iter_mut().for_each(|c| *c *= val));
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
    use core::{fmt::Display, ops::Mul};
    use crate::{matrix::{Matrix, MatrixError}, Complex, Float};

    #[derive(Debug)]
    pub struct Rotation<T: Float>(Matrix::<3, 3, T>);

    impl<T: Float> Rotation<T> {
        /// Makes the rotation matrix using the **Rodrigues' formula**.
        /// ### Rodrigues' formula
        /// $$\mbox{Rot}(\hat\omega, \theta) = e^{\left[\hat\omega\right] \theta} = I + \sim\theta\left[\hat\omega\right] + (1 - \cos \theta) \left[\hat\omega\right]^2 $$
        /// #### Example
        /// ```rust
        /// use vector::{matrix::Matrix, transformation::Rotation};
        ///
        /// let omega = Matrix::from([[0.], [0.866], [0.5]]);
        /// let theta_degrees = 30.;
        /// let rot = Rotation::from_axis_and_angle(omega, theta_degrees * std::f32::consts::PI / 180.);
        ///
        /// let answer = Matrix::from([[0.8660313, -0.25, 0.433], [0.25, 0.96650636, 0.058011007], [-0.433, 0.058011007, 0.8995249]]);
        ///
        /// assert_eq!(rot, answer);
        /// ```
        pub fn from_axis_and_angle(omega: Matrix<3, 1, T>, theta: T) -> Self {
            // 1. calculate skew symmetric matrix
            let skew_symmetric = Matrix::from([
                [T::zero().into(), -omega.get(2, 0).to_owned(), omega.get(1, 0).to_owned()],
                [omega.get(2, 0).to_owned(), T::zero().into(), -omega.get(0, 0).to_owned()],
                [-omega.get(1, 0).to_owned(), omega.get(0, 0).to_owned(), T::zero().into()],
            ]);

            let skew_symmetric_squared = &skew_symmetric * &skew_symmetric;

            // 2. Rodrigues' formula
            Self(Matrix::new_identity() + Complex::from(theta.sin()) * skew_symmetric + (Complex::from(T::one()) - Complex::from(theta.cos())) * skew_symmetric_squared)
        }

        /// Inverse of rotation matrices is same as transpose.
        /// $$R^{-1} = R^T$$
        pub fn inverse(&self) -> Self {
            Self(self.0.transpose())
        }
    }

    impl<T: Float> PartialEq<Matrix<3, 3, T>> for Rotation<T> {
        fn eq(&self, other: &Matrix<3, 3, T>) -> bool {
            &self.0 == other
        }
    }

    impl<T: Float> TryFrom<Matrix<3, 3, T>> for Rotation<T> {
        type Error = MatrixError;

        /// Creates Rotation matrix from a 3x3 matrix.
        /// Normally only 3 entries (out of 9) can be selcted independently in a rotation matrix. These correspond to the angles to be rotated by each axis.
        ///
        /// The rotation matrix has 9 elements. Hence 6 constraints need to be applied.
	    /// 1) The unit norm condition: $\hat{x}_b$, $\hat{y}_b$, $\hat{z}_b$ are all unit vectors. $$\begin{aligned} r_{11}^2 + r_{21}^2 + r_{31}^2 &= 1 \\ r_{12}^2 + r_{22}^2 + r_{32}^2 &= 1 \\ r_{13}^2 + r_{23}^2 + r_{33}^2 &= 1 \end{aligned}$$
	    /// 2) The orthogonality condition (columns are independent): $$\begin{aligned} \hat{x}_b . \hat{y}_b &= r_{11}r_{12} + r_{21}r_{22} + r_{31}r_{32} &= 0 \\ \hat{y}_b . \hat{z}_b &= r_{12}r_{13} + r_{22}r_{23} + r_{32}r_{32} &= 0 \\ \hat{x}_b . \hat{z}_b &= r_{11}r_{13} + r{21}r_{23} + r_{31}r_{33} &= 0 \end{aligned}$$
        ///
        /// Also for the frame to be right handed, an additional condition should be that the determinant should be `+1`.
        /// If the determinant is `-1`, it would lead to reflections or inversions.
        fn try_from(mut value: Matrix<3, 3, T>) -> Result<Self, Self::Error> {
            if value.determinant().magnitude() <= T::tolerence() {
                return Err(MatrixError::Singular);
            }

            // columns should be orthogonal
            for (c1, c2) in [(0, 1), (0, 2), (1, 2)] {
                if (0..3).map(|i| value.get(i, c1) * value.get(i, c2)).sum::<Complex<T>>().is_zero() {
                    return Err(MatrixError::Dependent);
                }
            }

            // normalize
            value.normalize_columns();

            // check orientation
            Ok(Self(value))
        }
    }

    impl<T: Float> core::ops::Neg for Rotation<T> {
        type Output = Self;

        fn neg(self) -> Self::Output {
            Self(-self.0)    
        }
    }

    #[derive(Debug)]
    pub struct Transformation<T: Float> {
        rotation: Rotation<T>,
        translation: Matrix<3, 1, T>,
    }

    impl<T: Float> Transformation<T> {
        /// Method to find inverse of the rotation matrix
        /// $$T^{-1} = \begin{bmatrix} R & p \\ 0 & 1 \end{bmatrix}^{-1} = \begin{bmatrix} R^T & -R^Tp \\ 0 & 1\end{bmatrix}$$
        fn inverse(&self) -> Self {
            let rotation = self.rotation.inverse();
            let translation = -(rotation.0.clone()) * &self.translation;

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
                    write!(f, "{}\t", self.rotation.0.get(i, j))?;
                }
                writeln!(f, "{}\t│", self.translation.get(i, 0))?;
            }

            writeln!(f, "│\t{0}\t{0}\t{0}\t{1}\t│", 0., 1.)
        }
    }
}
