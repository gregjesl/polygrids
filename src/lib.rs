use std::{ops::{Index, IndexMut, Mul, Add}, rc::Rc};
use nalgebra::RealField;

#[cfg(feature = "rand")]
use rand::{Rng, RngExt};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AxisWrapping<T: RealField + Copy> {
    None,
    Polar(T)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AxisError {
    InsufficentPoints,
    DupilicatePoints,
    NotMonotonicIncreasing,
    NotFinitePoint,
}

#[derive(Clone, Debug)]
pub struct Axis<T>
where T: RealField + Copy + From<f32>
{
    points: Vec<T>,
    wrapping: AxisWrapping<T>,
}

impl<T> Axis<T> 
where T: RealField + Copy + From<f32>
{
    pub fn new(points: Vec<T>, wrapping: AxisWrapping<T>) -> Result<Self, AxisError> {
        if points.len() < 2 {
            return Err(AxisError::InsufficentPoints);
        }

        for i in 1..points.len() {
            let point = &points[i];

            if !point.is_finite() {
                return Err(AxisError::NotFinitePoint);
            }

            if points.iter().filter(|&x| x == point).count() > 1 {
                return Err(AxisError::DupilicatePoints);
            }

            if point < &points[i - 1] {
                return Err(AxisError::NotMonotonicIncreasing);
            }
        }

        match wrapping {
            AxisWrapping::Polar(last) => {
                if !last.is_finite() {
                    return Err(AxisError::NotFinitePoint);
                }
                
                if last < points[points.len() - 1] {
                    return Err(AxisError::NotMonotonicIncreasing);
                }

                if last == *points.last().unwrap() {
                    return Err(AxisError::DupilicatePoints);
                }
            }
            AxisWrapping::None => {}
        }

        Ok(Axis { points, wrapping })
    }

    pub fn points(&self) -> &Vec<T> {
        &self.points
    }

    /// Creates a linear axis from start to end with the specified number of points.
    /// 
    /// # Examples
    /// ```
    /// use polygrids::Axis64;
    /// let axis = Axis64::linear(0.0, 10.0, 5).unwrap();
    /// assert_eq!(axis.points(), &vec![0.0, 2.5, 5.0, 7.5, 10.0]);
    /// ```
    pub fn linear(start: T, end: T, points: usize) -> Result<Self, AxisError> {
        if points < 2 {
            return Err(AxisError::InsufficentPoints);
        } else if start >= end {
            return Err(AxisError::NotMonotonicIncreasing);
        }
        let step = (end - start) / T::from((points - 1) as f32);
        let mut pts = Vec::with_capacity(points);
        for i in 0..points {
            pts.push(start + (T::from(i as f32)) * step);
        }
        Ok(Axis { points: pts, wrapping: AxisWrapping::None })
    }

    /// Creates a polar axis with the specified number of points from 0 to 2π.
    /// 
    /// # Examples
    /// ```
    /// use polygrids::Axis64;
    /// let axis = Axis64::polar_radians(4).unwrap();
    /// assert_eq!(axis.points(), &vec![
    ///     0.0, 
    ///     std::f64::consts::PI / 2.0, 
    ///     std::f64::consts::PI, 
    ///     3.0 * std::f64::consts::PI / 2.0
    /// ]);
    /// ```
    pub fn polar_radians(points: usize) -> Result<Self, AxisError> {
        if points < 2 {
            return Err(AxisError::InsufficentPoints);
        }
        let step = T::two_pi() / (T::from(points as f32));
        let mut pts = Vec::with_capacity(points);
        for i in 0..points {
            pts.push(T::from(i as f32) * step);
        }
        Ok(Axis { points: pts, wrapping: AxisWrapping::Polar(T::two_pi()) })
    }

    /// Creates a polar axis with the specified number of points from 0 to 2π.
    /// 
    /// # Examples
    /// ```
    /// use polygrids::Axis64;
    /// let axis = Axis64::polar_degrees(4).unwrap()  ;
    /// assert_eq!(axis.points(), &vec![
    ///     0.0, 
    ///     90.0, 
    ///     180.0, 
    ///     270.0
    /// ]);
    /// ```
    pub fn polar_degrees(points: usize) -> Result<Self, AxisError> {
        if points < 2 {
            return Err(AxisError::InsufficentPoints);
        }
        let step = T::from(360 as f32) / T::from(points as f32);
        let mut pts = Vec::with_capacity(points);
        for i in 0..points {
            pts.push(T::from(i as f32) * step);
        }
        Ok(Axis { points: pts, wrapping: AxisWrapping::Polar(T::from(360 as f32)) })
    }

    /// Given the point x, find the indices and weights of the two surrounding points.
    /// 
    /// # Returns 
    /// - `None` if x is out of bounds.
    /// - `Some([(i, 1.0), (i, 0.0)])` if x matches a point exactly.
    /// - `Some([(i0, w0), (i1, w1)])` where `axis[i0] < x < axis[i1]` otherwise.
    pub fn lookup(&self, x: T) -> Option<[(usize, T); 2]> {
        debug_assert!(self.points.iter().all(|v| v.is_finite()));

        // Check bounds
        if x < self.points[0] { return None }
        
        if x > *self.points.last().unwrap() {
            match self.wrapping {
                AxisWrapping::Polar(last) => {
                    if x == last { 
                        return Some([(0, T::one()), (0, T::zero())]);
                    } else if x < last {
                        let frac = (x - *self.points.last().unwrap()) / (last - *self.points.last().unwrap());
                        let lower_weight = T::one() - frac;
                        let upper_weight = frac;
                        let lower = (self.points.len() - 1, lower_weight);
                        let upper = (0, upper_weight);
                        return Some([lower, upper]);
                    } else {
                        return None;
                    }
                }
                AxisWrapping::None => { return None; }
            }
        }

        let mut lo = 0usize;
        let mut hi = self.points.len();
        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.points[mid] == x {
                return Some([(mid, T::one()), (mid, T::zero())]);
            } else if self.points[mid] < x {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        let i = lo;

        let frac = (x - self.points[i - 1]) / (self.points[i] - self.points[i - 1]);
        let lower_weight = T::one() - frac;
        let upper_weight = frac;
        let lower = (i - 1, lower_weight);
        let upper = (i, upper_weight);
        Some([lower, upper])
    }
}

#[cfg(feature = "rand")]
impl Axis<f64> {
    pub fn random_sample(&self, rng: &mut impl Rng) -> f64 {
        rng.random_range(self.points[0]..=*self.points.last().unwrap())
    }
}

impl<T> Default for Axis<T> 
where T: RealField + Copy + From<f32>
{
    fn default() -> Self {
        Axis::linear(T::zero(), T::one(), 2).unwrap()
    }
}

pub type Axis32 = Axis<f32>;
pub type Axis64 = Axis<f64>;


#[derive(Clone, Debug)]
pub struct LookupTable1D<G, T> 
where G: RealField + Copy + From<f32>
{
    axis: Rc<Axis<G>>,
    values: Vec<T>,
}

impl<G, T> LookupTable1D<G, T> 
where G: RealField + Copy + From<f32>
{
    pub fn new(axis: Rc<Axis<G>>, values: Vec<T>) -> Self {
        assert_eq!(axis.points.len(), values.len());
        LookupTable1D { axis, values }
    }

    pub fn fill<F>(axis: Rc<Axis<G>>, source: F) -> Self
    where F: Fn(G) -> T
    {
        let len = axis.points.len();
        let values = (0..len).map(|i| source(axis.points[i])).collect();
        Self::new(axis, values)
    }

    pub fn empty(axis: Rc<Axis<G>>) -> Self
    where T: Default + Clone
     {
        let len = axis.points.len();
        Self::new(axis, vec![T::default(); len])
    }

    pub fn axis(&self) -> &Axis<G> {
        &self.axis
    }

    pub fn serialize(&self) -> Vec<String> 
    where T: ToString
    {
        self.values.iter()
            .enumerate()
            .map(|(i, v)| format!("{},{}", self.axis.points[i], v.to_string()))
            .collect()
    }
}

impl<G, T> Index<usize> for LookupTable1D<G, T>
where G: RealField + Copy + From<f32>
{
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        &self.values[index]
    }
}

impl<G, T> IndexMut<usize> for LookupTable1D<G, T> 
where G: RealField + Copy + From<f32>
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.values[index]
    }
}

impl<G, T> Default for LookupTable1D<G, T>
where G: RealField + Copy + From<f32>,
      T: Default + Clone
{
    fn default() -> Self {
        LookupTable1D::empty(Rc::new(Axis::default()))
    }
}

impl<G, T> LookupTable1D<G, T>
where G: RealField + Copy + From<f32>,
      T: Mul<G, Output=T> + Add<T, Output = T> + Clone
{
    pub fn linear_interpolate(&self, x: G) -> Option<T> {
        let indices_weights = self.axis.lookup(x)?;

        let (i0, w0) = indices_weights[0];
        let (i1, w1) = indices_weights[1];

        debug_assert_eq!(w0 + w1, G::one());

        Some(self.values[i0].clone() * w0 + self.values[i1].clone() * w1)
    }
}