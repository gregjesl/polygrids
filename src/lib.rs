pub mod onedimensional;
pub mod twodimensional;

use nalgebra::RealField;
use onedimensional::LookupTable1D;
use std::{ops::{Add, Mul}, rc::Rc};

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

#[derive(Clone)]
pub enum LookupTable<G, T> 
where G: RealField + Copy + From<f32>
{
    Root(LookupTable1D<G, T>),
    Parent(Axis<G>, Vec<LookupTable<G, T>>),
}

impl<G, T> LookupTable<G, T> 
where G: RealField + Copy + From<f32>,
      T: Default + Clone
{
    pub fn new(axes: &[Axis<G>]) -> Self {
        match axes.len() {
            0 => panic!("At least one axis is required to create a LookupTable."),
            1 => LookupTable::Root(LookupTable1D::empty(Rc::new(axes[0].clone()))),
            _ => {
                let first_axis = axes[0].clone();
                let remaining_axes = &axes[1..];
                let mut children: Vec<LookupTable<G, T>> = Vec::with_capacity(first_axis.points().len());
                for _ in 0..first_axis.points().len() {
                    children.push(LookupTable::new(remaining_axes));
                }
                LookupTable::Parent(first_axis, children)
            }
        }
    }

    pub fn dimensions(&self) -> usize {
        match self {
            LookupTable::Root(_) => 1,
            LookupTable::Parent(_, children) => 1 + children[0].dimensions(),
        }
    }

    pub fn elements(&self) -> usize {
        match self {
            LookupTable::Root(table) => table.len(),
            LookupTable::Parent(_, children) => {
                children.first().unwrap().elements() * children.len()
            }
        }
    }

    pub fn slice(&self, index: usize) -> &LookupTable<G, T> {
        match self {
            LookupTable::Root(_) => panic!("Cannot slice a root LookupTable."),
            LookupTable::Parent(_, children) => &children[index],
        }
    }

    pub fn get(&self, indices: &[usize]) -> &T {
        match self {
            LookupTable::Root(table) => {
                assert_eq!(indices.len(), 1);
                &table[indices[0]]
            }
            LookupTable::Parent(_, children) => {
                assert!(indices.len() >= 1);
                let child = &children[indices[0]];
                child.get(&indices[1..])
            }
        }
    }

    pub fn set(&mut self, indices: &[usize], value: T) {
        match self {
            LookupTable::Root(table) => {
                assert_eq!(indices.len(), 1);
                table[indices[0]] = value;
            }
            LookupTable::Parent(_, children) => {
                assert!(indices.len() >= 1);
                let child = &mut children[indices[0]];
                child.set(&indices[1..], value);
            }
        }
    }

    pub fn serialize(&self) -> Vec<String> 
    where T: ToString
    {
        match self {
            LookupTable::Root(table) => {
                table.serialize()
            }
            LookupTable::Parent(axis, children) => {
                let mut result = Vec::with_capacity(self.elements());
                for (i, child) in children.iter().enumerate() {
                    for line in child.serialize().iter() {
                        result.push(format!("{},{}", axis.points()[i], line));
                    }
                }
                result
            }
        }
    }
}

impl<G, T> LookupTable<G, T>
where G: RealField + Copy + From<f32>,
      T: Mul<G, Output=T> + Add<T, Output = T> + Clone
{
    pub fn linear_interpolate(&self, xs: &[G]) -> Option<T> {
        match self {
            LookupTable::Root(table) => {
                assert_eq!(xs.len(), 1);
                table.linear_interpolate(xs[0])
            }
            LookupTable::Parent(axis, children) => {
                assert!(xs.len() >= 1);
                let indices_weights = axis.lookup(xs[0])?;
                let (i0, w0) = indices_weights[0];
                let (i1, w1) = indices_weights[1];

                let v0 = children[i0].linear_interpolate(&xs[1..])?;
                let v1 = children[i1].linear_interpolate(&xs[1..])?;

                Some(v0 * w0 + v1 * w1)
            }
        }
    }
}

pub type LookupTable32 = LookupTable<f32, f32>;
pub type LookupTable64 = LookupTable<f64, f64>;

/// Creates a cartesian heightmap using f32
pub fn cartesian_map_32(x: Axis32, y: Axis32) -> LookupTable32 {
    return LookupTable32::new(&[x, y]);
}

/// Creates a cartesian heightmap using f64
pub fn cartesian_map_64(x: Axis64, y: Axis64) -> LookupTable64 {
    return LookupTable64::new(&[x, y]);
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn two_norm(x: f64, y: f64) -> f64 {
        (x * x + y * y).sqrt()
    }

    #[test]
    fn test_2d_lookup_table() {
        let axis = Axis::linear(0.0, 2.0, 3).unwrap();
        let mut table = LookupTable::new(&[axis.clone(), axis]);
        assert_eq!(table.dimensions(), 2);
        assert_eq!(table.elements(), 9);
        for i in 0..3 {
            for j in 0..3 {
                let value = two_norm(i as f64, j as f64) * 10.0;
                table.set(&[i, j], value);
            }
        }

        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(table.get(&[i, j]), &(two_norm(i as f64, j as f64) * 10.0));
            }
        }
    }

    #[test]
    fn test_2d_lookup_interpolate() {
        let axis = Axis::linear(0.0, 1.0, 2).unwrap();
        let mut table = LookupTable::new(&[axis.clone(), axis]);
        assert_eq!(table.dimensions(), 2);
        table.set(&[0, 0], 0.0);
        table.set(&[0, 1], 10.0);
        table.set(&[1, 0], 10.0);
        table.set(&[1, 1], 20.0);

        assert_eq!(table.linear_interpolate(&[-0.1, 0.5]), None);
        assert_eq!(table.linear_interpolate(&[0.0, 0.0]), Some(0.0));
        assert_eq!(table.linear_interpolate(&[0.0, 0.5]), Some(5.0));
        assert_eq!(table.linear_interpolate(&[0.5, 0.0]), Some(5.0));
        assert_eq!(table.linear_interpolate(&[1.0, 0.5]), Some(15.0));
        assert_eq!(table.linear_interpolate(&[0.5, 1.0]), Some(15.0));
        assert_eq!(table.linear_interpolate(&[1.0, 1.0]), Some(20.0));
        assert_eq!(table.linear_interpolate(&[1.1, 0.5]), None);

        {
            let q11 = 0.0;
            let q12 = 10.0;
            let q21 = 10.0;
            let q22 = 20.0;
            let x1 = 0.0;
            let x2 = 1.0;
            let y1 = 0.0;
            let y2 = 1.0;

            let x = 0.3;
            let y = 0.6;

            let a = q11 * (y2 - y) + q12 * (y - y1);
            let b = q21 * (y2 - y) + q22 * (y - y1);
            let expected = a * (x2 - x) + b * (x - x1);
            let result = table.linear_interpolate(&[x, y]).unwrap();
            let diff = ((expected - result) as f64).abs();
            assert!(diff < 1e-10);
        }
    }

    #[test]
    fn test_serialize_lookup_table() {
        let axis1 = Axis::linear(0.0, 1.0, 2).unwrap();
        let axis2 = Axis::linear(0.0, 2.0, 3).unwrap();
        let mut table= LookupTable::new(&[axis1, axis2]);
        table.set(&[0, 0], 0.0);
        table.set(&[0, 1], 10.0);
        table.set(&[0, 2], 20.0);
        table.set(&[1, 0], 5.0);
        table.set(&[1, 1], 15.0);
        table.set(&[1, 2], 25.0);
        let serialized = table.serialize();
        let expected = vec![
            "0,0,0",
            "0,1,10",
            "0,2,20",
            "1,0,5",
            "1,1,15",
            "1,2,25",
        ];
        assert_eq!(serialized, expected);
    }
}