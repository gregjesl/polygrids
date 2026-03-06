use nalgebra::RealField;
use std::{rc::Rc, ops::{Index, IndexMut, Add, Mul}};
use crate::Axis;

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

    pub fn len(&self) -> usize {
        debug_assert_eq!(self.axis.points.len(), self.values.len());
        self.values.len()
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

pub type LookupTable1D32 = LookupTable1D<f32, f32>;
pub type LookupTable1D64 = LookupTable1D<f64, f64>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::rc::Rc;
    use crate::{AxisWrapping, Axis64};
    use nalgebra::Vector2;

    #[test]
    fn test_linear_axis_lookup() {
        let axis = Axis64::linear(0.0, 3.0, 4).unwrap();
        assert_eq!(axis.lookup(-1.0), None);
        assert_eq!(axis.lookup(0.0), Some([(0, 1.0), (0, 0.0)]));
        assert_eq!(axis.lookup(0.25), Some([(0, 0.75), (1, 0.25)]));
        assert_eq!(axis.lookup(1.0), Some([(1, 1.0), (1, 0.0)]));
        assert_eq!(axis.lookup(2.5), Some([(2, 0.5), (3, 0.5)]));
        assert_eq!(axis.lookup(3.0), Some([(3, 1.0), (3, 0.0)]));
        assert_eq!(axis.lookup(4.0), None);
    }

    #[cfg(feature = "rand")]
    #[test]
    fn test_wrapped_axis_lookup() {
        let axis = Axis64::new(vec![0.0, 1.0, 2.0, 3.0], AxisWrapping::Polar(4.0)).unwrap();
        assert_eq!(axis.lookup(-1.0), None);
        assert_eq!(axis.lookup(0.0), Some([(0, 1.0), (0, 0.0)]));
        assert_eq!(axis.lookup(0.5), Some([(0, 0.5), (1, 0.5)]));
        assert_eq!(axis.lookup(1.0), Some([(1, 1.0), (1, 0.0)]));
        assert_eq!(axis.lookup(2.5), Some([(2, 0.5), (3, 0.5)]));
        assert_eq!(axis.lookup(3.0), Some([(3, 1.0), (3, 0.0)]));
        assert_eq!(axis.lookup(3.25), Some([(3, 0.75), (0, 0.25)]));
        assert_eq!(axis.lookup(4.0), Some([(0, 1.0), (0, 0.0)]));
        assert_eq!(axis.lookup(4.25), None);
    }

    #[cfg(feature = "rand")]
    #[test]
    fn test_axis_random_sample() {
        let axis = Axis::new(vec![0.0, 1.0, 2.0, 3.0], AxisWrapping::None).unwrap();
        let mut rng = rand::rng();
        for _ in 0..100 {
            let sample = axis.random_sample(&mut rng);
            assert!(sample >= 0.0 && sample <= 3.0);
        }
    }

    #[test]
    fn test_linear_interpolation() {
        let axis = Rc::new(Axis::new(vec![0.0, 1.0, 2.0, 3.0], AxisWrapping::None).unwrap());
        let table = LookupTable1D64::new(axis, vec![0.0, 10.0, 20.0, 30.0]);
        assert_eq!(table.linear_interpolate(-1.0), None);
        assert_eq!(table.linear_interpolate(0.0), Some(0.0));
        assert_eq!(table.linear_interpolate(0.5), Some(5.0));
        assert_eq!(table.linear_interpolate(1.0), Some(10.0));
        assert_eq!(table.linear_interpolate(2.25), Some(22.5));
        assert_eq!(table.linear_interpolate(3.0), Some(30.0));
        assert_eq!(table.linear_interpolate(4.0), None);
    }

    #[test]
    fn test_linear_interpolation2() {
        let axis = Rc::new(Axis::new(vec![0.0, 1.0, 2.0, 3.0], AxisWrapping::None).unwrap());
        let table = LookupTable1D::new(axis, vec![
            Vector2::new(0.0, 0.0),
            Vector2::new(1.0, 10.0),
            Vector2::new(2.0, 20.0),
            Vector2::new(3.0, 30.0)
        ]);
        assert_eq!(table.linear_interpolate(-1.0), None);
        assert_eq!(table.linear_interpolate(0.0), Some(Vector2::new(0.0, 0.0)));
        assert_eq!(table.linear_interpolate(0.5), Some(Vector2::new(0.5, 5.0)));
        assert_eq!(table.linear_interpolate(1.0), Some(Vector2::new(1.0, 10.0)));
        assert_eq!(table.linear_interpolate(2.25), Some(Vector2::new(2.25, 22.5)));
        assert_eq!(table.linear_interpolate(3.0), Some(Vector2::new(3.0, 30.0)));
        assert_eq!(table.linear_interpolate(4.0), None);
    }
}