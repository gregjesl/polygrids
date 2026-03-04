
use std::{ops::{Index, IndexMut, Mul, Add}, rc::Rc};
use nalgebra::RealField;
use crate::{Axis32, Axis64};

use super::Axis;

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

pub type LookupTable1D32 = LookupTable1D<f32, f32>;
pub type LookupTable1D64 = LookupTable1D<f64, f64>;

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
                let mut children: Vec<LookupTable<G, T>> = Vec::with_capacity(first_axis.points.len());
                for _ in 0..first_axis.points.len() {
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
            LookupTable::Root(table) => table.values.len(),
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
                table.values[indices[0]] = value;
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
                        result.push(format!("{},{}", axis.points[i], line));
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