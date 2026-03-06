use super::{Vertex, VertexSource};
use nalgebra::{RealField, Vector2, Vector3};
use std::{rc::Rc, sync::{Arc, Mutex}, cell::RefCell};

pub type CartesianVertex32 = Vector2<f32>;
pub type CartesianVertex64 = Vector2<f64>;

impl<T> Vertex for Vector2<T> 
where T: RealField + Clone
{
    type Scalar = T;

    fn project(&self, distance: Self::Scalar) -> Vector3<Self::Scalar> {
        Vector3::new(self.x.clone(), self.y.clone(), distance)
    }
}

impl<T> VertexSource for Vector2<T>
where T: RealField + Clone
{
    fn midpoint(x: &Self, y: &Self) -> Self {
        if x < y {
            (x + y) / (T::one() + T::one())
        } else {
            (y + x) / (T::one() + T::one())
        }
    }
}

/// (I)ndexed (C)artesian (V)ertex (C)ollection with an f32 base
pub type ICVC32 = Rc<RefCell<Vec<(CartesianVertex32, (usize, usize))>>>;

/// (I)ndexed (C)artesian (V)ertex (C)ollection with an f64 base
pub type ICVC64 = Rc<RefCell<Vec<(CartesianVertex64, (usize, usize))>>>;

/// (A)tomic (I)ndexed (C)artesian (V)ertex (C)ollection with an f32 base
pub type AICVC32 = Arc<Mutex<Vec<(CartesianVertex32, (usize, usize))>>>;

/// (A)tomic (I)ndexed (C)artesian (V)ertex (C)ollection with an f64 base
pub type AICVC64 = Arc<Mutex<Vec<(CartesianVertex64, (usize, usize))>>>;