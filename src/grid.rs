use nalgebra::{RealField, Vector2, Vector3};
use std::{rc::Rc, sync::{Arc, Mutex}, cell::RefCell};

/// A vertex in two dimensions
pub trait Vertex {
    type Scalar: RealField + Clone;

    /// Map from two into three dimesions
    fn project(&self, distance: Self::Scalar) -> Vector3<Self::Scalar>;
}

impl<T> Vertex for Vector2<T> 
where T: RealField + Clone
{
    type Scalar = T;

    fn project(&self, distance: Self::Scalar) -> Vector3<Self::Scalar> {
        Vector3::new(self.x.clone(), self.y.clone(), distance)
    }
}

pub type CartesianVertex32 = Vector2<f32>;
pub type CartesianVertex64 = Vector2<f64>;

/// A vertex in two dimensions that can spawn other vertices
pub trait VertexSource
{
    /// Compute the midpoint 
    fn midpoint(x: &Self, y: &Self) -> Self;
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

pub trait IndexedVertexSource {
    type Scalar: RealField + Clone;
    type Vertex: Vertex<Scalar = Self::Scalar>;

    /// Get a vertex by index
    fn get(&self, index: usize) -> Option<Self::Vertex>;

    /// Get the number of vertices in the source
    fn len(&self) -> usize;
}

impl<V, T> IndexedVertexSource for Rc<Vec<V>>
where V: Vertex<Scalar = T> + Clone,
      T: RealField + Clone
{
    type Scalar = T;
    type Vertex = V;

    fn get(&self, index: usize) -> Option<Self::Vertex> {
        self.as_ref().get(index).cloned()
    }

    fn len(&self) -> usize {
        self.as_ref().len()
    }
}

impl<V, T> IndexedVertexSource for Arc<Vec<V>>
where V: Vertex<Scalar = T> + Clone,
      T: RealField + Clone
{
    type Scalar = T;
    type Vertex = V;

    fn get(&self, index: usize) -> Option<Self::Vertex> {
        self.as_ref().get(index).cloned()
    }

    fn len(&self) -> usize {
        self.as_ref().len()
    }
}

impl<V, T> IndexedVertexSource for Rc<RefCell<Vec<(V, (usize, usize))>>>
where V: Vertex<Scalar = T> + Clone,
      T: RealField + Clone
{
    type Scalar = T;
    type Vertex = V;

    fn get(&self, index: usize) -> Option<Self::Vertex> {
        self.borrow().get(index).cloned().map(|(vertex, _)| vertex)
    }

    fn len(&self) -> usize {
        self.borrow().len()
    }
}

impl<V, T> IndexedVertexSource for Arc<Mutex<Vec<(V, (usize, usize))>>>
where V: Vertex<Scalar = T> + Clone,
      T: RealField + Clone
{
    type Scalar = T;
    type Vertex = V;

    fn get(&self, index: usize) -> Option<Self::Vertex> {
        let guard = self.lock().unwrap();
        guard.get(index).cloned().map(|(vertex, _)| vertex)
    }

    fn len(&self) -> usize {
        let guard = self.lock().unwrap();
        guard.len()
    }
}

pub trait IndexedVertexSink
{
    type Scalar: RealField + Clone;
    type Vertex: Vertex<Scalar = Self::Scalar>;

    fn new_collection() -> Self;
    fn seed(&mut self, vertex: Self::Vertex) -> usize;
    fn midpoint(&mut self, v1: usize, v2: usize) -> Option<(Self::Vertex, usize)>;
}

impl<V, T> IndexedVertexSink for Rc<RefCell<Vec<(V, (usize, usize))>>>
where V: Vertex<Scalar = T> + VertexSource + Clone,
      T: RealField + Clone
{
    type Scalar = T;
    type Vertex = V;

    fn new_collection() -> Self {
        Rc::new(RefCell::new(Vec::new()))
    }

    fn seed(&mut self, vertex: Self::Vertex) -> usize {
        let mut guard = self.borrow_mut();
        let index = guard.len();
        guard.push((vertex, (index, index)));
        index
    }

    fn midpoint(&mut self, v1: usize, v2: usize) -> Option<(Self::Vertex, usize)> {
        debug_assert_ne!(v1, v2, "Vertices must be different");
        let key = if v1 < v2 { (v1, v2) } else { (v2, v1) };
        let mut guard = self.borrow_mut();
        if let Some(index) = guard.iter().position(|(_, k)| *k == key) {
            return Some((guard.get(index).expect("Invalid vertex index").0.clone(), index));
        }
        let midpoint = Self::Vertex::midpoint(
            &guard.get(v1)?.0, 
            &guard.get(v2)?.0
        );
        let index = guard.len();
        guard.push((midpoint.clone(), key));
        Some((midpoint, index))
    }
}

/// (I)ndexed (C)artesian (V)ertex (C)ollection with an f32 base
pub type ICVC32 = Rc<RefCell<Vec<(CartesianVertex32, (usize, usize))>>>;

/// (I)ndexed (C)artesian (V)ertex (C)ollection with an f64 base
pub type ICVC64 = Rc<RefCell<Vec<(CartesianVertex64, (usize, usize))>>>;

impl<V, T> IndexedVertexSink for Arc<Mutex<Vec<(V, (usize, usize))>>>
where V: Vertex<Scalar = T> + VertexSource + Clone,
      T: RealField + Clone
{
    type Scalar = T;
    type Vertex = V;

    fn new_collection() -> Self {
        Arc::new(Mutex::new(Vec::new()))
    }

    fn seed(&mut self, vertex: Self::Vertex) -> usize {
        let mut guard = self.lock().unwrap();
        let index = guard.len();
        guard.push((vertex, (index, index)));
        index
    }

    fn midpoint(&mut self, v1: usize, v2: usize) -> Option<(Self::Vertex, usize)> {
        debug_assert_ne!(v1, v2, "Vertices must be different");
        let key = if v1 < v2 { (v1, v2) } else { (v2, v1) };
        let mut guard = self.lock().expect("Unable to lock collection");
        if let Some(index) = guard.iter().position(|(_, k)| *k == key) {
            return Some((guard.get(index).expect("Invalid vertex index").0.clone(), index));
        }
        let midpoint = Self::Vertex::midpoint(
            &guard.get(v1)?.0, 
            &guard.get(v2)?.0
        );
        let index = guard.len();
        guard.push((midpoint.clone(), key));
        Some((midpoint, index))
    }
}

/// (A)tomic (I)ndexed (C)artesian (V)ertex (C)ollection with an f32 base
pub type AICVC32 = Arc<Mutex<Vec<(CartesianVertex32, (usize, usize))>>>;

/// (A)tomic (I)ndexed (C)artesian (V)ertex (C)ollection with an f64 base
pub type AICVC64 = Arc<Mutex<Vec<(CartesianVertex64, (usize, usize))>>>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_indexed_vertex() {
        // Create an indexed vertex collection
        let mut collection = ICVC64::new_collection();

        // Seed some vertices
        let i1 = collection.seed(CartesianVertex64::new(1.0, 0.0));
        let i2 = collection.seed(CartesianVertex64::new(0.0, 1.0));

        // Verify recall
        assert_eq!(collection.get(i1).unwrap(), CartesianVertex64::new(1.0, 0.0));
        assert_eq!(collection.get(i2).unwrap(), CartesianVertex64::new(0.0, 1.0));
        assert_eq!(collection.len(), 2);

        // Midpoint
        let (v3, i3) = collection.midpoint(i1, i2).unwrap();
        assert_eq!(v3, CartesianVertex64::new(0.5, 0.5));
        assert_eq!(i3, 2);
        assert_eq!(collection.len(), 3);
        let (v3_flip, i3_flip) = collection.midpoint(i2, i1).unwrap();
        assert_eq!(v3_flip, v3);
        assert_eq!(i3_flip, i3);
        assert_eq!(collection.len(), 3);
        let (v3_echo, i3_echo) = collection.midpoint(i1, i2).unwrap();
        assert_eq!(v3_echo, CartesianVertex64::new(0.5, 0.5));
        assert_eq!(i3_echo, 2);
        assert_eq!(collection.len(), 3);

        // Midpoint again
        let (v4, i4) = collection.midpoint(i1, i3).unwrap();
        assert_eq!(v4, CartesianVertex64::new(0.75, 0.25));
        assert_eq!(i4, 3);
        assert_eq!(collection.len(), 4);

        // Attempt to re-insert a midpoint
        let (v5, i5) = collection.midpoint(i1, i2).unwrap();
        assert_eq!(v5, CartesianVertex64::new(0.5, 0.5));
        assert_eq!(i5, 2);
        assert_eq!(collection.len(), 4);
    }
}