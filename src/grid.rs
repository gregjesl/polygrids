use std::{marker::PhantomData, ops::Index};
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

trait IndexedVertexSource<G> {
    fn get(&self, index: usize) -> Option<G>;
    fn len(&self) -> usize;
}

impl<G, T> IndexedVertexSource<G> for Rc<Vec<G>>
where G: Vertex<Scalar = T> + Clone,
      T: RealField
{
    fn get(&self, index: usize) -> Option<G> {
        self.as_ref().get(index).cloned()
    }

    fn len(&self) -> usize {
        self.as_ref().len()
    }
}

impl<G, T> IndexedVertexSource<G> for Arc<Vec<G>>
where G: Vertex<Scalar = T> + Clone,
      T: RealField
{
    fn get(&self, index: usize) -> Option<G> {
        self.as_ref().get(index).cloned()
    }

    fn len(&self) -> usize {
        self.as_ref().len()
    }
}

impl<G, T> IndexedVertexSource<G> for Rc<RefCell<Vec<(G, (usize, usize))>>>
where G: Vertex<Scalar = T> + Clone
{
    fn get(&self, index: usize) -> Option<G> {
        self.borrow().get(index).cloned().map(|(vertex, _)| vertex)
    }

    fn len(&self) -> usize {
        self.borrow().len()
    }
}

impl<G, T> IndexedVertexSource<G> for Arc<Mutex<Vec<(G, (usize, usize))>>>
where G: Vertex<Scalar = T> + Clone
{
    fn get(&self, index: usize) -> Option<G> {
        let guard = self.lock().unwrap();
        guard.get(index).cloned().map(|(vertex, _)| vertex)
    }

    fn len(&self) -> usize {
        let guard = self.lock().unwrap();
        guard.len()
    }
}

trait IndexedVertexSink<G, T> 
where G: Vertex<Scalar = T>,
      T: RealField + Clone
{
    fn seed(&mut self, vertex: G) -> usize;
    fn midpoint(&mut self, v1: usize, v2: usize) -> usize;
}

impl<G, T> IndexedVertexSink<G, T> for Rc<RefCell<Vec<(G, (usize, usize))>>>
where G: Vertex<Scalar = T> + VertexSource + Clone,
      T: RealField + Clone
{
    fn seed(&mut self, vertex: G) -> usize {
        let mut guard = self.borrow_mut();
        let index = guard.len();
        guard.push((vertex, (index, index)));
        index
    }

    fn midpoint(&mut self, v1: usize, v2: usize) -> usize {
        debug_assert_ne!(v1, v2, "Vertices must be different");
        let key = if v1 < v2 { (v1, v2) } else { (v2, v1) };
        let mut guard = self.borrow_mut();
        if let Some(index) = guard.iter().position(|(_, k)| *k == key) {
            return index;
        }
        let midpoint = G::midpoint(
            &guard.get(v1).expect("Invalid vertex index").0, 
            &guard.get(v2).expect("Invalid vertex index").0
        );
        let index = guard.len();
        guard.push((midpoint, key));
        index
    }
}

impl<G, T> IndexedVertexSink<G, T> for Arc<Mutex<Vec<(G, (usize, usize))>>>
where G: Vertex<Scalar = T> + VertexSource + Clone,
      T: RealField + Clone
{
    fn seed(&mut self, vertex: G) -> usize {
        let mut guard = self.lock().unwrap();
        let index = guard.len();
        guard.push((vertex, (index, index)));
        index
    }

    fn midpoint(&mut self, v1: usize, v2: usize) -> usize {
        debug_assert_ne!(v1, v2, "Vertices must be different");
        let key = if v1 < v2 { (v1, v2) } else { (v2, v1) };
        let mut guard = self.lock().unwrap();
        if let Some(index) = guard.iter().position(|(_, k)| *k == key) {
            return index;
        }
        let midpoint = G::midpoint(
            &guard.get(v1).expect("Invalid vertex index").0, 
            &guard.get(v2).expect("Invalid vertex index").0
        );
        let index = guard.len();
        guard.push((midpoint, key));
        index
    }
}

pub struct SharedVertex<G, T, C> 
where G: Vertex<Scalar = T> + VertexSource + Clone,
      T: RealField + Clone
{
    index: usize,
    collection: C,
    _markerg: PhantomData<G>,
    _markert: PhantomData<T>,
}

impl<G, T, C> SharedVertex<G, T, C> 
where G: Vertex<Scalar = T> + VertexSource + Clone,
      T: RealField + Clone,
      C: Default
{
    pub fn new_collection() -> C {
        C::default()
    }
}

impl<G, T, C> SharedVertex<G, T, C> 
where G: Vertex<Scalar = T> + VertexSource + Clone,
      T: RealField + Clone,
      C: IndexedVertexSink<G, T> + IndexedVertexSource<G> + Clone
{
    pub fn new(vertex: G, mut collection: C) -> Self {
        let index = collection.seed(vertex);
        Self {
            index,
            collection,
            _markerg: PhantomData,
            _markert: PhantomData,
        }
    }

    pub fn insert_midpoint(&mut self, other: &Self) -> Self {
        let index = self.collection.midpoint(self.index(), other.index());
        Self {
            index,
            collection: self.collection.clone(),
            _markerg: PhantomData,
            _markert: PhantomData,
        }
    }
}

impl<G, T, C> SharedVertex<G, T, C> 
where G: Vertex<Scalar = T> + VertexSource + Clone,
      T: RealField + Clone,
      C: IndexedVertexSource<G>
{
    fn vertex(&self) -> G {
        self.collection.get(self.index).expect("Invalid vertex index")
    }

    fn index(&self) -> usize {
        self.index
    }
}

impl<G, T, C> Vertex for SharedVertex<G, T, C> 
where G: Vertex<Scalar = T> + VertexSource + Clone,
      T: RealField + Clone,
      C: IndexedVertexSource<G>
{
    type Scalar = T;

    fn project(&self, distance: Self::Scalar) -> Vector3<Self::Scalar> {
        self.vertex().project(distance)
    }
}

/// Cartesian shared vertex with an f32 base
pub type CSV32 = SharedVertex<CartesianVertex32, f32, Rc<RefCell<Vec<(CartesianVertex32, (usize, usize))>>>>;

/// Cartesian shared vertex with an f64 base
pub type CSV64 = SharedVertex<CartesianVertex64, f64, Rc<RefCell<Vec<(CartesianVertex64, (usize, usize))>>>>;

/// Atomic cartesian shared vertex with an f32 base
pub type ACSV32 = SharedVertex<CartesianVertex32, f32, Arc<Mutex<Vec<(CartesianVertex32, (usize, usize))>>>>;

/// Atomic cartesian shared vertex with an f64 base
pub type ACSV64 = SharedVertex<CartesianVertex64, f64, Arc<Mutex<Vec<(CartesianVertex64, (usize, usize))>>>>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_indexed_vertex() {
        // Create an indexed vertex collection
        let collection = CSV64::new_collection();

        // Seed some vertices
        let mut v1 = CSV64::new(CartesianVertex64::new(1.0, 0.0), collection.clone());
        let v2 = CSV64::new(CartesianVertex64::new(0.0, 1.0), collection.clone());

        // Verify recall
        assert_eq!(v1.vertex(), CartesianVertex64::new(1.0, 0.0));
        assert_eq!(v1.index(), 0);
        assert_eq!(v2.vertex(), CartesianVertex64::new(0.0, 1.0));
        assert_eq!(v2.index(), 1);
        assert_eq!(collection.len(), 2);

        // Midpoint
        let v3 = v1.insert_midpoint(&v2);
        assert_eq!(v3.vertex(), CartesianVertex64::new(0.5, 0.5));
        assert_eq!(v3.index(), 2);
        assert_eq!(collection.len(), 3);

        // Midpoint again
        let v4 = v1.insert_midpoint(&v3);
        assert_eq!(v4.vertex(), CartesianVertex64::new(0.75, 0.25));
        assert_eq!(v4.index(), 3);
        assert_eq!(collection.len(), 4);

        // Attempt to re-insert a midpoint
        let v5 = v1.insert_midpoint(&v2);
        assert_eq!(v5.vertex(), CartesianVertex64::new(0.5, 0.5));
        assert_eq!(v5.index(), v3.index());
        assert_eq!(collection.len(), 4);

        /* 
        // Verify collection
        let vertices: Vec<CartesianVertex64> = collection.into();
        assert_eq!(vertices.len(), 4);
        assert_eq!(vertices[0], CartesianVertex64::new(1.0, 0.0));
        assert_eq!(vertices[1], CartesianVertex64::new(0.0, 1.0));
        assert_eq!(vertices[2], CartesianVertex64::new(0.5, 0.5));
        assert_eq!(vertices[3], CartesianVertex64::new(0.75, 0.25));
        */
    }
}