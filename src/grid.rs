use std::{marker::PhantomData, ops::Index};
use nalgebra::{RealField, Vector2, Vector3};
use std::{rc::Rc, cell::RefCell};

/// A vertex in two dimensions
pub trait Vertex<T> {
    /// Map from two into three dimesions
    fn project(&self, distance: T) -> Vector3<T>;
}

impl<T> Vertex<T> for Vector2<T> 
where T: RealField + Clone
{
    fn project(&self, distance: T) -> Vector3<T> {
        Vector3::new(self.x.clone(), self.y.clone(), distance)
    }
}

pub type CartesianVertex32 = Vector2<f32>;
pub type CartesianVertex64 = Vector2<f64>;

/// A vertex in two dimensions that can spawn other vertices
pub trait VertexSource<T>
{
    /// Compute the midpoint 
    fn midpoint(x: &Self, y: &Self) -> Self;
}

impl<T> VertexSource<T> for Vector2<T>
where T: RealField
{
    fn midpoint(x: &Self, y: &Self) -> Self {
        if x < y {
            (x + y) / (T::one() + T::one())
        } else {
            (y + x) / (T::one() + T::one())
        }
    }
}

struct IndexedVertexCollection<G, T>
where G: Vertex<T> + VertexSource<T> + Clone,
      T: RealField 
{
    map: Vec<(G, (usize, usize))>,
    _marker: std::marker::PhantomData<T>
}

impl<G, T> IndexedVertexCollection<G, T>
where G: Vertex<T> + VertexSource<T> + Clone,
      T: RealField
{
    fn new() -> Self {
        Self {
            map: Vec::new(),
            _marker: std::marker::PhantomData
        }
    }

    fn seed(&mut self, vertex: G) -> (&G, usize) {
        let index = self.map.len();
        let id = (index, index);
        self.map.push((vertex, id));
        let (vertex, echo) = self.map.last().unwrap();
        debug_assert_eq!(id, *echo, "ID should match the index");
        (vertex, index)
    }

    fn midpoint(&mut self, v1: usize, v2: usize) -> (&G, usize) {
        debug_assert_ne!(v1, v2, "Vertices must be different");
        let id = if v1 < v2 { (v1, v2) } else { (v2, v1) };
        if let Some(index) = self.map.iter().position(|(_, cid)| id == *cid) {
            let (vertex, _) = &self.map[index];
            return (vertex, index)
        } else {
            let g1 = self.map[v1].0.clone();
            let g2 = self.map[v2].0.clone();
            let midpoint = G::midpoint(&g1, &g2);
            let index = self.map.len();
            self.map.push((midpoint, id));
            let (vertex, echo) = self.map.last().unwrap();
            debug_assert_eq!(id, *echo, "ID should match the index");
            (vertex, index)
        }
    }
}

#[derive(Clone)]
pub struct SharedIndexedVertexCollection<G, T> 
where G: Vertex<T> + VertexSource<T> + Clone,
      T: RealField
{
    pub(crate) collection: Rc<RefCell<IndexedVertexCollection<G, T>>>
}

impl<G, T> SharedIndexedVertexCollection<G, T> 
where G: Vertex<T> + VertexSource<T> + Clone,
      T: RealField
{
    pub fn new() -> Self {
        Self {
            collection: Rc::new(RefCell::new(IndexedVertexCollection::new())),
        }
    }

    pub(crate) fn seed(&self, vertex: G) -> (G, usize) {
        let mut binding = self.collection.borrow_mut();
        let (vertex, index) = binding.seed(vertex);
        (vertex.clone(), index)
    }

    pub(crate) fn midpoint(&self, v1: usize, v2: usize) -> (G, usize) {
        let mut binding = self.collection.borrow_mut();
        let (vertex, index) = binding.midpoint(v1, v2);
        (vertex.clone(), index)
    }
}

impl<G, T> From<SharedIndexedVertexCollection<G, T>> for Vec<G>
where G: Vertex<T> + VertexSource<T> + Clone,
      T: RealField
{
    fn from(collection: SharedIndexedVertexCollection<G, T>) -> Self {
        let binding = collection.collection.borrow();
        binding.map.iter().map(|(vertex, _)| vertex.clone()).collect()
    }
}

pub type CartesianVertexCollection32 = SharedIndexedVertexCollection<CartesianVertex32, f32>;
pub type CartesianVertexCollection64 = SharedIndexedVertexCollection<CartesianVertex64, f64>;

pub struct SharedVertex<G, T> 
where G: Vertex<T> + VertexSource<T> + Clone,
      T: RealField
{
    index: usize,
    collection: Rc<RefCell<IndexedVertexCollection<G, T>>>
}

impl<G, T> SharedVertex<G, T> 
where G: Vertex<T> + VertexSource<T> + Clone,
      T: RealField
{
    pub fn new(vertex: G, collection: SharedIndexedVertexCollection<G, T>) -> Self {
        let (vertex, index) = collection.seed(vertex);
        Self {
            index,
            collection: collection.collection.clone(),
        }
    }

    pub fn vertex(&self) -> G {
        self.collection.borrow().map[self.index].0.clone()
    }

    pub fn index(&self) -> usize {
        self.index
    }
}

impl<G, T> Vertex<T> for SharedVertex<G, T>
where G: Vertex<T> + VertexSource<T> + Clone,
      T: RealField
{
    fn project(&self, distance: T) -> Vector3<T> {
        self.collection.borrow().map[self.index].0.project(distance)
    }
}

impl<G, T> VertexSource<T> for SharedVertex<G, T>
where G: Vertex<T> + VertexSource<T> + Clone,
      T: RealField
{
    fn midpoint(v1: &Self, v2: &Self) -> Self {
        debug_assert!(Rc::ptr_eq(&v1.collection, &v2.collection), "Vertices must belong to the same collection");
        let (vertex, index) = v1.collection.borrow_mut().midpoint(v1.index, v2.index);
        Self {
            index,
            collection: v1.collection.clone(),
        }
    }
}

pub type SharedCartesianVertex32 = SharedVertex<CartesianVertex32, f32>;
pub type SharedCartesianVertex64 = SharedVertex<CartesianVertex64, f64>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_indexed_vertex() {
        // Create an indexed vertex collection
        let collection = CartesianVertexCollection64::new();

        // Seed some vertices
        let v1 = SharedCartesianVertex64::new(CartesianVertex64::new(1.0, 0.0), collection.clone());
        let v2 = SharedCartesianVertex64::new(CartesianVertex64::new(0.0, 1.0), collection.clone());

        // Verify recall
        assert_eq!(v1.vertex(), CartesianVertex64::new(1.0, 0.0));
        assert_eq!(v1.index(), 0);
        assert_eq!(v2.vertex(), CartesianVertex64::new(0.0, 1.0));
        assert_eq!(v2.index(), 1);

        // Midpoint
        let v3 = SharedCartesianVertex64::midpoint(&v1, &v2);
        assert_eq!(v3.vertex(), CartesianVertex64::new(0.5, 0.5));
        assert_eq!(v3.index(), 2);

        // Midpoint again
        let v4 = SharedCartesianVertex64::midpoint(&v1, &v3);
        assert_eq!(v4.vertex(), CartesianVertex64::new(0.75, 0.25));
        assert_eq!(v4.index(), 3);

        // Verify collection
        let vertices: Vec<CartesianVertex64> = collection.into();
        assert_eq!(vertices.len(), 4);
        assert_eq!(vertices[0], CartesianVertex64::new(1.0, 0.0));
        assert_eq!(vertices[1], CartesianVertex64::new(0.0, 1.0));
        assert_eq!(vertices[2], CartesianVertex64::new(0.5, 0.5));
        assert_eq!(vertices[3], CartesianVertex64::new(0.75, 0.25));
    }
}