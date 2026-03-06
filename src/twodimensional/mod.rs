pub mod cartesian;
pub mod spherical;

use nalgebra::{RealField, Vector3};
use std::{rc::Rc, sync::{Arc, Mutex}, cell::RefCell, ops::{Index, IndexMut}};

/// A vertex in two dimensions
pub trait Vertex {
    type Scalar: RealField + Clone;

    /// Map from two into three dimesions
    fn project(&self, distance: Self::Scalar) -> Vector3<Self::Scalar>;
}

/// A vertex in two dimensions that can spawn other vertices
pub trait VertexSource
{
    /// Compute the midpoint 
    fn midpoint(x: &Self, y: &Self) -> Self;
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

#[derive(Clone, PartialEq, PartialOrd)]
pub struct Triangle<G, F>
where G: Vertex<Scalar = F>,
      F: nalgebra::RealField
{
    vertices: [G; 3]
}

impl<G, F> Triangle<G, F>
where G: Vertex<Scalar = F> + PartialOrd + Clone,
      F: nalgebra::RealField
{
    pub fn new(v0: G, v1: G, v2: G) -> Self {
        let mut vertices = [v0, v1, v2];
        vertices.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Triangle {
            vertices,
        }
    }

    pub fn center(&self) -> Vector3<F> {
        let c1 = self.vertices[0].project(F::one());
        let c2 = self.vertices[1].project(F::one());
        let c3 = self.vertices[2].project(F::one());
        let center = (c1 + c2 + c3) / (F::one() + F::one() + F::one());
        center.into()
    }

    pub fn vertices(&self) -> &[G; 3] {
        &self.vertices
    }
}

impl<G, F> Triangle<G, F>
where G: Vertex<Scalar = F> + VertexSource + PartialOrd + Clone,
      F: nalgebra::RealField
{
    /// Creates three triangles by connecting the vertices to an interior point
    pub fn subdivide3(&self, _: G) -> [Triangle<G, F>; 3] 
    {
        todo!()
    }

    /// Subdivides the triangle into for triangles by connecting the midpoints of each edge.
    pub fn subdivide4(&self) -> [Triangle<G, F>; 4] 
    {
        let m0 = G::midpoint(&self.vertices[0],&self.vertices[1]);
        let m1 = G::midpoint(&self.vertices[1],&self.vertices[2]);
        let m2 = G::midpoint(&self.vertices[2],&self.vertices[0]);

        debug_assert!(m0 == G::midpoint(&self.vertices[1],&self.vertices[0]));
        debug_assert!(m1 == G::midpoint(&self.vertices[2],&self.vertices[1]));
        debug_assert!(m2 == G::midpoint(&self.vertices[0],&self.vertices[2]));

        [
            Triangle::new(self.vertices[0].clone(), m0.clone(), m2.clone()),
            Triangle::new(self.vertices[1].clone(), m1.clone(), m0.clone()),
            Triangle::new(self.vertices[2].clone(), m2.clone(), m1.clone()),
            Triangle::new(m0, m1, m2),
        ]
    }
}

impl<G, F> Index<usize> for Triangle<G, F>
where G: Vertex<Scalar = F>,
      F: nalgebra::RealField
{
    type Output = G;

    fn index(&self, index: usize) -> &Self::Output {
        &self.vertices[index]
    }
}

impl<G, F> IndexMut<usize> for Triangle<G, F>
where G: Vertex<Scalar = F>,
      F: nalgebra::RealField
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.vertices[index]
    }
}

impl<G, F> IntoIterator for Triangle<G, F>
where G: Vertex<Scalar = F> + Clone,
      F: nalgebra::RealField
{
        type Item = G;
        type IntoIter = std::vec::IntoIter<G>;

    fn into_iter(self) -> std::vec::IntoIter<G> {
        self.vertices.to_vec().into_iter()
    }
}

#[derive(Clone, Debug)]
pub struct SharedTriangle<C> 
where C: Clone
{
    vertices: [usize; 3],
    collection: C
}

impl<C> SharedTriangle<C> 
where C: Clone
{
    pub fn new(v0: usize, v1: usize, v2: usize, collection: C) -> Self {
        let mut vertices = [v0, v1, v2];
        vertices.sort();
        SharedTriangle {
            vertices,
            collection,
        }
    }

    pub fn collection(&self) -> &C {
        &self.collection
    }

    pub fn edge_indices(&self) -> [[usize; 2]; 3] {
        let mut result = [
            [self.vertices[0], self.vertices[1]],
            [self.vertices[1], self.vertices[2]],
            [self.vertices[2], self.vertices[0]],
        ];
        for edge in &mut result {
            edge.sort();
        }
        result
    }
}

impl<C, V, T> SharedTriangle<C> 
where C: IndexedVertexSource<Scalar = T, Vertex = V> + Clone,
      V: Vertex<Scalar = T> + PartialOrd + Clone,
      T: RealField + Clone
{
    pub fn load(&self) -> Triangle<V, T> {
        let v0 = self.collection.get(self.vertices[0]).expect("Vertex index out of bounds");
        let v1 = self.collection.get(self.vertices[1]).expect("Vertex index out of bounds");
        let v2 = self.collection.get(self.vertices[2]).expect("Vertex index out of bounds");
        Triangle::new(v0, v1, v2)
    }
}

impl<C, V, T> SharedTriangle<C> 
where C: IndexedVertexSource<Scalar = T, Vertex = V> + IndexedVertexSink<Scalar = T, Vertex = V> + Clone,
      V: Vertex<Scalar = T> + VertexSource + PartialOrd + Clone,
      T: RealField + Clone
{
    pub fn subdivide4(&mut self) -> Option<[SharedTriangle<C>; 4]> {
        let (_, m0_index) = self.collection.midpoint(self.vertices[0],self.vertices[1])?;
        let (_, m1_index) = self.collection.midpoint(self.vertices[1],self.vertices[2])?;
        let (_, m2_index) = self.collection.midpoint(self.vertices[2],self.vertices[0])?;

        let collection = self.collection.clone();

        Some([
            SharedTriangle::new(self.vertices[0], m0_index, m2_index, collection.clone()),
            SharedTriangle::new(self.vertices[1], m1_index, m0_index, collection.clone()),
            SharedTriangle::new(self.vertices[2], m2_index, m1_index, collection.clone()),
            SharedTriangle::new(m0_index, m1_index, m2_index, collection.clone()),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cartesian::{CartesianVertex64, ICVC64};

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

    #[test]
    fn test_shared_triangle() {
        // Create an indexed vertex collection
        let mut collection = ICVC64::new_collection();

        // Create the initial triangle
        let v0 = CartesianVertex64::new(-1.0, 0.0);
        let v1 = CartesianVertex64::new(0.0, 1.0);
        let v2 = CartesianVertex64::new(1.0, 0.0);
        let i0 = collection.seed(v0);
        let i1 = collection.seed(v1);
        let i2 = collection.seed(v2);
        assert_eq!(collection.len(), 3);

        let mut triangle = SharedTriangle::new(i0, i1, i2, collection.clone());
        let parts = triangle.subdivide4().unwrap();
        assert_eq!(collection.len(), 6);
        for mut part in parts {
            part.subdivide4();
        }
        assert_eq!(collection.len(), 15);
    }
}