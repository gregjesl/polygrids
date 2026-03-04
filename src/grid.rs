use std::marker::PhantomData;

use nalgebra::{Vector2, Vector3};

/// A vertex in two dimensions
pub trait Vertex<T> {
    /// Map from two into three dimesions
    fn project(&self, distance: T) -> Vector3<T>;
}

/// A vertex in two dimensions that can spawn other vertices
pub trait VertexSource<T>: From<Vector3<T>> {

    /// Compute the midpoint 
    fn midpoint(x: &Self, y: &Self) -> Self;

    /// Compute the center of three vertices
    fn center(x: &Self, y: &Self, z: &Self) -> Self;
}

/// Vertex tagged with an ID. 
/// 
/// This can be used for shared vertices. 
pub struct EnumeratedVertex<V, T> 
where V: Vertex<T>
{
    id: usize,
    vertex: V,
    _phantom: PhantomData<T>
}