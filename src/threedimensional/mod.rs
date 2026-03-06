use nalgebra::{Vector3, Matrix3};

use crate::{
    Axis64,
    onedimensional::LookupTable1D,
    twodimensional::{Vertex, IndexedVertexSource, IndexedVertexSink, spherical::{Ray, Polyhedron, AtomicPolyhedron, Polyhedral, SharedRayCollection, AtomicRayCollection}}
};
use std::{
    rc::Rc,
    ops::{Add, Mul}
};

pub struct SphericalField<P, C, T> 
where P: Polyhedral<Collection = C>, 
      C: IndexedVertexSource<Scalar = f64, Vertex = Ray> + IndexedVertexSink<Scalar = f64, Vertex = Ray> + Clone, 
      T: Clone
{
    polyhedron: P,
    columns: Vec<LookupTable1D<f64, T>>,
}

impl<P,C,T> SphericalField<P, C, T> 
where P: Polyhedral<Collection = C>, 
      C: IndexedVertexSource<Scalar = f64, Vertex = Ray> + IndexedVertexSink<Scalar = f64, Vertex = Ray> + Clone, 
      T: Clone
{

    /// Create a spherical field using a fill function
    /// 
    /// # Arguments
    /// * `subdivisions` - The number of times to subdivide the base icosahedron. Each subdivision multiplies the number of triangles by 4.
    /// * `radial_axis` - The axis to use for the radial dimension. This will be the axis used for all radial columns. 
    /// * `fill` - A function that takes a ray and a distance and returns a value of type T. This will be used to fill the columns of the field.
    pub fn new<F>(subdivisions: usize, radial_axis: Axis64, fill: F) -> Self 
    where F: Fn(Ray, f64) -> T
    {
        let mut polyhedron = P::new();
        for _ in 0..subdivisions {
            polyhedron.subdivide();
        }
        let axisrc = Rc::new(radial_axis);
        
        let columns = polyhedron.collection().iter()
            .map(|v| LookupTable1D::fill(axisrc.clone(), |r| fill(v.clone(), r)))
            .collect();

        SphericalField { polyhedron, columns }
    }
}

impl<P, C> SphericalField<P, C, f64>
where P: Polyhedral<Collection = C>, 
      C: IndexedVertexSource<Scalar = f64, Vertex = Ray> + IndexedVertexSink<Scalar = f64, Vertex = Ray> + Clone
{
    pub fn interpolate(&self, ray: &Ray, radius: f64) -> Option<f64> {
        // Find the face
        let face = self.polyhedron.find_face(ray);

        // Get the indicies of the face
        let [i0, i1, i2] = *face.vertices();

        // Interpolate from the lookup tables
        let b = Vector3::new(
            self.columns[i0].linear_interpolate(radius)?,
            self.columns[i1].linear_interpolate(radius)?,
            self.columns[i2].linear_interpolate(radius)?
        );

        let matrix = Matrix3::from_rows(&[
            face.load()[0].project(radius).transpose(),
            face.load()[1].project(radius).transpose(),
            face.load()[2].project(radius).transpose()
        ]);

        let coefficients: Vector3<f64> = matrix.lu().solve(&b)?;
        let point = ray.project(radius);
        return Some(coefficients.dot(&point));
    }
}

pub type ScalarSphericalField = SphericalField<Polyhedron, SharedRayCollection, f64>;
pub type AtomicScalarSphericalField = SphericalField<AtomicPolyhedron, AtomicRayCollection, f64>;

impl<P, C> SphericalField<P, C, Vector3<f64>>
where P: Polyhedral<Collection = C>, 
      C: IndexedVertexSource<Scalar = f64, Vertex = Ray> + IndexedVertexSink<Scalar = f64, Vertex = Ray> + Clone
{
    pub fn interpolate(&self, ray: &Ray, radius: f64) -> Option<Vector3<f64>> {
        // Find the face
        let face = self.polyhedron.find_face(ray);

        // Get the indicies of the face
        let [i0, i1, i2] = *face.vertices();

        // Interpolate from the lookup tables
        let b = Matrix3::from_rows(&[
            self.columns[i0].linear_interpolate(radius)?.transpose(),
            self.columns[i1].linear_interpolate(radius)?.transpose(),
            self.columns[i2].linear_interpolate(radius)?.transpose()
        ]);

        let matrix = Matrix3::from_rows(&[
            face.load()[0].project(radius).transpose(),
            face.load()[1].project(radius).transpose(),
            face.load()[2].project(radius).transpose()
        ]);

        let coefficients = matrix.lu().solve(&b)?;
        let point = ray.project(radius);
        return Some((point.transpose() * coefficients).transpose());
    }
}

pub type ScalarSpherical3 = SphericalField<Polyhedron, SharedRayCollection, Vector3<f64>>;
pub type AtomicScalarSpherical3 = SphericalField<AtomicPolyhedron, AtomicRayCollection, Vector3<f64>>;