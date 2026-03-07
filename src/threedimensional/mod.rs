use std::rc::Rc;
use nalgebra::{Vector3, Matrix3};
use crate::{
    Axis64,
    onedimensional::LookupTable1D,
    twodimensional::{Vertex, IndexedVertexSource, spherical::{Ray, Polyhedron}}
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub struct SphericalField<T> 
{
    polyhedron: Polyhedron,
    columns: Vec<LookupTable1D<f64, T>>,
}

impl<T> SphericalField<T> 
where T: Clone
{

    /// Create a spherical field using a fill function
    /// 
    /// # Arguments
    /// * `subdivisions` - The number of times to subdivide the base icosahedron. Each subdivision multiplies the number of triangles by 4.
    /// * `radial_axis` - The axis to use for the radial dimension. This will be the axis used for all radial columns. 
    /// * `fill` - A function that takes a ray and a distance and returns a value of type T. This will be used to fill the columns of the field.
    /// 
    /// A multi-threaded version of this function exists, see [`SphericalField::parallel_new()`]
    /// 
    /// # Example
    /// 
    /// This example encodes cartesian coordinate in the spherical field
    /// 
    /// ```
    /// use polygrids::{Axis64, twodimensional::{Vertex, IndexedVertexSource}, threedimensional::SphericalField};
    /// 
    /// let axis = Axis64::linear(1.0, 10.0, 10).unwrap();
    /// let field = SphericalField::new(2, axis.clone(), {|ray, r|
    ///     ray.project(r)
    /// });
    /// 
    /// // Iterate through all of the vertices
    /// for i in 0..field.polyhedron().collection().len() {
    ///     // Iterate through all radial points
    ///     for j in 0..axis.points().len() {
    ///         let ray = field.polyhedron().collection().get(i).unwrap();
    ///         let r = axis.points()[j];
    /// 
    ///         // Reconstruct the value
    ///         assert_eq!(field.get(i, j).unwrap(), &ray.project(r));
    ///     }
    /// }
    /// ```
    pub fn new<F>(subdivisions: usize, radial_axis: Axis64, fill: F) -> Self 
    where F: Fn(Ray, f64) -> T
    {
        let mut polyhedron = Polyhedron::new();
        for _ in 0..subdivisions {
            polyhedron.subdivide();
        }

        let axisrc = Rc::new(radial_axis);

        #[cfg(not(feature = "progress"))]
        {
            let columns = polyhedron.collection().iter()
                .map(|v| LookupTable1D::fill(axisrc.clone(), |r| fill(v.clone(), r)))
                .collect();

            SphericalField { polyhedron, columns }
        }

        #[cfg(feature = "progress")]
        {
            use indicatif::{ProgressBar, ProgressStyle};
            let bar = ProgressBar::new(polyhedron.collection().len() as u64);
            bar.set_style(ProgressStyle::with_template("{msg} {eta}: {bar}").unwrap());
            bar.set_message("Filling scalar field");
            bar.finish_and_clear();
            let columns = polyhedron.collection().iter()
                .map(|v| {
                    let result = LookupTable1D::fill(axisrc.clone(), |r| fill(v.clone(), r));
                    bar.inc(1);
                    result
                })
                .collect();
            
            bar.finish_and_clear();

            SphericalField { polyhedron, columns }
        }
    }

    /// Create a spherical field using a fill function. The fill function is called using a parallel iterator. 
    /// 
    /// See [`SphericalField::new()`] for more details.  
    #[cfg(feature = "parallel")]
    pub fn parallel_new<F>(subdivisions: usize, radial_axis: Axis64, fill: F) -> Self 
    where F: Fn(Ray, f64) -> T + std::marker::Sync,
          T: Send
    {
        // Create the polyhedron

        use rayon::iter::IntoParallelRefIterator;
        let mut polyhedron = Polyhedron::new();
        for _ in 0..subdivisions {
            polyhedron.subdivide();
        }

        // Build the work queue
        let queue: Vec<Ray> = polyhedron.collection().iter().collect();
        let values: Vec<Vec<T>> = queue.par_iter()
            .map(|ray| {
                radial_axis.points().iter().map(|r| {
                    fill(*ray,*r)
                }).collect()
            })
            .collect();

        let axis_rc = Rc::new(radial_axis);

        let columns: Vec<_> = values.iter().map(| value| {
            LookupTable1D::new(axis_rc.clone(), value.clone())
        }).collect();

        Self { polyhedron, columns }
    }

    pub fn get(&self, ray: usize, radial: usize) -> Option<&T> {
        self.columns.get(ray)?.get(radial)
    }

    pub fn polyhedron(&self) -> &Polyhedron {
        &self.polyhedron
    }
}

impl SphericalField<f64> {
    /// Interpolate a value given a spherical coordinate
    /// 
    /// See [SphericalField<Vector3<f64>>::interpolate()] for the three-dimensional version
    /// # Returns
    /// - `Some(value)` if the coordinate lies in in the field, where `value` is the interpolated value
    /// - `None` if the coordinate is outside the field
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

impl SphericalField<Vector3<f64>>
{
    /// Interpolate a value given a spherical coordinate
    /// 
    /// See [SphericalField<f64>::interpolate()] for the scalar version
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

#[cfg(test)]
mod tests {
    use super::*;
    
    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel() {
        let axis = Axis64::linear(1.0, 10.0, 10).unwrap();
        let field = SphericalField::new(2, axis.clone(), {|ray, r| 
            ray.project(r)
        });

        let field_parallel = SphericalField::parallel_new(2, axis.clone(), {|ray, r| 
            ray.project(r)
        });

        assert_eq!(field.polyhedron().collection().len(), field_parallel.polyhedron().collection().len());
        for i in 0..field.polyhedron().collection().len() {
            for j in 0..axis.points().len() {
                let ray = field.polyhedron().collection().get(i).unwrap();
                assert_eq!(ray, field_parallel.polyhedron().collection().get(i).unwrap());
                assert_eq!(field.get(i, j).unwrap(), field_parallel.get(i, j).unwrap())
            }
        }
    }
}