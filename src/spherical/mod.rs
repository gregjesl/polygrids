mod faces;
use std::{cell::RefCell, rc::Rc, sync::{Arc, Mutex}};
use nalgebra::Vector3;
use crate::{geometry::SharedTriangle, grid::{IndexedVertexSink, IndexedVertexSource, Vertex, VertexSource}, spherical::faces::{FaceBranch, FaceTree}};
use angle_sc::{Angle, Degrees, Radians};

#[cfg(feature = "rand")]
use rand::Rng;

#[derive(Copy, Clone, Debug, Default, PartialEq, PartialOrd)]
pub struct Ray
{
    /// Latitude
    lat: Angle,

    /// Longitude
    lon: Angle
}

impl Ray
{
    pub fn new(lat: Angle, lon: Angle) -> Self
    {
        debug_assert!(lat.abs() <= Angle::from(Degrees(90.0)));
        debug_assert!(lon.abs() <= Angle::from(Degrees(180.0)));
        Self { lat, lon }
    }

    pub fn latitude(&self) -> &Angle
    {
        &self.lat
    }

    pub fn longitude(&self) -> &Angle
    {
        &self.lon
    }

    pub fn separation(&self, other: &Self) -> Angle
    {
        let v1 = self.project(1.0);
        let v2 = other.project(1.0);
        let dotprod = v1.dot(&v2).clamp(-1.0, 1.0);
        Radians(dotprod.acos()).into()
    }

    #[cfg(feature = "rand")]
    pub fn random<R: Rng + ?Sized>(rng: &mut R) -> Self
    {
        use rand::RngExt;

        let u = rng.random_range(0.0..1.0);
        let v = rng.random_range(0.0..1.0);
        let theta = 2.0 * std::f64::consts::PI * u;
        let phi: f64 = (2.0f64 * v - 1.0).acos();
        let x = theta.sin() * phi.cos();
        let y = theta.sin() * phi.sin();
        let z = phi.cos();
        debug_assert!(x.is_finite() && y.is_finite() && z.is_finite());
        let point = Vector3::new(x, y, z);
        point.into()
    }
}

impl Vertex for Ray {
    type Scalar = f64;

    fn project(&self, distance: f64) -> Vector3<f64> {
        let x = distance * self.lat.cos().0 * self.lon.cos().0;
        let y = distance * self.lat.cos().0 * self.lon.sin().0;
        let z = distance * self.lat.sin().0;
        Vector3::new(x, y, z)
    }
}

impl From<Vector3<f64>> for Ray {
    fn from(coord: Vector3<f64>) -> Self {
        let r = coord.norm();
        if r == 0.0 { return Ray { lat: Angle::default(), lon: Angle::default() }}
        let xynorm = (coord.x * coord.x + coord.y * coord.y).sqrt();
        let lat = Angle::from_y_x(coord.z, xynorm);
        let lon = Angle::from_y_x(coord.y, coord.x);
        Self { lat, lon }
    }
}

impl VertexSource for Ray {

    fn midpoint(x: &Self, y: &Self) -> Self {
        let c1 = x.project(1.0);
        let c2 = y.project(1.0);
        if x < y {
            let mid = (c1 + c2) / 2.0;
            mid.into()
        } else {
            let mid = (c2 + c1) / 2.0;
            mid.into()
        }
    }
}

pub type SharedRayCollection = Rc<RefCell<Vec<(Ray, (usize, usize))>>>;
pub type AtomicRayCollection = Arc<Mutex<Vec<(Ray, (usize, usize))>>>;

const TRIANGLES: [[usize; 3]; 20] = [
    [0, 1, 8],  [0, 1, 9],  [0, 4, 5],  [0, 4, 8], [0, 5, 9],  
    [1, 6, 7],  [1, 6, 8],  [1, 7, 9],  [2, 3, 10], [2, 3, 11],
    [2, 4, 5],  [2, 4, 10], [2, 5, 11],  [3, 7, 11], [3, 6, 7],
    [3, 6, 10], [4, 8, 10], [5, 9, 11], [6, 8, 10], [7, 9, 11],
];

#[derive(Clone)]
pub struct GenericPolyhedron<C>
where C: IndexedVertexSource<Scalar = f64, Vertex = Ray> + IndexedVertexSink<Scalar = f64, Vertex = Ray> + Clone
{
    faces: [FaceTree<C, Ray>; 20],
    vertices: C
}

pub type Polyhedron = GenericPolyhedron<SharedRayCollection>;
pub type AtomicPolyhedron = GenericPolyhedron<AtomicRayCollection>;

impl<C> GenericPolyhedron<C>
where C: IndexedVertexSource<Scalar = f64, Vertex = Ray> + IndexedVertexSink<Scalar = f64, Vertex = Ray> + Clone
{
    pub fn new() -> Self {
        let phi: f64 = (1.0 + 5.0_f64.sqrt()) / 2.0;
        
        let points = [
            Vector3::new(0.0, 1.0, phi),
            Vector3::new(0.0, -1.0, phi),
            Vector3::new(0.0, 1.0, -phi),
            Vector3::new(0.0, -1.0, -phi),

            Vector3::new(1.0, phi, 0.0),
            Vector3::new(-1.0, phi, 0.0),
            Vector3::new(1.0, -phi, 0.0),
            Vector3::new(-1.0, -phi, 0.0),

            Vector3::new(phi, 0.0, 1.0),
            Vector3::new(-phi, 0.0, 1.0),
            Vector3::new(phi, 0.0, -1.0),
            Vector3::new(-phi, 0.0, -1.0),
        ];

        let mut collection = C::new_collection();

        // Map the points to vertices
        points.iter()
            .enumerate()
            .for_each(|(v, p)| {
                let ray: Ray = p.clone().into();
                let index = collection.seed(ray);
                debug_assert_eq!(index, v);
            });
        
        let faces: [FaceTree<C, Ray>; 20] = TRIANGLES.into_iter()
            .map(|[i1, i2, i3]| {
                let triangle = SharedTriangle::new(i1, i2, i3, collection.clone());
                FaceTree::new(triangle)
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap_or_else(|_| panic!("Could not convert to array"));

        Self { faces, vertices: collection }
    }

    pub fn subdivide(&mut self) {
        for face in &mut self.faces {
            face.subdivide();
        }
    }

    pub fn triangles(&self) -> Vec<SharedTriangle<C>> {
        self.faces.iter()
            .flat_map(|f| f.leaves())
            .collect()
    }

    fn find_root_face(&self, vec: &Vector3<f64>) -> &FaceTree<C, Ray> {
        // Find the root face that maximizes the dot product
        let i = self.faces.iter().enumerate()
            .map(|(i, face_tree)| {
                let root_face = face_tree.root();
                let dot = root_face.dot_normal(&vec);
                (i, dot)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        &self.faces[i]
    }

    fn find_leaf(&self, ray: &Ray) -> &FaceBranch<C, Ray> {
        let vec = ray.project(1.0);

        // Load the face tree
        let face_tree = self.find_root_face(&vec);

        // Traverse the tree to find the leaf face
        let mut leaf_index = 0;
        loop {
            let branch = &face_tree[leaf_index];
            match &branch.children {
                Some(children) => {
                    // Find the child that maximizes the dot product
                    leaf_index = children.iter()
                        .map(|&child_index| {
                            let child_face = &face_tree[child_index].face;
                            let dot = child_face.dot_normal(&vec);
                            (child_index, dot)
                        })
                        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                        .map(|(child_index, _)| child_index)
                        .unwrap();
                },
                None => break,
            }
        }

        &face_tree[leaf_index]
    }

    /// Returns the face that the ray intersects
    pub fn find_face(&self, ray: &Ray) -> SharedTriangle<C> {
         self.find_leaf(ray).face.triangle().clone()
    }

    pub fn vertices(&self) -> &C {
        &self.vertices
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::IndexedVertexSource;

    #[test]
    fn test_icosahedron_seed() {
        let mut count: [usize; 12] = [0; 12];
        for triangle in TRIANGLES {
            for vertex in triangle {
                count[vertex] += 1;
            }
        }
        assert!(count.iter().all(|c| *c == 5));
    }

    #[test]
    fn test_icosahedron_distribution() {
        let mut count: [usize; 12] = [0; 12];
        for triangle in TRIANGLES {
            for vertex in triangle {
                count[vertex] += 1;
            }
        }
        assert!(count.iter().all(|c| *c == 5));
    }

    fn angle(divisions: usize) -> f64 {
        63.4 / 2.0_f64.powi(divisions as i32)
    }

        #[test]
    fn test_icosahedron_angle() {
        let poly = Polyhedron::new();
        for face in poly.faces {
            let triangle = face.root().triangle().load();
            let v0 = triangle.vertices()[0].project(1.0);
            let v1 = triangle.vertices()[1].project(1.0);
            let v2 = triangle.vertices()[2].project(1.0);
            let theta0 = v0.dot(&v1).acos().to_degrees();
            let theta1 = v1.dot(&v2).acos().to_degrees();
            let theta2 = v2.dot(&v0).acos().to_degrees();
            assert!([theta0, theta1, theta2].iter().all(|a| *a > 63.0 && *a < 64.0));
        }
    }
    
    const fn vertex_count(d: usize) -> usize {
        10 * 4_usize.pow(d as u32) + 2
    }

    #[test]
    fn test_flat_subdivision() {
        let poly = Polyhedron::new();
        let triangles = poly.triangles();
        assert_eq!(triangles.len(), 20);
        assert_eq!(poly.vertices().borrow().len(), vertex_count(0));

        // Subdivide
        let d1 = triangles.into_iter()
            .flat_map(|mut t| t.subdivide4().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(d1.len(), 80);
        assert_eq!(poly.vertices().borrow().len(), vertex_count(1));

        let d2 = d1.into_iter()
            .flat_map(|mut t| t.subdivide4().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(d2.len(), 320);
        assert_eq!(poly.vertices().borrow().len(), vertex_count(2));
    }

    #[test]
    fn test_subdivision() {
        let mut poly = Polyhedron::new();
    
        for i in 1..5 {
            poly.subdivide();
            let vertices = poly.vertices();
            // assert_eq!(vertices.len(), vertex_count(i));
            let triangles = poly.triangles();
            assert_eq!(triangles.len(), 20 * 4_usize.pow(i as u32));
            // Look for points that are too close to each other
            let min_angle = angle(i);
            for j in 0..vertices.len() {
                let vertex = vertices.get(j).unwrap();
                for k in 0..vertices.len() {
                    if j == k { continue; }
                    let vertex2 = vertices.get(k).unwrap();
                    let angle = Degrees::from(vertex.separation(&vertex2));
                    if angle.0 < 0.95 * min_angle {
                        panic!("Angle too small")
                    }
                }
            }
        }
    }
}