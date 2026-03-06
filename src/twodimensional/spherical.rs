use std::{cell::RefCell, rc::Rc, sync::{Arc, Mutex}, ops::Index};
use nalgebra::Vector3;
use super::{Vertex, VertexSource, Triangle, SharedTriangle, IndexedVertexSource, IndexedVertexSink};
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


#[derive(Clone, Debug)]
pub(crate) struct Face<C, V>
where C: IndexedVertexSource<Scalar = f64, Vertex = V> + Clone,
      V: Vertex<Scalar = f64> + Clone
{
    triangle: SharedTriangle<C>,
    normal: Vector3<f64>,
}

impl<C, V> Face<C, V>
where C: IndexedVertexSource<Scalar = f64, Vertex = V> + Clone,
      V: Vertex<Scalar = f64> + Clone
{
    pub fn dot_normal(&self, vector: &Vector3<f64>) -> f64 {
        self.normal.dot(vector)
    }
}

impl<C, V> Face<C, V>
where C: IndexedVertexSource<Scalar = f64, Vertex = V> + IndexedVertexSink<Scalar = f64, Vertex = V> + Clone,
      V: Vertex<Scalar = f64> + VertexSource + PartialOrd + Clone
{
    #[allow(dead_code)]
    pub fn new(triangle: Triangle<V, f64>, mut collection: C) -> Self {
        let normal = triangle.center();
        let v0 = collection.seed(triangle[0].clone());
        let v1 = collection.seed(triangle[1].clone());
        let v2 = collection.seed(triangle[2].clone());
        let shared = SharedTriangle::new(v0, v1, v2, collection);
        Self { triangle: shared, normal }
    }

    pub fn triangle(&self) -> &SharedTriangle<C> {
        &self.triangle
    }

    pub fn subdivide4(&mut self) -> Option<[Face<C, V>; 4]> {
        let sub_triangles = self.triangle.subdivide4()?;
        
        // Triangle 0
        let centers: Vec<_> = sub_triangles.iter()
            .map(|t| t.load().center())
            .collect();

        let result = [
            Self { triangle: sub_triangles[0].clone(), normal: centers[0] },
            Self { triangle: sub_triangles[1].clone(), normal: centers[1] },
            Self { triangle: sub_triangles[2].clone(), normal: centers[2] },
            Self { triangle: sub_triangles[3].clone(), normal: centers[3] },
        ];
        Some(result)
    }
}

impl<C, V> From<SharedTriangle<C>> for Face<C, V>
where C: IndexedVertexSource<Scalar = f64, Vertex = V> + Clone,
      V: Vertex<Scalar = f64> + PartialOrd + Clone
{
    fn from(triangle: SharedTriangle<C>) -> Self {
        let normal = triangle.load().center();
        Self { triangle, normal }
    }
}

/// A face that may have children
#[derive(Clone)]
pub(crate) struct FaceBranch<C, V>
where C: IndexedVertexSource<Scalar = f64, Vertex = V> + Clone,
      V: Vertex<Scalar = f64> + Clone
{
    pub face: Face<C, V>,
    pub children: Option<[usize; 4]>,
}

/// A root face that may have children
#[derive(Clone)]
pub(crate) struct FaceTree<C, V>
where C: IndexedVertexSource<Scalar = f64, Vertex = V> + Clone,
      V: Vertex<Scalar = f64> + Clone
{
    faces: Vec<FaceBranch<C, V>>,
}

impl<C, V> FaceTree<C, V>
where C: IndexedVertexSource<Scalar = f64, Vertex = V> + IndexedVertexSink<Scalar = f64, Vertex = V> + Clone,
      V: Vertex<Scalar = f64> + VertexSource + PartialOrd + Clone
{
    pub fn new(triangle: SharedTriangle<C>) -> Self
    {
        let root_face: Face<C, V> = triangle.into();
        let root_branch = FaceBranch {
            face: root_face,
            children: None,
        };
        Self {
            faces: vec![root_branch],
        }
    }

    pub fn root(&self) -> &Face<C, V> {
        &self.faces[0].face
    }

    fn insert(&mut self, face: Face<C, V>) -> usize {
        let index = self.faces.len();
        let branch = FaceBranch {
            face,
            children: None,
        };
        self.faces.push(branch);
        index
    }

    fn leaf_start(&self) -> usize {
        self.faces
            .iter()
            .enumerate()
            .find(|(_, f)| f.children.is_none())
            .map(|(i, _)| i)
            .unwrap()
    }

    pub fn leaves(&self) -> Vec<SharedTriangle<C>> {
        self.faces.iter()
            .filter(|f| f.children.is_none())
            .map(|f| f.face.triangle.clone())
            .collect()
    }

    pub fn subdivide(&mut self)
    {
        for leaf in self.leaf_start()..self.faces.len() {
            let face = &mut self.faces[leaf].face;
            let sub_faces = face.subdivide4().unwrap();
            let mut child_indices = [0; 4];
            for (i, sub_face) in sub_faces.into_iter().enumerate() {
                let child_index = self.insert(sub_face);
                child_indices[i] = child_index;
            }
            self.faces[leaf].children = Some(child_indices);
        }
    }
}

impl<C, V> Index<usize> for FaceTree<C, V>
where C: IndexedVertexSource<Scalar = f64, Vertex = V> + Clone,
      V: Vertex<Scalar = f64> + Clone
{
    type Output = FaceBranch<C, V>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.faces[index]
    }
}

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

    #[test]
    fn test_vertices() {
        let triangle: Triangle<Ray, f64> = Triangle::new(
            Vector3::new(1.0, 0.0, 0.0).into(),
            Vector3::new(0.0, 1.0, 0.0).into(),
            Vector3::new(0.0, 0.0, 1.0).into()
        );
        
        let mut collection = SharedRayCollection::new_collection();
        let v0 = collection.seed(triangle[0]);
        let v1 = collection.seed(triangle[1]);
        let v2 = collection.seed(triangle[2]);
        let shared = SharedTriangle::new(v0, v1, v2, collection.clone());
        let echo = shared.load();
        for i in 0..3 {
            assert_eq!(triangle[i], echo[i]);
        }
    
        // Formula = (2^n+1)(2^n+2)/2
        let mut tree = FaceTree::new(shared);
        assert_eq!(collection.len(), 3);
        tree.subdivide();
        assert_eq!(collection.len(), 6);
        tree.subdivide();
        assert_eq!(collection.len(), 15);
        tree.subdivide();
        assert_eq!(collection.len(), 45);
        tree.subdivide();
        assert_eq!(collection.len(), 153);
    }

    #[test]
    fn test_square() {
        let middle: Ray = Vector3::new(1.0, 0.0, 0.0).into();
        let east: Ray = Vector3::new(0.0, 1.0, 0.0).into();
        let west: Ray = Vector3::new(0.0, -1.0, 0.0).into();
        let north: Ray = Vector3::new(0.0, 0.0, 1.0).into();

        let mut collection = SharedRayCollection::new_collection();
        collection.seed(middle);
        collection.seed(east);
        collection.seed(west);
        collection.seed(north);

        let t1 = SharedTriangle::new(0, 1, 3, collection.clone());
        let t2 = SharedTriangle::new(0, 2, 3, collection.clone());
        let mut f1 = FaceTree::new(t1);
        let mut f2 = FaceTree::new(t2);
        assert_eq!(collection.len(), 4);

        f1.subdivide();
        f2.subdivide();
        assert_eq!(collection.len(), 9);

        f1.subdivide();
        f2.subdivide();
        assert_eq!(collection.len(), 25);
    }

    #[test]
    fn test_leaves() {
        let triangle: Triangle<Ray, f64> = Triangle::new(
            Vector3::new(1.0, 0.0, 0.0).into(),
            Vector3::new(0.0, 1.0, 0.0).into(),
            Vector3::new(0.0, 0.0, 1.0).into()
        );
        
        let mut collection = SharedRayCollection::new_collection();
        let v0 = collection.seed(triangle[0]);
        let v1 = collection.seed(triangle[1]);
        let v2 = collection.seed(triangle[2]);
        let shared = SharedTriangle::new(v0, v1, v2, collection.clone());
        let echo = shared.load();
        for i in 0..3 {
            assert_eq!(triangle[i], echo[i]);
        }
    
        // Formula = (2^n+1)(2^n+2)/2
        let mut tree = FaceTree::new(shared);
        assert_eq!(tree.leaves().len(), 1);
        for i in 1..6 {
            tree.subdivide();
            assert_eq!(tree.leaves().len(), 4_usize.pow(i));
        }
    }

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
            assert_eq!(vertices.len(), vertex_count(i));
            let triangles = poly.triangles();
            assert_eq!(triangles.len(), 20 * 4_usize.pow(i as u32));
        }
    }

    #[test]
    fn test_find() {
        let mut poly = Polyhedron::new();
        poly.subdivide();
        poly.subdivide();

        let ray = Ray::new(Angle::from(Degrees(30.0)), Angle::from(Degrees(45.0)));
        let face = poly.find_face(&ray);
        let triangle = face.load();
        let rays = triangle.vertices();
        let lower = rays
            .map(|r| Degrees::from(*r.latitude()).0)
            .into_iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let upper = rays
            .map(|r| Degrees::from(*r.latitude()).0)
            .into_iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        assert!(upper > lower);
        assert!(Degrees::from(*ray.latitude()).0 >= lower);
        assert!(Degrees::from(*ray.latitude()).0 <= upper);

        let left = rays
            .map(|r| Degrees::from(*r.longitude()).0)
            .into_iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        let right = rays
            .map(|r| Degrees::from(*r.longitude()).0)
            .into_iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        assert!(right > left);
        assert!(Degrees::from(*ray.longitude()).0 >= left);
        assert!(Degrees::from(*ray.longitude()).0 <= right);
    }
}