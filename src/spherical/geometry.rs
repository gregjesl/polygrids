use nalgebra::Vector3;
use angle_sc::{Angle, Degrees};
use std::{ops::{Index, IndexMut}, hash::Hash};
use crate::grid::{Vertex, VertexSource};

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

impl Eq for Ray {}

impl Ord for Ray
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.lat.partial_cmp(&other.lat) {
            Some(std::cmp::Ordering::Equal) => self.lon.partial_cmp(&other.lon).unwrap(),
            ord => ord.unwrap(),
        }
    }
}

impl Hash for Ray
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.lat.cos().0.to_bits().hash(state);
        self.lon.cos().0.to_bits().hash(state);
    }
}

impl Vertex<f64> for Ray {
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

impl VertexSource<f64> for Ray {

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

#[derive(Clone, PartialEq, PartialOrd)]
pub struct Triangle<G, F>
where G: Vertex<F>,
      F: nalgebra::RealField
{
    vertices: [G; 3],
    _marker: std::marker::PhantomData<F>,
}

impl<G, F> Triangle<G, F>
where G: Vertex<F> + VertexSource<F> + PartialOrd + Clone,
      F: nalgebra::RealField
{
    pub fn new(v0: G, v1: G, v2: G) -> Self {
        let mut vertices = [v0, v1, v2];
        vertices.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Triangle {
            vertices,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn center(&self) -> Vector3<F> {
        let c1 = self.vertices[0].project(F::one());
        let c2 = self.vertices[1].project(F::one());
        let c3 = self.vertices[2].project(F::one());
        let center = (c1 + c2 + c3) / (F::one() + F::one() + F::one());
        center.into()
    }

    pub fn subdivide(&self) -> [Triangle<G, F>; 4] 
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

    pub fn vertices(&self) -> &[G; 3] {
        &self.vertices
    }
}

impl<G, F> Index<usize> for Triangle<G, F>
where G: Vertex<F>,
      F: nalgebra::RealField
{
    type Output = G;

    fn index(&self, index: usize) -> &Self::Output {
        &self.vertices[index]
    }
}

impl<G, F> IndexMut<usize> for Triangle<G, F>
where G: Vertex<F>,
      F: nalgebra::RealField
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.vertices[index]
    }
}

impl<G, F> IntoIterator for Triangle<G, F>
where G: Vertex<F> + Clone,
      F: nalgebra::RealField
{
        type Item = G;
        type IntoIter = std::vec::IntoIter<G>;

    fn into_iter(self) -> std::vec::IntoIter<G> {
        self.vertices.to_vec().into_iter()
    }
}