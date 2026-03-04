use super::geometry::{Ray, Triangle};
use crate::{grid::{Vertex, VertexSource}};
use nalgebra::Vector3;

#[derive(Clone)]
pub struct Face<G>
where G: Vertex<f64>
{
    triangle: Triangle<G, f64>,
    normal: Vector3<f64>,
}

impl<G> Face<G>
where G: Vertex<f64>
{
    pub fn dot_normal(&self, vector: &Vector3<f64>) -> f64 {
        self.normal.dot(vector)
    }
}

impl<G> IntoIterator for Face<G>
where G: Vertex<f64> + Clone
{
    type Item = G;
    type IntoIter = std::vec::IntoIter<G>;

    fn into_iter(self) -> std::vec::IntoIter<G> {
        self.triangle.into_iter()
    }
}

impl<G> Face<G>
where G: Vertex<f64> + VertexSource<f64> + Clone + PartialOrd
{
    pub fn new(triangle: Triangle<G, f64>) -> Self {
        let normal = triangle.center().project(1.0).normalize();
        Face {
            triangle,
            normal,
        }
    }

    pub fn center(&self) -> G {
        self.triangle.center()
    }

    pub fn subdivide(&self) -> [Face<G>; 4] 
    {
        self.triangle.subdivide().map(Face::new)
    }
}

/// A face that may have children
#[derive(Clone)]
struct FaceBranch<G>
where G: Vertex<f64>
{
    pub face: Face<G>,
    pub children: Option<[usize; 4]>,
}

/// A root face that may have children
#[derive(Clone)]
struct FaceTree<G>
where G: Vertex<f64>
{
    faces: Vec<FaceBranch<G>>,
}

impl FaceTree<Ray>
{
    pub fn new(triangle: Triangle<Ray, f64>) -> Self {
        let root_face = Face::new(triangle);
        let root_branch = FaceBranch {
            face: root_face,
            children: None,
        };
        Self {
            faces: vec![root_branch],
        }
    }

    fn insert(&mut self, face: Face<Ray>) -> usize {
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

    pub fn subdivide(&mut self)
    {
        for leaf in self.leaf_start()..self.faces.len() {
            let face = &self.faces[leaf].face;
            let sub_faces = face.subdivide();
            let mut child_indices = [0; 4];
            for (i, sub_face) in sub_faces.into_iter().enumerate() {
                let child_index = self.insert(sub_face);
                child_indices[i] = child_index;
            }
            self.faces[leaf].children = Some(child_indices);
        }
    }
}