use std::{collections::HashSet, ops::Index};

use super::geometry::{Ray, Triangle};
use crate::{grid::{Vertex, VertexSource}};
use nalgebra::Vector3;

#[derive(Clone)]
pub struct Face<G>
where G: Vertex<f64>
{
    pub(crate) triangle: Triangle<G, f64>,
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
        let normal = triangle.center().normalize();
        Face {
            triangle,
            normal,
        }
    }

    pub fn center(&self) -> Vector3<f64> {
        self.triangle.center()
    }

    pub fn subdivide(&self) -> [Face<G>; 4] 
    {
        self.triangle.subdivide().map(Face::new)
    }
}

/// A face that may have children
#[derive(Clone)]
pub(crate) struct FaceBranch<G>
where G: Vertex<f64>
{
    pub face: Face<G>,
    pub children: Option<[usize; 4]>,
}

/// A root face that may have children
#[derive(Clone)]
pub(crate) struct FaceTree<G>
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

    pub fn root(&self) -> &Face<Ray> {
        &self.faces[0].face
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

    // Returns a list of all unique vertices in the tree
    pub fn vertices(&self) -> HashSet<Ray> {
        let mut set = HashSet::new();
        for face in &self.faces {
            for vertex in face.face.clone().into_iter() {
                set.insert(vertex);
            }
        }
        set
    }
}

impl Index<usize> for FaceTree<Ray>
{
    type Output = FaceBranch<Ray>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.faces[index]
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
    
        // Formula = (2^n+1)(2^n+2)/2
        let mut tree = FaceTree::new(triangle);
        assert_eq!(tree.vertices().len(), 3);
        tree.subdivide();
        assert_eq!(tree.vertices().len(), 6);
        tree.subdivide();
        assert_eq!(tree.vertices().len(), 15);
        tree.subdivide();
        assert_eq!(tree.vertices().len(), 45);
        tree.subdivide();
        assert_eq!(tree.vertices().len(), 153);
    }
}
