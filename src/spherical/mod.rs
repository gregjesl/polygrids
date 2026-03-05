mod geometry;
mod faces;
use nalgebra::Vector3;
use std::{collections::{BTreeSet, HashSet}, hash::Hash};

use geometry::{Triangle, Ray};
use faces::{FaceTree, FaceBranch};
use crate::grid::Vertex;

#[derive(Clone)]
pub struct Polyhedron {
    faces: [FaceTree<Ray>; 20],
}

impl Polyhedron {
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

        // Map the points to vertices
        let rays: Vec<Ray> = points.into_iter()
            .map(|vec: Vector3<f64>| vec.into())
            .collect();

        const TRIANGLES: [[usize; 3]; 20] = [
            [0, 1, 8],  [0, 1, 9],  [0, 4, 5],  [0, 4, 8], [0, 5, 9],  
            [1, 6, 7],  [1, 6, 8],  [1, 7, 9],  [2, 3, 10], [2, 3, 11],
            [2, 4, 5],  [2, 4, 10], [2, 5, 11],  [3, 7, 11], [3, 6, 7],
            [3, 6, 10], [4, 8, 10], [5, 9, 11], [6, 8, 10], [7, 9, 11],
        ];
        
        let faces: [FaceTree<Ray>; 20] = TRIANGLES.into_iter()
            .map(|[i1, i2, i3]| {
                let triangle = Triangle::new(rays[i1].clone(), rays[i2].clone(), rays[i3].clone());
                FaceTree::new(triangle)
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap_or_else(|_| panic!("Could not convert to array"));

        Self { faces }
    }

    pub fn subdivide(&mut self) {
        for face in &mut self.faces {
            face.subdivide();
        }
    }

    fn find_root_face(&self, vec: &Vector3<f64>) -> &FaceTree<Ray> {
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

    fn find_leaf(&self, ray: &Ray) -> &FaceBranch<Ray> {
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

    pub fn find_face(&self, ray: &Ray) -> &[Ray; 3] {
        let leaf_face = self.find_leaf(ray);
        leaf_face.face.triangle.vertices()
    }

    pub fn vertices(&self) -> HashSet<Ray> {
        let mut set = HashSet::new();
        for face_tree in &self.faces {
            let faceset = face_tree.vertices();
            set.extend(&faceset);
        }
        set
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    
    const fn vertex_count(d: usize) -> usize {
        10 * (d + 1).pow(2) + 2
    }

    #[test]
    fn test_vertices() {
        let mut poly = Polyhedron::new();
    
        for i in 0..5 {
            let vertices = poly.vertices();
            assert_eq!(vertices.len(), vertex_count(i));
            poly.subdivide();
        }
    }
}