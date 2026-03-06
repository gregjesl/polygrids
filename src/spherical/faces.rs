use crate::{geometry::{SharedTriangle, Triangle}, grid::{IndexedVertexSink, IndexedVertexSource, Vertex, VertexSource}};
use nalgebra::Vector3;

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
struct FaceBranch<C, V>
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

    pub fn leaves(&self) -> Vec<Triangle<V, f64>> {
        self.faces.iter()
            .filter(|f| f.children.is_none())
            .map(|t| t.face.triangle.load())
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{spherical::{Ray, SharedRayCollection}, grid::{IndexedVertexSink, IndexedVertexSource}};
    
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
}
