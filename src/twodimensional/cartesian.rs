use crate::twodimensional::{IndexedVertexSink, Triangle, SharedTriangle};

use super::{Vertex, VertexSource};
use nalgebra::{RealField, Vector2, Vector3};
use std::{cell::RefCell, fmt::Display, rc::Rc, sync::{Arc, Mutex}};

pub type CartesianVertex32 = Vector2<f32>;
pub type CartesianVertex64 = Vector2<f64>;

impl<T> Vertex for Vector2<T> 
where T: RealField + Clone
{
    type Scalar = T;

    fn project(&self, distance: Self::Scalar) -> Vector3<Self::Scalar> {
        Vector3::new(self.x.clone(), self.y.clone(), distance)
    }
}

impl<T> VertexSource for Vector2<T>
where T: RealField + Clone
{
    fn midpoint(x: &Self, y: &Self) -> Self {
        if x < y {
            (x + y) / (T::one() + T::one())
        } else {
            (y + x) / (T::one() + T::one())
        }
    }
}

/// (I)ndexed (C)artesian (V)ertex (C)ollection with an f32 base
pub type ICVC32 = Rc<RefCell<Vec<(CartesianVertex32, (usize, usize))>>>;

/// (I)ndexed (C)artesian (V)ertex (C)ollection with an f64 base
pub type ICVC64 = Rc<RefCell<Vec<(CartesianVertex64, (usize, usize))>>>;

/// (A)tomic (I)ndexed (C)artesian (V)ertex (C)ollection with an f32 base
pub type AICVC32 = Arc<Mutex<Vec<(CartesianVertex32, (usize, usize))>>>;

/// (A)tomic (I)ndexed (C)artesian (V)ertex (C)ollection with an f64 base
pub type AICVC64 = Arc<Mutex<Vec<(CartesianVertex64, (usize, usize))>>>;

pub struct SVG<T> 
where T: Display + Copy
{
    dimensions: (T, T),
    viewbox: (T, T, T, T),
    content: Vec<String>
}

impl<T> SVG<T> 
where T: Display + Copy
{
    pub fn new(dimensions: (T, T), viewbox: (T, T, T, T)) -> Self {
        SVG {
            dimensions,
            viewbox,
            content: Vec::new()
        }
    }

    pub fn add_content(&mut self, content: String) {
        self.content.push(content);
    }
}

impl<T> std::fmt::Display for SVG<T> 
where T: Display + Copy
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let content = self.content.join("\n");
        write!(f, "<svg width=\"{}\" height=\"{}\" viewBox=\"{} {} {} {}\" xmlns=\"http://www.w3.org/2000/svg\">\n{}\n</svg>", 
            self.dimensions.0, 
            self.dimensions.1, 
            self.viewbox.0, 
            self.viewbox.1, 
            self.viewbox.2, 
            self.viewbox.3, 
            content
        )
    }
}

impl<T> Triangle<Vector2<T>, T>
where T: RealField + Clone
{
    pub fn to_svg(&self, stroke: String, stroke_width: T, fill: String) -> String {
        let points = self.vertices.iter()
            .map(|v| format!("{},{}", v.x, v.y))
            .collect::<Vec<String>>()
            .join(" ");
        format!("<polygon points=\"{}\" stroke=\"{}\" stroke-width=\"{}\" fill=\"{}\" />", points, stroke, stroke_width, fill)
    }
}

/// Square map that starts with 4 triangles and 5 vertices
pub struct Base4SquareMap<T>
where T: RealField + Clone
{
    width: T,
    triangles: Vec<SharedTriangle<Rc<RefCell<Vec<(Vector2<T>, (usize, usize))>>>>>,
    vertices: Rc<RefCell<Vec<(Vector2<T>, (usize, usize))>>>
}

impl<T> Base4SquareMap<T>
where T: RealField + Copy
{
    pub fn new(width: T, subdivisions: usize) -> Self {
        let two = T::one() + T::one();
        let mut vertices = Rc::new(RefCell::new(Vec::new()));
        let halfwidth = width / two;
        
        let lowerleft = vertices.seed(Vector2::<T>::new(-halfwidth, -halfwidth));
        let lowerright = vertices.seed(Vector2::<T>::new(halfwidth, -halfwidth));
        let upperright = vertices.seed(Vector2::<T>::new(halfwidth, halfwidth));
        let upperleft = vertices.seed(Vector2::<T>::new(-halfwidth, halfwidth));
        let middle = vertices.seed(Vector2::<T>::new(T::zero(), T::zero()));

        let triangles = vec![
            SharedTriangle::new(lowerleft, lowerright, middle, vertices.clone()),
            SharedTriangle::new(lowerright, upperright, middle, vertices.clone()),
            SharedTriangle::new(upperright, upperleft, middle, vertices.clone()),
            SharedTriangle::new(upperleft, lowerleft, middle, vertices.clone()),
        ];

        let mut result = Base4SquareMap { width: width, triangles, vertices };
        for _ in 0..subdivisions {
            result.subdivide();
        }
        result
    }

    pub fn width(&self) -> T {
        self.width
    }

    pub fn vertices(&self) -> &Rc<RefCell<Vec<(Vector2<T>, (usize, usize))>>> {
        &self.vertices
    }

    pub fn subdivide(&mut self) {
        let mut new_triangles = Vec::with_capacity(self.triangles.len() * 4);
        for triangle in self.triangles.iter_mut() {
            let children = triangle.subdivide4().unwrap();
            children.into_iter().for_each(|child| new_triangles.push(child));
        }
        self.triangles = new_triangles;
    }

    pub fn load_triangles(&self) -> Vec<Triangle<Vector2<T>, T>> {
        self.triangles.iter().map(|t| t.load()).collect()
    }
}
