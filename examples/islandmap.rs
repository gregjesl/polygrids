use polygrids::twodimensional::cartesian::{SVG, Base4SquareMap};

fn main() {
    let map = Base4SquareMap::new(1000.0, 5);
    let mut svg: SVG<f64> = SVG::new((1000.0, 1000.0), (-500.0, -500.0, 1000.0, 1000.0));

    let all_triangles = map.load_triangles();
    let (island, water): (Vec<_>, Vec<_>) = all_triangles.into_iter()
        .partition(|v| v.center().norm() < 300.0);

    for triangle in water.iter() {
        svg.add_content(triangle.to_svg("blue".to_string(), 1.0,"blue".to_string()));
    }
    for triangle in island.iter() {
        svg.add_content(triangle.to_svg("green".to_string(), 1.0,"green".to_string()));
    }
    println!("{}", svg);
}