use nalgebra::Vector3;
use clap::Parser;
use polygrids::{
    Axis64, threedimensional::{SphericalField}, twodimensional::{Vertex, spherical::Ray}
};

/// Simple program to greet a person
#[derive(Parser, Debug)]
struct Args {
    /// Gravitational constant in m^3/s^2
    #[arg(long, default_value_t = 3.986004418e14)]
    mu: f64,

    /// Reference radius in meters
    #[arg(long, default_value_t = 6371e3)]
    radius: f64,

    /// J2 coefficient
    #[arg(long, default_value_t = 1.08263e-3)]
    j2: f64,

    /// Minimum altitude in meters
    #[arg(long, default_value_t = 100e3)]
    min: f64,

    /// Maximum altitude in meters
    #[arg(long, default_value_t = 300e3)]
    max: f64,

    /// Radial divisions
    #[arg(long, default_value_t = 101)]
    radial: usize,

    /// Subdivisions of the icosahedron
    #[arg(long, default_value_t = 5)]
    subdivisions: usize,

    /// Number of samples to check
    #[arg(long, default_value_t = 1000)]
    samples: usize,
}

fn j2(settings: &Args, point: Vector3<f64>) -> Vector3<f64> {
    let r = point.norm();
    let z = point.z;

    let point_accel = -settings.mu * point / r.powi(3);

    let factor = -settings.mu * settings.radius.powi(2) * settings.j2 / r.powi(5);
    let j2_accel = Vector3::new(
        7.5 * (z / r).powi(2) - 1.5 * point.x / r,
        7.5 * (z / r).powi(2) - 1.5 * point.y / r,
        7.5 * (z / r).powi(2) - 4.5 * z / r
    );

    return factor * j2_accel;
}

fn main() {
    let settings = Args::parse();
    assert!(settings.min < settings.max, "Minimum altitude must be less than maximum altitude");
    assert!(settings.radial > 1, "Radial divisions must be greater than 1");
    assert!(settings.min > 0.0, "Minimum altitude must be greater than 0");

    let axis = Axis64::linear(settings.min + settings.radius, settings.max + settings.radius, settings.radial).unwrap();

    let field = SphericalField::new(
        settings.subdivisions, axis.clone(), |ray, r| {
            j2(&settings, ray.project(r))
    });

    let mut results = Vec::with_capacity(settings.samples);
    let mut rng = rand::rng();
    for _ in 0..settings.samples {
        let ray = Ray::random(&mut rng);
        let radius = axis.random_sample(&mut rng);
        let accel = j2(&settings, ray.project(radius));
        let interp = field.interpolate(&ray, radius).unwrap();
        results.push((accel, interp));
    }
    let mean = results.iter().map(|(a, i)| (a - i).norm()).sum::<f64>() / settings.samples as f64;
    println!("Mean error: {:e} m/s^2", mean);
    let percent_error = results.iter().map(|(a, i)| (a - i).norm() / a.norm()).sum::<f64>() / settings.samples as f64;
    println!("Mean percent error: {:e} %", percent_error * 100.0);
}