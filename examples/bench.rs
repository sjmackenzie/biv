//! Matmul throughput per weight type
use biv::ops::{Input, Weight, matmul};
use std::time::Instant;

fn main() {
    let path = std::env::args().nth(1).unwrap();
    let g = biv_cli::open_gguf(&path).unwrap();
    for (name, ns) in [
        ("blk.0.ffn_up.weight", &[1usize, 16, 64][..]),
        ("blk.0.ffn_down.weight", &[1, 64]),
        ("blk.0.ssm_out.weight", &[1, 64]),
        ("output.weight", &[1]),
        ("blk.0.ssm_alpha.weight", &[1]),
    ] {
        let (info, data) = g.tensor(name).unwrap();
        let w = Weight::new(info.ty, info.dims[0] as usize, info.dims[1] as usize, data);
        for &n in ns {
            let x: Vec<f32> = (0..n * w.cols)
                .map(|i| ((i * 31) % 17) as f32 / 17.0 - 0.5)
                .collect();
            let mut out = vec![0.0; n * w.rows];
            matmul(&w, &Input::new(&x, n), &mut out); // warm (page in)
            let iters = if n == 1 { 20 } else { 3 };
            let t = Instant::now();
            for _ in 0..iters {
                matmul(&w, &Input::new(&x, n), &mut out);
            }
            let dt = t.elapsed().as_secs_f64() / iters as f64;
            println!(
                "{name:26} {:?} n={n:3}: {:7.2} ms  {:6.1} GB/s  {:6.1} GOP/s",
                info.ty,
                dt * 1e3,
                data.len() as f64 / dt / 1e9,
                2.0 * (n * w.cols * w.rows) as f64 / dt / 1e9
            );
        }
    }
}
