//! Metal vs CPU logits on the same sequence: prefill (matmul kernels), then decode steps (matvec kernels)
#[cfg(target_os = "macos")]
mod imp {
    use std::time::Instant;

    use biv::{
        engine::Engine, metal::MetalEngine, model::Model, sampler::argmax, tokenizer::Tokenizer,
    };

    fn top(l: &[f32], k: usize) -> Vec<(usize, f32)> {
        let mut v: Vec<_> = l.iter().copied().enumerate().collect();
        v.sort_by(|a, b| b.1.total_cmp(&a.1));
        v.truncate(k);
        v
    }

    fn report(what: &str, a: &[f32], b: &[f32]) {
        let maxdiff = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        let (ta, tb) = (top(a, 5), top(b, 5));
        let same = ta.iter().map(|x| x.0).eq(tb.iter().map(|x| x.0));
        println!(
            "{what}: max |diff| {maxdiff:.4}, top-5 {}",
            if same { "identical" } else { "DIFFERENT" }
        );
        if !same {
            println!("  cpu   {ta:?}\n  metal {tb:?}");
        }
    }

    pub fn main() {
        let path = std::env::args().nth(1).unwrap();
        let g = biv_cli::open_gguf(&path).unwrap();
        let tok = Tokenizer::from_gguf(&g).unwrap();
        let model = Model::load(&g).unwrap();
        let t = Instant::now();
        let metal = MetalEngine::new(&model, 256).unwrap();
        println!(
            "metal engine ready in {:?}: {}",
            t.elapsed(),
            metal.description()
        );

        let repeat: usize = std::env::args()
            .nth(2)
            .map(|s| s.parse().unwrap())
            .unwrap_or(1);
        let ids = tok.encode(&"Rust is a multi-paradigm, general-purpose programming language that emphasizes performance, type safety, and concurrency. It enforces memory safety, meaning that all references point to valid memory, without a garbage collector. Software developer Graydon Hoare created Rust. ".repeat(repeat));
        let ids = &ids[..ids.len() - 1]; // end mid-sentence: " Rust"
        let ids = ids.to_vec();
        let mut cpu = (&model).new_session(8192, 0).unwrap();
        let mut gpu = metal.new_session(8192, 0).unwrap();

        let t = Instant::now();
        let lc = cpu.forward(&ids);
        let tc = t.elapsed();
        let t = Instant::now();
        let lg = gpu.forward(&ids);
        let tg = t.elapsed();
        println!("prefill {} tokens: cpu {tc:?}, metal {tg:?}", ids.len());
        report("prefill", &lc, &lg);

        // decode a few greedy tokens with both, feeding the cpu choice to both
        let mut next = argmax(&lc);
        let mut text = vec![next];
        for step in 0..8 {
            let lc = cpu.forward(&[next]);
            let lg = gpu.forward(&[next]);
            report(&format!("decode {step}"), &lc, &lg);
            next = argmax(&lc);
            text.push(next);
        }
        println!("cpu text: {:?}", tok.decode(&text));

        // metal decode speed
        let n = 32;
        let t = Instant::now();
        for _ in 0..n {
            let l = gpu.forward(&[next]);
            next = argmax(&l);
        }
        println!(
            "metal decode: {:.1} tok/s",
            n as f64 / t.elapsed().as_secs_f64()
        );
    }
}

fn main() {
    #[cfg(target_os = "macos")]
    imp::main();
    #[cfg(not(target_os = "macos"))]
    eprintln!("compare needs the Metal backend (macOS)");
}
