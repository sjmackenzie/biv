//! Metal prefill attention (`attn_flash`) vs the split-K kernel: logits after
//! prefilling the same tokens in full batches and in passes of a few tokens,
//! then a few decode steps. `attention <gguf> [tokens]`
#[cfg(target_os = "macos")]
mod imp {
    use std::time::Instant;

    use biv::{
        engine::Engine, metal::MetalEngine, model::Model, sampler::argmax, tokenizer::Tokenizer,
    };

    fn report(what: &str, a: &[f32], b: &[f32]) {
        let maxdiff = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        let (ta, tb) = (argmax(a), argmax(b));
        println!(
            "{what}: max |diff| {maxdiff:.4}, top-1 {}",
            if ta == tb { "same" } else { "DIFFERENT" }
        );
    }

    pub fn main() {
        let path = std::env::args()
            .nth(1)
            .expect("usage: attention <gguf> [tokens]");
        let n: usize = std::env::args()
            .nth(2)
            .map(|s| s.parse().unwrap())
            .unwrap_or(1000);
        let g = biv_cli::open_gguf(&path).unwrap();
        let tok = Tokenizer::from_gguf(&g).unwrap();
        let model = Model::load(&g).unwrap();
        let metal = MetalEngine::new(&model, 256).unwrap();

        let text = "The history of computing is a story of abstraction layered upon abstraction: switches became gates, gates became circuits, circuits became processors, and processors became the substrate on which languages, operating systems and networks were built. ";
        let ids: Vec<_> = tok.encode(text).into_iter().cycle().take(n).collect();
        let mut flash = metal.new_session(n + 16, 0).unwrap();
        let mut split = metal.new_session(n + 16, 0).unwrap();

        let t = Instant::now();
        let lf = flash.forward(&ids);
        let tf = t.elapsed();
        // passes of 8 tokens use the split-K kernel
        let t = Instant::now();
        for c in ids[..n - 1].chunks(8) {
            split.prefill(c);
        }
        let ls = split.forward(&ids[n - 1..]);
        let ts = t.elapsed();
        println!("prefill {n} tokens: batches {tf:?}, passes of 8 {ts:?}");
        report("prefill", &ls, &lf);

        let mut next = argmax(&ls);
        for step in 0..4 {
            let ls = split.forward(&[next]);
            let lf = flash.forward(&[next]);
            report(&format!("decode {step}"), &ls, &lf);
            next = argmax(&ls);
        }
    }
}

fn main() {
    #[cfg(target_os = "macos")]
    imp::main();
    #[cfg(not(target_os = "macos"))]
    eprintln!("attention needs the Metal backend (macOS)");
}
