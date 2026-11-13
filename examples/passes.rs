//! The time of one batched pass for each mix of sequences and tokens.
//! `passes <gguf>`
//!
//! Each sequence holds the same prompt. A pass is a verify of `tokens`
//! tokens for each of `seqs` sequences, which gives logits for every token,
//! as a decode step with drafts does. Each case runs one pass to warm, then
//! `ITERS` timed passes, and prints the mean time of a pass and its ratio to
//! a pass of one token of one sequence.
use std::time::Instant;

use biv::{
    engine::{self, Engine, Pass, Work},
    model::Model,
    tokenizer::Tokenizer,
};

/// The prompt of each sequence
const PROMPT: &str =
    "Explain in detail how a CPU cache hierarchy works and why it matters for performance.";
/// The tokens of a pass and the rows of logits of a pass, at most
const PASS_TOKENS: usize = 256;
const PASS_LOGITS: usize = 64;
/// The context of each sequence, far above the tokens of the timed passes
const CONTEXT: usize = 4096;
/// The timed passes of each case, after one pass to warm
const ITERS: usize = 8;
/// Each case: the sequences of a pass and the tokens of each sequence. The
/// first case is the reference: one token of one sequence.
const CASES: [(usize, usize); 20] = [
    (1, 1),
    (1, 2),
    (1, 3),
    (1, 4),
    (1, 6),
    (1, 8),
    (1, 12),
    (1, 16),
    (1, 17),
    (1, 24),
    (1, 32),
    (1, 48),
    (2, 1),
    (4, 1),
    (8, 1),
    (16, 1),
    (2, 6),
    (3, 6),
    (4, 6),
    (8, 6),
];

fn main() {
    let path = std::env::args().nth(1).expect("usage: passes <gguf>");
    let g = biv_cli::open_gguf(&path).unwrap();
    let tok = Tokenizer::from_gguf(&g).unwrap();
    let model = Model::load(&g).unwrap();
    #[cfg(target_os = "macos")]
    let metal = biv::metal::MetalEngine::new(&model, PASS_TOKENS).unwrap();
    #[cfg(target_os = "macos")]
    let engine: &dyn Engine = &metal;
    #[cfg(not(target_os = "macos"))]
    let engine: &dyn Engine = &model;
    println!("{}", engine.description());

    let prompt = tok.encode(PROMPT);
    let mut batch = engine::new_batch(engine, PASS_TOKENS, PASS_LOGITS).unwrap();
    let mut reference = None;
    for (seqs, tokens) in CASES {
        let ids: Vec<usize> = (0..seqs)
            // a sequence verifies its drafts and one token more
            .map(|_| batch.add(CONTEXT, tokens - 1).unwrap())
            .collect();
        for &seq in &ids {
            batch.run(&[Work {
                seq,
                tokens: &prompt,
                pass: Pass::Prefill,
            }]);
        }
        let step: Vec<_> = prompt.iter().copied().cycle().take(tokens).collect();
        let work: Vec<Work<'_>> = ids
            .iter()
            .map(|&seq| Work {
                seq,
                tokens: &step,
                pass: Pass::Verify,
            })
            .collect();
        let mut pass = || {
            batch.run(&work);
            for &seq in &ids {
                batch.keep(seq, tokens);
            }
        };
        pass();
        let start = Instant::now();
        for _ in 0..ITERS {
            pass();
        }
        let ms = start.elapsed().as_secs_f64() * 1e3 / ITERS as f64;
        let base = *reference.get_or_insert(ms);
        println!(
            "seqs={seqs:2} tokens={tokens:2} rows={:3}: {ms:7.2} ms a pass, {:5.2}x one token",
            seqs * tokens,
            ms / base
        );
        for seq in ids {
            batch.remove(seq);
        }
    }
}
