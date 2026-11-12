//! `biv bench`: decode rates with 0 to 6 MTP drafts per step, greedy and
//! sampled, and with chats sharing passes through [`biv::batch`].
//!
//! Every answer is greedy unless a row says otherwise, so the text with drafts
//! must be the text without them: a difference is a bug, and fails the run.
//! Batched answers can part from those of the chat alone by rounding (the
//! matmul kernels depend on the rows of the whole pass), so they are counted.

use std::sync::Mutex;
use std::time::{Duration, Instant};

use biv::Error;
use biv::batch::{self, BatchOptions};
use biv::chat::{Chat, ChatOptions, Event};
use biv::engine::Engine;
use biv::sampler::{Sampler, SamplerConfig};
use biv::tokenizer::Tokenizer;

const PROMPTS: [&str; 8] = [
    "Explain in detail how a CPU cache hierarchy works and why it matters for performance.",
    "Write a Rust function that parses an IPv4 address from a string, with tests.",
    "Summarize the causes of the French Revolution in a few paragraphs.",
    "What are the trade-offs between B-trees and LSM trees for a storage engine?",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "How does public key cryptography work? Explain it to a curious teenager.",
    "Give a recipe for a vegetarian lasagna, with the steps in order.",
    "Compare TCP and QUIC: connection setup, head-of-line blocking, congestion control.",
];

/// What a run measures
struct Plan {
    /// prompts and tokens per answer of the draft rows
    prompts: usize,
    tokens: usize,
    drafts: &'static [usize],
    /// drafts of a sampled row (temperature 1, top-k 20, top-p 0.95)
    sampled: Option<usize>,
    /// prompts and tokens per answer of the batching rows, answered by a
    /// number of chats at a time, with a number of drafts: the first is the
    /// baseline
    batch_prompts: usize,
    batch_tokens: usize,
    batches: &'static [(usize, usize)],
}

const FAST: Plan = Plan {
    prompts: 2,
    tokens: 96,
    drafts: &[0, 1, 4],
    sampled: None,
    batch_prompts: 4,
    batch_tokens: 48,
    batches: &[(1, 0), (4, 0)],
};

const EXACT: Plan = Plan {
    prompts: 3,
    tokens: 128,
    drafts: &[0, 1, 2, 4, 6],
    sampled: Some(4),
    batch_prompts: 8,
    batch_tokens: 40,
    batches: &[(1, 0), (2, 0), (4, 0), (8, 0), (4, 1), (8, 1)],
};

/// One answer: its text, and what [`Chat::send`] counted
struct Answer {
    text: String,
    tokens: usize,
    drafted: usize,
    accepted: usize,
    /// from the end of the prompt
    decode: Duration,
}

fn answer(
    engine: &dyn Engine,
    tok: &Tokenizer,
    opts: &ChatOptions,
    sampler: &SamplerConfig,
    prompt: &str,
) -> Result<Answer, Error> {
    let mut chat = Chat::new(engine, tok, opts.clone())?;
    let mut sampler = Sampler::new(sampler.clone());
    let mut text = String::new();
    let mut start = Instant::now();
    let s = chat.send(
        prompt,
        &mut sampler,
        |ev| match ev {
            Event::PromptDone => start = Instant::now(),
            Event::Answer(a) => text.push_str(a),
            _ => {}
        },
        || false,
    )?;
    Ok(Answer {
        text,
        tokens: s.gen_tokens,
        drafted: s.drafted,
        accepted: s.accepted,
        decode: start.elapsed(),
    })
}

fn greedy() -> SamplerConfig {
    SamplerConfig {
        temperature: 0.0,
        ..Default::default()
    }
}

fn options(tokens: usize, draft: usize) -> ChatOptions {
    ChatOptions {
        thinking: false,
        max_tokens: tokens,
        context: 4096,
        draft: Some(draft),
        ..Default::default()
    }
}

fn rate(tokens: usize, d: Duration) -> f64 {
    tokens as f64 / d.as_secs_f64().max(1e-9)
}

/// Run the benchmark, `exact` for more answers, drafts and chats: returns
/// whether greedy drafting kept every text
pub fn run(engine: &dyn Engine, tok: &Tokenizer, mtp: bool, exact: bool) -> Result<bool, Error> {
    let plan = if exact { &EXACT } else { &FAST };
    let drafts: &[usize] = if mtp { plan.drafts } else { &[0] };
    let t0 = Instant::now();
    // the weights paged in and the pipelines built before anything is timed
    answer(
        engine,
        tok,
        &options(8, drafts[drafts.len() - 1]),
        &greedy(),
        "Hi",
    )?;

    println!(
        "{}: {} prompts x {} tokens, greedy{}",
        if exact { "exact" } else { "fast" },
        plan.prompts,
        plan.tokens,
        if mtp { "" } else { " (no MTP head: no drafts)" }
    );
    println!(
        "{:<9} {:>12}  {:>7}  {:>8}  {:>8}  same text",
        "drafts", "decode", "speedup", "accepted", "tok/step"
    );
    let prompts = &PROMPTS[..plan.prompts];
    let mut rows: Vec<(String, usize, SamplerConfig)> = drafts
        .iter()
        .map(|&d| (d.to_string(), d, greedy()))
        .collect();
    if let Some(d) = plan.sampled.filter(|_| mtp) {
        let sampled = SamplerConfig {
            seed: 1,
            ..Default::default()
        };
        rows.push((format!("{d} sampled"), d, sampled));
    }
    let (mut base, mut plain, mut kept) = (0.0, Vec::new(), true);
    for (name, d, sampler) in &rows {
        let answers: Vec<Answer> = prompts
            .iter()
            .map(|p| answer(engine, tok, &options(plan.tokens, *d), sampler, p))
            .collect::<Result<_, _>>()?;
        let tokens: usize = answers.iter().map(|a| a.tokens).sum();
        let r = rate(tokens, answers.iter().map(|a| a.decode).sum());
        let (drafted, accepted): (usize, usize) = answers
            .iter()
            .fold((0, 0), |(d, a), x| (d + x.drafted, a + x.accepted));
        let texts: Vec<String> = answers.into_iter().map(|a| a.text).collect();
        let same = if *d == 0 && sampler.temperature == 0.0 {
            (base, plain) = (r, texts);
            "-".to_string()
        } else if sampler.temperature > 0.0 {
            "-".to_string()
        } else {
            let n = texts.iter().zip(&plain).filter(|(a, b)| a == b).count();
            kept &= n == texts.len();
            let mark = if n == texts.len() { "" } else { "  DIFFERENT" };
            format!("{n}/{}{mark}", texts.len())
        };
        let (acc, step) = if drafted == 0 {
            ("-".to_string(), 1.0)
        } else {
            // a step is one token of the trunk and the drafts it confirmed
            let steps = tokens.saturating_sub(accepted).max(1);
            let pct = 100.0 * accepted as f64 / drafted as f64;
            (format!("{pct:.0}%"), tokens as f64 / steps as f64)
        };
        println!(
            "{name:<9} {r:6.1} tok/s  {:6.2}x  {acc:>8}  {step:8.2}  {same}",
            r / base
        );
    }

    println!(
        "\nbatching: {} prompts x {} tokens, greedy, each chat taking the next prompt when done",
        plan.batch_prompts, plan.batch_tokens
    );
    println!(
        "{:>5}  {:>6}  {:>12}  {:>7}  same answers",
        "chats", "drafts", "tok/s", "speedup"
    );
    let prompts = &PROMPTS[..plan.batch_prompts];
    let (mut base, mut alone) = (0.0, Vec::new());
    for &(j, d) in plan.batches.iter().filter(|b| mtp || b.1 == 0) {
        let opts = options(plan.batch_tokens, d);
        let next = Mutex::new(0);
        let texts = Mutex::new(vec![(String::new(), 0); prompts.len()]);
        let t = Instant::now();
        batch::serve(engine, &BatchOptions::default(), |h| {
            std::thread::scope(|s| {
                let workers: Vec<_> = (0..j)
                    .map(|_| {
                        s.spawn(|| -> Result<(), Error> {
                            loop {
                                let i = {
                                    let mut n = next.lock().unwrap();
                                    *n += 1;
                                    *n - 1
                                };
                                let Some(p) = prompts.get(i) else {
                                    return Ok(());
                                };
                                let a = answer(&h, tok, &opts, &greedy(), p)?;
                                texts.lock().unwrap()[i] = (a.text, a.tokens);
                            }
                        })
                    })
                    .collect();
                workers.into_iter().try_for_each(|w| w.join().unwrap())
            })
        })??;
        let texts: Vec<(String, usize)> = texts.into_inner().unwrap();
        let r = rate(texts.iter().map(|a| a.1).sum(), t.elapsed());
        let same = if alone.is_empty() {
            (base, alone) = (r, texts);
            "-".to_string()
        } else {
            let n = texts.iter().zip(&alone).filter(|(a, b)| a.0 == b.0).count();
            format!("{n}/{}", texts.len())
        };
        println!("{j:5}  {d:6}  {r:6.1} tok/s  {:6.2}x  {same}", r / base);
    }
    println!("\n{:.0}s", t0.elapsed().as_secs_f64());
    Ok(kept)
}
