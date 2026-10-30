//! Interactive chat with a Qwen3.8 GGUF model.

use std::error::Error;
use std::io::{BufRead, IsTerminal, Write};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use biv::chat::{Chat, ChatOptions, Event, ReasoningEffort, TurnStats};
use biv::engine::Engine;
use biv::gguf::{Gguf, Value};
use biv::model::Model;
use biv::sampler::{Sampler, SamplerConfig};
use biv::tokenizer::Tokenizer;
use biv_cli::{fingerprint, hf};

const DEFAULT_MODEL: &str = "hf.co/unsloth/Qwen3.8-27B-GGUF:Q4_1";

const USAGE: &str = "\
usage: biv [options]
       biv fingerprint [options] [--check <file>]

  -m, --model <path|name>   GGUF file, or a cached Hugging Face model as skelm
                            names it [default: hf.co/unsloth/Qwen3.8-27B-GGUF:Q4_1]
  -p, --prompt <text>       answer one message and exit
  -s, --system <text>       system prompt
      --no-think            disable the reasoning block
      --effort <level>      reasoning effort: low, medium, xhigh [default: xhigh]
  -c, --ctx <n>             context size in tokens [default: 16384]
  -n, --max-tokens <n>      tokens per answer, thinking included [default: 8192]
      --cpu                 run on the CPU instead of the GPU (Metal)
  -t, --threads <n>         CPU compute threads [default: all cores]
  -b, --batch <n>           prompt tokens per forward pass [default: 64 cpu, 256 metal]
  -d, --draft <n>           tokens drafted per step by the model's MTP head for
                            speculative decoding, 0 to disable [default: 1]
      --temp <f>            temperature, 0 for greedy [default: from the model, 1.0]
      --top-k <n>           [default: from the model, 20]
      --top-p <f>           [default: from the model, 0.95]
      --min-p <f>           [default: 0]
      --presence-penalty <f>  [default: 0]
      --repeat-penalty <f>  [default: 1]
      --seed <n>            [default: random]
  -h, --help

fingerprint prints, as key=value lines, what should decide the generated text
(model files, platform, backend, the options above) and the logits of a fixed
test run on this stack. With --check, it compares them with a saved fingerprint, exiting
with 1 when anything differs.

chat commands: /reset, /think, /nothink, /stats, /quit";

struct Args {
    /// `biv fingerprint`, with the file of `--check`
    fingerprint: Option<Option<String>>,
    model: String,
    prompt: Option<String>,
    threads: Option<usize>,
    cpu: bool,
    opts: ChatOptions,
    temp: Option<f32>,
    top_k: Option<usize>,
    top_p: Option<f32>,
    min_p: f32,
    presence_penalty: f32,
    repeat_penalty: f32,
    seed: u64,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        fingerprint: None,
        model: DEFAULT_MODEL.to_string(),
        prompt: None,
        threads: None,
        cpu: !cfg!(target_os = "macos"),
        opts: ChatOptions::default(),
        temp: None,
        top_k: None,
        top_p: None,
        min_p: 0.0,
        presence_penalty: 0.0,
        repeat_penalty: 1.0,
        seed: 0,
    };
    let mut it = std::env::args().skip(1).peekable();
    if it.next_if(|a| a == "fingerprint").is_some() {
        a.fingerprint = Some(None);
    }
    while let Some(flag) = it.next() {
        let mut val = || it.next().ok_or_else(|| format!("{flag} needs a value"));
        fn num<T: std::str::FromStr>(flag: &str, v: String) -> Result<T, String> {
            v.parse()
                .map_err(|_| format!("{flag}: invalid value '{v}'"))
        }
        match flag.as_str() {
            "-m" | "--model" => a.model = val()?,
            "-p" | "--prompt" => a.prompt = Some(val()?),
            "-s" | "--system" => a.opts.system = Some(val()?),
            "--no-think" => a.opts.thinking = false,
            "--effort" => {
                let v = val()?;
                a.opts.effort = ReasoningEffort::parse(&v)
                    .ok_or_else(|| format!("--effort: unknown level '{v}'"))?
            }
            "-c" | "--ctx" => a.opts.context = num(&flag, val()?)?,
            "-n" | "--max-tokens" => a.opts.max_tokens = num(&flag, val()?)?,
            "-t" | "--threads" => a.threads = Some(num(&flag, val()?)?),
            "-b" | "--batch" => a.opts.batch = Some(num(&flag, val()?)?),
            "-d" | "--draft" => a.opts.draft = num(&flag, val()?)?,
            "--cpu" => a.cpu = true,
            "--temp" => a.temp = Some(num(&flag, val()?)?),
            "--top-k" => a.top_k = Some(num(&flag, val()?)?),
            "--top-p" => a.top_p = Some(num(&flag, val()?)?),
            "--min-p" => a.min_p = num(&flag, val()?)?,
            "--presence-penalty" => a.presence_penalty = num(&flag, val()?)?,
            "--repeat-penalty" => a.repeat_penalty = num(&flag, val()?)?,
            "--seed" => a.seed = num(&flag, val()?)?,
            "--check" if a.fingerprint.is_some() => a.fingerprint = Some(Some(val()?)),
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            _ => return Err(format!("unknown option {flag}\n\n{USAGE}")),
        }
    }
    Ok(a)
}

fn default_threads() -> usize {
    // efficiency cores included: the matmuls are compute-bound enough to gain from them
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

struct Style {
    tty: bool,
}

impl Style {
    fn dim(&self) -> &'static str {
        if self.tty { "\x1b[2m" } else { "" }
    }
    fn bold(&self) -> &'static str {
        if self.tty { "\x1b[1m" } else { "" }
    }
    fn reset(&self) -> &'static str {
        if self.tty { "\x1b[0m" } else { "" }
    }
}

fn print_stats(s: &TurnStats, prompt_time: Duration, gen_time: Duration, style: &Style) {
    let rate = |n: usize, d: Duration| n as f64 / d.as_secs_f64().max(1e-9);
    eprintln!(
        "{}[prompt {} tok, {:.1} tok/s | answer {} tok, {:.2} tok/s{}{}]{}",
        style.dim(),
        s.prompt_tokens,
        rate(s.prompt_tokens, prompt_time),
        s.gen_tokens,
        rate(s.gen_tokens, gen_time),
        if s.drafted > 0 {
            format!(" | drafts {}/{} right", s.accepted, s.drafted)
        } else {
            String::new()
        },
        if s.truncated { " | truncated" } else { "" },
        style.reset()
    );
}

fn make_engine<'m>(
    model: &'m Model<'m>,
    cpu: bool,
    batch: Option<usize>,
) -> Result<Box<dyn Engine + 'm>, Box<dyn Error>> {
    if cpu {
        return Ok(Box::new(model));
    }
    #[cfg(target_os = "macos")]
    {
        Ok(Box::new(biv::metal::MetalEngine::new(
            model,
            batch.unwrap_or(256),
        )?))
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = batch;
        Err("no GPU backend on this platform, use --cpu".into())
    }
}

/// Print the fingerprint of the stack, or compare it with a saved one
fn run_fingerprint(
    path: &Path,
    gguf: &Gguf,
    tok: &Tokenizer,
    engine: &dyn Engine,
    generation: &fingerprint::Generation<'_>,
    threads: usize,
    check: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    let saved = match check {
        Some(file) => {
            let text = std::fs::read_to_string(file).map_err(|e| format!("{file}: {e}"))?;
            Some(fingerprint::Fingerprint::parse(&text).map_err(|e| format!("{file}: {e}"))?)
        }
        None => None,
    };
    let stack = fingerprint::stack(hf::model_sha256(path), gguf, engine, generation);
    let t0 = Instant::now();
    let probe = fingerprint::probe(engine, tok)?;
    eprintln!("probe run in {:.1}s", t0.elapsed().as_secs_f32());
    let doc = fingerprint::document(stack, probe, fingerprint::info(path, threads));
    let Some(saved) = saved else {
        print!("{}", doc.to_text());
        return Ok(());
    };
    let (lines, same) = fingerprint::compare(&saved, &doc, |id| tok.decode(&[id]))?;
    for l in lines {
        println!("{l}");
    }
    if !same {
        std::process::exit(1);
    }
    Ok(())
}

fn run() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let threads = args.threads.unwrap_or_else(default_threads);
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()?;

    let path = hf::resolve(&args.model)?;
    let t0 = Instant::now();
    let gguf = biv_cli::open_gguf(&path)?;
    let tok = Tokenizer::from_gguf(&gguf)?;
    let model = Model::load(&gguf)?;
    let engine = make_engine(&model, args.cpu, args.opts.batch)?;
    let name = gguf
        .get("general.name")
        .and_then(Value::as_str)
        .unwrap_or("model")
        .to_string();
    eprintln!(
        "loaded {name} ({}) in {:.1}s: {} layers, {}",
        path.file_name().unwrap_or_default().to_string_lossy(),
        t0.elapsed().as_secs_f32(),
        model.cfg.n_layer,
        engine.description()
    );

    // sampling defaults recommended by the model's publisher
    let gf = |k: &str| gguf.get(k).and_then(Value::as_f32);
    let defaults = SamplerConfig::default();
    let sampler_cfg = SamplerConfig {
        temperature: args
            .temp
            .or(gf("general.sampling.temp"))
            .unwrap_or(defaults.temperature),
        top_k: args
            .top_k
            .or(gf("general.sampling.top_k").map(|v| v as usize))
            .unwrap_or(defaults.top_k),
        top_p: args
            .top_p
            .or(gf("general.sampling.top_p"))
            .unwrap_or(defaults.top_p),
        min_p: args.min_p,
        presence_penalty: args.presence_penalty,
        repeat_penalty: args.repeat_penalty,
        // 0: a different seed every run
        seed: if args.seed != 0 {
            args.seed
        } else {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0)
        },
        ..defaults
    };
    let mut opts = args.opts.clone();
    opts.context = opts.context.min(model.cfg.context_length);
    if let Some(check) = &args.fingerprint {
        let generation = fingerprint::Generation {
            chat: &opts,
            sampler: &sampler_cfg,
            random_seed: args.seed == 0,
        };
        return run_fingerprint(
            &path,
            &gguf,
            &tok,
            engine.as_ref(),
            &generation,
            threads,
            check.as_deref(),
        );
    }
    let mut sampler = Sampler::new(sampler_cfg);
    let mut chat = Chat::new(engine.as_ref(), &tok, opts)?;

    let stop = Arc::new(AtomicBool::new(false));
    {
        let stop = stop.clone();
        // first Ctrl-C stops the answer, a second one while idle quits
        ctrlc::set_handler(move || {
            if stop.swap(true, Ordering::SeqCst) {
                std::process::exit(130);
            }
        })?;
    }

    let style = Style {
        tty: std::io::stdout().is_terminal(),
    };
    let answer = |chat: &mut Chat<'_>, text: &str, sampler: &mut Sampler| -> Result<(), String> {
        stop.store(false, Ordering::SeqCst);
        let mut out = std::io::stdout().lock();
        let mut in_think = false;
        let t0 = Instant::now();
        let mut t1 = None;
        let result = chat.send(
            text,
            sampler,
            |ev| {
                match ev {
                    Event::PromptDone => t1 = Some(Instant::now()),
                    Event::Thinking(t) => {
                        if !in_think {
                            in_think = true;
                            let _ = write!(out, "{}", style.dim());
                        }
                        let _ = write!(out, "{t}");
                    }
                    Event::ThinkingDone => {
                        in_think = false;
                        let _ = write!(out, "{}\n\n", style.reset());
                    }
                    Event::Answer(t) => {
                        let _ = write!(out, "{t}");
                    }
                    // no tools are defined, but the model may still call one
                    Event::ToolCall(c) => {
                        let _ = write!(
                            out,
                            "{}[tool call: {}{:?}]{}",
                            style.dim(),
                            c.name,
                            c.arguments,
                            style.reset()
                        );
                    }
                }
                let _ = out.flush();
            },
            || stop.load(Ordering::SeqCst),
        );
        let _ = writeln!(out, "{}", style.reset());
        drop(out);
        let stats = result.map_err(|e| match e {
            biv::Error::ContextFull { .. } => format!("{e}, /reset to start over"),
            e => e.to_string(),
        })?;
        let t1 = t1.unwrap_or(t0);
        print_stats(&stats, t1 - t0, t1.elapsed(), &style);
        // a Ctrl-C during the answer must not count towards quitting
        stop.store(false, Ordering::SeqCst);
        Ok(())
    };

    if let Some(p) = &args.prompt {
        return Ok(answer(&mut chat, p, &mut sampler)?);
    }

    eprintln!("type a message, /help for commands, Ctrl-C stops an answer, Ctrl-D quits");
    let stdin = std::io::stdin();
    loop {
        eprint!("{}> {}", style.bold(), style.reset());
        let _ = std::io::stderr().flush();
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            eprintln!();
            return Ok(());
        }
        // a trailing backslash continues the message on the next line
        while line.trim_end_matches(['\n', '\r']).ends_with('\\') {
            let t = line.trim_end_matches(['\n', '\r']).len();
            line.truncate(t - 1);
            line.push('\n');
            eprint!("{}. {}", style.dim(), style.reset());
            if stdin.lock().read_line(&mut line)? == 0 {
                break;
            }
        }
        let msg = line.trim();
        match msg {
            "" => continue,
            "/quit" | "/exit" => return Ok(()),
            "/help" => eprintln!("{}", USAGE.rsplit("\n\n").next().unwrap_or_default()),
            "/reset" => match chat.reset() {
                Ok(()) => eprintln!("conversation cleared"),
                Err(e) => eprintln!("error: {e}"),
            },
            "/think" | "/nothink" => {
                chat.opts.thinking = msg == "/think";
                eprintln!("thinking {}", if chat.opts.thinking { "on" } else { "off" });
            }
            "/stats" => eprintln!(
                "{} / {} context tokens, {:.1} MB of state",
                chat.n_past(),
                chat.opts.context,
                chat.state_bytes() as f64 / 1e6
            ),
            _ if msg.starts_with('/') => eprintln!("unknown command {msg}"),
            _ => {
                if let Err(e) = answer(&mut chat, msg, &mut sampler) {
                    eprintln!("error: {e}");
                }
            }
        }
    }
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
