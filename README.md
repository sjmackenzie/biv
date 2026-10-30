# biv — Brain In a Vat

Pure Rust chat inference for **Qwen3.8** GGUF models (llama.cpp architecture
`qwen35`), on Apple GPUs (Metal) or the CPU (aarch64 NEON, x86-64 AVX2). No
C/C++ code and no bindings: the GGUF loader, the quantized kernels (NEON `sdot`,
AVX2 `maddubs`, Metal shaders), the tokenizer, and the model graph are all in Rust
or MSL.

```sh
cargo build --release
./target/release/biv                      # hf.co/unsloth/Qwen3.8-27B-GGUF:Q4_1
./target/release/biv -m hf.co/unsloth/Qwen3.8-27B-GGUF:BF16 --no-think
./target/release/biv -m /path/to/model.gguf -p "one-shot question"
./target/release/biv --cpu                # skip the GPU
```

On x86-64 the AVX2 + FMA + F16C kernels (Haswell, Zen and later) are picked at run
time, so a default build runs on any x86-64 CPU and falls back to scalar code
without AVX2. `RUSTFLAGS="-C target-cpu=native"` additionally lets the compiler
use AVX2 in the non-kernel code (norms, softmax, sampling).

On macOS the Metal backend is the default. Its shaders are compiled from source
at startup, which takes about 0.5s, and the weights are used in place from the
mmap'd GGUF with no copy.

Models are looked up in the Hugging Face cache (`$HF_HUB_CACHE`, `$HF_HOME/hub`,
`~/.cache/huggingface/hub`). That's where skelm downloads them, and it uses the
same `hf.co/<owner>/<repo>:<quant>` names. Split GGUFs (`-00001-of-0000N`) are loaded
as one model.

Decoding is speculative by default: the model's MTP head drafts the next token,
and the trunk checks it in the same pass as the current one (`--draft <n>` sets
how many tokens are drafted, `0` turns it off). The output is the same as
without drafting, greedy or sampled.

In the chat, `/reset` clears the conversation, `/think` and `/nothink` toggle
reasoning for the next answers, `/stats` shows context use, Ctrl-C stops an
answer, and a trailing `\` continues a message on the next line. Run `--help` for
the sampling and context options. Sampling defaults come from the GGUF
(`general.sampling.*`).

## Fingerprint

`biv fingerprint [options]` prints a fingerprint of the inference stack, to
detect any change in what it generates. It takes the chat's options (`-m`,
`--cpu`, `-b`, `-d`, `--temp`, `--seed`…) and writes `key=value` lines (with
`\\`, `\n` and `\r` escaped in values), in two parts:

- the stack, what should decide the output besides the code: the model (with
  its SHA-256 when known), the OS release (its libm, and on macOS the compiler
  of the Metal shaders), the backend (CPU kernels or GPU, KV cache type, batch
  size), and the generation options. `stack_id` hashes it.
- `probe.*`, a fixed run on that stack: a prompt in passes of 64 and 16 tokens,
  12 tokens decoded greedily one per pass, the same from the restored prompt
  with drafts verified in one pass, and a sampled chat turn. Each pass records
  a hash of all its logits' bits (`.logits`) and the top 8 as `id:value`
  (`.top`). `output_id` hashes it. It takes about 5s on Metal and 30s on the CPU.

The code itself is not hashed: a change to it shows in the logits if it changes
the output, and not otherwise. `info.*` lines (CPU, threads, model path) don't
count. The hashes are 64-bit FNV-1a, to tell that something changed, not to
resist forgery.

The model's SHA-256 is given by the caller of `fingerprint::stack`, as a
`[u8; 32]`, or left out; for a split model, it is the XOR of the shards'.
`biv` takes it from the Hugging Face cache, which names each file after its
SHA-256; a change to the weights shows in the logits either way.

```sh
biv fingerprint > ref.txt
biv fingerprint --check ref.txt   # exits with 1 if anything differs
```

`--check` runs the probe again and reports the stack lines that changed, and for
the output: how many logits are identical, the largest difference of the top
logits, the first token chosen differently, and the chat text. The stack says
what changed, the probe whether the output did: a new OS can change the shaders'
code with no line changing, and a line can change with no effect on the output.

Runs repeat bit for bit: the Metal kernels use no atomics, and on the CPU each
output row is computed whole by one thread, so the thread count changes nothing
(4 and 12 threads give the same bits). How tokens are batched does change the
bits (`-b 16` splits the probe's passes of 64), and so does the backend: CPU and
Metal logits differ by up to about 0.4 on this probe.

## Layout

Two crates: `biv` (`engine/`) is the inference engine, with no I/O, and
`biv-cli` (the root, which builds the `biv` executable) is the command line
around it: files, mmap, the Hugging Face cache, the terminal.

Without its default features the engine is `no_std` + `alloc`: the model is
parsed from the GGUF's bytes (`Gguf::from_shards`), compute runs on one
thread and float functions come from `libm`. The x86 kernels are picked at run
time either way (cpuid). Its features:

| feature | brings |
|---|---|
| `std` (default) | std's float functions |
| `rayon` (default) | multithreaded CPU compute |
| `metal` | the Metal backend (macOS) |

| file | role |
|---|---|
| `engine/src/gguf.rs` | GGUF parser over in-memory shards, split files |
| `engine/src/quant.rs` | block formats (dequant for Q2_K…Q8_0, F16, BF16), int8 activations, NEON and AVX2 dot kernels with scalar references |
| `engine/src/ops.rs` | matmul (threaded with rayon), norms, rope, softmax |
| `engine/src/par.rs` | rayon's parallel iterators, or sequential ones of the same names |
| `engine/src/math.rs` | libm float functions without `std` |
| `engine/src/model.rs` | the `qwen35` graph: port of llama.cpp `src/models/qwen35.cpp` and `delta-net-base.cpp`; MTP drafting |
| `engine/src/engine.rs` | `Engine` / `Session` traits shared by the backends (draft / verify / keep for speculative decoding); CPU implementation |
| `engine/src/metal/mod.rs` | Metal backend: no-copy weight buffers, per-session KV/recurrent state, one command buffer per forward pass |
| `engine/src/metal/kernels.metal` | matvec / simdgroup matmul for each quant type (from ggml-metal), fused norm, rope, split-K (decode) and flash (prefill) attention, conv and gated delta-net kernels |
| `engine/src/tokenizer.rs` | byte-level BPE; the `qwen35` pre-tokenizer regex, matched by hand |
| `engine/src/chat.rs` | chat template (ChatML with `<think>`), incremental multi-turn state, the speculative decoding loop |
| `engine/src/sampler.rs` | temperature / top-k / top-p / min-p / penalties |
| `src/lib.rs` | GGUF files: shard discovery, mmap |
| `src/hf.rs` | Hugging Face cache lookup |
| `src/fingerprint.rs` | `biv fingerprint`: the stack, the probe, the comparison |
| `src/main.rs` | the chat CLI |
| `src/bin/gguf-dump.rs` | print a GGUF's metadata and tensors |

## Model notes

64 layers. Every 4th layer is gated full attention: GQA 24/4 heads of 256,
q/k RMS norm, NeoX rope on the first 64 dims (IMROPE reduces to this for
text), and a sigmoid output gate packed with Q. The other layers are Gated DeltaNet:
a causal conv (k=4), L2-normed q/k, and a delta rule over a 128×128 state per value
head (48 value heads, 16 key heads in tiled order). Each layer has a SwiGLU FFN.
Conversion has already folded `+1` into the norm weights and stores `A` as
`-exp(A_log)`.

The MTP block (`blk.64`) drafts tokens. Its row `p` is
`eh_proj · [enorm(embedding of token p+1) ‖ hnorm(normalized hidden state of token p)]`,
then a gated attention layer over its own history of rows and `shared_head_norm`;
the trunk's output head reads the next token. Every pass of the trunk also enters
its completed rows into that history. A second draft chains on the normalized
output of the first.

Speculative decoding (`Session::draft` / `verify` / `keep`): the trunk runs the
sampled token and the drafts in one pass, with logits for each. The drafts are
greedy, so sampling from the trunk and accepting a draft when it equals the
sampled token gives the same distribution as decoding without drafts. Rejected
tokens are dropped by moving `n_past` back for the KV caches; the delta net
states have one copy per drafted token plus one, the verify pass writes the state
after each token to the next copy, and `keep` only changes which copy is current.

The chat template keeps past reasoning in the history by default, so each turn
only appends tokens. The KV cache and the recurrent states are kept between turns,
and only the new message is processed.

Tools (`ChatOptions::tools`, JSON function definitions) go in the system block
as the chat template writes them. The model's calls come as `Event::ToolCall`
(function name and parameters, as the model wrote them), and their results go
back with `Chat::send_tool_responses`.

A conversation can be saved between turns (`Chat::save`) and restored in any
chat on the same engine (`Chat::restore`), which is how a system prompt is
cached: saved before the first message, the prefix holds the processed system
block, tools included. The recurrent states can't be rewound, so a prefix is a copy of the
whole sequence state: the KV cache of its tokens plus the delta net states,
about 150 MB for this model, plus 64 KB per token on Metal (128 KB on the CPU,
whose caches are f32).

## Verification

- Kernel tests: `cargo test --workspace --release` compares every SIMD kernel
  with its dequantize-then-dot reference. The x86-64 kernels can be tested on
  Apple silicon through Rosetta. It runs AVX2 code but only advertises it when
  asked: `ROSETTA_ADVERTISE_AVX=1 cargo test --workspace --release --target
  x86_64-apple-darwin` (without the variable, the same command tests the scalar
  fallback). The x86-64 build under Rosetta also generates the same greedy text
  as the native build. Rosetta timings are not representative of x86 hardware.
- Tokenization of the test prompts matches `llama-tokenize` exactly.
- Greedy generation matches llama.cpp (`llama-simple -ngl 0`) token for token
  on the same GGUF.
- `examples/consistency.rs`: the logits don't depend on how a prompt is split
  into forward passes, so state carry across chunks is exact.
- `examples/speculative.rs <gguf> [--cpu] [-n tokens] [-d draft]`: greedy
  answers with and without drafts are the same text, on Metal and on the CPU,
  with 1 to 3 drafts.
- `examples/compare.rs <gguf> [repeat]`: Metal vs CPU logits on the same prompt,
  then decode steps. The top-1 token matches at every step. Max logit differences
  are 0.2–0.4 on short prompts and up to a few units in the tail at ~700 tokens
  (f16 KV cache, half-precision matmul tiles).
- `examples/prefix.rs <gguf> [--cpu] [-n tokens]`: a chat restored from a saved
  system prompt or first turn gives the same greedy answer as the chat that
  saved it, on Metal and on the CPU, with or without drafts.
- `examples/tools.rs <gguf> [--cpu]`: the first prompt with tools has the tokens
  of the template text rendered by hand; the model calls both tools of a two-part
  question, answers from their results, and answers directly when no tool
  applies, with and without thinking; a restored prefix with tools makes the
  same calls.
- `examples/agent.rs <gguf> [--cpu] [--think] [task]`: an agent loop whose tools
  do I/O in a temporary workspace: list a directory, read a file, wait for a
  file that a simulated build writes after 10 s. The model waits for the build,
  reads what it needs and answers; paths outside the workspace are refused.
- `examples/attention.rs <gguf> [tokens]`: Metal logits after prefilling the
  same tokens in full batches (flash attention) and in passes of 8 (split-K)
  differ no more than between two splits that both use split-K.

## Performance (M2 Max 38-core GPU, Q4_1)

| | biv Metal | llama.cpp Metal | biv CPU (12 threads) | llama.cpp CPU |
|---|---|---|---|---|
| decode, 1 draft | 26–27 tok/s greedy, 23 sampled | | ~5.7 tok/s | |
| decode, no draft | 18–19 tok/s | 17.6 tok/s | ~5.3 tok/s | 4.5 tok/s |
| prompt (~20 tok) | 48 tok/s | 36.8 tok/s | ~9–10 tok/s | 5.0 tok/s |
| prompt (1541 tok) | 149 tok/s | | | |

BF16 on Metal decodes at about 7.4 tok/s without drafts, which means reading
about 400 GB/s of weights, close to the memory bandwidth, and 10.6–11.1 tok/s
with one.

Speculative decoding (`examples/speculative.rs`, greedy, 256 tokens): the MTP
head's draft is right 86–96% of the time (about 66% sampling at temperature 1).
On Metal a step is a draft (~5 ms: the MTP layer and the output head) and a
verify pass of 2 tokens (~1.2× a decode step: the matvec kernels read the weights
for the second token mostly from cache). Verify passes of 3 and 4 tokens cost
1.6× and 2×, so 2 or 3 drafts, right 76–92% and 69–85% of the time, are slower
than one. On the CPU a 2-token pass costs 1.45× a decode step, which leaves a
small gain.

Long contexts (`examples/longctx.rs <gguf> [depth...]`, Metal):

| context | 1k | 4k | 8k | 16k | 32k |
|---|---|---|---|---|---|
| decode | 18.1 tok/s | 17.8 | 17.5 | 17.0 | 16.2 |
| prefill up to it | 184 tok/s | 183 | 177 | 148 | 137 |

Decode attention is split-K ("flash decoding"). Each threadgroup takes a KV head with
its 6 query heads over one range of positions, so K/V is read once per group, and
at most 64 ranges are merged by a second kernel. Each simdgroup holds two query
heads; with three or more, q and the accumulators spill out of registers and the
kernel gets 2–6× slower. At 32k the attention of a layer reads 128 MB of f16 KV
cache in about 0.45 ms (~300 GB/s).

Batches of 32 tokens or more use a flash attention kernel on simdgroup matrices
instead: a threadgroup takes 16 query rows (token, head pairs of a KV head) and
walks the positions 64 at a time, Q·Kᵀ, online softmax, then P·V, with each
simdgroup holding a quarter of the head dimension of the output in registers.
It runs at about 6.2 TFLOPS from 2k to 32k positions, 2.4× the split-K kernel
over a 256-token batch, which took about 37% of prefill time at 16–32k. Its
loops over register arrays must be fully unrolled: left to the compiler, the
arrays went to memory and the kernel was slower than split-K.

## Next steps

- Ctrl-C during a long prefill.
- i8mm (`smmla`) CPU prompt kernel; f16 KV cache on CPU.
- AVX-VNNI / AVX-512 VNNI (`vpdpbusd`) kernels for x86-64. They are not testable
  under Rosetta, so they need real hardware.
- Matvec kernels that read the weights once for 2–4 tokens, for longer drafts.
  Looping over the tokens inside a simdgroup (weights in registers) was slower
  than a threadgroup per token for Q4_1: the per-token activation loads dominate.
- A 2-row CPU dot kernel, for speculative decoding on the CPU.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT), at your option.
