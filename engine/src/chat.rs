//! Multi-turn chat on top of the model, following the GGUF's chat template
//! (ChatML with `<think>` reasoning blocks).
//!
//! The template keeps past reasoning in the history (`preserve_thinking`
//! defaults to true), so every turn only appends to the sequence: the model
//! state is kept between turns and only the new tokens are processed. This
//! also suits the recurrent layers, whose state cannot be rewound (beyond the
//! last verify pass of speculative decoding).
//!
//! With tools, the model may answer with function calls ([`Event::ToolCall`]),
//! whose results go back with [`Chat::send_tool_responses`].
//!
//! A conversation can be saved at any point between turns and restored later,
//! in the same chat or another one on the same engine: [`Chat::save`] before the
//! first message saves the processed system prompt, so that conversations
//! starting with it don't process it again.

use alloc::collections::VecDeque;

use crate::Error;
use crate::engine::{Engine, Session, Snapshot};
use crate::prelude::*;
use crate::sampler::Sampler;
use crate::tokenizer::{PromptPart, StreamDecoder, TokenId, Tokenizer};

use PromptPart::{Content, Template};

/// The chat template's text before the tool definitions in the system block
const TOOLS_HEADER: &str = "# Tools\n\nYou have access to the following functions:\n\n<tools>";
/// And after them
const TOOLS_FOOTER: &str = "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningEffort {
    Low,
    Medium,
    XHigh,
}

impl ReasoningEffort {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "low" => Self::Low,
            "medium" => Self::Medium,
            "high" | "xhigh" => Self::XHigh,
            _ => return None,
        })
    }

    /// The system instruction the template adds when thinking is enabled
    fn instructions(self) -> Option<&'static str> {
        match self {
            Self::XHigh => Some(
                "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.",
            ),
            Self::Low => Some(
                "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.",
            ),
            Self::Medium => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ChatOptions {
    pub system: Option<String>,
    pub thinking: bool,
    pub effort: ReasoningEffort,
    /// per answer, thinking included
    pub max_tokens: usize,
    pub context: usize,
    /// prompt tokens processed per forward pass, by default the engine's choice
    pub batch: Option<usize>,
    /// tokens drafted by the MTP head per step, 0 for no speculative decoding
    pub draft: usize,
    /// functions the model may call, each a JSON object as the chat template's
    /// `tojson` writes it, e.g. `{"type": "function", "function": {"name":
    /// "get_weather", "description": "...", "parameters": {...}}}`
    pub tools: Vec<String>,
}

impl Default for ChatOptions {
    fn default() -> Self {
        ChatOptions {
            system: None,
            thinking: true,
            effort: ReasoningEffort::XHigh,
            max_tokens: 8192,
            context: 16384,
            batch: None,
            draft: 1,
            tools: Vec::new(),
        }
    }
}

/// Pieces of the answer, as they are generated
pub enum Event<'a> {
    /// the prompt is processed, generation starts
    PromptDone,
    Thinking(&'a str),
    Answer(&'a str),
    /// the reasoning block is complete
    ThinkingDone,
    /// a function call, to answer with [`Chat::send_tool_responses`]
    ToolCall(&'a ToolCall),
}

/// A function call the model made
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    pub name: String,
    /// parameters as the model wrote them: strings verbatim, other values as
    /// JSON (numbers, booleans, arrays, objects)
    pub arguments: Vec<(String, String)>,
}

impl ToolCall {
    /// From the text between `<tool_call>` and `</tool_call>`:
    /// `<function=NAME>`, then `<parameter=NAME>\nVALUE\n</parameter>` for each
    /// parameter, then `</function>`
    fn parse(text: &str) -> Option<Self> {
        let (name, mut rest) = text
            .trim_start()
            .strip_prefix("<function=")?
            .split_once('>')?;
        let mut arguments = Vec::new();
        loop {
            rest = rest.trim_start();
            let Some(param) = rest.strip_prefix("<parameter=") else {
                break;
            };
            let (param, after) = param.split_once('>')?;
            let (value, after) = after.split_once("</parameter>")?;
            let value = value.strip_prefix('\n').unwrap_or(value);
            let value = value.strip_suffix('\n').unwrap_or(value);
            arguments.push((param.trim().to_string(), value.to_string()));
            rest = after;
        }
        if rest.strip_prefix("</function>")?.trim().is_empty() {
            Some(ToolCall {
                name: name.trim().to_string(),
                arguments,
            })
        } else {
            None
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct TurnStats {
    pub prompt_tokens: usize,
    pub gen_tokens: usize,
    /// stopped by max_tokens or the context size rather than by the model
    pub truncated: bool,
    /// tokens drafted by the MTP head, and how many of them were right
    pub drafted: usize,
    pub accepted: usize,
    /// the function calls of the answer, as given to [`Event::ToolCall`]
    pub tool_calls: Vec<ToolCall>,
}

/// A conversation up to some point, see [`Chat::save`]
pub struct Prefix {
    snapshot: Snapshot,
    pending: Vec<TokenId>,
    generated: Vec<TokenId>,
}

impl Prefix {
    /// Tokens processed
    pub fn n_past(&self) -> usize {
        self.snapshot.n_past()
    }

    /// Memory held
    pub fn bytes(&self) -> usize {
        self.snapshot.bytes()
    }
}

pub struct Chat<'m> {
    engine: &'m dyn Engine,
    tok: &'m Tokenizer,
    pub opts: ChatOptions,
    session: Box<dyn Session + 'm>,
    /// generated but not yet processed tokens, fed before the next prompt
    pending: Vec<TokenId>,
    /// every generated token, for the sampler penalties
    generated: Vec<TokenId>,
    im_end: TokenId,
    think_end: TokenId,
    tool_call: TokenId,
    tool_call_end: TokenId,
}

impl<'m> Chat<'m> {
    pub fn new(
        engine: &'m dyn Engine,
        tok: &'m Tokenizer,
        opts: ChatOptions,
    ) -> Result<Self, Error> {
        let id = |s: &str| tok.token_id(s).ok_or_else(|| Error::MissingToken(s.into()));
        Ok(Chat {
            engine,
            tok,
            session: engine.new_session(opts.context, opts.draft)?,
            opts,
            pending: Vec::new(),
            generated: Vec::new(),
            im_end: id("<|im_end|>")?,
            think_end: id("</think>")?,
            tool_call: id("<tool_call>")?,
            tool_call_end: id("</tool_call>")?,
        })
    }

    pub fn reset(&mut self) -> Result<(), Error> {
        // release the old state before allocating the new one
        self.session = Box::new(Empty);
        self.session = self
            .engine
            .new_session(self.opts.context, self.opts.draft)?;
        self.pending.clear();
        self.generated.clear();
        Ok(())
    }

    pub fn n_past(&self) -> usize {
        self.session.n_past()
    }

    pub fn state_bytes(&self) -> usize {
        self.session.state_bytes()
    }

    /// Save the conversation so far, to [`Chat::restore`] it in this chat or
    /// another on the same engine. Before the first message, the system prompt
    /// is processed now, so that the prefix holds it.
    pub fn save(&mut self) -> Result<Prefix, Error> {
        if self.session.n_past() == 0 {
            let mut parts = Vec::new();
            self.system_block(&mut parts);
            let system = self.tok.encode_prompt(&parts);
            if system.len() >= self.opts.context {
                return Err(Error::ContextFull {
                    n_past: 0,
                    prompt: system.len(),
                    context: self.opts.context,
                });
            }
            self.prefill(&system);
        }
        Ok(Prefix {
            snapshot: self.session.snapshot()?,
            pending: self.pending.clone(),
            generated: self.generated.clone(),
        })
    }

    /// Go back to a saved conversation. Its system prompt, reasoning
    /// instructions and tools are those of the chat that saved it, whatever
    /// `opts` says.
    pub fn restore(&mut self, prefix: &Prefix) -> Result<(), Error> {
        self.session.restore(&prefix.snapshot)?;
        self.pending.clone_from(&prefix.pending);
        self.generated.clone_from(&prefix.generated);
        Ok(())
    }

    /// Append `tokens` to the sequence, in passes of the batch size
    fn prefill(&mut self, tokens: &[TokenId]) {
        let batch = self
            .opts
            .batch
            .unwrap_or_else(|| self.engine.batch_size())
            .max(1);
        for chunk in tokens.chunks(batch) {
            self.session.prefill(chunk);
        }
    }

    /// The system block, as the chat template writes it: the reasoning
    /// instructions, the tools, the system prompt; nothing without any of them
    fn system_block<'s>(&'s self, parts: &mut Vec<PromptPart<'s>>) {
        let system = self
            .opts
            .system
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty());
        let reasoning = if self.opts.thinking {
            self.opts.effort.instructions()
        } else {
            None
        };
        let tools = !self.opts.tools.is_empty();
        if reasoning.is_none() && !tools && system.is_none() {
            return;
        }
        parts.push(Template("<|im_start|>system\n"));
        if let Some(r) = reasoning {
            parts.push(Template(r));
            if tools || system.is_some() {
                parts.push(Template("\n\n"));
            }
        }
        if tools {
            parts.push(Template(TOOLS_HEADER));
            for tool in &self.opts.tools {
                parts.extend([Template("\n"), Content(tool.trim())]);
            }
            parts.push(Template(TOOLS_FOOTER));
            if system.is_some() {
                parts.push(Template("\n\n"));
            }
        }
        if let Some(s) = system {
            parts.push(Content(s));
        }
        parts.push(Template("<|im_end|>\n"));
    }

    /// The tokens of the next turn: what is left of the previous answer, the
    /// system block on the first turn, `message` and the generation prompt
    fn turn_tokens<'s>(&'s self, message: &[PromptPart<'s>]) -> Vec<TokenId> {
        let mut parts = Vec::new();
        if !self.pending.is_empty() {
            parts.push(Template("\n"));
        }
        if self.session.n_past() == 0 {
            self.system_block(&mut parts);
        }
        parts.extend_from_slice(message);
        parts.push(Template("<|im_start|>assistant\n"));
        parts.push(Template(if self.opts.thinking {
            "<think>\n"
        } else {
            "<think>\n\n</think>\n\n"
        }));
        let mut t = self.pending.clone();
        t.extend(self.tok.encode_prompt(&parts));
        t
    }

    /// Send a user message and stream the answer through `on_event`.
    /// `stop` is polled between tokens to interrupt the generation.
    pub fn send(
        &mut self,
        user: &str,
        sampler: &mut Sampler,
        on_event: impl FnMut(Event<'_>),
        stop: impl Fn() -> bool,
    ) -> Result<TurnStats, Error> {
        let prompt = self.turn_tokens(&[
            Template("<|im_start|>user\n"),
            Content(user.trim()),
            Template("<|im_end|>\n"),
        ]);
        self.answer(prompt, sampler, on_event, stop)
    }

    /// Send the results of the function calls of the last answer, in their
    /// order, and stream the answer as [`Chat::send`] does
    pub fn send_tool_responses(
        &mut self,
        responses: &[impl AsRef<str>],
        sampler: &mut Sampler,
        on_event: impl FnMut(Event<'_>),
        stop: impl Fn() -> bool,
    ) -> Result<TurnStats, Error> {
        let mut message = vec![Template("<|im_start|>user")];
        for r in responses {
            message.extend([
                Template("\n<tool_response>\n"),
                Content(r.as_ref().trim()),
                Template("\n</tool_response>"),
            ]);
        }
        message.push(Template("<|im_end|>\n"));
        let prompt = self.turn_tokens(&message);
        self.answer(prompt, sampler, on_event, stop)
    }

    /// Process `prompt` and generate the answer
    fn answer(
        &mut self,
        prompt: Vec<TokenId>,
        sampler: &mut Sampler,
        mut on_event: impl FnMut(Event<'_>),
        stop: impl Fn() -> bool,
    ) -> Result<TurnStats, Error> {
        if self.session.n_past() + prompt.len() + 1 >= self.opts.context {
            return Err(Error::ContextFull {
                n_past: self.session.n_past(),
                prompt: prompt.len(),
                context: self.opts.context,
            });
        }
        self.pending.clear();

        let mut stats = TurnStats {
            prompt_tokens: prompt.len(),
            ..Default::default()
        };
        let (body, last) = prompt.split_at(prompt.len() - 1);
        self.prefill(body);
        let mut logits = self.session.forward(last);
        on_event(Event::PromptDone);

        let mut thinking = self.opts.thinking;
        let mut decoder = StreamDecoder::default();
        // the template puts "\n\n" after </think>: don't echo it
        let mut skip_leading_newlines = !thinking;
        // newlines at the end of the text so far, shown only if more follows
        let mut held_newlines = String::new();
        // the text of the function call being generated
        let mut call: Option<String> = None;
        let mut spec = Speculation::default();
        loop {
            let next = sampler.sample(&mut logits, &self.generated);
            stats.gen_tokens += 1;
            self.generated.push(next);

            if next == self.im_end || self.tok.eog.contains(&next) {
                if let Some(text) = call.take() {
                    on_event(Event::Answer(&format!("<tool_call>{text}")));
                }
                // the end marker closes the message in the history
                self.pending.push(self.im_end);
                break;
            }
            if next == self.think_end && thinking {
                thinking = false;
                skip_leading_newlines = true;
                held_newlines.clear();
                on_event(Event::ThinkingDone);
            } else if next == self.tool_call && !thinking && call.is_none() {
                // the newlines before a call are not part of the answer
                held_newlines.clear();
                call = Some(String::new());
            } else if next == self.tool_call_end && call.is_some() {
                let text = call.take().unwrap_or_default();
                match ToolCall::parse(&text) {
                    Some(c) => {
                        on_event(Event::ToolCall(&c));
                        stats.tool_calls.push(c);
                    }
                    // not a call after all: show it as it was written
                    None => on_event(Event::Answer(&format!("<tool_call>{text}</tool_call>"))),
                }
                skip_leading_newlines = true;
            } else {
                let mut text = decoder.push(self.tok.token_bytes(next));
                if let Some(c) = &mut call {
                    c.push_str(&text);
                } else {
                    if skip_leading_newlines {
                        let trimmed = text.trim_start_matches('\n');
                        if !trimmed.is_empty() {
                            skip_leading_newlines = false;
                        }
                        text = trimmed.to_string();
                    }
                    let body = text.trim_end_matches('\n');
                    if !body.is_empty() {
                        let shown = core::mem::take(&mut held_newlines) + body;
                        on_event(if thinking {
                            Event::Thinking(&shown)
                        } else {
                            Event::Answer(&shown)
                        });
                    }
                    held_newlines.push_str(&text[body.len()..]);
                }
            }

            let full = self.session.n_past() - spec.drafts.len() + 2 >= self.opts.context;
            if stats.gen_tokens >= self.opts.max_tokens || full || stop() {
                stats.truncated = true;
                // close the message so the history stays well-formed
                self.pending.push(next);
                if thinking {
                    self.pending.push(self.think_end);
                }
                if let Some(text) = call.take() {
                    on_event(Event::Answer(&format!("<tool_call>{text}")));
                    self.pending.push(self.tool_call_end);
                }
                self.pending.push(self.im_end);
                break;
            }
            logits = self.advance(next, &mut spec, &mut stats);
        }
        self.settle(&mut spec);
        Ok(stats)
    }

    /// The logits after `next`: those of the last verify if `next` is its
    /// next draft, else from a new draft and verify
    fn advance(
        &mut self,
        next: TokenId,
        spec: &mut Speculation,
        stats: &mut TurnStats,
    ) -> Vec<f32> {
        if let Some(draft) = spec.drafts.pop_front() {
            let logits = spec.logits.pop_front().expect("logits for every draft");
            if draft == next {
                spec.kept += 1;
                stats.accepted += 1;
                return logits;
            }
        }
        self.settle(spec);

        // room for the drafts in the context
        let room = self.session.n_past() + 2 + self.opts.draft < self.opts.context;
        let drafts = if room {
            self.session.draft(next)
        } else {
            Vec::new()
        };
        if drafts.is_empty() {
            return self.session.forward(&[next]);
        }
        stats.drafted += drafts.len();
        let tokens: Vec<TokenId> = core::iter::once(next)
            .chain(drafts.iter().copied())
            .collect();
        let mut logits = VecDeque::from(self.session.verify(&tokens));
        let first = logits.pop_front().expect("logits for every token");
        *spec = Speculation {
            drafts: drafts.into(),
            logits,
            kept: 1,
        };
        first
    }

    /// Keep the verified tokens of the last draft that matched what was sampled
    fn settle(&mut self, spec: &mut Speculation) {
        if spec.kept > 0 {
            self.session.keep(spec.kept);
        }
        *spec = Speculation::default();
    }
}

/// The last verify: its drafts not yet compared with the sampled tokens, the
/// logits after each, and its tokens kept so far
#[derive(Default)]
struct Speculation {
    drafts: VecDeque<TokenId>,
    logits: VecDeque<Vec<f32>>,
    kept: usize,
}

/// Placeholder session while a new one is being created
struct Empty;

impl Session for Empty {
    fn forward(&mut self, _: &[TokenId]) -> Vec<f32> {
        unreachable!()
    }
    fn prefill(&mut self, _: &[TokenId]) {
        unreachable!()
    }
    fn draft(&mut self, _: TokenId) -> Vec<TokenId> {
        unreachable!()
    }
    fn verify(&mut self, _: &[TokenId]) -> Vec<Vec<f32>> {
        unreachable!()
    }
    fn keep(&mut self, _: usize) {
        unreachable!()
    }
    fn n_past(&self) -> usize {
        0
    }
    fn state_bytes(&self) -> usize {
        0
    }
    fn snapshot(&self) -> Result<Snapshot, Error> {
        unreachable!()
    }
    fn restore(&mut self, _: &Snapshot) -> Result<(), Error> {
        unreachable!()
    }
}
