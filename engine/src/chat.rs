//! Multi-turn chat on top of the model, following the GGUF's chat template
//! (ChatML with `<think>` reasoning blocks).
//!
//! The template keeps past reasoning in the history (`preserve_thinking`
//! defaults to true), so every turn only appends to the sequence: the model
//! state is kept between turns and only the new tokens are processed. This
//! also suits the recurrent layers, whose state cannot be rewound (beyond the
//! last verify pass of speculative decoding).
//!
//! A turn runs to its end in [`Chat::send`], or one pass of the model at a
//! time: [`Chat::begin`] opens it, each [`Chat::step`] does one pass and gives
//! the tokens chosen after it, and [`Chat::cancel`] ends it between two steps.
//! A caller that runs the model on its own schedule asks for the next pass
//! only when it has taken the tokens of the last one.
//!
//! With tools, the model may answer with function calls ([`Event::ToolCall`]),
//! whose results go back with [`Chat::send_tool_responses`].
//!
//! A conversation can be saved at any point between turns and restored later,
//! in the same chat or another one on the same engine: [`Chat::save`] before the
//! first message saves the processed system prompt, so that conversations
//! starting with it don't process it again. Its token ids alone
//! ([`Chat::transcript`]) take a few bytes per token and restore in any chat on
//! a model with the same vocabulary, by processing them again
//! ([`Chat::restore_transcript`]).

use alloc::collections::VecDeque;

use crate::Error;
use crate::engine::{DraftLogits, Engine, Session, Snapshot};
use crate::memory::SessionMemory;
use crate::prelude::*;
use crate::sampler::Sampler;
use crate::tokenizer::{PromptPart, StreamDecoder, TokenId, Tokenizer};
pub use crate::transcript::Transcript;

use PromptPart::{Content, Template};

/// The chat template's text before the tool definitions in the system block
const TOOLS_HEADER: &str = "# Tools\n\nYou have access to the following functions:\n\n<tools>";
/// And after them
const TOOLS_FOOTER: &str = "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

/// The tokens that the close of a stopped turn adds after the context: the
/// last token chosen, the end of the open block (`</think>` or `</tool_call>`)
/// and `<|im_end|>`. One block at most is open, because a call opens only
/// after the reasoning block. A turn starts only with this room above its
/// prompt, and stops at the end of the context while this room is left, so
/// that the session holds the close before the next prompt.
pub const CLOSE_ROOM_TOKENS: usize = 3;

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
    /// tokens drafted by the MTP head per step, 0 for none, by default the
    /// engine's choice
    pub draft: Option<usize>,
    /// tokens proposed per step from earlier text of the conversation, at
    /// most, 0 for none: when what was just generated appeared before, the
    /// tokens that followed it (a quote, code being rewritten). The session
    /// holds a copy of the recurrent states per token of a step.
    pub lookup: usize,
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
            draft: None,
            lookup: 0,
            tools: Vec::new(),
        }
    }
}

/// Pieces of the answer, as they are generated
pub enum Event<'a> {
    /// the prompt is processed, generation starts
    PromptDone,
    /// a token of the answer, as soon as it is chosen and before its text:
    /// the ids of [`TurnStats::tokens`], in order, the end marker included
    /// when the model writes it
    Token(TokenId),
    Thinking(&'a str),
    Answer(&'a str),
    /// the reasoning block is complete
    ThinkingDone,
    /// a function call, to answer with [`Chat::send_tool_responses`]
    ToolCall(&'a ToolCall),
}

/// A message of an earlier conversation, for [`Chat::send_with_history`]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Message<'a> {
    User(&'a str),
    /// the answer, without its reasoning
    Assistant(&'a str),
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

/// Why a turn ended
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stop {
    /// the model wrote its end marker
    End,
    /// the answer reached `max_tokens`
    MaxTokens,
    /// the session reached the end of the context
    ContextFull,
    /// the caller ended the turn: [`Chat::cancel`], or the `stop` function
    /// of [`Chat::send`]
    Cancel,
}

#[derive(Debug, Clone)]
pub struct TurnStats {
    pub prompt_tokens: usize,
    pub gen_tokens: usize,
    /// why the turn ended
    pub stop: Stop,
    /// tokens drafted by the MTP head or proposed from the conversation, and
    /// how many of them were right
    pub drafted: usize,
    pub accepted: usize,
    /// of those, proposed from the conversation, and right
    pub looked_up: usize,
    pub looked_up_accepted: usize,
    /// the function calls of the answer, as given to [`Event::ToolCall`]
    pub tool_calls: Vec<ToolCall>,
    /// the tokens sampled for the answer, in order, the end marker included
    /// when the model wrote it
    pub tokens: Vec<TokenId>,
}

/// What one [`Chat::step`] did
#[derive(Debug, Clone)]
pub enum Step {
    /// a pass over a part of the prompt; more of it is left
    Prompt,
    /// the tokens chosen after the pass, in order; the turn goes on
    Tokens(Vec<TokenId>),
    /// the tokens chosen after the last pass, in order, and the end of the
    /// turn
    Done(Vec<TokenId>, TurnStats),
}

/// An open turn: the state between two steps
struct Turn {
    stats: TurnStats,
    phase: Phase,
}

enum Phase {
    /// the prompt, of which `done` tokens are processed; it starts with the
    /// `pending` tokens that the chat held before the turn
    Prompt {
        prompt: Vec<TokenId>,
        done: usize,
        pending: usize,
    },
    /// the answer, with `next` chosen and not yet processed
    Answer { answer: Answer, next: TokenId },
}

/// The state of an answer between two steps
struct Answer {
    thinking: bool,
    decoder: StreamDecoder,
    /// the template puts "\n\n" after </think>: don't echo it
    skip_leading_newlines: bool,
    /// newlines at the end of the text so far, shown only if more follows
    held_newlines: String,
    /// the text of the function call being generated
    call: Option<String>,
    spec: Speculation,
}

impl Answer {
    fn new(thinking: bool) -> Self {
        Answer {
            thinking,
            decoder: StreamDecoder::default(),
            skip_leading_newlines: !thinking,
            held_newlines: String::new(),
            call: None,
            spec: Speculation::default(),
        }
    }
}

/// A conversation up to some point, see [`Chat::save`]
pub struct Prefix {
    snapshot: Snapshot,
    pending: Vec<TokenId>,
    generated: Vec<TokenId>,
    history: Vec<TokenId>,
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

/// The draft length of a chat's session: room to verify the MTP head's drafts
/// and the proposals from the conversation
fn session_draft(engine: &dyn Engine, opts: &ChatOptions) -> usize {
    opts.draft
        .unwrap_or_else(|| engine.draft_size())
        .max(opts.lookup)
}

pub struct Chat<'m> {
    engine: &'m dyn Engine,
    tok: &'m Tokenizer,
    pub opts: ChatOptions,
    session: Box<dyn Session + 'm>,
    /// tokens not yet processed, fed before the next prompt: the end of the
    /// last answer, or the rest of a cancelled prompt
    pending: Vec<TokenId>,
    /// every generated token, for the sampler penalties
    generated: Vec<TokenId>,
    /// every token of the sequence, for the proposals from the conversation
    history: Vec<TokenId>,
    /// consecutive steps whose proposal from the conversation was all right
    lookup_run: usize,
    /// the turn between [`Chat::begin`] and its end
    turn: Option<Turn>,
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
        let session = engine.new_session(opts.context, session_draft(engine, &opts))?;
        Self::from_session(
            engine,
            tok,
            opts,
            session,
            Vec::new(),
            Vec::new(),
            Vec::new(),
        )
    }

    /// A chat that goes on from `session`, a session of `engine` made as
    /// [`Chat::new`] makes one (`opts.context`, and the draft size of `opts`
    /// or of the engine, at least `opts.lookup`), for example one rebuilt by
    /// repeating the session calls of another chat, or one whose calls are
    /// recorded. `pending` are the tokens generated but not yet processed,
    /// fed before the next prompt, `generated` every token generated so far,
    /// for the sampler penalties, and `history` every token of the
    /// conversation, for prompt lookup; all empty for a session that holds
    /// nothing.
    /// A session that holds tokens gets no system block: its system prompt,
    /// reasoning instructions and tools are those it holds, whatever `opts`
    /// says, as for [`Chat::restore`]. Fails with [`Error::ContextFull`] when
    /// the session and `pending` hold more tokens than `opts.context`.
    pub fn from_session(
        engine: &'m dyn Engine,
        tok: &'m Tokenizer,
        opts: ChatOptions,
        session: Box<dyn Session + 'm>,
        pending: Vec<TokenId>,
        generated: Vec<TokenId>,
        history: Vec<TokenId>,
    ) -> Result<Self, Error> {
        if session.n_past() + pending.len() > opts.context {
            return Err(Error::ContextFull {
                n_past: session.n_past(),
                prompt: pending.len(),
                context: opts.context,
            });
        }
        let id = |s: &str| tok.token_id(s).ok_or_else(|| Error::MissingToken(s.into()));
        Ok(Chat {
            engine,
            tok,
            session,
            opts,
            pending,
            generated,
            history,
            lookup_run: 0,
            turn: None,
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
            .new_session(self.opts.context, session_draft(self.engine, &self.opts))?;
        self.pending.clear();
        self.generated.clear();
        self.history.clear();
        self.lookup_run = 0;
        self.turn = None;
        Ok(())
    }

    pub fn n_past(&self) -> usize {
        self.session.n_past()
    }

    /// The memory of the conversation's session
    pub fn memory(&self) -> SessionMemory {
        self.session.memory()
    }

    /// The memory of the session a chat on `engine` with `opts` would make,
    /// without making it
    pub fn session_memory(engine: &dyn Engine, opts: &ChatOptions) -> SessionMemory {
        engine.session_memory(opts.context, session_draft(engine, opts))
    }

    /// Every token sampled in this conversation, in order: the history the
    /// sampler penalties look at
    pub fn generated(&self) -> &[TokenId] {
        &self.generated
    }

    /// Every token of the conversation, prompts and answers, in order: the
    /// text that prompt lookup searches
    pub fn history(&self) -> &[TokenId] {
        &self.history
    }

    /// The tokens generated but not yet processed, fed before the next
    /// prompt: the end of the last answer, which the session doesn't hold yet
    pub fn pending(&self) -> &[TokenId] {
        &self.pending
    }

    /// Save the conversation so far, to [`Chat::restore`] it in this chat or
    /// another on the same engine. Before the first message, the system prompt
    /// is processed now, so that the prefix holds it.
    pub fn save(&mut self) -> Result<Prefix, Error> {
        self.no_turn()?;
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
            self.history.extend_from_slice(&system);
        }
        Ok(Prefix {
            snapshot: self.session.snapshot()?,
            pending: self.pending.clone(),
            generated: self.generated.clone(),
            history: self.history.clone(),
        })
    }

    /// Go back to a saved conversation. Its system prompt, reasoning
    /// instructions and tools are those of the chat that saved it, whatever
    /// `opts` says.
    pub fn restore(&mut self, prefix: &Prefix) -> Result<(), Error> {
        self.no_turn()?;
        self.session.restore(&prefix.snapshot)?;
        self.pending.clone_from(&prefix.pending);
        self.generated.clone_from(&prefix.generated);
        self.history.clone_from(&prefix.history);
        self.lookup_run = 0;
        Ok(())
    }

    /// Tokens of the conversation so far: those processed, and the end of the
    /// last answer, processed with the next message
    pub fn n_tokens(&self) -> usize {
        self.history.len() + self.pending.len()
    }

    /// The conversation so far as token ids, to [`Chat::restore_transcript`]
    /// in any chat whose model has the same vocabulary. Nothing is processed:
    /// before the first message it is empty, unlike [`Chat::save`].
    pub fn transcript(&self) -> Transcript {
        let mut tokens = Vec::with_capacity(self.n_tokens());
        tokens.extend_from_slice(&self.history);
        tokens.extend_from_slice(&self.pending);
        Transcript {
            tokens,
            pending: self.pending.len(),
            generated: self.generated.clone(),
        }
    }

    /// Replace the conversation with `transcript`, processing its tokens
    /// again. Its system prompt, reasoning instructions and tools are those
    /// of the chat that made it, whatever `opts` says, as with
    /// [`Chat::restore`]. The answers that follow are those of the chat
    /// that made it up to rounding: its tokens were processed in other
    /// passes. Nothing changes when it doesn't fit.
    pub fn restore_transcript(&mut self, transcript: &Transcript) -> Result<(), Error> {
        self.no_turn()?;
        let Transcript {
            tokens,
            pending,
            generated,
        } = transcript;
        if *pending > tokens.len() {
            return Err(Error::Transcript("more pending tokens than tokens"));
        }
        if *pending > 0 && *pending == tokens.len() {
            return Err(Error::Transcript("pending tokens without a conversation"));
        }
        let n_vocab = self.tok.n_vocab();
        if tokens
            .iter()
            .chain(generated)
            .any(|&t| t as usize >= n_vocab)
        {
            return Err(Error::Transcript("token id past the vocabulary"));
        }
        // the processed tokens and those of the next message's end
        if tokens.len() >= self.opts.context {
            return Err(Error::ContextFull {
                n_past: 0,
                prompt: tokens.len(),
                context: self.opts.context,
            });
        }
        if self.session.n_past() > 0 {
            self.reset()?;
        }
        let (processed, rest) = transcript.split();
        self.prefill(processed);
        self.history = processed.to_vec();
        self.pending = rest.to_vec();
        self.generated.clone_from(generated);
        self.lookup_run = 0;
        Ok(())
    }

    /// Fails with [`Error::TurnOpen`] while a turn is open
    fn no_turn(&self) -> Result<(), Error> {
        match self.turn {
            Some(_) => Err(Error::TurnOpen),
            None => Ok(()),
        }
    }

    /// Tokens drafted by the MTP head per step
    fn draft(&self) -> usize {
        self.opts.draft.unwrap_or_else(|| self.engine.draft_size())
    }

    /// Prompt tokens processed per pass
    fn batch(&self) -> usize {
        self.opts
            .batch
            .unwrap_or_else(|| self.engine.batch_size())
            .max(1)
    }

    /// Append `tokens` to the sequence, in passes of the batch size
    fn prefill(&mut self, tokens: &[TokenId]) {
        for chunk in tokens.chunks(self.batch()) {
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
    /// `stop` is polled between the steps of the answer ([`Chat::step`]) to
    /// interrupt the generation.
    pub fn send(
        &mut self,
        user: &str,
        sampler: &mut Sampler,
        on_event: impl FnMut(Event<'_>),
        stop: impl Fn() -> bool,
    ) -> Result<TurnStats, Error> {
        self.begin(user)?;
        self.run(sampler, on_event, stop)
    }

    /// Append `history`, messages of an earlier conversation that this chat
    /// did not generate, then send `user` as [`Chat::send`] does. The chat
    /// template writes an earlier answer with an empty reasoning block, as it
    /// does for an answer given without its reasoning. This is how a client
    /// that keeps the conversation itself, and sends all of it with each
    /// message, continues it on a new chat.
    pub fn send_with_history(
        &mut self,
        history: &[Message<'_>],
        user: &str,
        sampler: &mut Sampler,
        on_event: impl FnMut(Event<'_>),
        stop: impl Fn() -> bool,
    ) -> Result<TurnStats, Error> {
        self.begin_with_history(history, user)?;
        self.run(sampler, on_event, stop)
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
        self.begin_tool_responses(responses)?;
        self.run(sampler, on_event, stop)
    }

    /// Start the turn of a user message, as [`Chat::send`] does, and give the
    /// answer to the caller one [`Chat::step`] at a time. Nothing is processed
    /// yet. Fails with [`Error::TurnOpen`] while a turn is open, and with
    /// [`Error::ContextFull`] when the prompt doesn't fit.
    pub fn begin(&mut self, user: &str) -> Result<(), Error> {
        self.begin_with_history(&[], user)
    }

    /// Start a turn as [`Chat::send_with_history`] does, to [`Chat::step`]
    pub fn begin_with_history(&mut self, history: &[Message<'_>], user: &str) -> Result<(), Error> {
        let mut message = Vec::new();
        for m in history {
            match *m {
                Message::User(text) => message.extend([
                    Template("<|im_start|>user\n"),
                    Content(text.trim()),
                    Template("<|im_end|>\n"),
                ]),
                Message::Assistant(text) => message.extend([
                    Template("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
                    Content(text.trim()),
                    Template("<|im_end|>\n"),
                ]),
            }
        }
        message.extend([
            Template("<|im_start|>user\n"),
            Content(user.trim()),
            Template("<|im_end|>\n"),
        ]);
        let prompt = self.turn_tokens(&message);
        self.begin_prompt(prompt)
    }

    /// Start a turn as [`Chat::send_tool_responses`] does, to [`Chat::step`]
    pub fn begin_tool_responses(&mut self, responses: &[impl AsRef<str>]) -> Result<(), Error> {
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
        self.begin_prompt(prompt)
    }

    /// Open the turn of `prompt`, which starts with the pending tokens
    fn begin_prompt(&mut self, prompt: Vec<TokenId>) -> Result<(), Error> {
        if self.turn.is_some() {
            return Err(Error::TurnOpen);
        }
        if self.session.n_past() + prompt.len() + CLOSE_ROOM_TOKENS > self.opts.context {
            return Err(Error::ContextFull {
                n_past: self.session.n_past(),
                prompt: prompt.len(),
                context: self.opts.context,
            });
        }
        let pending = self.pending.len();
        self.pending.clear();
        let stats = TurnStats {
            prompt_tokens: prompt.len(),
            gen_tokens: 0,
            // the step that ends the turn sets the reason
            stop: Stop::End,
            drafted: 0,
            accepted: 0,
            looked_up: 0,
            looked_up_accepted: 0,
            tool_calls: Vec::new(),
            tokens: Vec::new(),
        };
        self.turn = Some(Turn {
            stats,
            phase: Phase::Prompt {
                prompt,
                done: 0,
                pending,
            },
        });
        Ok(())
    }

    /// Step the open turn to its end, polling `stop` after each step that
    /// gives tokens
    fn run(
        &mut self,
        sampler: &mut Sampler,
        mut on_event: impl FnMut(Event<'_>),
        stop: impl Fn() -> bool,
    ) -> Result<TurnStats, Error> {
        loop {
            match self.step(sampler, &mut on_event)? {
                Step::Done(_, stats) => return Ok(stats),
                Step::Tokens(_) if stop() => return self.cancel(&mut on_event),
                Step::Prompt | Step::Tokens(_) => {}
            }
        }
    }

    /// Do one pass of the model for the open turn: a pass over the next part
    /// of the prompt, at most the batch size, or the pass after the last token
    /// chosen (a forward, or a draft and its verify). Then choose the tokens
    /// that the pass gives: one, or more when drafts are right. The events of
    /// these tokens go to `on_event`. Give each step of a turn the same
    /// sampler. Between two steps the caller can [`Chat::cancel`] the turn.
    /// Fails with [`Error::NoTurn`] when no turn is open. After an error the
    /// turn is closed, as [`Chat::send`] closes it.
    pub fn step(
        &mut self,
        sampler: &mut Sampler,
        mut on_event: impl FnMut(Event<'_>),
    ) -> Result<Step, Error> {
        let Turn { mut stats, phase } = self.turn.take().ok_or(Error::NoTurn)?;
        let (answer, logits) = match phase {
            Phase::Prompt {
                prompt,
                done,
                pending,
            } => {
                // every token of the prompt but the last, in passes of the
                // batch size, then a forward of the last
                let body = prompt.len() - 1;
                if done < body {
                    let end = body.min(done + self.batch());
                    self.session.prefill(&prompt[done..end]);
                    self.history.extend_from_slice(&prompt[done..end]);
                    self.turn = Some(Turn {
                        stats,
                        phase: Phase::Prompt {
                            prompt,
                            done: end,
                            pending,
                        },
                    });
                    return Ok(Step::Prompt);
                }
                let last = &prompt[body..];
                self.history.extend_from_slice(last);
                let logits = self.session.forward(last);
                on_event(Event::PromptDone);
                (Answer::new(self.opts.thinking), logits)
            }
            Phase::Answer { mut answer, next } => {
                let logits = self.advance(next, &mut answer.spec, &mut stats, sampler);
                (answer, logits)
            }
        };
        self.choose_tokens(stats, answer, logits, sampler, &mut on_event)
    }

    /// End the open turn before its end, as a `stop` of [`Chat::send`] does:
    /// the message is closed so the history stays well-formed. During the
    /// prompt, the rest of it is processed with the next message; before the
    /// first step, the chat is as it was before the turn. Fails with
    /// [`Error::NoTurn`] when no turn is open.
    pub fn cancel(&mut self, mut on_event: impl FnMut(Event<'_>)) -> Result<TurnStats, Error> {
        let Turn { mut stats, phase } = self.turn.take().ok_or(Error::NoTurn)?;
        stats.stop = Stop::Cancel;
        match phase {
            Phase::Prompt {
                prompt,
                done: 0,
                pending,
            } => self.pending.extend_from_slice(&prompt[..pending]),
            Phase::Prompt { prompt, done, .. } => {
                self.pending.extend_from_slice(&prompt[done..]);
                self.close_message(self.opts.thinking, false);
            }
            Phase::Answer { mut answer, next } => {
                self.close_answer(&mut answer, next, &mut on_event);
                self.settle(&mut answer.spec);
            }
        }
        Ok(stats)
    }

    /// Choose tokens from `logits`, the logits after the last pass, and from
    /// those of its right drafts, until the next token needs a pass or the
    /// turn ends
    fn choose_tokens(
        &mut self,
        mut stats: TurnStats,
        mut a: Answer,
        mut logits: Vec<f32>,
        sampler: &mut Sampler,
        on_event: &mut impl FnMut(Event<'_>),
    ) -> Result<Step, Error> {
        let first = stats.tokens.len();
        loop {
            let next = match self.choose(&mut logits, &a.spec, sampler) {
                Ok(next) => next,
                Err(e) => {
                    // the tokens so far are in the session: close the
                    // message, as a stop does, so the history stays
                    // well-formed for the next turn
                    self.close_message(a.thinking, a.call.is_some());
                    self.settle(&mut a.spec);
                    return Err(e);
                }
            };
            stats.gen_tokens += 1;
            stats.tokens.push(next);
            self.generated.push(next);
            sampler.accept(next);
            on_event(Event::Token(next));

            if next == self.im_end || self.tok.eog.contains(&next) {
                if let Some(text) = a.call.take() {
                    on_event(Event::Answer(&format!("<tool_call>{text}")));
                }
                // the end marker closes the message in the history
                self.pending.push(self.im_end);
                stats.stop = Stop::End;
                return Ok(self.finish(stats, a, first));
            }
            if next == self.think_end && a.thinking {
                a.thinking = false;
                a.skip_leading_newlines = true;
                a.held_newlines.clear();
                on_event(Event::ThinkingDone);
            } else if next == self.tool_call && !a.thinking && a.call.is_none() {
                // the newlines before a call are not part of the answer
                a.held_newlines.clear();
                a.call = Some(String::new());
            } else if next == self.tool_call_end && a.call.is_some() {
                let text = a.call.take().unwrap_or_default();
                match ToolCall::parse(&text) {
                    Some(c) => {
                        on_event(Event::ToolCall(&c));
                        stats.tool_calls.push(c);
                    }
                    // not a call after all: show it as it was written
                    None => on_event(Event::Answer(&format!("<tool_call>{text}</tool_call>"))),
                }
                a.skip_leading_newlines = true;
            } else {
                let mut text = a.decoder.push(self.tok.token_bytes(next));
                if let Some(c) = &mut a.call {
                    c.push_str(&text);
                } else {
                    if a.skip_leading_newlines {
                        let trimmed = text.trim_start_matches('\n');
                        if !trimmed.is_empty() {
                            a.skip_leading_newlines = false;
                        }
                        text = trimmed.to_string();
                    }
                    let body = text.trim_end_matches('\n');
                    if !body.is_empty() {
                        let shown = core::mem::take(&mut a.held_newlines) + body;
                        on_event(if a.thinking {
                            Event::Thinking(&shown)
                        } else {
                            Event::Answer(&shown)
                        });
                    }
                    a.held_newlines.push_str(&text[body.len()..]);
                }
            }

            let full = self.session.n_past() - a.spec.drafts.len() + CLOSE_ROOM_TOKENS
                >= self.opts.context;
            // the first reason that holds names the end, in this order
            let reason = if stats.gen_tokens >= self.opts.max_tokens {
                Some(Stop::MaxTokens)
            } else if full {
                Some(Stop::ContextFull)
            } else {
                None
            };
            if let Some(reason) = reason {
                stats.stop = reason;
                self.close_answer(&mut a, next, on_event);
                return Ok(self.finish(stats, a, first));
            }
            // a right draft has its logits already: no pass is needed
            if a.spec.drafts.front() == Some(&next) {
                logits = self.advance(next, &mut a.spec, &mut stats, sampler);
                continue;
            }
            let tokens = stats.tokens[first..].to_vec();
            self.turn = Some(Turn {
                stats,
                phase: Phase::Answer { answer: a, next },
            });
            return Ok(Step::Tokens(tokens));
        }
    }

    /// The last step of a turn, whose tokens start at `first` in `stats`
    fn finish(&mut self, stats: TurnStats, mut a: Answer, first: usize) -> Step {
        self.settle(&mut a.spec);
        Step::Done(stats.tokens[first..].to_vec(), stats)
    }

    /// Close the answer of a stopped turn into `pending`: `next`, the last
    /// token chosen and not yet processed, then the ends of the open blocks
    fn close_answer(
        &mut self,
        a: &mut Answer,
        next: TokenId,
        on_event: &mut impl FnMut(Event<'_>),
    ) {
        self.pending.push(next);
        let call_open = a.call.is_some();
        if let Some(text) = a.call.take() {
            on_event(Event::Answer(&format!("<tool_call>{text}")));
        }
        self.close_message(a.thinking, call_open);
    }

    /// Close an open message into `pending`: the end of the reasoning block
    /// while `thinking`, the end of a call while `call_open`, then the end of
    /// the message
    fn close_message(&mut self, thinking: bool, call_open: bool) {
        if thinking {
            self.pending.push(self.think_end);
        }
        if call_open {
            self.pending.push(self.tool_call_end);
        }
        self.pending.push(self.im_end);
    }

    /// The token after the sequence, from `logits`, the logits after it. With
    /// a draft drawn from a distribution q (the MTP head's, while sampling),
    /// the draft is taken with a chance of p/q, else a token is drawn from
    /// max(p - q, 0) (speculative sampling): either way the token follows the
    /// sampler's distribution p. A draft of the likeliest token, or one from
    /// the conversation, is right when the sampled token is the same.
    fn choose(
        &self,
        logits: &mut [f32],
        spec: &Speculation,
        sampler: &mut Sampler,
    ) -> Result<TokenId, Error> {
        let (Some(&draft), Some(Some(q))) = (spec.drafts.front(), spec.q.front()) else {
            return sampler.sample(logits, &self.generated);
        };
        let p = normalized(sampler.distribution(logits, &self.generated)?);
        Ok(speculative_choice(&p, q, draft, sampler))
    }

    /// The logits after `next`: those of the last verify if `next` is its
    /// next draft, else from a new draft and verify
    fn advance(
        &mut self,
        next: TokenId,
        spec: &mut Speculation,
        stats: &mut TurnStats,
        sampler: &mut Sampler,
    ) -> Vec<f32> {
        self.history.push(next);
        if let Some(draft) = spec.drafts.pop_front() {
            spec.q.pop_front();
            let logits = spec.logits.pop_front().expect("logits for every draft");
            if draft == next {
                spec.kept += 1;
                stats.accepted += 1;
                if spec.looked_up {
                    stats.looked_up_accepted += 1;
                }
                return logits;
            }
        }
        self.settle(spec);

        // room for the drafts in the context
        let room = self.opts.context.saturating_sub(self.session.n_past() + 2);
        let (drafts, q, looked_up) = self.propose(next, room, sampler);
        if drafts.is_empty() {
            return self.session.forward(&[next]);
        }
        stats.drafted += drafts.len();
        if looked_up {
            stats.looked_up += drafts.len();
        }
        let tokens: Vec<TokenId> = core::iter::once(next)
            .chain(drafts.iter().copied())
            .collect();
        let mut logits = VecDeque::from(self.session.verify(&tokens));
        let first = logits.pop_front().expect("logits for every token");
        *spec = Speculation {
            proposed: drafts.len(),
            drafts: drafts.into(),
            q: q.into(),
            logits,
            kept: 1,
            looked_up,
        };
        first
    }

    /// The tokens to verify after `next`, at most `room`, the distributions
    /// they were drawn from (none for the likeliest token or one from the
    /// conversation), and whether they come from the conversation rather than
    /// the MTP head. A continuation found
    /// after a long match is taken alone; after a short one, only if the MTP
    /// head drafts its first tokens too. It is cut to `LOOKUP_SHORT` tokens
    /// unless the last steps' continuations were all right: a copy is running.
    fn propose(
        &mut self,
        next: TokenId,
        room: usize,
        sampler: &mut Sampler,
    ) -> (Vec<TokenId>, Vec<Option<Dist>>, bool) {
        // a draft passes no mask: while one is set, each step chooses one
        // token through it
        if sampler.masked() {
            return (Vec::new(), Vec::new(), false);
        }
        let mut found = if self.opts.lookup > 0 {
            continuation(&self.history, self.opts.lookup)
        } else {
            None
        };
        let long = if self.lookup_run >= LOOKUP_RUN {
            self.opts.lookup
        } else {
            self.opts.lookup.min(LOOKUP_SHORT)
        };
        if let Some((_, c)) = &mut found {
            c.truncate(long.min(room));
        }
        if let Some((m, c)) = &found
            && *m >= LOOKUP_STRONG
            && !c.is_empty()
        {
            let c = found.unwrap().1;
            return (c, Vec::new(), true);
        }
        let mut q = Vec::new();
        let draft = self.draft();
        let drafts = if draft > 0 {
            let greedy = sampler.greedy();
            self.session.draft(next, draft.min(room), &mut |l| {
                if greedy {
                    return l.argmax();
                }
                let d = normalized(sampler.distribution_of(l.candidates()));
                let t = sampler.draw(&d);
                q.push(Some(d));
                t
            })
        } else {
            Vec::new()
        };
        match found {
            Some((_, c))
                if c.len() >= LOOKUP_AGREE
                    && drafts.len() >= LOOKUP_AGREE
                    && c[..LOOKUP_AGREE] == drafts[..LOOKUP_AGREE] =>
            {
                (c, Vec::new(), true)
            }
            _ => (drafts, q, false),
        }
    }

    /// Keep the verified tokens of the last draft that matched what was sampled
    fn settle(&mut self, spec: &mut Speculation) {
        if spec.kept > 0 {
            self.session.keep(spec.kept);
        }
        if spec.looked_up && spec.kept == spec.proposed + 1 {
            self.lookup_run += 1;
        } else if spec.kept > 0 {
            self.lookup_run = 0;
        }
        *spec = Speculation::default();
    }
}

/// (token, probability) pairs
type Dist = Vec<(TokenId, f32)>;

/// `draft`, drawn from `q`, with a chance of p/q, else a token drawn from
/// max(p - q, 0): a token drawn from `p` (both normalized)
fn speculative_choice(
    p: &[(TokenId, f32)],
    q: &[(TokenId, f32)],
    draft: TokenId,
    sampler: &mut Sampler,
) -> TokenId {
    let prob = |d: &[(TokenId, f32)], t| d.iter().find(|c| c.0 == t).map_or(0.0, |c| c.1);
    if sampler.uniform() * prob(q, draft) < prob(p, draft) {
        return draft;
    }
    let residual: Dist = p
        .iter()
        .map(|&(t, pt)| (t, (pt - prob(q, t)).max(0.0)))
        .filter(|c| c.1 > 0.0)
        .collect();
    if residual.is_empty() {
        sampler.draw(p)
    } else {
        sampler.draw(&residual)
    }
}

/// `d` scaled to a sum of 1
fn normalized(mut d: Dist) -> Dist {
    let sum: f32 = d.iter().map(|c| c.1).sum();
    for c in &mut d {
        c.1 /= sum;
    }
    d
}

/// A match of at least this many tokens is proposed without the MTP head's
/// agreement, and one of fewer is not proposed at all (`LOOKUP_MIN`)
const LOOKUP_STRONG: usize = 8;
const LOOKUP_MIN: usize = 3;
/// Leading tokens a continuation and the MTP head's drafts must share to
/// propose the continuation after a short match
const LOOKUP_AGREE: usize = 2;
/// Tokens proposed from the conversation per step, unless the last
/// `LOOKUP_RUN` steps' proposals were all right
const LOOKUP_SHORT: usize = 7;
const LOOKUP_RUN: usize = 2;
/// Longest match looked for
const LOOKUP_MAX_MATCH: usize = 32;

/// What followed the latest earlier occurrence of the longest suffix of
/// `history` (at least `LOOKUP_MIN` tokens long), up to `n` tokens, and the
/// suffix length. The occurrence may overlap the suffix: a repeating pattern
/// continues with its own period.
fn continuation(history: &[TokenId], n: usize) -> Option<(usize, Vec<TokenId>)> {
    let len = history.len();
    let last = *history.last()?;
    let (mut best, mut at) = (0, 0);
    for j in (0..len - 1).rev() {
        if history[j] != last {
            continue;
        }
        let m = (1..LOOKUP_MAX_MATCH.min(j + 1))
            .take_while(|&k| history[j - k] == history[len - 1 - k])
            .count()
            + 1;
        if m > best {
            (best, at) = (m, j);
            if m == LOOKUP_MAX_MATCH {
                break;
            }
        }
    }
    if best < LOOKUP_MIN {
        return None;
    }
    // history followed by the continuation, where position len + i repeats j + 1 + i
    let mut c: Vec<TokenId> = Vec::with_capacity(n);
    for i in 0..n {
        let p = at + 1 + i;
        c.push(if p < len { history[p] } else { c[p - len] });
    }
    Some((best, c))
}

/// The last verify: its drafts not yet compared with the sampled tokens, the
/// logits after each, and its tokens kept so far
#[derive(Default)]
struct Speculation {
    drafts: VecDeque<TokenId>,
    /// the distribution each draft was drawn from, if not a single token
    q: VecDeque<Option<Dist>>,
    logits: VecDeque<Vec<f32>>,
    kept: usize,
    /// drafts of the verify, and whether they come from the conversation
    proposed: usize,
    looked_up: bool,
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
    fn draft(
        &mut self,
        _: TokenId,
        _: usize,
        _: &mut dyn FnMut(&DraftLogits<'_>) -> TokenId,
    ) -> Vec<TokenId> {
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
    fn memory(&self) -> SessionMemory {
        SessionMemory::default()
    }
    fn snapshot(&self) -> Result<Snapshot, Error> {
        unreachable!()
    }
    fn restore(&mut self, _: &Snapshot) -> Result<(), Error> {
        unreachable!()
    }
}

#[cfg(test)]
mod tests {
    use super::{continuation, speculative_choice};
    use crate::prelude::*;
    use crate::sampler::{Sampler, SamplerConfig};

    #[test]
    fn speculative_choice_follows_the_target() {
        // q overlaps p partly, puts mass on a token p excludes, misses one p has
        let p = [(0, 0.5), (1, 0.3), (2, 0.2)];
        let q = [(0, 0.2), (1, 0.6), (3, 0.2)];
        let mut s = Sampler::new(SamplerConfig {
            seed: 7,
            ..Default::default()
        });
        let n = 400_000;
        let mut count = [0usize; 4];
        let mut accepted = 0;
        for _ in 0..n {
            let d = s.draw(&q);
            let t = speculative_choice(&p, &q, d, &mut s);
            accepted += (t == d) as usize;
            count[t as usize] += 1;
        }
        for &(t, pt) in p.iter() {
            let f = count[t as usize] as f64 / n as f64;
            assert!((f - pt as f64).abs() < 0.005, "token {t}: {f} vs {pt}");
        }
        assert_eq!(count[3], 0);
        // the acceptance is the overlap: min(p, q) summed = 0.2 + 0.3
        let a = accepted as f64 / n as f64;
        assert!((a - 0.5).abs() < 0.005, "acceptance {a}");
    }

    #[test]
    fn continuation_of_the_longest_latest_match() {
        // "1 2 3 4 5" then "9 2 3 4 5 6", now "... 3 4 5": the longer match wins
        let h = [7, 1, 2, 3, 4, 5, 8, 9, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5];
        assert_eq!(continuation(&h, 2), Some((5, vec![8, 9])));
        // too short a match
        assert_eq!(continuation(&[1, 2, 3, 9, 2, 3], 2), None);
        assert_eq!(continuation(&[5], 2), None);
    }

    #[test]
    fn continuation_overlapping_its_match() {
        // a period of 2: "a b a b a" continues "b a b"
        let h = [0, 1, 0, 1, 0];
        assert_eq!(continuation(&h, 3), Some((3, vec![1, 0, 1])));
        // a run of one token
        assert_eq!(continuation(&[4, 4, 4, 4], 3), Some((3, vec![4, 4, 4])));
    }
}
