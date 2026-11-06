//! The close of a turn that stops at the end of the context. Only the
//! tokenizer comes from the model file; a scripted engine chooses each token,
//! so the tests need no GPU, but they are ignored without a model file:
//! `BIV_TEST_MODEL=<gguf> cargo test --release --test chat_close -- --ignored`

use biv::Error;
use biv::chat::{CLOSE_ROOM_TOKENS, Chat, ChatOptions, Event, Stop, TurnStats};
use biv::engine::{DraftLogits, Engine, Session, Snapshot};
use biv::sampler::{Sampler, SamplerConfig};
use biv::tokenizer::{TokenId, Tokenizer};

/// The variable that names the GGUF file
const MODEL_ENV: &str = "BIV_TEST_MODEL";
const MESSAGE: &str = "What time is it?";
/// A text that the tokenizer gives as one token
const WORD: &str = " word";
/// A function that the model can call
const TOOL: &str = r#"{"type": "function", "function": {"name": "get_time", "description": "The time now", "parameters": {"type": "object", "properties": {}}}}"#;
/// The context of the turn that measures the prompt: far above any prompt
const WIDE_CONTEXT: usize = 4096;
/// The answer of the turn that measures the prompt: one token
const PROBE_TOKENS: usize = 1;
/// The tokens that the scripted engine can choose in one session, more than
/// any turn of these tests generates
const SCRIPT_TOKENS: usize = 16;
/// The per-answer cap, above the script, so that only the context stops a turn
const MAX_TOKENS: usize = SCRIPT_TOKENS + 1;
/// Prompt tokens per prefill of the scripted engine
const SCRIPT_BATCH: usize = 64;
/// The scripted engine drafts nothing
const NO_DRAFT: usize = 0;
/// The tokens that a turn generates before the context leaves only the room
/// of the close
const STEPS_BEFORE_THE_CLOSE: usize = 2;
/// A greedy sampler: the one-hot logits of the script choose each token
const GREEDY: f32 = 0.0;
/// The logit of the scripted token; every other token has 0
const CHOSEN_LOGIT: f32 = 1.0;

fn model_path() -> String {
    std::env::var(MODEL_ENV).unwrap_or_else(|_| panic!("{MODEL_ENV} must name a GGUF file"))
}

/// An engine whose sessions choose the tokens of `script`, in order: the
/// logits of each forward put the next token of the script first
struct Script {
    tokens: Vec<TokenId>,
    vocab: usize,
}

struct ScriptSession<'s> {
    script: &'s Script,
    n_past: usize,
    forwards: usize,
}

impl Script {
    /// A session that holds `n_past` tokens, as a replay would rebuild one
    fn session_at(&self, n_past: usize) -> Box<dyn Session + '_> {
        Box::new(ScriptSession {
            script: self,
            n_past,
            forwards: 0,
        })
    }
}

impl Engine for Script {
    fn new_session(&self, _context: usize, _draft: usize) -> Result<Box<dyn Session + '_>, Error> {
        Ok(self.session_at(0))
    }

    fn batch_size(&self) -> usize {
        SCRIPT_BATCH
    }

    fn draft_size(&self) -> usize {
        NO_DRAFT
    }

    fn description(&self) -> String {
        "script".into()
    }

    fn numerics(&self) -> Vec<(&'static str, String)> {
        Vec::new()
    }
}

impl Session for ScriptSession<'_> {
    fn forward(&mut self, tokens: &[TokenId]) -> Vec<f32> {
        self.n_past += tokens.len();
        let token = *self
            .script
            .tokens
            .get(self.forwards)
            .expect("the script holds a token for each forward");
        self.forwards += 1;
        let mut logits = vec![0.0; self.script.vocab];
        logits[token as usize] = CHOSEN_LOGIT;
        logits
    }

    fn prefill(&mut self, tokens: &[TokenId]) {
        self.n_past += tokens.len();
    }

    fn draft(
        &mut self,
        _next: TokenId,
        _n: usize,
        _pick: &mut dyn FnMut(&DraftLogits<'_>) -> TokenId,
    ) -> Vec<TokenId> {
        Vec::new()
    }

    fn verify(&mut self, _tokens: &[TokenId]) -> Vec<Vec<f32>> {
        unreachable!("a session that drafts nothing verifies nothing")
    }

    fn keep(&mut self, _n: usize) {
        unreachable!("a session that verifies nothing keeps nothing")
    }

    fn n_past(&self) -> usize {
        self.n_past
    }

    fn state_bytes(&self) -> usize {
        0
    }

    fn snapshot(&self) -> Result<Snapshot, Error> {
        unreachable!("these tests take no snapshot")
    }

    fn restore(&mut self, _snapshot: &Snapshot) -> Result<(), Error> {
        unreachable!("these tests restore no snapshot")
    }
}

fn options(context: usize, thinking: bool, tools: Vec<String>) -> ChatOptions {
    ChatOptions {
        system: None,
        thinking,
        context,
        max_tokens: MAX_TOKENS,
        draft: Some(NO_DRAFT),
        lookup: 0,
        tools,
        ..ChatOptions::default()
    }
}

fn sampler() -> Sampler {
    Sampler::new(SamplerConfig {
        temperature: GREEDY,
        ..SamplerConfig::default()
    })
}

fn send(chat: &mut Chat<'_>) -> Result<TurnStats, Error> {
    chat.send(MESSAGE, &mut sampler(), |_: Event<'_>| {}, || false)
}

fn id(tok: &Tokenizer, text: &str) -> TokenId {
    tok.token_id(text)
        .unwrap_or_else(|| panic!("the tokenizer has {text}"))
}

fn word(tok: &Tokenizer) -> TokenId {
    let ids = tok.encode(WORD);
    assert_eq!(ids.len(), 1, "{WORD:?} is one token");
    ids[0]
}

/// The prompt tokens of the first turn with `thinking` and `tools`
fn prompt_tokens(engine: &Script, tok: &Tokenizer, thinking: bool, tools: &[String]) -> usize {
    let opts = ChatOptions {
        max_tokens: PROBE_TOKENS,
        ..options(WIDE_CONTEXT, thinking, tools.to_vec())
    };
    let mut chat = Chat::new(engine, tok, opts).expect("chat");
    send(&mut chat).expect("turn").prompt_tokens
}

/// One turn of `script` in a context with STEPS_BEFORE_THE_CLOSE steps above
/// the prompt and the room of the close. The turn stops at the end of the
/// context with its close in the context, and a chat goes on from it.
fn close_at_the_end_of_the_context(
    tok: &Tokenizer,
    script: Vec<TokenId>,
    thinking: bool,
    tools: Vec<String>,
) -> Vec<TokenId> {
    let engine = Script {
        tokens: script,
        vocab: tok.n_vocab(),
    };
    let prompt = prompt_tokens(&engine, tok, thinking, &tools);
    let context = prompt + STEPS_BEFORE_THE_CLOSE + CLOSE_ROOM_TOKENS;
    let opts = options(context, thinking, tools);
    let mut chat = Chat::new(&engine, tok, opts.clone()).expect("chat");
    let stats = send(&mut chat).expect("turn");
    assert_eq!(stats.stop, Stop::ContextFull);
    // a step for each token below the room of the close, and the last token
    assert_eq!(stats.gen_tokens, STEPS_BEFORE_THE_CLOSE + 1);
    assert_eq!(chat.n_past() + chat.pending().len(), context);

    let pending = chat.pending().to_vec();
    let session = engine.session_at(chat.n_past());
    Chat::from_session(
        &engine,
        tok,
        opts,
        session,
        pending.clone(),
        chat.generated().to_vec(),
        chat.history().to_vec(),
    )
    .expect("a chat goes on from the turn");
    pending
}

#[test]
#[ignore = "needs a model file"]
fn a_turn_that_fills_the_context_while_thinking_keeps_its_close_in_the_context() {
    let gguf = biv_cli::open_gguf(model_path()).expect("gguf");
    let tok = Tokenizer::from_gguf(&gguf).expect("tokenizer");
    let script = vec![word(&tok); SCRIPT_TOKENS];
    let pending = close_at_the_end_of_the_context(&tok, script, true, Vec::new());
    assert_eq!(pending[1..], [id(&tok, "</think>"), id(&tok, "<|im_end|>")]);
}

#[test]
#[ignore = "needs a model file"]
fn a_turn_that_fills_the_context_in_a_call_keeps_its_close_in_the_context() {
    let gguf = biv_cli::open_gguf(model_path()).expect("gguf");
    let tok = Tokenizer::from_gguf(&gguf).expect("tokenizer");
    let mut script = vec![id(&tok, "<tool_call>")];
    script.resize(SCRIPT_TOKENS, word(&tok));
    let pending = close_at_the_end_of_the_context(&tok, script, false, vec![TOOL.into()]);
    assert_eq!(
        pending[1..],
        [id(&tok, "</tool_call>"), id(&tok, "<|im_end|>")]
    );
}

/// A call opens only after the reasoning block: a <tool_call> while thinking
/// is text of the reasoning, so a close holds one block end at most
#[test]
#[ignore = "needs a model file"]
fn a_call_does_not_open_while_thinking() {
    let gguf = biv_cli::open_gguf(model_path()).expect("gguf");
    let tok = Tokenizer::from_gguf(&gguf).expect("tokenizer");
    let mut script = vec![id(&tok, "<tool_call>")];
    script.resize(SCRIPT_TOKENS, word(&tok));
    let pending = close_at_the_end_of_the_context(&tok, script, true, vec![TOOL.into()]);
    assert_eq!(pending[1..], [id(&tok, "</think>"), id(&tok, "<|im_end|>")]);
}

/// A prompt that leaves less than the room of the close is refused, and a
/// prompt that leaves the room gives one token and its close
#[test]
#[ignore = "needs a model file"]
fn a_prompt_needs_the_room_of_the_close() {
    let gguf = biv_cli::open_gguf(model_path()).expect("gguf");
    let tok = Tokenizer::from_gguf(&gguf).expect("tokenizer");
    let engine = Script {
        tokens: vec![word(&tok); SCRIPT_TOKENS],
        vocab: tok.n_vocab(),
    };
    let prompt = prompt_tokens(&engine, &tok, true, &[]);

    let short = options(prompt + CLOSE_ROOM_TOKENS - 1, true, Vec::new());
    let mut chat = Chat::new(&engine, &tok, short).expect("chat");
    assert!(matches!(send(&mut chat), Err(Error::ContextFull { .. })));

    let context = prompt + CLOSE_ROOM_TOKENS;
    let room = options(context, true, Vec::new());
    let mut chat = Chat::new(&engine, &tok, room).expect("chat");
    let stats = send(&mut chat).expect("turn");
    assert_eq!(stats.stop, Stop::ContextFull);
    assert_eq!(stats.gen_tokens, 1);
    assert_eq!(chat.n_past() + chat.pending().len(), context);
}
