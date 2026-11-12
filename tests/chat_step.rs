//! A turn driven one pass at a time: Chat::begin, Chat::step and
//! Chat::cancel. Only the tokenizer comes from the model file; a scripted
//! engine chooses each token, so the tests need no GPU, but they are ignored
//! without a model file:
//! `BIV_TEST_MODEL=<gguf> cargo test --release --test chat_step -- --ignored`

mod common;

use biv::Error;
use biv::chat::{Chat, ChatOptions, Event, Step, Stop, TurnStats};
use biv::tokenizer::{TokenId, Tokenizer};
use common::{NO_DRAFT, Script, model_path, sampler};

/// A message whose prompt is longer than the batch, so that the prompt takes
/// more than one step
const LONG_MESSAGE_WORDS: usize = 200;
/// A text that the tokenizer gives as one token
const WORD: &str = " word";
/// The tokens that the scripted engine can choose in one session
const SCRIPT_TOKENS: usize = 16;
/// The per-answer cap, below the script, so that it stops each turn
const MAX_TOKENS: usize = 6;
/// The prompt tokens of each pass of these tests
const BATCH: usize = 32;
/// A context far above any turn of these tests
const CONTEXT: usize = 4096;

fn options(thinking: bool) -> ChatOptions {
    ChatOptions {
        system: None,
        thinking,
        context: CONTEXT,
        max_tokens: MAX_TOKENS,
        batch: Some(BATCH),
        draft: Some(NO_DRAFT),
        lookup: 0,
        ..ChatOptions::default()
    }
}

fn message() -> String {
    WORD.repeat(LONG_MESSAGE_WORDS)
}

fn word(tok: &Tokenizer) -> TokenId {
    let ids = tok.encode(WORD);
    assert_eq!(ids.len(), 1, "{WORD:?} is one token");
    ids[0]
}

fn id(tok: &Tokenizer, text: &str) -> TokenId {
    tok.token_id(text)
        .unwrap_or_else(|| panic!("the tokenizer has {text}"))
}

/// The tokens and steps of a turn stepped to its end
struct Stepped {
    stats: TurnStats,
    prompt_steps: usize,
    tokens: Vec<Vec<TokenId>>,
}

fn step_to_the_end(chat: &mut Chat<'_>) -> Stepped {
    let mut s = sampler();
    let mut prompt_steps = 0;
    let mut tokens = Vec::new();
    loop {
        match chat.step(&mut s, |_: Event<'_>| {}).expect("step") {
            Step::Prompt => prompt_steps += 1,
            Step::Tokens(t) => tokens.push(t),
            Step::Done(t, stats) => {
                tokens.push(t);
                return Stepped {
                    stats,
                    prompt_steps,
                    tokens,
                };
            }
        }
    }
}

/// A turn of steps is the turn of send: the same calls of the session, the
/// same tokens and the same chat after it. The prompt takes a step for each
/// batch of its tokens but the last, and with no drafts each step of the
/// answer chooses one token.
#[test]
#[ignore = "needs a model file"]
fn a_stepped_turn_is_the_turn_of_send() {
    let gguf = biv_cli::open_gguf(model_path()).expect("gguf");
    let tok = Tokenizer::from_gguf(&gguf).expect("tokenizer");
    let script = vec![word(&tok); SCRIPT_TOKENS];

    let sent_engine = Script::new(script.clone(), tok.n_vocab());
    let mut sent = Chat::new(&sent_engine, &tok, options(true)).expect("chat");
    let by_send = sent
        .send(&message(), &mut sampler(), |_: Event<'_>| {}, || false)
        .expect("turn");

    let stepped_engine = Script::new(script, tok.n_vocab());
    let mut chat = Chat::new(&stepped_engine, &tok, options(true)).expect("chat");
    chat.begin(&message()).expect("begin");
    let stepped = step_to_the_end(&mut chat);

    assert_eq!(stepped.stats.stop, Stop::MaxTokens);
    assert_eq!(stepped.stats.tokens, by_send.tokens);
    assert_eq!(stepped.stats.prompt_tokens, by_send.prompt_tokens);
    assert_eq!(*stepped_engine.fed.borrow(), *sent_engine.fed.borrow());
    assert_eq!(chat.pending(), sent.pending());
    assert_eq!(chat.history(), sent.history());
    assert_eq!(chat.generated(), sent.generated());

    assert_eq!(
        stepped.prompt_steps,
        (by_send.prompt_tokens - 1).div_ceil(BATCH)
    );
    assert_eq!(stepped.tokens.len(), MAX_TOKENS);
    assert!(stepped.tokens.iter().all(|t| t.len() == 1));
    assert_eq!(stepped.tokens.concat(), by_send.tokens);
}

/// A cancel before the first step leaves the chat as it was: the next turn
/// is the turn of a new chat
#[test]
#[ignore = "needs a model file"]
fn a_cancel_before_the_first_step_leaves_the_chat_as_it_was() {
    let gguf = biv_cli::open_gguf(model_path()).expect("gguf");
    let tok = Tokenizer::from_gguf(&gguf).expect("tokenizer");
    let script = vec![word(&tok); SCRIPT_TOKENS];

    let engine = Script::new(script.clone(), tok.n_vocab());
    let mut chat = Chat::new(&engine, &tok, options(true)).expect("chat");
    chat.begin(&message()).expect("begin");
    let stats = chat.cancel(|_: Event<'_>| {}).expect("cancel");
    assert_eq!(stats.stop, Stop::Cancel);
    assert_eq!(stats.gen_tokens, 0);
    assert_eq!(chat.n_past(), 0);
    assert!(chat.pending().is_empty());
    assert!(chat.history().is_empty());
    assert!(engine.fed.borrow().is_empty());

    let new_engine = Script::new(script, tok.n_vocab());
    let mut new = Chat::new(&new_engine, &tok, options(true)).expect("chat");
    chat.begin(&message()).expect("begin");
    new.begin(&message()).expect("begin");
    let after = step_to_the_end(&mut chat);
    let fresh = step_to_the_end(&mut new);
    assert_eq!(after.stats.tokens, fresh.stats.tokens);
    assert_eq!(*engine.fed.borrow(), *new_engine.fed.borrow());
}

/// A cancel in the prompt keeps the rest of the prompt and the close of the
/// message for the next turn
#[test]
#[ignore = "needs a model file"]
fn a_cancel_in_the_prompt_keeps_its_rest_and_the_close() {
    let gguf = biv_cli::open_gguf(model_path()).expect("gguf");
    let tok = Tokenizer::from_gguf(&gguf).expect("tokenizer");
    let engine = Script::new(vec![word(&tok); SCRIPT_TOKENS], tok.n_vocab());
    let mut chat = Chat::new(&engine, &tok, options(true)).expect("chat");
    chat.begin(&message()).expect("begin");
    let first = chat.step(&mut sampler(), |_: Event<'_>| {}).expect("step");
    assert!(matches!(first, Step::Prompt));
    let stats = chat.cancel(|_: Event<'_>| {}).expect("cancel");

    assert_eq!(stats.stop, Stop::Cancel);
    assert_eq!(stats.gen_tokens, 0);
    assert_eq!(chat.n_past(), BATCH);
    assert_eq!(chat.history().len(), BATCH);
    let close = [id(&tok, "</think>"), id(&tok, "<|im_end|>")];
    let pending = chat.pending();
    assert_eq!(pending.len(), stats.prompt_tokens - BATCH + close.len());
    assert_eq!(pending[pending.len() - close.len()..], close);
}

/// A cancel in the answer closes the message after the last token chosen,
/// as a stop of send does
#[test]
#[ignore = "needs a model file"]
fn a_cancel_in_the_answer_closes_the_message() {
    let gguf = biv_cli::open_gguf(model_path()).expect("gguf");
    let tok = Tokenizer::from_gguf(&gguf).expect("tokenizer");
    let engine = Script::new(vec![word(&tok); SCRIPT_TOKENS], tok.n_vocab());
    let mut chat = Chat::new(&engine, &tok, options(true)).expect("chat");
    chat.begin(&message()).expect("begin");
    let mut s = sampler();
    let chosen = loop {
        match chat.step(&mut s, |_: Event<'_>| {}).expect("step") {
            Step::Prompt => {}
            Step::Tokens(t) => break t,
            Step::Done(..) => panic!("the turn ends at MAX_TOKENS, after this step"),
        }
    };
    let stats = chat.cancel(|_: Event<'_>| {}).expect("cancel");

    assert_eq!(stats.stop, Stop::Cancel);
    assert_eq!(stats.tokens, chosen);
    assert_eq!(
        chat.pending(),
        [word(&tok), id(&tok, "</think>"), id(&tok, "<|im_end|>")]
    );
}

/// A chat with an open turn refuses a second turn and a save, and a chat
/// with no turn refuses a step and a cancel
#[test]
#[ignore = "needs a model file"]
fn an_open_turn_is_stepped_or_cancelled_first() {
    let gguf = biv_cli::open_gguf(model_path()).expect("gguf");
    let tok = Tokenizer::from_gguf(&gguf).expect("tokenizer");
    let engine = Script::new(vec![word(&tok); SCRIPT_TOKENS], tok.n_vocab());
    let mut chat = Chat::new(&engine, &tok, options(true)).expect("chat");

    assert!(matches!(
        chat.step(&mut sampler(), |_: Event<'_>| {}),
        Err(Error::NoTurn)
    ));
    assert!(matches!(chat.cancel(|_: Event<'_>| {}), Err(Error::NoTurn)));

    chat.begin(&message()).expect("begin");
    assert!(matches!(chat.begin(&message()), Err(Error::TurnOpen)));
    assert!(matches!(
        chat.send(&message(), &mut sampler(), |_: Event<'_>| {}, || false),
        Err(Error::TurnOpen)
    ));
    assert!(matches!(chat.save(), Err(Error::TurnOpen)));
}
