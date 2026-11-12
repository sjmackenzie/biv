//! The counts of the passes of the batch scheduler. Only the tokenizer comes
//! from the model file; a scripted engine chooses each token, so the tests
//! need no GPU, but they are ignored without a model file:
//! `BIV_TEST_MODEL=<gguf> cargo test --release --test batch_passes -- --ignored`

mod common;

use std::thread;
use std::time::Duration;

use biv::batch::{self, BatchOptions, Passes};
use biv::chat::{Chat, ChatOptions, Event};
use biv::engine::Engine;
use biv::tokenizer::{TokenId, Tokenizer};
use common::{NO_DRAFT, Script, model_path, sampler};

/// A text that the tokenizer gives as one token
const WORD: &str = " word";
/// The words of the message, more than one pass of prompt tokens
const MESSAGE_WORDS: usize = 100;
/// The tokens that the scripted engine can choose in one session
const SCRIPT_TOKENS: usize = 16;
/// The per-answer cap, below the script, so that it stops each turn
const MAX_TOKENS: usize = 6;
/// The prompt tokens of each pass of these tests
const BATCH: usize = 32;
/// A context far above any turn of these tests
const CONTEXT: usize = 4096;
/// The batch of the scheduler: the tokens and the logits of one pass
const PASS_TOKENS: usize = 64;
const PASS_LOGITS: usize = 8;
/// The wait of the scheduler for the sessions of the last pass
const PASS_WAIT: Duration = Duration::from_millis(5);
/// The chats of the test of two chats
const TWO_CHATS: usize = 2;

fn options() -> ChatOptions {
    ChatOptions {
        system: None,
        thinking: false,
        context: CONTEXT,
        max_tokens: MAX_TOKENS,
        batch: Some(BATCH),
        draft: Some(NO_DRAFT),
        lookup: 0,
        ..ChatOptions::default()
    }
}

fn batch_options() -> BatchOptions {
    BatchOptions {
        tokens: Some(PASS_TOKENS),
        logits: PASS_LOGITS,
        wait: PASS_WAIT,
    }
}

fn word(tok: &Tokenizer) -> TokenId {
    let ids = tok.encode(WORD);
    assert_eq!(ids.len(), 1, "{WORD:?} is one token");
    ids[0]
}

/// One turn of `chats` chats at the same time, each on a thread of its own,
/// and the passes of the scheduler after them
fn passes_of(engine: &Script, tok: &Tokenizer, chats: usize) -> Passes {
    let message = WORD.repeat(MESSAGE_WORDS);
    batch::serve(engine, &batch_options(), |handle| {
        thread::scope(|scope| {
            for _ in 0..chats {
                scope.spawn(|| {
                    let engine: &dyn Engine = &handle;
                    let mut chat = Chat::new(engine, tok, options()).expect("chat");
                    chat.send(&message, &mut sampler(), |_: Event<'_>| {}, || false)
                        .expect("send");
                });
            }
        });
        handle.passes()
    })
    .expect("serve")
}

/// The passes count each token that the sessions took. With one chat no
/// pass holds the tokens of two sequences.
#[test]
#[ignore = "needs a model file"]
fn the_passes_of_one_chat_hold_its_tokens_and_none_is_shared() {
    let gguf = biv_cli::open_gguf(model_path()).expect("gguf");
    let tok = Tokenizer::from_gguf(&gguf).expect("tokenizer");
    let engine = Script::new(vec![word(&tok); SCRIPT_TOKENS], tok.n_vocab());

    let passes = passes_of(&engine, &tok, 1);

    assert_eq!(passes.tokens, engine.fed.borrow().len() as u64);
    assert_eq!(passes.shared, 0);
    assert!(passes.passes > 1, "the prompt is more than one pass");
}

/// With two chats the passes hold the tokens of both, the same tokens as
/// two turns alone, in at most as many passes
#[test]
#[ignore = "needs a model file"]
fn the_passes_of_two_chats_hold_the_tokens_of_both() {
    let gguf = biv_cli::open_gguf(model_path()).expect("gguf");
    let tok = Tokenizer::from_gguf(&gguf).expect("tokenizer");
    let alone_engine = Script::new(vec![word(&tok); SCRIPT_TOKENS], tok.n_vocab());
    let alone = passes_of(&alone_engine, &tok, 1);
    let engine = Script::new(vec![word(&tok); SCRIPT_TOKENS], tok.n_vocab());

    let passes = passes_of(&engine, &tok, TWO_CHATS);

    assert_eq!(passes.tokens, engine.fed.borrow().len() as u64);
    assert_eq!(passes.tokens, TWO_CHATS as u64 * alone.tokens);
    assert!(passes.shared <= passes.passes);
    assert!(passes.passes <= TWO_CHATS as u64 * alone.passes);
}
