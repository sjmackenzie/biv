//! A chat that goes on from the messages of an earlier conversation. Only
//! the tokenizer comes from the model file; a scripted engine chooses each
//! token, so the test needs no GPU, but it is ignored without a model file:
//! `BIV_TEST_MODEL=<gguf> cargo test --release --test chat_history -- --ignored`

mod common;

use biv::chat::{Chat, ChatOptions, Message, ReasoningEffort};
use biv::tokenizer::{PromptPart, Tokenizer};
use common::{NO_DRAFT, Script, model_path, sampler};

use PromptPart::{Content, Template};

const CONTEXT: usize = 4096;
/// The answer of the turn: one token, enough to see the prompt
const ONE_TOKEN: usize = 1;
const FIRST: &str = "Name a prime number.";
const ANSWER: &str = "Seven.";
const SECOND: &str = "And the next one?";
/// A text that the tokenizer gives as one token, the answer of the script
const WORD: &str = " word";

/// The chat template of Qwen3.8 for a user message, an earlier answer with
/// no reasoning, and the next user message, with thinking on and no system
/// block (reasoning effort medium gives no reasoning instructions)
fn template(tok: &Tokenizer) -> Vec<biv::tokenizer::TokenId> {
    tok.encode_prompt(&[
        Template("<|im_start|>user\n"),
        Content(FIRST),
        Template("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"),
        Content(ANSWER),
        Template("<|im_end|>\n<|im_start|>user\n"),
        Content(SECOND),
        Template("<|im_end|>\n<|im_start|>assistant\n<think>\n"),
    ])
}

#[test]
#[ignore = "needs a model file"]
fn a_chat_takes_the_messages_of_an_earlier_conversation_as_the_template_writes_them() {
    let gguf = biv_cli::open_gguf(model_path()).expect("gguf");
    let tok = Tokenizer::from_gguf(&gguf).expect("tokenizer");
    let word = tok.encode(WORD);
    let engine = Script::new(word, tok.n_vocab());
    let opts = ChatOptions {
        system: None,
        thinking: true,
        effort: ReasoningEffort::Medium,
        context: CONTEXT,
        max_tokens: ONE_TOKEN,
        draft: Some(NO_DRAFT),
        lookup: 0,
        tools: Vec::new(),
        ..ChatOptions::default()
    };
    let mut chat = Chat::new(&engine, &tok, opts).expect("chat");
    let history = [Message::User(FIRST), Message::Assistant(ANSWER)];
    let stats = chat
        .send_with_history(&history, SECOND, &mut sampler(), |_| {}, || false)
        .expect("turn");
    let expected = template(&tok);
    assert_eq!(stats.prompt_tokens, expected.len());
    assert_eq!(*engine.fed.borrow(), expected);
}
