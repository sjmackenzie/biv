//! Compute backends behind one interface: an engine holds the weights, a
//! session holds one sequence (kv cache, recurrent states).

use core::any::Any;

use crate::Error;
use crate::model::{Model, State, StateSnapshot};
use crate::par;
use crate::prelude::*;
use crate::tokenizer::TokenId;

pub trait Engine {
    /// A new empty sequence able to hold `context` tokens, verifying up to
    /// `draft` tokens after the chosen one per step, and drafting with the
    /// model's MTP head when it has one and `draft` isn't 0
    fn new_session(&self, context: usize, draft: usize) -> Result<Box<dyn Session + '_>, Error>;

    /// Prompt tokens per forward pass this backend works best with
    fn batch_size(&self) -> usize;

    /// Tokens drafted per step this backend works best with
    fn draft_size(&self) -> usize;

    fn description(&self) -> String;

    /// What decides the numbers this backend computes, besides the code, the
    /// weights and how the tokens are batched: kernel variants, cache types,
    /// sizes, as (name, value) pairs. Equal pairs on the same code and model
    /// should give the same bits.
    fn numerics(&self) -> Vec<(&'static str, String)>;
}

/// A sequence. Speculative decoding is three calls: [`Session::draft`] guesses
/// the tokens after the one just chosen, [`Session::verify`] appends the chosen
/// token and the guesses, and [`Session::keep`] keeps the ones that held.
pub trait Session {
    /// Append `tokens`, returning the logits of the last one
    fn forward(&mut self, tokens: &[TokenId]) -> Vec<f32>;

    /// Append `tokens` without computing logits
    fn prefill(&mut self, tokens: &[TokenId]);

    /// Up to `n` tokens likely to follow `next`, the token chosen after the
    /// sequence, at most the session's draft length, each chosen by `pick`
    /// from the MTP head's logits ([`DraftLogits::argmax`] for the likeliest).
    /// Empty when the session doesn't draft.
    fn draft(
        &mut self,
        next: TokenId,
        n: usize,
        pick: &mut dyn FnMut(&DraftLogits<'_>) -> TokenId,
    ) -> Vec<TokenId>;

    /// Append `tokens` provisionally, returning the logits after each one:
    /// the chosen token and up to the session's draft length more, drafted or
    /// not. [`Session::keep`] must follow.
    fn verify(&mut self, tokens: &[TokenId]) -> Vec<Vec<f32>>;

    /// Keep the first `n` tokens of the last verify, dropping the others
    fn keep(&mut self, n: usize);

    /// Tokens in the sequence
    fn n_past(&self) -> usize;

    /// Memory held by the sequence state
    fn state_bytes(&self) -> usize;

    /// A copy of the sequence, to [`Session::restore`] in sessions of the same
    /// engine. Not between [`Session::verify`] and [`Session::keep`].
    fn snapshot(&self) -> Result<Snapshot, Error>;

    /// Replace the sequence with `snapshot`, taken from a session of the same
    /// engine. The draft length may differ, but a drafting session needs a
    /// snapshot from a drafting one: the MTP head's history is part of it.
    fn restore(&mut self, snapshot: &Snapshot) -> Result<(), Error>;
}

/// The MTP head's logits for a drafted token: `logits[i]` is for token
/// `tokens[i]`, or for token `i` when the head covers the whole vocabulary
pub struct DraftLogits<'a> {
    pub logits: &'a [f32],
    pub tokens: Option<&'a [TokenId]>,
}

impl DraftLogits<'_> {
    pub fn token(&self, i: usize) -> TokenId {
        self.tokens.map_or(i as TokenId, |t| t[i])
    }

    /// The likeliest token
    pub fn argmax(&self) -> TokenId {
        self.token(crate::sampler::argmax(self.logits) as usize)
    }

    /// (token, logit) pairs
    pub fn candidates(&self) -> Vec<(TokenId, f32)> {
        self.logits
            .iter()
            .enumerate()
            .map(|(i, &l)| (self.token(i), l))
            .collect()
    }
}

/// The state of a sequence after its tokens, in the layout of the backend that
/// made it: the attention caches of those tokens, and the recurrent states,
/// which only processing the whole sequence again could rebuild
pub struct Snapshot {
    n_past: usize,
    bytes: usize,
    data: Box<dyn Any + Send + Sync>,
}

impl Snapshot {
    pub(crate) fn new<T: Any + Send + Sync>(n_past: usize, bytes: usize, data: T) -> Self {
        Snapshot {
            n_past,
            bytes,
            data: Box::new(data),
        }
    }

    pub(crate) fn data<T: Any>(&self) -> Result<&T, Error> {
        self.data
            .downcast_ref()
            .ok_or(Error::Snapshot("taken from another backend"))
    }

    /// Tokens in the sequence
    pub fn n_past(&self) -> usize {
        self.n_past
    }

    /// Memory held
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

impl<T: Engine + ?Sized> Engine for &T {
    fn new_session(&self, context: usize, draft: usize) -> Result<Box<dyn Session + '_>, Error> {
        (**self).new_session(context, draft)
    }

    fn batch_size(&self) -> usize {
        (**self).batch_size()
    }

    fn draft_size(&self) -> usize {
        (**self).draft_size()
    }

    fn description(&self) -> String {
        (**self).description()
    }

    fn numerics(&self) -> Vec<(&'static str, String)> {
        (**self).numerics()
    }
}

impl Engine for Model<'_> {
    fn new_session(&self, context: usize, draft: usize) -> Result<Box<dyn Session + '_>, Error> {
        Ok(Box::new(CpuSession {
            model: self,
            state: self.new_state(context, draft > 0)?,
            draft,
        }))
    }

    fn batch_size(&self) -> usize {
        64
    }

    // a pass of 2 tokens costs 1.45x one of 1, and the cost grows from there
    fn draft_size(&self) -> usize {
        1
    }

    fn description(&self) -> String {
        format!("cpu, {} threads", par::threads())
    }

    // not the thread count: each output row is computed whole by one thread
    fn numerics(&self) -> Vec<(&'static str, String)> {
        vec![
            ("kind", "cpu".into()),
            ("kernels", crate::quant::kernels().into()),
            ("kv_cache", "f32".into()),
        ]
    }
}

struct CpuSession<'m> {
    model: &'m Model<'m>,
    state: State,
    draft: usize,
}

impl Session for CpuSession<'_> {
    fn forward(&mut self, tokens: &[TokenId]) -> Vec<f32> {
        self.model.forward(&mut self.state, tokens)
    }

    fn prefill(&mut self, tokens: &[TokenId]) {
        self.model.prefill(&mut self.state, tokens)
    }

    fn draft(
        &mut self,
        next: TokenId,
        n: usize,
        pick: &mut dyn FnMut(&DraftLogits<'_>) -> TokenId,
    ) -> Vec<TokenId> {
        self.model
            .draft(&mut self.state, next, n.min(self.draft), pick)
    }

    fn verify(&mut self, tokens: &[TokenId]) -> Vec<Vec<f32>> {
        self.model.verify(&mut self.state, tokens)
    }

    fn keep(&mut self, n: usize) {
        self.model.keep(&mut self.state, n)
    }

    fn n_past(&self) -> usize {
        self.state.n_past
    }

    fn state_bytes(&self) -> usize {
        self.state.kv_bytes()
    }

    fn snapshot(&self) -> Result<Snapshot, Error> {
        let snap = self.state.snapshot()?;
        Ok(Snapshot::new(self.state.n_past, snap.bytes(), snap))
    }

    fn restore(&mut self, snapshot: &Snapshot) -> Result<(), Error> {
        self.state.restore(snapshot.data::<StateSnapshot>()?)
    }
}
