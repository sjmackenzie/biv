//! Compute backends behind one interface: an engine holds the weights, a
//! session holds one sequence (kv cache, recurrent states).

use core::any::Any;

use crate::Error;
use crate::model::{Model, State, StateSnapshot};
use crate::par;
use crate::prelude::*;
use crate::tokenizer::TokenId;

pub trait Engine {
    /// A new empty sequence able to hold `context` tokens, drafting up to
    /// `draft` tokens per step when the model has an MTP head
    fn new_session(&self, context: usize, draft: usize) -> Result<Box<dyn Session + '_>, Error>;

    /// Prompt tokens per forward pass this backend works best with
    fn batch_size(&self) -> usize;

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

    /// Tokens likely to follow `next`, the token chosen after the sequence.
    /// Empty when the session doesn't draft.
    fn draft(&mut self, next: TokenId) -> Vec<TokenId>;

    /// Append `tokens` provisionally, returning the logits after each one.
    /// [`Session::keep`] must follow.
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

    fn draft(&mut self, next: TokenId) -> Vec<TokenId> {
        self.model.draft(&mut self.state, next, self.draft)
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
