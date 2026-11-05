//! Token sampling: penalties, temperature, top-k, top-p, min-p.

use crate::Error;
use crate::gguf::Gguf;
#[cfg(not(any(feature = "std", test)))]
use crate::math::Float;
use crate::prelude::*;
use crate::tokenizer::TokenId;

/// The sampling values a model's publisher can store in the GGUF

#[derive(Debug, Clone)]
pub struct SamplerConfig {
    /// 0 = greedy
    pub temperature: f32,
    /// 0 = disabled
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    /// subtracted once from the logit of every token already generated
    pub presence_penalty: f32,
    /// divides positive / multiplies negative logits of recent tokens
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    /// of the random generator
    pub seed: u64,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        // the model card's recommended thinking-mode settings
        SamplerConfig {
            temperature: 1.0,
            top_k: 20,
            top_p: 0.95,
            min_p: 0.0,
            presence_penalty: 0.0,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
            seed: 0,
        }
    }
}

/// Values that replace those of [`SamplerConfig::from_gguf`]; `None` keeps
/// the GGUF's value, or the [`SamplerConfig::default`] one for a value the
/// GGUF doesn't store
#[derive(Debug, Clone)]
pub struct SamplerOverrides {
    pub temperature: Option<f32>,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub repeat_penalty: Option<f32>,
}

impl SamplerConfig {
    /// The sampling values recommended by the model's publisher, from the
    /// GGUF, with `overrides` over them. Fails with [`Error::ZeroSeed`] when
    /// `seed` is 0: the caller names the seed, so that a run can be repeated.
    pub fn from_gguf(gguf: &Gguf, overrides: SamplerOverrides, seed: u64) -> Result<Self, Error> {
        if seed == 0 {
            return Err(Error::ZeroSeed);
        }
        let stored = &gguf.metadata.general.sampling;
        let defaults = SamplerConfig::default();
        Ok(SamplerConfig {
            temperature: overrides
                .temperature
                .or(stored.temp)
                .unwrap_or(defaults.temperature),
            top_k: overrides
                .top_k
                .or(stored.top_k.map(|v| v as usize))
                .unwrap_or(defaults.top_k),
            top_p: overrides.top_p.or(stored.top_p).unwrap_or(defaults.top_p),
            min_p: overrides.min_p.or(stored.min_p).unwrap_or(defaults.min_p),
            presence_penalty: overrides
                .presence_penalty
                .unwrap_or(defaults.presence_penalty),
            repeat_penalty: overrides.repeat_penalty.unwrap_or(defaults.repeat_penalty),
            seed,
            ..defaults
        })
    }
}

pub struct Sampler {
    pub cfg: SamplerConfig,
    rng: u64,
}

impl Sampler {
    pub fn new(cfg: SamplerConfig) -> Self {
        let seed = if cfg.seed != 0 {
            cfg.seed
        } else {
            0x9E37_79B9_7F4A_7C15
        };
        Sampler { cfg, rng: seed | 1 }
    }

    /// The random generator's state after the tokens sampled so far, to
    /// [`Sampler::resume`] from, e.g. with a [`crate::chat::Prefix`] of the
    /// same point
    pub fn rng_state(&self) -> u64 {
        self.rng
    }

    /// A sampler that goes on from `rng_state`, as [`Sampler::rng_state`]
    /// gave it: it samples what the sampler that gave it would have sampled.
    /// Fails with [`Error::ZeroRngState`] on 0, a state the generator never
    /// holds.
    pub fn resume(cfg: SamplerConfig, rng_state: u64) -> Result<Self, Error> {
        if rng_state == 0 {
            return Err(Error::ZeroRngState);
        }
        Ok(Sampler {
            cfg,
            rng: rng_state,
        })
    }

    /// xorshift64*, uniform in [0, 1)
    pub fn uniform(&mut self) -> f32 {
        self.rng ^= self.rng >> 12;
        self.rng ^= self.rng << 25;
        self.rng ^= self.rng >> 27;
        let v = self.rng.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (v >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Whether the sampler picks the most likely token
    pub fn greedy(&self) -> bool {
        self.cfg.temperature <= 0.0
    }

    /// Pick the next token; `recent` is the generated history used by the penalties
    pub fn sample(&mut self, logits: &mut [f32], recent: &[TokenId]) -> TokenId {
        self.penalize(logits, recent);
        if self.greedy() {
            return argmax(logits);
        }
        let cand = self.probs(enumerate(logits));
        self.draw(&cand)
    }

    /// The distribution [`Sampler::sample`] draws from, as (token,
    /// probability) pairs, best first, their sum 1 or a bit less. Not greedy.
    pub fn distribution(&self, logits: &mut [f32], recent: &[TokenId]) -> Vec<(TokenId, f32)> {
        self.penalize(logits, recent);
        self.probs(enumerate(logits))
    }

    /// The distribution of (token, logit) candidates, as
    /// [`Sampler::distribution`] without the penalties. Not greedy.
    pub fn distribution_of(&self, cand: Vec<(TokenId, f32)>) -> Vec<(TokenId, f32)> {
        self.probs(cand)
    }

    /// A token of (token, weight) pairs, with a chance proportional to its weight
    pub fn draw(&mut self, cand: &[(TokenId, f32)]) -> TokenId {
        let total: f32 = cand.iter().map(|c| c.1).sum();
        let mut r = self.uniform() * total;
        for &(t, p) in cand {
            r -= p;
            if r <= 0.0 {
                return t;
            }
        }
        cand.last().unwrap().0
    }

    fn penalize(&self, logits: &mut [f32], recent: &[TokenId]) {
        let c = &self.cfg;
        if c.presence_penalty != 0.0 || c.repeat_penalty != 1.0 {
            let window = &recent[recent.len().saturating_sub(c.repeat_last_n)..];
            let mut seen = window.to_vec();
            seen.sort_unstable();
            seen.dedup();
            for &t in &seen {
                let l = &mut logits[t as usize];
                if c.repeat_penalty != 1.0 {
                    *l = if *l > 0.0 {
                        *l / c.repeat_penalty
                    } else {
                        *l * c.repeat_penalty
                    };
                }
                *l -= c.presence_penalty;
            }
        }
    }

    /// top-k, softmax with temperature, min-p, top-p of (token, logit) candidates
    fn probs(&self, mut cand: Vec<(TokenId, f32)>) -> Vec<(TokenId, f32)> {
        let c = &self.cfg;
        // candidates, best first
        let k = if c.top_k > 0 {
            c.top_k.min(cand.len())
        } else {
            cand.len()
        };
        if k < cand.len() {
            cand.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
            cand.truncate(k);
        }
        cand.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

        // softmax with temperature
        let max = cand[0].1;
        let mut sum = 0.0;
        for (_, l) in cand.iter_mut() {
            *l = ((*l - max) / c.temperature).exp();
            sum += *l;
        }
        for (_, p) in cand.iter_mut() {
            *p /= sum;
        }

        if c.min_p > 0.0 {
            let floor = cand[0].1 * c.min_p;
            cand.retain(|&(_, p)| p >= floor);
        }
        if c.top_p < 1.0 {
            let mut acc = 0.0;
            let mut keep = cand.len();
            for (i, &(_, p)) in cand.iter().enumerate() {
                acc += p;
                if acc >= c.top_p {
                    keep = i + 1;
                    break;
                }
            }
            cand.truncate(keep);
        }
        cand
    }
}

fn enumerate(logits: &[f32]) -> Vec<(TokenId, f32)> {
    logits
        .iter()
        .enumerate()
        .map(|(i, &l)| (i as TokenId, l))
        .collect()
}

pub fn argmax(v: &[f32]) -> TokenId {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as TokenId)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::Shard;

    const SEED: u64 = 7;
    const KEY_TEMPERATURE: &str = "general.sampling.temp";
    const KEY_TOP_K: &str = "general.sampling.top_k";
    const KEY_TOP_P: &str = "general.sampling.top_p";
    /// GGUF metadata value types
    const TYPE_U32: u32 = 4;
    const TYPE_F32: u32 = 6;
    const GGUF_MAGIC: &[u8] = b"GGUF";
    const GGUF_VERSION: u32 = 3;

    fn string(out: &mut Vec<u8>, s: &str) {
        out.extend((s.len() as u64).to_le_bytes());
        out.extend(s.as_bytes());
    }

    /// A GGUF with no tensors and the metadata `kv`, as (key, type, little-endian value)
    fn gguf(kv: &[(&str, u32, [u8; 4])]) -> Gguf {
        let mut b = Vec::new();
        b.extend(GGUF_MAGIC);
        b.extend(GGUF_VERSION.to_le_bytes());
        b.extend(0u64.to_le_bytes());
        b.extend((kv.len() as u64).to_le_bytes());
        for (key, ty, value) in kv {
            string(&mut b, key);
            b.extend(ty.to_le_bytes());
            b.extend(value);
        }
        let shard: Shard = Box::new(b);
        Gguf::from_shards(vec![shard]).unwrap()
    }

    fn no_overrides() -> SamplerOverrides {
        SamplerOverrides {
            temperature: None,
            top_k: None,
            top_p: None,
            min_p: None,
            presence_penalty: None,
            repeat_penalty: None,
        }
    }

    /// Tokens sampled before and after the state is taken
    const BEFORE: usize = 5;
    const AFTER: usize = 20;
    const VOCAB: usize = 50;

    /// Logits that leave many candidates, so that the draws decide the tokens
    fn logits() -> Vec<f32> {
        (0..VOCAB).map(|i| (i % 7) as f32 * 0.1).collect()
    }

    fn draws(s: &mut Sampler, n: usize) -> Vec<TokenId> {
        (0..n).map(|_| s.sample(&mut logits(), &[])).collect()
    }

    #[test]
    fn resume_goes_on_from_the_state() {
        let cfg = SamplerConfig {
            top_k: 0,
            top_p: 1.0,
            seed: SEED,
            ..SamplerConfig::default()
        };
        let mut a = Sampler::new(cfg.clone());
        draws(&mut a, BEFORE);
        let mut b = Sampler::resume(cfg.clone(), a.rng_state()).unwrap();
        assert_eq!(draws(&mut a, AFTER), draws(&mut b, AFTER));
        // a new sampler with the same seed starts over instead
        let mut c = Sampler::new(cfg);
        let mut d = Sampler::new(c.cfg.clone());
        draws(&mut d, BEFORE);
        assert_ne!(draws(&mut c, AFTER), draws(&mut d, AFTER));
    }

    #[test]
    fn resume_refuses_state_zero() {
        assert!(matches!(
            Sampler::resume(SamplerConfig::default(), 0),
            Err(Error::ZeroRngState)
        ));
    }

    #[test]
    fn from_gguf_refuses_seed_zero() {
        let g = gguf(&[]);
        assert!(matches!(
            SamplerConfig::from_gguf(&g, no_overrides(), 0),
            Err(Error::ZeroSeed)
        ));
    }

    #[test]
    fn from_gguf_takes_stored_values_then_overrides() {
        let g = gguf(&[
            (KEY_TEMPERATURE, TYPE_F32, 0.6f32.to_le_bytes()),
            (KEY_TOP_K, TYPE_U32, 40u32.to_le_bytes()),
            (KEY_TOP_P, TYPE_F32, 0.8f32.to_le_bytes()),
        ]);
        let c = SamplerConfig::from_gguf(&g, no_overrides(), SEED).unwrap();
        assert_eq!((c.temperature, c.top_k, c.top_p), (0.6, 40, 0.8));
        assert_eq!(c.seed, SEED);

        let over = SamplerOverrides {
            temperature: Some(0.0),
            top_k: Some(5),
            min_p: Some(0.1),
            ..no_overrides()
        };
        let c = SamplerConfig::from_gguf(&g, over, SEED).unwrap();
        assert_eq!(
            (c.temperature, c.top_k, c.top_p, c.min_p),
            (0.0, 5, 0.8, 0.1)
        );
    }

    #[test]
    fn from_gguf_without_stored_values_keeps_the_defaults() {
        let c = SamplerConfig::from_gguf(&gguf(&[]), no_overrides(), SEED).unwrap();
        let d = SamplerConfig::default();
        assert_eq!(
            (c.temperature, c.top_k, c.top_p, c.min_p, c.repeat_last_n),
            (d.temperature, d.top_k, d.top_p, d.min_p, d.repeat_last_n)
        );
    }
}
