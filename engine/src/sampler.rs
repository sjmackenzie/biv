//! Token sampling: penalties, temperature, top-k, top-p, min-p.

#[cfg(not(any(feature = "std", test)))]
use crate::math::Float;
use crate::prelude::*;
use crate::tokenizer::TokenId;

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

    /// xorshift64*, uniform in [0, 1)
    fn uniform(&mut self) -> f32 {
        self.rng ^= self.rng >> 12;
        self.rng ^= self.rng << 25;
        self.rng ^= self.rng >> 27;
        let v = self.rng.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (v >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Pick the next token; `recent` is the generated history used by the penalties
    pub fn sample(&mut self, logits: &mut [f32], recent: &[TokenId]) -> TokenId {
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

        if c.temperature <= 0.0 {
            return argmax(logits);
        }

        // candidates, best first
        let mut cand: Vec<(TokenId, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &l)| (i as TokenId, l))
            .collect();
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

        let total: f32 = cand.iter().map(|c| c.1).sum();
        let mut r = self.uniform() * total;
        for &(t, p) in &cand {
            r -= p;
            if r <= 0.0 {
                return t;
            }
        }
        cand.last().unwrap().0
    }
}

pub fn argmax(v: &[f32]) -> TokenId {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as TokenId)
        .unwrap_or(0)
}
