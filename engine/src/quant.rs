//! ggml tensor types: block layouts, dequantization and dot products.
//!
//! Activations are quantized to int8 in blocks of 32 ([`Q8Act`]), each with
//! its own f32 scale. Every quantized weight type dots against that single
//! activation format: the k-quants' 16/32-element sub-blocks line up with the
//! activation blocks, so the per-sub-block scales fold into the float
//! accumulation (ported from ggml's `vec_dot_*` kernels).

#[cfg(not(any(feature = "std", test)))]
use crate::math::Float;
use crate::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    BF16,
}

impl GgmlType {
    pub fn from_u32(v: u32) -> Option<Self> {
        Some(match v {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            30 => Self::BF16,
            _ => return None,
        })
    }

    /// (elements per block, bytes per block)
    pub fn block(self) -> (usize, usize) {
        match self {
            Self::F32 => (1, 4),
            Self::F16 | Self::BF16 => (1, 2),
            Self::Q4_0 => (32, 18),
            Self::Q4_1 => (32, 20),
            Self::Q5_0 => (32, 22),
            Self::Q5_1 => (32, 24),
            Self::Q8_0 => (32, 34),
            Self::Q2K => (256, 84),
            Self::Q3K => (256, 110),
            Self::Q4K => (256, 144),
            Self::Q5K => (256, 176),
            Self::Q6K => (256, 210),
        }
    }

    pub fn row_size(self, n: usize) -> usize {
        let (be, bb) = self.block();
        debug_assert!(n.is_multiple_of(be));
        n / be * bb
    }

    /// Whether dot products use the float activations rather than [`Q8Act`]
    pub fn uses_f32_act(self) -> bool {
        matches!(self, Self::F32 | Self::F16 | Self::BF16)
    }

    /// Whether [`dot`] has a kernel for this type (the others can only be dequantized)
    pub fn has_dot(self) -> bool {
        matches!(
            self,
            Self::F32
                | Self::F16
                | Self::BF16
                | Self::Q4_0
                | Self::Q4_1
                | Self::Q8_0
                | Self::Q4K
                | Self::Q5K
                | Self::Q6K
        )
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub fn f16_to_f32(h: u16) -> f32 {
    // hardware conversion (the f16 vector types are not stable in std::arch)
    let r: f32;
    unsafe {
        core::arch::asm!(
            "fmov {t:s}, {h:w}",
            "fcvt {r:s}, {t:h}",
            h = in(reg) h as u32,
            t = out(vreg) _,
            r = lateout(vreg) r,
            options(pure, nomem, nostack)
        );
    }
    r
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
pub fn f16_to_f32(h: u16) -> f32 {
    // branch-free conversion (from ggml's fp16 fallback / FP16 library)
    let w = (h as u32) << 16;
    let sign = w & 0x8000_0000;
    let two_w = w.wrapping_add(w);
    let exp_offset = 0xE0u32 << 23;
    let normalized = f32::from_bits((two_w >> 4) + exp_offset) * f32::from_bits(0x0780_0000); // 2^-112
    let magic_mask = 126u32 << 23;
    let denormalized = f32::from_bits((two_w >> 17) | magic_mask) - 0.5;
    let bits = if two_w < (1 << 27) {
        denormalized.to_bits()
    } else {
        normalized.to_bits()
    };
    f32::from_bits(sign | bits)
}

#[inline]
pub fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

#[inline]
fn rd16(b: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([b[off], b[off + 1]])
}

#[inline]
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// Dequantize one row of `n` elements
pub fn dequantize_row(ty: GgmlType, src: &[u8], dst: &mut [f32]) {
    let n = dst.len();
    match ty {
        GgmlType::F32 => {
            for (i, d) in dst.iter_mut().enumerate() {
                *d = f32::from_le_bytes(src[4 * i..4 * i + 4].try_into().unwrap());
            }
        }
        GgmlType::F16 => {
            for (i, d) in dst.iter_mut().enumerate() {
                *d = f16_to_f32(rd16(src, 2 * i));
            }
        }
        GgmlType::BF16 => {
            for (i, d) in dst.iter_mut().enumerate() {
                *d = bf16_to_f32(rd16(src, 2 * i));
            }
        }
        GgmlType::Q4_0 => {
            for (b, y) in src.chunks_exact(18).zip(dst.chunks_exact_mut(32)) {
                let d = f16_to_f32(rd16(b, 0));
                for j in 0..16 {
                    y[j] = ((b[2 + j] & 0xF) as i32 - 8) as f32 * d;
                    y[j + 16] = ((b[2 + j] >> 4) as i32 - 8) as f32 * d;
                }
            }
        }
        GgmlType::Q4_1 => {
            for (b, y) in src.chunks_exact(20).zip(dst.chunks_exact_mut(32)) {
                let d = f16_to_f32(rd16(b, 0));
                let m = f16_to_f32(rd16(b, 2));
                for j in 0..16 {
                    y[j] = (b[4 + j] & 0xF) as f32 * d + m;
                    y[j + 16] = (b[4 + j] >> 4) as f32 * d + m;
                }
            }
        }
        GgmlType::Q5_0 => {
            for (b, y) in src.chunks_exact(22).zip(dst.chunks_exact_mut(32)) {
                let d = f16_to_f32(rd16(b, 0));
                let qh = u32::from_le_bytes(b[2..6].try_into().unwrap());
                for j in 0..16 {
                    let h0 = ((qh >> j) << 4) & 0x10;
                    let h1 = (qh >> (j + 12)) & 0x10;
                    y[j] = (((b[6 + j] & 0xF) as u32 | h0) as i32 - 16) as f32 * d;
                    y[j + 16] = (((b[6 + j] >> 4) as u32 | h1) as i32 - 16) as f32 * d;
                }
            }
        }
        GgmlType::Q5_1 => {
            for (b, y) in src.chunks_exact(24).zip(dst.chunks_exact_mut(32)) {
                let d = f16_to_f32(rd16(b, 0));
                let m = f16_to_f32(rd16(b, 2));
                let qh = u32::from_le_bytes(b[4..8].try_into().unwrap());
                for j in 0..16 {
                    let h0 = ((qh >> j) << 4) & 0x10;
                    let h1 = (qh >> (j + 12)) & 0x10;
                    y[j] = ((b[8 + j] & 0xF) as u32 | h0) as f32 * d + m;
                    y[j + 16] = ((b[8 + j] >> 4) as u32 | h1) as f32 * d + m;
                }
            }
        }
        GgmlType::Q8_0 => {
            for (b, y) in src.chunks_exact(34).zip(dst.chunks_exact_mut(32)) {
                let d = f16_to_f32(rd16(b, 0));
                for j in 0..32 {
                    y[j] = b[2 + j] as i8 as f32 * d;
                }
            }
        }
        GgmlType::Q2K => {
            for (b, y) in src.chunks_exact(84).zip(dst.chunks_exact_mut(256)) {
                let (scales, qs) = (&b[0..16], &b[16..80]);
                let d = f16_to_f32(rd16(b, 80));
                let min = f16_to_f32(rd16(b, 82));
                let mut yi = 0;
                let mut is = 0;
                for n in 0..2 {
                    let q = &qs[32 * n..];
                    for shift in [0, 2, 4, 6] {
                        for half in 0..2 {
                            let sc = scales[is];
                            is += 1;
                            let (dl, ml) = (d * (sc & 0xF) as f32, min * (sc >> 4) as f32);
                            for l in 0..16 {
                                y[yi] = dl * ((q[l + 16 * half] >> shift) & 3) as f32 - ml;
                                yi += 1;
                            }
                        }
                    }
                }
            }
        }
        GgmlType::Q3K => {
            for (b, y) in src.chunks_exact(110).zip(dst.chunks_exact_mut(256)) {
                let (hm, qs, sraw) = (&b[0..32], &b[32..96], &b[96..108]);
                let d_all = f16_to_f32(rd16(b, 108));
                // unpack the 16 6-bit scales
                let mut scales = [0i8; 16];
                for j in 0..16 {
                    let lo = if j < 8 {
                        sraw[j] & 0xF
                    } else {
                        sraw[j - 8] >> 4
                    };
                    let hi = (sraw[8 + j % 4] >> (2 * (j / 4))) & 3;
                    scales[j] = ((lo | (hi << 4)) as i8).wrapping_sub(32);
                }
                let mut yi = 0;
                let mut is = 0;
                let mut m = 1u8;
                for n in 0..2 {
                    let q = &qs[32 * n..];
                    for shift in [0, 2, 4, 6] {
                        for half in 0..2 {
                            let dl = d_all * scales[is] as f32;
                            is += 1;
                            for l in 0..16 {
                                let idx = l + 16 * half;
                                let v = ((q[idx] >> shift) & 3) as i8
                                    - if hm[idx] & m != 0 { 0 } else { 4 };
                                y[yi] = dl * v as f32;
                                yi += 1;
                            }
                        }
                        m <<= 1;
                    }
                }
            }
        }
        GgmlType::Q4K => {
            for (b, y) in src.chunks_exact(144).zip(dst.chunks_exact_mut(256)) {
                let d = f16_to_f32(rd16(b, 0));
                let min = f16_to_f32(rd16(b, 2));
                let (scales, qs) = (&b[4..16], &b[16..144]);
                for j in 0..4 {
                    let (s1, m1) = scale_min_k4(2 * j, scales);
                    let (s2, m2) = scale_min_k4(2 * j + 1, scales);
                    let (d1, m1) = (d * s1 as f32, min * m1 as f32);
                    let (d2, m2) = (d * s2 as f32, min * m2 as f32);
                    let q = &qs[32 * j..32 * j + 32];
                    for l in 0..32 {
                        y[64 * j + l] = d1 * (q[l] & 0xF) as f32 - m1;
                        y[64 * j + 32 + l] = d2 * (q[l] >> 4) as f32 - m2;
                    }
                }
            }
        }
        GgmlType::Q5K => {
            for (b, y) in src.chunks_exact(176).zip(dst.chunks_exact_mut(256)) {
                let d = f16_to_f32(rd16(b, 0));
                let min = f16_to_f32(rd16(b, 2));
                let (scales, qh, qs) = (&b[4..16], &b[16..48], &b[48..176]);
                for j in 0..4 {
                    let (s1, m1) = scale_min_k4(2 * j, scales);
                    let (s2, m2) = scale_min_k4(2 * j + 1, scales);
                    let (d1, m1) = (d * s1 as f32, min * m1 as f32);
                    let (d2, m2) = (d * s2 as f32, min * m2 as f32);
                    let (u1, u2) = (1u8 << (2 * j), 2u8 << (2 * j));
                    let q = &qs[32 * j..32 * j + 32];
                    for l in 0..32 {
                        let h1 = if qh[l] & u1 != 0 { 16 } else { 0 };
                        let h2 = if qh[l] & u2 != 0 { 16 } else { 0 };
                        y[64 * j + l] = d1 * ((q[l] & 0xF) + h1) as f32 - m1;
                        y[64 * j + 32 + l] = d2 * ((q[l] >> 4) + h2) as f32 - m2;
                    }
                }
            }
        }
        GgmlType::Q6K => {
            for (b, y) in src.chunks_exact(210).zip(dst.chunks_exact_mut(256)) {
                let d = f16_to_f32(rd16(b, 208));
                for n in 0..2 {
                    let ql = &b[64 * n..];
                    let qh = &b[128 + 32 * n..];
                    let sc = &b[192 + 8 * n..];
                    let y = &mut y[128 * n..];
                    for l in 0..32 {
                        let is = l / 16;
                        let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i32 - 32;
                        let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                        let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                        let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                        y[l] = d * (sc[is] as i8) as f32 * q1 as f32;
                        y[l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
                        y[l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
                        y[l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
                    }
                }
            }
        }
    }
    debug_assert_eq!(n % ty.block().0, 0);
}

/// Int8 activations, one f32 scale per 32 elements
#[derive(Default, Clone)]
pub struct Q8Act {
    pub qs: Vec<i8>,
    /// per block scale
    pub d: Vec<f32>,
    /// per block `d * sum(qs)`
    pub s: Vec<f32>,
}

impl Q8Act {
    pub fn quantize(x: &[f32]) -> Self {
        let mut q = Q8Act::default();
        q.quantize_into(x);
        q
    }

    pub fn quantize_into(&mut self, x: &[f32]) {
        assert!(
            x.len().is_multiple_of(32),
            "activation length must be a multiple of 32"
        );
        let nb = x.len() / 32;
        self.qs.resize(x.len(), 0);
        self.d.resize(nb, 0.0);
        self.s.resize(nb, 0.0);
        for (b, (xb, qb)) in x
            .chunks_exact(32)
            .zip(self.qs.chunks_exact_mut(32))
            .enumerate()
        {
            let amax = xb.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let d = amax / 127.0;
            let id = if d != 0.0 { 1.0 / d } else { 0.0 };
            let mut sum = 0i32;
            for (q, &v) in qb.iter_mut().zip(xb) {
                let r = (v * id).round() as i32;
                *q = r as i8;
                sum += r;
            }
            self.d[b] = d;
            self.s[b] = d * sum as f32;
        }
    }

    /// The view of `n` elements starting at element `off` (a multiple of 32)
    pub fn slice(&self, off: usize, n: usize) -> Q8Slice<'_> {
        Q8Slice {
            qs: &self.qs[off..off + n],
            d: &self.d[off / 32..(off + n) / 32],
            s: &self.s[off / 32..(off + n) / 32],
        }
    }
}

#[derive(Clone, Copy)]
pub struct Q8Slice<'a> {
    pub qs: &'a [i8],
    pub d: &'a [f32],
    pub s: &'a [f32],
}

/// Dot product of one weight row against activations (`xf` float, `xq` int8)
#[inline]
pub fn dot(ty: GgmlType, w: &[u8], xf: &[f32], xq: Option<Q8Slice<'_>>) -> f32 {
    match ty {
        GgmlType::F32 => dot_f32_bytes(w, xf),
        GgmlType::F16 => dot_f16(w, xf),
        GgmlType::BF16 => dot_bf16(w, xf),
        _ => {
            let xq = xq.expect("quantized activations required");
            match ty {
                GgmlType::Q4_0 => dot_q4_0(w, xq),
                GgmlType::Q4_1 => dot_q4_1(w, xq),
                GgmlType::Q8_0 => dot_q8_0(w, xq),
                GgmlType::Q4K => dot_q4k(w, xq),
                GgmlType::Q5K => dot_q5k(w, xq),
                GgmlType::Q6K => dot_q6k(w, xq),
                _ => unreachable!("no dot kernel for {ty:?}"),
            }
        }
    }
}

/// Dot products of one weight row against four activation rows. Types with a
/// dedicated kernel unpack each weight block once for the four rows.
#[inline]
pub fn dot_x4(ty: GgmlType, w: &[u8], xf: [&[f32]; 4], xq: Option<[Q8Slice<'_>; 4]>) -> [f32; 4] {
    #[cfg(target_arch = "aarch64")]
    if ty == GgmlType::Q4_1 {
        return unsafe { neon::dot_q4_1_x4(w, xq.expect("quantized activations required")) };
    }
    #[cfg(target_arch = "x86_64")]
    if ty == GgmlType::Q4_1 {
        let xq = || xq.expect("quantized activations required");
        match x86::level() {
            x86::Level::AvxVnni => return unsafe { x86::vnni::dot_q4_1_x4(w, xq()) },
            x86::Level::Avx2 => return unsafe { x86::avx2::dot_q4_1_x4(w, xq()) },
            x86::Level::Scalar => {}
        }
    }
    core::array::from_fn(|t| dot(ty, w, xf[t], xq.map(|q| q[t])))
}

// ---------------------------------------------------------------------------
// float kernels

#[allow(unreachable_code)]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "aarch64")]
    return unsafe { neon::dot_f32(a.as_ptr(), b.as_ptr(), a.len()) };
    #[cfg(target_arch = "x86_64")]
    if x86::has_avx2() {
        return unsafe { x86::dot_f32(a.as_ptr(), b.as_ptr(), a.len()) };
    }
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// `y += a * x`
pub fn axpy(y: &mut [f32], a: f32, x: &[f32]) {
    assert_eq!(y.len(), x.len());
    #[cfg(target_arch = "x86_64")]
    if x86::has_avx2() {
        return unsafe { x86::axpy(y.as_mut_ptr(), a, x.as_ptr(), x.len()) };
    }
    // NEON is baseline on aarch64, so this vectorizes there
    for (y, x) in y.iter_mut().zip(x) {
        *y += a * x;
    }
}

fn dot_f32_bytes(w: &[u8], x: &[f32]) -> f32 {
    // tensor data offsets are 32-byte aligned in gguf, so f32 rows are aligned
    let (pre, wf, _) = unsafe { w.align_to::<f32>() };
    if pre.is_empty() {
        dot_f32(&wf[..x.len()], x)
    } else {
        w.chunks_exact(4)
            .zip(x)
            .map(|(b, x)| f32::from_le_bytes(b.try_into().unwrap()) * x)
            .sum()
    }
}

fn dot_f16(w: &[u8], x: &[f32]) -> f32 {
    assert!(w.len() >= 2 * x.len());
    #[cfg(target_arch = "x86_64")]
    if x86::has_avx2() {
        return unsafe { x86::dot_f16(w.as_ptr(), x.as_ptr(), x.len()) };
    }
    x.iter()
        .enumerate()
        .map(|(i, x)| f16_to_f32(rd16(w, 2 * i)) * x)
        .sum()
}

#[allow(unreachable_code)]
fn dot_bf16(w: &[u8], x: &[f32]) -> f32 {
    assert!(w.len() >= 2 * x.len());
    #[cfg(target_arch = "aarch64")]
    return unsafe { neon::dot_bf16(w.as_ptr(), x.as_ptr(), x.len()) };
    #[cfg(target_arch = "x86_64")]
    if x86::has_avx2() {
        return unsafe { x86::dot_bf16(w.as_ptr(), x.as_ptr(), x.len()) };
    }
    x.iter()
        .enumerate()
        .map(|(i, x)| bf16_to_f32(rd16(w, 2 * i)) * x)
        .sum()
}

// ---------------------------------------------------------------------------
// quantized kernels: NEON (dotprod) on aarch64, AVX-VNNI or AVX2 on x86-64
// when the CPU has it, the scalar reference otherwise

macro_rules! quant_dots {
    ($($name:ident),*) => {$(
        #[allow(unreachable_code)]
        fn $name(w: &[u8], x: Q8Slice<'_>) -> f32 {
            #[cfg(target_arch = "aarch64")]
            return unsafe { neon::$name(w, x) };
            #[cfg(target_arch = "x86_64")]
            match x86::level() {
                x86::Level::AvxVnni => return unsafe { x86::vnni::$name(w, x) },
                x86::Level::Avx2 => return unsafe { x86::avx2::$name(w, x) },
                x86::Level::Scalar => {}
            }
            scalar::$name(w, x)
        }
    )*};
}

quant_dots!(dot_q4_0, dot_q4_1, dot_q8_0, dot_q4k, dot_q5k, dot_q6k);

/// The dot kernels this machine runs: `neon` (with dotprod) on aarch64,
/// `avxvnni` or `avx2` on x86-64 CPUs that have them, `scalar` otherwise
pub fn kernels() -> &'static str {
    #[cfg(target_arch = "aarch64")]
    return "neon";
    #[cfg(target_arch = "x86_64")]
    match x86::level() {
        x86::Level::AvxVnni => return "avxvnni",
        x86::Level::Avx2 => return "avx2",
        x86::Level::Scalar => {}
    }
    #[allow(unreachable_code)]
    "scalar"
}

/// Reference kernels: dequantize a block, then integer dot per 32 activations
#[allow(dead_code)]
pub mod scalar {
    use super::*;

    fn idot(a: &[i32], b: &[i8]) -> i32 {
        a.iter().zip(b).map(|(&a, &b)| a * b as i32).sum()
    }

    pub fn dot_q4_0(w: &[u8], x: Q8Slice<'_>) -> f32 {
        let mut sum = 0.0;
        for (i, b) in w.chunks_exact(18).take(x.d.len()).enumerate() {
            let d = f16_to_f32(rd16(b, 0));
            let mut q = [0i32; 32];
            for j in 0..16 {
                q[j] = (b[2 + j] & 0xF) as i32 - 8;
                q[j + 16] = (b[2 + j] >> 4) as i32 - 8;
            }
            sum += d * x.d[i] * idot(&q, &x.qs[32 * i..32 * i + 32]) as f32;
        }
        sum
    }

    pub fn dot_q4_1(w: &[u8], x: Q8Slice<'_>) -> f32 {
        let mut sum = 0.0;
        for (i, b) in w.chunks_exact(20).take(x.d.len()).enumerate() {
            let d = f16_to_f32(rd16(b, 0));
            let m = f16_to_f32(rd16(b, 2));
            let mut q = [0i32; 32];
            for j in 0..16 {
                q[j] = (b[4 + j] & 0xF) as i32;
                q[j + 16] = (b[4 + j] >> 4) as i32;
            }
            sum += d * x.d[i] * idot(&q, &x.qs[32 * i..32 * i + 32]) as f32 + m * x.s[i];
        }
        sum
    }

    pub fn dot_q8_0(w: &[u8], x: Q8Slice<'_>) -> f32 {
        let mut sum = 0.0;
        for (i, b) in w.chunks_exact(34).take(x.d.len()).enumerate() {
            let d = f16_to_f32(rd16(b, 0));
            let q: Vec<i32> = b[2..34].iter().map(|&v| v as i8 as i32).collect();
            sum += d * x.d[i] * idot(&q, &x.qs[32 * i..32 * i + 32]) as f32;
        }
        sum
    }

    pub fn dot_q4k(w: &[u8], x: Q8Slice<'_>) -> f32 {
        let mut sum = 0.0;
        for (i, b) in w.chunks_exact(144).take(x.d.len() / 8).enumerate() {
            let d = f16_to_f32(rd16(b, 0));
            let min = f16_to_f32(rd16(b, 2));
            let (scales, qs) = (&b[4..16], &b[16..144]);
            for sb in 0..8 {
                let (sc, m) = scale_min_k4(sb, scales);
                let q4 = &qs[32 * (sb / 2)..];
                let q: Vec<i32> = (0..32)
                    .map(|l| ((q4[l] >> (4 * (sb % 2))) & 0xF) as i32)
                    .collect();
                let xb = 8 * i + sb;
                sum += d * sc as f32 * x.d[xb] * idot(&q, &x.qs[32 * xb..32 * xb + 32]) as f32
                    - min * m as f32 * x.s[xb];
            }
        }
        sum
    }

    pub fn dot_q5k(w: &[u8], x: Q8Slice<'_>) -> f32 {
        let mut sum = 0.0;
        for (i, b) in w.chunks_exact(176).take(x.d.len() / 8).enumerate() {
            let d = f16_to_f32(rd16(b, 0));
            let min = f16_to_f32(rd16(b, 2));
            let (scales, qh, qs) = (&b[4..16], &b[16..48], &b[48..176]);
            for sb in 0..8 {
                let (sc, m) = scale_min_k4(sb, scales);
                let q4 = &qs[32 * (sb / 2)..];
                let q: Vec<i32> = (0..32)
                    .map(|l| {
                        (((q4[l] >> (4 * (sb % 2))) & 0xF) | (((qh[l] >> sb) & 1) << 4)) as i32
                    })
                    .collect();
                let xb = 8 * i + sb;
                sum += d * sc as f32 * x.d[xb] * idot(&q, &x.qs[32 * xb..32 * xb + 32]) as f32
                    - min * m as f32 * x.s[xb];
            }
        }
        sum
    }

    pub fn dot_q6k(w: &[u8], x: Q8Slice<'_>) -> f32 {
        let mut sum = 0.0;
        for (i, b) in w.chunks_exact(210).take(x.d.len() / 8).enumerate() {
            let d = f16_to_f32(rd16(b, 208));
            let mut q = [0i32; 256];
            let mut sc16 = [0i32; 16];
            for n in 0..2 {
                let ql = &b[64 * n..];
                let qh = &b[128 + 32 * n..];
                let sc = &b[192 + 8 * n..];
                for l in 0..32 {
                    let is = l / 16;
                    q[128 * n + l] = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i32 - 32;
                    q[128 * n + l + 32] =
                        ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                    q[128 * n + l + 64] = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                    q[128 * n + l + 96] =
                        ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                    for k in 0..4 {
                        sc16[8 * n + is + 2 * k] = sc[is + 2 * k] as i8 as i32;
                    }
                }
            }
            for xb in 0..8 {
                let mut isum = 0;
                for half in 0..2 {
                    let g = 2 * xb + half;
                    let off = 16 * g;
                    isum +=
                        sc16[g] * idot(&q[off..off + 16], &x.qs[256 * i + off..256 * i + off + 16]);
                }
                sum += d * x.d[8 * i + xb] * isum as f32;
            }
        }
        sum
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::{Q8Slice, f16_to_f32, rd16, scale_min_k4};
    use core::arch::aarch64::*;

    #[target_feature(enable = "neon")]
    pub unsafe fn dot_f32(a: *const f32, b: *const f32, n: usize) -> f32 {
        unsafe {
            let mut acc = [vdupq_n_f32(0.0); 4];
            let mut i = 0;
            while i + 16 <= n {
                for k in 0..4 {
                    acc[k] = vfmaq_f32(
                        acc[k],
                        vld1q_f32(a.add(i + 4 * k)),
                        vld1q_f32(b.add(i + 4 * k)),
                    );
                }
                i += 16;
            }
            let mut s = vaddvq_f32(vaddq_f32(
                vaddq_f32(acc[0], acc[1]),
                vaddq_f32(acc[2], acc[3]),
            ));
            while i < n {
                s += *a.add(i) * *b.add(i);
                i += 1;
            }
            s
        }
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn dot_bf16(w: *const u8, x: *const f32, n: usize) -> f32 {
        unsafe {
            let w = w as *const u16;
            let mut acc0 = vdupq_n_f32(0.0);
            let mut acc1 = vdupq_n_f32(0.0);
            let mut i = 0;
            while i + 8 <= n {
                let v = vld1q_u16(w.add(i));
                let lo = vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(v)));
                let hi = vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_high_u16(v)));
                acc0 = vfmaq_f32(acc0, lo, vld1q_f32(x.add(i)));
                acc1 = vfmaq_f32(acc1, hi, vld1q_f32(x.add(i + 4)));
                i += 8;
            }
            let mut s = vaddvq_f32(vaddq_f32(acc0, acc1));
            while i < n {
                s += f32::from_bits((*w.add(i) as u32) << 16) * *x.add(i);
                i += 1;
            }
            s
        }
    }

    #[inline(always)]
    unsafe fn dot32(acc: int32x4_t, a0: int8x16_t, a1: int8x16_t, x: *const i8) -> int32x4_t {
        unsafe { vdotq_s32(vdotq_s32(acc, a0, vld1q_s8(x)), a1, vld1q_s8(x.add(16))) }
    }

    #[target_feature(enable = "neon,dotprod")]
    pub unsafe fn dot_q4_0(w: &[u8], x: Q8Slice<'_>) -> f32 {
        unsafe {
            let nb = x.d.len();
            let mut acc = vdupq_n_f32(0.0);
            let m4 = vdupq_n_u8(0x0F);
            let s8 = vdupq_n_s8(8);
            for i in 0..nb {
                let b = w.as_ptr().add(18 * i);
                let d = f16_to_f32(rd16(w, 18 * i));
                let v = vld1q_u8(b.add(2));
                let lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v, m4)), s8);
                let hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(v)), s8);
                let p = dot32(vdupq_n_s32(0), lo, hi, x.qs.as_ptr().add(32 * i));
                acc = vfmaq_n_f32(acc, vcvtq_f32_s32(p), d * x.d[i]);
            }
            vaddvq_f32(acc)
        }
    }

    #[inline(always)]
    unsafe fn rd16p(p: *const u8) -> u16 {
        unsafe { (p as *const u16).read_unaligned() }
    }

    /// The nibbles of a q4 block as two vectors: elements 0..16 and 16..32
    #[inline(always)]
    unsafe fn q4_nibbles(qs: *const u8) -> (int8x16_t, int8x16_t) {
        unsafe {
            let v = vld1q_u8(qs);
            (
                vreinterpretq_s8_u8(vandq_u8(v, vdupq_n_u8(0x0F))),
                vreinterpretq_s8_u8(vshrq_n_u8::<4>(v)),
            )
        }
    }

    #[target_feature(enable = "neon,dotprod")]
    pub unsafe fn dot_q4_1(w: &[u8], x: Q8Slice<'_>) -> f32 {
        unsafe {
            let nb = x.d.len();
            debug_assert!(w.len() >= 20 * nb);
            let (wp, xq, xd, xs) = (w.as_ptr(), x.qs.as_ptr(), x.d.as_ptr(), x.s.as_ptr());
            let mut acc0 = vdupq_n_f32(0.0);
            let mut acc1 = vdupq_n_f32(0.0);
            let mut summs0 = 0.0f32;
            let mut summs1 = 0.0f32;
            let mut i = 0;
            while i + 2 <= nb {
                let b0 = wp.add(20 * i);
                let b1 = b0.add(20);
                let (l0, h0) = q4_nibbles(b0.add(4));
                let (l1, h1) = q4_nibbles(b1.add(4));
                let p0 = dot32(vdupq_n_s32(0), l0, h0, xq.add(32 * i));
                let p1 = dot32(vdupq_n_s32(0), l1, h1, xq.add(32 * i + 32));
                acc0 = vfmaq_n_f32(acc0, vcvtq_f32_s32(p0), f16_to_f32(rd16p(b0)) * *xd.add(i));
                acc1 = vfmaq_n_f32(
                    acc1,
                    vcvtq_f32_s32(p1),
                    f16_to_f32(rd16p(b1)) * *xd.add(i + 1),
                );
                summs0 += f16_to_f32(rd16p(b0.add(2))) * *xs.add(i);
                summs1 += f16_to_f32(rd16p(b1.add(2))) * *xs.add(i + 1);
                i += 2;
            }
            if i < nb {
                let b0 = wp.add(20 * i);
                let (l0, h0) = q4_nibbles(b0.add(4));
                let p0 = dot32(vdupq_n_s32(0), l0, h0, xq.add(32 * i));
                acc0 = vfmaq_n_f32(acc0, vcvtq_f32_s32(p0), f16_to_f32(rd16p(b0)) * *xd.add(i));
                summs0 += f16_to_f32(rd16p(b0.add(2))) * *xs.add(i);
            }
            vaddvq_f32(vaddq_f32(acc0, acc1)) + summs0 + summs1
        }
    }

    /// One q4_1 row against four activation rows: the block is unpacked once
    #[target_feature(enable = "neon,dotprod")]
    pub unsafe fn dot_q4_1_x4(w: &[u8], x: [Q8Slice<'_>; 4]) -> [f32; 4] {
        unsafe {
            let nb = x[0].d.len();
            debug_assert!(w.len() >= 20 * nb);
            let wp = w.as_ptr();
            let xq = [
                x[0].qs.as_ptr(),
                x[1].qs.as_ptr(),
                x[2].qs.as_ptr(),
                x[3].qs.as_ptr(),
            ];
            let xd = [
                x[0].d.as_ptr(),
                x[1].d.as_ptr(),
                x[2].d.as_ptr(),
                x[3].d.as_ptr(),
            ];
            let xs = [
                x[0].s.as_ptr(),
                x[1].s.as_ptr(),
                x[2].s.as_ptr(),
                x[3].s.as_ptr(),
            ];
            let mut acc = [vdupq_n_f32(0.0); 4];
            let mut summs = [0.0f32; 4];
            for i in 0..nb {
                let b = wp.add(20 * i);
                let (lo, hi) = q4_nibbles(b.add(4));
                let d = f16_to_f32(rd16p(b));
                let m = f16_to_f32(rd16p(b.add(2)));
                for t in 0..4 {
                    let p = dot32(vdupq_n_s32(0), lo, hi, xq[t].add(32 * i));
                    acc[t] = vfmaq_n_f32(acc[t], vcvtq_f32_s32(p), d * *xd[t].add(i));
                    summs[t] += m * *xs[t].add(i);
                }
            }
            [
                vaddvq_f32(acc[0]) + summs[0],
                vaddvq_f32(acc[1]) + summs[1],
                vaddvq_f32(acc[2]) + summs[2],
                vaddvq_f32(acc[3]) + summs[3],
            ]
        }
    }

    #[target_feature(enable = "neon,dotprod")]
    pub unsafe fn dot_q8_0(w: &[u8], x: Q8Slice<'_>) -> f32 {
        unsafe {
            let nb = x.d.len();
            let mut acc = vdupq_n_f32(0.0);
            for i in 0..nb {
                let b = w.as_ptr().add(34 * i + 2) as *const i8;
                let p = dot32(
                    vdupq_n_s32(0),
                    vld1q_s8(b),
                    vld1q_s8(b.add(16)),
                    x.qs.as_ptr().add(32 * i),
                );
                acc = vfmaq_n_f32(acc, vcvtq_f32_s32(p), f16_to_f32(rd16(w, 34 * i)) * x.d[i]);
            }
            vaddvq_f32(acc)
        }
    }

    #[target_feature(enable = "neon,dotprod")]
    pub unsafe fn dot_q4k(w: &[u8], x: Q8Slice<'_>) -> f32 {
        unsafe {
            let nsb = x.d.len() / 8;
            let m4 = vdupq_n_u8(0x0F);
            let mut sum = 0.0f32;
            for i in 0..nsb {
                let b = &w[144 * i..144 * (i + 1)];
                let d = f16_to_f32(rd16(b, 0));
                let min = f16_to_f32(rd16(b, 2));
                let scales = &b[4..16];
                let qs = b.as_ptr().add(16);
                let xq = x.qs.as_ptr().add(256 * i);
                for j in 0..4 {
                    let v0 = vld1q_u8(qs.add(32 * j));
                    let v1 = vld1q_u8(qs.add(32 * j + 16));
                    let lo_p = dot32(
                        vdupq_n_s32(0),
                        vreinterpretq_s8_u8(vandq_u8(v0, m4)),
                        vreinterpretq_s8_u8(vandq_u8(v1, m4)),
                        xq.add(64 * j),
                    );
                    let hi_p = dot32(
                        vdupq_n_s32(0),
                        vreinterpretq_s8_u8(vshrq_n_u8::<4>(v0)),
                        vreinterpretq_s8_u8(vshrq_n_u8::<4>(v1)),
                        xq.add(64 * j + 32),
                    );
                    let (s1, m1) = scale_min_k4(2 * j, scales);
                    let (s2, m2) = scale_min_k4(2 * j + 1, scales);
                    let (xa, xb) = (8 * i + 2 * j, 8 * i + 2 * j + 1);
                    sum += d
                        * (s1 as f32 * x.d[xa] * vaddvq_s32(lo_p) as f32
                            + s2 as f32 * x.d[xb] * vaddvq_s32(hi_p) as f32)
                        - min * (m1 as f32 * x.s[xa] + m2 as f32 * x.s[xb]);
                }
            }
            sum
        }
    }

    #[target_feature(enable = "neon,dotprod")]
    pub unsafe fn dot_q5k(w: &[u8], x: Q8Slice<'_>) -> f32 {
        unsafe {
            let nsb = x.d.len() / 8;
            let m4 = vdupq_n_u8(0x0F);
            let b16 = vdupq_n_u8(16);
            let mut sum = 0.0f32;
            for i in 0..nsb {
                let b = &w[176 * i..176 * (i + 1)];
                let d = f16_to_f32(rd16(b, 0));
                let min = f16_to_f32(rd16(b, 2));
                let scales = &b[4..16];
                let qh0 = vld1q_u8(b.as_ptr().add(16));
                let qh1 = vld1q_u8(b.as_ptr().add(32));
                let qs = b.as_ptr().add(48);
                let xq = x.qs.as_ptr().add(256 * i);
                for j in 0..4 {
                    let v0 = vld1q_u8(qs.add(32 * j));
                    let v1 = vld1q_u8(qs.add(32 * j + 16));
                    let u1 = vdupq_n_u8(1 << (2 * j));
                    let u2 = vdupq_n_u8(2 << (2 * j));
                    let lo0 = vorrq_u8(vandq_u8(v0, m4), vandq_u8(vtstq_u8(qh0, u1), b16));
                    let lo1 = vorrq_u8(vandq_u8(v1, m4), vandq_u8(vtstq_u8(qh1, u1), b16));
                    let hi0 = vorrq_u8(vshrq_n_u8::<4>(v0), vandq_u8(vtstq_u8(qh0, u2), b16));
                    let hi1 = vorrq_u8(vshrq_n_u8::<4>(v1), vandq_u8(vtstq_u8(qh1, u2), b16));
                    let lo_p = dot32(
                        vdupq_n_s32(0),
                        vreinterpretq_s8_u8(lo0),
                        vreinterpretq_s8_u8(lo1),
                        xq.add(64 * j),
                    );
                    let hi_p = dot32(
                        vdupq_n_s32(0),
                        vreinterpretq_s8_u8(hi0),
                        vreinterpretq_s8_u8(hi1),
                        xq.add(64 * j + 32),
                    );
                    let (s1, m1) = scale_min_k4(2 * j, scales);
                    let (s2, m2) = scale_min_k4(2 * j + 1, scales);
                    let (xa, xb) = (8 * i + 2 * j, 8 * i + 2 * j + 1);
                    sum += d
                        * (s1 as f32 * x.d[xa] * vaddvq_s32(lo_p) as f32
                            + s2 as f32 * x.d[xb] * vaddvq_s32(hi_p) as f32)
                        - min * (m1 as f32 * x.s[xa] + m2 as f32 * x.s[xb]);
                }
            }
            sum
        }
    }

    #[target_feature(enable = "neon,dotprod")]
    pub unsafe fn dot_q6k(w: &[u8], x: Q8Slice<'_>) -> f32 {
        unsafe {
            let nsb = x.d.len() / 8;
            let m4 = vdupq_n_u8(0x0F);
            let m3 = vdupq_n_u8(0x03);
            let s32 = vdupq_n_s8(32);
            let mut sum = 0.0f32;
            for i in 0..nsb {
                let b = w.as_ptr().add(210 * i);
                let d = f16_to_f32(rd16(w, 210 * i + 208));
                let xq = x.qs.as_ptr().add(256 * i);
                for n in 0..2 {
                    let ql = b.add(64 * n);
                    let qh = b.add(128 + 32 * n);
                    let sc = b.add(192 + 8 * n) as *const i8;
                    let qh0 = vld1q_u8(qh);
                    let qh1 = vld1q_u8(qh.add(16));
                    let ql0 = vld1q_u8(ql);
                    let ql1 = vld1q_u8(ql.add(16));
                    let ql2 = vld1q_u8(ql.add(32));
                    let ql3 = vld1q_u8(ql.add(48));
                    let mk = |lo: uint8x16_t, h: uint8x16_t| {
                        vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(lo, vshlq_n_u8::<4>(h))), s32)
                    };
                    let q = [
                        mk(vandq_u8(ql0, m4), vandq_u8(qh0, m3)),
                        mk(vandq_u8(ql1, m4), vandq_u8(qh1, m3)),
                        mk(vandq_u8(ql2, m4), vandq_u8(vshrq_n_u8::<2>(qh0), m3)),
                        mk(vandq_u8(ql3, m4), vandq_u8(vshrq_n_u8::<2>(qh1), m3)),
                        mk(vshrq_n_u8::<4>(ql0), vandq_u8(vshrq_n_u8::<4>(qh0), m3)),
                        mk(vshrq_n_u8::<4>(ql1), vandq_u8(vshrq_n_u8::<4>(qh1), m3)),
                        mk(vshrq_n_u8::<4>(ql2), vshrq_n_u8::<6>(qh0)),
                        mk(vshrq_n_u8::<4>(ql3), vshrq_n_u8::<6>(qh1)),
                    ];
                    let xh = xq.add(128 * n);
                    for k in 0..4 {
                        let pa = vdotq_s32(vdupq_n_s32(0), q[2 * k], vld1q_s8(xh.add(32 * k)));
                        let pb =
                            vdotq_s32(vdupq_n_s32(0), q[2 * k + 1], vld1q_s8(xh.add(32 * k + 16)));
                        let isum = vaddvq_s32(pa) * *sc.add(2 * k) as i32
                            + vaddvq_s32(pb) * *sc.add(2 * k + 1) as i32;
                        sum += d * x.d[8 * i + 4 * n + k] * isum as f32;
                    }
                }
            }
            sum
        }
    }
}

/// x86-64 kernels, picked at run time so that a baseline build still runs
/// everywhere: AVX2 + FMA + F16C (Haswell / Zen and later), and the same int8
/// kernels with AVX-VNNI (Alder Lake / Zen 5 and later). AVX2 multiplies int8
/// with `maddubs` (unsigned x signed bytes, pairs summed into i16) then `madd`
/// into i32; VNNI's `vpdpbusd` does both in one instruction. Signed weights go
/// through the `sign` trick. No pair sum can saturate its i16: weights are at
/// most 128 in magnitude and activations at most 127, so both give the same
/// i32 lanes, which go through the same float operations: the VNNI kernels'
/// results are bit for bit those of the AVX2 ones.
#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::{Q8Slice, scale_min_k4};
    use core::arch::x86_64::*;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    pub enum Level {
        Scalar,
        Avx2,
        AvxVnni,
    }

    /// The best kernels the CPU can run: the cpuid result is cached, so this
    /// is a load, and a constant when the build targets all the features
    #[inline]
    pub fn level() -> Level {
        use core::sync::atomic::{AtomicU8, Ordering};
        const UNKNOWN: u8 = 0;
        static DETECTED: AtomicU8 = AtomicU8::new(UNKNOWN);

        if cfg!(all(
            target_feature = "avx2",
            target_feature = "fma",
            target_feature = "f16c",
            target_feature = "avxvnni"
        )) {
            return Level::AvxVnni;
        }
        let level = match DETECTED.load(Ordering::Relaxed) {
            UNKNOWN => {
                let level = detect();
                DETECTED.store(level as u8 + 1, Ordering::Relaxed);
                level
            }
            1 => Level::Scalar,
            2 => Level::Avx2,
            _ => Level::AvxVnni,
        };
        if cfg!(all(
            target_feature = "avx2",
            target_feature = "fma",
            target_feature = "f16c"
        )) {
            level.max(Level::Avx2)
        } else {
            level
        }
    }

    #[inline]
    pub fn has_avx2() -> bool {
        level() >= Level::Avx2
    }

    /// AVX2, FMA and F16C from cpuid, and the OS saving the AVX registers
    /// on context switches (XCR0), without which AVX faults; then AVX-VNNI
    #[cold]
    fn detect() -> Level {
        let leaf1 = __cpuid(1);
        let bit = |r: u32, b: u32| r & (1 << b) != 0;
        let (fma, osxsave, avx, f16c) = (
            bit(leaf1.ecx, 12),
            bit(leaf1.ecx, 27),
            bit(leaf1.ecx, 28),
            bit(leaf1.ecx, 29),
        );
        if !(fma && osxsave && avx && f16c) || __cpuid(0).eax < 7 {
            return Level::Scalar;
        }
        // SAFETY: OSXSAVE is set, so xgetbv is available
        let xcr0 = unsafe { _xgetbv(0) };
        let ymm_saved = xcr0 & 0b110 == 0b110; // SSE and AVX state
        if !(ymm_saved && bit(__cpuid_count(7, 0).ebx, 5)) {
            return Level::Scalar;
        }
        if bit(__cpuid_count(7, 1).eax, 4) {
            Level::AvxVnni
        } else {
            Level::Avx2
        }
    }

    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn f16(p: *const u8) -> f32 {
        unsafe {
            _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(
                (p as *const u16).read_unaligned() as i32,
            )))
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn load(p: *const u8) -> __m256i {
        unsafe { _mm256_loadu_si256(p as *const __m256i) }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn loadx(p: *const i8) -> __m256i {
        unsafe { _mm256_loadu_si256(p as *const __m256i) }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    fn hsum(v: __m256) -> f32 {
        let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        _mm_cvtss_f32(_mm_add_ss(s, _mm_movehdup_ps(s)))
    }

    /// `acc += p * d`, `p` converted from i32
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    fn fma_i32(acc: __m256, p: __m256i, d: f32) -> __m256 {
        _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(d), acc)
    }

    /// The 16 bytes of a q4 block as 32 values: low nibbles (elements 0..16),
    /// then high nibbles (16..32)
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn q4_nibbles(p: *const u8) -> __m256i {
        unsafe {
            let v = _mm_loadu_si128(p as *const __m128i);
            let m4 = _mm_set1_epi8(0x0F);
            _mm256_set_m128i(
                _mm_and_si128(_mm_srli_epi16::<4>(v), m4),
                _mm_and_si128(v, m4),
            )
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_f32(a: *const f32, b: *const f32, n: usize) -> f32 {
        unsafe {
            let mut acc = [_mm256_setzero_ps(); 4];
            let mut i = 0;
            while i + 32 <= n {
                for (k, acc) in acc.iter_mut().enumerate() {
                    *acc = _mm256_fmadd_ps(
                        _mm256_loadu_ps(a.add(i + 8 * k)),
                        _mm256_loadu_ps(b.add(i + 8 * k)),
                        *acc,
                    );
                }
                i += 32;
            }
            while i + 8 <= n {
                acc[0] =
                    _mm256_fmadd_ps(_mm256_loadu_ps(a.add(i)), _mm256_loadu_ps(b.add(i)), acc[0]);
                i += 8;
            }
            let mut s = hsum(_mm256_add_ps(
                _mm256_add_ps(acc[0], acc[1]),
                _mm256_add_ps(acc[2], acc[3]),
            ));
            while i < n {
                s += *a.add(i) * *b.add(i);
                i += 1;
            }
            s
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn axpy(y: *mut f32, a: f32, x: *const f32, n: usize) {
        unsafe {
            let va = _mm256_set1_ps(a);
            let mut i = 0;
            while i + 8 <= n {
                _mm256_storeu_ps(
                    y.add(i),
                    _mm256_fmadd_ps(va, _mm256_loadu_ps(x.add(i)), _mm256_loadu_ps(y.add(i))),
                );
                i += 8;
            }
            while i < n {
                *y.add(i) += a * *x.add(i);
                i += 1;
            }
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_bf16(w: *const u8, x: *const f32, n: usize) -> f32 {
        unsafe {
            let w = w as *const u16;
            let widen = |p: *const u16| {
                _mm256_castsi256_ps(_mm256_slli_epi32::<16>(_mm256_cvtepu16_epi32(
                    _mm_loadu_si128(p as *const __m128i),
                )))
            };
            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            let mut i = 0;
            while i + 16 <= n {
                acc0 = _mm256_fmadd_ps(widen(w.add(i)), _mm256_loadu_ps(x.add(i)), acc0);
                acc1 = _mm256_fmadd_ps(widen(w.add(i + 8)), _mm256_loadu_ps(x.add(i + 8)), acc1);
                i += 16;
            }
            if i + 8 <= n {
                acc0 = _mm256_fmadd_ps(widen(w.add(i)), _mm256_loadu_ps(x.add(i)), acc0);
                i += 8;
            }
            let mut s = hsum(_mm256_add_ps(acc0, acc1));
            while i < n {
                s += f32::from_bits((w.add(i).read_unaligned() as u32) << 16) * *x.add(i);
                i += 1;
            }
            s
        }
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot_f16(w: *const u8, x: *const f32, n: usize) -> f32 {
        unsafe {
            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            let mut i = 0;
            while i + 16 <= n {
                let h0 = _mm256_cvtph_ps(_mm_loadu_si128(w.add(2 * i) as *const __m128i));
                let h1 = _mm256_cvtph_ps(_mm_loadu_si128(w.add(2 * i + 16) as *const __m128i));
                acc0 = _mm256_fmadd_ps(h0, _mm256_loadu_ps(x.add(i)), acc0);
                acc1 = _mm256_fmadd_ps(h1, _mm256_loadu_ps(x.add(i + 8)), acc1);
                i += 16;
            }
            let mut s = hsum(_mm256_add_ps(acc0, acc1));
            while i < n {
                s += f16(w.add(2 * i)) * *x.add(i);
                i += 1;
            }
            s
        }
    }

    /// The int8 dot kernels, compiled for the target features `$feat`, with
    /// `$dot(u, s)` summing the products of unsigned bytes `u` and signed bytes
    /// `s` into 8 i32 (each of 4 consecutive bytes)
    macro_rules! int8_kernels {
        ($m:ident, $feat:literal, |$u:ident, $s:ident| $dot:expr) => {
            pub mod $m {
                use super::*;

                #[inline]
                #[target_feature(enable = $feat)]
                fn dot_u8_i8($u: __m256i, $s: __m256i) -> __m256i {
                    $dot
                }

                /// Products of signed bytes `a` and `s`, summed into 8 i32
                #[inline]
                #[target_feature(enable = $feat)]
                fn dot_i8_i8(a: __m256i, s: __m256i) -> __m256i {
                    dot_u8_i8(_mm256_sign_epi8(a, a), _mm256_sign_epi8(s, a))
                }

                #[target_feature(enable = $feat)]
                pub unsafe fn dot_q4_0(w: &[u8], x: Q8Slice<'_>) -> f32 {
                    unsafe {
                        let nb = x.d.len();
                        assert!(w.len() >= 18 * nb && x.qs.len() >= 32 * nb);
                        let (wp, xq) = (w.as_ptr(), x.qs.as_ptr());
                        let s8 = _mm256_set1_epi8(8);
                        let mut acc = [_mm256_setzero_ps(); 2];
                        for i in 0..nb {
                            let b = wp.add(18 * i);
                            let q = _mm256_sub_epi8(q4_nibbles(b.add(2)), s8);
                            acc[i & 1] = fma_i32(
                                acc[i & 1],
                                dot_i8_i8(q, loadx(xq.add(32 * i))),
                                f16(b) * x.d[i],
                            );
                        }
                        hsum(_mm256_add_ps(acc[0], acc[1]))
                    }
                }

                #[target_feature(enable = $feat)]
                pub unsafe fn dot_q4_1(w: &[u8], x: Q8Slice<'_>) -> f32 {
                    unsafe {
                        let nb = x.d.len();
                        assert!(w.len() >= 20 * nb && x.qs.len() >= 32 * nb && x.s.len() >= nb);
                        let (wp, xq) = (w.as_ptr(), x.qs.as_ptr());
                        let mut acc = [_mm256_setzero_ps(); 2];
                        let mut summs = 0.0f32;
                        for i in 0..nb {
                            let b = wp.add(20 * i);
                            let p = dot_u8_i8(q4_nibbles(b.add(4)), loadx(xq.add(32 * i)));
                            acc[i & 1] = fma_i32(acc[i & 1], p, f16(b) * x.d[i]);
                            summs += f16(b.add(2)) * x.s[i];
                        }
                        hsum(_mm256_add_ps(acc[0], acc[1])) + summs
                    }
                }

                /// One q4_1 row against four activation rows: the block is unpacked once
                #[target_feature(enable = $feat)]
                pub unsafe fn dot_q4_1_x4(w: &[u8], x: [Q8Slice<'_>; 4]) -> [f32; 4] {
                    unsafe {
                        let nb = x[0].d.len();
                        assert!(w.len() >= 20 * nb);
                        assert!(
                            x.iter().all(|x| x.qs.len() >= 32 * nb
                                && x.d.len() >= nb
                                && x.s.len() >= nb)
                        );
                        let wp = w.as_ptr();
                        let mut acc = [_mm256_setzero_ps(); 4];
                        let mut summs = [0.0f32; 4];
                        for i in 0..nb {
                            let b = wp.add(20 * i);
                            let q = q4_nibbles(b.add(4));
                            let (d, m) = (f16(b), f16(b.add(2)));
                            for t in 0..4 {
                                let p = dot_u8_i8(q, loadx(x[t].qs.as_ptr().add(32 * i)));
                                acc[t] = fma_i32(acc[t], p, d * *x[t].d.get_unchecked(i));
                                summs[t] += m * *x[t].s.get_unchecked(i);
                            }
                        }
                        core::array::from_fn(|t| hsum(acc[t]) + summs[t])
                    }
                }

                #[target_feature(enable = $feat)]
                pub unsafe fn dot_q8_0(w: &[u8], x: Q8Slice<'_>) -> f32 {
                    unsafe {
                        let nb = x.d.len();
                        assert!(w.len() >= 34 * nb && x.qs.len() >= 32 * nb);
                        let (wp, xq) = (w.as_ptr(), x.qs.as_ptr());
                        let mut acc = [_mm256_setzero_ps(); 2];
                        for i in 0..nb {
                            let b = wp.add(34 * i);
                            let p = dot_i8_i8(load(b.add(2)), loadx(xq.add(32 * i)));
                            acc[i & 1] = fma_i32(acc[i & 1], p, f16(b) * x.d[i]);
                        }
                        hsum(_mm256_add_ps(acc[0], acc[1]))
                    }
                }

                #[target_feature(enable = $feat)]
                pub unsafe fn dot_q4k(w: &[u8], x: Q8Slice<'_>) -> f32 {
                    unsafe {
                        let nsb = x.d.len() / 8;
                        assert!(
                            w.len() >= 144 * nsb && x.qs.len() >= 256 * nsb && x.s.len() >= 8 * nsb
                        );
                        let m4 = _mm256_set1_epi8(0x0F);
                        let mut acc_lo = _mm256_setzero_ps();
                        let mut acc_hi = _mm256_setzero_ps();
                        let mut mins = 0.0f32;
                        for i in 0..nsb {
                            let b = w.as_ptr().add(144 * i);
                            let (d, dmin) = (f16(b), f16(b.add(2)));
                            let scales = &w[144 * i + 4..144 * i + 16];
                            let xq = x.qs.as_ptr().add(256 * i);
                            for j in 0..4 {
                                // 32 bytes: low nibbles are sub-block 2j, high nibbles 2j + 1
                                let v = load(b.add(16 + 32 * j));
                                let lo = _mm256_and_si256(v, m4);
                                let hi = _mm256_and_si256(_mm256_srli_epi16::<4>(v), m4);
                                let (s1, m1) = scale_min_k4(2 * j, scales);
                                let (s2, m2) = scale_min_k4(2 * j + 1, scales);
                                let (xa, xb) = (8 * i + 2 * j, 8 * i + 2 * j + 1);
                                acc_lo = fma_i32(
                                    acc_lo,
                                    dot_u8_i8(lo, loadx(xq.add(64 * j))),
                                    d * s1 as f32 * x.d[xa],
                                );
                                acc_hi = fma_i32(
                                    acc_hi,
                                    dot_u8_i8(hi, loadx(xq.add(64 * j + 32))),
                                    d * s2 as f32 * x.d[xb],
                                );
                                mins += dmin * (m1 as f32 * x.s[xa] + m2 as f32 * x.s[xb]);
                            }
                        }
                        hsum(_mm256_add_ps(acc_lo, acc_hi)) - mins
                    }
                }

                #[target_feature(enable = $feat)]
                pub unsafe fn dot_q5k(w: &[u8], x: Q8Slice<'_>) -> f32 {
                    unsafe {
                        let nsb = x.d.len() / 8;
                        assert!(
                            w.len() >= 176 * nsb && x.qs.len() >= 256 * nsb && x.s.len() >= 8 * nsb
                        );
                        let m4 = _mm256_set1_epi8(0x0F);
                        let b16 = _mm256_set1_epi8(16);
                        // 16 where bit `bit` of the high-bit byte is set
                        let high = |qh: __m256i, bit: usize| {
                            let m = _mm256_set1_epi8((1u8 << bit) as i8);
                            _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(qh, m), m), b16)
                        };
                        let mut acc_lo = _mm256_setzero_ps();
                        let mut acc_hi = _mm256_setzero_ps();
                        let mut mins = 0.0f32;
                        for i in 0..nsb {
                            let b = w.as_ptr().add(176 * i);
                            let (d, dmin) = (f16(b), f16(b.add(2)));
                            let scales = &w[176 * i + 4..176 * i + 16];
                            let qh = load(b.add(16));
                            let xq = x.qs.as_ptr().add(256 * i);
                            for j in 0..4 {
                                let v = load(b.add(48 + 32 * j));
                                let lo = _mm256_or_si256(_mm256_and_si256(v, m4), high(qh, 2 * j));
                                let hi = _mm256_or_si256(
                                    _mm256_and_si256(_mm256_srli_epi16::<4>(v), m4),
                                    high(qh, 2 * j + 1),
                                );
                                let (s1, m1) = scale_min_k4(2 * j, scales);
                                let (s2, m2) = scale_min_k4(2 * j + 1, scales);
                                let (xa, xb) = (8 * i + 2 * j, 8 * i + 2 * j + 1);
                                acc_lo = fma_i32(
                                    acc_lo,
                                    dot_u8_i8(lo, loadx(xq.add(64 * j))),
                                    d * s1 as f32 * x.d[xa],
                                );
                                acc_hi = fma_i32(
                                    acc_hi,
                                    dot_u8_i8(hi, loadx(xq.add(64 * j + 32))),
                                    d * s2 as f32 * x.d[xb],
                                );
                                mins += dmin * (m1 as f32 * x.s[xa] + m2 as f32 * x.s[xb]);
                            }
                        }
                        hsum(_mm256_add_ps(acc_lo, acc_hi)) - mins
                    }
                }

                #[target_feature(enable = $feat)]
                pub unsafe fn dot_q6k(w: &[u8], x: Q8Slice<'_>) -> f32 {
                    unsafe {
                        let nsb = x.d.len() / 8;
                        assert!(w.len() >= 210 * nsb && x.qs.len() >= 256 * nsb);
                        let m4 = _mm256_set1_epi8(0x0F);
                        let m3 = _mm256_set1_epi8(0x03);
                        let s32 = _mm256_set1_epi8(32);
                        let mut acc = [_mm256_setzero_ps(); 2];
                        for i in 0..nsb {
                            let b = w.as_ptr().add(210 * i);
                            let d = f16(b.add(208));
                            let xq = x.qs.as_ptr().add(256 * i);
                            for n in 0..2 {
                                let ql0 = load(b.add(64 * n));
                                let ql1 = load(b.add(64 * n + 32));
                                let qh = load(b.add(128 + 32 * n));
                                let sc = b.add(192 + 8 * n) as *const i8;
                                // 6-bit values in 0..64 (the -32 offset is applied below)
                                let mk = |lo: __m256i, h: __m256i| {
                                    _mm256_or_si256(
                                        lo,
                                        _mm256_slli_epi16::<4>(_mm256_and_si256(h, m3)),
                                    )
                                };
                                let q = [
                                    mk(_mm256_and_si256(ql0, m4), qh),
                                    mk(_mm256_and_si256(ql1, m4), _mm256_srli_epi16::<2>(qh)),
                                    mk(
                                        _mm256_and_si256(_mm256_srli_epi16::<4>(ql0), m4),
                                        _mm256_srli_epi16::<4>(qh),
                                    ),
                                    mk(
                                        _mm256_and_si256(_mm256_srli_epi16::<4>(ql1), m4),
                                        _mm256_srli_epi16::<6>(qh),
                                    ),
                                ];
                                for (k, &q) in q.iter().enumerate() {
                                    let xv = loadx(xq.add(128 * n + 32 * k));
                                    let p = _mm256_sub_epi16(
                                        _mm256_maddubs_epi16(q, xv),
                                        _mm256_maddubs_epi16(s32, xv),
                                    );
                                    // 16 elements per scale: i16 lanes 0..8 and 8..16
                                    let scl = _mm256_set_m128i(
                                        _mm_set1_epi16(*sc.add(2 * k + 1) as i16),
                                        _mm_set1_epi16(*sc.add(2 * k) as i16),
                                    );
                                    acc[k & 1] = fma_i32(
                                        acc[k & 1],
                                        _mm256_madd_epi16(p, scl),
                                        d * x.d[8 * i + 4 * n + k],
                                    );
                                }
                            }
                        }
                        hsum(_mm256_add_ps(acc[0], acc[1]))
                    }
                }
            }
        };
    }

    int8_kernels!(avx2, "avx2,fma,f16c", |u, s| _mm256_madd_epi16(
        _mm256_maddubs_epi16(u, s),
        _mm256_set1_epi16(1)
    ));
    int8_kernels!(vnni, "avx2,fma,f16c,avxvnni", |u, s| {
        _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), u, s)
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    #[test]
    fn avx2_detection_matches_std() {
        use std::arch::is_x86_feature_detected;
        let std = is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
            && is_x86_feature_detected!("f16c");
        assert_eq!(x86::has_avx2(), std);
        let vnni = std && is_x86_feature_detected!("avxvnni");
        assert_eq!(x86::level() == x86::Level::AvxVnni, vnni);
    }

    /// The VNNI kernels give the AVX2 kernels' results bit for bit
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn vnni_matches_avx2() {
        if x86::level() < x86::Level::AvxVnni {
            return;
        }
        use x86::{avx2, vnni};
        type Kernel = unsafe fn(&[u8], Q8Slice<'_>) -> f32;
        let mut seed = 0x0bad_cafe_dead_beefu64;
        let n = 1024;
        let x: Vec<f32> = (0..n)
            .map(|i| ((i * 7919) % 211) as f32 / 50.0 - 2.0)
            .collect();
        // full-range activations, -127 and 127 included
        let xs: [Vec<f32>; 4] = core::array::from_fn(|t| {
            x.iter()
                .enumerate()
                .map(|(i, v)| {
                    if i % 5 == t {
                        1e3 * v.signum()
                    } else {
                        v * (t as f32 - 1.5)
                    }
                })
                .collect()
        });
        let xqs: Vec<Q8Act> = xs.iter().map(|x| Q8Act::quantize(x)).collect();
        let pairs: [(GgmlType, Kernel, Kernel); 6] = [
            (GgmlType::Q4_0, avx2::dot_q4_0, vnni::dot_q4_0),
            (GgmlType::Q4_1, avx2::dot_q4_1, vnni::dot_q4_1),
            (GgmlType::Q8_0, avx2::dot_q8_0, vnni::dot_q8_0),
            (GgmlType::Q4K, avx2::dot_q4k, vnni::dot_q4k),
            (GgmlType::Q5K, avx2::dot_q5k, vnni::dot_q5k),
            (GgmlType::Q6K, avx2::dot_q6k, vnni::dot_q6k),
        ];
        for (ty, a, b) in pairs {
            let mut w = rand_row(ty, n, &mut seed);
            if ty == GgmlType::Q8_0 {
                // -128, the weight the sign trick can't negate
                w[2] = 0x80;
            }
            for xq in &xqs {
                let (a, b) = unsafe { (a(&w, xq.slice(0, n)), b(&w, xq.slice(0, n))) };
                assert_eq!(a.to_bits(), b.to_bits(), "{ty:?}: {a} vs {b}");
            }
        }
        let w = rand_row(GgmlType::Q4_1, n, &mut seed);
        let x4 = core::array::from_fn(|t| xqs[t].slice(0, n));
        let (a, b) = unsafe { (avx2::dot_q4_1_x4(&w, x4), vnni::dot_q4_1_x4(&w, x4)) };
        assert_eq!(a.map(f32::to_bits), b.map(f32::to_bits));
    }

    fn rand_bytes(n: usize, seed: &mut u64) -> Vec<u8> {
        (0..n)
            .map(|_| {
                *seed ^= *seed << 13;
                *seed ^= *seed >> 7;
                *seed ^= *seed << 17;
                (*seed >> 24) as u8
            })
            .collect()
    }

    /// random blocks, with sane f16 scales so the values stay finite
    fn rand_row(ty: GgmlType, n: usize, seed: &mut u64) -> Vec<u8> {
        let mut b = rand_bytes(ty.row_size(n), seed);
        let (_, bs) = ty.block();
        let half = 0x2E66u16.to_le_bytes(); // ~0.1
        let neg = 0xAE66u16.to_le_bytes(); // ~-0.1
        for blk in b.chunks_exact_mut(bs) {
            match ty {
                GgmlType::Q4_0 | GgmlType::Q8_0 => blk[0..2].copy_from_slice(&half),
                GgmlType::Q4_1 => {
                    blk[0..2].copy_from_slice(&half);
                    blk[2..4].copy_from_slice(&neg);
                }
                GgmlType::Q4K | GgmlType::Q5K => {
                    blk[0..2].copy_from_slice(&half);
                    blk[2..4].copy_from_slice(&half);
                }
                GgmlType::Q6K => blk[208..210].copy_from_slice(&half),
                _ => {}
            }
        }
        b
    }

    #[test]
    fn kernels_match_dequantized_dot() {
        let mut seed = 0x1234_5678_9abc_def1u64;
        let n = 1024;
        let x: Vec<f32> = (0..n)
            .map(|i| ((i * 7919) % 211) as f32 / 50.0 - 2.0)
            .collect();
        let xq = Q8Act::quantize(&x);
        // dequantized activations, so only the weight path is compared
        let xd: Vec<f32> = (0..n).map(|i| xq.qs[i] as f32 * xq.d[i / 32]).collect();
        for ty in [
            GgmlType::Q4_0,
            GgmlType::Q4_1,
            GgmlType::Q8_0,
            GgmlType::Q4K,
            GgmlType::Q5K,
            GgmlType::Q6K,
        ] {
            let w = rand_row(ty, n, &mut seed);
            let mut wd = vec![0.0; n];
            dequantize_row(ty, &w, &mut wd);
            let reference: f32 = wd.iter().zip(&xd).map(|(a, b)| a * b).sum();
            let got = dot(ty, &w, &x, Some(xq.slice(0, n)));
            let tol = 1e-3 * reference.abs().max(1.0);
            assert!(
                (got - reference).abs() < tol,
                "{ty:?}: {got} vs {reference}"
            );
            let sc = match ty {
                GgmlType::Q4_0 => scalar::dot_q4_0(&w, xq.slice(0, n)),
                GgmlType::Q4_1 => scalar::dot_q4_1(&w, xq.slice(0, n)),
                GgmlType::Q8_0 => scalar::dot_q8_0(&w, xq.slice(0, n)),
                GgmlType::Q4K => scalar::dot_q4k(&w, xq.slice(0, n)),
                GgmlType::Q5K => scalar::dot_q5k(&w, xq.slice(0, n)),
                GgmlType::Q6K => scalar::dot_q6k(&w, xq.slice(0, n)),
                _ => unreachable!(),
            };
            assert!(
                (sc - reference).abs() < tol,
                "{ty:?} scalar: {sc} vs {reference}"
            );
            // four different activation rows
            let xs: Vec<Vec<f32>> = (0..4)
                .map(|t| x.iter().map(|v| v * (t as f32 - 1.5)).collect())
                .collect();
            let xqs: Vec<Q8Act> = xs.iter().map(|x| Q8Act::quantize(x)).collect();
            let x4 = dot_x4(
                ty,
                &w,
                core::array::from_fn(|t| &xs[t][..]),
                Some(core::array::from_fn(|t| xqs[t].slice(0, n))),
            );
            for t in 0..4 {
                let one = dot(ty, &w, &xs[t], Some(xqs[t].slice(0, n)));
                assert!(
                    (x4[t] - one).abs() < tol,
                    "{ty:?} x4 row {t}: {} vs {one}",
                    x4[t]
                );
            }
        }
    }

    #[test]
    fn bf16_and_f16() {
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0xC000), -2.0);
        assert_eq!(f16_to_f32(0x0001), 5.960464477539063e-8);
        assert_eq!(bf16_to_f32(0x3F80), 1.0);
        let x: Vec<f32> = (0..37).map(|i| i as f32 * 0.25).collect();
        let w: Vec<u8> = x
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let want: f32 = x.iter().map(|v| v * v).sum();
        assert!((dot(GgmlType::BF16, &w, &x, None) - want).abs() < 1e-3);
        let w16: Vec<u8> = x
            .iter()
            .flat_map(|&v| f32_to_f16_exact(v).to_le_bytes())
            .collect();
        assert!((dot(GgmlType::F16, &w16, &x, None) - want).abs() < 1e-3);
        let w32: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert!((dot(GgmlType::F32, &w32, &x, None) - want).abs() < 1e-3);
        assert!((dot_f32(&x, &x) - want).abs() < 1e-3);
        let mut y = x.clone();
        axpy(&mut y, 2.0, &x);
        assert!(y.iter().zip(&x).all(|(y, x)| *y == 3.0 * x));
    }

    /// f16 bits of a value with an exact f16 form (multiples of 1/4 below 512)
    fn f32_to_f16_exact(v: f32) -> u16 {
        if v == 0.0 {
            return 0;
        }
        let b = v.to_bits();
        let exp = ((b >> 23) & 0xFF) as i32 - 127 + 15;
        ((b >> 16) & 0x8000) as u16 | ((exp as u16) << 10) | ((b >> 13) & 0x3FF) as u16
    }
}
