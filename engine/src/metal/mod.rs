//! Metal backend: the whole forward pass of a batch is encoded into one
//! command buffer. Weights are not copied: the GPU reads the mmap'd GGUF
//! through no-copy buffers (unified memory). Small vectors (norms, conv
//! weights) are uploaded; activations and states live in shared buffers.

use std::ffi::c_void;
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLLibrary, MTLResourceOptions, MTLSize,
};

use crate::Error;
use crate::engine::{DraftLogits, Engine, Session, Snapshot};
use crate::model::{Config, DRAFT_CAP, DraftVocab, Layer, Mixer, Model, copy_head_rows};
use crate::ops::Weight;
use crate::prelude::*;
use crate::quant::GgmlType;
use crate::tokenizer::TokenId;

const SOURCE: &str = include_str!("kernels.metal");

type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
type Device = Retained<ProtocolObject<dyn MTLDevice>>;
type Encoder = ProtocolObject<dyn MTLComputeCommandEncoder>;

/// Virtual memory page, the alignment of no-copy buffers
const PAGE: usize = if cfg!(target_arch = "aarch64") {
    16384
} else {
    4096
};

/// Matmul inputs of up to this many rows use the matvec kernels, one
/// threadgroup per input row: the weights of the second and third mostly come
/// from the cache (a verify pass of 2 tokens costs 1.2x a decode step, 3 1.6x)
const MV_MAX_ROWS: usize = 3;
/// Matmul inputs of up to this many rows use the `mmv` kernels, which read the
/// weights once for 8 input rows (on an M2 Max, a pass of 4 to 8 tokens costs
/// 1.75-1.9x a decode step, 12 to 16 3.2-3.4x); more use the `mm`
/// kernels, whose tiles are 32 rows (~4.7x from 5 to 32 tokens)
const MMV_MAX_ROWS: usize = 16;
/// Input rows and weight rows (2 simdgroups of 8) per `mmv` threadgroup
const MMV_ROWS: usize = 8;
const MMV_WEIGHT_ROWS: usize = 16;

/// Attention positions per split-K range, at least
const ATTN_CHUNK: usize = 32;
/// Split-K ranges per (token, head), at most. About 64 ranges keep the GPU
/// busy at any context length; more only add merge work.
const ATTN_MAX_SPLITS: usize = 64;
/// (query heads per kv head, simdgroups per threadgroup) of the `attn_split`
/// kernels, see the instantiations in the shaders
const ATTN_GROUPS: &[(usize, usize)] = &[(1, 2), (2, 2), (4, 2), (6, 3), (8, 4)];
const ATTN_HEAD_DIMS: &[usize] = &[128, 256];
/// Batches of at least this many tokens use the `attn_flash` kernel, which is
/// 2.2-2.4x faster from 64 tokens; with fewer query rows (decode, verify) it
/// can't fill the GPU and the split-K kernel is faster
const ATTN_FLASH_MIN: usize = 32;
/// Query rows, positions per step and simdgroups of the `attn_flash` kernels,
/// see the instantiations in the shaders. The K / V caches hold a multiple of
/// `ATTN_FLASH_C` positions.
const ATTN_FLASH_R: usize = 16;
const ATTN_FLASH_C: usize = 64;
const ATTN_FLASH_NSG: usize = 4;
/// Verify batches (2 tokens to `ATTN_FLASH_MIN`) of a sequence of at least
/// this many tokens use the split `attn_flash` kernel, which reads each K / V
/// tile once for all the tokens and heads of a block rather than once per
/// token: 13% less for a 5-token verify at 32k positions on M2 Max. Shorter,
/// its ranges are too short for its overheads.
const ATTN_FLASH_SPLIT_CTX: usize = 2048;

#[repr(C)]
struct MatArgs {
    k: i32,
    rows: i32,
    n: i32,
    pad: i32,
    w_row_bytes: u64,
    x_row_bytes: u64,
}

#[repr(C)]
struct EltArgs {
    n: i32,
}

#[repr(C)]
struct NormArgs {
    n: i32,
    eps: f32,
    gated: i32,
    ostride: i32,
}

#[repr(C)]
struct AttnArgs {
    n_head: i32,
    n_kv: i32,
    hd: i32,
    n_rot: i32,
    pos0: i32,
    eps: f32,
    scale: f32,
    splits: i32,
    chunk: i32,
    n: i32,
}

#[repr(C)]
struct ConvArgs {
    conv_dim: i32,
    n: i32,
    slot: i32,
    slots: i32,
    save: i32,
}

#[repr(C)]
struct GdnArgs {
    n: i32,
    hk: i32,
    hv: i32,
    conv_dim: i32,
    eps: f32,
    slot: i32,
    slots: i32,
    save: i32,
}

/// A buffer binding: buffer and byte offset
#[derive(Clone, Copy)]
struct B<'a>(&'a Buffer, usize);

impl<'a> B<'a> {
    fn at(buf: &'a Buffer) -> Self {
        B(buf, 0)
    }

    /// `n` f32 further
    fn f32_off(self, n: usize) -> Self {
        B(self.0, self.1 + 4 * n)
    }
}

/// A weight matrix seen by the GPU
#[derive(Clone)]
struct GpuWeight {
    buf: Buffer,
    offset: usize,
    ty: GgmlType,
    cols: usize,
    rows: usize,
    row_bytes: usize,
}

/// The compute kernels, from `kernels.metal`, each with its MSL name. The
/// attention kernels are compiled for several head sizes and group sizes:
/// their variant here is the one of the model's geometry.
macro_rules! kernels {
    (|$hd:ident, $group:ident| $($(#[$m:meta])* $v:ident => $name:expr,)*) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        enum Kernel {
            $($(#[$m])* $v,)*
        }

        impl Kernel {
            /// in declaration order: `ALL[k as usize] == k`
            const ALL: &[Kernel] = &[$(Kernel::$v,)*];

            /// The MSL function, for attention heads of `hd` floats and query
            /// groups of `group` heads
            fn name(self, $hd: usize, $group: usize) -> String {
                let _ = ($hd, $group);
                match self {
                    $(Kernel::$v => String::from($name),)*
                }
            }
        }
    };
}

kernels! {
    |hd, group|
    MvQ4_0 => "mv_q4_0",
    MvQ4_1 => "mv_q4_1",
    MvQ5K => "mv_q5_K",
    MvQ6K => "mv_q6_K",
    MvF32 => "mv_f32",
    MvF16 => "mv_f16",
    MvBf16 => "mv_bf16",
    MmF32 => "mm_f32",
    MmF16 => "mm_f16",
    MmBf16 => "mm_bf16",
    MmQ4_0 => "mm_q4_0",
    MmQ4_1 => "mm_q4_1",
    MmQ8_0 => "mm_q8_0",
    MmQ4K => "mm_q4_K",
    MmQ5K => "mm_q5_K",
    MmQ6K => "mm_q6_K",
    MmvF32 => "mmv_f32",
    MmvF16 => "mmv_f16",
    MmvBf16 => "mmv_bf16",
    MmvQ4_0 => "mmv_q4_0",
    MmvQ4_1 => "mmv_q4_1",
    MmvQ8_0 => "mmv_q8_0",
    MmvQ4K => "mmv_q4_K",
    MmvQ5K => "mmv_q5_K",
    MmvQ6K => "mmv_q6_K",
    AddInplace => "add_inplace",
    Swiglu => "swiglu",
    RmsNorm => "rms_norm",
    AttnPrep => "attn_prep",
    AttnReduce => "attn_reduce",
    /// split-K attention, for short batches
    AttnSplit => format!("attn_split_{hd}_{group}"),
    AttnFlash => format!("attn_flash_{hd}"),
    /// flash attention split along the positions, for long contexts
    AttnFlashSplit => format!("attn_flash_split_{hd}"),
    SsmConv => "ssm_conv",
    Gdn128 => "gdn_128",
}

/// The pipeline of every [`Kernel`], indexed by it
struct Kernels {
    pipelines: Vec<Pipeline>,
}

impl Kernels {
    fn get(&self, k: Kernel) -> &Pipeline {
        &self.pipelines[k as usize]
    }
}

fn mv_kernel(ty: GgmlType) -> Option<(Kernel, usize, usize)> {
    // (kernel, rows per threadgroup, threads per threadgroup)
    Some(match ty {
        GgmlType::Q4_0 => (Kernel::MvQ4_0, 8, 64),
        GgmlType::Q4_1 => (Kernel::MvQ4_1, 8, 64),
        GgmlType::Q5K => (Kernel::MvQ5K, 2, 64),
        GgmlType::Q6K => (Kernel::MvQ6K, 4, 64),
        GgmlType::F32 => (Kernel::MvF32, 4, 128),
        GgmlType::F16 => (Kernel::MvF16, 4, 128),
        GgmlType::BF16 => (Kernel::MvBf16, 4, 128),
        _ => return None,
    })
}

/// The `mmv` kernels, for 4 to 16 input rows
fn mmv_kernel(ty: GgmlType) -> Option<Kernel> {
    Some(match ty {
        GgmlType::F32 => Kernel::MmvF32,
        GgmlType::F16 => Kernel::MmvF16,
        GgmlType::BF16 => Kernel::MmvBf16,
        GgmlType::Q4_0 => Kernel::MmvQ4_0,
        GgmlType::Q4_1 => Kernel::MmvQ4_1,
        GgmlType::Q8_0 => Kernel::MmvQ8_0,
        GgmlType::Q4K => Kernel::MmvQ4K,
        GgmlType::Q5K => Kernel::MmvQ5K,
        GgmlType::Q6K => Kernel::MmvQ6K,
        _ => return None,
    })
}

fn mm_kernel(ty: GgmlType) -> Option<Kernel> {
    Some(match ty {
        GgmlType::F32 => Kernel::MmF32,
        GgmlType::F16 => Kernel::MmF16,
        GgmlType::BF16 => Kernel::MmBf16,
        GgmlType::Q4_0 => Kernel::MmQ4_0,
        GgmlType::Q4_1 => Kernel::MmQ4_1,
        GgmlType::Q8_0 => Kernel::MmQ8_0,
        GgmlType::Q4K => Kernel::MmQ4K,
        GgmlType::Q5K => Kernel::MmQ5K,
        GgmlType::Q6K => Kernel::MmQ6K,
        _ => return None,
    })
}

enum GpuMixer {
    Attention {
        wq: GpuWeight,
        wk: GpuWeight,
        wv: GpuWeight,
        wo: GpuWeight,
        q_norm: Buffer,
        k_norm: Buffer,
    },
    DeltaNet {
        wqkv: GpuWeight,
        wgate: GpuWeight,
        wbeta: GpuWeight,
        walpha: GpuWeight,
        wout: GpuWeight,
        conv1d: Buffer,
        dt_bias: Buffer,
        a: Buffer,
        norm: Buffer,
    },
}

struct GpuLayer {
    attn_norm: Buffer,
    post_norm: Buffer,
    mixer: GpuMixer,
    ffn_gate: GpuWeight,
    ffn_up: GpuWeight,
    ffn_down: GpuWeight,
}

/// The MTP head, see [`crate::model::Mtp`]
struct GpuMtp {
    enorm: Buffer,
    hnorm: Buffer,
    eh_proj: GpuWeight,
    layer: GpuLayer,
    head_norm: Buffer,
}

pub struct MetalEngine<'m> {
    model: &'m Model<'m>,
    device: Device,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    k: Kernels,
    layers: Vec<GpuLayer>,
    mtp: Option<GpuMtp>,
    output_norm: Buffer,
    output: GpuWeight,
    inv_freq: Buffer,
    attn_nsg: usize,
    max_batch: usize,
}

fn upload(device: &Device, data: &[f32]) -> Result<Buffer, Error> {
    let bytes = (data.len() * 4).max(4);
    let buf = device
        .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
        .ok_or(Error::OutOfMemory { bytes })?;
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr(),
            buf.contents().as_ptr() as *mut f32,
            data.len(),
        );
    }
    Ok(buf)
}

fn zeroed(device: &Device, bytes: usize) -> Result<Buffer, Error> {
    let buf = device
        .newBufferWithLength_options(bytes.max(4), MTLResourceOptions::StorageModeShared)
        .ok_or(Error::OutOfMemory { bytes })?;
    unsafe { std::ptr::write_bytes(buf.contents().as_ptr() as *mut u8, 0, bytes) };
    Ok(buf)
}

/// Expose mmap'd weights to the GPU without copying them
fn wrap(device: &Device, w: &Weight<'_>) -> Result<GpuWeight, Error> {
    let addr = w.data.as_ptr() as usize;
    let start = addr & !(PAGE - 1);
    let end = (addr + w.data.len()).next_multiple_of(PAGE);
    let ptr =
        NonNull::new(start as *mut c_void).ok_or_else(|| backend("metal: null weight pointer"))?;
    // read-only weights need no hazard tracking
    let opts =
        MTLResourceOptions::StorageModeShared | MTLResourceOptions::HazardTrackingModeUntracked;
    let buf = unsafe {
        device.newBufferWithBytesNoCopy_length_options_deallocator(ptr, end - start, opts, None)
    }
    .ok_or_else(|| backend("metal: cannot map the weights (no-copy buffer)"))?;
    if mm_kernel(w.ty).is_none() {
        return Err(backend(format!("metal: no matmul kernel for {:?}", w.ty)));
    }
    if !w.cols.is_multiple_of(32) {
        return Err(backend(format!(
            "metal: matmul width {} is not a multiple of 32",
            w.cols
        )));
    }
    Ok(GpuWeight {
        buf,
        offset: addr - start,
        ty: w.ty,
        cols: w.cols,
        rows: w.rows,
        row_bytes: w.ty.row_size(w.cols),
    })
}

fn backend(msg: impl Into<String>) -> Error {
    Error::Backend(msg.into())
}

fn gpu_layer(device: &Device, l: &Layer<'_>) -> Result<GpuLayer, Error> {
    let mixer = match &l.mixer {
        Mixer::Attention(a) => GpuMixer::Attention {
            wq: wrap(device, &a.wq)?,
            wk: wrap(device, &a.wk)?,
            wv: wrap(device, &a.wv)?,
            wo: wrap(device, &a.wo)?,
            q_norm: upload(device, &a.q_norm)?,
            k_norm: upload(device, &a.k_norm)?,
        },
        Mixer::DeltaNet(d) => GpuMixer::DeltaNet {
            wqkv: wrap(device, &d.wqkv)?,
            wgate: wrap(device, &d.wgate)?,
            wbeta: wrap(device, &d.wbeta)?,
            walpha: wrap(device, &d.walpha)?,
            wout: wrap(device, &d.wout)?,
            conv1d: upload(device, &d.conv1d)?,
            dt_bias: upload(device, &d.dt_bias)?,
            a: upload(device, &d.a)?,
            norm: upload(device, &d.norm)?,
        },
    };
    Ok(GpuLayer {
        attn_norm: upload(device, &l.attn_norm)?,
        post_norm: upload(device, &l.post_norm)?,
        mixer,
        ffn_gate: wrap(device, &l.ffn_gate)?,
        ffn_up: wrap(device, &l.ffn_up)?,
        ffn_down: wrap(device, &l.ffn_down)?,
    })
}

impl<'m> MetalEngine<'m> {
    pub fn new(model: &'m Model<'m>, max_batch: usize) -> Result<Self, Error> {
        let c = &model.cfg;
        if c.ssm_state != 128 || c.conv_kernel != 4 {
            return Err(backend(
                "metal: the delta net kernels expect state size 128 and conv kernel 4",
            ));
        }
        let group = c.n_head / c.n_head_kv;
        let attn_nsg = ATTN_GROUPS.iter().find(|g| g.0 == group).map(|g| g.1);
        let Some(attn_nsg) = attn_nsg.filter(|_| {
            ATTN_HEAD_DIMS.contains(&c.head_dim)
                && c.n_head.is_multiple_of(c.n_head_kv)
                && c.n_rot.is_multiple_of(64)
        }) else {
            return Err(backend("metal: unsupported attention head geometry"));
        };

        let device = MTLCreateSystemDefaultDevice().ok_or_else(|| backend("metal: no device"))?;
        let library = device
            .newLibraryWithSource_options_error(&NSString::from_str(SOURCE), None)
            .map_err(|e| {
                backend(format!(
                    "metal: shader compilation failed: {}",
                    e.localizedDescription()
                ))
            })?;
        let mut pipelines = Vec::with_capacity(Kernel::ALL.len());
        for (i, &k) in Kernel::ALL.iter().enumerate() {
            debug_assert_eq!(k as usize, i);
            let name = k.name(c.head_dim, group);
            let f = library
                .newFunctionWithName(&NSString::from_str(&name))
                .ok_or_else(|| backend(format!("metal: missing kernel {name}")))?;
            let p = device
                .newComputePipelineStateWithFunction_error(&f)
                .map_err(|e| {
                    backend(format!(
                        "metal: pipeline {name}: {}",
                        e.localizedDescription()
                    ))
                })?;
            pipelines.push(p);
        }
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| backend("metal: no command queue"))?;

        let layers = model
            .layers
            .iter()
            .map(|l| gpu_layer(&device, l))
            .collect::<Result<_, _>>()?;
        let mtp = match &model.mtp {
            Some(m) => Some(GpuMtp {
                enorm: upload(&device, &m.enorm)?,
                hnorm: upload(&device, &m.hnorm)?,
                eh_proj: wrap(&device, &m.eh_proj)?,
                layer: gpu_layer(&device, &m.layer)?,
                head_norm: upload(&device, &m.head_norm)?,
            }),
            None => None,
        };

        Ok(MetalEngine {
            output_norm: upload(&device, &model.output_norm)?,
            output: wrap(&device, &model.output)?,
            inv_freq: upload(&device, &model.inv_freq)?,
            model,
            device,
            queue,
            k: Kernels { pipelines },
            attn_nsg,
            layers,
            mtp,
            max_batch: max_batch.max(1),
        })
    }

    fn cfg(&self) -> &Config {
        &self.model.cfg
    }

    fn dispatch(
        &self,
        enc: &Encoder,
        kernel: Kernel,
        grid: [usize; 3],
        threads: [usize; 3],
        shmem: usize,
    ) {
        enc.setComputePipelineState(self.k.get(kernel));
        if shmem > 0 {
            unsafe { enc.setThreadgroupMemoryLength_atIndex(shmem, 0) };
        }
        let size = |v: [usize; 3]| MTLSize {
            width: v[0],
            height: v[1],
            depth: v[2],
        };
        enc.dispatchThreadgroups_threadsPerThreadgroup(size(grid), size(threads));
    }

    fn args<T>(enc: &Encoder, args: &T) {
        unsafe { enc.setBytes_length_atIndex(NonNull::from(args).cast(), size_of::<T>(), 0) };
    }

    fn bind(enc: &Encoder, bufs: &[B<'_>]) {
        for (i, b) in bufs.iter().enumerate() {
            unsafe { enc.setBuffer_offset_atIndex(Some(b.0), b.1, i + 1) };
        }
    }

    /// `out[t][r] = dot(w.row(r), x[t])` for `n` input rows
    fn matmul(&self, enc: &Encoder, w: &GpuWeight, x: B<'_>, n: usize, out: B<'_>) {
        let args = MatArgs {
            k: w.cols as i32,
            rows: w.rows as i32,
            n: n as i32,
            pad: 0,
            w_row_bytes: w.row_bytes as u64,
            x_row_bytes: (w.cols * 4) as u64,
        };
        Self::args(enc, &args);
        Self::bind(enc, &[B(&w.buf, w.offset), x, out]);
        match mv_kernel(w.ty) {
            // the float32 weights are small vectors' projections, too few rows to fill mmv
            Some((name, rows_per_tg, threads))
                if n <= MV_MAX_ROWS || (w.ty == GgmlType::F32 && n <= 8) =>
            {
                self.dispatch(
                    enc,
                    name,
                    [w.rows.div_ceil(rows_per_tg), n, 1],
                    [threads, 1, 1],
                    0,
                )
            }
            _ if n <= MMV_MAX_ROWS => self.dispatch(
                enc,
                mmv_kernel(w.ty).expect("checked at load"),
                [w.rows.div_ceil(MMV_WEIGHT_ROWS), n.div_ceil(MMV_ROWS), 1],
                [64, 1, 1],
                0,
            ),
            _ => {
                let name = mm_kernel(w.ty).expect("checked at load");
                self.dispatch(
                    enc,
                    name,
                    [n.div_ceil(32), w.rows.div_ceil(64), 1],
                    [128, 1, 1],
                    8192,
                )
            }
        }
    }

    /// rms norm of `rows` rows of `width`, optionally gated by silu(gate)
    #[allow(clippy::too_many_arguments)]
    fn rms_norm(
        &self,
        enc: &Encoder,
        x: B<'_>,
        w: &Buffer,
        out: B<'_>,
        gate: Option<B<'_>>,
        rows: usize,
        width: usize,
    ) {
        self.rms_norm_strided(enc, x, w, out, gate, rows, width, width)
    }

    /// [`MetalEngine::rms_norm`] with output rows `ostride` floats apart
    #[allow(clippy::too_many_arguments)]
    fn rms_norm_strided(
        &self,
        enc: &Encoder,
        x: B<'_>,
        w: &Buffer,
        out: B<'_>,
        gate: Option<B<'_>>,
        rows: usize,
        width: usize,
        ostride: usize,
    ) {
        let args = NormArgs {
            n: width as i32,
            eps: self.cfg().eps,
            gated: gate.is_some() as i32,
            ostride: ostride as i32,
        };
        Self::args(enc, &args);
        Self::bind(enc, &[x, B::at(w), out, gate.unwrap_or(x)]);
        let threads = if width >= 1024 { 256 } else { 32 };
        self.dispatch(enc, Kernel::RmsNorm, [rows, 1, 1], [threads, 1, 1], 0);
    }

    fn elementwise(&self, enc: &Encoder, kernel: Kernel, a: B<'_>, b: B<'_>, n: usize) {
        Self::args(enc, &EltArgs { n: n as i32 });
        Self::bind(enc, &[a, b]);
        self.dispatch(enc, kernel, [n.div_ceil(256), 1, 1], [256, 1, 1], 0);
    }
}

impl Engine for MetalEngine<'_> {
    fn new_session(&self, context: usize, draft: usize) -> Result<Box<dyn Session + '_>, Error> {
        Ok(Box::new(MetalSession::new(self, context, draft)?))
    }

    fn batch_size(&self) -> usize {
        self.max_batch
    }

    // a verify pass of 5 tokens costs 1.75x one of 1 (the matmul kernels read
    // the weights once for 8 tokens); 5 drafts are about as fast, 3 and 6 slower
    fn draft_size(&self) -> usize {
        4
    }

    fn description(&self) -> String {
        format!("metal, {}", self.device.name())
    }

    // the batch size sets the split-K attention's splits
    fn numerics(&self) -> Vec<(&'static str, String)> {
        vec![
            ("kind", "metal".into()),
            ("device", self.device.name().to_string()),
            ("kv_cache", "f16".into()),
            ("max_batch", self.max_batch.to_string()),
        ]
    }
}

/// `len` f32 of a shared buffer, from `offset` f32
#[allow(clippy::mut_from_ref)]
fn host(buf: &Buffer, offset: usize, len: usize) -> &mut [f32] {
    assert!(4 * (offset + len) <= buf.length());
    unsafe {
        std::slice::from_raw_parts_mut((buf.contents().as_ptr() as *mut f32).add(offset), len)
    }
}

/// `len` bytes of a shared buffer, from `offset`
#[allow(clippy::mut_from_ref)]
fn host_bytes(buf: &Buffer, offset: usize, len: usize) -> &mut [u8] {
    assert!(offset + len <= buf.length());
    unsafe { std::slice::from_raw_parts_mut((buf.contents().as_ptr() as *mut u8).add(offset), len) }
}

/// A copy of `len` bytes of a shared buffer, from `offset`
fn copy_out(buf: &Buffer, offset: usize, len: usize) -> Result<Vec<u8>, Error> {
    let mut v = Vec::new();
    v.try_reserve_exact(len)
        .map_err(|_| Error::OutOfMemory { bytes: len })?;
    v.extend_from_slice(host_bytes(buf, offset, len));
    Ok(v)
}

/// See [`Session::snapshot`]: the f16 kv caches of the tokens, the current
/// slot of the delta net states, the MTP head's history
struct MetalSnapshot {
    layers: Vec<LayerSnapshot>,
    mtp: Option<MtpSnapshot>,
}

enum LayerSnapshot {
    Attention { k: Vec<u8>, v: Vec<u8> },
    DeltaNet { conv: Vec<u8>, s: Vec<u8> },
}

struct MtpSnapshot {
    k: Vec<u8>,
    v: Vec<u8>,
    rows: usize,
    last: Vec<u8>,
}

enum LayerState {
    Attention {
        k: Buffer,
        v: Buffer,
    },
    /// `slots` copies of each state, see `ssm_conv` in the shaders
    DeltaNet {
        conv: Buffer,
        s: Buffer,
    },
}

/// Activation buffers, sized for `max_batch` tokens
struct Scratch {
    x: Buffer,
    h: Buffer,
    y: Buffer,
    /// q+gate / qkv / ffn gate
    a1: Buffer,
    /// z / ffn up
    a2: Buffer,
    k: Buffer,
    v: Buffer,
    q: Buffer,
    /// attention / delta net output
    o: Buffer,
    conv_out: Buffer,
    /// split-K attention partials: accumulators and (max, sum)
    part_o: Buffer,
    part_ml: Buffer,
    alpha: Buffer,
    beta: Buffer,
    /// normalized hidden states, the input of the output and MTP heads
    feat: Buffer,
    /// logits of up to `slots` tokens
    logits: Buffer,
}

/// The MTP head's history, see [`crate::model::Model::draft`]
struct MtpState {
    kv: LayerState,
    /// rows entered: row `p` pairs token `p + 1` with the hidden state of token `p`
    rows: usize,
    /// embeddings of the row tokens
    emb: Buffer,
    /// input rows, `[enorm(embedding) ‖ hnorm(hidden)]`
    input: Buffer,
    /// normalized hidden state of the last token, waiting for its successor
    last: Buffer,
    /// normalized output of the last draft, the hidden state of the next
    chain: Buffer,
    /// the output head rows of the draft vocabulary
    head: Option<(DraftVocab, Buffer)>,
}

#[derive(Clone, Copy, PartialEq)]
enum Pass {
    /// no logits
    Prefill,
    /// logits of the last token
    Forward,
    /// logits of every token, and the delta net states after each
    Verify,
}

struct MetalSession<'e> {
    e: &'e MetalEngine<'e>,
    layers: Vec<LayerState>,
    s: Scratch,
    mtp: Option<MtpState>,
    /// tokens per draft
    draft: usize,
    /// copies of the delta net states (one more than the draft length), and
    /// the current one
    slots: usize,
    slot: usize,
    /// first position of the last verify, until `keep`
    verify_from: Option<usize>,
    context: usize,
    n_past: usize,
    bytes: usize,
}

impl<'e> MetalSession<'e> {
    fn new(e: &'e MetalEngine<'e>, context: usize, draft: usize) -> Result<Self, Error> {
        let c = e.cfg();
        let dev = &e.device;
        let kv_dim = c.n_head_kv * c.head_dim;
        // f16 kv caches, see `attn_flash`
        let kv_bytes = 2 * context.next_multiple_of(ATTN_FLASH_C) * kv_dim;
        // a verify pass is one chunk
        let draft = if e.mtp.is_some() {
            draft.min(e.max_batch - 1)
        } else {
            0
        };
        let slots = draft + 1;
        let mut bytes = 0;
        let mut alloc = |n: usize| {
            bytes += n;
            zeroed(dev, n)
        };
        let mut layers = Vec::with_capacity(c.n_layer);
        for &r in &c.recurrent {
            layers.push(if r {
                LayerState::DeltaNet {
                    conv: alloc(slots * 4 * (c.conv_kernel - 1) * c.conv_dim())?,
                    s: alloc(slots * 4 * c.ssm_v_heads * c.ssm_state * c.ssm_state)?,
                }
            } else {
                LayerState::Attention {
                    k: alloc(kv_bytes)?,
                    v: alloc(kv_bytes)?,
                }
            });
        }
        let b = e.max_batch;
        let qd = c.n_head * c.head_dim;
        let s = Scratch {
            x: alloc(4 * b * c.n_embd)?,
            h: alloc(4 * b * c.n_embd)?,
            y: alloc(4 * b * c.n_embd)?,
            a1: alloc(4 * b * c.conv_dim().max(2 * qd).max(c.n_ff))?,
            a2: alloc(4 * b * c.n_ff.max(c.d_inner()))?,
            k: alloc(4 * b * kv_dim)?,
            v: alloc(4 * b * kv_dim)?,
            q: alloc(4 * b * qd)?,
            o: alloc(4 * b * qd.max(c.d_inner()))?,
            conv_out: alloc(4 * b * c.conv_dim())?,
            part_o: alloc(4 * attn_parts(c, b) * c.head_dim)?,
            part_ml: alloc(4 * 2 * attn_parts(c, b))?,
            alpha: alloc(4 * b * c.ssm_v_heads)?,
            beta: alloc(4 * b * c.ssm_v_heads)?,
            feat: alloc(4 * b * c.n_embd)?,
            logits: alloc(4 * slots * c.n_vocab)?,
        };
        let mtp = if draft > 0 {
            Some(MtpState {
                kv: LayerState::Attention {
                    k: alloc(kv_bytes)?,
                    v: alloc(kv_bytes)?,
                },
                rows: 0,
                emb: alloc(4 * b * c.n_embd)?,
                input: alloc(4 * b * 2 * c.n_embd)?,
                last: alloc(4 * c.n_embd)?,
                chain: alloc(4 * c.n_embd)?,
                head: match DraftVocab::new(c.n_vocab) {
                    Some(vocab) => {
                        let out = &e.model.output;
                        let rb = out.data.len() / out.rows;
                        let buf = alloc(DRAFT_CAP * rb)?;
                        let n = vocab.len();
                        copy_head_rows(out, vocab.tokens(0..n), host_bytes(&buf, 0, n * rb));
                        Some((vocab, buf))
                    }
                    None => None,
                },
            })
        } else {
            None
        };
        Ok(MetalSession {
            e,
            layers,
            s,
            mtp,
            draft,
            slots,
            slot: 0,
            verify_from: None,
            context,
            n_past: 0,
            bytes,
        })
    }

    /// Bytes of one slot of a delta net's conv and recurrent states
    fn slot_bytes(&self) -> (usize, usize) {
        let c = self.e.cfg();
        (
            4 * (c.conv_kernel - 1) * c.conv_dim(),
            4 * c.ssm_v_heads * c.ssm_state * c.ssm_state,
        )
    }

    fn run(&mut self, tokens: &[TokenId], pass: Pass) -> Vec<Vec<f32>> {
        if let Some((vocab, buf)) = self.mtp.as_mut().and_then(|m| m.head.as_mut()) {
            let added = vocab.observe(
                tokens,
                if pass == Pass::Verify {
                    1
                } else {
                    tokens.len()
                },
            );
            let out = &self.e.model.output;
            let rb = out.data.len() / out.rows;
            let dst = host_bytes(buf, added.start * rb, added.len() * rb);
            copy_head_rows(out, vocab.tokens(added), dst);
        }
        let mut out = Vec::new();
        let chunks: Vec<&[TokenId]> = tokens.chunks(self.e.max_batch).collect();
        for (i, chunk) in chunks.iter().enumerate() {
            out = self.run_chunk(
                chunk,
                if i == chunks.len() - 1 {
                    pass
                } else {
                    Pass::Prefill
                },
            );
        }
        out
    }

    /// Encode into one command buffer and wait for it
    fn submit(&self, encode: impl FnOnce(&Encoder)) {
        let cb = self
            .e
            .queue
            .commandBuffer()
            .expect("metal: no command buffer");
        let enc = cb
            .computeCommandEncoder()
            .expect("metal: no compute encoder");
        encode(&enc);
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
        if cb.status() == MTLCommandBufferStatus::Error {
            let msg = cb
                .error()
                .map(|e| e.localizedDescription().to_string())
                .unwrap_or_default();
            panic!("metal: command buffer failed: {msg}");
        }
    }

    fn run_chunk(&mut self, tokens: &[TokenId], pass: Pass) -> Vec<Vec<f32>> {
        let e = self.e;
        let c = e.cfg();
        let n = tokens.len();
        let pos0 = self.n_past;
        assert!(
            pos0 + n <= self.context,
            "context overflow ({} + {n} > {})",
            pos0,
            self.context
        );
        assert!(
            pass != Pass::Verify || n <= self.slots,
            "verify of {n} tokens, {} at most",
            self.slots
        );
        let emb = c.n_embd;
        let s = &self.s;

        // the embedding lookup happens on the CPU, straight into the shared buffer
        let x_host = host(&s.x, 0, n * emb);
        for (t, &tok) in tokens.iter().enumerate() {
            e.model
                .tok_embd
                .dequantize_row(tok as usize, &mut x_host[t * emb..(t + 1) * emb]);
        }
        // the MTP rows this pass completes, from the first without a row to
        // the one before the last token, and the embeddings of their tokens
        let mtp_rows = self
            .mtp
            .as_ref()
            .map(|m| (m.rows, pos0 + n - 1))
            .filter(|(first, end)| first < end);
        if let (Some(m), Some((first, end))) = (&self.mtp, mtp_rows) {
            let from = first + 1 - pos0;
            host(&m.emb, 0, (end - first) * emb).copy_from_slice(&x_host[from * emb..]);
        }

        let all = self.mtp.is_some() || pass == Pass::Verify;
        let (x, feat, logits) = (B::at(&s.x), B::at(&s.feat), B::at(&s.logits));
        self.submit(|enc| {
            for (layer, ls) in e.layers.iter().zip(&self.layers) {
                self.encode_layer(enc, layer, ls, n, pos0, pass == Pass::Verify);
            }
            if pass == Pass::Prefill && !all {
                return;
            }

            // normalized hidden states: every row for the MTP head and verify, else the last
            let from = if all { 0 } else { n - 1 };
            e.rms_norm(
                enc,
                x.f32_off(from * emb),
                &e.output_norm,
                feat.f32_off(from * emb),
                None,
                n - from,
                emb,
            );
            match pass {
                Pass::Prefill => {}
                Pass::Forward => e.matmul(enc, &e.output, feat.f32_off((n - 1) * emb), 1, logits),
                Pass::Verify => e.matmul(enc, &e.output, feat, n, logits),
            }

            if let (Some(gm), Some(m), Some((first, end))) = (&e.mtp, &self.mtp, mtp_rows) {
                // the first row's hidden state is the previous pass's last, unless a draft entered it
                let prev = (first < pos0).then(|| B::at(&m.last));
                let rest = feat.f32_off((first.max(pos0) - pos0) * emb);
                self.encode_mtp_input(enc, gm, m, end - first, prev, rest);
                e.matmul(enc, &gm.eh_proj, B::at(&m.input), end - first, x);
                self.encode_layer(enc, &gm.layer, &m.kv, end - first, first, false);
            }
        });

        self.n_past += n;
        if let Some(m) = &mut self.mtp {
            if let Some((_, end)) = mtp_rows {
                m.rows = end;
            }
            host(&m.last, 0, emb).copy_from_slice(host(&s.feat, (n - 1) * emb, emb));
        }
        let rows = match pass {
            Pass::Prefill => 0,
            Pass::Forward => 1,
            Pass::Verify => n,
        };
        (0..rows)
            .map(|t| host(&s.logits, t * c.n_vocab, c.n_vocab).to_vec())
            .collect()
    }

    /// MTP input rows: `enorm` of the embeddings in `m.emb` and `hnorm` of the
    /// hidden states, `prev` for the first row if given, then `rest`
    fn encode_mtp_input(
        &self,
        enc: &Encoder,
        gm: &GpuMtp,
        m: &MtpState,
        rows: usize,
        prev: Option<B<'_>>,
        rest: B<'_>,
    ) {
        let e = self.e;
        let emb = e.cfg().n_embd;
        let input = B::at(&m.input);
        e.rms_norm_strided(
            enc,
            B::at(&m.emb),
            &gm.enorm,
            input,
            None,
            rows,
            emb,
            2 * emb,
        );
        let mut right = input.f32_off(emb);
        let mut rows = rows;
        if let Some(prev) = prev {
            e.rms_norm_strided(enc, prev, &gm.hnorm, right, None, 1, emb, 2 * emb);
            right = right.f32_off(2 * emb);
            rows -= 1;
        }
        if rows > 0 {
            e.rms_norm_strided(enc, rest, &gm.hnorm, right, None, rows, emb, 2 * emb);
        }
    }

    /// One decoder layer over the residual stream `s.x` of `n` tokens at
    /// `pos0`; with `save`, delta nets keep their states after each token
    #[allow(clippy::too_many_arguments)]
    fn encode_layer(
        &self,
        enc: &Encoder,
        layer: &GpuLayer,
        ls: &LayerState,
        n: usize,
        pos0: usize,
        save: bool,
    ) {
        let e = self.e;
        let c = e.cfg();
        let s = &self.s;
        let (emb, hd) = (c.n_embd, c.head_dim);
        let (x, h, y) = (B::at(&s.x), B::at(&s.h), B::at(&s.y));
        let (slot, slots, save) = (self.slot as i32, self.slots as i32, save as i32);

        e.rms_norm(enc, x, &layer.attn_norm, h, None, n, emb);
        match (&layer.mixer, ls) {
            (
                GpuMixer::DeltaNet {
                    wqkv,
                    wgate,
                    wbeta,
                    walpha,
                    wout,
                    conv1d,
                    dt_bias,
                    a,
                    norm,
                },
                LayerState::DeltaNet { conv, s: st },
            ) => {
                let (sd, hk, hv) = (c.ssm_state, c.ssm_k_heads, c.ssm_v_heads);
                let conv_dim = c.conv_dim();
                e.matmul(enc, wqkv, h, n, B::at(&s.a1));
                e.matmul(enc, wgate, h, n, B::at(&s.a2));
                e.matmul(enc, wbeta, h, n, B::at(&s.beta));
                e.matmul(enc, walpha, h, n, B::at(&s.alpha));

                MetalEngine::args(
                    enc,
                    &ConvArgs {
                        conv_dim: conv_dim as i32,
                        n: n as i32,
                        slot,
                        slots,
                        save,
                    },
                );
                MetalEngine::bind(
                    enc,
                    &[B::at(&s.a1), B::at(conv1d), B::at(conv), B::at(&s.conv_out)],
                );
                e.dispatch(
                    enc,
                    Kernel::SsmConv,
                    [conv_dim.div_ceil(256), 1, 1],
                    [256, 1, 1],
                    0,
                );

                let args = GdnArgs {
                    n: n as i32,
                    hk: hk as i32,
                    hv: hv as i32,
                    conv_dim: conv_dim as i32,
                    eps: c.eps,
                    slot,
                    slots,
                    save,
                };
                MetalEngine::args(enc, &args);
                MetalEngine::bind(
                    enc,
                    &[
                        B::at(&s.conv_out),
                        B::at(&s.alpha),
                        B::at(&s.beta),
                        B::at(dt_bias),
                        B::at(a),
                        B::at(st),
                        B::at(&s.o),
                    ],
                );
                e.dispatch(enc, Kernel::Gdn128, [sd / 4, hv, 1], [32, 4, 1], 0);

                // norm(o) * silu(z), per head, in place
                e.rms_norm(
                    enc,
                    B::at(&s.o),
                    norm,
                    B::at(&s.o),
                    Some(B::at(&s.a2)),
                    n * hv,
                    sd,
                );
                e.matmul(enc, wout, B::at(&s.o), n, y);
            }
            (
                GpuMixer::Attention {
                    wq,
                    wk,
                    wv,
                    wo,
                    q_norm,
                    k_norm,
                },
                LayerState::Attention { k: kc, v: vc },
            ) => {
                e.matmul(enc, wq, h, n, B::at(&s.a1));
                e.matmul(enc, wk, h, n, B::at(&s.k));
                e.matmul(enc, wv, h, n, B::at(&s.v));

                let flash_split =
                    (2..ATTN_FLASH_MIN).contains(&n) && pos0 + n >= ATTN_FLASH_SPLIT_CTX;
                let (splits, chunk) = if flash_split {
                    flash_splits(c, pos0, n, e.max_batch)
                } else {
                    attn_splits(c, pos0, n, e.max_batch)
                };
                let args = AttnArgs {
                    n_head: c.n_head as i32,
                    n_kv: c.n_head_kv as i32,
                    hd: hd as i32,
                    n_rot: c.n_rot as i32,
                    pos0: pos0 as i32,
                    eps: c.eps,
                    scale: 1.0 / (hd as f32).sqrt(),
                    splits: splits as i32,
                    chunk: chunk as i32,
                    n: n as i32,
                };
                MetalEngine::args(enc, &args);
                MetalEngine::bind(
                    enc,
                    &[
                        B::at(&s.a1),
                        B::at(&s.k),
                        B::at(&s.v),
                        B::at(q_norm),
                        B::at(k_norm),
                        B::at(&e.inv_freq),
                        B::at(&s.q),
                        B::at(kc),
                        B::at(vc),
                    ],
                );
                e.dispatch(
                    enc,
                    Kernel::AttnPrep,
                    [c.n_head + c.n_head_kv, n, 1],
                    [32, 1, 1],
                    0,
                );

                let group = c.n_head / c.n_head_kv;
                if n >= ATTN_FLASH_MIN {
                    MetalEngine::args(enc, &args);
                    MetalEngine::bind(
                        enc,
                        &[B::at(&s.q), B::at(kc), B::at(vc), B::at(&s.a1), B::at(&s.o)],
                    );
                    // queries (half), scores, and a float per row
                    let shmem = ATTN_FLASH_R * (2 * hd + 4 * ATTN_FLASH_C + 4);
                    let blocks = (n * group).div_ceil(ATTN_FLASH_R);
                    e.dispatch(
                        enc,
                        Kernel::AttnFlash,
                        [blocks, c.n_head_kv, 1],
                        [32 * ATTN_FLASH_NSG, 1, 1],
                        shmem,
                    );
                } else {
                    MetalEngine::args(enc, &args);
                    MetalEngine::bind(
                        enc,
                        &[
                            B::at(&s.q),
                            B::at(kc),
                            B::at(vc),
                            B::at(&s.part_o),
                            B::at(&s.part_ml),
                        ],
                    );
                    if flash_split {
                        let shmem = ATTN_FLASH_R * (2 * hd + 4 * ATTN_FLASH_C + 4);
                        let blocks = (n * group).div_ceil(ATTN_FLASH_R);
                        e.dispatch(
                            enc,
                            Kernel::AttnFlashSplit,
                            [blocks, c.n_head_kv, splits],
                            [32 * ATTN_FLASH_NSG, 1, 1],
                            shmem,
                        );
                    } else {
                        let shmem = 4 * (2 * e.attn_nsg * group + group * hd);
                        e.dispatch(
                            enc,
                            Kernel::AttnSplit,
                            [c.n_head_kv, splits, n],
                            [32 * e.attn_nsg, 1, 1],
                            shmem,
                        );
                    }

                    MetalEngine::args(enc, &args);
                    MetalEngine::bind(
                        enc,
                        &[
                            B::at(&s.part_o),
                            B::at(&s.part_ml),
                            B::at(&s.a1),
                            B::at(&s.o),
                        ],
                    );
                    e.dispatch(enc, Kernel::AttnReduce, [c.n_head, n, 1], [hd, 1, 1], 0);
                }

                e.matmul(enc, wo, B::at(&s.o), n, y);
            }
            _ => unreachable!("layer / state kind mismatch"),
        }
        e.elementwise(enc, Kernel::AddInplace, x, y, n * emb);

        e.rms_norm(enc, x, &layer.post_norm, h, None, n, emb);
        e.matmul(enc, &layer.ffn_gate, h, n, B::at(&s.a1));
        e.matmul(enc, &layer.ffn_up, h, n, B::at(&s.a2));
        e.elementwise(enc, Kernel::Swiglu, B::at(&s.a1), B::at(&s.a2), n * c.n_ff);
        e.matmul(enc, &layer.ffn_down, B::at(&s.a1), n, y);
        e.elementwise(enc, Kernel::AddInplace, x, y, n * emb);
    }
}

/// Split-K attention partials the scratch buffers hold: one per head and range
/// for a single token, one per head and token for a full batch
fn attn_parts(c: &Config, max_batch: usize) -> usize {
    c.n_head * ATTN_MAX_SPLITS.max(max_batch)
}

/// Split-K ranges and their length for `n` tokens after `pos0`: enough ranges
/// to fill the GPU when there are few tokens, bounded by the partial buffers
fn attn_splits(c: &Config, pos0: usize, n: usize, max_batch: usize) -> (usize, usize) {
    let n_ctx = pos0 + n;
    let splits = n_ctx
        .div_ceil(ATTN_CHUNK)
        .min(ATTN_MAX_SPLITS)
        .min(attn_parts(c, max_batch) / (n * c.n_head))
        .max(1);
    (splits, n_ctx.div_ceil(splits))
}

/// Split-K ranges and their length, a multiple of `ATTN_FLASH_C`, for the
/// split `attn_flash` kernel on `n` tokens after `pos0`
fn flash_splits(c: &Config, pos0: usize, n: usize, max_batch: usize) -> (usize, usize) {
    let n_ctx = pos0 + n;
    let max = (attn_parts(c, max_batch) / (n * c.n_head)).clamp(1, ATTN_MAX_SPLITS);
    let chunk = n_ctx.div_ceil(max).next_multiple_of(ATTN_FLASH_C);
    (n_ctx.div_ceil(chunk), chunk)
}

impl Session for MetalSession<'_> {
    fn forward(&mut self, tokens: &[TokenId]) -> Vec<f32> {
        self.run(tokens, Pass::Forward)
            .pop()
            .expect("logits requested")
    }

    fn prefill(&mut self, tokens: &[TokenId]) {
        self.run(tokens, Pass::Prefill);
    }

    fn draft(
        &mut self,
        next: TokenId,
        n: usize,
        pick: &mut dyn FnMut(&DraftLogits<'_>) -> TokenId,
    ) -> Vec<TokenId> {
        let (Some(gm), Some(m)) = (&self.e.mtp, &self.mtp) else {
            return Vec::new();
        };
        let e = self.e;
        let c = e.cfg();
        let emb = c.n_embd;
        // the last token's row waits for its successor, `next`
        assert_eq!(
            m.rows + 1,
            self.n_past,
            "draft must follow a forward pass or keep"
        );
        let base = m.rows;
        let n = n.min(self.draft);
        let mut drafts = Vec::with_capacity(n);
        let mut tok = next;
        for i in 0..n.min(self.context - base) {
            e.model
                .tok_embd
                .dequantize_row(tok as usize, host(&m.emb, 0, emb));
            self.submit(|enc| {
                // the first row is history, the chained ones are speculative
                // and overwritten by the next rows entered
                let hidden = if i == 0 { &m.last } else { &m.chain };
                self.encode_mtp_input(enc, gm, m, 1, None, B::at(hidden));
                e.matmul(enc, &gm.eh_proj, B::at(&m.input), 1, B::at(&self.s.x));
                self.encode_layer(enc, &gm.layer, &m.kv, 1, base + i, false);
                e.rms_norm(
                    enc,
                    B::at(&self.s.x),
                    &gm.head_norm,
                    B::at(&m.chain),
                    None,
                    1,
                    emb,
                );
                let head = match &m.head {
                    Some((vocab, buf)) if !vocab.full() => Some(GpuWeight {
                        buf: buf.clone(),
                        offset: 0,
                        rows: vocab.len(),
                        ..e.output.clone()
                    }),
                    _ => None,
                };
                e.matmul(
                    enc,
                    head.as_ref().unwrap_or(&e.output),
                    B::at(&m.chain),
                    1,
                    B::at(&self.s.logits),
                );
            });
            tok = match &m.head {
                Some((vocab, _)) if !vocab.full() => pick(&DraftLogits {
                    logits: host(&self.s.logits, 0, vocab.len()),
                    tokens: Some(vocab.tokens(0..vocab.len())),
                }),
                _ => pick(&DraftLogits {
                    logits: host(&self.s.logits, 0, c.n_vocab),
                    tokens: None,
                }),
            };
            drafts.push(tok);
        }
        if let Some(m) = &mut self.mtp {
            m.rows = base + 1;
        }
        drafts
    }

    fn verify(&mut self, tokens: &[TokenId]) -> Vec<Vec<f32>> {
        let pos0 = self.n_past;
        let logits = self.run_chunk(tokens, Pass::Verify);
        self.verify_from = Some(pos0);
        logits
    }

    fn keep(&mut self, n: usize) {
        let pos0 = self.verify_from.take().expect("keep without verify");
        let verified = self.n_past - pos0;
        assert!(
            n >= 1 && n <= verified,
            "keep {n} of {verified} verified tokens"
        );
        // the slot after the n-th token holds the delta net states to go on from
        self.slot = (self.slot + n) % self.slots;
        self.n_past = pos0 + n;
        // attention caches are overwritten from the new position
        if let Some(m) = &mut self.mtp {
            let emb = self.e.cfg().n_embd;
            m.rows = m.rows.min(self.n_past - 1);
            host(&m.last, 0, emb).copy_from_slice(host(&self.s.feat, (n - 1) * emb, emb));
        }
    }

    fn n_past(&self) -> usize {
        self.n_past
    }

    fn state_bytes(&self) -> usize {
        self.bytes
    }

    fn snapshot(&self) -> Result<Snapshot, Error> {
        assert!(
            self.verify_from.is_none(),
            "snapshot between verify and keep"
        );
        let c = self.e.cfg();
        let kv_row = 2 * c.n_head_kv * c.head_dim;
        let (conv_slot, s_slot) = self.slot_bytes();
        let mut bytes = 0;
        let mut copy = |buf: &Buffer, offset: usize, len: usize| {
            bytes += len;
            copy_out(buf, offset, len)
        };
        let mut layers = Vec::with_capacity(self.layers.len());
        for ls in &self.layers {
            layers.push(match ls {
                LayerState::Attention { k, v } => LayerSnapshot::Attention {
                    k: copy(k, 0, self.n_past * kv_row)?,
                    v: copy(v, 0, self.n_past * kv_row)?,
                },
                LayerState::DeltaNet { conv, s } => LayerSnapshot::DeltaNet {
                    conv: copy(conv, self.slot * conv_slot, conv_slot)?,
                    s: copy(s, self.slot * s_slot, s_slot)?,
                },
            });
        }
        let mtp = match &self.mtp {
            Some(MtpState {
                kv: LayerState::Attention { k, v },
                rows,
                last,
                ..
            }) => Some(MtpSnapshot {
                k: copy(k, 0, rows * kv_row)?,
                v: copy(v, 0, rows * kv_row)?,
                rows: *rows,
                last: copy(last, 0, 4 * c.n_embd)?,
            }),
            Some(_) => unreachable!("the MTP head has an attention layer"),
            None => None,
        };
        Ok(Snapshot::new(
            self.n_past,
            bytes,
            MetalSnapshot { layers, mtp },
        ))
    }

    fn restore(&mut self, snapshot: &Snapshot) -> Result<(), Error> {
        let snap: &MetalSnapshot = snapshot.data()?;
        let n_past = snapshot.n_past();
        if n_past > self.context {
            return Err(Error::Snapshot("more tokens than the context holds"));
        }
        let c = self.e.cfg();
        let kv_row = 2 * c.n_head_kv * c.head_dim;
        let (conv_slot, s_slot) = self.slot_bytes();
        let fits = self.layers.len() == snap.layers.len()
            && self
                .layers
                .iter()
                .zip(&snap.layers)
                .all(|(ls, s)| match (ls, s) {
                    (LayerState::Attention { .. }, LayerSnapshot::Attention { k, .. }) => {
                        k.len() == n_past * kv_row
                    }
                    (LayerState::DeltaNet { .. }, LayerSnapshot::DeltaNet { conv, s }) => {
                        conv.len() == conv_slot && s.len() == s_slot
                    }
                    _ => false,
                })
            && match (&self.mtp, &snap.mtp) {
                (Some(_), Some(m)) => m.k.len() == m.rows * kv_row && m.last.len() == 4 * c.n_embd,
                (Some(_), None) => {
                    return Err(Error::Snapshot("no MTP history, for a drafting session"));
                }
                (None, _) => true,
            };
        if !fits {
            return Err(Error::Snapshot("taken from another model"));
        }

        // the states go to slot 0
        for (ls, s) in self.layers.iter().zip(&snap.layers) {
            match (ls, s) {
                (LayerState::Attention { k, v }, LayerSnapshot::Attention { k: sk, v: sv }) => {
                    host_bytes(k, 0, sk.len()).copy_from_slice(sk);
                    host_bytes(v, 0, sv.len()).copy_from_slice(sv);
                }
                (LayerState::DeltaNet { conv, s }, LayerSnapshot::DeltaNet { conv: sc, s: ss }) => {
                    host_bytes(conv, 0, sc.len()).copy_from_slice(sc);
                    host_bytes(s, 0, ss.len()).copy_from_slice(ss);
                }
                _ => unreachable!("checked above"),
            }
        }
        if let (Some(m), Some(sm)) = (&mut self.mtp, &snap.mtp) {
            let LayerState::Attention { k, v } = &m.kv else {
                unreachable!("the MTP head has an attention layer")
            };
            host_bytes(k, 0, sm.k.len()).copy_from_slice(&sm.k);
            host_bytes(v, 0, sm.v.len()).copy_from_slice(&sm.v);
            host_bytes(&m.last, 0, sm.last.len()).copy_from_slice(&sm.last);
            m.rows = sm.rows;
        }
        self.slot = 0;
        self.n_past = n_past;
        self.verify_from = None;
        Ok(())
    }
}
