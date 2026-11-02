//! GGUF container reader: metadata key/values and tensors, over the bytes of
//! the file (typically a memory mapping, owned by the [`Gguf`]).
//!
//! The metadata is read into the nested structs of [`Metadata`], one level
//! per component of the dotted keys.
//!
//! Split files (`-00001-of-0000N.gguf`) are parsed together; tensor data of
//! each shard stays in its own buffer.

mod metadata;

use hashbrown::HashMap;

pub use metadata::*;

use crate::Error;
use crate::prelude::*;
use crate::quant::GgmlType;

const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" little endian
const DEFAULT_ALIGNMENT: u64 = 32;

#[derive(Debug, Clone)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    Str(String),
    Array(Vec<Value>),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        Some(match *self {
            Value::U8(v) => v as u64,
            Value::I8(v) => v as u64,
            Value::U16(v) => v as u64,
            Value::I16(v) => v as u64,
            Value::U32(v) => v as u64,
            Value::I32(v) => v as u64,
            Value::U64(v) => v,
            Value::I64(v) => v as u64,
            Value::Bool(v) => v as u64,
            _ => return None,
        })
    }

    pub fn as_f32(&self) -> Option<f32> {
        match *self {
            Value::F32(v) => Some(v),
            Value::F64(v) => Some(v as f32),
            _ => self.as_u64().map(|v| v as f32),
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// ggml order: `dims[0]` is the contiguous (row) dimension
    pub dims: Vec<u64>,
    pub ty: GgmlType,
    shard: usize,
    /// absolute offset in the shard file
    offset: u64,
}

impl TensorInfo {
    pub fn n_elements(&self) -> u64 {
        self.dims.iter().product()
    }

    pub fn n_bytes(&self) -> usize {
        self.ty.row_size(self.n_elements() as usize)
    }
}

/// The bytes of one GGUF file
pub type Shard = Box<dyn AsRef<[u8]> + Send + Sync>;

pub struct Gguf {
    pub metadata: Metadata,
    /// the keys with no field in [`Metadata`], or a value of another type
    /// than their field's, in file order
    pub extra: Vec<(String, Value)>,
    pub tensors: HashMap<String, TensorInfo>,
    /// tensor names in file order
    pub tensor_order: Vec<String>,
    shards: Vec<Shard>,
}

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], Error> {
        if self.pos + n > self.buf.len() {
            return Err(Error::Truncated);
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn u8(&mut self) -> Result<u8, Error> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16, Error> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32, Error> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, Error> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn string(&mut self) -> Result<String, Error> {
        let len = self.u64()? as usize;
        let bytes = self.take(len)?;
        Ok(String::from_utf8_lossy(bytes).into_owned())
    }

    fn value(&mut self, ty: u32) -> Result<Value, Error> {
        Ok(match ty {
            0 => Value::U8(self.u8()?),
            1 => Value::I8(self.u8()? as i8),
            2 => Value::U16(self.u16()?),
            3 => Value::I16(self.u16()? as i16),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(self.u32()? as i32),
            6 => Value::F32(f32::from_bits(self.u32()?)),
            7 => Value::Bool(self.u8()? != 0),
            8 => Value::Str(self.string()?),
            9 => {
                let elem_ty = self.u32()?;
                let n = self.u64()? as usize;
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    v.push(self.value(elem_ty)?);
                }
                Value::Array(v)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(self.u64()? as i64),
            12 => Value::F64(f64::from_bits(self.u64()?)),
            _ => return Err(Error::UnknownValueType(ty)),
        })
    }
}

impl Gguf {
    /// Parse a GGUF from the contents of its shard files, in order
    pub fn from_shards(shards: Vec<Shard>) -> Result<Self, Error> {
        let mut gguf = Gguf {
            metadata: Metadata::default(),
            extra: Vec::new(),
            tensors: HashMap::new(),
            tensor_order: Vec::new(),
            shards: Vec::new(),
        };
        for (i, shard) in shards.into_iter().enumerate() {
            gguf.parse_shard(i, (*shard).as_ref())?;
            gguf.shards.push(shard);
        }
        Ok(gguf)
    }

    fn parse_shard(&mut self, shard: usize, buf: &[u8]) -> Result<(), Error> {
        let mut c = Cursor { buf, pos: 0 };
        if c.u32()? != GGUF_MAGIC {
            return Err(Error::BadMagic);
        }
        let version = c.u32()?;
        if version < 2 {
            return Err(Error::UnsupportedVersion(version));
        }
        let n_tensors = c.u64()?;
        let n_kv = c.u64()?;

        for _ in 0..n_kv {
            let key = c.string()?;
            let ty = c.u32()?;
            let value = c.value(ty)?;
            // split shards repeat `split.*` keys; the first shard holds the real metadata
            if self.contains(&key) {
                continue;
            }
            if let Err(value) = self.metadata.set(&key, value) {
                self.extra.push((key, value));
            }
        }

        let mut infos = Vec::with_capacity(n_tensors as usize);
        for _ in 0..n_tensors {
            let name = c.string()?;
            let n_dims = c.u32()? as usize;
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(c.u64()?);
            }
            let raw_ty = c.u32()?;
            let Some(ty) = GgmlType::from_u32(raw_ty) else {
                return Err(Error::UnknownTensorType {
                    tensor: name,
                    ty: raw_ty,
                });
            };
            let offset = c.u64()?;
            infos.push(TensorInfo {
                name,
                dims,
                ty,
                shard,
                offset,
            });
        }

        let alignment = self
            .metadata
            .general
            .alignment
            .map_or(DEFAULT_ALIGNMENT, u64::from);
        let data_start = (c.pos as u64).div_ceil(alignment) * alignment;

        for mut info in infos {
            info.offset += data_start;
            if info.offset as usize + info.n_bytes() > buf.len() {
                return Err(Error::TensorOutOfBounds(info.name));
            }
            self.tensor_order.push(info.name.clone());
            self.tensors.insert(info.name.clone(), info);
        }
        Ok(())
    }

    /// Whether the file has `key`
    pub fn contains(&self, key: &str) -> bool {
        self.metadata.contains(key) || self.extra.iter().any(|(k, _)| k == key)
    }

    /// The value of `key`, from [`Metadata`] or `extra`
    pub fn get(&self, key: &str) -> Option<Value> {
        self.metadata.get(key).or_else(|| {
            self.extra
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v.clone())
        })
    }

    /// Every key and its value: those of [`Metadata`], then `extra`
    pub fn entries(&self) -> Vec<(String, Value)> {
        let mut out = Vec::new();
        self.metadata
            .visit("", &mut |k, v| out.push((String::from(k), v)));
        out.extend(self.extra.iter().cloned());
        out
    }

    pub fn tensor(&self, name: &str) -> Option<(&TensorInfo, &[u8])> {
        let info = self.tensors.get(name)?;
        let start = info.offset as usize;
        Some((
            info,
            &(*self.shards[info.shard]).as_ref()[start..start + info.n_bytes()],
        ))
    }
}
