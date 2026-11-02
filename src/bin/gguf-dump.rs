//! Print the metadata and tensor table of a GGUF file.

use biv::gguf::Value;

fn show(v: &Value) -> String {
    match v {
        Value::Str(s) if s.len() > 200 => format!(
            "{:?}… ({} bytes)",
            &s[..s.floor_char_boundary(200)],
            s.len()
        ),
        Value::Str(s) => format!("{s:?}"),
        Value::Array(a) if a.len() > 16 => {
            let head: Vec<_> = a.iter().take(8).map(show).collect();
            format!("[{}, …] ({} items)", head.join(", "), a.len())
        }
        Value::Array(a) => format!("[{}]", a.iter().map(show).collect::<Vec<_>>().join(", ")),
        // the struct fields have their own types, not the file's: no `U32(…)`
        Value::U8(x) => x.to_string(),
        Value::I8(x) => x.to_string(),
        Value::U16(x) => x.to_string(),
        Value::I16(x) => x.to_string(),
        Value::U32(x) => x.to_string(),
        Value::I32(x) => x.to_string(),
        Value::U64(x) => x.to_string(),
        Value::I64(x) => x.to_string(),
        Value::F32(x) => format!("{x:?}"),
        Value::F64(x) => format!("{x:?}"),
        Value::Bool(x) => x.to_string(),
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: gguf-dump <file.gguf> [--full-key KEY]");
    let full_key = match (args.next().as_deref(), args.next()) {
        (Some("--full-key"), Some(k)) => Some(k),
        _ => None,
    };
    let gguf = biv_cli::open_gguf(&path).unwrap_or_else(|e| panic!("{e}"));

    if let Some(key) = full_key {
        match gguf.get(&key) {
            Some(Value::Str(s)) => println!("{s}"),
            Some(v) => println!("{}", show(&v)),
            None => eprintln!("no key {key}"),
        }
        return;
    }

    let mut entries = gguf.entries();
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    for (k, v) in entries {
        println!("{k} = {}", show(&v));
    }
    println!();
    for name in &gguf.tensor_order {
        let t = &gguf.tensors[name];
        println!("{name:48} {:?} {:?}", t.ty, t.dims);
    }
}
