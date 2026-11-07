fn main() {
    let path = std::env::args().nth(1).unwrap();
    let t = std::time::Instant::now();
    let tok = biv_cli::open_tokenizer(&path).unwrap();
    eprintln!(
        "load {:?}, vocab {}, eos {} {:?}, eog {:?}",
        t.elapsed(),
        tok.n_vocab(),
        tok.eos,
        tok.token_text(tok.eos),
        tok.eog
    );
    let specials: Vec<_> = (0..tok.n_vocab() as u32)
        .filter(|&i| tok.is_special(i))
        .map(|i| format!("{i}:{}", tok.token_text(i)))
        .collect();
    println!("{} specials: {}", specials.len(), specials.join(" "));
    for s in [
        "Hello world! It's 2026, isn't it?   ok\n\n  x",
        "<|im_start|>user\nBonjour, ça va? 你好世界 🦀<|im_end|>\n<think>\n",
    ] {
        let ids = tok.encode_with_special(s);
        println!("{ids:?}\n{:?}", tok.decode(&ids));
    }
}
