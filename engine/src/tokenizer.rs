//! GPT-2 style byte-level BPE, as stored in GGUF (`tokenizer.ggml.model = gpt2`)
//! with the `qwen35` pre-tokenizer.

use hashbrown::HashMap;
use unicode_properties::{GeneralCategoryGroup, UnicodeGeneralCategory};

use crate::Error;
use crate::gguf::{Gguf, need};
use crate::prelude::*;

const TOKEN_TYPE_CONTROL: i64 = 3;
const TOKEN_TYPE_USER_DEFINED: i64 = 4;

pub type TokenId = u32;

/// A piece of a prompt, see [`Tokenizer::encode_prompt`]
#[derive(Debug, Clone, Copy)]
pub enum PromptPart<'a> {
    /// the chat template's own text, where special token strings are tokens
    Template(&'a str),
    /// message content, where they are plain text
    Content(&'a str),
}

pub struct Tokenizer {
    /// token text, in the GPT-2 byte-to-unicode encoding (or raw for special tokens)
    tokens: Vec<String>,
    /// decoded bytes of each token
    bytes: Vec<Vec<u8>>,
    is_special: Vec<bool>,
    token_to_id: HashMap<String, TokenId>,
    /// (left, right) -> (rank, merged)
    merges: HashMap<(TokenId, TokenId), (u32, TokenId)>,
    /// token id of each raw byte
    byte_tokens: [TokenId; 256],
    /// special tokens, longest first, for splitting text that may contain them
    specials: Vec<(String, TokenId)>,
    pub bos: Option<TokenId>,
    pub eos: TokenId,
    /// tokens that end a generation
    pub eog: Vec<TokenId>,
}

/// GPT-2's reversible byte <-> printable unicode mapping
fn byte_to_unicode() -> [char; 256] {
    let mut map = ['\0'; 256];
    let mut n = 0u32;
    for b in 0..256u32 {
        let printable = (b'!' as u32..=b'~' as u32).contains(&b)
            || (0xA1..=0xAC).contains(&b)
            || (0xAE..=0xFF).contains(&b);
        map[b as usize] = if printable {
            char::from_u32(b).unwrap()
        } else {
            n += 1;
            char::from_u32(255 + n).unwrap()
        };
    }
    map
}

impl Tokenizer {
    pub fn from_gguf(gguf: &Gguf) -> Result<Self, Error> {
        let m = &gguf.metadata;
        let ggml = &m.tokenizer.ggml;
        let model = need!(m, tokenizer.ggml.model)?;
        let pre = ggml.pre.as_deref().unwrap_or("qwen35");
        if model != "gpt2" || pre != "qwen35" {
            return Err(Error::UnsupportedTokenizer {
                model: model.clone(),
                pre: pre.into(),
            });
        }
        let tokens: Vec<String> = need!(m, tokenizer.ggml.tokens)?.clone();
        let types: Vec<i64> = match &ggml.token_type {
            Some(a) => a.iter().map(|&t| t.into()).collect(),
            None => vec![1; tokens.len()],
        };

        let unicode_to_byte: HashMap<char, u8> = byte_to_unicode()
            .iter()
            .enumerate()
            .map(|(b, &c)| (c, b as u8))
            .collect();

        let is_special: Vec<bool> = types
            .iter()
            .map(|&t| t == TOKEN_TYPE_CONTROL || t == TOKEN_TYPE_USER_DEFINED)
            .collect();
        let bytes: Vec<Vec<u8>> = tokens
            .iter()
            .zip(&is_special)
            .map(|(t, &special)| {
                if special {
                    t.as_bytes().to_vec()
                } else {
                    t.chars()
                        .map(|c| unicode_to_byte.get(&c).copied())
                        .collect::<Option<Vec<u8>>>()
                        .unwrap_or_else(|| t.as_bytes().to_vec())
                }
            })
            .collect();

        let token_to_id: HashMap<String, TokenId> = tokens
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i as TokenId))
            .collect();

        let mut byte_tokens = [0; 256];
        for (b, c) in byte_to_unicode().iter().enumerate() {
            let c = c.to_string();
            byte_tokens[b] = *token_to_id.get(&c).ok_or(Error::MissingToken(c))?;
        }

        let mut merges = HashMap::new();
        for (rank, merge) in need!(m, tokenizer.ggml.merges)?.iter().enumerate() {
            let Some((l, r)) = merge.split_once(' ') else {
                continue;
            };
            let (Some(&li), Some(&ri), Some(&mi)) = (
                token_to_id.get(l),
                token_to_id.get(r),
                token_to_id.get(&format!("{l}{r}")),
            ) else {
                continue;
            };
            merges.entry((li, ri)).or_insert((rank as u32, mi));
        }

        let mut specials: Vec<(String, TokenId)> = tokens
            .iter()
            .enumerate()
            .filter(|(i, t)| is_special[*i] && !t.is_empty())
            .map(|(i, t)| (t.clone(), i as TokenId))
            .collect();
        specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        let eos = *need!(m, tokenizer.ggml.eos_token_id)?;
        let mut eog = vec![eos];
        for t in ["<|im_end|>", "<|endoftext|>", "<|end_of_text|>"] {
            if let Some(&i) = token_to_id.get(t)
                && !eog.contains(&i)
            {
                eog.push(i);
            }
        }

        Ok(Tokenizer {
            tokens,
            bytes,
            is_special,
            token_to_id,
            merges,
            byte_tokens,
            specials,
            bos: ggml.bos_token_id,
            eos,
            eog,
        })
    }

    pub fn n_vocab(&self) -> usize {
        self.tokens.len()
    }

    pub fn token_id(&self, text: &str) -> Option<TokenId> {
        self.token_to_id.get(text).copied()
    }

    pub fn is_special(&self, id: TokenId) -> bool {
        self.is_special.get(id as usize).copied().unwrap_or(false)
    }

    pub fn token_text(&self, id: TokenId) -> &str {
        &self.tokens[id as usize]
    }

    /// The raw bytes a token stands for (a token may hold a partial UTF-8 sequence)
    pub fn token_bytes(&self, id: TokenId) -> &[u8] {
        &self.bytes[id as usize]
    }

    pub fn decode(&self, ids: &[TokenId]) -> String {
        let bytes: Vec<u8> = ids
            .iter()
            .flat_map(|&i| self.token_bytes(i).iter().copied())
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Tokenize plain text; special token strings are not recognized
    pub fn encode(&self, text: &str) -> Vec<TokenId> {
        let mut out = Vec::new();
        self.encode_into(text, &mut out);
        out
    }

    /// Tokenize text, mapping special token strings (`<|im_start|>`, …) to their ids
    pub fn encode_with_special(&self, text: &str) -> Vec<TokenId> {
        self.encode_prompt(&[PromptPart::Template(text)])
    }

    /// Tokenize a rendered prompt. Special token strings are recognized in the
    /// template's text only, and the text between special tokens is tokenized
    /// whole, across parts, as it would be in one string.
    pub fn encode_prompt(&self, parts: &[PromptPart<'_>]) -> Vec<TokenId> {
        let mut out = Vec::new();
        let mut text = String::new();
        for part in parts {
            match *part {
                PromptPart::Content(s) => text.push_str(s),
                PromptPart::Template(mut rest) => {
                    while let Some((pos, len, id)) = self.find_special(rest) {
                        text.push_str(&rest[..pos]);
                        self.encode_into(&text, &mut out);
                        text.clear();
                        out.push(id);
                        rest = &rest[pos + len..];
                    }
                    text.push_str(rest);
                }
            }
        }
        self.encode_into(&text, &mut out);
        out
    }

    /// The earliest special token string in `text`: position, length and id
    fn find_special(&self, text: &str) -> Option<(usize, usize, TokenId)> {
        // specials are sorted longest first, so the first found at a position
        // is the longest there
        let mut best: Option<(usize, usize, TokenId)> = None;
        for (s, id) in &self.specials {
            if let Some(pos) = text.find(s.as_str())
                && best.is_none_or(|(bp, _, _)| pos < bp)
            {
                best = Some((pos, s.len(), *id));
            }
        }
        best
    }

    fn encode_into(&self, text: &str, out: &mut Vec<TokenId>) {
        for piece in pre_split(text) {
            self.bpe(piece.as_bytes(), out);
        }
    }

    fn bpe(&self, piece: &[u8], out: &mut Vec<TokenId>) {
        let mut syms: Vec<TokenId> = piece
            .iter()
            .map(|&b| self.byte_tokens[b as usize])
            .collect();
        loop {
            let best = syms
                .windows(2)
                .enumerate()
                .filter_map(|(i, w)| {
                    self.merges
                        .get(&(w[0], w[1]))
                        .map(|&(rank, m)| (rank, i, m))
                })
                .min();
            let Some((rank, _, _)) = best else { break };
            // merge every occurrence of this pair, left to right
            let mut merged = Vec::with_capacity(syms.len());
            let mut i = 0;
            while i < syms.len() {
                if i + 1 < syms.len()
                    && let Some(&(r, m)) = self.merges.get(&(syms[i], syms[i + 1]))
                    && r == rank
                {
                    merged.push(m);
                    i += 2;
                } else {
                    merged.push(syms[i]);
                    i += 1;
                }
            }
            syms = merged;
        }
        out.extend(syms);
    }
}

/// llama.cpp's `LLAMA_VOCAB_PRE_TYPE_QWEN35` split, a regex:
///
/// ```text
/// (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}
/// | ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
/// ```
///
/// matched by hand. The pieces cover the whole text: every char starts one
/// of the alternatives.
fn pre_split(text: &str) -> impl Iterator<Item = &str> {
    let mut rest = text;
    core::iter::from_fn(move || {
        if rest.is_empty() {
            return None;
        }
        let (piece, tail) = rest.split_at(piece_len(rest));
        rest = tail;
        Some(piece)
    })
}

/// The classes of the regex: `\p{L}`, `\p{M}`, `\p{N}`, `\s`, and none of them
#[derive(Clone, Copy, PartialEq, Eq)]
enum Class {
    Letter,
    Mark,
    Number,
    Space,
    Other,
}

fn class(c: char) -> Class {
    if c.is_ascii() {
        return match c {
            'a'..='z' | 'A'..='Z' => Class::Letter,
            '0'..='9' => Class::Number,
            '\t'..='\r' | ' ' => Class::Space,
            _ => Class::Other,
        };
    }
    // White_Space, as `\s`; no whitespace char is a letter, mark or number
    if c.is_whitespace() {
        return Class::Space;
    }
    match c.general_category_group() {
        GeneralCategoryGroup::Letter => Class::Letter,
        GeneralCategoryGroup::Mark => Class::Mark,
        GeneralCategoryGroup::Number => Class::Number,
        _ => Class::Other,
    }
}

/// End of the run of chars accepted by `f` in `s`, from byte `from`
fn run(s: &str, from: usize, f: impl Fn(char) -> bool) -> usize {
    from + s[from..].find(|c| !f(c)).unwrap_or(s.len() - from)
}

/// Byte length of the piece at the start of `s`: the first alternative of the
/// regex that matches there, as long as backtracking makes it
fn piece_len(s: &str) -> usize {
    use Class::*;
    let mut chars = s.chars();
    let c0 = chars.next().expect("a piece of an empty text");
    let (k0, k1) = (class(c0), chars.next().map(class));
    let letter = |c| matches!(class(c), Letter | Mark);

    // '[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD]
    if c0 == '\'' {
        let lower = |i: usize| s.as_bytes().get(i).map(u8::to_ascii_lowercase);
        match (lower(1), lower(2)) {
            (Some(b's' | b't' | b'm' | b'd'), _) => return 2,
            (Some(b'r' | b'v'), Some(b'e')) | (Some(b'l'), Some(b'l')) => return 3,
            _ => {}
        }
    }
    // [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
    if matches!(k1, Some(Letter | Mark))
        && !matches!(c0, '\r' | '\n')
        && !matches!(k0, Letter | Number)
    {
        return run(s, c0.len_utf8(), letter);
    }
    if matches!(k0, Letter | Mark) {
        return run(s, 0, letter);
    }
    // \p{N}
    if k0 == Number {
        return c0.len_utf8();
    }
    // ' ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*'
    let start = match (c0, k0, k1) {
        (_, Other, _) => Some(0),
        (' ', _, Some(Other)) => Some(1),
        _ => None,
    };
    if let Some(start) = start {
        let end = run(s, start, |c| class(c) == Other);
        return run(s, end, |c| matches!(c, '\r' | '\n'));
    }

    // whitespace is left
    debug_assert!(k0 == Space);
    let ws = run(s, 0, |c| class(c) == Space);
    // \s*[\r\n]+: up to the run's last newline
    if let Some(nl) = s[..ws].rfind(['\r', '\n']) {
        return nl + 1;
    }
    // \s+(?!\S): the run, but for its last char when a non-space follows,
    // which then starts the next piece; \s+: a single space before a non-space
    let last = s[..ws].chars().next_back().map_or(0, char::len_utf8);
    if ws == s.len() || ws == last {
        ws
    } else {
        ws - last
    }
}

/// Turns a stream of tokens into text, holding back incomplete UTF-8 sequences
#[derive(Default)]
pub struct StreamDecoder {
    pending: Vec<u8>,
}

impl StreamDecoder {
    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        match core::str::from_utf8(&self.pending) {
            Ok(s) => {
                let s = s.to_string();
                self.pending.clear();
                s
            }
            Err(e) => {
                let valid = e.valid_up_to();
                // an invalid (not just incomplete) sequence: emit it lossily
                let cut = if e.error_len().is_some() {
                    self.pending.len()
                } else {
                    valid
                };
                let s = String::from_utf8_lossy(&self.pending[..cut]).into_owned();
                self.pending.drain(..cut);
                s
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The regex the pre-tokenizer implements, as llama.cpp has it
    const QWEN35_PRE: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    fn check(re: &fancy_regex::Regex, text: &str) {
        let want: Vec<&str> = re.find_iter(text).map(|m| m.unwrap().as_str()).collect();
        let got: Vec<&str> = pre_split(text).collect();
        assert_eq!(got, want, "split of {text:?}");
    }

    #[test]
    fn pre_split_matches_regex() {
        let re = fancy_regex::Regex::new(QWEN35_PRE).unwrap();
        for text in [
            "",
            "Hello world! It's 2026, isn't it?   ok\n\n  x",
            "I'LL 'Re 'sure' 'x 'd' '",
            "e\u{301}t\u{e9} \u{301}x नमस्ते ١٢٣ ½ Ⅻ 你好 🦀",
            "  \t\n \r\n  trailing   ",
            "a\u{a0}\u{a0}b\u{3000}c \u{2028}d \u{85}",
            "x  ",
            "   ",
            " .\n\r\n!",
        ] {
            check(&re, text);
        }
        // random sequences of pieces that exercise every alternative
        let pool = [
            "a",
            "Z",
            "é",
            "e\u{301}",
            "\u{301}",
            "ß",
            "中",
            "😀",
            "👍🏽",
            "\u{200d}",
            "\u{feff}",
            "0",
            "9",
            "½",
            "Ⅻ",
            "٣",
            "क",
            "ि",
            "्",
            " ",
            "  ",
            "\t",
            "\n",
            "\r",
            "\r\n",
            "\u{a0}",
            "\u{3000}",
            "\u{2028}",
            "\u{85}",
            "\u{b}",
            "\u{0}",
            "\u{1f}",
            "'",
            "'s",
            "'S",
            "'re",
            "'VE",
            "'lL",
            "'d",
            "'m",
            "'t",
            "'x",
            "'r",
            "'l",
            ".",
            ",",
            "!",
            "-",
            "_",
            "(",
            "\"",
            "/",
            "$",
            "€",
            "+",
            "<|im_start|>",
        ];
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let mut rand = |n: usize| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % n as u64) as usize
        };
        for _ in 0..50_000 {
            let len = 1 + rand(12);
            let text: String = (0..len).map(|_| pool[rand(pool.len())]).collect();
            check(&re, &text);
        }
    }
}
