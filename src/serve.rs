//! A server of the chat completions route of the OpenAI API, for clients
//! that speak it: `POST /v1/chat/completions`, one full answer for each
//! request, with no stream. Each request holds the whole conversation, so
//! each one runs on a new chat. The server answers one request at a time;
//! a request that comes while one runs is refused with `server_busy`.
//!
//! It takes only the fields below, and refuses any other field by name,
//! so that a client never gets an answer made without a setting that it
//! sent:
//! - `model`: the model name that the server was started with
//! - `messages`: objects with `role` and `content` (a string). The system
//!   messages come first; then user and assistant messages, the last one
//!   from the user.
//! - `max_tokens`: the most tokens of the answer, the reasoning included
//! - `reasoning_effort`: `none` (no reasoning block), `low`, `medium`,
//!   `high` or `xhigh`, as the Qwen3.8 chat template takes them (`high` is
//!   `xhigh`)
//! - `stream`: `false` only
//!
//! The answer gives `choices[0].message.content`, the reasoning as
//! `choices[0].message.reasoning_content`, `finish_reason` (`stop` when the
//! model ended its answer, `length` at max_tokens or at the end of the
//! context) and `usage`.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::sync::mpsc::{Receiver, SyncSender, TrySendError, sync_channel};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use biv::chat::{Chat, ChatOptions, Event, Message, ReasoningEffort, Stop, TurnStats};
use biv::engine::Engine;
use biv::sampler::{Sampler, SamplerConfig};
use biv::tokenizer::Tokenizer;
use serde_json::{Map, Value, json};

/// The one route of the server
pub const ROUTE: &str = "/v1/chat/completions";
const METHOD: &str = "POST";
/// The longest body of a request
const MAX_BODY_BYTES: usize = 16 << 20;
/// The longest line of the head of a request
const MAX_HEAD_LINE_BYTES: usize = 8 << 10;
/// The most lines of the head of a request
const MAX_HEAD_LINES: usize = 128;
/// The time that a client has to send its request
const READ_TIMEOUT: Duration = Duration::from_secs(30);
/// The requests that wait while one runs: none
const WAITING_REQUESTS: usize = 0;

const CONTENT_LENGTH: &str = "content-length";
const TRANSFER_ENCODING: &str = "transfer-encoding";

const FIELD_MODEL: &str = "model";
const FIELD_MESSAGES: &str = "messages";
const FIELD_MAX_TOKENS: &str = "max_tokens";
const FIELD_REASONING_EFFORT: &str = "reasoning_effort";
const FIELD_STREAM: &str = "stream";
const REQUEST_FIELDS: [&str; 5] = [
    FIELD_MODEL,
    FIELD_MESSAGES,
    FIELD_MAX_TOKENS,
    FIELD_REASONING_EFFORT,
    FIELD_STREAM,
];
const FIELD_ROLE: &str = "role";
const FIELD_CONTENT: &str = "content";
const MESSAGE_FIELDS: [&str; 2] = [FIELD_ROLE, FIELD_CONTENT];

const ROLE_SYSTEM: &str = "system";
const ROLE_USER: &str = "user";
const ROLE_ASSISTANT: &str = "assistant";
/// The chat template joins the system messages with a newline
const SYSTEM_SEPARATOR: &str = "\n";

/// `reasoning_effort` for an answer with no reasoning block
const EFFORT_NONE: &str = "none";

/// Why a request gets no answer: the HTTP status, and the code that names
/// the refusal in the body
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refusal {
    RequestUnreadable(String),
    HeadTooLong,
    MethodNotAllowed(String),
    RouteNotFound(String),
    ContentLengthMissing,
    ChunkedBodyUnsupported,
    BodyTooLarge(usize),
    BodyNotJson(String),
    BodyNotObject,
    FieldUnsupported(String),
    FieldMissing(&'static str),
    FieldInvalid(&'static str),
    ModelNotServed(String),
    StreamUnsupported,
    ReasoningEffortUnsupported(String),
    MessageFieldUnsupported(String),
    RoleUnsupported(String),
    SystemMessageNotFirst,
    LastMessageNotUser,
    NoUserMessage,
    ContextLengthExceeded(String),
    ToolCallUnsupported,
    EngineFailed(String),
    ServerBusy,
}

impl Refusal {
    fn status(&self) -> (u16, &'static str) {
        use Refusal::*;
        match self {
            MethodNotAllowed(_) => (405, "Method Not Allowed"),
            RouteNotFound(_) => (404, "Not Found"),
            ContentLengthMissing => (411, "Length Required"),
            BodyTooLarge(_) | HeadTooLong => (413, "Content Too Large"),
            ChunkedBodyUnsupported => (501, "Not Implemented"),
            ToolCallUnsupported | EngineFailed(_) => (500, "Internal Server Error"),
            ServerBusy => (503, "Service Unavailable"),
            _ => (400, "Bad Request"),
        }
    }

    /// The name of the refusal, as the `code` of the error
    pub fn code(&self) -> &'static str {
        use Refusal::*;
        match self {
            RequestUnreadable(_) => "request_unreadable",
            HeadTooLong => "head_too_long",
            MethodNotAllowed(_) => "method_not_allowed",
            RouteNotFound(_) => "route_not_found",
            ContentLengthMissing => "content_length_missing",
            ChunkedBodyUnsupported => "chunked_body_unsupported",
            BodyTooLarge(_) => "body_too_large",
            BodyNotJson(_) => "body_not_json",
            BodyNotObject => "body_not_object",
            FieldUnsupported(_) => "field_unsupported",
            FieldMissing(_) => "field_missing",
            FieldInvalid(_) => "field_invalid",
            ModelNotServed(_) => "model_not_served",
            StreamUnsupported => "stream_unsupported",
            ReasoningEffortUnsupported(_) => "reasoning_effort_unsupported",
            MessageFieldUnsupported(_) => "message_field_unsupported",
            RoleUnsupported(_) => "role_unsupported",
            SystemMessageNotFirst => "system_message_not_first",
            LastMessageNotUser => "last_message_not_user",
            NoUserMessage => "no_user_message",
            ContextLengthExceeded(_) => "context_length_exceeded",
            ToolCallUnsupported => "tool_call_unsupported",
            EngineFailed(_) => "engine_failed",
            ServerBusy => "server_busy",
        }
    }

    fn message(&self) -> String {
        use Refusal::*;
        match self {
            RequestUnreadable(e) => format!("the request cannot be read: {e}"),
            HeadTooLong => "the head of the request is too long".into(),
            MethodNotAllowed(m) => format!("method {m}: the route takes {METHOD} only"),
            RouteNotFound(p) => format!("no route {p}: the server serves {ROUTE}"),
            ContentLengthMissing => "the request has no Content-Length".into(),
            ChunkedBodyUnsupported => "a chunked body is not supported".into(),
            BodyTooLarge(n) => format!("a body of {n} bytes is above {MAX_BODY_BYTES}"),
            BodyNotJson(e) => format!("the body is not JSON: {e}"),
            BodyNotObject => "the body is not a JSON object".into(),
            FieldUnsupported(f) => format!("the field {f} is not supported"),
            FieldMissing(f) => format!("the field {f} is required"),
            FieldInvalid(f) => format!("the field {f} has a value of the wrong type or range"),
            ModelNotServed(m) => format!("the model {m} is not the model of this server"),
            StreamUnsupported => "stream must be false: the server gives one full answer".into(),
            ReasoningEffortUnsupported(e) => {
                format!("reasoning_effort {e} is not supported: none, low, medium, high or xhigh")
            }
            MessageFieldUnsupported(f) => format!("the message field {f} is not supported"),
            RoleUnsupported(r) => {
                format!("the role {r} is not supported: system, user or assistant")
            }
            SystemMessageNotFirst => "a system message comes after another message".into(),
            LastMessageNotUser => "the last message is not from the user".into(),
            NoUserMessage => "the request has no user message".into(),
            ContextLengthExceeded(e) => e.clone(),
            ToolCallUnsupported => {
                "the model called a function, and tools are not supported".into()
            }
            EngineFailed(e) => e.clone(),
            ServerBusy => "the server answers one request at a time".into(),
        }
    }

    fn body(&self) -> String {
        json!({
            "error": {
                "message": self.message(),
                "type": "invalid_request_error",
                "code": self.code(),
            }
        })
        .to_string()
    }
}

/// One request, read from a client
#[derive(Debug, Clone, PartialEq)]
pub struct Request {
    /// the system messages, joined as the chat template joins them
    pub system: Option<String>,
    pub history: Vec<(Role, String)>,
    pub user: String,
    pub max_tokens: usize,
    /// None for no reasoning block
    pub effort: Option<ReasoningEffort>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
}

/// One line of the head of a request, into `line`
fn head_line<R: BufRead>(stream: &mut R, line: &mut String) -> Result<(), Refusal> {
    line.clear();
    let n = (&mut *stream)
        .take(MAX_HEAD_LINE_BYTES as u64)
        .read_line(line)
        .map_err(|e| Refusal::RequestUnreadable(e.to_string()))?;
    if n == 0 {
        return Err(Refusal::RequestUnreadable("the connection closed".into()));
    }
    if !line.ends_with('\n') {
        return Err(Refusal::HeadTooLong);
    }
    Ok(())
}

/// The body of an HTTP/1.1 request of the route, or why it is refused
fn read_http<R: BufRead>(stream: &mut R) -> Result<Vec<u8>, Refusal> {
    let mut line = String::new();
    head_line(stream, &mut line)?;
    let mut words = line.split_whitespace();
    let (Some(method), Some(path)) = (words.next(), words.next()) else {
        return Err(Refusal::RequestUnreadable(format!("request line {line:?}")));
    };
    let (method, path) = (method.to_string(), path.to_string());

    let mut length = None;
    let mut lines = 0;
    loop {
        head_line(stream, &mut line)?;
        let header = line.trim_end_matches(['\r', '\n']);
        if header.is_empty() {
            break;
        }
        lines += 1;
        if lines > MAX_HEAD_LINES {
            return Err(Refusal::HeadTooLong);
        }
        let Some((name, value)) = header.split_once(':') else {
            return Err(Refusal::RequestUnreadable(format!("header {header:?}")));
        };
        let (name, value) = (name.trim().to_ascii_lowercase(), value.trim());
        if name == TRANSFER_ENCODING {
            return Err(Refusal::ChunkedBodyUnsupported);
        }
        if name == CONTENT_LENGTH {
            let n: usize = value
                .parse()
                .map_err(|_| Refusal::RequestUnreadable(format!("Content-Length {value:?}")))?;
            length = Some(n);
        }
    }
    if path != ROUTE {
        return Err(Refusal::RouteNotFound(path));
    }
    if method != METHOD {
        return Err(Refusal::MethodNotAllowed(method));
    }
    let length = length.ok_or(Refusal::ContentLengthMissing)?;
    if length > MAX_BODY_BYTES {
        return Err(Refusal::BodyTooLarge(length));
    }
    let mut body = vec![0; length];
    stream
        .read_exact(&mut body)
        .map_err(|e| Refusal::RequestUnreadable(e.to_string()))?;
    Ok(body)
}

fn string_field<'v>(
    object: &'v Map<String, Value>,
    name: &'static str,
) -> Result<Option<&'v str>, Refusal> {
    match object.get(name) {
        None => Ok(None),
        Some(Value::String(s)) => Ok(Some(s)),
        Some(_) => Err(Refusal::FieldInvalid(name)),
    }
}

fn effort_of(word: &str) -> Result<Option<ReasoningEffort>, Refusal> {
    if word == EFFORT_NONE {
        return Ok(None);
    }
    // the levels of the chat template: low, medium and xhigh, high for xhigh
    ReasoningEffort::parse(word)
        .map(Some)
        .ok_or_else(|| Refusal::ReasoningEffortUnsupported(word.into()))
}

/// The request of `body`, for a server of `model`
pub fn parse_request(body: &[u8], model: &str) -> Result<Request, Refusal> {
    let value: Value =
        serde_json::from_slice(body).map_err(|e| Refusal::BodyNotJson(e.to_string()))?;
    let Value::Object(object) = value else {
        return Err(Refusal::BodyNotObject);
    };
    if let Some(field) = object
        .keys()
        .find(|k| !REQUEST_FIELDS.contains(&k.as_str()))
    {
        return Err(Refusal::FieldUnsupported(field.clone()));
    }
    let asked = string_field(&object, FIELD_MODEL)?.ok_or(Refusal::FieldMissing(FIELD_MODEL))?;
    if asked != model {
        return Err(Refusal::ModelNotServed(asked.into()));
    }
    match object.get(FIELD_STREAM) {
        None | Some(Value::Bool(false)) => {}
        Some(Value::Bool(true)) => return Err(Refusal::StreamUnsupported),
        Some(_) => return Err(Refusal::FieldInvalid(FIELD_STREAM)),
    }
    let max_tokens = object
        .get(FIELD_MAX_TOKENS)
        .ok_or(Refusal::FieldMissing(FIELD_MAX_TOKENS))?
        .as_u64()
        .filter(|&n| n > 0)
        .and_then(|n| usize::try_from(n).ok())
        .ok_or(Refusal::FieldInvalid(FIELD_MAX_TOKENS))?;
    let effort = effort_of(
        string_field(&object, FIELD_REASONING_EFFORT)?
            .ok_or(Refusal::FieldMissing(FIELD_REASONING_EFFORT))?,
    )?;

    let Some(Value::Array(messages)) = object.get(FIELD_MESSAGES) else {
        return Err(match object.get(FIELD_MESSAGES) {
            None => Refusal::FieldMissing(FIELD_MESSAGES),
            Some(_) => Refusal::FieldInvalid(FIELD_MESSAGES),
        });
    };
    let mut system: Option<String> = None;
    let mut turns = Vec::new();
    for m in messages {
        let Value::Object(m) = m else {
            return Err(Refusal::FieldInvalid(FIELD_MESSAGES));
        };
        if let Some(field) = m.keys().find(|k| !MESSAGE_FIELDS.contains(&k.as_str())) {
            return Err(Refusal::MessageFieldUnsupported(field.clone()));
        }
        let role = string_field(m, FIELD_ROLE)?.ok_or(Refusal::FieldMissing(FIELD_ROLE))?;
        let content =
            string_field(m, FIELD_CONTENT)?.ok_or(Refusal::FieldMissing(FIELD_CONTENT))?;
        match role {
            ROLE_SYSTEM if turns.is_empty() => {
                let content = content.trim();
                // the template leaves out an empty system message
                if !content.is_empty() {
                    system = Some(match system {
                        Some(s) => s + SYSTEM_SEPARATOR + content,
                        None => content.to_string(),
                    });
                }
            }
            ROLE_SYSTEM => return Err(Refusal::SystemMessageNotFirst),
            ROLE_USER => turns.push((Role::User, content.to_string())),
            ROLE_ASSISTANT => turns.push((Role::Assistant, content.to_string())),
            other => return Err(Refusal::RoleUnsupported(other.into())),
        }
    }
    let user = match turns.pop() {
        Some((Role::User, user)) => user,
        Some((Role::Assistant, _)) => return Err(Refusal::LastMessageNotUser),
        None => return Err(Refusal::NoUserMessage),
    };
    Ok(Request {
        system,
        history: turns,
        user,
        max_tokens,
        effort,
    })
}

/// The finish_reason of the OpenAI API for why a turn stopped
fn finish_reason(stop: Stop) -> &'static str {
    match stop {
        Stop::End => "stop",
        Stop::MaxTokens | Stop::ContextFull => "length",
        Stop::Cancel => unreachable!("the server never stops a turn"),
    }
}

/// The body of the answer to a request
pub fn answer_body(
    model: &str,
    id: u64,
    content: &str,
    reasoning: &str,
    stats: &TurnStats,
) -> String {
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    json!({
        "id": format!("chatcmpl-{id}"),
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": ROLE_ASSISTANT,
                "content": content,
                "reasoning_content": reasoning,
            },
            "finish_reason": finish_reason(stats.stop),
        }],
        "usage": {
            "prompt_tokens": stats.prompt_tokens,
            "completion_tokens": stats.gen_tokens,
            "total_tokens": stats.prompt_tokens + stats.gen_tokens,
        },
    })
    .to_string()
}

fn write_response(stream: &mut TcpStream, status: (u16, &str), body: &str) {
    let (code, reason) = status;
    let head = format!(
        "HTTP/1.1 {code} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    if let Err(e) = stream
        .write_all(head.as_bytes())
        .and_then(|()| stream.write_all(body.as_bytes()))
        .and_then(|()| stream.flush())
    {
        eprintln!("serve: the answer was not sent: {e}");
    }
}

fn refuse(stream: &mut TcpStream, refusal: &Refusal) {
    eprintln!("serve: refused {}: {}", refusal.code(), refusal.message());
    write_response(stream, refusal.status(), &refusal.body());
}

/// Read each connection of `listener`, and hand its body to the chat
/// thread, or refuse it when the chat thread is busy
fn accept(listener: TcpListener, chats: SyncSender<(TcpStream, Vec<u8>)>) {
    for stream in listener.incoming() {
        let mut stream = match stream {
            Ok(s) => s,
            Err(e) => {
                eprintln!("serve: a connection failed: {e}");
                continue;
            }
        };
        if let Err(e) = stream.set_read_timeout(Some(READ_TIMEOUT)) {
            eprintln!("serve: no read timeout: {e}");
            continue;
        }
        let body = match stream.try_clone() {
            Ok(s) => read_http(&mut BufReader::new(s)),
            Err(e) => Err(Refusal::RequestUnreadable(e.to_string())),
        };
        match body {
            Err(refusal) => refuse(&mut stream, &refusal),
            Ok(body) => match chats.try_send((stream, body)) {
                Ok(()) => {}
                Err(TrySendError::Full((mut stream, _))) => {
                    refuse(&mut stream, &Refusal::ServerBusy)
                }
                Err(TrySendError::Disconnected(_)) => return,
            },
        }
    }
}

/// A listener on `address`, which must be a loopback address: the server
/// has no authentication
pub fn listen(address: &str) -> Result<TcpListener, String> {
    let addrs: Vec<SocketAddr> = address
        .to_socket_addrs()
        .map_err(|e| format!("--listen {address}: {e}"))?
        .collect();
    if addrs.is_empty() || !addrs.iter().all(|a| a.ip().is_loopback()) {
        return Err(format!(
            "--listen {address}: not a loopback address; the server has no authentication"
        ));
    }
    TcpListener::bind(&addrs[..]).map_err(|e| format!("--listen {address}: {e}"))
}

/// What a server answers with
pub struct Server<'m> {
    pub engine: &'m dyn Engine,
    pub tok: &'m Tokenizer,
    /// the model name that a request must give
    pub model: String,
    /// the options of each chat; each request sets the system prompt, the
    /// reasoning and max_tokens
    pub opts: ChatOptions,
    pub sampler: SamplerConfig,
    /// a new seed for each request, else the seed of `sampler` for each one
    pub random_seed: bool,
}

impl Server<'_> {
    /// Serve the requests of `listener`, one at a time, until the process
    /// ends
    pub fn serve(&self, listener: TcpListener) -> std::io::Result<()> {
        let (chats, requests): (_, Receiver<(TcpStream, Vec<u8>)>) = sync_channel(WAITING_REQUESTS);
        std::thread::Builder::new()
            .name("accept".into())
            .spawn(move || accept(listener, chats))?;
        for (id, (mut stream, body)) in (1..).zip(requests) {
            match self.answer(id, &body) {
                Ok(answer) => write_response(&mut stream, (200, "OK"), &answer),
                Err(refusal) => refuse(&mut stream, &refusal),
            }
        }
        Ok(())
    }

    fn answer(&self, id: u64, body: &[u8]) -> Result<String, Refusal> {
        let request = parse_request(body, &self.model)?;
        let mut opts = self.opts.clone();
        opts.system = request.system;
        opts.thinking = request.effort.is_some();
        if let Some(effort) = request.effort {
            opts.effort = effort;
        }
        opts.max_tokens = request.max_tokens;
        let mut chat = Chat::new(self.engine, self.tok, opts)
            .map_err(|e| Refusal::EngineFailed(e.to_string()))?;
        let mut cfg = self.sampler.clone();
        if self.random_seed {
            cfg.seed = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_err(|e| Refusal::EngineFailed(e.to_string()))?
                .as_nanos() as u64;
        }
        let mut sampler = Sampler::new(cfg);
        let history: Vec<Message<'_>> = request
            .history
            .iter()
            .map(|(role, text)| match role {
                Role::User => Message::User(text),
                Role::Assistant => Message::Assistant(text),
            })
            .collect();

        let (mut content, mut reasoning) = (String::new(), String::new());
        let t0 = Instant::now();
        let mut t1 = None;
        let stats = chat
            .send_with_history(
                &history,
                &request.user,
                &mut sampler,
                |ev| match ev {
                    Event::PromptDone => t1 = Some(Instant::now()),
                    Event::Thinking(t) => reasoning.push_str(t),
                    Event::Answer(t) => content.push_str(t),
                    Event::Token(_) | Event::ThinkingDone | Event::ToolCall(_) => {}
                },
                || false,
            )
            .map_err(|e| match e {
                biv::Error::ContextFull { .. } => Refusal::ContextLengthExceeded(e.to_string()),
                e => Refusal::EngineFailed(e.to_string()),
            })?;
        if !stats.tool_calls.is_empty() {
            return Err(Refusal::ToolCallUnsupported);
        }
        let t1 = t1.unwrap_or(t0);
        let (prompt_s, gen_s) = ((t1 - t0).as_secs_f64(), t1.elapsed().as_secs_f64());
        eprintln!(
            "serve: request {id}: {} prompt tokens in {prompt_s:.2}s ({:.1} tok/s), {} answer tokens in {gen_s:.2}s ({:.1} tok/s), {}",
            stats.prompt_tokens,
            stats.prompt_tokens as f64 / prompt_s,
            stats.gen_tokens,
            stats.gen_tokens as f64 / gen_s,
            finish_reason(stats.stop),
        );
        Ok(answer_body(&self.model, id, &content, &reasoning, &stats))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODEL: &str = "qwen";
    const ASKED_TOKENS: usize = 512;

    fn body(extra: &str) -> String {
        format!(
            r#"{{"model":"{MODEL}","stream":false,"max_tokens":{ASKED_TOKENS},"reasoning_effort":"none","messages":[{{"role":"system","content":"Be terse."}},{{"role":"user","content":"a"}},{{"role":"assistant","content":"b"}},{{"role":"user","content":"c"}}]{extra}}}"#
        )
    }

    fn parse(text: &str) -> Result<Request, Refusal> {
        parse_request(text.as_bytes(), MODEL)
    }

    #[test]
    fn a_request_of_the_fields_of_the_route_gives_each_value() {
        let r = parse(&body("")).expect("request");
        assert_eq!(r.system.as_deref(), Some("Be terse."));
        assert_eq!(
            r.history,
            [
                (Role::User, "a".to_string()),
                (Role::Assistant, "b".to_string())
            ]
        );
        assert_eq!(r.user, "c");
        assert_eq!(r.max_tokens, ASKED_TOKENS);
        assert_eq!(r.effort, None);
    }

    #[test]
    fn each_reasoning_effort_of_the_template_is_taken() {
        for (word, effort) in [
            ("low", Some(ReasoningEffort::Low)),
            ("medium", Some(ReasoningEffort::Medium)),
            ("high", Some(ReasoningEffort::XHigh)),
            ("xhigh", Some(ReasoningEffort::XHigh)),
            ("none", None),
        ] {
            assert_eq!(effort_of(word), Ok(effort), "{word}");
        }
        assert_eq!(
            effort_of("minimal"),
            Err(Refusal::ReasoningEffortUnsupported("minimal".into()))
        );
    }

    #[test]
    fn a_field_that_the_server_does_not_take_is_refused_by_name() {
        let r = parse(&body(r#","temperature":0.5"#));
        assert_eq!(r, Err(Refusal::FieldUnsupported("temperature".into())));
        let message = r#"{"model":"qwen","max_tokens":8,"reasoning_effort":"none","messages":[{"role":"user","content":"a","name":"x"}]}"#;
        assert_eq!(
            parse(message),
            Err(Refusal::MessageFieldUnsupported("name".into()))
        );
    }

    #[test]
    fn each_required_field_is_refused_by_name_when_it_is_missing() {
        for field in [
            FIELD_MODEL,
            FIELD_MAX_TOKENS,
            FIELD_REASONING_EFFORT,
            FIELD_MESSAGES,
        ] {
            let mut v: Value = serde_json::from_str(&body("")).expect("json");
            v.as_object_mut().expect("object").remove(field);
            assert_eq!(
                parse(&v.to_string()),
                Err(Refusal::FieldMissing(field)),
                "{field}"
            );
        }
    }

    #[test]
    fn a_request_that_the_server_cannot_answer_as_asked_is_refused() {
        let other = body("").replace(r#""model":"qwen""#, r#""model":"other""#);
        assert_eq!(parse(&other), Err(Refusal::ModelNotServed("other".into())));
        let stream = body("").replace(r#""stream":false"#, r#""stream":true"#);
        assert_eq!(parse(&stream), Err(Refusal::StreamUnsupported));
        let zero = body("").replace(r#""max_tokens":512"#, r#""max_tokens":0"#);
        assert_eq!(parse(&zero), Err(Refusal::FieldInvalid(FIELD_MAX_TOKENS)));
        let tool = body("").replace(r#""role":"assistant""#, r#""role":"tool""#);
        assert_eq!(parse(&tool), Err(Refusal::RoleUnsupported("tool".into())));
        let late = body("").replace(r#""role":"assistant""#, r#""role":"system""#);
        assert_eq!(parse(&late), Err(Refusal::SystemMessageNotFirst));
        let last = body("").replace(
            r#"{"role":"user","content":"c"}"#,
            r#"{"role":"assistant","content":"c"}"#,
        );
        assert_eq!(parse(&last), Err(Refusal::LastMessageNotUser));
        assert_eq!(parse("[]"), Err(Refusal::BodyNotObject));
        assert!(matches!(parse("{"), Err(Refusal::BodyNotJson(_))));
    }

    fn http(text: &str) -> Result<Vec<u8>, Refusal> {
        read_http(&mut text.as_bytes())
    }

    #[test]
    fn an_http_request_gives_its_body_or_a_named_refusal() {
        let ok = format!("POST {ROUTE} HTTP/1.1\r\nHost: x\r\ncontent-length: 2\r\n\r\n{{}}");
        assert_eq!(http(&ok), Ok(b"{}".to_vec()));
        let get = format!("GET {ROUTE} HTTP/1.1\r\n\r\n");
        assert_eq!(http(&get), Err(Refusal::MethodNotAllowed("GET".into())));
        let path = "POST /v1/completions HTTP/1.1\r\n\r\n";
        assert_eq!(
            http(path),
            Err(Refusal::RouteNotFound("/v1/completions".into()))
        );
        let no_length = format!("POST {ROUTE} HTTP/1.1\r\n\r\n");
        assert_eq!(http(&no_length), Err(Refusal::ContentLengthMissing));
        let chunked = format!("POST {ROUTE} HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n");
        assert_eq!(http(&chunked), Err(Refusal::ChunkedBodyUnsupported));
        let large = format!(
            "POST {ROUTE} HTTP/1.1\r\nContent-Length: {}\r\n\r\n",
            MAX_BODY_BYTES + 1
        );
        assert_eq!(http(&large), Err(Refusal::BodyTooLarge(MAX_BODY_BYTES + 1)));
    }

    #[test]
    fn the_answer_gives_the_content_first_and_the_usage() {
        let stats = TurnStats {
            prompt_tokens: 3,
            gen_tokens: 2,
            stop: Stop::MaxTokens,
            drafted: 0,
            accepted: 0,
            looked_up: 0,
            looked_up_accepted: 0,
            tool_calls: Vec::new(),
            tokens: Vec::new(),
        };
        let text = answer_body(MODEL, 1, "hi \"you\"", "", &stats);
        let v: Value = serde_json::from_str(&text).expect("json");
        assert_eq!(v["choices"][0]["message"]["content"], "hi \"you\"");
        assert_eq!(v["choices"][0]["finish_reason"], "length");
        assert_eq!(v["usage"]["prompt_tokens"], 3);
        assert_eq!(v["usage"]["completion_tokens"], 2);
        // a client that reads the first "content" key reads the answer
        let first = text.find(r#""content":"#).expect("content");
        assert!(text[first..].starts_with(r#""content":"hi \"you\"""#));
    }
}
