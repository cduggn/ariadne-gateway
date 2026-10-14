//! The gateway's wire shapes: what it reads from a worker's answer and what it writes as JSON.

use ariadne_decide::WorkerState;
use serde::Serialize;
use serde_json::{Map, Value};

use super::Usage;
use crate::fleet::{Phase, WorkerStatus};
use crate::log;

/// The usage and the first finish reason of a chat completion, read with Go's `encoding/json`
/// rules: a value of the wrong type anywhere in the decoded shape fails the whole decode, an
/// integer field takes only an integer literal, a missing or `null` field reads as zero, and a
/// key matches its field exactly or else ignoring ASCII case. `None` when the body carried no
/// usage object or did not decode.
pub(super) fn usage_of(body: &[u8]) -> Option<Usage> {
    let Ok(Value::Object(top)) = serde_json::from_slice::<Value>(body) else {
        return None;
    };
    let finish_reason = match field(&top, "choices") {
        None | Some(Value::Null) => String::new(),
        Some(Value::Array(choices)) => {
            let mut reasons = choices.iter().map(|c| match c {
                Value::Null => Some(String::new()),
                Value::Object(c) => match field(c, "finish_reason") {
                    None | Some(Value::Null) => Some(String::new()),
                    Some(Value::String(s)) => Some(s.clone()),
                    Some(_) => None,
                },
                _ => None,
            });
            let first = reasons.next().unwrap_or(Some(String::new()))?;
            if reasons.any(|r| r.is_none()) {
                return None;
            }
            first
        }
        Some(_) => return None,
    };
    let Some(Value::Object(usage)) = field(&top, "usage") else {
        return None;
    };
    let cached_tokens = match field(usage, "prompt_tokens_details") {
        None | Some(Value::Null) => 0,
        Some(Value::Object(details)) => int(field(details, "cached_tokens"))?,
        Some(_) => return None,
    };
    Some(Usage {
        prompt_tokens: int(field(usage, "prompt_tokens"))?,
        cached_tokens,
        completion_tokens: int(field(usage, "completion_tokens"))?,
        finish_reason,
    })
}

fn field<'a>(m: &'a Map<String, Value>, key: &str) -> Option<&'a Value> {
    m.get(key).or_else(|| {
        m.iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(key))
            .map(|(_, v)| v)
    })
}

/// A Go `int` field. A negative count, which no worker sends, reads as zero.
fn int(v: Option<&Value>) -> Option<u64> {
    match v {
        None | Some(Value::Null) => Some(0),
        Some(Value::Number(n)) => n.as_i64().map(|n| u64::try_from(n).unwrap_or(0)),
        Some(_) => None,
    }
}

/// The OpenAI error shape, the only error body the client parses.
#[derive(Serialize)]
pub(super) struct ApiError<'a> {
    pub error: ApiErrorBody<'a>,
}

#[derive(Serialize)]
pub(super) struct ApiErrorBody<'a> {
    pub message: &'a str,
    #[serde(rename = "type")]
    pub kind: &'a str,
    pub code: &'a str,
}

/// One pod on `/debug/workers`. `phase` is the status phase as text, because Go's wire form of
/// the phase is a number.
#[derive(Serialize)]
pub(super) struct DebugWorker<'a> {
    pod: &'a str,
    phase: &'static str,
    status: StatusJson<'a>,
    view: StateJson<'a>,
}

impl<'a> DebugWorker<'a> {
    /// `view` is `None` for a pod the gate does not know, which Go showed as the zero state.
    pub(super) fn new(st: &'a WorkerStatus, view: Option<&'a WorkerState>) -> Self {
        let v = view.map(|ws| &ws.view);
        Self {
            pod: &st.pod,
            phase: st.phase.as_str(),
            status: StatusJson {
                pod: &st.pod,
                phase: match st.phase {
                    Phase::Down => 0,
                    Phase::Warming => 1,
                    Phase::Ready => 2,
                },
                last_scrape: st
                    .last_scrape
                    .map_or_else(|| "0001-01-01T00:00:00Z".to_owned(), log::wall_time),
                last_probe_s: st.last_probe_s,
                probes: st.probes,
                restarts: st.restarts,
            },
            view: StateJson {
                view: ViewJson {
                    pod: v.map_or("", |v| v.pod.as_str()),
                    kv_pool_tokens: v.map_or(0, |v| v.kv_pool_tokens),
                    kv_usage: v.map_or(0.0, |v| v.kv_usage),
                    reserved_tokens: v.map_or(0, |v| v.reserved_tokens),
                    in_flight: v.map_or(0, |v| v.in_flight),
                    queued: v.map_or(0, |v| v.queued),
                    max_inflight: v.map_or(0, |v| v.max_inflight),
                    mean_service_s: v.map_or(0.0, |v| v.mean_service_s),
                    ttft_p50_s: v.map_or(0.0, |v| v.ttft_p50_s),
                    ttft_p99_s: v.map_or(0.0, |v| v.ttft_p99_s),
                    waiting_rising: v.is_some_and(|v| v.waiting_rising),
                },
                ready: view.is_some_and(|ws| ws.ready),
                age: view.map_or(0, |ws| i64::try_from(ws.age.as_nanos()).unwrap_or(i64::MAX)),
            },
        }
    }
}

/// `fleet.WorkerStatus` as Go marshalled it, with no json tags.
#[derive(Serialize)]
#[serde(rename_all = "PascalCase")]
struct StatusJson<'a> {
    pod: &'a str,
    phase: u8,
    last_scrape: String,
    last_probe_s: f64,
    probes: u64,
    restarts: u64,
}

/// `decide.WorkerState`; `Age` is a Go `time.Duration`, integer nanoseconds.
#[derive(Serialize)]
#[serde(rename_all = "PascalCase")]
struct StateJson<'a> {
    view: ViewJson<'a>,
    ready: bool,
    age: i64,
}

#[derive(Serialize)]
struct ViewJson<'a> {
    #[serde(rename = "Pod")]
    pod: &'a str,
    #[serde(rename = "KVPoolTokens")]
    kv_pool_tokens: u64,
    #[serde(rename = "KVUsage")]
    kv_usage: f64,
    #[serde(rename = "ReservedTokens")]
    reserved_tokens: u64,
    #[serde(rename = "InFlight")]
    in_flight: usize,
    #[serde(rename = "Queued")]
    queued: usize,
    #[serde(rename = "MaxInflight")]
    max_inflight: usize,
    #[serde(rename = "MeanServiceS")]
    mean_service_s: f64,
    #[serde(rename = "TTFTp50S")]
    ttft_p50_s: f64,
    #[serde(rename = "TTFTp99S")]
    ttft_p99_s: f64,
    #[serde(rename = "WaitingRising")]
    waiting_rising: bool,
}

/// A header value as the Go string `Header.Get` returned, with each byte that is not part of
/// valid UTF-8 shown as U+FFFD, as Go's JSON encoders show it.
pub(super) fn go_text(b: &[u8]) -> String {
    let mut out = String::with_capacity(b.len());
    let mut rest = b;
    loop {
        match std::str::from_utf8(rest) {
            Ok(s) => {
                out.push_str(s);
                return out;
            }
            Err(e) => {
                let (valid, bad) = rest.split_at(e.valid_up_to());
                out.push_str(&String::from_utf8_lossy(valid));
                let n = e.error_len().unwrap_or(bad.len());
                out.extend(std::iter::repeat_n('\u{fffd}', n));
                rest = &bad[n..];
            }
        }
    }
}

/// The type Go's server gave a body written without one: none for an empty body, else text
/// unless the first 512 bytes hold a byte no text has. Go's HTML, XML and media signatures are
/// not recognised; a worker always names its type.
pub(super) fn sniff(body: &[u8]) -> Option<&'static str> {
    if body.is_empty() {
        return None;
    }
    let binary = body
        .iter()
        .take(512)
        .any(|&b| matches!(b, 0x00..=0x08 | 0x0b | 0x0e..=0x1a | 0x1c..=0x1f));
    Some(if binary {
        "application/octet-stream"
    } else {
        "text/plain; charset=utf-8"
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn usage(prompt: u64, cached: u64, completion: u64, finish: &str) -> Option<Usage> {
        Some(Usage {
            prompt_tokens: prompt,
            cached_tokens: cached,
            completion_tokens: completion,
            finish_reason: finish.to_owned(),
        })
    }

    #[test]
    fn usage_of_reads_what_go_decoded() {
        let cases: [(&str, Option<Usage>); 12] = [
            (
                r#"{"usage":{"prompt_tokens":10,"completion_tokens":3,"prompt_tokens_details":{"cached_tokens":4}},"choices":[{"finish_reason":"stop"},{"finish_reason":"length"}]}"#,
                usage(10, 4, 3, "stop"),
            ),
            (r#"{"usage":{}}"#, usage(0, 0, 0, "")),
            (
                r#"{"USAGE":{"Prompt_Tokens":7},"choices":[null]}"#,
                usage(7, 0, 0, ""),
            ),
            (r#"{"usage":null}"#, None),
            (r#"{"choices":[]}"#, None),
            (r#"{"usage":{"prompt_tokens":1.0}}"#, None),
            (r#"{"usage":{"prompt_tokens":"1"}}"#, None),
            (r#"{"usage":{},"choices":{}}"#, None),
            (r#"{"usage":{},"choices":[{"finish_reason":1}]}"#, None),
            (r#"{"usage":{"prompt_tokens_details":[]}}"#, None),
            ("not json", None),
            ("[1]", None),
        ];
        for (body, want) in cases {
            assert_eq!(usage_of(body.as_bytes()), want, "{body}");
        }
    }

    #[test]
    fn go_text_replaces_each_invalid_byte() {
        assert_eq!(go_text(b"a\xe2\x82Ab\xff"), "a\u{fffd}\u{fffd}Ab\u{fffd}");
        assert_eq!(go_text("é".as_bytes()), "é");
    }

    #[test]
    fn sniff_tells_text_from_binary() {
        assert_eq!(sniff(b""), None);
        assert_eq!(sniff(b"{\"a\":1}\n"), Some("text/plain; charset=utf-8"));
        assert_eq!(sniff(b"\x00\x01"), Some("application/octet-stream"));
    }
}
