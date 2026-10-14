//! The gateway's JSON log, byte for byte what Go's `slog.NewJSONHandler` wrote, and the JSON text
//! encoding Go's `encoding/json` gives the HTTP bodies, which differs from slog's in two escapes.

use std::fmt;
use std::io::{self, Write};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::SystemTime;

use chrono::{DateTime, FixedOffset, Local, Offset, Timelike};
use serde::Serialize;
use serde_json::Value;
use serde_json::ser::{CharEscape, Formatter, Serializer};

use crate::overflow;

/// A line's level, with slog's numeric severity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    Debug,
    Info,
    Warn,
    Error,
}

impl Level {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Debug => "DEBUG",
            Self::Info => "INFO",
            Self::Warn => "WARN",
            Self::Error => "ERROR",
        }
    }

    const fn severity(self) -> i64 {
        match self {
            Self::Debug => -4,
            Self::Info => 0,
            Self::Warn => 4,
            Self::Error => 8,
        }
    }
}

impl From<overflow::Level> for Level {
    fn from(l: overflow::Level) -> Self {
        match l {
            overflow::Level::Debug => Self::Debug,
            overflow::Level::Info => Self::Info,
            overflow::Level::Warn => Self::Warn,
            overflow::Level::Error => Self::Error,
        }
    }
}

/// The lowest severity a `Logger` writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Threshold(i64);

impl Threshold {
    pub const INFO: Self = Self(Level::Info.severity());

    /// Parses what `slog.Level.UnmarshalText` accepts: a level name in any case, optionally with
    /// a signed offset, as in "warn" or "INFO+2".
    pub fn parse(s: &str) -> Option<Self> {
        let (name, offset) = match s.find(['+', '-']) {
            Some(i) => (&s[..i], go_atoi(&s[i..])?),
            None => (s, 0),
        };
        let base = [Level::Debug, Level::Info, Level::Warn, Level::Error]
            .into_iter()
            .find(|l| name.eq_ignore_ascii_case(l.as_str()))?;
        base.severity().checked_add(offset).map(Self)
    }
}

/// `strconv.Atoi`: an optional sign and decimal digits, nothing else.
pub fn go_atoi(s: &str) -> Option<i64> {
    let digits = s.strip_prefix(['+', '-']).unwrap_or(s);
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    s.parse().ok()
}

/// Writes one JSON object per line. Cloning is cheap and every clone writes to the same sink.
#[derive(Clone)]
pub struct Logger {
    threshold: Threshold,
    sink: Arc<Mutex<Box<dyn Write + Send>>>,
}

impl fmt::Debug for Logger {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Logger")
            .field("threshold", &self.threshold)
            .finish_non_exhaustive()
    }
}

impl Logger {
    pub fn new(threshold: Threshold, sink: impl Write + Send + 'static) -> Self {
        Self {
            threshold,
            sink: Arc::new(Mutex::new(Box::new(sink))),
        }
    }

    pub fn discard() -> Self {
        Self::new(Threshold(i64::MAX), io::sink())
    }

    pub fn log(&self, level: Level, msg: &str, attrs: &[(&str, Value)]) {
        if level.severity() < self.threshold.0 {
            return;
        }
        let mut out = line(Local::now().fixed_offset(), level, msg, attrs);
        out.push(b'\n');
        let mut sink = self.sink.lock().unwrap_or_else(PoisonError::into_inner);
        // slog drops a line it cannot write; so does this.
        let _ = sink.write_all(&out);
    }

    pub fn info(&self, msg: &str, attrs: &[(&str, Value)]) {
        self.log(Level::Info, msg, attrs);
    }

    pub fn warn(&self, msg: &str, attrs: &[(&str, Value)]) {
        self.log(Level::Warn, msg, attrs);
    }

    /// This logger as the overflow circuit's `Log` callback.
    pub fn overflow_log(&self) -> overflow::Log {
        let log = self.clone();
        Arc::new(move |level, msg, attrs| log.log(level.into(), msg, attrs))
    }
}

/// One line without its newline: `time`, `level`, `msg`, then the attributes in order.
pub fn line(
    time: DateTime<FixedOffset>,
    level: Level,
    msg: &str,
    attrs: &[(&str, Value)],
) -> Vec<u8> {
    let head = [
        ("time", Value::from(rfc3339_nano(time))),
        ("level", Value::from(level.as_str())),
        ("msg", Value::from(msg)),
    ];
    let mut out = vec![b'{'];
    let pairs = head
        .iter()
        .map(|(k, v)| (*k, v))
        .chain(attrs.iter().map(|(k, v)| (*k, v)));
    for (i, (k, v)) in pairs.enumerate() {
        if i > 0 {
            out.push(b',');
        }
        out.extend(encode(&k, Flavor::Slog));
        out.push(b':');
        out.extend(encode(v, Flavor::Slog));
    }
    out.push(b'}');
    out
}

/// Which Go encoder's escaping to follow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flavor {
    /// `log/slog`: no HTML escaping, and `\b` and `\f` as `\u0008` and `\u000c`.
    Slog,
    /// `encoding/json` with its default HTML escaping of `<`, `>` and `&`.
    Json,
}

/// `v` as Go would encode it. Floats take Go's form (`1e+21`, `0.000025`), and U+2028 and U+2029
/// are escaped as both Go encoders do.
pub fn encode(v: &(impl Serialize + ?Sized), flavor: Flavor) -> Vec<u8> {
    let mut out = Vec::new();
    let mut ser = Serializer::with_formatter(&mut out, GoFormatter(flavor));
    // Serializing into a Vec fails only for a map with non-string keys, which no caller builds.
    let _ = v.serialize(&mut ser);
    out
}

struct GoFormatter(Flavor);

impl Formatter for GoFormatter {
    fn write_f64<W: ?Sized + Write>(&mut self, w: &mut W, v: f64) -> io::Result<()> {
        w.write_all(go_float(v).as_bytes())
    }

    fn write_f32<W: ?Sized + Write>(&mut self, w: &mut W, v: f32) -> io::Result<()> {
        self.write_f64(w, f64::from(v))
    }

    fn write_char_escape<W: ?Sized + Write>(&mut self, w: &mut W, e: CharEscape) -> io::Result<()> {
        let short: &[u8] = match (e, self.0) {
            (CharEscape::Quote, _) => b"\\\"",
            (CharEscape::ReverseSolidus, _) => b"\\\\",
            (CharEscape::Solidus, _) => b"\\/",
            (CharEscape::LineFeed, _) => b"\\n",
            (CharEscape::CarriageReturn, _) => b"\\r",
            (CharEscape::Tab, _) => b"\\t",
            (CharEscape::Backspace, Flavor::Json) => b"\\b",
            (CharEscape::FormFeed, Flavor::Json) => b"\\f",
            (CharEscape::Backspace, Flavor::Slog) => return write!(w, "\\u0008"),
            (CharEscape::FormFeed, Flavor::Slog) => return write!(w, "\\u000c"),
            (CharEscape::AsciiControl(b), _) => return write!(w, "\\u{b:04x}"),
        };
        w.write_all(short)
    }

    fn write_string_fragment<W: ?Sized + Write>(
        &mut self,
        w: &mut W,
        fragment: &str,
    ) -> io::Result<()> {
        let mut start = 0;
        for (i, c) in fragment.char_indices() {
            let escape = match c {
                '\u{2028}' | '\u{2029}' => true,
                '<' | '>' | '&' => self.0 == Flavor::Json,
                _ => false,
            };
            if escape {
                w.write_all(&fragment.as_bytes()[start..i])?;
                write!(w, "\\u{:04x}", u32::from(c))?;
                start = i + c.len_utf8();
            }
        }
        w.write_all(&fragment.as_bytes()[start..])
    }
}

/// `encoding/json`'s float form: shortest digits, exponent form below 1e-6 and from 1e21.
pub fn go_float(f: f64) -> String {
    let abs = f.abs();
    if abs != 0.0 && (abs < 1e-6 || abs >= 1e21) {
        let s = format!("{f:e}");
        match s.split_once('e') {
            Some((mantissa, exp)) if !exp.starts_with('-') => format!("{mantissa}e+{exp}"),
            _ => s,
        }
    } else {
        format!("{f}")
    }
}

/// `time.RFC3339Nano`: trailing zeros of the fraction dropped, `Z` for a zero offset.
pub fn rfc3339_nano(t: DateTime<FixedOffset>) -> String {
    let mut s = t.format("%Y-%m-%dT%H:%M:%S").to_string();
    let nanos = t.nanosecond() % 1_000_000_000;
    if nanos != 0 {
        let frac = format!("{nanos:09}");
        s.push('.');
        s.push_str(frac.trim_end_matches('0'));
    }
    if t.offset().fix().local_minus_utc() == 0 {
        s.push('Z');
    } else {
        s.push_str(&t.format("%:z").to_string());
    }
    s
}

/// A wall time in the local zone, as Go's `time.Time` marshals it.
pub fn wall_time(t: SystemTime) -> String {
    rfc3339_nano(DateTime::<Local>::from(t).fixed_offset())
}

#[cfg(test)]
mod tests {
    use chrono::TimeZone;

    use super::*;

    fn at(offset_s: i32, nanos: u32) -> DateTime<FixedOffset> {
        FixedOffset::east_opt(offset_s)
            .and_then(|tz| tz.with_ymd_and_hms(2026, 10, 14, 8, 28, 22).single())
            .and_then(|t| t.with_nanosecond(nanos))
            .expect("valid test time")
    }

    #[test]
    fn line_matches_slog_byte_for_byte() {
        // Recorded from Go 1.27's slog.NewJSONHandler with the same attributes.
        let attrs = [
            (
                "s",
                Value::from("a<b>&c\"\\\u{8}\u{c}\n\t\u{1}\u{2028}\u{fffd}\u{fffd}A é"),
            ),
            ("f", Value::from(1e21)),
            ("g", Value::from(12.3)),
            ("h", Value::from(0.0)),
            ("i", Value::from(1e-7)),
            ("j", Value::from(123_456_789.0)),
            ("k", Value::from(1e20)),
            ("e", Value::from("x<y")),
            ("pods", Value::from(vec!["a<b", "c"])),
            ("b", Value::from(true)),
            ("n", Value::from(5)),
            ("err", Value::Null),
        ];
        let got = line(at(3600, 844_162_000), Level::Info, "m<&>", &attrs);
        assert_eq!(
            String::from_utf8(got).expect("utf-8"),
            r#"{"time":"2026-10-14T08:28:22.844162+01:00","level":"INFO","msg":"m<&>","s":"a<b>&c\"\\\u0008\u000c\n\t\u0001\u2028��A é","f":1e+21,"g":12.3,"h":0,"i":1e-7,"j":123456789,"k":100000000000000000000,"e":"x<y","pods":["a<b","c"],"b":true,"n":5,"err":null}"#
        );
    }

    #[test]
    fn json_flavor_matches_encoding_json() {
        let v = serde_json::json!({
            "f": [1e21, 0.1, 1e-7, 100.0, 2.5e-5],
            "s": "a<b>&c\"\\\u{8}\u{c}\n\t\u{1}\u{2028}A é\u{7f}",
        });
        assert_eq!(
            String::from_utf8(encode(&v, Flavor::Json)).expect("utf-8"),
            "{\"f\":[1e+21,0.1,1e-7,100,0.000025],\"s\":\"a\\u003cb\\u003e\\u0026c\\\"\\\\\\b\\f\\n\\t\\u0001\\u2028A é\u{7f}\"}"
        );
    }

    #[test]
    fn time_formats_as_rfc3339_nano() {
        assert_eq!(rfc3339_nano(at(0, 120_000_000)), "2026-10-14T08:28:22.12Z");
        assert_eq!(rfc3339_nano(at(3600, 0)), "2026-10-14T08:28:22+01:00");
        assert_eq!(
            rfc3339_nano(at(-34200, 1)),
            "2026-10-14T08:28:22.000000001-09:30"
        );
    }

    #[test]
    fn threshold_parses_what_slog_accepts() {
        let cases = [
            ("debug", Some(-4)),
            ("INFO", Some(0)),
            ("Warn", Some(4)),
            ("error", Some(8)),
            ("info+2", Some(2)),
            ("WARN-1", Some(3)),
            ("loud", None),
            ("info+", None),
            ("", None),
            ("info+x", None),
        ];
        for (text, want) in cases {
            assert_eq!(Threshold::parse(text).map(|t| t.0), want, "{text:?}");
        }
    }

    #[test]
    fn logger_drops_lines_below_its_threshold() {
        #[derive(Clone, Default)]
        struct Shared(Arc<Mutex<Vec<u8>>>);
        impl Write for Shared {
            fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
                self.0.lock().expect("lock").write(buf)
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        let buf = Shared::default();
        let log = Logger::new(Threshold::parse("warn").expect("level"), buf.clone());
        log.info("quiet", &[]);
        (log.overflow_log())(overflow::Level::Warn, "loud", &[("n", Value::from(1))]);
        let text = String::from_utf8(buf.0.lock().expect("lock").clone()).expect("utf-8");
        assert_eq!(text.lines().count(), 1);
        assert!(
            text.ends_with("\"level\":\"WARN\",\"msg\":\"loud\",\"n\":1}\n"),
            "{text}"
        );
    }
}
