//! Go's `time.Duration` text form, so flags and log lines keep the syntax the
//! Go binaries used (`300ms`, `1m30s`, `1.5s`).

use std::fmt;
use std::str::FromStr;
use std::time::Duration;

const MICROSECOND: u64 = 1_000;
const MILLISECOND: u64 = 1_000_000;
const SECOND: u64 = 1_000_000_000;

/// Go's largest magnitude, the absolute value of the most negative Duration.
const LIMIT: u64 = 1 << 63;

/// A signed nanosecond count, as Go's `time.Duration` is. It parses with
/// `time.ParseDuration`'s grammar and displays as `Duration.String()` does.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GoDuration(i64);

impl GoDuration {
    pub const fn from_nanos(nanos: i64) -> Self {
        Self(nanos)
    }

    pub const fn nanos(self) -> i64 {
        self.0
    }

    /// The duration as a std `Duration`. A negative duration becomes zero,
    /// which is how Go's timers treat one.
    pub fn to_std(self) -> Duration {
        Duration::from_nanos(u64::try_from(self.0).unwrap_or(0))
    }
}

/// Saturates at Go's maximum, about 292 years.
impl From<Duration> for GoDuration {
    fn from(d: Duration) -> Self {
        Self(i64::try_from(d.as_nanos()).unwrap_or(i64::MAX))
    }
}

/// Why a string is not a Go duration. The messages are `time.ParseDuration`'s.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParseDurationError {
    Invalid { input: String },
    MissingUnit { input: String },
    UnknownUnit { unit: String, input: String },
}

/// Quotes as Go's `time` package does, with every non-ASCII or control byte
/// as a `\x` escape.
fn quote(s: &str) -> String {
    let mut out = String::from("\"");
    for b in s.bytes() {
        match b {
            b'"' | b'\\' => {
                out.push('\\');
                out.push(char::from(b));
            }
            0x20..=0x7f => out.push(char::from(b)),
            _ => out.push_str(&format!("\\x{b:02x}")),
        }
    }
    out.push('"');
    out
}

impl fmt::Display for ParseDurationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid { input } => write!(f, "time: invalid duration {}", quote(input)),
            Self::MissingUnit { input } => {
                write!(f, "time: missing unit in duration {}", quote(input))
            }
            Self::UnknownUnit { unit, input } => write!(
                f,
                "time: unknown unit {} in duration {}",
                quote(unit),
                quote(input)
            ),
        }
    }
}

impl std::error::Error for ParseDurationError {}

fn unit_nanos(unit: &str) -> Option<u64> {
    Some(match unit {
        "ns" => 1,
        // U+00B5 micro sign and U+03BC Greek mu, both of which Go accepts.
        "us" | "\u{b5}s" | "\u{3bc}s" => MICROSECOND,
        "ms" => MILLISECOND,
        "s" => SECOND,
        "m" => 60 * SECOND,
        "h" => 3600 * SECOND,
        _ => return None,
    })
}

fn digits(s: &str) -> (&str, &str) {
    s.split_at(s.bytes().take_while(u8::is_ascii_digit).count())
}

impl FromStr for GoDuration {
    type Err = ParseDurationError;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        let invalid = || ParseDurationError::Invalid {
            input: input.to_owned(),
        };
        let (negative, mut s) = match input.as_bytes().first() {
            Some(b'-') => (true, &input[1..]),
            Some(b'+') => (false, &input[1..]),
            _ => (false, input),
        };
        if s == "0" {
            return Ok(Self(0));
        }
        if s.is_empty() {
            return Err(invalid());
        }

        let mut total: u64 = 0;
        while !s.is_empty() {
            let (whole, rest) = digits(s);
            let mut value: u64 = 0;
            for b in whole.bytes() {
                value = value
                    .checked_mul(10)
                    .and_then(|v| v.checked_add(u64::from(b - b'0')))
                    .filter(|v| *v <= LIMIT)
                    .ok_or_else(invalid)?;
            }
            let (fraction, rest) = match rest.strip_prefix('.') {
                Some(after) => digits(after),
                None => ("", rest),
            };
            if whole.is_empty() && fraction.is_empty() {
                return Err(invalid());
            }

            let unit_len = rest
                .bytes()
                .take_while(|b| *b != b'.' && !b.is_ascii_digit())
                .count();
            if unit_len == 0 {
                return Err(ParseDurationError::MissingUnit {
                    input: input.to_owned(),
                });
            }
            let (unit, rest) = rest.split_at(unit_len);
            let Some(unit_ns) = unit_nanos(unit) else {
                return Err(ParseDurationError::UnknownUnit {
                    unit: unit.to_owned(),
                    input: input.to_owned(),
                });
            };
            s = rest;

            value = value
                .checked_mul(unit_ns)
                .filter(|v| *v <= LIMIT)
                .ok_or_else(invalid)?;

            // Go stops reading fraction digits once they overflow, and so
            // ignores the rest instead of failing.
            let (mut numerator, mut scale) = (0u64, 1f64);
            for b in fraction.bytes() {
                let Some(next) = numerator
                    .checked_mul(10)
                    .and_then(|v| v.checked_add(u64::from(b - b'0')))
                    .filter(|v| *v <= LIMIT)
                else {
                    break;
                };
                numerator = next;
                scale *= 10.0;
            }
            if numerator > 0 {
                // Go computes the fraction in float64 too, so the rounding
                // matches digit for digit.
                let part = (numerator as f64 * (unit_ns as f64 / scale)) as u64;
                value = value
                    .checked_add(part)
                    .filter(|v| *v <= LIMIT)
                    .ok_or_else(invalid)?;
            }
            total = total
                .checked_add(value)
                .filter(|v| *v <= LIMIT)
                .ok_or_else(invalid)?;
        }

        if negative {
            return Ok(Self(0i64.wrapping_sub_unsigned(total)));
        }
        i64::try_from(total).map(Self).map_err(|_| invalid())
    }
}

/// `value / 10^precision` as an integer part and a fraction with its
/// trailing zeros dropped, the dot included only when a fraction remains.
fn split_fraction(value: u64, precision: usize) -> (u64, String) {
    let divisor = 10u64.pow(precision as u32);
    let (whole, rem) = (value / divisor, value % divisor);
    if rem == 0 {
        return (whole, String::new());
    }
    let padded = format!("{rem:0precision$}");
    (whole, format!(".{}", padded.trim_end_matches('0')))
}

impl fmt::Display for GoDuration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let nanos = self.0.unsigned_abs();
        if nanos == 0 {
            return f.write_str("0s");
        }
        if self.0 < 0 {
            f.write_str("-")?;
        }
        if nanos < SECOND {
            let (precision, unit) = match nanos {
                n if n < MICROSECOND => (0, "ns"),
                n if n < MILLISECOND => (3, "\u{b5}s"),
                _ => (6, "ms"),
            };
            let (whole, fraction) = split_fraction(nanos, precision);
            return write!(f, "{whole}{fraction}{unit}");
        }
        let (seconds, fraction) = split_fraction(nanos, 9);
        let (hours, minutes, seconds) = (seconds / 3600, seconds / 60 % 60, seconds % 60);
        if hours > 0 {
            write!(f, "{hours}h{minutes}m")?;
        } else if minutes > 0 {
            write!(f, "{minutes}m")?;
        }
        write!(f, "{seconds}{fraction}s")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MS: i64 = 1_000_000;
    const S: i64 = 1_000_000_000;

    #[test]
    fn parses_what_go_parses() {
        let cases: &[(&str, i64)] = &[
            ("0", 0),
            ("+0", 0),
            ("-0", 0),
            ("0s", 0),
            ("5ns", 5),
            ("300ms", 300 * MS),
            ("50ms", 50 * MS),
            ("2s", 2 * S),
            ("60s", 60 * S),
            ("1.5s", 1500 * MS),
            ("1m30s", 90 * S),
            ("1h2m3s", 3723 * S),
            ("-1.5h", -5400 * S),
            ("+5m", 300 * S),
            (".5s", 500 * MS),
            ("1.s", S),
            ("1us", 1_000),
            ("1\u{b5}s", 1_000),
            ("1\u{3bc}s", 1_000),
            ("1.0000000001s", S),
            ("2562047h47m16.854775807s", i64::MAX),
            ("-2562047h47m16.854775808s", i64::MIN),
            ("0.3333333333333333333h", 20 * 60 * S),
            ("0.100000000000000000000h", 6 * 60 * S),
            ("0.830103483285477580700h", 2_988_372_539_827),
        ];
        for (input, want) in cases {
            assert_eq!(
                input.parse::<GoDuration>().map(GoDuration::nanos),
                Ok(*want),
                "{input}"
            );
        }
    }

    #[test]
    fn rejects_what_go_rejects() {
        let invalid = |input: &str| ParseDurationError::Invalid {
            input: input.to_owned(),
        };
        let missing = |input: &str| ParseDurationError::MissingUnit {
            input: input.to_owned(),
        };
        let unknown = |unit: &str, input: &str| ParseDurationError::UnknownUnit {
            unit: unit.to_owned(),
            input: input.to_owned(),
        };
        let cases = [
            ("", invalid("")),
            ("+", invalid("+")),
            ("-", invalid("-")),
            ("s", invalid("s")),
            (".", invalid(".")),
            (".s", invalid(".s")),
            ("-.s", invalid("-.s")),
            (" 1s", invalid(" 1s")),
            ("3", missing("3")),
            ("1s5", missing("1s5")),
            ("3000000h", invalid("3000000h")),
            (
                "2562047h47m16.854775808s",
                invalid("2562047h47m16.854775808s"),
            ),
            ("9223372036854775808ns", invalid("9223372036854775808ns")),
            ("99999999999999999999s", invalid("99999999999999999999s")),
            ("99999999999999999999d", invalid("99999999999999999999d")),
            ("1d", unknown("d", "1d")),
            ("1 s", unknown(" s", "1 s")),
            ("1S", unknown("S", "1S")),
            ("1s ", unknown("s ", "1s ")),
        ];
        for (input, want) in cases {
            assert_eq!(input.parse::<GoDuration>(), Err(want), "{input:?}");
        }
    }

    #[test]
    fn error_messages_are_gos() {
        let message = |input: &str| input.parse::<GoDuration>().map_err(|e| e.to_string());
        assert_eq!(
            message("abc"),
            Err("time: invalid duration \"abc\"".to_owned())
        );
        assert_eq!(
            message("5"),
            Err("time: missing unit in duration \"5\"".to_owned())
        );
        assert_eq!(
            message("5d"),
            Err("time: unknown unit \"d\" in duration \"5d\"".to_owned())
        );
        assert_eq!(
            message("1\u{b5}"),
            Err("time: unknown unit \"\\xc2\\xb5\" in duration \"1\\xc2\\xb5\"".to_owned())
        );
    }

    #[test]
    fn displays_as_go_does() {
        let cases: &[(i64, &str)] = &[
            (0, "0s"),
            (1, "1ns"),
            (1_100, "1.1\u{b5}s"),
            (2_200_000, "2.2ms"),
            (500 * MS, "500ms"),
            (50 * MS, "50ms"),
            (2 * S, "2s"),
            (1500 * MS, "1.5s"),
            (3_300_000_000, "3.3s"),
            (60 * S, "1m0s"),
            (90 * S, "1m30s"),
            (125 * S, "2m5s"),
            (4 * 60 * S + 5_001_000_000, "4m5.001s"),
            (3600 * S, "1h0m0s"),
            (5 * 3600 * S + 6 * 60 * S + 7_001_000_000, "5h6m7.001s"),
            (8 * 60 * S + 1, "8m0.000000001s"),
            (-1500 * MS, "-1.5s"),
            (-1, "-1ns"),
            (i64::MAX, "2562047h47m16.854775807s"),
            (i64::MIN, "-2562047h47m16.854775808s"),
        ];
        for (nanos, want) in cases {
            assert_eq!(GoDuration::from_nanos(*nanos).to_string(), *want);
        }
    }

    #[test]
    fn converts_to_and_from_std() {
        assert_eq!(
            GoDuration::from_nanos(300 * MS).to_std(),
            Duration::from_millis(300)
        );
        assert_eq!(GoDuration::from_nanos(-S).to_std(), Duration::ZERO);
        assert_eq!(
            GoDuration::from(Duration::from_millis(1500)).to_string(),
            "1.5s"
        );
        assert_eq!(GoDuration::from(Duration::MAX).nanos(), i64::MAX);
    }
}
