//! The KV hop's cost rule. It hops only when the history is long enough and
//! the copy is estimated to be cheaper than the prefill.

use std::fmt;

use ariadne_decide::Sticky;

use crate::goduration::GoDuration;

/// What one move did: the `result` label on `orch_hop_total` and the `X-Hop`
/// header value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Outcome {
    /// The history was shorter than `min_tokens`, so the destination
    /// recomputed it.
    BelowThreshold,
    /// The cost rule estimated the prefill to be faster than the copy.
    RecomputeCheaper,
    /// The source held the KV and the destination was told to pull it.
    Hopped,
    /// The hop was attempted and did not complete, so the destination
    /// recomputed the history.
    Failed,
    /// `max_inflight` hops were already running, so this one was skipped.
    Busy,
}

impl Outcome {
    /// Every outcome, for pre-registering the metric's labels.
    pub const ALL: [Self; 5] = [
        Self::BelowThreshold,
        Self::RecomputeCheaper,
        Self::Hopped,
        Self::Failed,
        Self::Busy,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BelowThreshold => "below_threshold",
            Self::RecomputeCheaper => "recompute_cheaper",
            Self::Hopped => "hopped",
            Self::Failed => "failed",
            Self::Busy => "busy",
        }
    }
}

/// Why the cost rule chose to recompute.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Skip {
    BelowThreshold,
    RecomputeCheaper,
}

impl Skip {
    pub const fn outcome(self) -> Outcome {
        match self {
            Self::BelowThreshold => Outcome::BelowThreshold,
            Self::RecomputeCheaper => Outcome::RecomputeCheaper,
        }
    }
}

/// The cost rule's inputs. The rates describe the deployment and should come
/// from a measurement on the real node. Until then they are estimates, and
/// the rule is only as good as they are.
///
/// The integer fields are signed, as the Go flags are, so `validate` can name
/// a negative value the way the Go binary did at startup.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Config {
    /// The shortest history worth hopping. Two extra round trips are not
    /// worth saving a short prefill.
    pub min_tokens: i64,
    /// The prefix every worker already holds from warm-up. Neither path pays
    /// for it, so it is left out of both costs.
    pub shared_prefix_tokens: i64,
    pub kv_bytes_per_token: i64,
    /// Copy bandwidth between two workers.
    pub transfer_bytes_per_s: f64,
    /// A worker's uncached prefill rate.
    pub prefill_tokens_per_s: f64,
    /// The fixed cost of a hop: the hold request and the handshake before
    /// the pull.
    pub overhead: GoDuration,
    /// Bounds the source request, so a busy source means a recompute rather
    /// than a stalled request.
    pub timeout: GoDuration,
    /// Bounds the destination request once it carries a hop, because a
    /// destination pulling from an engine that is gone hangs until the
    /// gateway's upstream timeout. Past it the gateway resends without the
    /// hop.
    pub receive_timeout: GoDuration,
    /// Caps hops running at once. Each sends the source a request outside
    /// the gateway's admission and may pin its blocks until pulled, so a
    /// burst of moves must not turn into a burst of hops.
    pub max_inflight: i64,
}

/// Sized for Qwen3-8B-AWQ on a 20 GiB A100 slice. The KV size is from the
/// model's config. The prefill rate is serving/fit.py's estimate for that
/// slice, 12,000 tokens in 3.15 s. The bandwidth is a conservative guess for
/// TCP between two pods on one host. Measure both rates before trusting the
/// rule.
pub const DEFAULT_CONFIG: Config = Config {
    min_tokens: 8192,
    shared_prefix_tokens: 3899,
    kv_bytes_per_token: 147_456,
    transfer_bytes_per_s: 2e9,
    prefill_tokens_per_s: 3800.0,
    overhead: GoDuration::from_nanos(50_000_000),
    timeout: GoDuration::from_nanos(10_000_000_000),
    receive_timeout: GoDuration::from_nanos(30_000_000_000),
    max_inflight: 4,
};

impl Default for Config {
    fn default() -> Self {
        DEFAULT_CONFIG
    }
}

/// One field that cannot drive the rule, holding the bad value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Invalid {
    MinTokens(i64),
    SharedPrefixTokens(i64),
    KvBytesPerToken(i64),
    TransferBytesPerS(f64),
    PrefillTokensPerS(f64),
    Overhead(GoDuration),
    Timeout(GoDuration),
    ReceiveTimeout(GoDuration),
    MaxInflight(i64),
}

impl fmt::Display for Invalid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MinTokens(v) => write!(f, "min tokens {v} is negative"),
            Self::SharedPrefixTokens(v) => write!(f, "shared prefix {v} is negative"),
            Self::KvBytesPerToken(v) => write!(f, "KV bytes per token {v} must be positive"),
            Self::TransferBytesPerS(v) => {
                write!(f, "transfer rate {} B/s must be positive", GoFloat(*v))
            }
            Self::PrefillTokensPerS(v) => {
                write!(f, "prefill rate {} tokens/s must be positive", GoFloat(*v))
            }
            Self::Overhead(v) => write!(f, "overhead {v} is negative"),
            Self::Timeout(v) => write!(f, "timeout {v} must be positive"),
            Self::ReceiveTimeout(v) => write!(f, "receive timeout {v} must be positive"),
            Self::MaxInflight(v) => write!(f, "max in-flight hops {v} must be at least 1"),
        }
    }
}

/// Every field of a `Config` that cannot drive the rule, in field order.
/// Displays one per line, as Go's `errors.Join` does.
#[derive(Clone, Debug, PartialEq)]
pub struct ConfigError(Vec<Invalid>);

impl ConfigError {
    pub fn fields(&self) -> &[Invalid] {
        &self.0
    }
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, invalid) in self.0.iter().enumerate() {
            if i > 0 {
                f.write_str("\n")?;
            }
            write!(f, "{invalid}")?;
        }
        Ok(())
    }
}

impl std::error::Error for ConfigError {}

/// The cost rule's estimate for one move. `tokens` is the history beyond the
/// shared prefix, the part one of the two paths has to pay for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Estimate {
    pub tokens: u64,
    pub hop_cost: GoDuration,
    pub recompute_cost: GoDuration,
}

/// The cost rule's answer for one move.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Plan {
    Hop(Estimate),
    Skip(Skip, Estimate),
}

impl Plan {
    pub const fn estimate(&self) -> &Estimate {
        match self {
            Self::Hop(estimate) | Self::Skip(_, estimate) => estimate,
        }
    }
}

impl Config {
    /// Reports every field that cannot drive the rule.
    pub fn validate(&self) -> Result<(), ConfigError> {
        let checks = [
            (self.min_tokens < 0, Invalid::MinTokens(self.min_tokens)),
            (
                self.shared_prefix_tokens < 0,
                Invalid::SharedPrefixTokens(self.shared_prefix_tokens),
            ),
            (
                self.kv_bytes_per_token <= 0,
                Invalid::KvBytesPerToken(self.kv_bytes_per_token),
            ),
            (
                !positive(self.transfer_bytes_per_s),
                Invalid::TransferBytesPerS(self.transfer_bytes_per_s),
            ),
            (
                !positive(self.prefill_tokens_per_s),
                Invalid::PrefillTokensPerS(self.prefill_tokens_per_s),
            ),
            (self.overhead.nanos() < 0, Invalid::Overhead(self.overhead)),
            (self.timeout.nanos() <= 0, Invalid::Timeout(self.timeout)),
            (
                self.receive_timeout.nanos() <= 0,
                Invalid::ReceiveTimeout(self.receive_timeout),
            ),
            (
                self.max_inflight < 1,
                Invalid::MaxInflight(self.max_inflight),
            ),
        ];
        let invalid: Vec<Invalid> = checks
            .into_iter()
            .filter_map(|(bad, invalid)| bad.then_some(invalid))
            .collect();
        if invalid.is_empty() {
            Ok(())
        } else {
            Err(ConfigError(invalid))
        }
    }

    /// Applies the cost rule to a run whose last step's prompt was `history`
    /// tokens long.
    ///
    /// A rate that is not positive has no estimate, and its cost reads as the
    /// longest duration. Such a config never hops, though `validate` rejects
    /// it first in the gateway.
    pub fn decide(&self, history: u64) -> Plan {
        let beyond = i128::from(history) - i128::from(self.shared_prefix_tokens);
        let tokens = u64::try_from(beyond.max(0)).unwrap_or(u64::MAX);
        let bytes = tokens as f64 * self.kv_bytes_per_token as f64;
        let copy = per_rate(bytes, self.transfer_bytes_per_s);
        let recompute = per_rate(tokens as f64, self.prefill_tokens_per_s);
        let hop_cost = copy.map(|copy| self.overhead.nanos().saturating_add(copy.nanos()));
        let estimate = Estimate {
            tokens,
            hop_cost: GoDuration::from_nanos(hop_cost.unwrap_or(i64::MAX)),
            recompute_cost: recompute.unwrap_or(GoDuration::from_nanos(i64::MAX)),
        };
        if i128::from(history) < i128::from(self.min_tokens) {
            return Plan::Skip(Skip::BelowThreshold, estimate);
        }
        match (hop_cost, recompute) {
            (Some(hop), Some(recompute)) if hop < recompute.nanos() => Plan::Hop(estimate),
            _ => Plan::Skip(Skip::RecomputeCheaper, estimate),
        }
    }
}

/// Reports whether a placement moved a run off a worker that still holds its
/// history. A run with no binding has no history anywhere, a kept binding
/// needs no copy, and a worker that restarted lost its cache.
pub const fn moved(sticky: Sticky) -> bool {
    match sticky {
        Sticky::BrokenLoad | Sticky::BrokenShed => true,
        Sticky::None | Sticky::New | Sticky::Hit | Sticky::BrokenGone => false,
    }
}

/// False for NaN, which Go's `<= 0` check lets through.
fn positive(rate: f64) -> bool {
    rate > 0.0
}

fn per_rate(amount: f64, rate: f64) -> Option<GoDuration> {
    positive(rate).then(|| seconds(amount / rate))
}

/// Rounds to the nearest nanosecond, so a cost that is exact in decimal does
/// not land a nanosecond short through floating point. Saturates where Go's
/// conversion is implementation-defined.
fn seconds(s: f64) -> GoDuration {
    GoDuration::from_nanos((s * 1e9).round() as i64)
}

/// A float as Go's `%g` prints it: the shortest digits that round-trip, in
/// exponent form when the exponent is below -4 or at least 6.
struct GoFloat(f64);

impl fmt::Display for GoFloat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let v = self.0;
        if v.is_nan() {
            return f.write_str("NaN");
        }
        if v.is_infinite() {
            return f.write_str(if v > 0.0 { "+Inf" } else { "-Inf" });
        }
        let scientific = format!("{v:e}");
        let Some((mantissa, exp)) = scientific.split_once('e') else {
            return write!(f, "{v}");
        };
        let exp: i32 = exp.parse().unwrap_or(0);
        if v == 0.0 || (-4..6).contains(&exp) {
            return write!(f, "{v}");
        }
        let sign = if exp < 0 { '-' } else { '+' };
        write!(f, "{mantissa}e{sign}{:02}", exp.unsigned_abs())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MS: i64 = 1_000_000;

    fn ms(n: i64) -> GoDuration {
        GoDuration::from_nanos(n * MS)
    }

    /// Round numbers: 1 MB per token over 1 GB/s is 1 ms per token to copy,
    /// and 1,000 tokens/s is 1 ms per token to recompute, so the overhead
    /// alone decides a tie.
    const RULE: Config = Config {
        min_tokens: 1000,
        shared_prefix_tokens: 200,
        kv_bytes_per_token: 1_000_000,
        transfer_bytes_per_s: 1e9,
        prefill_tokens_per_s: 1000.0,
        overhead: GoDuration::from_nanos(10 * MS),
        timeout: GoDuration::from_nanos(1000 * MS),
        receive_timeout: GoDuration::from_nanos(1000 * MS),
        max_inflight: 1,
    };

    fn estimate(tokens: u64, hop_ms: i64, recompute_ms: i64) -> Estimate {
        Estimate {
            tokens,
            hop_cost: ms(hop_ms),
            recompute_cost: ms(recompute_ms),
        }
    }

    #[test]
    fn decide() {
        let cases: [(&str, Config, u64, Plan); 4] = [
            (
                "history below the threshold recomputes",
                RULE,
                999,
                Plan::Skip(Skip::BelowThreshold, estimate(799, 809, 799)),
            ),
            (
                "equal per-token costs lose to the overhead",
                RULE,
                1200,
                Plan::Skip(Skip::RecomputeCheaper, estimate(1000, 1010, 1000)),
            ),
            (
                "a faster link hops",
                Config {
                    transfer_bytes_per_s: 2e9,
                    ..RULE
                },
                1200,
                Plan::Hop(estimate(1000, 510, 1000)),
            ),
            (
                "a history inside the shared prefix costs nothing either way",
                Config {
                    min_tokens: 0,
                    ..RULE
                },
                150,
                Plan::Skip(Skip::RecomputeCheaper, estimate(0, 10, 0)),
            ),
        ];
        for (name, cfg, history, want) in cases {
            assert_eq!(cfg.decide(history), want, "{name}");
        }
    }

    /// Plans the Go `Config.Decide` returned for the same inputs, run with Go
    /// 1.27 against internal/hop/policy.go.
    #[test]
    fn decide_matches_go() {
        let cases: [(Config, u64, Plan); 4] = [
            (
                DEFAULT_CONFIG,
                16000,
                Plan::Hop(Estimate {
                    tokens: 12101,
                    hop_cost: GoDuration::from_nanos(942_182_528),
                    recompute_cost: GoDuration::from_nanos(3_184_473_684),
                }),
            ),
            (
                DEFAULT_CONFIG,
                8192,
                Plan::Hop(Estimate {
                    tokens: 4293,
                    hop_cost: GoDuration::from_nanos(366_514_304),
                    recompute_cost: GoDuration::from_nanos(1_129_736_842),
                }),
            ),
            (
                DEFAULT_CONFIG,
                8191,
                Plan::Skip(
                    Skip::BelowThreshold,
                    Estimate {
                        tokens: 4292,
                        hop_cost: GoDuration::from_nanos(366_440_576),
                        recompute_cost: GoDuration::from_nanos(1_129_473_684),
                    },
                ),
            ),
            (
                Config {
                    prefill_tokens_per_s: 2e9 / 147_456.0,
                    ..DEFAULT_CONFIG
                },
                100_000,
                Plan::Skip(
                    Skip::RecomputeCheaper,
                    Estimate {
                        tokens: 96101,
                        hop_cost: GoDuration::from_nanos(7_135_334_528),
                        recompute_cost: GoDuration::from_nanos(7_085_334_528),
                    },
                ),
            ),
        ];
        for (cfg, history, want) in cases {
            assert_eq!(cfg.decide(history), want, "decide({history})");
        }
    }

    #[test]
    fn a_rate_that_is_not_positive_never_hops() {
        for rate in [0.0, -1.0, f64::NAN] {
            for cfg in [
                Config {
                    transfer_bytes_per_s: rate,
                    ..RULE
                },
                Config {
                    prefill_tokens_per_s: rate,
                    ..RULE
                },
            ] {
                let plan = cfg.decide(1_000_000);
                assert!(
                    matches!(plan, Plan::Skip(Skip::RecomputeCheaper, _)),
                    "decide with rate {rate} = {plan:?}"
                );
            }
        }
    }

    #[test]
    fn decide_saturates_instead_of_overflowing() {
        let plan = DEFAULT_CONFIG.decide(u64::MAX);
        assert_eq!(
            plan,
            Plan::Skip(
                Skip::RecomputeCheaper,
                Estimate {
                    tokens: u64::MAX - 3899,
                    hop_cost: GoDuration::from_nanos(i64::MAX),
                    recompute_cost: GoDuration::from_nanos(i64::MAX),
                }
            )
        );
    }

    #[test]
    fn default_config_hops_a_long_history_and_skips_a_short_one() {
        assert_eq!(DEFAULT_CONFIG.validate(), Ok(()));
        assert!(matches!(
            DEFAULT_CONFIG.decide(4000),
            Plan::Skip(Skip::BelowThreshold, _)
        ));
        assert!(matches!(DEFAULT_CONFIG.decide(16000), Plan::Hop(_)));
    }

    fn zero() -> Config {
        Config {
            min_tokens: 0,
            shared_prefix_tokens: 0,
            kv_bytes_per_token: 0,
            transfer_bytes_per_s: 0.0,
            prefill_tokens_per_s: 0.0,
            overhead: GoDuration::from_nanos(0),
            timeout: GoDuration::from_nanos(0),
            receive_timeout: GoDuration::from_nanos(0),
            max_inflight: 0,
        }
    }

    #[test]
    fn validate_names_every_bad_field() {
        let err = Config {
            min_tokens: -1,
            shared_prefix_tokens: -1,
            overhead: GoDuration::from_nanos(-1),
            ..zero()
        }
        .validate()
        .expect_err("validate passed a zero config");
        assert_eq!(
            err.to_string(),
            "min tokens -1 is negative\n\
             shared prefix -1 is negative\n\
             KV bytes per token 0 must be positive\n\
             transfer rate 0 B/s must be positive\n\
             prefill rate 0 tokens/s must be positive\n\
             overhead -1ns is negative\n\
             timeout 0s must be positive\n\
             receive timeout 0s must be positive\n\
             max in-flight hops 0 must be at least 1"
        );
    }

    #[test]
    fn validate_prints_rates_as_go_does() {
        let err = Config {
            transfer_bytes_per_s: -2.5e9,
            prefill_tokens_per_s: -0.00001,
            ..DEFAULT_CONFIG
        }
        .validate()
        .expect_err("negative rates passed");
        assert_eq!(
            err.to_string(),
            "transfer rate -2.5e+09 B/s must be positive\n\
             prefill rate -1e-05 tokens/s must be positive"
        );
        let err = Config {
            transfer_bytes_per_s: -123_456.0,
            prefill_tokens_per_s: -0.0001,
            ..DEFAULT_CONFIG
        }
        .validate()
        .expect_err("negative rates passed");
        assert_eq!(
            err.to_string(),
            "transfer rate -123456 B/s must be positive\n\
             prefill rate -0.0001 tokens/s must be positive"
        );
    }

    #[test]
    fn moved_only_when_the_old_worker_still_holds_the_history() {
        let cases = [
            (Sticky::None, false),
            (Sticky::New, false),
            (Sticky::Hit, false),
            (Sticky::BrokenLoad, true),
            (Sticky::BrokenShed, true),
            (Sticky::BrokenGone, false),
        ];
        for (sticky, want) in cases {
            assert_eq!(moved(sticky), want, "moved({})", sticky.as_str());
        }
    }

    #[test]
    fn outcome_labels() {
        let labels: Vec<&str> = Outcome::ALL.iter().map(|o| o.as_str()).collect();
        assert_eq!(
            labels,
            [
                "below_threshold",
                "recompute_cheaper",
                "hopped",
                "failed",
                "busy"
            ]
        );
    }
}
