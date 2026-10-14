//! Reads a vLLM worker's Prometheus `/metrics` body into the few numbers the
//! Gate needs. The parser follows `prometheus/common` v0.72's text parser,
//! which the Go gateway used, so the same bodies are accepted and rejected.

use std::collections::HashMap;
use std::fmt;

/// Caps one /metrics body, so a misbehaving worker cannot exhaust the
/// gateway's memory through the scrape path.
pub const MAX_METRICS_BYTES: usize = 8 << 20;

/// Caps the labels on one sample line. vLLM sends about ten, and the
/// duplicate-label check is quadratic in this number.
pub const MAX_LABELS_PER_SAMPLE: usize = 64;

/// The families `parse_metrics` reads. Only these keep their points and
/// buckets; every other family is validated and then forgotten.
const READ: [&str; 5] = [KV_USAGE, KV_USAGE_LEGACY, WAITING, CACHE_CONFIG, TTFT];
const KV_USAGE: &str = "vllm:kv_cache_usage_perc";
const KV_USAGE_LEGACY: &str = "vllm:gpu_cache_usage_perc";
const WAITING: &str = "vllm:num_requests_waiting";
const CACHE_CONFIG: &str = "vllm:cache_config_info";
const TTFT: &str = "vllm:time_to_first_token_seconds";

/// One cumulative histogram bucket: the observations at or under `le`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bucket {
    pub le: f64,
    pub count: f64,
}

/// One Prometheus histogram's buckets in ascending bound order, +Inf last.
/// Two histograms with the same bounds can be subtracted bucket by bucket.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Histogram {
    pub buckets: Vec<Bucket>,
}

/// One worker's /metrics body reduced to what the Gate needs, with every
/// number already checked, so the admission decision can trust it.
/// `kv_usage` is a ratio in 0..=1, and `None` when the body carried no KV
/// usage metric, because without it the sample says nothing about the
/// worker's load. `kv_pool_tokens` is the KV pool's capacity in tokens, or 0
/// when vLLM did not export its cache config. `waiting` is finite and not
/// negative, and every TTFT bucket has a real bound and a finite count.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Sample {
    pub kv_usage: Option<f64>,
    pub kv_pool_tokens: u64,
    pub waiting: f64,
    pub ttft: Histogram,
}

/// A body that is not Prometheus text exposition, and the line that broke.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseError {
    pub line: usize,
    pub msg: String,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "parse metrics: text format parsing error in line {}: {}",
            self.line, self.msg
        )
    }
}

impl std::error::Error for ParseError {}

/// Reads one Prometheus text exposition body into a Sample. It reads at most
/// `MAX_METRICS_BYTES`. It sums gauges across label sets, and histogram
/// buckets with the same bound. KV usage is normalised per label set,
/// because older vLLM releases exported it as a percentage.
///
/// The numbers come from the worker and are not trusted. A KV usage that is
/// not finite reads as a full cache, so garbage fails closed, and a finite
/// one is clamped to 0..=1. A waiting count that is not finite or negative
/// reads as 0. A pool label that is negative or not a number counts as 0.
pub fn parse_metrics(body: &[u8]) -> Result<Sample, ParseError> {
    let families = Families::parse(&body[..body.len().min(MAX_METRICS_BYTES)])?;

    let mut sample = Sample::default();
    if let Some(family) = families.first(&[KV_USAGE, KV_USAGE_LEGACY]) {
        let usage = family
            .points
            .iter()
            .fold(0.0, |sum, p| sum + ratio(p.value));
        sample.kv_usage = Some(if usage.is_finite() {
            usage.clamp(0.0, 1.0)
        } else {
            1.0
        });
    }
    if let Some(family) = families.first(&[WAITING]) {
        let waiting = family.points.iter().fold(0.0, |sum, p| sum + p.value);
        sample.waiting = if waiting.is_finite() && waiting > 0.0 {
            waiting
        } else {
            0.0
        };
    }
    if let Some(family) = families.first(&[CACHE_CONFIG]) {
        sample.kv_pool_tokens = family.points.iter().fold(0u64, |sum, p| {
            sum.saturating_add(
                p.label_u64("num_gpu_blocks")
                    .saturating_mul(p.label_u64("block_size")),
            )
        });
    }
    if let Some(family) = families.first(&[TTFT]) {
        sample.ttft = histogram(&family.buckets);
    }
    Ok(sample)
}

/// Turns a usage that may be a percentage into a 0 to 1 ratio.
fn ratio(v: f64) -> f64 {
    if v > 1.0 { v / 100.0 } else { v }
}

/// Sums a family's buckets across label sets by bound.
fn histogram(buckets: &[Bucket]) -> Histogram {
    let mut sorted = buckets.to_vec();
    sorted.sort_by(|a, b| a.le.total_cmp(&b.le));
    let mut merged: Vec<Bucket> = Vec::with_capacity(sorted.len());
    for bucket in sorted {
        match merged.last_mut() {
            Some(last) if last.le == bucket.le => last.count += bucket.count,
            _ => merged.push(bucket),
        }
    }
    Histogram { buckets: merged }
}

impl Histogram {
    /// Returns the q quantile of the observations in `self` that are not in
    /// `prev`, interpolating linearly inside the chosen bucket as
    /// Prometheus's histogram_quantile does. `prev` may be empty. A `prev`
    /// whose bounds differ is ignored, because the two cannot be subtracted.
    /// The +Inf bucket returns the highest finite bound. An empty delta
    /// returns 0.
    pub fn quantile(&self, prev: &Histogram, q: f64) -> f64 {
        let n = self.buckets.len();
        let same_bounds = prev.buckets.len() == n
            && self
                .buckets
                .iter()
                .zip(&prev.buckets)
                .all(|(a, b)| a.le == b.le);
        let delta: Vec<f64> = if same_bounds {
            self.buckets
                .iter()
                .zip(&prev.buckets)
                .map(|(a, b)| a.count - b.count)
                .collect()
        } else {
            self.buckets.iter().map(|b| b.count).collect()
        };
        let Some(&total) = delta.last() else {
            return 0.0;
        };
        if total <= 0.0 {
            return 0.0;
        }
        let rank = q.clamp(0.0, 1.0) * total;

        // Go's sort.Search, step for step. A worker restart can leave the
        // delta non-monotonic, and another bisection could then land on a
        // different bucket than the Go gateway did.
        let (mut b, mut hi) = (0, n);
        while b < hi {
            let mid = b + (hi - b) / 2;
            if delta[mid] >= rank {
                hi = mid;
            } else {
                b = mid + 1;
            }
        }

        if b >= n - 1 {
            return if n < 2 { 0.0 } else { self.buckets[n - 2].le };
        }
        let (start, before) = if b > 0 {
            (self.buckets[b - 1].le, delta[b - 1])
        } else {
            (0.0, 0.0)
        };
        let (end, count) = (self.buckets[b].le, delta[b]);
        if count <= before {
            return end;
        }
        start + (end - start) * (rank - before) / (count - before)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Counter,
    Gauge,
    Untyped,
    Summary,
    Histogram,
}

/// One gauge, counter or untyped sample line.
struct Point {
    labels: Vec<(String, String)>,
    value: f64,
}

impl Point {
    /// Reads a count label, or 0 when it is missing, negative or not a
    /// number.
    fn label_u64(&self, name: &str) -> u64 {
        self.labels
            .iter()
            .find(|(n, _)| n == name)
            .and_then(|(_, v)| v.parse().ok())
            .unwrap_or(0)
    }
}

/// One metric family. `points` holds the lines of a scalar family and
/// `buckets` the bucket lines of a histogram, both only when `kept`. A
/// summary or histogram keeps no points, so it reads as zero where a scalar
/// was expected.
#[derive(Default)]
struct Family {
    kept: bool,
    kind: Option<Kind>,
    has_help: bool,
    sampled: bool,
    points: Vec<Point>,
    buckets: Vec<Bucket>,
}

/// Which series of a summary or histogram a line's metric name refers to.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Series {
    Plain,
    Count,
    Sum,
}

#[derive(Default)]
struct Families {
    by_name: HashMap<String, usize>,
    all: Vec<Family>,
}

impl Families {
    fn parse(body: &[u8]) -> Result<Self, ParseError> {
        let mut families = Self::default();
        let mut lines = body.split(|b| *b == b'\n');
        // The piece after the last newline is not a line. The Go parser
        // accepts only blanks there and calls anything else a truncated body.
        let tail = lines.next_back().unwrap_or_default();
        let mut number = 0;
        for line in lines {
            number += 1;
            families
                .line(line)
                .map_err(|msg| ParseError { line: number, msg })?;
        }
        if tail.iter().any(|b| !is_blank(*b)) {
            return Err(ParseError {
                line: number + 1,
                msg: "unexpected end of input stream".to_owned(),
            });
        }
        Ok(families)
    }

    /// Returns the first family under any of `names` that has a sample.
    /// vLLM has renamed metrics across releases, and reading zero from a
    /// renamed metric would make the Gate confidently wrong.
    fn first(&self, names: &[&str]) -> Option<&Family> {
        names
            .iter()
            .filter_map(|name| self.by_name.get(*name))
            .map(|i| &self.all[*i])
            .find(|family| family.sampled)
    }

    /// Finds the family a metric name belongs to, creating it when new. A
    /// `_count`, `_sum` or `_bucket` name belongs to the summary or histogram
    /// declared under its stem, and is its own family otherwise.
    fn resolve(&mut self, name: &str) -> (usize, Series) {
        if let Some(i) = self.by_name.get(name) {
            return (*i, Series::Plain);
        }
        let (count, sum, bucket) = (
            name.strip_suffix("_count").filter(|stem| !stem.is_empty()),
            name.strip_suffix("_sum").filter(|stem| !stem.is_empty()),
            name.strip_suffix("_bucket").filter(|stem| !stem.is_empty()),
        );
        let series = match (count, sum) {
            (Some(_), _) => Series::Count,
            (None, Some(_)) => Series::Sum,
            (None, None) => Series::Plain,
        };
        let declared = |stem: Option<&str>, kind: Kind| {
            let i = *self.by_name.get(stem?)?;
            (self.all[i].kind == Some(kind)).then_some(i)
        };
        if let Some(i) = declared(count.or(sum), Kind::Summary) {
            return (i, series);
        }
        if let Some(i) = declared(count.or(sum).or(bucket), Kind::Histogram) {
            return (i, series);
        }
        self.all.push(Family {
            kept: READ.contains(&name),
            ..Family::default()
        });
        self.by_name.insert(name.to_owned(), self.all.len() - 1);
        (self.all.len() - 1, Series::Plain)
    }

    fn line(&mut self, line: &[u8]) -> Result<(), String> {
        let mut c = Cursor { line, pos: 0 };
        c.skip_blanks();
        match c.cur() {
            b'\n' => Ok(()),
            b'#' => self.comment(c),
            _ => self.sample(c),
        }
    }

    fn comment(&mut self, mut c: Cursor<'_>) -> Result<(), String> {
        c.advance_past_blanks();
        let keyword = c.token();
        if c.cur() == b'\n' || (keyword != b"HELP" && keyword != b"TYPE") {
            return Ok(());
        }
        c.advance_past_blanks();
        let name = c.name(is_metric_name_byte)?;
        if c.cur() == b'\n' {
            return Ok(());
        }
        if !is_blank(c.cur()) {
            return Err("invalid metric name in comment".to_owned());
        }
        let (i, _) = self.resolve(name);
        c.advance_past_blanks();
        if c.cur() == b'\n' {
            return Ok(());
        }
        let family = &mut self.all[i];
        if keyword == b"HELP" {
            if family.has_help {
                return Err(format!("second HELP line for metric name {name:?}"));
            }
            c.check_help_escapes()?;
            family.has_help = true;
            return Ok(());
        }
        if family.kind.is_some() {
            return Err(format!(
                "second TYPE line for metric name {name:?}, or TYPE reported after samples"
            ));
        }
        let text = c.rest();
        let kind = [
            ("counter", Kind::Counter),
            ("gauge", Kind::Gauge),
            ("untyped", Kind::Untyped),
            ("summary", Kind::Summary),
            ("histogram", Kind::Histogram),
            ("gauge_histogram", Kind::Histogram),
            ("gaugehistogram", Kind::Histogram),
        ]
        .into_iter()
        .find(|(word, _)| text.eq_ignore_ascii_case(word.as_bytes()));
        let Some((_, kind)) = kind else {
            return Err(format!(
                "unknown metric type {:?}",
                String::from_utf8_lossy(text)
            ));
        };
        family.kind = Some(kind);
        Ok(())
    }

    fn sample(&mut self, mut c: Cursor<'_>) -> Result<(), String> {
        let name = c.name(is_metric_name_byte)?;
        if name.is_empty() {
            return Err("invalid metric name".to_owned());
        }
        let (i, series) = self.resolve(name);
        let kind = *self.all[i].kind.get_or_insert(Kind::Untyped);
        c.skip_blanks();

        let mut labels: Vec<(&str, String)> = Vec::new();
        let mut le = None;
        if c.cur() == b'{' {
            let mut seen = 0;
            loop {
                c.advance_past_blanks();
                if c.cur() == b'}' {
                    break;
                }
                seen += 1;
                if seen > MAX_LABELS_PER_SAMPLE {
                    return Err(format!(
                        "more than {MAX_LABELS_PER_SAMPLE} labels for metric {name:?}"
                    ));
                }
                let label = c.name(is_label_name_byte)?;
                if label.is_empty() {
                    return Err(format!("invalid label name for metric {name:?}"));
                }
                c.skip_blanks();
                if c.cur() != b'=' {
                    return Err(format!(
                        "expected '=' after label name, found {:?}",
                        char::from(c.cur())
                    ));
                }
                if label == "__name__" {
                    return Err("label name \"__name__\" is reserved".to_owned());
                }
                // A summary's quantile and a histogram's le name the series
                // and are not labels of the metric.
                let series_label = matches!(
                    (kind, label),
                    (Kind::Summary, "quantile") | (Kind::Histogram, "le")
                );
                if !series_label && labels.iter().any(|(n, _)| *n == label) {
                    return Err(format!("duplicate label names for metric {name:?}"));
                }
                c.advance_past_blanks();
                if c.cur() != b'"' {
                    return Err(format!(
                        "expected '\"' at start of label value, found {:?}",
                        char::from(c.cur())
                    ));
                }
                let value = c.label_value()?;
                if series_label {
                    let bound = parse_float(&value).ok_or_else(|| {
                        format!("expected float as value for '{label}' label, got {value:?}")
                    })?;
                    if kind == Kind::Histogram {
                        le = Some(bound);
                    }
                }
                c.advance_past_blanks();
                let last = match c.cur() {
                    b',' => false,
                    b'}' => true,
                    _ => return Err(format!("unexpected end of label value {value:?}")),
                };
                if !series_label {
                    labels.push((label, value));
                }
                if last {
                    break;
                }
            }
            c.advance_past_blanks();
        }

        let token = c.token();
        let value = std::str::from_utf8(token)
            .ok()
            .and_then(parse_float)
            .ok_or_else(|| {
                format!(
                    "expected float as value, got {:?}",
                    String::from_utf8_lossy(token)
                )
            })?;

        let family = &mut self.all[i];
        family.sampled = true;
        match (kind, series) {
            (Kind::Counter | Kind::Gauge | Kind::Untyped, _) => {
                if family.kept {
                    let labels = labels.into_iter().map(|(n, v)| (n.to_owned(), v));
                    family.points.push(Point {
                        labels: labels.collect(),
                        value,
                    });
                }
            }
            (Kind::Summary, _) | (Kind::Histogram, Series::Sum) => {}
            (Kind::Histogram, Series::Count) => {
                if value < 0.0 {
                    return Err(format!("negative count for histogram {name:?}"));
                }
            }
            (Kind::Histogram, Series::Plain) => {
                // A NaN bound is no bucket, as in the Go parser.
                if let Some(le) = le.filter(|le| !le.is_nan()) {
                    if value < 0.0 {
                        return Err(format!("negative bucket population for histogram {name:?}"));
                    }
                    // An infinite or NaN count would poison every quantile.
                    if family.kept && value.is_finite() {
                        family.buckets.push(Bucket { le, count: value });
                    }
                }
            }
        }

        if c.cur() == b'\n' {
            return Ok(());
        }
        c.advance_past_blanks();
        let token = c.token();
        if std::str::from_utf8(token)
            .ok()
            .and_then(|t| t.parse::<i64>().ok())
            .is_none()
        {
            return Err(format!(
                "expected integer as timestamp, got {:?}",
                String::from_utf8_lossy(token)
            ));
        }
        let rest = c.rest();
        if !rest.is_empty() {
            return Err(format!(
                "spurious string after timestamp: {:?}",
                String::from_utf8_lossy(rest)
            ));
        }
        Ok(())
    }
}

/// A position in one line. Past the end it reads as the newline that ended
/// the line, which is what the Go parser's states branch on.
struct Cursor<'a> {
    line: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn cur(&self) -> u8 {
        self.line.get(self.pos).copied().unwrap_or(b'\n')
    }

    fn skip_blanks(&mut self) {
        while is_blank(self.cur()) {
            self.pos += 1;
        }
    }

    /// Steps over the current byte, then over any blanks.
    fn advance_past_blanks(&mut self) {
        self.pos += 1;
        self.skip_blanks();
    }

    fn take_while(&mut self, keep: impl Fn(u8) -> bool) -> &'a [u8] {
        let rest = self.line.get(self.pos..).unwrap_or_default();
        let len = rest.iter().take_while(|b| keep(**b)).count();
        self.pos += len;
        &rest[..len]
    }

    /// The bytes up to the next blank or the end of the line.
    fn token(&mut self) -> &'a [u8] {
        self.take_while(|b| !is_blank(b))
    }

    fn rest(&mut self) -> &'a [u8] {
        self.take_while(|_| true)
    }

    /// A metric or label name, which may be empty. Quoted UTF-8 names, which
    /// vLLM does not emit, are refused instead of parsed.
    fn name(&mut self, is_name_byte: fn(u8, bool) -> bool) -> Result<&'a str, String> {
        if self.cur() == b'"' {
            return Err("quoted metric and label names are not supported".to_owned());
        }
        if !is_name_byte(self.cur(), true) {
            return Ok("");
        }
        let name = self.take_while(|b| is_name_byte(b, false));
        if self.cur() == b'"' {
            return Err("quoted metric and label names are not supported".to_owned());
        }
        // Name bytes are ASCII, so this cannot fail.
        std::str::from_utf8(name).map_err(|e| e.to_string())
    }

    /// The label value whose opening quote is the current byte. Leaves the
    /// closing quote as the current byte.
    fn label_value(&mut self) -> Result<String, String> {
        let mut value = Vec::new();
        loop {
            self.pos += 1;
            match self.cur() {
                b'"' => break,
                b'\n' => {
                    return Err(format!(
                        "label value {:?} contains unescaped new-line",
                        String::from_utf8_lossy(&value)
                    ));
                }
                b'\\' => {
                    self.pos += 1;
                    match self.cur() {
                        b @ (b'"' | b'\\') => value.push(b),
                        b'n' => value.push(b'\n'),
                        b => return Err(invalid_escape(b)),
                    }
                }
                b => value.push(b),
            }
        }
        String::from_utf8(value).map_err(|e| {
            format!(
                "invalid label value {:?}",
                String::from_utf8_lossy(e.as_bytes())
            )
        })
    }

    /// Checks the escapes of a HELP text, the only part of it that can fail.
    fn check_help_escapes(&mut self) -> Result<(), String> {
        let mut text = self.rest().iter();
        while let Some(b) = text.next() {
            if *b == b'\\' {
                match text.next().copied().unwrap_or(b'\n') {
                    b'\\' | b'n' | b'"' => {}
                    b => return Err(invalid_escape(b)),
                }
            }
        }
        Ok(())
    }
}

fn invalid_escape(b: u8) -> String {
    format!("invalid escape sequence '\\{}'", char::from(b))
}

fn is_blank(b: u8) -> bool {
    b == b' ' || b == b'\t'
}

fn is_label_name_byte(b: u8, first: bool) -> bool {
    b.is_ascii_alphabetic() || b == b'_' || (!first && b.is_ascii_digit())
}

fn is_metric_name_byte(b: u8, first: bool) -> bool {
    is_label_name_byte(b, first) || b == b':'
}

/// Parses a float as the Go parser does: `strconv.ParseFloat` without hex
/// floats or digit separators.
fn parse_float(s: &str) -> Option<f64> {
    if s.contains(['p', 'P', '_']) {
        return None;
    }
    let v: f64 = s.parse().ok()?;
    let unsigned = s.trim_start_matches(['+', '-']).as_bytes();
    let spelled =
        |word: &str| unsigned.len() >= 3 && unsigned[..3].eq_ignore_ascii_case(word.as_bytes());
    // Go takes no sign on a NaN, and reports a number too large for a
    // float64 as an error where Rust returns infinity.
    if (v.is_nan() && unsigned.len() != s.len()) || (v.is_infinite() && !spelled("inf")) {
        return None;
    }
    Some(v)
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    /// A cut of a real vLLM /metrics body, with one engine and one model
    /// label set. The TTFT histogram holds 100 observations, 50 under 0.25s
    /// and 99 under 1s.
    pub(in crate::fleet) const VLLM_METRICS: &str = r#"# HELP vllm:cache_config_info Information of the LLMEngine CacheConfig
# TYPE vllm:cache_config_info gauge
vllm:cache_config_info{block_size="16",cache_dtype="auto",calculate_kv_scales="False",cpu_offload_gb="0",enable_prefix_caching="True",gpu_memory_utilization="0.9",num_cpu_blocks="0",num_gpu_blocks="4941",sliding_window="None",swap_space="4"} 1.0
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="Qwen/Qwen3-8B-AWQ"} 2.0
# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{engine="0",model_name="Qwen/Qwen3-8B-AWQ"} 3.0
# HELP vllm:kv_cache_usage_perc KV-cache usage. 1 means 100 percent usage.
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{engine="0",model_name="Qwen/Qwen3-8B-AWQ"} 0.42
# HELP vllm:prompt_tokens_total Number of prefill tokens processed.
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{engine="0",model_name="Qwen/Qwen3-8B-AWQ"} 812345.0
# HELP vllm:time_to_first_token_seconds Histogram of time to first token in seconds.
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_sum{engine="0",model_name="Qwen/Qwen3-8B-AWQ"} 41.7
vllm:time_to_first_token_seconds_bucket{engine="0",le="0.1",model_name="Qwen/Qwen3-8B-AWQ"} 0.0
vllm:time_to_first_token_seconds_bucket{engine="0",le="0.25",model_name="Qwen/Qwen3-8B-AWQ"} 50.0
vllm:time_to_first_token_seconds_bucket{engine="0",le="0.5",model_name="Qwen/Qwen3-8B-AWQ"} 50.0
vllm:time_to_first_token_seconds_bucket{engine="0",le="1.0",model_name="Qwen/Qwen3-8B-AWQ"} 99.0
vllm:time_to_first_token_seconds_bucket{engine="0",le="2.5",model_name="Qwen/Qwen3-8B-AWQ"} 100.0
vllm:time_to_first_token_seconds_bucket{engine="0",le="+Inf",model_name="Qwen/Qwen3-8B-AWQ"} 100.0
vllm:time_to_first_token_seconds_count{engine="0",model_name="Qwen/Qwen3-8B-AWQ"} 100.0
"#;

    fn must_parse(body: &str) -> Sample {
        parse_metrics(body.as_bytes()).expect("parse_metrics")
    }

    fn hist(bounds: &[f64], counts: &[f64]) -> Histogram {
        assert_eq!(bounds.len(), counts.len());
        Histogram {
            buckets: bounds
                .iter()
                .zip(counts)
                .map(|(le, count)| Bucket {
                    le: *le,
                    count: *count,
                })
                .collect(),
        }
    }

    fn bounds(h: &Histogram) -> Vec<f64> {
        h.buckets.iter().map(|b| b.le).collect()
    }

    fn counts(h: &Histogram) -> Vec<f64> {
        h.buckets.iter().map(|b| b.count).collect()
    }

    fn close(got: f64, want: f64) -> bool {
        (got - want).abs() <= 1e-9
    }

    #[test]
    fn parse_metrics_reads_vllm_body() {
        let s = must_parse(VLLM_METRICS);

        assert_eq!(s.kv_usage, Some(0.42));
        assert_eq!(s.kv_pool_tokens, 79056);
        assert_eq!(s.waiting, 3.0);
        assert_eq!(
            s.ttft,
            hist(
                &[0.1, 0.25, 0.5, 1.0, 2.5, f64::INFINITY],
                &[0.0, 50.0, 50.0, 99.0, 100.0, 100.0],
            )
        );
    }

    #[test]
    fn parse_metrics_normalises_legacy_percent_name() {
        let s = must_parse(
            "# TYPE vllm:gpu_cache_usage_perc gauge\nvllm:gpu_cache_usage_perc{model_name=\"m\"} 42.0\n",
        );
        assert_eq!(s.kv_usage, Some(0.42));
        assert_eq!(s.kv_pool_tokens, 0, "without cache_config_info");
    }

    #[test]
    fn parse_metrics_sums_across_label_sets() {
        let s = must_parse(
            r#"# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{engine="0"} 0.1
vllm:kv_cache_usage_perc{engine="1"} 0.2
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{engine="0"} 1.0
vllm:num_requests_waiting{engine="1"} 4.0
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{engine="0",le="0.5"} 3.0
vllm:time_to_first_token_seconds_bucket{engine="0",le="+Inf"} 5.0
vllm:time_to_first_token_seconds_bucket{engine="1",le="0.5"} 2.0
vllm:time_to_first_token_seconds_bucket{engine="1",le="+Inf"} 7.0
"#,
        );
        let usage = s.kv_usage.expect("kv_usage");
        assert!(close(usage, 0.3), "kv_usage = {usage}, want 0.3");
        assert_eq!(s.waiting, 5.0);
        assert_eq!(counts(&s.ttft), [5.0, 12.0]);
    }

    #[test]
    fn parse_metrics_without_kv_usage_is_not_ok() {
        let s = must_parse(
            "# TYPE vllm:num_requests_waiting gauge\nvllm:num_requests_waiting{engine=\"0\"} 3.0\n",
        );
        assert_eq!(s.kv_usage, None, "without a KV usage metric");
        assert_eq!(s.waiting, 3.0);
    }

    #[test]
    fn parse_metrics_rejects_malformed_body() {
        let err = parse_metrics(b"vllm:kv_cache_usage_perc{engine=\"0\" 0.5\n")
            .expect_err("want a parse error");
        assert_eq!(
            err.to_string(),
            "parse metrics: text format parsing error in line 1: unexpected end of label value \"0\""
        );
    }

    #[test]
    fn parse_metrics_fails_closed_on_garbage_numbers() {
        let usage = |v: &str| {
            must_parse(&format!(
                "# TYPE vllm:kv_cache_usage_perc gauge\nvllm:kv_cache_usage_perc{{engine=\"0\"}} {v}\n"
            ))
            .kv_usage
        };
        for (v, want) in [
            ("-0.5", 0.0),
            ("NaN", 1.0),
            ("+Inf", 1.0),
            ("-Inf", 1.0),
            ("250", 1.0),
            ("0.5", 0.5),
        ] {
            assert_eq!(usage(v), Some(want), "kv_usage from {v}");
        }

        let waiting = |v: &str| {
            must_parse(&format!(
                "# TYPE vllm:num_requests_waiting gauge\nvllm:num_requests_waiting 1\nvllm:num_requests_waiting{{e=\"1\"}} {v}\n"
            ))
            .waiting
        };
        for (v, want) in [("-5", 0.0), ("NaN", 0.0), ("+Inf", 0.0), ("2", 3.0)] {
            assert_eq!(waiting(v), want, "waiting from 1 + {v}");
        }

        let pool = |blocks: &str, size: &str| {
            must_parse(&format!(
                "vllm:cache_config_info{{num_gpu_blocks=\"{blocks}\",block_size=\"{size}\"}} 1\nvllm:cache_config_info{{num_gpu_blocks=\"2\",block_size=\"16\"}} 1\n"
            ))
            .kv_pool_tokens
        };
        assert_eq!(pool("-4941", "16"), 32, "negative blocks count as 0");
        assert_eq!(pool("x", "16"), 32, "non-numeric blocks count as 0");
        assert_eq!(
            pool("18446744073709551615", "16"),
            u64::MAX,
            "an overflowing product saturates"
        );
    }

    #[test]
    fn parse_metrics_drops_buckets_with_bad_counts() {
        let s = must_parse(
            r#"# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{le="0.5"} NaN
vllm:time_to_first_token_seconds_bucket{le="1"} +Inf
vllm:time_to_first_token_seconds_bucket{le="NaN"} 3
vllm:time_to_first_token_seconds_bucket{le="+Inf"} 4
"#,
        );
        assert_eq!(s.ttft, hist(&[f64::INFINITY], &[4.0]));
    }

    #[test]
    fn parse_metrics_refuses_too_many_labels() {
        let line = |n: usize| {
            let labels: Vec<String> = (0..n).map(|i| format!("l{i}=\"v\"")).collect();
            format!("other_metric{{{}}} 1\n", labels.join(","))
        };
        assert!(parse_metrics(line(MAX_LABELS_PER_SAMPLE).as_bytes()).is_ok());
        let err = parse_metrics(line(MAX_LABELS_PER_SAMPLE + 1).as_bytes())
            .expect_err("want a parse error");
        assert_eq!(
            err.to_string(),
            "parse metrics: text format parsing error in line 1: more than 64 labels for metric \"other_metric\""
        );
    }

    #[test]
    fn parse_metrics_keeps_validating_unread_families() {
        let err = parse_metrics(b"other{a=\"1\",a=\"2\"} 1\n").expect_err("duplicate label");
        assert_eq!(
            err.to_string(),
            "parse metrics: text format parsing error in line 1: duplicate label names for metric \"other\""
        );
        let families = Families::parse(VLLM_METRICS.as_bytes()).expect("parse");
        let stored: usize = families
            .all
            .iter()
            .map(|f| f.points.len() + f.buckets.len())
            .sum();
        assert_eq!(
            stored, 9,
            "points of the four read families and the six TTFT buckets"
        );
    }

    #[test]
    fn quantile_interpolates_inside_bucket() {
        let h = must_parse(VLLM_METRICS).ttft;
        let none = Histogram::default();

        for (q, want) in [(0.5, 0.25), (0.99, 1.0), (0.25, 0.175)] {
            let got = h.quantile(&none, q);
            assert!(close(got, want), "p{q} = {got}, want {want}");
        }
    }

    #[test]
    fn quantile_subtracts_prev() {
        let h = must_parse(VLLM_METRICS).ttft;
        let prev = hist(&bounds(&h), &[10.0; 6]);
        let shifted: Vec<f64> = counts(&h).iter().map(|c| c + 10.0).collect();
        let cur = hist(&bounds(&h), &shifted);

        for (q, want) in [(0.5, 0.25), (0.99, 1.0)] {
            let got = cur.quantile(&prev, q);
            assert!(close(got, want), "p{q} of delta = {got}, want {want}");
        }
    }

    #[test]
    fn quantile_edges() {
        let h = must_parse(VLLM_METRICS).ttft;
        let none = Histogram::default();

        assert_eq!(h.quantile(&h, 0.5), 0.0, "p50 of empty delta");
        assert_eq!(none.quantile(&none, 0.5), 0.0, "p50 of empty histogram");

        let tail = hist(&bounds(&h), &[0.0, 0.0, 0.0, 0.0, 0.0, 4.0]);
        assert_eq!(tail.quantile(&none, 0.5), 2.5, "p50 in +Inf bucket");
    }
}
