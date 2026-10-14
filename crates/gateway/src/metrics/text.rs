//! The Prometheus text exposition as client_golang v1.24 writes it: families sorted by name,
//! series sorted by their label values in label-name order, label pairs sorted by name with `le`
//! last, and floats in Go's `strconv.FormatFloat(v, 'g', -1, 64)` form.

use std::collections::BTreeMap;
use std::fmt::Write as _;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Kind {
    Counter,
    Gauge,
    Histogram,
}

impl Kind {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Counter => "counter",
            Self::Gauge => "gauge",
            Self::Histogram => "histogram",
        }
    }
}

/// One metric family: its name, help text, type and label names.
#[derive(Debug)]
pub(super) struct Spec<const N: usize> {
    pub name: &'static str,
    pub help: &'static str,
    pub kind: Kind,
    pub labels: [&'static str; N],
}

/// The histogram's upper bounds, 5 ms to 120 s, the client's timeout.
pub(super) const DURATION_BUCKETS: [f64; 14] = [
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0,
];

#[derive(Debug, Clone, Default, PartialEq)]
pub(super) struct Histogram {
    /// Observations per bucket, not cumulative. One above the last bound counts only in `count`.
    buckets: [u64; DURATION_BUCKETS.len()],
    count: u64,
    sum: f64,
}

impl Histogram {
    pub(super) fn observe(&mut self, v: f64) {
        if let Some(i) = DURATION_BUCKETS.iter().position(|bound| v <= *bound) {
            self.buckets[i] += 1;
        }
        self.count += 1;
        self.sum += v;
    }
}

#[derive(Debug, Default)]
struct Family {
    header: String,
    /// Each series' lines, keyed by its label values in label-name order.
    series: BTreeMap<Vec<String>, String>,
}

/// A scrape being assembled. Samples may arrive in any order; `finish` sorts them. A family
/// with no sample is not written, as client_golang skips a vector with no children.
#[derive(Debug, Default)]
pub(super) struct Exposition(BTreeMap<&'static str, Family>);

impl Exposition {
    pub(super) fn sample<const N: usize>(&mut self, spec: &Spec<N>, values: [&str; N], value: f64) {
        let (lines, labels) = self.series(spec, values);
        let _ = writeln!(
            lines,
            "{}{} {}",
            spec.name,
            braces(&labels),
            go_float(value)
        );
    }

    pub(super) fn histogram<const N: usize>(
        &mut self,
        spec: &Spec<N>,
        values: [&str; N],
        h: &Histogram,
    ) {
        let (lines, labels) = self.series(spec, values);
        let sep = if labels.is_empty() { "" } else { "," };
        let mut cumulative = 0;
        for (bound, n) in DURATION_BUCKETS.iter().zip(h.buckets) {
            cumulative += n;
            let le = go_float(*bound);
            let _ = writeln!(
                lines,
                "{}_bucket{{{labels}{sep}le=\"{le}\"}} {cumulative}",
                spec.name
            );
        }
        let _ = writeln!(
            lines,
            "{}_bucket{{{labels}{sep}le=\"+Inf\"}} {}",
            spec.name, h.count
        );
        let _ = writeln!(
            lines,
            "{}_sum{} {}",
            spec.name,
            braces(&labels),
            go_float(h.sum)
        );
        let _ = writeln!(lines, "{}_count{} {}", spec.name, braces(&labels), h.count);
    }

    /// The lines of one series, and its label pairs rendered in name order.
    fn series<const N: usize>(
        &mut self,
        spec: &Spec<N>,
        values: [&str; N],
    ) -> (&mut String, String) {
        let mut pairs: Vec<(&str, &str)> = spec.labels.into_iter().zip(values).collect();
        pairs.sort_unstable_by_key(|(name, _)| *name);
        let mut labels = String::new();
        for (i, (name, value)) in pairs.iter().enumerate() {
            let sep = if i == 0 { "" } else { "," };
            let _ = write!(labels, "{sep}{name}=\"{}\"", escape(value));
        }
        let family = self.0.entry(spec.name).or_insert_with(|| Family {
            header: format!(
                "# HELP {0} {1}\n# TYPE {0} {2}\n",
                spec.name,
                spec.help,
                spec.kind.as_str()
            ),
            series: BTreeMap::new(),
        });
        let key = pairs.iter().map(|(_, v)| (*v).to_owned()).collect();
        (family.series.entry(key).or_default(), labels)
    }

    pub(super) fn finish(self) -> String {
        let mut out = String::new();
        for family in self.0.into_values() {
            out.push_str(&family.header);
            for lines in family.series.into_values() {
                out.push_str(&lines);
            }
        }
        out
    }
}

fn braces(labels: &str) -> String {
    if labels.is_empty() {
        String::new()
    } else {
        format!("{{{labels}}}")
    }
}

fn escape(value: &str) -> String {
    value
        .replace('\\', r"\\")
        .replace('"', "\\\"")
        .replace('\n', r"\n")
}

/// `v` as Go's `strconv.FormatFloat(v, 'g', -1, 64)` writes it, with client_golang's spellings
/// of zero and the infinities. Shortest digits either way; the exponent form when the decimal
/// exponent is below -4 or at least 6, with a sign and at least two exponent digits.
pub(super) fn go_float(v: f64) -> String {
    if v == 0.0 {
        return "0".to_owned();
    }
    if v.is_nan() {
        return "NaN".to_owned();
    }
    if v.is_infinite() {
        return if v > 0.0 { "+Inf" } else { "-Inf" }.to_owned();
    }
    let sci = format!("{v:e}");
    let (mantissa, exp) = sci.split_once('e').unwrap_or((&sci, "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    if (-4..6).contains(&exp) {
        format!("{v}")
    } else {
        let sign = if exp < 0 { '-' } else { '+' };
        format!("{mantissa}e{sign}{:02}", exp.unsigned_abs())
    }
}
