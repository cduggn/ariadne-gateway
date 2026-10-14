//! Go's `http.ServeMux` routing (as of Go 1.27) over the gateway's fixed patterns: unclean paths redirect, a
//! known path with the wrong method is a 405 naming the allowed methods, anything else a 404.

use http::Method;

/// One registered pattern.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Route {
    Chat,
    Healthz,
    Readyz,
    Workers,
    WorkerMetrics(String),
    Metrics,
}

impl Route {
    /// The `Allow` header for the route's path. A GET pattern also serves HEAD.
    const fn allow(&self) -> &'static str {
        match self {
            Self::Chat => "POST",
            Self::Healthz
            | Self::Readyz
            | Self::Workers
            | Self::WorkerMetrics(_)
            | Self::Metrics => "GET, HEAD",
        }
    }

    fn accepts(&self, m: &Method) -> bool {
        match self {
            Self::Chat => m == Method::POST,
            Self::Healthz
            | Self::Readyz
            | Self::Workers
            | Self::WorkerMetrics(_)
            | Self::Metrics => m == Method::GET || m == Method::HEAD,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Dispatch {
    Serve(Route),
    /// 307 to this location, the escaped cleaned path.
    Redirect(String),
    MethodNotAllowed {
        allow: &'static str,
    },
    NotFound,
    /// The path has a malformed percent escape, which Go's server refuses before routing.
    BadRequest,
}

/// Routes one request. `path` is the escaped request path and `query` the raw query.
/// `metrics` says whether `GET /metrics` is registered.
pub(super) fn dispatch(
    method: &Method,
    path: &str,
    query: Option<&str>,
    metrics: bool,
) -> Dispatch {
    if unescape(path).is_none() {
        return Dispatch::BadRequest;
    }
    let clean = clean_path(path);
    if clean != path {
        let mut location = clean;
        if let Some(q) = query.filter(|q| !q.is_empty()) {
            location.push('?');
            location.push_str(q);
        }
        return Dispatch::Redirect(location);
    }
    let Some(route) = route(path, metrics) else {
        return Dispatch::NotFound;
    };
    if route.accepts(method) {
        Dispatch::Serve(route)
    } else {
        Dispatch::MethodNotAllowed {
            allow: route.allow(),
        }
    }
}

/// Matches the unescaped segments of a clean path.
fn route(path: &str, metrics: bool) -> Option<Route> {
    let segments: Vec<Vec<u8>> = path
        .strip_prefix('/')?
        .split('/')
        .map(unescape)
        .collect::<Option<_>>()?;
    let segs: Vec<&[u8]> = segments.iter().map(Vec::as_slice).collect();
    match segs.as_slice() {
        [b"v1", b"chat", b"completions"] => Some(Route::Chat),
        [b"healthz"] => Some(Route::Healthz),
        [b"readyz"] => Some(Route::Readyz),
        [b"debug", b"workers"] => Some(Route::Workers),
        [b"debug", b"workers", pod, b"metrics"] if !pod.is_empty() => Some(Route::WorkerMetrics(
            String::from_utf8_lossy(pod).into_owned(),
        )),
        [b"metrics"] if metrics => Some(Route::Metrics),
        _ => None,
    }
}

/// Go's `cleanPath`: `path.Clean` on a rooted path, keeping a trailing slash.
fn clean_path(p: &str) -> String {
    let mut kept: Vec<&str> = Vec::new();
    for seg in p.split('/') {
        match seg {
            "" | "." => {}
            ".." => {
                kept.pop();
            }
            s => kept.push(s),
        }
    }
    let mut out = format!("/{}", kept.join("/"));
    if p.ends_with('/') && out != "/" {
        out.push('/');
    }
    out
}

/// Percent-decodes `s`, or `None` for a `%` not followed by two hex digits.
fn unescape(s: &str) -> Option<Vec<u8>> {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while let Some(&c) = b.get(i) {
        if c == b'%' {
            let hex = |j: usize| b.get(j).and_then(|h| char::from(*h).to_digit(16));
            let byte = hex(i + 1)? * 16 + hex(i + 2)?;
            out.push(u8::try_from(byte).ok()?);
            i += 3;
        } else {
            out.push(c);
            i += 1;
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_follows_go_servemux() {
        let get = Method::GET;
        let cases: Vec<(Method, &str, Option<&str>, bool, Dispatch)> = vec![
            (
                Method::POST,
                "/v1/chat/completions",
                None,
                false,
                Dispatch::Serve(Route::Chat),
            ),
            (
                get.clone(),
                "/v1/chat/completions",
                None,
                false,
                Dispatch::MethodNotAllowed { allow: "POST" },
            ),
            (
                get.clone(),
                "/healthz",
                None,
                false,
                Dispatch::Serve(Route::Healthz),
            ),
            (
                Method::HEAD,
                "/readyz",
                None,
                false,
                Dispatch::Serve(Route::Readyz),
            ),
            (
                Method::POST,
                "/healthz",
                None,
                false,
                Dispatch::MethodNotAllowed { allow: "GET, HEAD" },
            ),
            (get.clone(), "/nope", None, false, Dispatch::NotFound),
            (get.clone(), "/healthz/", None, false, Dispatch::NotFound),
            (get.clone(), "/metrics", None, false, Dispatch::NotFound),
            (
                Method::POST,
                "/metrics",
                None,
                true,
                Dispatch::MethodNotAllowed { allow: "GET, HEAD" },
            ),
            (
                get.clone(),
                "/metrics",
                None,
                true,
                Dispatch::Serve(Route::Metrics),
            ),
            (
                get.clone(),
                "/debug/workers",
                None,
                false,
                Dispatch::Serve(Route::Workers),
            ),
            (
                get.clone(),
                "/debug/workers/vllm%2D0/metrics",
                None,
                false,
                Dispatch::Serve(Route::WorkerMetrics("vllm-0".to_owned())),
            ),
            (
                get.clone(),
                "//healthz",
                Some("a=1"),
                false,
                Dispatch::Redirect("/healthz?a=1".to_owned()),
            ),
            (
                get.clone(),
                "/debug/./workers/../../healthz",
                None,
                false,
                Dispatch::Redirect("/healthz".to_owned()),
            ),
            (
                get.clone(),
                "/x/../a%20b",
                None,
                false,
                Dispatch::Redirect("/a%20b".to_owned()),
            ),
            (get.clone(), "/heal%zzhz", None, false, Dispatch::BadRequest),
        ];
        for (method, path, query, metrics, want) in cases {
            assert_eq!(
                dispatch(&method, path, query, metrics),
                want,
                "{method} {path}"
            );
        }
    }
}
