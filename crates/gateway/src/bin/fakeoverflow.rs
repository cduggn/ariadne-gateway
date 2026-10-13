//! Runs one fake overflow backend for the laptop demo. It serves GET
//! /v1/models and POST /v1/chat/completions the way `fake::overflow` does in
//! the tests, checks a bearer key when given one, and answers every chat
//! with an inconclusive diagnosis.

use std::process::ExitCode;

use ariadne_gateway::fake::overflow::Backend;
use ariadne_gateway::goduration::GoDuration;
use clap::{ArgAction, Parser};
use http::StatusCode;
use tokio::net::TcpListener;

#[derive(Parser)]
#[command(name = "fakeoverflow")]
struct Args {
    /// address to serve on
    #[arg(long, default_value = "127.0.0.1:18010")]
    listen: String,

    /// bearer key every request must carry; empty accepts any
    #[arg(long, default_value = "")]
    key: String,

    /// status every chat answers, e.g. 402 to play an out-of-credit backend
    #[arg(long, default_value = "200", value_parser = parse_status)]
    status: StatusCode,

    /// delay before every chat answer
    #[arg(long, default_value = "500ms")]
    latency: GoDuration,

    /// answer the probe with 503, as a backend that is out would
    #[arg(
        long,
        action = ArgAction::Set,
        num_args = 0..=1,
        require_equals = true,
        default_value = "false",
        default_missing_value = "true",
        value_parser = parse_bool,
    )]
    down: bool,
}

fn parse_status(s: &str) -> Result<StatusCode, String> {
    let code: u16 = s.parse().map_err(|_| "parse error".to_owned())?;
    StatusCode::from_u16(code).map_err(|_| format!("{code} is not an HTTP status"))
}

/// Accepts what Go's `strconv.ParseBool` does, so `--down=false` still works.
fn parse_bool(s: &str) -> Result<bool, String> {
    match s {
        "1" | "t" | "T" | "TRUE" | "true" | "True" => Ok(true),
        "0" | "f" | "F" | "FALSE" | "false" | "False" => Ok(false),
        _ => Err("parse error".to_owned()),
    }
}

#[tokio::main]
async fn main() -> ExitCode {
    let args = Args::parse();

    let backend = Backend::new();
    backend.set(|s| {
        s.key = Some(args.key.clone()).filter(|key| !key.is_empty());
        s.status = args.status;
        s.latency = args.latency.to_std();
        s.down = args.down;
    });
    eprintln!(
        "fakeoverflow on {}: status {}, {} latency, key {}, down {}",
        args.listen,
        args.status.as_u16(),
        args.latency,
        !args.key.is_empty(),
        args.down,
    );
    let served = async {
        let listener = TcpListener::bind(&args.listen).await?;
        axum::serve(listener, backend.router()).await
    };
    match served.await {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("{err}");
            ExitCode::FAILURE
        }
    }
}
