//! Runs one fake vLLM worker for the laptop demo. It serves /metrics and
//! /v1/chat/completions the way `fake::vllm` does in the tests, and answers
//! every run with a few read-tool calls and then an inconclusive diagnosis.

use std::process::ExitCode;

use ariadne_gateway::fake::vllm::Worker;
use ariadne_gateway::goduration::GoDuration;
use clap::Parser;
use tokio::net::TcpListener;

#[derive(Parser)]
#[command(name = "fakevllm")]
struct Args {
    /// address to serve on
    #[arg(long, default_value = "127.0.0.1:18001")]
    listen: String,

    /// read-tool calls per run before it submits
    #[arg(long, default_value = "3", allow_negative_numbers = true)]
    steps: i64,

    /// delay before every chat answer
    #[arg(long, default_value = "300ms")]
    latency: GoDuration,

    /// KV cache usage reported on /metrics, 0..1
    #[arg(long, default_value = "0.1", allow_negative_numbers = true)]
    kv_usage: f64,
}

#[tokio::main]
async fn main() -> ExitCode {
    let args = Args::parse();

    let worker = Worker::new();
    worker.set(|s| {
        // A negative step count submits at once, as zero does.
        s.steps = usize::try_from(args.steps).unwrap_or(0);
        s.latency = args.latency.to_std();
        s.kv_usage = args.kv_usage;
    });
    eprintln!(
        "fakevllm on {}: {} steps, {} latency",
        args.listen, args.steps, args.latency,
    );
    let served = async {
        let listener = TcpListener::bind(&args.listen).await?;
        axum::serve(listener, worker.router()).await
    };
    match served.await {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("{err}");
            ExitCode::FAILURE
        }
    }
}
