use std::time::Duration;

use ariadne_decide::{DataClass, Priority, Request, TenantState, Timestamp};

mod pick;
mod request;
mod shed;

const FIXED_NOW: Timestamp = Timestamp::from_epoch_offset(Duration::from_secs(1_000_000));

const AMPLE_TENANT: TenantState = TenantState {
    available_tokens: 1_000_000.0,
    rate_per_s: 1000.0,
};

fn after(secs: u64) -> Timestamp {
    FIXED_NOW.saturating_add(Duration::from_secs(secs))
}

/// A realistic doctor step-3 request: two messages, a tools array, sampling params the gateway
/// does not care about, and an explicit `max_completion_tokens`.
const DOCTOR_BODY: &str = r#"{
  "model": "Qwen/Qwen3-8B-AWQ",
  "messages": [
    {"role": "system", "content": "You are a Kubernetes cluster doctor."},
    {"role": "user", "content": "Diagnose the crashloop on pod api-7d9f4."}
  ],
  "tools": [
    {"type": "function", "function": {"name": "get_pod_logs", "parameters": {"type": "object"}}}
  ],
  "max_completion_tokens": 768,
  "temperature": 0.7,
  "top_k": 20,
  "chat_template_kwargs": {"enable_thinking": false}
}"#;

/// The request the Go tests wrote as `Request{Priority: p, Deadline: d}`.
fn request(priority: Priority, deadline: Timestamp) -> Request {
    Request {
        id: String::new(),
        run: None,
        step: 0,
        tenant: String::new(),
        app: String::new(),
        priority,
        class: DataClass::Restricted,
        body_bytes: 0,
        max_out: 0,
        arrived: FIXED_NOW,
        deadline,
    }
}
