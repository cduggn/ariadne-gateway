//! In-process fakes of the gateway's upstreams, for tests and the laptop demo.

pub mod overflow;
pub mod vllm;

use std::io;
use std::net::SocketAddr;

use axum::Router;
use tokio::net::TcpListener;
use tokio::sync::oneshot;

/// A router being served on a loopback port. Dropping it stops the listener
/// and closes idle connections, as `httptest.Server.Close` does.
#[derive(Debug)]
pub struct Served {
    addr: SocketAddr,
    _shutdown: oneshot::Sender<()>,
}

impl Served {
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// The base URL, `http://127.0.0.1:<port>` with no trailing slash.
    pub fn url(&self) -> String {
        format!("http://{}", self.addr)
    }
}

/// Serves `router` on `127.0.0.1:0` from a background task.
pub async fn serve(router: Router) -> io::Result<Served> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    let (shutdown, dropped) = oneshot::channel::<()>();
    tokio::spawn(async move {
        // A test has nowhere to report a serve error, and its requests fail.
        let _ = axum::serve(listener, router)
            .with_graceful_shutdown(async move {
                let _ = dropped.await;
            })
            .await;
    });
    Ok(Served {
        addr,
        _shutdown: shutdown,
    })
}
