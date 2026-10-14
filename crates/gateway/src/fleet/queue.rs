use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::{Arc, Mutex};

use ariadne_decide::{Priority, Reason, Timestamp};
use tokio::sync::oneshot;

use super::lock;
use crate::clock::Clock;

/// Why `Queue::acquire` turned a request away.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QueueRefusal {
    UnknownPod,
    /// The queue was full, or an interactive arrival displaced this batch waiter.
    Full,
    /// The deadline passed before a slot came free.
    Timeout,
}

impl QueueRefusal {
    /// The shed label the caller reports.
    pub const fn reason(self) -> Reason {
        match self {
            Self::UnknownPod => Reason::NoEligiblePod,
            Self::Full => Reason::QueueFull,
            Self::Timeout => Reason::TimeoutQueue,
        }
    }
}

impl fmt::Display for QueueRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "gateway queue: {}", self.reason().as_str())
    }
}

impl std::error::Error for QueueRefusal {}

/// What ended a waiter's wait, sent under the queue lock. A grant is counted in flight by the
/// sender, so the waiter owns that slot from then on, whether or not it is still listening.
#[derive(Debug)]
enum Wake {
    Granted,
    Displaced,
}

#[derive(Debug)]
struct Waiter {
    id: u64,
    wake: oneshot::Sender<Wake>,
}

/// Each lane is FIFO, and the interactive lane drains first.
#[derive(Debug, Default)]
struct Lanes {
    interactive: VecDeque<Waiter>,
    batch: VecDeque<Waiter>,
}

impl Lanes {
    fn lane(&mut self, prio: Priority) -> &mut VecDeque<Waiter> {
        match prio {
            Priority::Interactive => &mut self.interactive,
            Priority::Batch => &mut self.batch,
        }
    }

    fn len(&self) -> usize {
        self.interactive.len() + self.batch.len()
    }

    fn remove(&mut self, prio: Priority, id: u64) -> bool {
        let lane = self.lane(prio);
        let before = lane.len();
        lane.retain(|w| w.id != id);
        lane.len() != before
    }
}

#[derive(Debug)]
struct Worker {
    in_flight: usize,
    max: usize,
    max_queued: usize,
    lanes: Lanes,
}

impl Worker {
    fn grant_waiters(&mut self) {
        while self.in_flight < self.max {
            let Some(head) = self
                .lanes
                .interactive
                .pop_front()
                .or_else(|| self.lanes.batch.pop_front())
            else {
                return;
            };
            // A waiter that stopped listening takes no slot.
            if head.wake.send(Wake::Granted).is_ok() {
                self.in_flight += 1;
            }
        }
    }
}

#[derive(Debug, Default)]
struct State {
    workers: HashMap<String, Worker>,
    next_waiter: u64,
}

/// One pod's in-flight count and the length of each lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Depth {
    pub in_flight: usize,
    pub interactive: usize,
    pub batch: usize,
}

impl Depth {
    pub const fn queued(&self) -> usize {
        self.interactive + self.batch
    }
}

/// The gateway's per-worker admission queue. One mutex guards every worker, because grants are
/// rare and short, and one lock keeps `depth` consistent across lanes. The lock is never held
/// across an `.await`, and the queue spawns no tasks: whoever frees a slot wakes the next waiter.
#[derive(Debug, Clone)]
pub struct Queue {
    clock: Clock,
    state: Arc<Mutex<State>>,
}

/// One held in-flight slot. Dropping it returns the slot and grants it to the next waiter, so a
/// slot is released exactly once on every exit path.
pub struct Slot {
    state: Arc<Mutex<State>>,
    pod: String,
}

impl fmt::Debug for Slot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Slot").field("pod", &self.pod).finish()
    }
}

impl Drop for Slot {
    fn drop(&mut self) {
        let mut st = lock(&self.state);
        if let Some(w) = st.workers.get_mut(&self.pod) {
            w.in_flight = w.in_flight.saturating_sub(1);
            w.grant_waiters();
        }
    }
}

enum Entry<'q> {
    Slot(Slot),
    Waiting(Waiting<'q>),
}

/// A waiter in its lane. However the wait ends, including the `acquire` future being dropped,
/// the waiter leaves its lane, and a slot granted to it that nobody will use is released.
struct Waiting<'q> {
    queue: &'q Queue,
    pod: &'q str,
    prio: Priority,
    id: u64,
    wake: oneshot::Receiver<Wake>,
    left: bool,
}

impl Waiting<'_> {
    /// Under the queue lock a waiter is either still in its lane or has been sent its wake, so
    /// this returns `None` exactly when it removed a waiter nobody had woken.
    fn leave(&mut self) -> Option<Wake> {
        let mut st = lock(&self.queue.state);
        let removed = st
            .workers
            .get_mut(self.pod)
            .is_some_and(|w| w.lanes.remove(self.prio, self.id));
        if removed {
            None
        } else {
            self.wake.try_recv().ok()
        }
    }
}

impl Drop for Waiting<'_> {
    fn drop(&mut self) {
        if !self.left && matches!(self.leave(), Some(Wake::Granted)) {
            drop(self.queue.slot(self.pod));
        }
    }
}

impl Queue {
    /// A queue for `pods`, each with `max_inflight` slots and room for `max_queued` waiters across
    /// both lanes. Deadlines are read on `clock`.
    pub fn new(pods: &[String], max_inflight: usize, max_queued: usize, clock: Clock) -> Self {
        let workers = pods
            .iter()
            .map(|pod| {
                let w = Worker {
                    in_flight: 0,
                    max: max_inflight,
                    max_queued,
                    lanes: Lanes::default(),
                };
                (pod.clone(), w)
            })
            .collect();
        Self {
            clock,
            state: Arc::new(Mutex::new(State {
                workers,
                next_waiter: 0,
            })),
        }
    }

    /// Takes an in-flight slot on `pod`, or waits for one until `deadline`. A free slot is granted
    /// at once only when no one is waiting, so a newcomer never passes the lanes. A full queue
    /// refuses a batch arrival; an interactive arrival displaces the newest batch waiter instead,
    /// which then returns `Full`. A deadline not in the future returns `Timeout` unless a slot is
    /// free. A waiter whose deadline passes just after it was granted keeps the slot.
    pub async fn acquire(
        &self,
        pod: &str,
        prio: Priority,
        deadline: Timestamp,
    ) -> Result<Slot, QueueRefusal> {
        let mut waiting = match self.enter(pod, prio, deadline)? {
            Entry::Slot(slot) => return Ok(slot),
            Entry::Waiting(waiting) => waiting,
        };
        let woken = tokio::select! {
            wake = &mut waiting.wake => Some(wake.ok()),
            () = tokio::time::sleep(self.clock.until(deadline)) => None,
        };
        let wake = woken.unwrap_or_else(|| waiting.leave());
        waiting.left = true;
        match wake {
            Some(Wake::Granted) => Ok(self.slot(pod)),
            Some(Wake::Displaced) => Err(QueueRefusal::Full),
            None => Err(QueueRefusal::Timeout),
        }
    }

    fn enter<'q>(
        &'q self,
        pod: &'q str,
        prio: Priority,
        deadline: Timestamp,
    ) -> Result<Entry<'q>, QueueRefusal> {
        let mut st = lock(&self.state);
        let State {
            workers,
            next_waiter,
        } = &mut *st;
        let w = workers.get_mut(pod).ok_or(QueueRefusal::UnknownPod)?;
        if w.in_flight < w.max && w.lanes.len() == 0 {
            w.in_flight += 1;
            return Ok(Entry::Slot(self.slot(pod)));
        }
        if deadline <= self.clock.now() {
            return Err(QueueRefusal::Timeout);
        }
        if w.lanes.len() >= w.max_queued {
            let displaced = match prio {
                Priority::Batch => None,
                Priority::Interactive => w.lanes.batch.pop_back(),
            };
            let displaced = displaced.ok_or(QueueRefusal::Full)?;
            // A failed send means that waiter already stopped listening; it is gone either way.
            let _ = displaced.wake.send(Wake::Displaced);
        }
        *next_waiter += 1;
        let (tx, rx) = oneshot::channel();
        w.lanes.lane(prio).push_back(Waiter {
            id: *next_waiter,
            wake: tx,
        });
        Ok(Entry::Waiting(Waiting {
            queue: self,
            pod,
            prio,
            id: *next_waiter,
            wake: rx,
            left: false,
        }))
    }

    /// A handle on a slot already counted in flight.
    fn slot(&self, pod: &str) -> Slot {
        Slot {
            state: Arc::clone(&self.state),
            pod: pod.to_owned(),
        }
    }

    /// `pod`'s in-flight count and lane lengths. An unknown pod reports zeros.
    pub fn depth(&self, pod: &str) -> Depth {
        let st = lock(&self.state);
        st.workers.get(pod).map_or(Depth::default(), |w| Depth {
            in_flight: w.in_flight,
            interactive: w.lanes.interactive.len(),
            batch: w.lanes.batch.len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use tokio::task::JoinHandle;

    use super::*;

    const POD: &str = "vllm-0";

    type Outcome = Result<Slot, QueueRefusal>;

    fn queue(max_inflight: usize, max_queued: usize) -> Queue {
        Queue::new(&[POD.to_owned()], max_inflight, max_queued, Clock::system())
    }

    fn far(q: &Queue) -> Timestamp {
        q.clock.now().saturating_add(Duration::from_secs(3600))
    }

    fn acquire_async(q: &Queue, prio: Priority, deadline: Timestamp) -> JoinHandle<Outcome> {
        let q = q.clone();
        tokio::spawn(async move { q.acquire(POD, prio, deadline).await })
    }

    fn depth(q: &Queue) -> (usize, usize, usize) {
        let d = q.depth(POD);
        (d.in_flight, d.interactive, d.batch)
    }

    /// Yields until the spawned acquirers have reached their lanes.
    async fn wait_depth(q: &Queue, want: (usize, usize, usize)) {
        for _ in 0..10_000 {
            if depth(q) == want {
                return;
            }
            tokio::task::yield_now().await;
        }
        assert_eq!(depth(q), want, "Depth");
    }

    async fn join(h: JoinHandle<Outcome>) -> Outcome {
        tokio::time::timeout(Duration::from_secs(2), h)
            .await
            .expect("Acquire did not return in time")
            .expect("acquire task")
    }

    async fn must_slot(h: JoinHandle<Outcome>) -> Slot {
        join(h)
            .await
            .expect("Acquire returned a refusal, want a slot")
    }

    async fn must_refusal(h: JoinHandle<Outcome>, want: QueueRefusal) {
        assert_eq!(join(h).await.err(), Some(want));
    }

    async fn assert_still_waiting(h: &JoinHandle<Outcome>) {
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(
            !h.is_finished(),
            "Acquire returned while the worker was full"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn grants_free_slots_then_waits_for_release() {
        let q = queue(2, 32);

        let s1 = must_slot(acquire_async(&q, Priority::Batch, far(&q))).await;
        assert_eq!(depth(&q), (1, 0, 0));
        let s2 = must_slot(acquire_async(&q, Priority::Batch, far(&q))).await;
        assert_eq!(depth(&q), (2, 0, 0));

        let third = acquire_async(&q, Priority::Interactive, far(&q));
        wait_depth(&q, (2, 1, 0)).await;
        assert_still_waiting(&third).await;

        drop(s1);
        let s3 = must_slot(third).await;
        assert_eq!(depth(&q), (2, 0, 0));

        drop((s2, s3));
        assert_eq!(depth(&q), (0, 0, 0));
    }

    #[tokio::test(start_paused = true)]
    async fn release_grants_interactive_lane_first_then_fifo() {
        let q = queue(1, 32);
        let holder = must_slot(acquire_async(&q, Priority::Batch, far(&q))).await;

        let (order_tx, mut order) = tokio::sync::mpsc::unbounded_channel();
        let mut tasks = Vec::new();
        for (name, prio, want) in [
            ("B1", Priority::Batch, (1, 0, 1)),
            ("I1", Priority::Interactive, (1, 1, 1)),
            ("B2", Priority::Batch, (1, 1, 2)),
            ("I2", Priority::Interactive, (1, 2, 2)),
        ] {
            let (q2, tx, deadline) = (q.clone(), order_tx.clone(), far(&q));
            tasks.push(tokio::spawn(async move {
                let slot = q2.acquire(POD, prio, deadline).await.expect(name);
                tx.send(name).expect("order channel");
                drop(slot);
            }));
            wait_depth(&q, want).await;
        }

        drop(holder);
        for (i, want) in ["I1", "I2", "B1", "B2"].into_iter().enumerate() {
            let got = tokio::time::timeout(Duration::from_secs(2), order.recv()).await;
            assert_eq!(got, Ok(Some(want)), "grant {}", i + 1);
        }
        for t in tasks {
            t.await.expect("waiter task");
        }
        assert_eq!(depth(&q), (0, 0, 0));
    }

    #[tokio::test(start_paused = true)]
    async fn queue_full_refuses_batch_and_displaces_newest_batch_for_interactive() {
        let q = queue(1, 2);
        let holder = must_slot(acquire_async(&q, Priority::Batch, far(&q))).await;

        let b1 = acquire_async(&q, Priority::Batch, far(&q));
        wait_depth(&q, (1, 0, 1)).await;
        let b2 = acquire_async(&q, Priority::Batch, far(&q));
        wait_depth(&q, (1, 0, 2)).await;

        let b3 = acquire_async(&q, Priority::Batch, far(&q));
        must_refusal(b3, QueueRefusal::Full).await;
        assert_eq!(depth(&q), (1, 0, 2));

        let i1 = acquire_async(&q, Priority::Interactive, far(&q));
        must_refusal(b2, QueueRefusal::Full).await;
        wait_depth(&q, (1, 1, 1)).await;
        assert_still_waiting(&b1).await;

        drop(holder);
        let s = must_slot(i1).await;
        assert_eq!(depth(&q), (1, 0, 1));

        drop(s);
        drop(must_slot(b1).await);
        assert_eq!(depth(&q), (0, 0, 0));
    }

    #[tokio::test(start_paused = true)]
    async fn queue_full_with_no_batch_waiters_refuses_interactive() {
        let q = queue(1, 1);
        let holder = must_slot(acquire_async(&q, Priority::Batch, far(&q))).await;

        let i1 = acquire_async(&q, Priority::Interactive, far(&q));
        wait_depth(&q, (1, 1, 0)).await;
        must_refusal(
            acquire_async(&q, Priority::Interactive, far(&q)),
            QueueRefusal::Full,
        )
        .await;

        drop(holder);
        drop(must_slot(i1).await);
        assert_eq!(depth(&q), (0, 0, 0));
    }

    #[tokio::test(start_paused = true)]
    async fn deadline_in_queue_returns_timeout_queue() {
        let q = queue(1, 32);
        tokio::time::advance(Duration::from_secs(2)).await;
        let past =
            || Timestamp::from_epoch_offset(q.clock.now().epoch_offset() - Duration::from_secs(1));
        let holder = must_slot(acquire_async(&q, Priority::Batch, far(&q))).await;

        let soon = q.clock.now().saturating_add(Duration::from_millis(15));
        must_refusal(
            acquire_async(&q, Priority::Batch, soon),
            QueueRefusal::Timeout,
        )
        .await;
        assert_eq!(depth(&q), (1, 0, 0));

        let late = q.acquire(POD, Priority::Interactive, past()).await;
        assert_eq!(late.err(), Some(QueueRefusal::Timeout));
        assert_eq!(depth(&q), (1, 0, 0));

        drop(holder);
        let s = q
            .acquire(POD, Priority::Interactive, past())
            .await
            .expect("a past deadline with a free slot gets the slot");
        assert_eq!(depth(&q), (1, 0, 0));
        drop(s);
        assert_eq!(depth(&q), (0, 0, 0));
    }

    #[tokio::test(start_paused = true)]
    async fn dropping_the_future_removes_the_waiter() {
        let q = queue(1, 32);
        let holder = must_slot(acquire_async(&q, Priority::Batch, far(&q))).await;

        let waiting = acquire_async(&q, Priority::Interactive, far(&q));
        wait_depth(&q, (1, 1, 0)).await;
        waiting.abort();
        assert!(waiting.await.is_err_and(|e| e.is_cancelled()), "cancelled");
        assert_eq!(depth(&q), (1, 0, 0));

        drop(holder);
        assert_eq!(depth(&q), (0, 0, 0));
    }

    #[tokio::test(start_paused = true)]
    async fn dropping_a_granted_waiter_releases_its_slot() {
        let q = queue(1, 32);
        let holder = must_slot(acquire_async(&q, Priority::Batch, far(&q))).await;
        let granted = acquire_async(&q, Priority::Batch, far(&q));
        let next = acquire_async(&q, Priority::Batch, far(&q));
        wait_depth(&q, (1, 0, 2)).await;

        drop(holder);
        assert_eq!(depth(&q), (1, 0, 1), "granted before it ran");
        granted.abort();
        assert!(granted.await.is_err_and(|e| e.is_cancelled()), "cancelled");

        drop(must_slot(next).await);
        assert_eq!(depth(&q), (0, 0, 0));
    }

    #[tokio::test(start_paused = true)]
    async fn each_release_grants_one_waiter_in_order() {
        let q = queue(1, 32);
        let holder = must_slot(acquire_async(&q, Priority::Batch, far(&q))).await;

        let w1 = acquire_async(&q, Priority::Batch, far(&q));
        wait_depth(&q, (1, 0, 1)).await;
        let w2 = acquire_async(&q, Priority::Batch, far(&q));
        wait_depth(&q, (1, 0, 2)).await;

        drop(holder);
        let s1 = must_slot(w1).await;
        assert_eq!(depth(&q), (1, 0, 1));
        assert_still_waiting(&w2).await;

        drop(s1);
        let s2 = must_slot(w2).await;
        assert_eq!(depth(&q), (1, 0, 0));

        drop(s2);
        assert_eq!(depth(&q), (0, 0, 0));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn no_slot_leak_under_race() {
        let q = queue(4, 32);
        let mut seed: u64 = 7;
        let mut tasks = Vec::new();
        for i in 0..200u64 {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let wait = Duration::from_micros((seed >> 33) % 8000);
            let prio = if i % 3 == 0 {
                Priority::Interactive
            } else {
                Priority::Batch
            };
            let q = q.clone();
            tasks.push(tokio::spawn(async move {
                let deadline = q.clock.now().saturating_add(wait);
                match q.acquire(POD, prio, deadline).await {
                    Ok(slot) => {
                        tokio::time::sleep(Duration::from_millis(1)).await;
                        drop(slot);
                        true
                    }
                    Err(r) => {
                        assert!(
                            matches!(r, QueueRefusal::Timeout | QueueRefusal::Full),
                            "refusal {r}"
                        );
                        false
                    }
                }
            }));
        }
        let mut grants = 0;
        for t in tasks {
            grants += usize::from(t.await.expect("acquirer"));
        }
        assert!(
            grants >= 4,
            "grants = {grants}, want at least the 4 free slots"
        );
        assert_eq!(depth(&q), (0, 0, 0));
    }

    #[tokio::test]
    async fn unknown_pod_returns_no_eligible_pod() {
        let q = queue(16, 32);
        let got = q.acquire("vllm-9", Priority::Interactive, far(&q)).await;
        assert_eq!(got.err(), Some(QueueRefusal::UnknownPod));
        assert_eq!(QueueRefusal::UnknownPod.reason(), Reason::NoEligiblePod);
        assert_eq!(q.depth("vllm-9"), Depth::default());
    }

    #[test]
    fn refusal_error() {
        assert_eq!(QueueRefusal::Full.to_string(), "gateway queue: queue_full");
        assert_eq!(QueueRefusal::Full.reason(), Reason::QueueFull);
        assert_eq!(QueueRefusal::Timeout.reason(), Reason::TimeoutQueue);
    }
}
