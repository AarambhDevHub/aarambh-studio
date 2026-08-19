use std::sync::mpsc::{self, Receiver, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use aarambh_studio_core::{AarambhError, Result};
use aarambh_studio_inference::{
    FinishReason, GenerationConfig, GenerationOutput, GenerationPhase, GenerationSession,
    GenerationStep, InferenceEngine,
};
use aarambh_studio_safety::{
    PiiPolicy, SafeStreamEvent, SafetyPolicy, StreamingSafetyFilter, detect_pii,
};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender, unbounded_channel};

use crate::metrics::ServerMetrics;
use crate::prefix_cache::{PrefixCache, PrefixLookup};

#[derive(Debug, Clone)]
/// Runtime controls for continuous request scheduling.
pub struct BatcherConfig {
    /// Maximum simultaneously active generation sessions.
    pub max_batch_size: usize,
    /// Maximum requests waiting for admission.
    pub queue_capacity: usize,
    /// Idle wait before checking the admission queue again.
    pub batch_wait: Duration,
    /// Maximum prompt tokens processed by one prefill forward pass.
    pub prefill_chunk_size: usize,
}

impl Default for BatcherConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 8,
            queue_capacity: 128,
            batch_wait: Duration::from_millis(2),
            prefill_chunk_size: 128,
        }
    }
}

#[derive(Debug, Clone)]
/// One validated generation request submitted to the scheduler.
pub struct GenerationRequest {
    /// Safety-checked and role-formatted prompt.
    pub prompt: String,
    /// Inference generation controls.
    pub config: GenerationConfig,
    /// Whether output deltas should be emitted before completion.
    pub stream: bool,
}

#[derive(Debug)]
/// Event delivered from the inference worker to one HTTP request.
pub enum GenerationEvent {
    /// Safety-approved text delta.
    Delta(String),
    /// Complete generation result.
    Completed(Box<GenerationOutput>),
    /// Output was stopped by the safety layer.
    SafetyBlocked(String),
    /// Inference failed.
    Failed(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Failure to submit work to the bounded scheduler queue.
pub enum SubmitError {
    /// Queue capacity is exhausted.
    QueueFull,
    /// Inference worker has stopped.
    WorkerStopped,
}

#[derive(Clone)]
/// Cloneable handle used by HTTP handlers to submit generation jobs.
pub struct BatcherHandle {
    sender: SyncSender<Control>,
    metrics: Arc<ServerMetrics>,
    worker: Arc<Mutex<Option<thread::JoinHandle<()>>>>,
}

impl BatcherHandle {
    /// Start a dedicated continuous-batching inference worker.
    pub fn start(
        engine: InferenceEngine,
        config: BatcherConfig,
        safety_policy: Option<SafetyPolicy>,
        metrics: Arc<ServerMetrics>,
    ) -> Result<Self> {
        Self::start_with_prefix_cache(engine, config, safety_policy, metrics, None)
    }

    /// Start a worker with an optional prompt-prefix cache for prefill reuse.
    pub fn start_with_prefix_cache(
        engine: InferenceEngine,
        config: BatcherConfig,
        safety_policy: Option<SafetyPolicy>,
        metrics: Arc<ServerMetrics>,
        prefix_cache: Option<Arc<PrefixCache>>,
    ) -> Result<Self> {
        if config.max_batch_size == 0
            || config.queue_capacity == 0
            || config.prefill_chunk_size == 0
        {
            return Err(AarambhError::Config(
                "batch size and queue capacity must be greater than zero".into(),
            ));
        }
        let (sender, receiver) = mpsc::sync_channel(config.queue_capacity);
        let worker_metrics = metrics.clone();
        let worker = thread::Builder::new()
            .name("aarambh-inference".to_string())
            .spawn(move || {
                run_worker(
                    engine,
                    receiver,
                    config,
                    safety_policy,
                    worker_metrics,
                    prefix_cache,
                )
            })
            .map_err(AarambhError::Io)?;
        Ok(Self {
            sender,
            metrics,
            worker: Arc::new(Mutex::new(Some(worker))),
        })
    }

    /// Submit a request without waiting when the queue is full.
    pub fn submit(
        &self,
        request: GenerationRequest,
    ) -> std::result::Result<UnboundedReceiver<GenerationEvent>, SubmitError> {
        let (tx, rx) = unbounded_channel();
        self.metrics.request_queued();
        match self.sender.try_send(Control::Submit(Job { request, tx })) {
            Ok(()) => Ok(rx),
            Err(TrySendError::Full(_)) => {
                self.metrics.request_queue_rollback();
                self.metrics.request_rejected();
                Err(SubmitError::QueueFull)
            }
            Err(TrySendError::Disconnected(_)) => {
                self.metrics.request_queue_rollback();
                Err(SubmitError::WorkerStopped)
            }
        }
    }

    /// Request worker shutdown and wait for its thread to exit.
    pub fn shutdown(&self) {
        let _ = self.sender.send(Control::Shutdown);
        if let Ok(mut guard) = self.worker.lock()
            && let Some(worker) = guard.take()
        {
            let _ = worker.join();
        }
    }
}

enum Control {
    Submit(Job),
    Shutdown,
}

struct Job {
    request: GenerationRequest,
    tx: UnboundedSender<GenerationEvent>,
}

struct ActiveRequest {
    request: GenerationRequest,
    session: GenerationSession,
    tx: UnboundedSender<GenerationEvent>,
    filter: Option<StreamingSafetyFilter>,
    stop_buffer: StopBuffer,
    visible_text: String,
    retries: usize,
}

impl ActiveRequest {
    fn new(
        request: GenerationRequest,
        session: GenerationSession,
        tx: UnboundedSender<GenerationEvent>,
        policy: Option<&SafetyPolicy>,
    ) -> Self {
        Self {
            request,
            session,
            tx,
            filter: policy.cloned().map(StreamingSafetyFilter::new),
            stop_buffer: StopBuffer::default(),
            visible_text: String::new(),
            retries: 0,
        }
    }

    fn restart(&mut self, engine: &InferenceEngine, policy: Option<&SafetyPolicy>) -> Result<()> {
        self.session = engine.prepare_session(&self.request.prompt, self.request.config.clone())?;
        self.filter = policy.cloned().map(StreamingSafetyFilter::new);
        self.stop_buffer = StopBuffer::default();
        self.visible_text.clear();
        self.retries += 1;
        Ok(())
    }

    fn process_step(&mut self, step: GenerationStep) -> Option<String> {
        let fragments = if step.phase == GenerationPhase::Answer {
            self.stop_buffer.push(
                &step.token_text,
                &self.request.config.stop_sequences,
                self.session.finish_reason() == Some(FinishReason::StopSequence),
            )
        } else {
            vec![step.token_text.clone()]
        };
        for fragment in fragments {
            let mut safe_step = step.clone();
            safe_step.token_text = fragment;
            let events = match &mut self.filter {
                Some(filter) => filter.push_step(&safe_step),
                None if safe_step.phase == GenerationPhase::Answer => {
                    vec![SafeStreamEvent::Text(safe_step.token_text)]
                }
                None => Vec::new(),
            };
            if let Some(reason) = self.apply_safe_events(events) {
                return Some(reason);
            }
        }
        None
    }

    fn finish_filter(&mut self) -> Option<String> {
        for fragment in self.stop_buffer.finish() {
            let step = GenerationStep {
                step: self.session.completion_tokens(),
                token_id: 0,
                token_text: fragment,
                candidates: Vec::new(),
                phase: GenerationPhase::Answer,
                forced: false,
            };
            let events = match &mut self.filter {
                Some(filter) => filter.push_step(&step),
                None => vec![SafeStreamEvent::Text(step.token_text)],
            };
            if let Some(reason) = self.apply_safe_events(events) {
                return Some(reason);
            }
        }
        let events = self
            .filter
            .as_mut()
            .map(StreamingSafetyFilter::finish)
            .unwrap_or_default();
        self.apply_safe_events(events)
    }

    fn apply_safe_events(&mut self, events: Vec<SafeStreamEvent>) -> Option<String> {
        for event in events {
            match event {
                SafeStreamEvent::Text(text) => {
                    self.visible_text.push_str(&text);
                    if self.request.stream && self.tx.send(GenerationEvent::Delta(text)).is_err() {
                        return Some("client disconnected".to_string());
                    }
                }
                SafeStreamEvent::Blocked(reason) => return Some(reason),
            }
        }
        None
    }
}

#[derive(Default)]
struct StopBuffer {
    pending: String,
}

impl StopBuffer {
    fn push(&mut self, fragment: &str, stops: &[String], matched: bool) -> Vec<String> {
        self.pending.push_str(fragment);
        if matched {
            if let Some(stop_len) = stops
                .iter()
                .filter(|stop| self.pending.ends_with(stop.as_str()))
                .map(String::len)
                .max()
            {
                self.pending.truncate(self.pending.len() - stop_len);
            }
            return self.finish();
        }
        let hold = stops
            .iter()
            .map(|stop| longest_suffix_prefix(&self.pending, stop))
            .max()
            .unwrap_or(0);
        let emit_len = self.pending.len().saturating_sub(hold);
        if emit_len == 0 {
            return Vec::new();
        }
        let suffix = self.pending.split_off(emit_len);
        let text = std::mem::replace(&mut self.pending, suffix);
        vec![text]
    }

    fn finish(&mut self) -> Vec<String> {
        if self.pending.is_empty() {
            Vec::new()
        } else {
            vec![std::mem::take(&mut self.pending)]
        }
    }
}

fn longest_suffix_prefix(text: &str, stop: &str) -> usize {
    let max = text.len().min(stop.len().saturating_sub(1));
    (1..=max)
        .rev()
        .find(|&len| {
            text.is_char_boundary(text.len() - len)
                && stop.is_char_boundary(len)
                && text[text.len() - len..] == stop[..len]
        })
        .unwrap_or(0)
}

fn run_worker(
    engine: InferenceEngine,
    receiver: Receiver<Control>,
    config: BatcherConfig,
    safety_policy: Option<SafetyPolicy>,
    metrics: Arc<ServerMetrics>,
    prefix_cache: Option<Arc<PrefixCache>>,
) {
    let mut active = Vec::<ActiveRequest>::new();
    let mut shutting_down = false;
    loop {
        while !shutting_down && active.len() < config.max_batch_size {
            match receiver.try_recv() {
                Ok(Control::Submit(job)) => admit_job(
                    &engine,
                    job,
                    safety_policy.as_ref(),
                    &metrics,
                    &mut active,
                    config.prefill_chunk_size,
                    prefix_cache.as_ref(),
                ),
                Ok(Control::Shutdown) => shutting_down = true,
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    shutting_down = true;
                    break;
                }
            }
        }

        if active.is_empty() {
            if shutting_down {
                break;
            }
            match receiver.recv_timeout(config.batch_wait) {
                Ok(Control::Submit(job)) => admit_job(
                    &engine,
                    job,
                    safety_policy.as_ref(),
                    &metrics,
                    &mut active,
                    config.prefill_chunk_size,
                    prefix_cache.as_ref(),
                ),
                Ok(Control::Shutdown) => shutting_down = true,
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => shutting_down = true,
            }
            continue;
        }

        let mut index = 0;
        while index < active.len() {
            if active[index].tx.is_closed() {
                active.swap_remove(index);
                metrics.request_cancelled();
                continue;
            }
            let step = match active[index].session.advance(engine.tokenizer()) {
                Ok(step) => step,
                Err(error) => {
                    let request = active.swap_remove(index);
                    let _ = request.tx.send(GenerationEvent::Failed(error.to_string()));
                    metrics.inference_error();
                    metrics.request_completed();
                    continue;
                }
            };
            if let Some(step) = step {
                metrics.generated_token();
                if let Some(reason) = active[index].process_step(step) {
                    if reason == "client disconnected" {
                        active.swap_remove(index);
                        metrics.request_cancelled();
                        continue;
                    }
                    if retry_or_block(
                        &engine,
                        &mut active[index],
                        safety_policy.as_ref(),
                        &metrics,
                    ) {
                        index += 1;
                    } else {
                        active.swap_remove(index);
                        metrics.request_completed();
                    }
                    continue;
                }
            }
            if active[index].session.is_finished() {
                let mut request = active.swap_remove(index);
                if let Some(_reason) = request.finish_filter() {
                    if retry_or_block(&engine, &mut request, safety_policy.as_ref(), &metrics) {
                        active.push(request);
                    } else {
                        metrics.request_completed();
                    }
                    continue;
                }
                match request.session.into_output() {
                    Ok(mut output) => {
                        if let Some(reason) = structured_pii_block(&output, safety_policy.as_ref())
                        {
                            let _ = request.tx.send(GenerationEvent::SafetyBlocked(reason));
                            metrics.safety_blocked();
                        } else {
                            if output.tool_call.is_none() {
                                output.text = request.visible_text.clone();
                                output.answer_text = request.visible_text;
                            }
                            let _ = request
                                .tx
                                .send(GenerationEvent::Completed(Box::new(output)));
                        }
                    }
                    Err(error) => {
                        let _ = request.tx.send(GenerationEvent::Failed(error.to_string()));
                        metrics.inference_error();
                    }
                }
                metrics.request_completed();
                continue;
            }
            index += 1;
        }

        if !active.is_empty() {
            let mut sessions = active
                .iter_mut()
                .map(|request| &mut request.session)
                .collect::<Vec<_>>();
            metrics.decode_batch(sessions.len());
            if let Err(error) = engine.decode_sessions(&mut sessions) {
                metrics.inference_error();
                for request in active.drain(..) {
                    let _ = request.tx.send(GenerationEvent::Failed(error.to_string()));
                    metrics.request_completed();
                }
            }
        }
    }

    for request in active {
        let _ = request
            .tx
            .send(GenerationEvent::Failed("server shutting down".to_string()));
        metrics.request_completed();
    }
}

fn admit_job(
    engine: &InferenceEngine,
    job: Job,
    policy: Option<&SafetyPolicy>,
    metrics: &ServerMetrics,
    active: &mut Vec<ActiveRequest>,
    prefill_chunk_size: usize,
    prefix_cache: Option<&Arc<PrefixCache>>,
) {
    metrics.request_admitted();
    let lookup_metrics = metrics;
    let lookup = |prompt_ids: &[u32]| -> Option<(aarambh_studio_inference::KvCache, usize)> {
        let cache = prefix_cache?;
        match cache.lookup(prompt_ids) {
            PrefixLookup::Hit { cache, matched_len } => {
                lookup_metrics.prefix_cache_hit(matched_len as u64);
                Some((cache, matched_len))
            }
            PrefixLookup::Miss => {
                lookup_metrics.prefix_cache_miss();
                None
            }
        }
    };
    let store = |prompt_ids: &[u32], cache: &aarambh_studio_inference::KvCache| {
        if let Some(store_cache) = prefix_cache {
            store_cache.store(prompt_ids, cache);
        }
    };
    match engine.prepare_session_with_prefix_cache(
        &job.request.prompt,
        job.request.config.clone(),
        prefill_chunk_size,
        lookup,
        store,
    ) {
        Ok(session) => active.push(ActiveRequest::new(job.request, session, job.tx, policy)),
        Err(error) => {
            let _ = job.tx.send(GenerationEvent::Failed(error.to_string()));
            metrics.inference_error();
            metrics.request_completed();
        }
    }
}

fn retry_or_block(
    engine: &InferenceEngine,
    request: &mut ActiveRequest,
    policy: Option<&SafetyPolicy>,
    metrics: &ServerMetrics,
) -> bool {
    let max_regenerations = policy.map(|policy| policy.max_regenerations).unwrap_or(0);
    if !request.request.stream
        && request.retries < max_regenerations
        && request.restart(engine, policy).is_ok()
    {
        return true;
    }
    let _ = request.tx.send(GenerationEvent::SafetyBlocked(
        "output blocked by safety".to_string(),
    ));
    metrics.safety_blocked();
    false
}

fn structured_pii_block(
    output: &GenerationOutput,
    policy: Option<&SafetyPolicy>,
) -> Option<String> {
    let policy = policy?;
    if output.tool_call.is_none() || policy.output_pii == PiiPolicy::Off {
        return None;
    }
    let findings = detect_pii(&output.text);
    if findings.is_empty() || policy.output_pii == PiiPolicy::Warn {
        None
    } else {
        Some("tool call contains PII and cannot be safely redacted".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_buffer_does_not_release_partial_stop() {
        let mut buffer = StopBuffer::default();
        let stops = vec!["STOP".to_string()];
        assert_eq!(buffer.push("hello ST", &stops, false), ["hello "]);
        assert!(buffer.push("OP", &stops, true).is_empty());
    }

    #[test]
    fn suffix_prefix_handles_utf8_boundaries() {
        assert_eq!(longest_suffix_prefix("hello ", "stop"), 0);
        assert_eq!(longest_suffix_prefix("hello st", "stop"), 2);
    }
}
