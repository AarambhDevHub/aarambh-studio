# SELF_LEARNING_V4.md — aarambh-studio v4.0

> From first principles. From zero. From Rust.
>
> Companion to `SELF_LEARNING.md`, `SELF_LEARNING_V2.md`, and
> `SELF_LEARNING_V3.md`. This document covers **only what v4.0 adds** —
> how MLA, audio, sparse MoE dispatch, test-time compute scaling,
> RLAIF, sandboxed tool execution, multi-agent orchestration, and RAG
> each interact with the self-learning loop. Sections continue
> numbering from v3's Section 39. Everything in the three prior
> documents (Online GRPO, Self-Critique, Experience Replay, CPU vs GPU
> mode, catastrophic forgetting protection, forgetting diagnostics,
> vision/video/document grounding, chain-aware replay) is unchanged and
> continues to work exactly as documented for every session type it
> already covers. v4.0 is the **final planned version** — see §55.

---

## Table of Contents

40. [Why This Is a Separate Document](#40-why-this-is-a-separate-document)
41. [What Changes, What Doesn't](#41-what-changes-what-doesnt)
42. [Self-Learning on the MLA Attention Stack](#42-self-learning-on-the-mla-attention-stack)
43. [Audio-Grounded Verification](#43-audio-grounded-verification)
44. [Self-Learning With Sparse MoE Dispatch](#44-self-learning-with-sparse-moe-dispatch)
45. [Test-Time Compute Scaling Inside Self-Learning](#45-test-time-compute-scaling-inside-self-learning)
46. [RLAIF Inside the Self-Learning Loop](#46-rlaif-inside-the-self-learning-loop)
47. [Self-Learning With Sandboxed Tool Execution](#47-self-learning-with-sandboxed-tool-execution)
48. [Self-Learning With Multi-Agent Orchestration](#48-self-learning-with-multi-agent-orchestration)
49. [RAG-Grounded Self-Learning](#49-rag-grounded-self-learning)
50. [Hardware Gating (v4 Additions)](#50-hardware-gating-v4-additions)
51. [Full Loop Flow (v4, All Modalities and Capabilities)](#51-full-loop-flow-v4-all-modalities-and-capabilities)
52. [CLI Commands (v4)](#52-cli-commands-v4)
53. [Crate Structure Additions](#53-crate-structure-additions)
54. [What to Expect (v4)](#54-what-to-expect-v4)
55. [Context Management, Chat-Template Compatibility, and Red-Team Coverage in Self-Learning Sessions](#55-context-management-chat-template-compatibility-and-red-team-coverage-in-self-learning-sessions)
56. [Known Limitations (v4-Specific) and Closing Note](#56-known-limitations-v4-specific-and-closing-note)

---

## 40. Why This Is a Separate Document

v4.0's architecture changes (`ARCHITECTURE_V4.md` §55–65) touch the
self-learning loop for reasons distinct from why v2 and v3 each needed
their own pass:

- **Execution changes what a "turn" can do.** Every self-learning
  design through v3 assumed a turn's model output was text, or text
  plus a tool-call *request*. v4 §61 means a turn can now trigger real
  execution with real side effects (within the sandbox). The
  self-learning loop's verification step (§43, §47 below) has to
  account for the fact that a turn's "result" may now include the
  output of something that actually ran, not just something the model
  said it wanted to run.
- **Multiple attention kinds change what "the model" means during a
  session.** With MLA (v4 §55) joining Gated DeltaNet and DSA (v3), a
  single checkpoint's forward pass may now route through three
  different attention mechanisms depending on layer index. Nothing
  about self-learning's math changes because of this — see §42 for why
  — but it is worth stating explicitly rather than leaving it implicit.
- **Test-time compute scaling and RLAIF are new signal sources that sit
  adjacent to, not inside, the existing online-GRPO loop**, and the
  question of whether/how they interact with self-learning specifically
  (as opposed to offline training) needed its own answer (§45, §46).
- **This is the final version.** Unlike v2 and v3, this document does
  not need to reason about what a hypothetical next version might touch
  — see §55's closing note.

## 41. What Changes, What Doesn't

**Unchanged, and verified unchanged by regression tests:**
- Text-only self-learning on i3 — byte-for-byte identical to v1.
- Image-grounded self-learning on Kaggle — byte-for-byte identical to
  v2.
- Video- and document-grounded self-learning on Kaggle — byte-for-byte
  identical to v3.
- Forgetting diagnostics (v3 §27), the Manas export schema (v3 §28),
  MoE routing-drift diagnostics (v3 §31), and chain-aware replay
  (v3 §33) — all unchanged in mechanism; §44 and §48 below extend their
  *coverage*, not their underlying math.
- Existing `replay_buffer.jsonl`, `replay_buffer_v2.jsonl`, and v3's
  replay schema all load exactly as before. No migration step is
  required for any prior session type moving to v4.0.

**New, additive only:**
- MLA-aware replay sampling is identical to Full/GatedDeltaNet-aware
  replay — no new field, no new behaviour (§42 explains why).
- Audio-grounded verification, following the exact
  verifier-vs-self-critique split v2 §17 established (§43).
- Routing-drift diagnostics extended to cover sparse-dispatch sessions
  (§44).
- An optional record of which test-time-scaling selection strategy (if
  any) produced a given replay entry's completion (§45).
- RLAIF-sourced replay entries carry a provenance marker distinguishing
  them from GRPO- or DPO-sourced entries (§46).
- Execution-result-aware replay entries, extending v3 §29's
  `tool-use` topic to cover entries where a tool call was actually
  executed rather than only proposed (§47).
- Orchestrator-level replay entries alongside per-sub-agent entries,
  extending v3 §33's chain-aware pattern one level further (§48).
- An optional `retrieved_context_ref` field on replay entries, following
  the exact caching-reference pattern v2 §17 established for
  `image_ref` (§49).

## 42. Self-Learning on the MLA Attention Stack

The short version: **nothing changes, and that is the point.**

> **Status: Verified for v4.0.0-alpha.1 (Phase 41).** The
> `mla_training_backward_reaches_mla_parameters` test confirms gradients reach
> the MLA down-projection (`kv_a_proj`), latent norm, value up-projection
> (`up_v`), and output projection (`o_proj`) — the §42 reachability argument.
> MLA's Q/K gradient behaviour on the CPU candle-fallback attention path
> matches the existing GQA path (full Q/K gradients flow under the CUDA/flash
> path used in real training); MLA is wired into the identical attention path,
> so self-learning's gradient orthogonalisation reaches MLA weights
> consistently with every other attention kind.

Online GRPO's math (`SELF_LEARNING.md` §5) operates on log-probabilities
of generated tokens under the current policy. It has no dependency on
*how* those log-probabilities were computed internally — whether a
given layer used Full attention, Gated DeltaNet's recurrent state, or
MLA's compressed-latent reconstruction is entirely opaque to the
training loop above it. This is the same reasoning
`SELF_LEARNING_V3.md` §30 already established for Gated DeltaNet and
DSA, extended to a third attention kind without needing a new argument.

**What is worth verifying, and is covered by a regression test below:**
gradient orthogonalisation (`SELF_LEARNING.md` §8) must correctly reach
MLA's down/up-projection weights the same way it reaches every other
trainable weight — an MLA layer is not exempt from the anti-forgetting
defence just because it is new.

## 43. Audio-Grounded Verification

Follows the exact split `SELF_LEARNING_V2.md` §17 established for
vision and `SELF_LEARNING_V3.md` §32 extended to video and documents:
checkable audio questions (e.g. "what word was spoken at this
timestamp," for a task type where the underlying free/public dataset
provides a ground-truth transcript) route to a grounded verifier that
checks against the actual audio content; open-ended audio questions
("describe the mood of this clip") fall back to self-critique, with the
same noise caveats every self-critique path has carried since v1.

Reuses `SELF_LEARNING_V2.md` §17's cached-token reference pattern
directly — a new `audio_ref` field, following the identical shape as
`image_ref`/`video_ref`/`document_ref`. No new verification
architecture; the existing verifier-vs-critique split is simply applied
to a fourth modality.

## 44. Self-Learning With Sparse MoE Dispatch

> **Status: Shipped for v4.0.0-alpha.3 (Phase 43).** The
> `DispatchKind` enum and `sparse_grouped_dispatch` are implemented
> (`aarambh-studio-core`, `aarambh-studio-nn`); the aux-loss-
> dispatch-independence invariant this section relies on is verified by
> the `load_balancing_loss_value_is_unaffected_by_dispatch_kind` unit
> test. See `docs/phase43_sparse_moe.md`.

`ARCHITECTURE_V4.md` §57 established that `DispatchKind` only changes
the *compute path* of an MoE forward pass, never the router's training
objective. This holds for the self-learning loop as well: an online
GRPO update batch trains the router identically regardless of whether
the forward pass that produced the sampled completion used
`DenseMasked` or `Sparse` dispatch.

**What does extend:** `SELF_LEARNING_V3.md` §31's routing-drift
diagnostic — which tracks which experts activate for replayed/probed
examples to detect routing drift specifically — now also records which
`DispatchKind` was active during a given probe, purely so that a
routing-drift flag can be correctly attributed rather than mistakenly
correlated with a dispatch-path change that has no bearing on the
router's actual learned behaviour.

## 45. Test-Time Compute Scaling Inside Self-Learning

> **Status: Verified for v4.0.0-alpha.5 (Phase 45).** The
> `SelectionStrategy` enum and the `BestOfNEngine` wrapper are implemented
> (`aarambh-studio-inference`: `best_of_n.rs`, `self_consistency.rs`,
> `process_reward.rs`); the eval harness records single-sample vs best-of-N
> accuracy deltas in the scorecard's `details` map via
> `EvalConfig.best_of_n`. A session may pass a `SelectionStrategy` to the
> self-learning loop's own N-completion sampling — the loop's replay-entry
> metadata field that records the strategy is the instrumentation this
> section describes, left as an open question rather than an asserted
> learning-outcome claim. See `docs/phase45_test_time.md`.

Test-time compute scaling (`ARCHITECTURE_V4.md` §59) is fundamentally
an *inference-time* technique — generate N candidates, select one. The
self-learning loop's existing N-completion sampling
(`SELF_LEARNING.md` §12) already generates multiple candidates per
turn, for a different purpose (feeding the replay buffer with a spread
of attempts). v4.0 does not merge these two mechanisms into one — they
remain conceptually distinct — but a session may optionally use
test-time-scaling's `SelectionStrategy` (v4 §59) to choose which of the
self-learning loop's own N completions gets scored and replayed, rather
than defaulting to the highest-raw-score completion alone.

When this option is enabled, the replay entry's metadata records which
`SelectionStrategy` (if any) was used, purely for post-hoc analysis of
whether selection-aware replay produces measurably different learning
outcomes than the existing default — an open question this document
does not claim to have answered, only instrumented.

## 46. RLAIF Inside the Self-Learning Loop

> **Status: Verified for v4.0.0-alpha.6 (Phase 46).** The `RlaifConfig`,
> `JudgeGenerator`/`CandidateSampler` traits, position-swap bias correction,
> and `(chosen, rejected)` DPO-schema output are implemented
> (`aarambh-studio-finetune`: `rlaif.rs`); the `finetune rlaif` CLI
> subcommand wires policy + judge `InferenceEngine`s. The generated pairs
> feed into the unmodified `finetune dpo` pipeline. See `docs/phase46_rlaif.md`.

RLAIF (`ARCHITECTURE_V4.md` §60) was designed as an **offline**
data-generation front end for the existing DPO pipeline — a judge model
scores self-sampled pairs, producing (chosen, rejected) data fed into
`finetune dpo`. This is deliberately kept separate from the *online*
self-learning loop's own GRPO-based update mechanism
(`SELF_LEARNING.md` §5): the two operate on different training
objectives (DPO's pairwise preference loss vs. GRPO's group-relative
policy loss) and are not merged into a single online RLAIF-in-the-loop
mechanism in v4.0.

What v4.0 *does* add: a replay entry that was scored by the existing
self-critique path (`SELF_LEARNING.md` §7) may optionally carry a
`provenance: "self_critique" | "rlaif_judge"` marker if an RLAIF judge
pass was separately run over the same session's completions offline —
purely so that anyone analysing the replay buffer afterward can
distinguish which scoring mechanism produced a given entry's score.
This is a labelling addition, not a change to how scores feed into
online GRPO's math.

## 47. Self-Learning With Sandboxed Tool Execution

This is the section with the most genuinely new content in v4.0,
because Phase 47's execution capability (`ARCHITECTURE_V4.md` §61)
changes what a self-learning turn's "result" can be.

**Before v4.0** (v2 §26, v3 §46): a tool-use replay entry recorded a
proposed tool call and, if the developer's own integration executed it
and fed the result back, the *text* of that result. The self-learning
loop never executed anything itself.

**v4.0:** when sandboxed execution (v4 §61) is enabled for a
self-learning session, the loop can trigger execution directly, subject
to the exact same closed-world allowlist, authorization, timeout, and
resource-ceiling boundaries `ARCHITECTURE_V4.md` §61 defines for any
other execution path — self-learning sessions are not a privileged or
exempted execution context.

```
Self-learning turn proposes a tool call (grammar-constrained JSON,
v2 §30)
        │
        ▼
Same closed-world allowlist + authorization check as any other
execution path (v4 §61) — a self-learning session has NO elevated
privilege here
        │
        ▼
Execution result (success or a recorded refusal/timeout/error) becomes
part of the turn's outcome
        │
        ▼
Verification: for execution results with a checkable expected outcome
(e.g. a whitelisted lookup with a known-correct answer in the
self-learning corpus), route to a grounded verifier; otherwise,
self-critique
        │
        ▼
Replay entry (topic: "tool-use", extended per v3 §29's schema) records
whether the entry involved actual execution or remained emit-only,
via an `executed: bool` field
```

**A refusal is a valid, informative outcome, not an error to discard.**
If a self-learning session proposes an unauthorized or unlisted tool
call, the resulting hard refusal (v4 §61) is itself recorded as a
low-scored replay entry — the model gets signal that the proposal was
inappropriate, the same way any other incorrect proposal would be
scored low and replayed, rather than the turn being silently dropped
from the loop.

> **Implemented in v4.0.0-alpha.7.** `crates/aarambh-studio-agent/src/sandbox.rs`
> ships the closed-world `ToolExecutor` trait, `ToolSandbox`, and
> `SandboxedToolProvider` (a `ToolResultProvider`), so a self-learning
> session that enables sandboxed execution uses the exact same allowlist,
> authorization, timeout, and resource-ceiling boundaries as any other
> execution path — it is not a privileged or exempted context.

## 48. Self-Learning With Multi-Agent Orchestration

Extends v3 §33's chain-aware replay pattern one level further. A
multi-agent orchestration session (`ARCHITECTURE_V4.md` §62) produces
both an orchestrator-level outcome and multiple sub-agent-level
outcomes. v4.0's answer, consistent with v3 §33's existing logic:
**each sub-agent's chain contributes its own replay entries** (per
v3 §33's existing per-step, chain-aware weighting, unchanged), and the
**orchestrator's own delegation and aggregation reasoning** contributes
a separate, additional replay entry of its own — scored on whether the
overall delegated task succeeded, using the same score-propagation
logic v3 §33 already established, applied at one additional level of
nesting rather than as a new mechanism.

The three hard bounds from `ARCHITECTURE_V4.md` §62 (max sub-agent
count, max total execution time, sandbox-scope containment) apply
identically inside a self-learning session as they do in any other
orchestration context — self-learning does not relax them.

## 49. RAG-Grounded Self-Learning

When a self-learning session has retrieval enabled (`ARCHITECTURE_V4.md`
§63), a turn's prompt may include retrieved context spliced in ahead of
the user's question, exactly as any other RAG-augmented generation
would. This section documents one addition: an optional
`retrieved_context_ref` field on the replay entry, following the exact
caching-reference pattern `SELF_LEARNING_V2.md` §17 established for
`image_ref` — the retrieved chunks themselves are not duplicated into
the replay buffer, only referenced, keeping the buffer's storage
footprint the same class of cost it has always been.

Verification for RAG-grounded turns follows the existing split:
questions checkable against the retrieved source content route to a
grounded verifier (comparing the answer against the actual retrieved
text, the same principle `SELF_LEARNING_V3.md` §32 used for
video/document grounding); open-ended questions fall back to
self-critique.

## 50. Hardware Gating (v4 Additions)

Extending `SELF_LEARNING_V3.md` §34's gating table:

| Session type | Hardware | Reasoning |
|---|---|---|
| Text-only, image-only, video/document-grounded (v1–v3, unchanged) | per v3 §34 | unchanged |
| MLA-retrofit-checkpoint sessions (v4, new) | i3 or Kaggle, same as the underlying checkpoint's other requirements | MLA changes cache footprint, not the hardware class a session needs — a session's gate follows its modality/execution requirements exactly as before, not its attention kind |
| Audio-grounded (v4, new) | Kaggle only | Frozen audio-encoder forward pass per turn — same cost class as image/video/document grounding |
| Sparse-MoE-dispatch sessions (v4, new) | Kaggle only for `Sparse` dispatch to actually engage; i3 sessions silently use `DenseMasked` per `ARCHITECTURE_V4.md` §57's documented fallback | Consistent with the architecture doc — not a new gate, a restatement of the existing dispatch-kind fallback rule as it applies to self-learning specifically |
| Test-time compute scaling, small N (v4, new) | i3 | Follows the existing v1 §12 N-completion CPU-safe budget |
| Test-time compute scaling, larger N (v4, new) | Kaggle | Cost scales with N |
| RLAIF judge passes (v4, new, offline) | Kaggle | Judge-model inference at self-sampling scale, same cost class as any Large-scale inference workload |
| Sandboxed tool execution, text/tool-result only (v4, new) | i3 | Lightweight orchestration overhead, same as v3 §34's existing tool-use rule |
| Sandboxed tool execution, multimodal results (v4, new) | Kaggle only | Inherits the existing vision/video/document/audio gate the moment any executed tool's result is non-text |
| Multi-agent orchestration (v4, new) | i3 (orchestration) + Kaggle (any multimodal sub-chain) | Same inheritance rule as execution, applied recursively to sub-agents |
| RAG-grounded sessions (v4, new) | i3 | Embedding head and index are CPU-capable by design |

Same discipline as every prior version: a session refuses to start on
hardware it is not gated for, with a clear error message, rather than
silently degrading or producing misleading results.

## 51. Full Loop Flow (v4, All Modalities and Capabilities)

```
User turn arrives (text, and optionally image/video/document/audio
input, and optionally as part of a multi-step tool chain, and
optionally with retrieval or orchestration enabled)
        │
        ▼
Hardware gate check (§50) — refuses cleanly if session type exceeds
the current machine's capability
        │
        ▼
[If RAG enabled] Retrieve context (§49), splice into prompt
        │
        ▼
Model generates a response — hybrid attention stack including MLA
(§42), MoE routing with dispatch-kind-agnostic training (§44), MTP
heads available as speculative-decode draft (v3 §41), optionally
selected via test-time-scaling strategy (§45)
        │
        ▼
[If tool call proposed] Grammar-constrained JSON validated (v2 §30) ->
sandboxed execution check (§47) -> execute or hard-refuse
        │
        ▼
[If orchestration enabled] Delegate to sub-agent chains (§48), each
independently gated and sandboxed, results aggregated
        │
        ▼
Verification: checkable question -> grounded verifier (text: v1 §7,
vision: v2 §17, video/document: v3 §32, audio: §43, RAG-grounded: §49)
              open-ended -> self-critique (all modalities, same noise
              caveats documented since v1)
        │
        ▼
Score -> replay buffer entry (v3 §29's schema, extended with
`executed`, `provenance`, `retrieved_context_ref` per this document;
chain-aware per v3 §33, orchestrator-aware per §48 where applicable)
        │
        ▼
Online GRPO update batch (SELF_LEARNING.md §5, gradient
orthogonalisation per §8, attention-kind-agnostic per §42,
dispatch-kind-agnostic per §44)
        │
        ▼
Forgetting diagnostics run (v3 §27): capability probes,
forgetting_delta(), routing-drift check (v3 §31, dispatch-kind-aware
per §44)
        │
        ▼
Session summary: scores, any flagged forgetting deltas, any flagged
routing drift, any recorded execution refusals — surfaced to you, not
silently swallowed
```

## 52. CLI Commands (v4)

```
[ ] aarambh-studio selflearn --config <cfg> --audio --hardware kaggle
[ ] aarambh-studio selflearn --config <cfg> --rag --index my_index/
[ ] aarambh-studio selflearn --config <cfg> --agent --tools tools.json \
      --execute --sandbox-config sandbox.toml --max-steps 8
[ ] aarambh-studio selflearn --config <cfg> --orchestrate \
      --max-sub-agents 4 --max-total-seconds 300
[ ] aarambh-studio selflearn --config <cfg> --best-of-n 4 \
      --selection self-consistency
[ ] aarambh-studio selflearn --config <cfg> --forgetting-report
      # unchanged from v3 §36 — still runs a standalone diagnostic
      # pass without a live session
```

## 53. Crate Structure Additions

```
crates/aarambh-studio-selflearn/
└── src/
    ├── ...v1/v2/v3 modules unchanged (online_grpo.rs, critique.rs,
    │      replay.rs, vision_cache.rs, vision_verifier.rs, gating.rs,
    │      forgetting_hook.rs, routing_drift.rs, video_verifier.rs,
    │      document_verifier.rs, chain_replay.rs)...
    ├── audio_verifier.rs        ← extends vision_verifier.rs's split
    │                              for checkable audio-QA types (§43)
    ├── execution_replay.rs      ← replay entry construction for
    │                              executed (not just proposed) tool
    │                              calls, including refusals (§47)
    ├── orchestrator_replay.rs   ← orchestrator-level replay entries
    │                              alongside sub-agent chain_replay.rs
    │                              entries (§48)
    ├── rag_cache.rs             ← retrieved_context_ref caching
    │                              pattern, mirrors vision_cache.rs (§49)
    └── selection_metadata.rs    ← optional SelectionStrategy provenance
                                    tagging on replay entries (§45)
```

All new modules are additive to the existing crate — no existing v1/v2/
v3 module is renamed, removed, or restructured. This is the sixth
version in a row this crate has grown this way.

## 54. What to Expect (v4)

- Text-only, image-only, video-only, and document-only sessions behave
  byte-for-byte as before — this document changes nothing about them.
- Audio-grounded sessions are Kaggle-only, following the same reasoning
  every prior modality established.
- MLA and sparse MoE dispatch are transparent to the self-learning
  loop's training math — you should not expect to see any behavioural
  difference in online GRPO's update quality attributable to these
  changes, only to the underlying model's improved efficiency.
- Sandboxed execution inside a self-learning session is off by default
  and requires explicit operator authorization, identical to any other
  execution context (v4 §61) — a self-learning session does not get an
  easier path to execution than any other integration.
- Refusals during execution are recorded and scored, not discarded —
  expect to see them in the replay buffer and in session summaries.
- Multi-agent self-learning sessions will take measurably longer per
  turn than single-agent sessions, bounded by the configured
  `--max-sub-agents` and `--max-total-seconds` ceilings — this is
  expected, not a performance regression to chase.

## 55. Context Management, Chat-Template Compatibility, and Red-Team Coverage in Self-Learning Sessions

> **Status: shipped in v4.0.0-alpha.12 (Phase 52).** The session-start
> `chat_template_version` gate is wired into `SelfLearnLoop::from_paths`; a
> mismatch refuses to start the session. Self-learning sessions default to
> `Reject` context policy.

Three retrofit additions from `ARCHITECTURE_V4.md` §66–67 apply to
self-learning sessions specifically, covered here rather than folded
silently into an existing section since none of the three change the
loop's core math — only what happens at its edges.

**Chat-template-version check at session start.** A self-learning
session refuses to start if the checkpoint's `chat_template_version`
(§66) does not match what the session's configured replay buffer schema
expects — the same fail-loud-not-silent discipline the hardware gates
(§50) already follow, applied to template compatibility instead of
hardware capability. This matters specifically for self-learning
because a session that ran for hours against a mismatched template
would produce replay entries built on a misinterpreted prompt
structure, silently corrupting the buffer rather than failing at the
one point (session start) where catching it is cheap.

**Context-truncation policy applies unchanged.** Long self-learning
sessions — particularly ones using multi-step tool chains (v3 §33) or
orchestration (§48 above) — are exactly the kind of long-running,
many-turn sessions `ARCHITECTURE_V4.md` §66's `ContextTruncationPolicy`
was written for. Self-learning sessions default to `Reject` rather than
`SlidingWindow` or `Summarize` specifically because silently dropping
context mid-session would mean scoring and replaying turns whose
verifier no longer has the full picture it originally reasoned
over — the same "fail loud, not silent" instinct that governs every
other edge case in this document.

**Self-learning is in scope for red-team evaluation, not exempt from
it.** `ARCHITECTURE_V4.md` §67's adversarial corpus includes cases that
specifically target the self-learning loop's own replay-scoring and
execution paths (e.g. attempting to get a self-learning session to
execute an unauthorized tool via a crafted turn, or attempting to
inject content that manipulates the self-critique score rather than
answering honestly). A self-learning session does not get a lighter
adversarial bar than any other execution context in the project.

## 56. Known Limitations (v4-Specific) and Closing Note

- Selection-strategy-aware replay (§45) is instrumented but its actual
  effect on learning outcomes versus the existing default is an open
  question this document does not claim to have answered — it is
  logged for analysis, not presented as a proven improvement.
- The RLAIF/self-critique provenance marker (§46) is a labelling
  addition only; it does not change how a score of a given magnitude
  affects the online GRPO update regardless of which mechanism produced
  it.
- Fine-grained per-step credit assignment inside orchestrated
  multi-agent chains (§48) is not attempted — orchestrator-level and
  sub-agent-level entries are scored independently, using the same
  chain-outcome-weighting logic v3 §33 already established, not a new
  finer-grained mechanism.
- As with every prior version, routing-drift and forgetting diagnostics
  detect *that* something changed, not definitively *why* —
  distinguishing legitimate learning from actual forgetting remains a
  judgment call informed by the diagnostic, not automated by it.

**Closing note.** This is the fourth and final self-learning companion
document planned for aarambh-studio. `ARCHITECTURE_V4.md` §69 explains the
reasoning for why v4.0 is the project's final planned version in full;
the same reasoning applies here. Every mechanism documented across
`SELF_LEARNING.md`, `SELF_LEARNING_V2.md`, `SELF_LEARNING_V3.md`, and
this document continues to work exactly as specified — there is no
planned v5 that would supersede or extend it further. Known limitations
listed above and in each prior document's own limitations section
remain the honest, final word on what this self-learning loop does and
does not do.
