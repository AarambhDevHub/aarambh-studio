# Sampling Defaults

> **Phase 52 reference** (`ARCHITECTURE_V4.md` §66). One canonical answer to
> "what sampling settings should I use?", consolidating guidance that was
> previously scattered informally across `ARCHITECTURE.md`,
> `ARCHITECTURE_V2.md`, `ARCHITECTURE_V3.md`, and `ROADMAP*.md`.

The Aarambh-studio sampler (`aarambh-studio-inference::Sampler`) supports
temperature, top-k, top-p (nucleus), greedy, and seed-based reproducibility.
This document is the single source of truth for recommended defaults per use
case. All values are starting points — measure on your own model and adjust.

## Canonical reference table

| Use case | Temperature | Top-p | Top-k | Greedy | Rationale |
|---|---|---|---|---|---|
| **Deterministic tool-call generation** (Phase 26 / 47 / 48) | `0.0` (greedy) | — | — | yes | Tool-call JSON must be deterministic and parseable. Any stochasticity risks a malformed call that the closed-world allowlist then refuses. Greedy is mandatory here, not a preference. |
| **Math / code verification** (Phase 7 thinking, Phase 39 max-thinking) | `0.0`–`0.1` | `1.0` | — | prefer greedy | A single wrong token in a derivation or a program is a wrong answer. Disable stochasticity; let the thinking budget carry the reasoning, not the sampler. |
| **Open-ended chat** (server `/v1/chat/completions`, default) | `0.7` | `0.9` | unset | no | Balanced: coherent and grounded, with enough variety to avoid loops. This is the server default (`ServeConfig::default()` resolves to the thinking-mode default sampler; see below). |
| **Creative writing** | `0.9`–`1.1` | `0.95` | unset | no | Higher temperature for novelty; top-p keeps the long tail from going incoherent. Above ~1.2 the model degrades to word salad on these scales — measure before exceeding. |
| **Self-consistency / best-of-N** (Phase 45) | `0.7`–`0.9` | `0.95` | `50` | no | Each sample must be diverse enough that majority-vote / verifier selection is meaningful, but each individual sample must still be coherent. Top-k=50 bounds the tail. |
| **Self-learning / online GRPO rollout** (Phase 12 / 46) | `0.8` | `0.95` | unset | no | Rollouts need exploration breadth for the verifier to discriminate, but each rollout must remain a plausible completion to score. |

## Thinking-mode defaults

The server resolves sampling defaults from the request's `reasoning_effort`
(`ARCHITECTURE_V3.md` §48.3) when the caller omits `temperature`/`top_p`.
Explicit request parameters are **never** overridden.

| `reasoning_effort` | Default temperature | Default top-p |
|---|---|---|
| `none` (no thinking) | `0.7` | `0.9` |
| `low` / `medium` / `max` | `0.7` | `0.9` |

Thinking modes do not raise the temperature — the reasoning happens inside the thinking block, not via noisier sampling. A hotter sampler would make the
thinking less reliable, not more.
the thinking less reliable, not more.

## When to deviate

- **Reproducibility.** Always pass a `seed` (the sampler is seed-based) when you
  need a bit-exact replay — eval harness runs, regression baselines, red-team
  case replays. Two runs with the same seed, prompt, and sampler config produce
  identical output.
- **Context-policy interaction.** Under `ContextTruncationPolicy::Reject`
  (Phase 52), a session that would exceed the context window errors loudly
  rather than silently dropping turns. Do not compensate by lowering
  `max_tokens` to "fit" — the policy is telling you the session is too long;
  shorten the input or switch to `SlidingWindow` consciously.
- **Safety-sensitive generation.** For anything execution-sensitive (sandboxed
  tool execution, orchestration), prefer greedy tool-call generation and the
  `Reject` context policy. Silently dropping a turn in a tool chain changes the
  meaning of the session; the defaults here are chosen so that does not happen
  by accident.

## Forbidden / unsupported

The server rejects (HTTP 400) the following, regardless of use case — they are
not a matter of taste:

- `n != 1` (only one choice is generated).
- `parallel_tool_calls = true` (one tool call per turn, by design).
- Non-zero `frequency_penalty` / `presence_penalty` (not implemented).
- `logprobs = true` (log probabilities are not produced).

## Versioning

This table is part of the model's I/O contract and is versioned alongside the
chat template (`chat_template_version`, Phase 52). A change to these defaults
that changes observable generation behaviour is a template-shape change and
bumps `chat_template_version`.
