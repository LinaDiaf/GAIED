# Real-time workflow monitoring

The monitor wraps the existing `provenance.py` → `knowledge_graph.py` →
`verification.py` pipeline. It consumes one event at a time, keeps an isolated
record snapshot for each `run_id`, and re-runs the deterministic verifier for
that run after each record update. Re-evaluation is run-scoped so records from
different runs are never joined. A late parent record or corrected claim can
change an earlier finding; changes are recorded as finding-history transitions.
This favors conservative correctness and implementation reuse over an
incremental rewrite of the scientific checks.

## Run the monitor and simulation

Start the local API:

```bash
cd provenance_analytics
python -m realtime_monitor.server --host 127.0.0.1 --port 8765
```

In another terminal, run the deterministic mock agent:

```bash
python -m realtime_monitor.simulated_agent
```

The simulator sends a mock calculation and an inconsistent claim, polls for
feedback, then submits a mock correction. Its values are test fixtures, not
scientific results. Run the existing and new tests with:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q -p no:cacheprovider
```

The default audit database is `output/realtime_audit.sqlite3`; it can be
overridden with `--db`. The service binds to localhost by default. Keep it
behind an authenticated, access-controlled boundary before exposing it beyond
the local machine.

## Event and HTTP contract

`POST /v1/events` accepts version `1.0` envelopes. Each event needs a unique
`event_id`, explicit `run_id`, `source.system`, `event_type`, and object
`payload`. Replaying an identical event is idempotent; reusing its ID for
different content is rejected.

Supported event types:

- `record.upsert`: include `source_record_id` and `payload.record`. The record
  must contain the same explicit Flowcept/task identifier. New versions replace
  the current record in that run; the audit database retains each submitted
  event.
- `agent.response`: include `payload.intervention_id`, a response `message`,
  and optionally `record_updates` with explicit `{record_id, record}` entries.
  A response is **verified resolved** only when a corresponding verifier check
  becomes `SUPPORTED` with matching scope and source-record evidence. A verbal
  response alone is not resolution.
- `run.completed`: records an explicit campaign/run completion event; it does
  not imply that every calculation or candidate was captured. Send it only
  after the branch has actually finished all monitored agents. Findings that
  depend on the absence of a workflow-scoped result are held until this marker.
  A calculation with an explicit terminal status but no output can be flagged;
  active calculations are not interrupted merely because their output has not
  arrived yet. The monitor never infers campaign completion from a single
  workflow or task status.

`GET /v1/feedback?run_id=...` returns queued instructions for that run.
Feedback is delivered at least once: repeated polls may return an instruction
until an `agent.response` with its `intervention_id` acknowledges it. Consumers
should deduplicate by `intervention_id`.
`GET /v1/audit?run_id=...` returns paged event, finding-transition, and
intervention histories. The default page size is 100; use `limit` and separate
`event_offset`, `finding_offset`, or `intervention_offset` query parameters.

Events larger than 5 MiB are rejected. The API uses the Python standard
library; it does not require the Flowcept, GAIED, LangGraph, Academy, LLM, or
chemistry packages.

## Flowcept and GAIED adapters

`FlowceptAdapter.to_event(record, run_id=..., event_id=...)` maps a decoded
Flowcept record to the standard envelope without importing Flowcept.
`FlowceptJSONLTailAdapter` can tail an append-only Flowcept JSONL buffer while
it is being written:

```bash
python -m realtime_monitor.flowcept_jsonl --path path/to/flowcept.jsonl --run-id campaign-123 --from-start
```

By default the tailer starts at the end and forwards only new complete lines.
Use `--from-start` to replay existing lines while continuing to tail, or
`--replay` to send a finished file once. Replaying is safe because event IDs are
derived deterministically from the run, record ID, and content. For message
queues, connect a Flowcept-supported consumer to `FlowceptAdapter` and
`MonitorClient`; this repository does not start or configure a broker.

The bundled GAIED integration implements the two producer/agent seams:

1. At the point a Flowcept record becomes available, call the Flowcept adapter
   and send its event to `POST /v1/events`.
2. At a safe agent control point, poll `GET /v1/feedback` and pass returned
   instructions through that framework's normal agent-message/state mechanism.
   After responding, publish an `agent.response` event with actual changed
   trace records if there are any.

After the branch confirms all monitored agents have finished, it should send
one explicit `run.completed` event. Use a campaign-level `run_id` when the run
contains multiple workflow/agent IDs; this keeps their records in the same
isolated campaign snapshot while still requiring explicit provenance links for
cross-workflow comparisons.

Example producer-side code:

```python
from realtime_monitor import FlowceptAdapter
from realtime_monitor.client import MonitorClient

client = MonitorClient("http://127.0.0.1:8765")
event = FlowceptAdapter.to_event(
    flowcept_record,
    run_id=explicit_run_id,
    event_id=stable_event_id,
)
client.send_event(event.to_dict())
```

`realtime_bridge.py` publishes Flowcept events and agent responses;
`mol_design_agents.py` polls feedback at the LangGraph planning boundary; and
`run-07.py` sends the terminal campaign event. The root
`compose.gaied.yml` starts the analyzer API, Redis, and GAIED together. The
monitor API itself remains independent of Flowcept, GAIED, and the broker.

`SQLiteFeedbackAdapter` is the default pollable queue. Applications with a
native control channel can provide a `CallbackFeedbackAdapter`. Instructions
are deterministic and tied to a finding's reason code and source evidence.
They request missing source information or review; they never supply invented
values. Unhandled or non-actionable findings do not trigger interventions.

## Correctness boundaries

- Findings retain the offline verifier's `SUPPORTED`, `CONTRADICTED`, and
  `INCONCLUSIVE` meanings. A trace conflict is not independent scientific
  validation.
- Missing, ambiguous, or late evidence is not treated as false. No cross-run
  matching, inferred identity, unit, value, or parent link is performed.
- Each run snapshot is re-evaluated after new records arrive; the verifier
  currently recomputes findings across that run rather than maintaining
  per-check incremental indexes. Event ingestion and audit are incremental,
  but very large runs may need a future incremental-check optimization.
- SQLite stores submitted events, current record snapshots, finding history,
  and intervention/response states. The current record snapshot is an index;
  the event log is the audit source.
- The monitor cannot force an agent to follow an instruction. Integration must
  wire the feedback adapter into the branch's normal control loop and preserve
  that branch's safety policy.
- No online retrieval, external scientific evidence, LLM judgment, or
  molecular calculations are performed by this monitor.
