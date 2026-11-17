# Real GAIED live run

The analyzer and verifier are bundled in `provenance_analytics/`. The live
bridge forwards records from the real Flowcept interceptor, passes analyzer
feedback to the next GAIED LangGraph planning call, and sends the resulting
agent response and newly emitted records back for verification. The web page
at `http://127.0.0.1:8765/` reads the monitor's SQLite audit API.

## Required runtime

GAIED's checked-in `SOMEAPI_URL` is a placeholder. Before launching, configure
the endpoint and authentication expected by that model gateway. The adapter
accepts `SOMEAPI_URL`, `SOMEAPI_API_KEY` (Bearer authentication), and optional
`SomeAPI_USER`. Keep credentials in environment variables; do not put them in
the repository. A custom gateway must return a string `response` field. For a
standard OpenAI-compatible `/chat/completions` endpoint, the adapter sends
`messages` and reads `choices[0].message.content`; set
`SOMEAPI_OPENAI_COMPATIBLE=1` if the endpoint uses that protocol at another
path. Tool selection continues to use GAIED's existing JSON tool-call prompt.

The GAIED image installs dependencies from `environment.yml` and
`pyproject.toml`; it also needs RDKit, xtb-python with its native xTB runtime,
Academy/Parsl, Flowcept, LangGraph, and Redis. The Compose stack starts Redis
for the GAIED service. This example uses Academy's `RedisExchangeFactory`.

## Start

From the GAIED repository root, the Compose stack keeps the analyzer API, Redis, and GAIED on a private
container network. It publishes the dashboard/API only at `127.0.0.1:8765` and
stores the audit database in a Docker volume. Copy the safe example and set
the LLM endpoint and credentials in the private `.env` file. `.env` is
excluded from version control and Docker build contexts:

```powershell
Copy-Item .env.example .env
notepad .env
# For an OpenAI-compatible gateway, set SOMEAPI_OPENAI_COMPATIBLE=1 and
# replace both model names with values accepted by that API.
$env:PATH = "$env:LOCALAPPDATA\Programs\DockerDesktop\resources\bin;$env:PATH"
docker compose -f compose.gaied.yml up -d --build
docker compose -f compose.gaied.yml logs -f gaied
```

Open `http://127.0.0.1:8765/`. The GAIED logs print the Flowcept campaign ID;
enter that ID in the dashboard. Press Ctrl+C to stop following the logs while
leaving the dashboard and completed containers available. Stop the stack with
`docker compose -f compose.gaied.yml down` when finished.

The default run is one seed (`CNC(N)=O`) and two GAIED iterations. It retains
the planner, tool selection, RDKit 3D geometry, xTB optimization/evaluation,
conclusion, critique, plan update, Flowcept capture, and Academy execution.
The xTB optimizer is capped at 80 steps and its local worker count defaults to
one. Increase those environment settings only when the host can support it.

The dashboard expects the Flowcept `campaign_id` shown by GAIED in the
`FLOWCEPT_CAMPAIGN_ID` environment for the process. Copy that value into Run
ID. A response marked `verified_resolved` requires a matching verifier finding
supported by the changed trace evidence; an acknowledgement by itself remains
unresolved. Trace-consistency results do not establish scientific truth.

## Honest run status

A simulator/API test is not a GAIED experiment. Only records emitted by the
just-started Flowcept campaign count as a real experiment.
