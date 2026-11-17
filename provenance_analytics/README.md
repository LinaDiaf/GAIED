# Offline Scientific Provenance Verifier

Checks whether claims, rankings, calculations, and provenance links agree with
the local Flowcept/GAIED JSONL trace. Agreement with a trace is **not proof of
scientific truth**.

## Run

From this directory, install the analyzer and test dependencies:

```bash
python -m pip install -r requirements-dev.txt
python main.py --input path/to/flowcept.jsonl
python -m pytest -q -p no:cacheprovider tests
```

`provenance.py` reads records; `knowledge_graph.py` extracts facts and builds
the graph; `verification.py` checks trace evidence; `main.py` writes reports.
Outputs are in `output/`: a short Markdown summary, detailed JSON findings,
and `knowledge_graph.json`.

The full GAIED + analyzer stack is launched from the repository root with
`docker compose -f compose.gaied.yml up -d --build`; open
`http://127.0.0.1:8765/`. The deterministic simulator remains available only
for API tests. Real GAIED integration steps and external runtime requirements
are in [the live GAIED guide](docs/GAIED_LIVE_RUN.md).

**SUPPORTED** means a check agrees with the trace; **CONTRADICTED** means
comparable trace evidence conflicts; **INCONCLUSIVE** means evidence is
missing or ambiguous. Ranking payload checks only compare recorded values.

This version is offline and does not independently validate science. The
current data lacks explicit units and calculation protocols, so scientific
comparisons remain inconclusive. A parent ID not found in this export is not
necessarily invalid. Workflows are not matched without explicit provenance.

For the optional real-time event API, Flowcept/agent adapters, mock simulation,
and GAIED integration contract, see [docs/REALTIME_MONITORING.md](docs/REALTIME_MONITORING.md).
