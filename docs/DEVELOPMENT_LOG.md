# Development campaigns

Campaigns run during development, on earlier versions of `pgmem` (Qwen 3.5 9B through Ollama, real xTB,
two agents, three iterations). The code changed between them, so they are not part of the evaluation. Each
observation is classified as handled automatically by the existing rules, not covered (a general problem type), or a
bug that was fixed.

| Campaign | Handled automatically | Not covered | Bug fixed |
|---|---|---|---|
| real1 (two seeds; cut short by the stop rule) | guidance delivered; an agent ignored "do not propose these invalid SMILES again", recorded as not applied; M3 filled by the agents with kinds and statuses | | proposals compared with the memory too late (false "already measured"); list-typed procedure fields |
| real2 (cut short) | genuine repeated proposals detected; invalid-SMILES failures diagnosed; a wrong LLM explanation kept as `candidate_explanation`; M4 procedure derived from two experiences | the same failure diagnosed twice (memory and provenance) | formatting of procedure actions |
| real3 (complete) | reuse of another agent's result; full guidance cycle (6 applied, 5 effective, 1 not effective); contradicting evidence added without overwriting; triple computation detected; repeated proposals stopped after targeted guidance | simultaneous duplicate computations (lock added); effectiveness judged once and agent-declared epistemic status (documented limitations) | evidence of knowledge about a failed molecule; misleading task-error text; guidance counting units; unusable LLM answers not logged; "applied" test unsuited to invalid-SMILES guidance (RDKit test); procedure never drafted again after an unusable LLM answer |
| lock1 (same seed twice; namespace reused) | iterations without a tool call end instead of blocking the agent; invalid-SMILES guidance judged by RDKit; M4 procedure created | a molecule whose failure is confirmed can be proposed and computed again in a later iteration (documented limitation) | an agent blocked for 12 minutes (unbounded tool-call retries in the original loop); an iteration without proposals counted as applied guidance; M4 status counting deliveries as applications |
| lock2 (same seed twice) | one computation for a molecule proposed by both agents at once; guidance not applied, then applied and effective; no procedure without repeated failures | evidence cited by an agent for a wrong statement (documented limitation) | |
| pilot (first evaluation design: 6 scenarios × 4 conditions × 1, Qwen 3.5 9B) | no successful computation repeated with a memory (all remaining duplicates are failed molecules computed again); 10 of 10 statements resting on non-converged results flagged by the analysis agent (0 of 6 without it); every non-converged result diagnosed | no contradiction detected (recall 0); random failures leave no pattern to learn, so the ineffective-procedure scenario never triggered; re-proposals frequent (26 to 42 %) | scenario design: failures by chemical motif with a switch at a fixed number of iterations; retrieval was not task-aware; M2 kinds filtered after ranking; M1 deduplication split by SMILES spelling; injected failure message revealed the injection |

These observations shaped the current design (`docs/PGMEM.md`) and evaluation (`docs/EVALUATION.md`).

Operational issues, unrelated to the system: Ollama stopped (real2, first try); a too short stop rule given in the
commands (real1, real2).

## Pilots of pgmem v3 (evaluation design pgmem-v3.1)

**Pilot with Qwen 3.5 9B** (16 runs, results folder `pilot_qwen9b_v3`): the analysis agent could not keep pace
(most guidance created after the end: 1 to 5 delivered of 11 to 21); 37 % of the proposals were invalid SMILES;
nitrile 12 % and CF3 10 % of valid proposals. Led to: LLM tasks queued behind the rules, JSON answers with line
breaks accepted, goal kept in the memory, change of model.

**Model choice** (remote PC: RX 9070 XT, 16 GB VRAM, 31 GB RAM): `batiai/qwen3.6-35b:iq3` (Qwen3.6-35B-A3B, MoE,
~3B active, imatrix IQ3, 13 GB, built for Ollama; the Unsloth Qwen3.6 GGUFs do not load in Ollama). With the
client's settings (`think: false`, `num_ctx` 8192) it runs 100 % on the GPU (14 GB). Probe (`motif_failures` x `off`
x 2): 2 % invalid SMILES (1 of 52), about 5 s per agent LLM call (9B: about 9 s), campaigns of 170-200 s. Nitrile
was never proposed: the failing motifs became CF3 (31 %) then nitro (14 %), the most proposed motifs that neither
seed carries (amide and heteroaromatic nitrogen are the seeds' motifs).

Operational: Docker (WSL) was stopped twice by Windows for lack of memory (System event 2004, then Hyper-V switch
events): Ollama's `llama-server` keeps 14.7 GB of the model in RAM and the WSL VM could take about 15 GB, against a
commit limit of 33.4 GB (small page file, drive C nearly full). Fixed with `%USERPROFILE%\.wslconfig`
(`memory=10GB`, `swap=4GB`, swap file on D). Interrupted runs are restarted from scratch by the runner.

**Pilot with Qwen3.6-35B-A3B** (16 runs, one per cell, so tendencies only; results `results/eval/pilot`, zip
`pilot_qwen36.zip`):

| Observation | Values |
|---|---|
| analysis agent keeps pace | guidance delivered / never delivered: motif 14 / 2, corrupted 6 / 3, invalid 8 / 6, redundant 1 / 2 (the undelivered ones mostly created during the agents' last iteration); time in rules 18-43 s per campaign of 123-243 s; LLM task wait 6-10 s |
| invalid SMILES | 0 to 3 per campaign; no error or rejected JSON answer in the logs |
| exploration and redundancy | valid distinct molecules, full vs off: 15 vs 6, 15 vs 12, 15 vs 9, 13 vs 12; no redundant computation in full (off: up to 5); structured also explores more than off, so part of the gain comes from the graph memory, not the analysis agent |
| motif failures (phase 1) | full: CF3 share 50 % then 0 %, 3 failures (off 6), no repeated failure (off 2, structured 3); the 4 recurring-failure guidance items applied and effective |
| motif failures (phase 2) | nitro only 5-8 % of computations: about 3 computations after the switch over the 4 conditions; the ineffective-procedure rule never exercised; full kept avoiding CF3 after it stopped failing (8 % vs 46 % in off) |
| corrupted results | 4 statements rested on corrupted results, 1 flagged; corrupted_result precision 0.33, recall 0.5; the agents themselves noticed that the values looked wrong |
| goal drift | 5 diagnoses, all quotes found in the agents' texts; 3 genuine (an agent turning to "diagnose and correct the computational methodology", one confusing the ionization energy with a stability energy), 1 false positive (an observation on perfluorinated compounds) |
| best IE | polluted in full by atoms and reference molecules ([F] 24.8 eV, [O], HF, water): the agents benchmarked xTB, whose values they found too high (benzene 14.2 eV vs 9.24 eV measured) |
| other | `query_memory` used once in 16 runs (the memory acts through the context shown at each node); repeated-proposal guidance rarely applied (1 of 6 in motif); failure explanations advised Gaussian SCF keywords the agents cannot apply (0 of 4 applied) |

Fixed after this pilot: explanation and procedure prompts state that the agents only choose molecules; the
repeated-proposal guidance no longer repeats molecules or shows an empty part. Proposed before the main series,
pending the author's decision: amide as the phase-2 motif, 5 iterations in every scenario, the best-IE metric
restricted to molecules with at least two heavy atoms and no radical.

