from __future__ import annotations

import asyncio
import logging
import os
import time
import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from realtime_bridge import publish_run_completed

from mol_design_agents import XTBConfig
from mol_design_agents import XTBSimulationAgent
import parsl
from parsl import Config
from parsl import HighThroughputExecutor
from parsl.concurrent import ParslPoolExecutor
from parsl.providers import LocalProvider

from academy.exchange import RedisExchangeFactory
from academy.manager import Manager

from flowcept import Flowcept

logger = logging.getLogger(__name__)


def _successful_xtb_results(campaign_id: str) -> list[tuple[str, float]]:
    """Read actual successful calculation records from the live analyzer."""
    base = os.environ.get("PROVENANCE_MONITOR_URL", "http://127.0.0.1:8765").rstrip("/")
    query = urlencode({"run_id": campaign_id, "limit": 1000})
    request = Request(f"{base}/v1/audit?{query}")
    with urlopen(request, timeout=10) as response:
        audit = json.loads(response.read().decode("utf-8"))
    results = []
    for item in audit.get("events", []):
        record = item.get("event", {}).get("payload", {}).get("record", {})
        if record.get("activity_id") != "compute_ionization_energy":
            continue
        used = record.get("used") if isinstance(record.get("used"), dict) else {}
        generated = record.get("generated") if isinstance(record.get("generated"), dict) else {}
        value = generated.get("ionization_energy")
        if record.get("status") == "FINISHED" and isinstance(value, (int, float)):
            smiles = used.get("smiles")
            if isinstance(smiles, str):
                results.append((smiles, float(value)))
    return results


async def main() -> int:
    logging.basicConfig(level=logging.INFO)

    config = Config(
        executors=[
            HighThroughputExecutor(
                provider=LocalProvider(
                    worker_init=os.environ.get(
                        'GAIED_WORKER_INIT',
                        f'cd "{os.getcwd()}" && export PYTHONPATH="{os.getcwd()}:$PYTHONPATH"',
                    ),
                ),
                max_workers_per_node=1,
            ),
        ],
    )
    executor = ParslPoolExecutor(config)

    async with await Manager.from_exchange_factory(
        factory=RedisExchangeFactory(
            os.environ.get('GAIED_REDIS_HOST', 'localhost'),
            int(os.environ.get('GAIED_REDIS_PORT', '6379')),
        ),
        executors=executor,
    ) as manager:
        seeds = [os.environ.get('GAIED_SEED', 'CNC(N)=O')]
        max_iterations = max(2, int(os.environ.get('GAIED_MAX_ITERATIONS', '2')))
        print(
            f'Starting one-seed GAIED campaign: {seeds[0]} '
            f'({max_iterations} iterations maximum)',
            flush=True,
        )

        agents = []
        for molecule in seeds:
            agents.append(
                await manager.launch(
                    XTBSimulationAgent,
                    args=(
                        XTBConfig(),
                        molecule,
                        max_iterations,
                        3,   # no_improvement_tolerance
                    ),
                ),
            )

        print('Starting discovery campaign')
        print('=' * 80)
        # Track each agent's last known molecule count; exit when all stable.
        started_at = time.monotonic()
        max_runtime = float(os.environ.get('GAIED_MAX_RUNTIME_SECONDS', '1200'))
        active: dict[int, tuple] = {i: (agent, -1) for i, agent in enumerate(agents)}
        agent_errors: list[int] = []
        try:
            while active:
                await asyncio.sleep(max(5.0, float(os.environ.get('GAIED_REPORT_INTERVAL_SECONDS', '10'))))
                finished = []
                for i, (agent, last_count) in list(active.items()):
                    try:
                        status, report = await asyncio.gather(
                            asyncio.wait_for(agent.campaign_status(), timeout=10.0),
                            asyncio.wait_for(agent.report(), timeout=10.0),
                        )
                        count = len(report)
                        print(f'Progress report from agent {i}')
                        print(f'Number of molecules simulated: {count}')
                        print(f'Five best molecules: {report[:5]}')
                        print(
                            f"Workflow stage: {'complete' if status['complete'] else 'running'}; "
                            f"iterations: {status['iterations_completed']}; "
                            f"elapsed: {int(time.monotonic() - started_at)}s",
                            flush=True,
                        )
                        if status['complete']:
                            if not report:
                                agent_errors.append(i)
                                print(
                                    f'Agent {i} finished without a successful xTB result.',
                                    flush=True,
                                )
                            else:
                                print(f'Agent {i} finished after its configured iterations.')
                            finished.append(i)
                        else:
                            active[i] = (agent, count)
                    except Exception as exc:
                        if "mailbox for agentid" in str(exc).casefold() and "terminated" in str(exc).casefold():
                            try:
                                results = _successful_xtb_results(os.environ['FLOWCEPT_CAMPAIGN_ID'])
                            except Exception:
                                logger.exception('Could not read final xTB results from the analyzer')
                                results = []
                            print(f'Agent {i} workflow ended; {len(results)} successful xTB result(s) recorded.', flush=True)
                            baseline = next((value for smiles, value in results if smiles == os.environ.get('GAIED_SEED', 'CNC(N)=O')), None)
                            novel = [(smiles, value) for smiles, value in results if smiles != os.environ.get('GAIED_SEED', 'CNC(N)=O')]
                            if baseline is not None and any(value > baseline for _, value in novel):
                                winner = max(novel, key=lambda item: item[1])
                                print(f'Objective achieved in recorded xTB results: {winner[0]} = {winner[1]} eV > seed {baseline} eV.', flush=True)
                            else:
                                logger.error('Objective not achieved: no novel candidate has a recorded ionization energy above the seed baseline.')
                                agent_errors.append(i)
                            finished.append(i)
                            continue
                        # Status calls can be delayed while Academy is running a
                        # real local LLM/xTB action. A delayed report is not a
                        # completed campaign; keep polling until lifecycle says done.
                        logger.warning(
                            'Agent %d status unavailable while work may still be running: %s',
                            i, exc,
                        )
                        print(
                            f'Agent {i} busy; continuing to monitor.',
                            flush=True,
                        )
                        active[i] = (agent, last_count)
                if time.monotonic() - started_at > max_runtime:
                    logger.error('Campaign exceeded its %.0f-second safety limit.', max_runtime)
                    agent_errors.extend(i for i in active if i not in agent_errors)
                    finished.extend(active)
                for i in finished:
                    del active[i]
                print('=' * 80)
            print('All agents completed.')
            if agent_errors:
                raise RuntimeError(
                    f'GAIED agents failed or the measured xTB results did not beat the seed baseline: {agent_errors}'
                )
        except (KeyboardInterrupt, asyncio.CancelledError):
            print('\nShutting down agents...')

    return 0


if __name__ == '__main__':
    import signal
    import threading

    exit_code = 1
    run_status = 'failed'
    run_reason = 'GAIED campaign did not finish successfully.'
    with Flowcept() as flowcept:
        campaign_id = flowcept.campaign_id
        print(f'Flowcept campaign ID: {campaign_id}', flush=True)
        print(
            f'GAIED live profile: seed={os.environ.get("GAIED_SEED", "CNC(N)=O")}, '
            f'max_iterations={os.environ.get("GAIED_MAX_ITERATIONS", "2")}, '
            f'xTB workers={os.environ.get("GAIED_XTB_WORKERS", "1")}',
            flush=True,
        )
        os.environ['FLOWCEPT_CAMPAIGN_ID'] = flowcept.campaign_id
        try:
            exit_code = asyncio.run(main())
            run_status = 'completed' if exit_code == 0 else 'failed'
            run_reason = 'GAIED workflow finished with successful xTB results.' if exit_code == 0 else 'GAIED workflow reported an agent failure.'
        except Exception as exc:
            logger.exception('GAIED workflow failed.')
            run_reason = f'{type(exc).__name__}: {exc}'
            exit_code = 1
        finally:
            # Suppress further Ctrl+C so cleanup isn't interrupted.
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            # Run Parsl cleanup in a daemon thread; if it hangs, we still exit.
            def _parsl_cleanup():
                try:
                    parsl.dfk().cleanup()
                except Exception:
                    pass
            t = threading.Thread(target=_parsl_cleanup, daemon=True)
            t.start()
            t.join(timeout=15)
    if campaign_id:
        publish_run_completed(campaign_id, status=run_status, reason=run_reason)
    # os._exit bypasses atexit handlers (avoids the "DFK still running" warning).
    os._exit(exit_code)
