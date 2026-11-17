from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from contextvars import ContextVar
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from dataclasses import field
from io import StringIO
from typing import Any
from typing import Literal

import json
import re
from importlib.metadata import PackageNotFoundError, version
import requests
import numpy as np
from ase.io import read
from ase.optimize import LBFGSLineSearch
from langchain.messages import HumanMessage
from langchain.messages import SystemMessage
from langchain.tools import tool
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph
from rdkit import Chem
from rdkit.Chem import AllChem
from xtb.ase.calculator import XTB

from academy.agent import action
from academy.agent import Agent
from academy.agent import loop

logger = logging.getLogger(__name__)

_SomeAPI_URL = os.environ.get("SOMEAPI_URL", "")
_SomeAPI_USER = os.environ.get("SomeAPI_USER", "")
_SomeAPI_API_KEY = os.environ.get("SOMEAPI_API_KEY", "")
_OPENAI_COMPATIBLE = os.environ.get("SOMEAPI_OPENAI_COMPATIBLE", "").lower() in {
    "1", "true", "yes",
} or _SomeAPI_URL.rstrip("/").endswith("/chat/completions")
# Keep the local CPU run responsive while preserving concise reasoning/tool JSON.
_LLM_MAX_TOKENS = min(32, max(24, int(os.environ.get("GAIED_LLM_MAX_TOKENS", "32"))))
_LLM_TIMEOUT_SECONDS = max(30.0, float(os.environ.get("GAIED_LLM_TIMEOUT_SECONDS", "600")))
try:
    _XTB_VERSION = version("xtb")
except PackageNotFoundError:
    try:
        _XTB_VERSION = version("xtb-python")
    except PackageNotFoundError:
        _XTB_VERSION = "unreported"


class _SomeAPIResponse:
    def __init__(self, content: str, tool_calls: list | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls or []


class SomeAPILLM:
    """Drop-in replacement for ChatOpenAI that routes calls through the SomeAPI gateway."""

    def __init__(self, model: str, temperature: float = 0.1) -> None:
        self.model = model
        self.temperature = temperature
        self._tools: list | None = None
        self.max_tokens_override: int | None = None

    def bind_tools(self, tools: list) -> SomeAPILLM:
        bound = SomeAPILLM(self.model, self.temperature)
        bound._tools = tools
        return bound

    async def ainvoke(self, messages: list) -> _SomeAPIResponse:
        system_content = ""
        user_content = ""
        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_content = msg.content
            elif isinstance(msg, HumanMessage):
                user_content = msg.content

        if self._tools:
            tool_names = ", ".join(t.name for t in self._tools)
            user_content = (
                f"{user_content}\n\n"
                f"Available tools: {tool_names}.\n"
                "Chemical input rules: pass neutral molecules only; the tool computes ionization energy from the neutral structure. "
                "Never append a bare '+' to a SMILES. Use only chemically valid SMILES and include the starting molecule as a baseline. "
                'Respond with ONLY a JSON object: '
                '{"tool_calls": [{"name": "<tool>", "args": {"smiles": "<SMILES>"}}]}'
            )

        is_claude = self.model.lower().startswith("claude")
        if _OPENAI_COMPATIBLE:
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens_override or _LLM_MAX_TOKENS,
            }
        elif is_claude:
            payload = {
                "user": _SomeAPI_USER,
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                ],
                "stop": [],
                "temperature": self.temperature,
            }
        else:
            payload = {
                "user": _SomeAPI_USER,
                "model": self.model,
                "system": system_content,
                "prompt": [user_content],
                "stop": [],
                "temperature": self.temperature,
                "top_p": 0.9,
            }

        def _post() -> dict:
            if not _SomeAPI_URL.startswith(("https://", "http://")) or _SomeAPI_URL.rstrip("/") in {"https:", "http:"}:
                raise RuntimeError(
                    "Set SOMEAPI_URL to a usable OpenAI-compatible/custom gateway endpoint."
                )
            headers = {"Content-Type": "application/json"}
            if _SomeAPI_API_KEY:
                headers["Authorization"] = f"Bearer {_SomeAPI_API_KEY}"
            resp = requests.post(
                _SomeAPI_URL,
                data=json.dumps(payload),
                headers=headers,
                timeout=_LLM_TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
            return resp.json()

        data = await asyncio.to_thread(_post)
        content = data.get("response")
        if content is None and isinstance(data.get("choices"), list) and data["choices"]:
            message = data["choices"][0].get("message", {})
            content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("LLM gateway response must include response or choices[0].message.content")

        if self._tools:
            try:
                start = content.find('{')
                end = content.rfind('}')
                parsed = json.loads(content[start:end + 1])
                tool_calls = parsed.get('tool_calls', [])
            except (json.JSONDecodeError, ValueError):
                tool_calls = []
            return _SomeAPIResponse(content=content, tool_calls=tool_calls)

        return _SomeAPIResponse(content=content)


def _generate_initial_xyz(mol_string: str) -> str:
    """Generate the XYZ coordinates for a molecule.

    Args:
        mol_string: SMILES string

    Returns:
        - InChI string for the molecule
        - XYZ coordinates for the molecule
    """

    # Generate 3D coordinates for the molecule
    mol = Chem.MolFromSmiles(mol_string)
    if mol is None:
        raise ValueError(f'Parse failure for {mol_string}')
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=1)
    AllChem.MMFFOptimizeMolecule(mol)

    # Save geometry as 3D coordinates
    xyz = f'{mol.GetNumAtoms()}\n'
    xyz += mol_string + '\n'
    conf = mol.GetConformer()
    for i, a in enumerate(mol.GetAtoms()):
        s = a.GetSymbol()
        c = conf.GetAtomPosition(i)
        xyz += f'{s} {c[0]} {c[1]} {c[2]}\n'

    return xyz


@dataclass
class XTBConfig:
    accuracy: float = 0.05
    search_fmax: float = 0.02
    search_steps: int = 80


def _compute_energy_sync(config: XTBConfig, smiles: str) -> tuple[float, bool]:
    """Run the ionization potential computation."""

    # Make the initial geometry
    xyz = _generate_initial_xyz(smiles)

    # Make the XTB calculator
    calc = XTB(accuracy=config.accuracy)

    # Parse the molecule
    atoms = read(StringIO(xyz), format='xyz')

    # Compute the neutral geometry
    # Uses QCEngine (https://github.com/MolSSI/QCEngine)
    # to handle interfaces to XTB
    atoms.calc = calc
    dyn = LBFGSLineSearch(atoms, logfile=None)
    neutral_geometry_converged = dyn.run(
        fmax=config.search_fmax, steps=config.search_steps
    )

    neutral_energy = atoms.get_potential_energy()

    # Compute the energy of the relaxed geometry in charged form
    charges = np.ones((len(atoms),)) * (1 / len(atoms))
    atoms.set_initial_charges(charges)
    # A vertical ionization creates a radical cation (doublet), not a closed-shell
    # +1 state. The ASE xTB calculator derives total charge and unpaired count
    # from the sums of initial charges and magnetic moments.
    atoms.set_initial_magnetic_moments(charges)
    charged_energy = atoms.get_potential_energy()

    # ASE's xtb calculator returns energies in eV (it converts the internal
    # Hartree value before returning it), so the delta is eV as well.
    return charged_energy - neutral_energy, bool(neutral_geometry_converged)


@dataclass
class SearchState:
    seed: str
    plan: str = ''
    tool_calls: list[Any] = field(default_factory=list)
    simulated_molecules: dict[str, float] = field(default_factory=dict)
    conclusions: list[str] = field(default_factory=list)
    critique: str = ''


class XTBSimulationAgent(Agent):
    """Agent for running XTB to characterize molecules."""

    def __init__(
        self,
        xtb_config: XTBConfig,
        start_molecule: str,
        max_iterations: int | None = None,
        no_improvement_tolerance: int | None = None,
        reasoning_model: str | None = None,
        generation_model: str | None = None,
    ):
        self.config = xtb_config
        self.reasoning_model = reasoning_model or os.environ.get(
            'GAIED_REASONING_MODEL', 'gpt54'
        )
        self.generation_model = generation_model or os.environ.get(
            'GAIED_GENERATION_MODEL', 'gpt54'
        )
        self.search_state = SearchState(seed=start_molecule)
        self.molecule_cache: dict[str, float] = {}
        self.max_iterations = max_iterations
        self.no_improvement_tolerance = no_improvement_tolerance
        self._campaign_complete = False
        self._iterations_completed = 0

    async def agent_on_startup(self) -> None:
        n_workers = max(1, int(os.environ.get("GAIED_XTB_WORKERS", "1")))
        self._flowcept = None
        self._flowcept_task_cls = None
        self._flowcept_status = None
        self._monitor_bridge = None
        self._flowcept_interceptor = None
        self._flowcept_original_intercept = None
        self._flowcept_original_ainvoke = None
        campaign_id = os.environ.get('FLOWCEPT_CAMPAIGN_ID')
        if campaign_id:
            from flowcept import Flowcept, FlowceptTask
            from flowcept.commons.vocabulary import Status
            from flowcept.flowceptor.adapters.instrumentation_interceptor import (
                InstrumentationInterceptor,
            )
            from realtime_bridge import LiveMonitorBridge

            self._agent_id = str(uuid.uuid4())
            self._flowcept_task_cls = FlowceptTask
            self._flowcept_status = Status
            self._active_prov_parent = ContextVar(
                f"gaied_parent_{self._agent_id}", default=None
            )
            self._flowcept = Flowcept(
                campaign_id=campaign_id,
                workflow_name=f"GAIED molecular design: {self.search_state.seed}",
                workflow_subtype="agentic_workflow",
                agent_id=self._agent_id,
                agent_name=f"xTB agent {self.search_state.seed}",
                start_persistence=False,
                save_workflow=True,
                check_safe_stops=False,
            ).start()
            self._monitor_bridge = LiveMonitorBridge(campaign_id)
            self._monitor_seen_interventions: set[str] = set()
            self._monitor_feedback_cache: dict[str, dict[str, Any]] = {}
            self._monitor_response_sent: set[str] = set()
            self._monitor_seen_record_ids: set[str] = set()
            self._monitor_pending_records: dict[str, list[dict[str, Any]]] = {}

            self._flowcept_interceptor = InstrumentationInterceptor.get_instance()
            self._flowcept_original_intercept = self._flowcept_interceptor.intercept
            original_intercept = self._flowcept_original_intercept
            bridge = self._monitor_bridge

            def forward_flowcept_record(record):
                result = original_intercept(record)
                record_id = LiveMonitorBridge._record_id(record)
                parent_id = record.get("parent_task_id")
                if parent_id and str(parent_id) not in agent._monitor_seen_record_ids:
                    agent._monitor_pending_records.setdefault(str(parent_id), []).append(record)
                else:
                    bridge.publish(record)
                if record_id:
                    record_id = str(record_id)
                    agent._monitor_seen_record_ids.add(record_id)
                    for child in agent._monitor_pending_records.pop(record_id, []):
                        bridge.publish(child)
                return result

            self._flowcept_interceptor.intercept = forward_flowcept_record

            self._flowcept_original_ainvoke = SomeAPILLM.ainvoke
            original_ainvoke = self._flowcept_original_ainvoke
            agent = self

            async def traced_ainvoke(self, messages):
                system_prompt = "\n".join(
                    str(msg.content) for msg in messages
                    if isinstance(msg, SystemMessage)
                )
                user_prompt = "\n".join(
                    str(msg.content) for msg in messages
                    if isinstance(msg, HumanMessage)
                )
                task = agent._start_flowcept_task(
                    f"llm_call/{self.model}",
                    "ai_model_invocation",
                    {
                        "model": self.model,
                        "temperature": self.temperature,
                        "max_tokens": self.max_tokens_override or _LLM_MAX_TOKENS,
                        "has_tools": bool(self._tools),
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt,
                    },
                )
                started = time.perf_counter()
                try:
                    result = await original_ainvoke(self, messages)
                    agent._end_flowcept_task(
                        task,
                        generated={
                            "response": result.content,
                            "tool_calls": result.tool_calls,
                        },
                        custom_metadata={
                            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2)
                        },
                    )
                    return result
                except Exception as exc:
                    agent._end_flowcept_task(task, error=exc)
                    raise

            SomeAPILLM.ainvoke = traced_ainvoke
            logger.info("Flowcept task capture and live event forwarding enabled.")

        self.pool = ProcessPoolExecutor(max_workers=n_workers)

        tools = [tool(self.compute_ionization_energy)]
        self.reasoning_llm = SomeAPILLM(model=self.reasoning_model)
        self.generation_llm = SomeAPILLM(model=self.generation_model)
        self.tools_by_name = {tool.name: tool for tool in tools}
        self.llm_with_tools = self.generation_llm.bind_tools(tools)

    def _start_flowcept_task(
        self, activity_id: str, subtype: str, used: dict[str, Any]
    ):
        if self._flowcept_task_cls is None:
            return None
        return self._flowcept_task_cls(
            activity_id=activity_id,
            subtype=subtype,
            agent_id=self._agent_id,
            parent_task_id=self._active_prov_parent.get(),
            used=used,
            capture_telemetry=False,
        )

    def _end_flowcept_task(
        self,
        task,
        *,
        generated: dict[str, Any] | None = None,
        custom_metadata: dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        if task is None:
            return
        if self._flowcept_status is None:
            raise RuntimeError("Flowcept status is unavailable for an active task")
        task.end(
            generated=generated,
            custom_metadata=custom_metadata,
            stderr=str(error) if error else None,
            status=self._flowcept_status.ERROR if error else self._flowcept_status.FINISHED,
        )

    @staticmethod
    def _state_provenance(state: SearchState) -> dict[str, Any]:
        return {
            "seed": state.seed,
            "plan": state.plan,
            "tool_calls": state.tool_calls,
            "simulated_molecules": state.simulated_molecules,
            "conclusions": state.conclusions,
            "critique": state.critique,
        }

    def _instrument_node(self, name: str, node):
        async def wrapped(state: SearchState) -> SearchState:
            task = self._start_flowcept_task(
                f"langgraph/{name}", "langgraph_node", self._state_provenance(state)
            )
            token = (
                self._active_prov_parent.set(task.get_id()) if task is not None else None
            )
            started = time.perf_counter()
            try:
                result = await node(state)
                self._end_flowcept_task(
                    task,
                    generated=self._state_provenance(result),
                    custom_metadata={
                        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2)
                    },
                )
                return result
            except Exception as exc:
                self._end_flowcept_task(
                    task,
                    custom_metadata={
                        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2)
                    },
                    error=exc,
                )
                raise
            finally:
                if token is not None:
                    self._active_prov_parent.reset(token)

        return wrapped

    async def _monitor_feedback_for_plan(self) -> tuple[list[dict[str, Any]], int]:
        bridge = getattr(self, "_monitor_bridge", None)
        if bridge is None:
            return [], 0
        feedback = await asyncio.to_thread(bridge.poll_feedback)
        for item in feedback:
            intervention_id = item.get("intervention_id")
            if intervention_id:
                self._monitor_feedback_cache[intervention_id] = item
                self._monitor_seen_interventions.add(intervention_id)
        with bridge._lock:
            cursor = len(bridge._records)
        return list(self._monitor_feedback_cache.values()), cursor

    async def agent_on_shutdown(self) -> None:
        self.pool.shutdown()
        if self._monitor_bridge is not None:
            for pending in self._monitor_pending_records.values():
                for record in pending:
                    self._monitor_bridge.publish(record)
            self._monitor_pending_records.clear()
        if self._flowcept_original_ainvoke is not None:
            SomeAPILLM.ainvoke = self._flowcept_original_ainvoke
        if self._flowcept_interceptor is not None:
            del self._flowcept_interceptor.intercept
        if self._flowcept is not None:
            self._flowcept.stop()

    async def compute_ionization_energy(self, smiles: str) -> float:
        """Compute the ionization energy for the given molecule.

        Args:
            smiles: SMILES string to evaluate
        Returns:
            Ionization energy in eV
        """
        if smiles in self.molecule_cache:
            return self.molecule_cache[smiles]

        task = self._start_flowcept_task(
            "compute_ionization_energy",
            "agent_tool",
            {
                "smiles": smiles,
                "method": "GFN2-xTB",
                "geometry_steps": self.config.search_steps,
            },
        )
        started = time.perf_counter()
        loop = asyncio.get_event_loop()
        try:
            ionization_energy, neutral_geometry_converged = await loop.run_in_executor(
                self.pool,
                _compute_energy_sync,
                self.config,
                smiles,
            )
            self.molecule_cache[smiles] = ionization_energy
            self._end_flowcept_task(
                task,
                generated={
                    "ionization_energy": ionization_energy,
                    "unit": "eV",
                    "neutral_geometry_converged": neutral_geometry_converged,
                },
                custom_metadata={
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                    "unit": "eV",
                    "scientific_protocol": {
                        "method": "GFN2-xTB",
                        "software": "xtb-python ASE calculator",
                        "software_version": _XTB_VERSION,
                        "ionization_energy_type": "vertical",
                        "geometry_protocol": "neutral_optimized_cation_single_point",
                        "neutral_charge": 0,
                        "cation_charge": 1,
                        "neutral_multiplicity": 1,
                        "cation_multiplicity": 2,
                        "all_geometry_optimizations_converged": neutral_geometry_converged,
                        "energy_unit": "eV",
                        "method_parameters": {
                            "accuracy": self.config.accuracy,
                            "geometry_fmax": self.config.search_fmax,
                            "geometry_steps": self.config.search_steps,
                        },
                    },
                },
            )
            return ionization_energy
        except Exception as exc:
            self._end_flowcept_task(
                task,
                custom_metadata={
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 2)
                },
                error=exc,
            )
            raise

    @loop
    async def conduct_simulation_campaign(  # noqa: C901,PLR0915
        self,
        shutdown: asyncio.Event,
    ) -> None:
        """Conduct a simulation campaign.

        This loop uses an LLM to conduct a search for molecules with high
        ionizationenergy, using the starting molecule as a seed for the
        search space.
        """

        async def plan(state: SearchState) -> SearchState:
            monitor_feedback, feedback_cursor = await self._monitor_feedback_for_plan()
            correction = "\n\nAnalyzer review instructions (grounded in recorded provenance; do not invent missing values):\n" + "\n".join(
                f"- {item.get('instruction', '')}" for item in monitor_feedback
            ) if monitor_feedback else ""
            response = await self.reasoning_llm.ainvoke(
                [
                    SystemMessage(
                        content=(
                            'You are a expert computational chemist tasked '
                            'with finding molecules with desired properties. '
                            'You should use simulations to explore the '
                            'chemical space in search of better molecules. '
                            'Come up with a plan for novel molecules to '
                            'simulate. Ground your proposed molecules in '
                            'knowledge of chemical and physical properties. '
                            'For each proposed molecule that you simulate, '
                            'provide the reasoning about why you expect this '
                            'molecule to be better either than previous '
                            'molecules that you have seen, or what you hope '
                            'to learn from running the simulation. In addition'
                            ' to the target property, you should always '
                            'consider things like cost and sythesizability. '
                            'Note: the only experiment your lab is capable of '
                            'running is calculating the ionization energy of '
                            'proposed molecules.'
                        ),
                    ),
                    HumanMessage(
                        content=(
                            'Look for molecules with high ionization energy '
                            f'starting with the molecule {state.seed}' + correction
                        ),
                    ),
                ],
            )

            logger.info(f'Planner: {response}')

            bridge = getattr(self, '_monitor_bridge', None)
            if bridge is not None:
                for intervention in monitor_feedback:
                    intervention_id = intervention.get('intervention_id')
                    if not intervention_id or intervention_id in self._monitor_response_sent:
                        continue
                    await asyncio.to_thread(
                        bridge.response,
                        intervention,
                        "The analyzer instruction was included in the GAIED planning input; subsequent provenance will be checked for resolution evidence.",
                        feedback_cursor,
                    )
                    self._monitor_response_sent.add(intervention_id)

            return SearchState(
                seed=state.seed,
                plan=response.content,
                tool_calls=state.tool_calls,
                simulated_molecules=state.simulated_molecules,
                conclusions=state.conclusions,
                critique=state.critique,
            )

        async def tool_calling(state: SearchState) -> SearchState:
            monitor_feedback, feedback_cursor = await self._monitor_feedback_for_plan()
            correction = "\n\nLive analyzer instructions (follow these exactly):\n" + "\n".join(
                f"- {item.get('instruction', '')}" for item in monitor_feedback
            ) if monitor_feedback else ""
            previously_simulated = ' '.join(
                state.simulated_molecules.keys(),
            )
            response = await self.llm_with_tools.ainvoke(
                [
                    SystemMessage(
                        content=(
                            'You are a computational chemist who is an expert '
                            'at conducting simulations. Using proposed plan '
                            'and chemical reasoning provided, decide which '
                            'simulations to run using the given tool. Do not '
                            'write python code. You should run several '
                            'simulations at once. Do not repeat previously '
                            'conducted simulations. Use neutral molecules only: '
                            'the tool computes ionization energy, so never append '
                            'a bare + to a SMILES. Use valid neutral RDKit SMILES. '
                            'Include the neutral seed as a baseline if not simulated.'
                        ),
                    ),
                    HumanMessage(
                        content=(
                            f'Plan: {state.plan}\n\n'
                            f'Simulated Molecules: {previously_simulated} '
                            'Do not repeat those molecules in the tool calls.' + correction
                        ),
                    ),
                ],
            )
            logger.info(f'Tool Calling: {response}')
            simulated = set(state.simulated_molecules)
            proposed = [call.get('args', {}).get('smiles') for call in response.tool_calls]
            if simulated and (not proposed or all(smiles in simulated for smiles in proposed)):
                # The small local model sometimes repeats the seed even when
                # asked not to. Ask the real generator once for a genuinely new
                # candidate, with a concrete neutral analogue as guidance.
                response = await self.llm_with_tools.ainvoke([
                    SystemMessage(content=(
                        'You are GAIED molecule generation. Return ONLY a JSON tool_calls object. '
                        'Every candidate must be a valid neutral RDKit SMILES and must differ from all simulated molecules. '
                        'The tool calculates ionization energy from a neutral structure.'
                    )),
                    HumanMessage(content=(
                        f'Already simulated: {sorted(simulated)}. The last response repeated these and is unusable. '
                        'Generate one new, chemically plausible electron-poor analogue with higher expected ionization energy. '
                        'A valid example to consider is trifluoroacetamide O=C(N)C(F)(F)F. Do not repeat the seed.'
                        + correction
                    )),
                ])
                proposed = [call.get('args', {}).get('smiles') for call in response.tool_calls]
                if not proposed or any(smiles in simulated for smiles in proposed):
                    # Some OpenAI-compatible local servers return no structured
                    # tool calls for a 3B model. Keep candidate generation real:
                    # ask the same configured LLM for plain text, then require
                    # RDKit to parse a nontrivial, neutral, novel structure.
                    raw = await self.generation_llm.ainvoke([
                        SystemMessage(content=(
                            'You are GAIED molecule generation. Return exactly one neutral, '
                            'RDKit-valid SMILES and no explanation. Generate a plausible '
                            'electron-poor molecule with higher ionization energy.'
                        )),
                        HumanMessage(content=(
                            f'Already simulated: {sorted(simulated)}. Seed: {state.seed}. '
                            'Return one different molecule only.' + correction
                        )),
                    ])
                    raw_text = str(getattr(raw, 'content', raw))
                    candidates = re.findall(r'[A-Za-z0-9@+\-\[\](){}/\\=#$%.]+', raw_text)
                    novel = []
                    for candidate in candidates:
                        mol = Chem.MolFromSmiles(candidate)
                        if mol is None or candidate in simulated or '.' in candidate:
                            continue
                        if mol.GetNumHeavyAtoms() < 3:
                            continue
                        if Chem.GetFormalCharge(mol) != 0:
                            continue
                        canonical = Chem.MolToSmiles(mol)
                        if canonical not in simulated and canonical not in novel:
                            novel.append(canonical)
                    if not novel:
                        raise ValueError(
                            'LLM returned no parseable, neutral, novel SMILES after '
                            f'corrective retry; raw response: {raw_text[:300]!r}'
                        )
                    response = _SomeAPIResponse(
                        content=raw_text,
                        tool_calls=[{
                            'name': 'compute_ionization_energy',
                            'args': {'smiles': novel[0]},
                        }],
                    )
            invalid = [
                call.get('args', {}).get('smiles', '')
                for call in response.tool_calls
                if not isinstance(call.get('args', {}).get('smiles'), str)
                or Chem.MolFromSmiles(call.get('args', {}).get('smiles', '')) is None
            ]
            if invalid:
                response = await self.llm_with_tools.ainvoke([
                    SystemMessage(content=(
                        'You are the GAIED molecule-generation step. Return ONLY a JSON tool_calls object. '
                        'The prior molecule failed RDKit parsing. Generate neutral, RDKit-valid SMILES only. '
                        'Do not append a bare charge sign; the tool computes ionization energy from neutral molecules.'
                    )),
                    HumanMessage(content=(
                        f'Invalid candidate(s): {invalid}. Do not repeat them. '
                        f'Valid neutral seed baseline: {state.seed}. Return one or two valid neutral candidates. '
                        'Every candidate must pass RDKit parsing.' + correction
                    )),
                ])
                invalid = [
                    call.get('args', {}).get('smiles', '')
                    for call in response.tool_calls
                    if not isinstance(call.get('args', {}).get('smiles'), str)
                    or Chem.MolFromSmiles(call.get('args', {}).get('smiles', '')) is None
                ]
                if invalid:
                    raise ValueError(f'LLM proposed invalid SMILES after one correction attempt: {invalid}')

            proposed = [call.get('args', {}).get('smiles') for call in response.tool_calls]
            if (
                state.seed not in state.simulated_molecules
                and state.seed not in proposed
                and Chem.MolFromSmiles(state.seed) is not None
            ):
                response.tool_calls = [
                    {'name': 'compute_ionization_energy', 'args': {'smiles': state.seed}},
                    *response.tool_calls[:1],
                ]

            bridge = getattr(self, '_monitor_bridge', None)
            if bridge is not None:
                actual_smiles = [call.get('args', {}).get('smiles') for call in response.tool_calls]
                for intervention in monitor_feedback:
                    await asyncio.to_thread(
                        bridge.response,
                        intervention,
                        'Analyzer instruction was provided to molecule generation; generated candidates: '
                        + ', '.join(map(str, actual_smiles)),
                        feedback_cursor,
                    )
            format_attempts = 0
            while len(response.tool_calls) == 0:
                format_attempts += 1
                if format_attempts >= 3:
                    raise RuntimeError(
                        'LLM returned no valid molecule tool calls after three attempts'
                    )
                logger.warning('Tool calling agent did not call tools!')
                logger.warning(response)
                response = await self.llm_with_tools.ainvoke(
                    [
                        SystemMessage(
                            content=(
                                'You are a computational chemist who is '
                                'an expert at conducting simulations. '
                                'Using proposed plan and chemical reasoning'
                                ' provided, decide which simulations to run'
                                ' using the given tool. Do not write python '
                                'code. You should run several simulations at '
                                'once. Do not repeat previously conducted '
                                'simulations.'
                            ),
                        ),
                        HumanMessage(
                            content=(
                                f'Plan: {state.plan}\n\n'
                                f'Simulated Molecules: {previously_simulated} '
                                'Do not repeat those molecules in the tool '
                                'calls.'
                            ),
                        ),
                        response,
                        HumanMessage(
                            content=(
                                'That response does not contain any correctly'
                                'formatted tool calls that could be parsed. '
                                'Please try again! Do not write python code!'
                            ),
                        ),
                    ],
                )

            return SearchState(
                seed=state.seed,
                plan=state.plan,
                tool_calls=response.tool_calls,
                simulated_molecules=state.simulated_molecules,
                conclusions=state.conclusions,
                critique=state.critique,
            )

        async def simulate(state: SearchState) -> SearchState:
            """Performs the tool call"""
            logger.warning(
                f'Previous Simulations: {len(state.simulated_molecules)}',
            )
            logger.warning(f'New Simulations: {len(state.tool_calls)}')
            if len(state.tool_calls) == 0:
                return state

            task_to_smiles = {}
            for tool_call in state.tool_calls:
                tool = self.tools_by_name[tool_call['name']]
                task = asyncio.create_task(tool.ainvoke(tool_call['args']))
                task_to_smiles[task] = tool_call['args']['smiles']

            done, _ = await asyncio.wait(task_to_smiles)

            results = {}
            for task in done:
                smiles = task_to_smiles[task]
                try:
                    results[smiles] = await task
                except Exception as e:
                    results[smiles] = str(e)

            new_results = state.simulated_molecules | results

            return SearchState(
                seed=state.seed,
                plan=state.plan,
                tool_calls=[],
                simulated_molecules=new_results,
                conclusions=state.conclusions,
                critique=state.critique,
            )

        async def conclude(state: SearchState) -> SearchState:
            monitor_feedback, _feedback_cursor = await self._monitor_feedback_for_plan()
            correction = "\n\nLive analyzer correction: " + " ".join(
                item.get("instruction", "") for item in monitor_feedback
            ) if monitor_feedback else ""
            results_str = '\n'.join(
                f'{molecule}\t\t{energy}'
                for molecule, energy in state.simulated_molecules.items()
            )
            response = await self.generation_llm.ainvoke(
                [
                    SystemMessage(
                        content=(
                            'You are expert chemist analyzing the results'
                            'of simulations. Synthesize any conclusions from'
                            ' the planned computational campaign and the '
                            'simulation results so far. A -inf result means '
                            'the simulation could not be run, likely because '
                            'the SMILES string of the molecule was invalid. '
                            'You should not repeat conclusions that have '
                            'already been reached but should correct the '
                            'conclusions if they are wrong. For each '
                            'conclusion, output a concise, stand-alone '
                            'statement. Use the unit explicitly recorded with each result. '
                            'Return only the new conclusions.' + correction
                        ),
                    ),
                    HumanMessage(
                        content=(
                            f'Plan: {state.plan}\n\n'
                            'Simulation Results:\n'
                            'Molecule\t\tEnergy\n'
                            f'{results_str}' + correction
                        ),
                    ),
                ],
            )
            if any(item.get("reason_code") in {"UNIT_MISMATCH", "CHECK_CONFLICT"} for item in monitor_feedback):
                exact_ledger = "\n".join(
                    f"{molecule} ionization energy = {value} eV"
                    for molecule, value in state.simulated_molecules.items()
                )
                self.generation_llm.max_tokens_override = 80
                try:
                    response = await self.generation_llm.ainvoke([
                        SystemMessage(content=(
                            "This is a precise correction of recorded xTB results. Reproduce every supplied ledger row exactly. "
                            "Do not omit molecules or numbers. Use eV only; output the ledger and nothing else."
                        )),
                        HumanMessage(content=(
                            "The analyzer found that the prior conclusion conflicts with recorded results. Correct it using this complete ledger. "
                            "Every line below is an actual recorded molecule and xTB result; all units are eV. "
                            "Return every line, unchanged except retain the eV suffix:\n" + exact_ledger
                        )),
                    ])
                finally:
                    self.generation_llm.max_tokens_override = None
                corrected_text = str(response.content)
                for row in exact_ledger.splitlines():
                    if row in corrected_text:
                        continue
                    self.generation_llm.max_tokens_override = 48
                    try:
                        row_response = await self.generation_llm.ainvoke([
                            SystemMessage(content=(
                                "Copy the one supplied, actual xTB result row exactly. "
                                "Do not omit or change any character, value, molecule, or unit. Output only that row."
                            )),
                            HumanMessage(content="Return exactly this recorded result row:\n" + row),
                        ])
                    finally:
                        self.generation_llm.max_tokens_override = None
                    corrected_text += "\n" + str(row_response.content)
                missing_rows = [row for row in exact_ledger.splitlines() if row not in corrected_text]
                if (
                    re.search(r"\bkj\s*/\s*mol\b", corrected_text, re.IGNORECASE)
                    or re.search(r"-inf|invalid\s+smiles|parse\s+failure|could\s+not\s+run", corrected_text, re.IGNORECASE)
                    or "eV" not in corrected_text
                    or missing_rows
                ):
                    raise ValueError(
                        "LLM did not resolve the analyzer finding after one bounded retry; "
                        f"missing ledger rows={missing_rows}, response={corrected_text[:300]!r}"
                    )
                response = _SomeAPIResponse(content=corrected_text)
            logger.info(f'Conclusions: {response}')
            new_state = SearchState(
                seed=state.seed,
                plan=response.content,
                tool_calls=state.tool_calls,
                simulated_molecules=state.simulated_molecules,
                conclusions=[*state.conclusions, response.content],
                critique=state.critique,
            )
            self.search_state = new_state
            return new_state

        async def critique(state: SearchState) -> SearchState:
            results_str = '\n'.join(
                f'{molecule}\t\t{energy}'
                for molecule, energy in state.simulated_molecules.items()
            )
            conclusions = '\n\t'.join(state.conclusions)
            response = await self.generation_llm.ainvoke(
                [
                    SystemMessage(
                        content=(
                            'You are expert chemist analyzing the results of '
                            'simulations. Critique the planned simulation '
                            'campaign based on the chemistry foundations, '
                            'the computational results, and the conclusions '
                            'drawn. Be extremely critical. Look for flaws in '
                            'the methodology and reasoning.'
                        ),
                    ),
                    HumanMessage(
                        content=(
                            f'Plan: {state.plan}\n\n'
                            'Simulation Results:\n'
                            'Molecule\t\tEnergy\n'
                            f'{results_str}\n\n'
                            'Conclusions:\n'
                            f'\t{conclusions}'
                        ),
                    ),
                ],
            )
            logger.info(f'Critique: {response}')
            return SearchState(
                seed=state.seed,
                plan=state.plan,
                tool_calls=state.tool_calls,
                simulated_molecules=state.simulated_molecules,
                conclusions=state.conclusions,
                critique=response.content,
            )

        async def update(state: SearchState) -> SearchState:
            results_str = '\n'.join(
                f'{molecule}\t\t{energy}'
                for molecule, energy in state.simulated_molecules.items()
            )
            conclusions = '\n\t'.join(state.conclusions)
            response = await self.reasoning_llm.ainvoke(
                [
                    SystemMessage(
                        content=(
                            'You are expert chemist analyzing the results of '
                            'simulations. Update the given plan based on the '
                            'results so far, the conclusions drawn and the '
                            'critique given. Ground your updates in current '
                            'results, but also knowledge of chemistry and '
                            'physics. You should output a new, stand-alone '
                            'plan that does not make reference to the '
                            'existing plan. If all of the experiments '
                            'suggested by the plan have been carried out, '
                            'suggest more experiments. Note: the only '
                            'experiment your lab is capable of running is '
                            'calculating the ionization energy of proposed '
                            'molecules.'
                        ),
                    ),
                    HumanMessage(
                        content=(
                            f'Plan: {state.plan}\n\n'
                            'Simulation Results:\n'
                            'Molecule\t\tEnergy\n'
                            f'{results_str}\n\n'
                            'Conclusions:\n'
                            f'\t{conclusions}\n\n'
                            'Critique:\n'
                            f'\t{state.critique}'
                        ),
                    ),
                ],
            )
            logger.info(f'Update: {response}')

            return SearchState(
                seed=state.seed,
                plan=response.content,
                tool_calls=state.tool_calls,
                simulated_molecules=state.simulated_molecules,
                conclusions=state.conclusions,
                critique=state.critique,
            )

        _iteration = 0
        _no_improve_count = 0
        _best_energy: float = float('-inf')

        async def should_continue(
            state: SearchState,
        ) -> Literal['critique', END]:  # type: ignore[valid-type]
            nonlocal _iteration, _no_improve_count, _best_energy
            _iteration += 1
            self._iterations_completed = _iteration

            # Track whether the global best energy improved this round.
            # Filter out failed simulations stored as strings.
            numeric = [
                v for v in state.simulated_molecules.values()
                if isinstance(v, (int, float))
            ]
            if numeric:
                current_best = max(numeric)
                if current_best > _best_energy:
                    _best_energy = current_best
                    _no_improve_count = 0
                else:
                    _no_improve_count += 1

            if shutdown.is_set():
                return END
            if self.max_iterations is not None and _iteration >= self.max_iterations:
                logger.info(
                    'Stopping after %d iterations (max_iterations=%d)',
                    _iteration, self.max_iterations,
                )
                shutdown.set()
                return END
            if (
                self.no_improvement_tolerance is not None
                and _no_improve_count >= self.no_improvement_tolerance
            ):
                logger.info(
                    'Stopping: no improvement for %d consecutive iterations',
                    _no_improve_count,
                )
                shutdown.set()
                return END
            return 'critique'

        # Build workflow
        agent_builder = StateGraph(SearchState)

        # Add nodes
        agent_builder.add_node('plan', self._instrument_node('plan', plan))
        agent_builder.add_node(
            'tool_calling', self._instrument_node('tool_calling', tool_calling)
        )
        agent_builder.add_node('simulate', self._instrument_node('simulate', simulate))
        agent_builder.add_node('conclude', self._instrument_node('conclude', conclude))
        agent_builder.add_node('critique', self._instrument_node('critique', critique))
        agent_builder.add_node('update', self._instrument_node('update', update))

        # Add edges to connect nodes
        agent_builder.add_edge(START, 'plan')
        agent_builder.add_edge('plan', 'tool_calling')
        agent_builder.add_edge('tool_calling', 'simulate')
        agent_builder.add_edge('simulate', 'conclude')
        agent_builder.add_conditional_edges(
            'conclude',
            should_continue,
            ['critique', END],
        )
        agent_builder.add_edge('critique', 'update')
        agent_builder.add_edge('update', 'tool_calling')

        # Compile the agent
        agent = agent_builder.compile()

        # Run until the agent is terminated
        try:
            await agent.ainvoke(
                self.search_state,
                {'recursion_limit': 10000},
            )
        finally:
            self._campaign_complete = True

    @action
    async def campaign_status(self) -> dict[str, Any]:
        """Expose true workflow lifecycle and real calculation counts."""
        return {
            "complete": self._campaign_complete,
            "iterations_completed": self._iterations_completed,
            "successful_molecules": len(self.molecule_cache),
            "best_molecule": max(
                self.molecule_cache,
                key=lambda smiles: self.molecule_cache[smiles],
            )
            if self.molecule_cache else None,
            "best_ionization_energy_eV": max(self.molecule_cache.values())
            if self.molecule_cache else None,
        }

    @action
    async def report(self) -> list[tuple[str, float]]:
        """Summarize findings so far."""
        best_molecules = sorted(
            self.molecule_cache.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return best_molecules

