import warnings
warnings.simplefilter("ignore", DeprecationWarning)

import os
import sys
from pathlib import Path
import pandas as pd
from typing import List
import json
from datetime import datetime
import asyncio
import threading
import argparse
from mmengine import DictAction

root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from src.logger import logger
from src.config import config
from src.models import model_manager
from src.metric import question_scorer
from src.agent import create_agent, prepare_response
from src.registry import DATASET

append_answer_lock = threading.Lock()


def _serialize_steps(steps) -> list:
    """Serialize memory steps to JSON-safe dicts, stripping large binary fields."""
    serialized = []
    for step in steps:
        try:
            d = step.dict()
            d.pop("model_input_messages", None)
            d.pop("observations_images", None)
            if "model_output_message" in d and isinstance(d["model_output_message"], dict):
                d["model_output_message"].pop("raw", None)
            d["_step_type"] = type(step).__name__
            serialized.append(d)
        except Exception:
            serialized.append({"_step_type": type(step).__name__, "_raw": str(step)[:2000]})
    return serialized

def append_answer(entry: dict, jsonl_file: str) -> None:
    jsonl_file = Path(jsonl_file)
    jsonl_file.parent.mkdir(parents=True, exist_ok=True)
    with append_answer_lock, open(jsonl_file, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(entry) + "\n")
    assert os.path.exists(jsonl_file), "File not found!"
    print("Answer exported to file:", jsonl_file.resolve())


def _has_agent_error(row: pd.Series) -> bool:
    """True when the run failed (timeout, crash, etc.) and should be retried on resume."""
    if "agent_error" not in row.index:
        return False
    err = row["agent_error"]
    if pd.isna(err):
        return False
    return bool(str(err).strip())


def filter_answers(answers_file):
    answer_df = pd.read_json(answers_file, lines=True)

    filttered_df = []
    for row in answer_df.iterrows():
        row = row[1]

        truth = row["true_answer"]

        # Test split: keep any completed run (no agent_error), including abstention
        # "Unable to determine", so a resume does not re-run those questions.
        if truth == "?":
            if not _has_agent_error(row):
                filttered_df.append(row)
        else:
            prediction = row["prediction"]
            if prediction is None or pd.isna(prediction):
                continue
            prediction = str(prediction)
            if question_scorer(prediction, truth):
                filttered_df.append(row)

    filttered_df = pd.DataFrame(filttered_df)
    filttered_df.to_json(answers_file, lines=True, orient='records')

    logger.info(f"Previous answers filtered! {len(answer_df)} -> {len(filttered_df)}")

def get_tasks_to_run(answers_file, dataset) -> List[dict]:

    data = dataset.data

    logger.info(f"Loading answers from {answers_file}...")
    # DRA_RESUME_PRESERVE_ALL=1 opts out of filter_answers, which normally
    # drops wrong/errored validation rows from the dra.jsonl and lets the
    # resume re-attempt them. That default behavior:
    #   (a) wastes budget (each rerun consumes per-Q timeout + tokens), and
    #   (b) gives the model a second chance, which contaminates E0 skill
    #       training — a harder question the model failed once should NOT
    #       get to retry with a different (possibly successful) trajectory
    #       that influences the learned skill library.
    # With this flag, every prior attempt — correct, wrong, errored, or
    # "Unable to determine" — counts as done and is skipped on resume.
    # Use when fairness > retry-on-transient-failure.
    preserve_all = os.environ.get("DRA_RESUME_PRESERVE_ALL", "").strip() in ("1", "true", "True", "yes")
    try:
        if os.path.exists(answers_file):
            if preserve_all:
                logger.info(
                    "DRA_RESUME_PRESERVE_ALL=1: skipping filter_answers — "
                    "every attempted task_id (any outcome) will be treated as done."
                )
            else:
                logger.info("Filtering answers starting.")
                filter_answers(answers_file)
                logger.info("Filtering answers ending.")

            df = pd.read_json(answers_file, lines=True)
            if "task_id" not in df.columns:
                logger.warning(f"Answers file {answers_file} does not contain 'task_id' column. Please check the file format.")
                return []
            done_questions = df["task_id"].tolist()
            logger.info(f"Found {len(done_questions)} previous results!")
        else:
            done_questions = []
    except Exception as e:
        logger.warning("Error when loading records: ", e)
        logger.warning("No usable records! ▶️ Starting new.")
        done_questions = []
    return [line for line in data.to_dict(orient="records") if line["task_id"] not in done_questions]

TRANSIENT_ERROR_KEYWORDS = [
    "Connection refused", "Connection reset", "Connection error",
    "ConnectionError",
    "Internal Server Error", "503", "502", "RemoteDisconnected",
    "ConnectionAbortedError",
]
MAX_RETRIES = 3
RETRY_WAIT_SECS = 60

# Hard wall-clock guard — see docs/handoffs/HANDOFF_THROUGHPUT_REFACTOR.md §P1.
# `asyncio.wait_for(timeout=per_question_timeout)` alone does not enforce a
# hard wall-clock cap: after it calls `task.cancel()`, `finally:` cleanup
# blocks in sub-tools (e.g. `src/tools/auto_browser.py:152-160` which waits up
# to 15 s for `browser_agent.close()`) still run to completion before control
# returns, extending wall time. Observed 2026-04-20: E2 task 023e9d44 ran
# 3298 s against a nominal 1800 s cap.
#
# Fix: build the agent call as an explicit task, shield it so the outer
# `wait_for` cannot auto-cancel the inner, then on TimeoutError cancel
# manually and give cleanup at most CLEANUP_GRACE_SECS to unwind before
# abandoning. `CLEANUP_GRACE_SECS=30` is a 2× margin over the known 15 s
# browser close bound; the already-landed per-call 120 s HTTP timeout
# (fbd0dd1) reaps any stuck LLM call shortly after.
CLEANUP_GRACE_SECS = 30

def _is_transient_error(error: Exception) -> bool:
    err_str = str(error)
    return any(kw in err_str for kw in TRANSIENT_ERROR_KEYWORDS)

async def answer_single_question(config, example):

    augmented_question = example["question"]
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    per_question_timeout = getattr(config, "per_question_timeout_secs", 1800)

    if example["file_name"]:
        prompt_use_files = "\n\nTo solve the task above, you will have to use these attached files:\n"
        file_description = f" - Attached file: {example['file_name']}"
        prompt_use_files += file_description
        augmented_question += prompt_use_files

    output = None
    intermediate_steps = []
    parsing_error = False
    iteration_limit_exceeded = False
    raised_exception = False
    exception = None
    attempts_made = 0

    for attempt in range(MAX_RETRIES + 1):
        attempts_made = attempt + 1
        try:
            agent = await create_agent(config)
            if attempt == 0:
                logger.visualize_agent_tree(agent)
            logger.info(f"Task Id: {example['task_id']}, Final Answer: {example['true_answer']}")

            # Build the agent call as a named task so we can cancel it
            # explicitly on timeout; shield it so `wait_for` does not issue
            # its own auto-cancel. See CLEANUP_GRACE_SECS comment above.
            agent_task = asyncio.create_task(
                agent.run(task=augmented_question),
                name=f"run_{example['task_id']}",
            )
            try:
                final_result = await asyncio.wait_for(
                    asyncio.shield(agent_task),
                    timeout=per_question_timeout,
                )
            except asyncio.TimeoutError:
                agent_task.cancel()
                try:
                    # Bounded cleanup. CRITICAL: we `shield(agent_task)` so
                    # this inner `wait_for` does NOT re-invoke the same
                    # "await cancelled task to unwind" pathology we're
                    # guarding against — if the task is ignoring
                    # cancellation, awaiting it directly would block here
                    # exactly as the original bug does in production.
                    # Instead, shield + timeout gives us a strict
                    # CLEANUP_GRACE_SECS ceiling. The task keeps running in
                    # the background; the per-call 120 s HTTP timeout
                    # (src/models/openaillm.py) and the 15 s browser close
                    # guard (src/tools/auto_browser.py) reap its resources
                    # on their own schedules.
                    await asyncio.wait_for(
                        asyncio.shield(agent_task), timeout=CLEANUP_GRACE_SECS
                    )
                except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                    logger.error(
                        f"Task {example['task_id']}: cancel() did not "
                        f"complete in {CLEANUP_GRACE_SECS}s — abandoning. "
                        f"Proceeding with next task."
                    )
                    # Silence asyncio's "Task was destroyed but it is
                    # pending!" warning that would otherwise fire at
                    # event-loop shutdown. The task continues to run in
                    # the background, bounded by the already-landed HTTP
                    # + browser cleanup timeouts (see CLEANUP_GRACE_SECS
                    # docstring above). This callback registers an
                    # acknowledgement so asyncio considers the task
                    # "retrieved" even though we intentionally drop it.
                    agent_task.add_done_callback(lambda _t: None)
                # Re-raise to hand control to the outer TimeoutError branch,
                # preserving the exact log message + state-mutation semantics
                # the pre-change code produced.
                raise asyncio.TimeoutError(
                    f"Per-question timeout ({per_question_timeout}s) exceeded"
                ) from None

            agent_memory = await agent.write_memory_to_messages(summary_mode=True)
            reformulation_model_id = getattr(config.agent_config, 'model_id', 'gpt-4.1')
            final_result = await prepare_response(
                augmented_question,
                agent_memory,
                reformulation_model=model_manager.registed_models[reformulation_model_id],
            )

            output = str(final_result)
            for memory_step in agent.memory.steps:
                memory_step.model_input_messages = None
            intermediate_steps = _serialize_steps(agent.memory.steps)
            intermediate_steps_str = [str(step) for step in agent.memory.steps]

            parsing_error = any("AgentParsingError" in step for step in intermediate_steps_str)
            iteration_limit_exceeded = "Agent stopped due to iteration limit or time limit." in output
            raised_exception = False
            break

        except asyncio.TimeoutError:
            logger.warning(f"Question timed out after {per_question_timeout}s: {augmented_question[:80]}")
            output = None
            intermediate_steps = []
            iteration_limit_exceeded = True
            exception = asyncio.TimeoutError(f"Per-question timeout ({per_question_timeout}s) exceeded")
            raised_exception = True
            break

        except Exception as e:
            if _is_transient_error(e) and attempt < MAX_RETRIES:
                logger.warning(
                    f"Transient error on attempt {attempt + 1}/{MAX_RETRIES + 1}, "
                    f"retrying in {RETRY_WAIT_SECS}s: {str(e)[:200]}"
                )
                await asyncio.sleep(RETRY_WAIT_SECS)
                continue

            logger.info(f"Error on {augmented_question[:80]}: {e}")
            output = None
            intermediate_steps = []
            exception = e
            raised_exception = True
            break
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    annotated_example = {
        "agent_name": config.agent_config.name,
        "question": example["question"],
        "augmented_question": augmented_question,
        "prediction": output,
        "output": output,
        "intermediate_steps": intermediate_steps,
        "parsing_error": parsing_error,
        "iteration_limit_exceeded": iteration_limit_exceeded,
        "agent_error": str(exception) if raised_exception else None,
        "start_time": start_time,
        "end_time": end_time,
        "task": example["task"],
        "task_id": example["task_id"],
        "true_answer": example["true_answer"],
        "attempts": attempts_made,
    }
    append_answer(annotated_example, config.save_path)

def parse_args():
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument("--config", default=os.path.join(root, "configs", "config_gaia.py"), help="config file path")

    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args

async def main():
    # Parse command line arguments
    args = parse_args()

    # Initialize the configuration
    config.init_config(args.config, args)

    # Initialize the logger
    logger.init_logger(log_path=config.log_path)
    logger.info(f"| Logger initialized at: {config.log_path}")
    try:
        logger.info(f"| Config:\n{config.pretty_text}")
    except Exception as _pt_err:
        logger.warning(f"| Config: pretty_text failed ({type(_pt_err).__name__}); dumping raw dict.")
        logger.info(f"| Config (raw): {dict(config)}")

    # Registed models
    model_manager.init_models(use_local_proxy=getattr(config, 'use_local_proxy', True))
    logger.info("| Registed models: %s", ", ".join(model_manager.registed_models.keys()))
    
    # Load dataset
    dataset = DATASET.build(config.dataset)
    logger.info(f"| Loaded dataset: {len(dataset)} examples.")

    # Load answers
    tasks_to_run = get_tasks_to_run(config.save_path, dataset)
    max_samples = getattr(config, "max_samples", None)
    if max_samples is not None:
        tasks_to_run = tasks_to_run[:int(max_samples)]
        logger.info(f"| Limited to {len(tasks_to_run)} tasks (max_samples={max_samples}).")
    else:
        logger.info(f"| Loaded {len(tasks_to_run)} tasks to run.")

    # Run tasks — streaming worker pool (§P2 of HANDOFF_THROUGHPUT_REFACTOR.md).
    # Replaces the previous batch-gather loop: under that scheme a slow
    # question blocked its batch peers for up to 1800 s even if they were
    # idle, because `asyncio.gather` waits for the slowest in each batch.
    # With a semaphore + TaskGroup, as soon as any worker finishes the
    # next queued task starts — straggler stalls go away.
    #
    # Safety: `answer_single_question` handles its own exceptions on all
    # currently-reachable paths (every path writes a jsonl row via
    # `append_answer` before returning). The real safety net, though, is
    # `_bounded` below — it catches anything unhandled so TaskGroup
    # cannot cancel siblings on a crashed worker. A future change to
    # `answer_single_question` (e.g. a new exception raised from
    # `_serialize_steps` or `append_answer` itself) would still be
    # contained by `_bounded`'s `except Exception` without needing
    # `answer_single_question` to be audited again. Catching Exception
    # (NOT BaseException) keeps CancelledError propagation intact so
    # Ctrl-C still cancels the whole run.
    concurrency = max(1, int(getattr(config, "concurrency", 4)))
    sem = asyncio.Semaphore(concurrency)

    async def _bounded(task_item):
        async with sem:
            try:
                return await answer_single_question(config, task_item)
            except Exception as e:
                logger.exception(
                    f"| worker crashed on task "
                    f"{task_item.get('task_id', '?')}: {e}"
                )
                return None

    async with asyncio.TaskGroup() as tg:
        for task_item in tasks_to_run:
            tg.create_task(_bounded(task_item))

    logger.info(
        f"| All {len(tasks_to_run)} tasks complete (streaming worker pool, "
        f"concurrency={concurrency})."
    )

if __name__ == '__main__':
    asyncio.run(main())