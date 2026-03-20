"""Codex (GPT-5.4) advisor — consulted at key decision points."""

import subprocess
import logging
import json
from pathlib import Path

log = logging.getLogger(__name__)

CODEX_LOG = Path(__file__).parent / "codex_log.md"
SESSION_FILE = Path(__file__).parent / ".codex_session"


def _run_codex(prompt: str, timeout: int = 120) -> str | None:
    """Run codex exec and return output. Returns None on failure."""
    try:
        result = subprocess.run(
            [
                "codex", "exec",
                "-s", "read-only",
                "-m", "gpt-5.4",
                "-c", "model_reasoning_effort='high'",
                prompt,
            ],
            capture_output=True, text=True, timeout=timeout,
            cwd=str(Path(__file__).parent),
        )
        output = result.stdout.strip()
        if result.returncode != 0:
            log.warning(f"Codex returned {result.returncode}: {result.stderr[:200]}")
        return output if output else None
    except subprocess.TimeoutExpired:
        log.warning(f"Codex timed out after {timeout}s")
        return None
    except FileNotFoundError:
        log.warning("Codex CLI not found")
        return None
    except Exception as e:
        log.warning(f"Codex error: {e}")
        return None


def _log_advice(event: str, advice: str):
    """Append Codex advice to persistent log."""
    import time
    now = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    entry = f"\n## [{now}] {event}\n\n{advice}\n"
    with open(CODEX_LOG, "a") as f:
        f.write(entry)


def on_round_scored(round_num: int, score: float, round_z: float | None):
    """Called after a round scores and GT is harvested."""
    log.info(f"Consulting Codex on round {round_num} results...")

    cal_file = Path(__file__).parent / "calibration.json"
    cal_summary = ""
    if cal_file.exists():
        cal = json.loads(cal_file.read_text())
        z_vals = cal.get("round_z", {})
        cal_summary = f"Round z values: {json.dumps(z_vals)}, z_mean={cal.get('z_mean', '?')}"

    prompt = f"""Round {round_num} just scored {score:.1f}. {cal_summary}

Our z-conditioned Dirichlet model uses P(class|features,z) = intercept + slope*z.
Backtested: oracle-z LORO avg 82.9, z-mean LORO avg 69.9, global-mean LORO avg 70.1.

The z=mean approach is equivalent to global mean. Gains only come from knowing z.
Mid-sim survival is unreliable for z estimation (round 4: 100% mid-sim, 4-18% end-state).

Questions:
1. Given round {round_num} scored {score:.1f}, does this suggest our z estimate was off?
2. Are there better signals than survival_rate to estimate z from mid-sim observations?
3. Should we adjust confidence (currently 30) in the z-conditioned prior?
4. Any quick wins for the next round?

Be concise (under 200 words). Actionable suggestions only."""

    advice = _run_codex(prompt)
    if advice:
        _log_advice(f"Round {round_num} scored {score:.1f}", advice)
        log.info(f"Codex advice logged for round {round_num}")
    return advice


def on_pre_solve(round_num: int, dynamics: dict | None, z_estimate: float | None):
    """Called before submitting predictions for a new round."""
    log.info(f"Consulting Codex before solving round {round_num}...")

    dyn_str = json.dumps(dynamics) if dynamics else "no dynamics yet"
    prompt = f"""About to submit predictions for round {round_num}.
Estimated z={z_estimate}, dynamics: {dyn_str}

Our z-conditioned model: P(class|key,z) = intercept + slope*z.
z range from GT: 0.018 (catastrophic) to 0.419 (healthy).

Should we use z={z_estimate} or adjust? Consider:
- Mid-sim survival is unreliable (round 4: 100% mid-sim → 4-18% end-state)
- z_mean={z_estimate} hedges across regimes
- Only extreme signals (survival<0.5, food<10) reliably indicate catastrophe

Reply in under 100 words: recommended z value and reasoning."""

    advice = _run_codex(prompt, timeout=90)
    if advice:
        _log_advice(f"Pre-solve round {round_num}, z={z_estimate}", advice)
        log.info(f"Codex pre-solve advice for round {round_num}")
    return advice


def on_calibration_update(n_files: int, round_z: dict):
    """Called after recalibration with new GT data."""
    log.info("Consulting Codex on calibration update...")

    prompt = f"""Just recalibrated with {n_files} GT files.
Round z values: {json.dumps(round_z)}

Our z-conditioned linear model has 27 keys. Codex previously found:
- Oracle z LORO: 82.9 avg
- Global mean LORO: 70.1 avg
- More keys (310, 475) don't help (overfit)

With {n_files} files now, should we:
1. Add more feature keys (more data reduces overfitting)?
2. Try non-linear z conditioning (quadratic, piecewise)?
3. Change the confidence scaling?

Reply in under 150 words. Only suggest changes with clear expected gain."""

    advice = _run_codex(prompt, timeout=90)
    if advice:
        _log_advice(f"Calibration update ({n_files} files)", advice)
        log.info("Codex calibration advice logged")
    return advice
