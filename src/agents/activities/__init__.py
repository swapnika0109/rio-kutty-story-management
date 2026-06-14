"""Shared helpers for activity agents."""

import json

from ...utils.logger import setup_logger

logger = setup_logger(__name__)


# Per-metric pass thresholds. A metric is treated as "needs fixing" only when
# its score is BELOW its threshold. Filtering by score is robust to whatever
# free-form reason text the judge writes (string-match filters were fragile —
# Gemini's wording varies enough that pass reasons leaked through).
#
# Mirrors _ACTIVITY_HARD_METRICS and PASS_THRESHOLD in evaluation_agent.py.
# Kept here as a local copy (no cross-package import) because evaluation_agent
# pulls in DeepEval and Gemini SDKs that we don't want loaded just to format a
# retry prompt.
_METRIC_THRESHOLDS: dict[str, float] = {
    # Rubric collapsed to 3 metrics. Must mirror _ACTIVITY_HARD_METRICS +
    # _ACTIVITY_SOFT_METRICS in evaluation_agent.py.
    "non_toxicity":        0.8,
    "engagability":        0.6,
    "age_appropriateness": 0.6,
}


def _prepend_retry_feedback(prompt: str, state: dict, activity_type: str) -> str:
    """If the prior eval pass failed for this activity, prepend a corrective
    block listing ONLY the metrics that scored below their threshold — and
    their reasons — so the next generation targets actual failures instead
    of chasing metrics that already passed.

    No-op on first attempt (no _eval_<type> in state).
    """
    eval_record = (state.get("activities") or {}).get(f"_eval_{activity_type}")
    if not eval_record or eval_record.get("passed"):
        return prompt

    reason = eval_record.get("reason") or ""
    metrics = eval_record.get("metrics") or {}
    metric_reasons = eval_record.get("metric_reasons") or {}

    lines = [
        "PREVIOUS ATTEMPT FAILED EVALUATION — you MUST fix these issues:",
        f"- Overall: {reason}",
    ]
    fixable_metrics: list[str] = []
    for metric, score in metrics.items():
        threshold = _METRIC_THRESHOLDS.get(metric)
        if threshold is None:
            # Unknown metric — surface it just in case rather than silently drop.
            threshold = 0.6
        if score is None or score >= threshold:
            continue
        why = metric_reasons.get(metric) or "(no reason provided)"
        lines.append(f"- {metric} ({score:.2f}<{threshold}): {why}")
        fixable_metrics.append(metric)

    if not fixable_metrics:
        # Eval said FAILED but every individual metric is at/above its
        # threshold — fall back to surfacing the overall reason alone.
        # Happens occasionally with judge-inconsistency or rounding edges.
        lines.append("- (No specific metric below threshold — see overall reason above.)")

    # Show the agent its OWN previous attempt so it REVISES that draft to fix the
    # issues above, instead of regenerating blind and possibly repeating them.
    prev_activity = (state.get("activities") or {}).get(activity_type)
    if prev_activity is not None:
        try:
            prev_json = json.dumps(prev_activity, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            prev_json = str(prev_activity)
        lines.append("")
        lines.append("YOUR PREVIOUS ATTEMPT (revise THIS to fix the issues above; keep what already worked):")
        lines.append(prev_json)

    lines.append("")
    lines.append("Regenerate the activity addressing each issue above. Stay tightly tied to the story's characters, events, and setting.")
    lines.append("")

    logger.info(
        f"[{activity_type}] Prepending retry feedback (prev score={eval_record.get('score')}, "
        f"fixable_metrics={fixable_metrics})"
    )
    return "\n".join(lines) + "\n" + prompt
