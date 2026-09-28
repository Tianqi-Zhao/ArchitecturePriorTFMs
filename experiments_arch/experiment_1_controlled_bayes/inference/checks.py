"""Small inference checks run explicitly before collecting experimental material."""

from dataclasses import replace

import numpy as np

from ..artifacts.protocol import Gate, Protocol
from .base import PredictionRequest
from .collector import state_plan


def check_task(adapter, task, projection_x, protocol=Protocol(source_tokens=(0,)), *,
               gradients=False):
    """Check sampled queries with full context; raise immediately on a mismatch."""
    p = protocol
    adapter.validate_protocol(p)
    gates = state_plan(p, adapter.token_groups(p.layers))
    limit = max(2, p.query_batch_size + 1)
    queries = {"projection": projection_x[:limit], "evaluation": task.evaluation_x[:limit]}

    def predict(phase, gate, batch_size, *, with_gradients=False):
        x = queries[phase]
        request = PredictionRequest(task, x, gate, p.task_seed(task.task_id))
        output = adapter.predict_batch([request], gradients=with_gradients,
                                       batch_size=batch_size)[0]
        if output.values.shape != (len(x),) or not np.isfinite(output.values).all():
            raise ValueError(f"task={task.task_id} {phase}: invalid predictions")
        if with_gradients and (
                output.gradient_mean is None or output.gradient_count != len(x)
                or output.gradient_mean.shape != (x.shape[1],)
                or not np.isfinite(output.gradient_mean).all()):
            raise ValueError(f"task={task.task_id} {phase}: incomplete or invalid input gradients")
        return output.values

    def compare(actual, expected, phase, gate, message):
        np.testing.assert_allclose(
            actual, expected, atol=p.atol, rtol=p.rtol,
            err_msg=f"task={task.task_id} {phase} {gate}: {message}")

    def checked_prediction(phase, gate):
        value = predict(phase, gate, p.query_batch_size,
                        with_gradients=gradients and phase == "projection")
        alternate = max(1, min(p.query_batch_size // 2, len(queries[phase]) // 2))
        if alternate == p.query_batch_size:
            alternate += 1
        compare(predict(phase, gate, alternate), value, phase, gate,
                "query chunk invariance failed")
        return value

    baseline = {phase: checked_prediction(phase, Gate()) for phase in queries}
    for gate in gates[1:]:
        zero = replace(gate, epsilon=0.0)
        for phase in queries:
            compare(predict(phase, zero, p.query_batch_size), baseline[phase], phase, zero,
                    "zero gate differs from baseline")
            checked_prediction(phase, gate)
            compare(predict(phase, Gate(), p.query_batch_size), baseline[phase], phase, gate,
                    "baseline not restored after intervention")
    return {"task_id": int(task.task_id), "states": len(gates),
            "projection_queries": len(queries["projection"]),
            "evaluation_queries": len(queries["evaluation"])}
