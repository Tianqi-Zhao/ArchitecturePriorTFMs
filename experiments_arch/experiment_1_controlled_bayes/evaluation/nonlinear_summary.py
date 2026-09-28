"""Task-weighted summaries and paired cluster bootstrap for saved IV.7 results."""

from collections import Counter
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from ..artifacts.nonlinear_protocol import ALGORITHM_VERSION, METRICS, RATIOS, SUMMARY_SCHEMA, SummaryConfig
from ..artifacts.nonlinear_results import NonlinearEvaluation
from ..artifacts.nonlinear_storage import analysis_provenance
from ..artifacts.storage import new_directory, reference, sha256, write_json
from .nonlinear_metrics import NonlinearCalculator


@dataclass
class TaskSeries:
    values: np.ndarray
    denominators: np.ndarray | None
    status: np.ndarray
    reason: np.ndarray

    @classmethod
    def read(cls, result, task_ids, metric, statistic):
        denominator = None
        if statistic == "ratio_of_means":
            values, denominator = NonlinearCalculator.ratio_components(result.metrics, metric)
            names = (RATIOS[metric][0], *RATIOS[metric][1])
            status, reason = cls._join_status(result, names)
        elif metric in RATIOS:
            values = result.ratios[metric]
            status, reason = result.ratio_status[metric], result.ratio_reason[metric]
        else:
            values = result.metrics[metric]
            status, reason = result.metric_status[metric], result.metric_reason[metric]
        n = len(task_ids)
        series = cls(np.full(n, np.nan), None if denominator is None else np.full(n, np.nan),
                     np.full(n, "unavailable", dtype=object), np.full(n, "task missing", dtype=object))
        for i, task_id in enumerate(task_ids):
            row = result.index.get(int(task_id))
            if row is not None:
                series.values[i], series.status[i], series.reason[i] = values[row], status[row], reason[row]
                if denominator is not None:
                    series.denominators[i] = denominator[row]
        return series

    @staticmethod
    def _join_status(result, names):
        ok = np.ones(len(result.status), dtype=bool)
        status = np.full(len(result.status), "ok", dtype=object)
        reason = np.full(len(result.status), "", dtype=object)
        for name in names:
            missing = result.metric_status[name] != "ok"
            reason = np.where(ok & missing, result.metric_reason[name], reason)
            ok &= ~missing
        status = np.where(ok, "ok", "unavailable")
        return status, reason


class ClusterBootstrap:
    """Resample whole batches, retaining task weights and a shared draw across models."""

    def __init__(self, config=SummaryConfig()):
        self.config = config

    def summarize(self, series, batch_ids, *, threshold, key):
        valid = np.logical_and.reduce([item.status == "ok" for item in series])
        exclusions = Counter()
        for i in np.flatnonzero(~valid):
            exclusions["; ".join(f"{j}: {item.status[i]}: {item.reason[i]}"
                                  for j, item in enumerate(series) if item.status[i] != "ok")] += 1
        batches, inverse = np.unique(np.asarray(batch_ids)[valid], return_inverse=True)
        n, b = int(valid.sum()), len(batches)
        result = dict(value=np.nan, ci_low=np.nan, ci_high=np.nan,
                      n_eligible=len(valid), n_valid=n, n_excluded=int((~valid).sum()),
                      n_batches=b, n_bootstrap_valid=0, status="unavailable", reason="no valid tasks",
                      ci_status="unavailable", ci_reason="no valid tasks",
                      excluded_reasons=json.dumps(dict(sorted(exclusions.items())), sort_keys=True))
        if not n:
            return result
        sizes = np.bincount(inverse, minlength=b)
        totals = [np.bincount(inverse, weights=item.values[valid], minlength=b) for item in series]
        denominators = [None if item.denominators is None else
                        np.bincount(inverse, weights=item.denominators[valid], minlength=b)
                        for item in series]

        def statistic(weights):
            count = weights @ sizes
            estimates = []
            for total, denominator in zip(totals, denominators):
                mean = (weights @ total) / count
                if denominator is not None:
                    den = (weights @ denominator) / count
                    quotient = np.full(np.shape(mean), np.nan)
                    np.divide(mean, den, out=quotient, where=np.isfinite(den) & (den > threshold))
                    mean = quotient
                estimates.append(mean)
            return estimates[0] if len(estimates) == 1 else estimates[1] - estimates[0]

        result["value"] = float(statistic(np.ones(b, dtype=np.int64)))
        if not np.isfinite(result["value"]):
            result.update(reason="aggregate denominator at or below threshold",
                          ci_reason="point estimate unavailable")
            return result
        result.update(status="ok", reason="", ci_reason="fewer than two valid batches")
        if b < 2:
            return result
        digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).digest()
        seed = np.random.SeedSequence([self.config.bootstrap_seed, int.from_bytes(digest[:16], "big")])
        rng = np.random.Generator(np.random.PCG64(seed))
        draws = []
        for start in range(0, self.config.bootstrap_repeats, 128):
            weights = rng.multinomial(b, np.full(b, 1 / b),
                                      size=min(128, self.config.bootstrap_repeats - start))
            draws.append(statistic(weights))
        draws = np.concatenate(draws)
        result["n_bootstrap_valid"] = int(np.isfinite(draws).sum())
        if result["n_bootstrap_valid"] != self.config.bootstrap_repeats:
            result["ci_reason"] = "some bootstrap denominators are at or below threshold"
            return result
        tail = (1 - self.config.confidence_level) / 2
        low, high = np.quantile(draws, [tail, 1 - tail])
        result.update(ci_low=float(low), ci_high=float(high), ci_status="ok", ci_reason="")
        return result


class NonlinearSummarizer:
    def __init__(self, evaluations, *, labels, comparisons=(), config=SummaryConfig()):
        if not evaluations or len(evaluations) != len(labels) or len(set(labels)) != len(labels):
            raise ValueError("provide one unique label per evaluation")
        if any(not label.strip() or "\n" in label or "\r" in label for label in labels):
            raise ValueError("model labels must be nonempty single-line text")
        self.results = {label: NonlinearEvaluation(path) for label, path in zip(labels, evaluations)}
        if len({result.sha256 for result in self.results.values()}) != len(evaluations):
            raise ValueError("duplicate evaluation input")
        self.comparisons = [tuple(pair) for pair in comparisons]
        if len(set(self.comparisons)) != len(self.comparisons):
            raise ValueError("duplicate comparison")
        for pair in self.comparisons:
            if len(pair) != 2 or pair[0] == pair[1] or any(label not in self.results for label in pair):
                raise ValueError("comparisons require two distinct supplied labels")
            a, b = (self.results[label] for label in pair)
            if a.comparison_config != b.comparison_config:
                raise ValueError("paired evaluations require shared materials and matching evaluation settings")
            self._alignment([a, b])
        self.config, self.bootstrap = config, ClusterBootstrap(config)
        self.has_valid_results = False

    @staticmethod
    def _alignment(results):
        task_ids = sorted(set().union(*(result.index for result in results)))
        rows = []
        for task_id in task_ids:
            sources = [result for result in results if task_id in result.index]
            metadata = [tuple(int(result.tasks[name][result.index[task_id]]) for name in
                              ("source_row", "original_batch_id", "n_eval")) for result in sources]
            if any(item != metadata[0] for item in metadata):
                raise ValueError("paired task source rows, batch IDs or evaluation lengths differ")
            rows.append(dict(task_id=task_id, source_row=metadata[0][0],
                             original_batch_id=metadata[0][1], n_eval=metadata[0][2]))
        return task_ids, rows

    def _summarize(self, labels):
        results = [self.results[label] for label in labels]
        task_ids, metadata = self._alignment(results)
        batch_ids = np.array([row["original_batch_id"] for row in metadata])
        ids = [result.sha256 for result in results]
        paired = len(results) == 2
        group_id = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        identity = dict(group_id=group_id, model_a=labels[0], model_b=labels[1] if paired else "",
                        evaluation_a=ids[0], evaluation_b=ids[1] if paired else "",
                        source_test_sha256=results[0].source_test_sha256,
                        d=results[0].prior.d, k=results[0].prior.k)
        summaries, tasks = [], []
        for metric in (*METRICS, *RATIOS):
            statistics = ("mean_of_ratios", "ratio_of_means") if metric in RATIOS else ("task_mean",)
            for statistic in statistics:
                series = [TaskSeries.read(result, task_ids, metric, statistic) for result in results]
                row = self.bootstrap.summarize(
                    series, batch_ids, threshold=results[0].evaluation_config.denominator_threshold,
                    key=dict(inputs=ids, metric=metric, statistic=statistic))
                summaries.append(dict(**identity, metric=metric, statistic=statistic, **row))
                if statistic == "ratio_of_means":
                    continue
                for i, task in enumerate(metadata):
                    available = all(item.status[i] == "ok" for item in series)
                    reason = "; ".join(f"{labels[j]}: {item.status[i]}: {item.reason[i]}"
                                       for j, item in enumerate(series) if item.status[i] != "ok")
                    value = series[0].values[i]
                    if paired:
                        value = series[1].values[i] - value if available else np.nan
                    tasks.append(dict(**identity, **task, metric=metric,
                                      value_a=series[0].values[i], value_b=series[1].values[i] if paired else np.nan,
                                      value=value, status=("ok" if available else "unavailable") if paired
                                      else series[0].status[i], reason=reason))
        return summaries, tasks

    @staticmethod
    def _write_csv(path, rows):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            for row in rows:
                writer.writerow({key: "" if isinstance(value, (float, np.floating)) and not np.isfinite(value)
                                 else value for key, value in row.items()})

    def _report(self, summaries, paired):
        lines = ["# IV.7 Total Nonlinear Energy Evaluation", "",
                 "Tasks receive equal weight; paired differences are B-A. E is error relative to Bayes.", "",
                 "## Inputs and coverage", "", "| Model | Input ID | Successful / total tasks |",
                 "| --- | --- | --- |"]
        for label, result in sorted(self.results.items()):
            label = label.replace("|", "\\|")
            lines.append(f"| {label} | {result.sha256} | {result.counts['ok']} / {len(result.index)} |")
        for title, rows in (("Single-model results", summaries), ("Paired comparison (B-A)", paired)):
            if not rows:
                continue
            lines += ["", f"## {title}", "", "| Model | Metric | Statistic | Value | Interval | Valid tasks / candidates |",
                      "| --- | --- | --- | --- | --- | --- |"]
            for row in rows:
                label = (f"{row['model_b']} - {row['model_a']}" if row["model_b"] else row["model_a"]).replace("|", "\\|")
                value = f"{row['value']:.6g}" if row["status"] == "ok" else "unavailable"
                interval = (f"[{row['ci_low']:.6g}, {row['ci_high']:.6g}]" if row["ci_status"] == "ok"
                            else row["ci_reason"])
                lines.append(f"| {label} | {row['metric']} | {row['statistic']} | {value} | {interval} | "
                             f"{row['n_valid']} / {row['n_eligible']} |")
        lines += ["", f"Resampled original batches {self.config.bootstrap_repeats} times at "
                  f"confidence level {self.config.confidence_level:g}, seed={self.config.bootstrap_seed}.",
                  "Intervals describe task-sampling uncertainty for the fixed saved queries. "
                  "See the CSV for exclusion reasons and valid batch counts.", ""]
        return "\n".join(lines)

    def run(self, output_dir):
        destination = Path(output_dir).expanduser().resolve()
        summaries, tasks, paired_summaries, paired_tasks = [], [], [], []
        for label in sorted(self.results):
            aggregate, individual = self._summarize([label])
            summaries.extend(aggregate)
            tasks.extend(individual)
        for labels in sorted(self.comparisons):
            aggregate, individual = self._summarize(labels)
            paired_summaries.extend(aggregate)
            paired_tasks.extend(individual)
        self.has_valid_results = any(row["status"] == "ok" for row in summaries)
        with new_directory(destination) as directory:
            tables = {"task_metrics.csv": tasks, "summary.csv": summaries}
            if self.comparisons:
                tables.update({"paired_tasks.csv": paired_tasks, "paired_summary.csv": paired_summaries})
            for name, rows in tables.items():
                self._write_csv(directory / name, rows)
            (directory / "report.md").write_text(self._report(summaries, paired_summaries), encoding="utf-8")
            write_json(directory / "summary.json", dict(
                schema_version=SUMMARY_SCHEMA, algorithm_version=ALGORITHM_VERSION,
                kind="nonlinear_summary", complete=True, config=asdict(self.config),
                inputs={label: reference(result.path, destination) for label, result in sorted(self.results.items())},
                models={label: result.metadata["run"]["model"] for label, result in sorted(self.results.items())},
                comparisons=[dict(a=a, b=b, direction="B-A") for a, b in sorted(self.comparisons)],
                groups={label: result.comparison_config for label, result in sorted(self.results.items())},
                counts={label: result.counts for label, result in sorted(self.results.items())},
                statistics_unit="task", cluster_unit="original_batch_id", rng="PCG64",
                validation_scope="results_and_metadata",
                outputs={name: dict(path=name, sha256=sha256(directory / name)) for name in (*tables, "report.md")},
                provenance=analysis_provenance()))
        return destination / "summary.json"
