"""Fresh executions of the report protocols, independent of historical artifacts.

Only data and published source/configuration files are inputs. Saved fits belong
to the new output directory and are accepted only with matching input, source,
parameter and row fingerprints. Historical result JSONs are comparison targets,
never predictions or model inputs.
"""
from __future__ import annotations

from dataclasses import replace
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np

from src.evaluation import best_f1_threshold
from src.feature_preprocessing import FeaturePreprocessor, tree_feature_matrices, load_feature_metadata
from src.run_boosting_cv import _array_sha256, _metrics, _resolved_parameters, _variant
from src.run_preprocessing_cv import _as_binary_labels, _load_feature_names, _sha256, _stratified_folds


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


class HistoricalMedianPreprocessor(FeaturePreprocessor):
    """Preserve the report's numeric median fill, including coded categories.

    The maintained preprocessor uses modes for coded categories. Its cleaning,
    output order and binary maps remain applicable; recompute only the numeric
    fill and scaling states using the historical median rule, on fitting rows.
    """

    def fit(self, features):
        super().fit(features)
        states = []
        for state in self._states:
            if state.auxiliary_only or state.one_hot_categories:
                states.append(state)
                continue
            values, _ = self._clean_values(features[:, state.index], state.missing_codes, state.zero_codes)
            missing = ~np.isfinite(values)
            original = values.copy()
            for code, mapped in state.binary_map.items():
                values[original == code] = mapped
            fill = float(np.median(values[~missing]))
            imputed = np.where(missing, fill, values)
            if state.semantic_type == "binary" and state.binary_map:
                mean, scale = 0.0, 1.0
            else:
                mean, scale = float(np.mean(imputed)), float(np.std(imputed))
                if scale == 0.0:
                    scale = 1.0
            states.append(replace(state, fill_value=fill, mean=mean, scale=scale))
        self._states = states
        return self


def historical_designs(variant, train, valid, names, metadata):
    if variant["preprocessing"]["name"] != "codebook_aware":
        raise ValueError("The report linear baselines require codebook-aware preprocessing.")
    processor = HistoricalMedianPreprocessor(names, metadata, variant["preprocessing"]).fit(train)
    return processor.transform(train), processor.transform(valid), list(processor.output_feature_names)


class historical_linear_runner:
    """Temporarily supply the historical designs to a sequential linear runner."""
    def __init__(self, module):
        self.module, self.previous = module, None

    def __enter__(self):
        self.previous = self.module._designs_for_variant
        self.module._designs_for_variant = historical_designs

    def __exit__(self, kind, value, traceback):
        self.module._designs_for_variant = self.previous


def split_rows(labels, fraction=0.2, seed=20260918):
    """Exact original stratified holdout, with signed-label class order."""
    rng = np.random.default_rng(seed)
    training, validation = [], []
    for label in sorted(np.unique(labels)):
        rows = np.flatnonzero(labels == label)
        rng.shuffle(rows)
        count = int(round(rows.size * fraction))
        validation.append(rows[:count])
        training.append(rows[count:])
    return np.sort(np.concatenate(training)), np.sort(np.concatenate(validation))


def model_class(backend):
    if backend == "phase14":
        from src.numpy_boosting_phase14 import HistogramGradientBoostingClassifier
    elif backend == "phase34":
        from src.numpy_boosting import HistogramGradientBoostingClassifier
    elif backend == "categorical":
        from src.numpy_boosting_categorical import HistogramGradientBoostingClassifier
    elif backend == "structural":
        from src.numpy_boosting_structural import HistogramGradientBoostingClassifier
    else:
        raise ValueError(f"Unknown report backend: {backend}")
    return HistogramGradientBoostingClassifier


def sampling_positions(labels, method, ratio, seed, scores=None):
    """Original Phase36 choice and Phase37 nested-permutation procedures."""
    positive, negative = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    if not positive.size or not negative.size:
        raise ValueError("Sampling requires both classes.")
    rng = np.random.default_rng(seed)
    target = min(negative.size, int(ratio * positive.size))
    if method == "random_permutation_prefix":
        chosen = rng.permutation(negative)[:target]
    elif method == "random":
        chosen = rng.choice(negative, target, replace=False)
    elif method == "hybrid_hard":
        if scores is None or scores.shape != labels.shape or not np.all(np.isfinite(scores)):
            raise ValueError("Hybrid sampling requires finite training-only pilot scores.")
        uniform = rng.choice(negative, target // 2, replace=False)
        remaining = np.setdiff1d(negative, uniform, assume_unique=True)
        chosen = np.r_[uniform, remaining[np.lexsort((remaining, -scores[remaining]))[:target-uniform.size]]]
    else:
        raise ValueError("Unknown sampling method.")
    return np.sort(np.r_[positive, chosen])


class ReportRun:
    def __init__(self, root, data_dir, output, smoke=False, smoke_rows=1200):
        self.root, self.data_dir = Path(root).resolve(), Path(data_dir).resolve()
        self.output, self.smoke = Path(output).resolve(), bool(smoke)
        if self.output != self.root and not self.output.is_relative_to(self.root):
            raise ValueError("Report output must be inside this checkout so generated runner records remain portable.")
        self.output.mkdir(parents=True, exist_ok=True)
        sources = [*self.root.joinpath("src").glob("*.py"),
                   *self.root.joinpath("tools").glob("*.py"), self.root / "implementations.py",
                   *self.root.joinpath("configs").rglob("*.json")]
        self.identity = {"sources": {p.relative_to(self.root).as_posix(): _sha256(p) for p in sorted(sources)},
                         "inputs": {n: _sha256(self.data_dir / n) for n in
                                    ("x_train.csv", "y_train.csv", "x_test.csv", "sample_submission.csv")},
                         "environment": {"python": platform.python_version(), "numpy": np.__version__},
                         "smoke": self.smoke, "smoke_rows": smoke_rows if smoke else None}
        manifest = self.output / "run.json"
        if manifest.exists() and read(manifest) != self.identity:
            raise ValueError("Output directory belongs to different sources, data, environment or mode; use a new directory.")
        write(manifest, self.identity)
        from tools.prepare_data import prepare
        self.cache = self.output / "data"
        cache_receipt = self.cache / "receipt.json"
        if cache_receipt.exists():
            receipt = read(cache_receipt)
            if receipt["run_fingerprint"] != fingerprint(self.identity):
                raise ValueError("Prepared numeric cache belongs to another run.")
            for name, expected in receipt["files"].items():
                if _sha256(self.cache / name) != expected:
                    raise ValueError("Prepared numeric cache fingerprint differs: " + name)
        prepare(self.data_dir, self.cache)
        if not cache_receipt.exists():
            write(cache_receipt, {"run_fingerprint": fingerprint(self.identity),
                  "files": {p.name: _sha256(p) for p in self.cache.glob("*.npy")}})
        self.x = np.load(self.cache / "x_train_float32.npy", mmap_mode="r")
        self.test = np.load(self.cache / "x_test_float32.npy", mmap_mode="r")
        signed = np.load(self.cache / "y_train_int8.npy")
        self.y = _as_binary_labels(signed)
        self.names = _load_feature_names(self.data_dir / "x_train.csv")
        self.metadata = load_feature_metadata(self.root / "configs/feature_metadata.json")
        development, validation = split_rows(signed)
        if smoke:
            # Deterministic stratified subset; no full-data metric assertions.
            chosen = []
            rng = np.random.default_rng(20261022)
            for label in (0, 1):
                candidates = np.flatnonzero(self.y == label)
                count = max(60, int(smoke_rows * np.mean(self.y == label)))
                chosen.extend(rng.choice(candidates, min(count, candidates.size), replace=False))
            self.active = np.sort(np.asarray(chosen, dtype=np.int64))
            dp, vp = split_rows(signed[self.active])
            development, validation = self.active[dp], self.active[vp]
            self.test_rows = np.arange(min(100, len(self.test)), dtype=np.int64)
        else:
            self.active = np.arange(len(self.y), dtype=np.int64)
            self.test_rows = np.arange(len(self.test), dtype=np.int64)
        self.development = development
        np.savez_compressed(self.cache / "split.npz", development_indices=development, validation_indices=validation)
        self.run_id = fingerprint(self.identity)

    def suite(self, recipe):
        suite = read(self.root / recipe)
        if "preprocessing_config_path" in suite:
            plan = _variant(read(self.root / suite["preprocessing_config_path"]), suite["preprocessing_variant"])["preprocessing"]
            plan = self.smoke_plan(plan)
        else:
            plan = None
        return suite, plan

    def smoke_plan(self, plan):
        """A declared workflow fixture excludes columns too sparse in its subset.

        Full executions never use this transformation. Small official-data
        subsets cannot reliably populate rare questionnaire branches in all
        nested/selector folds; silently filling them would change the method.
        """
        if not self.smoke or plan.get("name") == "raw_tree_passthrough":
            return plan
        plan = dict(plan)
        processor = FeaturePreprocessor(self.names, self.metadata, plan)
        excluded = list(plan.get("exclude_features", []))
        for name in processor._retained_names():
            entry = processor.metadata_by_name[name]
            raw = self.x[self.active, self.names.index(name)]
            clean, _ = processor._clean_values(raw, processor._missing_codes(name, entry), processor._zero_codes(name, entry))
            if int(np.isfinite(clean).sum()) < 50:
                excluded.append(name)
        plan["exclude_features"] = sorted(set(excluded))
        imputation = plan.get("imputation", {})
        required = set(imputation.get("anchor_features", [])) | set(imputation.get("nonresponse_features", [])) | set(imputation.get("continuous_blank_features", []))
        if required & set(excluded):
            raise ValueError("The smoke cohort cannot exercise required imputation features; increase --smoke-rows.")
        path = self.output / "smoke_fixture.json"
        fixture = read(path) if path.exists() else {"mode": "reduced workflow only",
            "rows": len(self.active), "active_rows_sha256": _array_sha256(self.active),
            "minimum_observed_support": 50, "plans": {},
            "warning": "Full report runs use the unmodified published plans and complete rows."}
        fixture["plans"][fingerprint(plan)] = {"excluded_features": plan["exclude_features"]}
        write(path, fixture)
        return plan

    def adjusted_parameters(self, parameters):
        parameters = dict(parameters)
        if self.smoke:
            parameters.update(n_estimators=3, min_samples_leaf=2, max_depth=min(parameters.get("max_depth", 3), 3))
        return parameters

    def matrix_pair(self, train_rows, validation_rows, plan, test=False):
        raw_train = np.asarray(self.x[train_rows], dtype=np.float32)
        raw_valid = np.asarray((self.test if test else self.x)[validation_rows], dtype=np.float32)
        counts = plan.get("derived_counts", {}).get("groups")
        clean_plan = {k: v for k, v in plan.items() if k != "derived_counts"}
        train, valid, names = tree_feature_matrices(raw_train, raw_valid, self.names, self.metadata, clean_plan)
        if counts:
            from src.count_features import append_count_features
            train, nt = append_count_features(train, names, raw_train, self.names, counts)
            valid, nv = append_count_features(valid, names, raw_valid, self.names, counts)
            if nt != nv:
                raise ValueError("Train/test count feature order differs.")
            names = nt
        return train, valid, names

    def fit_model(self, train, labels, valid, parameters, backend, identity):
        parameters = self.adjusted_parameters(parameters)
        if self.smoke and type(parameters.get("max_features")) is int:
            parameters["max_features"] = min(parameters["max_features"], train.shape[1])
        description = {"run": self.run_id, "identity": identity, "backend": backend, "parameters": parameters,
                       "train": _array_sha256(train), "labels": _array_sha256(labels), "valid": _array_sha256(valid)}
        key = fingerprint(description)
        directory = self.output / "fits" / key
        directory.mkdir(parents=True, exist_ok=True)
        receipt, predictions, checkpoint = directory / "completed.json", directory / "probabilities.npy", directory / "model.pkl"
        cls = model_class(backend)
        if receipt.exists():
            completed = read(receipt)
            if completed["description"] != description or _sha256(predictions) != completed["probabilities_sha256"] or _sha256(checkpoint) != completed["checkpoint_sha256"]:
                raise ValueError("Saved fit fingerprint differs.")
            model, metadata = cls.load_checkpoint(checkpoint)
            if metadata != description:
                raise ValueError("Saved model belongs to another fit.")
            return np.load(predictions, allow_pickle=False), model
        print(f"FIT {key[:12]} rows={len(labels)} features={train.shape[1]} backend={backend}", flush=True)
        if checkpoint.exists():
            model, metadata = cls.load_checkpoint(checkpoint)
            if metadata != description:
                raise ValueError("Partial model belongs to another fit.")
        else:
            model = cls(**parameters)
        steps = list(range(25, parameters["n_estimators"] + 1, 25))
        if parameters["n_estimators"] not in steps:
            steps.append(parameters["n_estimators"])
        def progress(step, fitted):
            fitted.save_checkpoint(checkpoint, description)
            print(f"TREES {key[:12]} {step}/{parameters['n_estimators']}", flush=True)
        if checkpoint.exists():
            prior = len(model.trees_)
            if prior < parameters["n_estimators"]:
                model.continue_fit(train, labels, parameters["n_estimators"]-prior, [s for s in steps if s > prior], progress)
        else:
            model.fit(train, labels, steps, progress)
        model.save_checkpoint(checkpoint, description)
        p = model.predict_proba(valid)[:, 1]
        np.save(predictions, p, allow_pickle=False)
        write(receipt, {"description": description, "probabilities_sha256": _sha256(predictions),
                        "checkpoint_sha256": _sha256(checkpoint)})
        return p, model

    def select_columns(self, train_rows, plan, parameters, seed, count):
        from tools.phase29_feature_selection import selector_partition, PermutationPredictor, importance, ordered_columns, score
        selection_key = fingerprint({"run": self.run_id, "train": _array_sha256(train_rows), "plan": plan,
                                     "parameters": parameters, "seed": seed})
        path = self.output / "selection" / (selection_key + ".json")
        if path.exists():
            result = read(path)
            if result["selection_key"] != selection_key or fingerprint(result["ranking"]) != result["ranking_sha256"]:
                raise ValueError("Saved feature ranking differs.")
            return ordered_columns(result["ranking"], count)[0]
        fit, holdout, sample = selector_partition(self.y[train_rows], train_rows, seed, 5, 100 if self.smoke else 10000)
        train, valid, names = self.matrix_pair(train_rows[fit], train_rows[holdout], plan)
        p, model = self.fit_model(train, self.y[train_rows[fit]], valid, parameters, "phase14",
                                  {"selector": selection_key})
        threshold = best_f1_threshold(self.y[train_rows[holdout]], p).threshold
        inverse = np.full(len(train_rows), -1, dtype=np.int64)
        inverse[holdout] = np.arange(len(holdout))
        predictor = PermutationPredictor(model, valid[inverse[sample]])
        rank_labels = self.y[train_rows[sample]]
        original, _ = score(rank_labels, predictor.original_probabilities, threshold)
        rng = np.random.default_rng(seed + 200000)
        permutations = [rng.permutation(len(sample)) for _ in range(3)]
        ranking = [{"column_index": i, "feature": name,
                    **importance(predictor, [i], rank_labels, threshold, permutations, original),
                    "split_count": int(predictor.counts[i])} for i, name in enumerate(names)]
        write(path, {"selection_key": selection_key, "training_rows_sha256": _array_sha256(train_rows),
                     "selector_fit_rows_sha256": _array_sha256(train_rows[fit]),
                     "holdout_rows_sha256": _array_sha256(train_rows[holdout]),
                     "ranking": ranking, "ranking_sha256": fingerprint(ranking)})
        return ordered_columns(ranking, count)[0]

    def tree_fit(self, suite, plan, train_rows, valid_rows, number, selector_index, entry, test=False):
        if np.intersect1d(train_rows, valid_rows).size and not test:
            raise ValueError("Training and evaluation rows overlap.")
        parameters = {**suite["models"][0]["parameters"], "n_estimators": suite["models"][0]["checkpoints"][-1]}
        backend = entry.get("backend", "phase14")
        eligible = train_rows.copy()
        sampling = suite.get("sampling")
        if sampling:
            seed = sampling["seed"] + number
            method, ratio = sampling["method"], sampling["negative_to_positive_ratio"]
            scores = None
            if method == "hybrid_hard":
                pilot_rows = train_rows[sampling_positions(self.y[train_rows], "random", ratio, seed)]
                pt, pv, pn = self.matrix_pair(pilot_rows, train_rows, plan)
                pp = _resolved_parameters({**parameters, "n_estimators": sampling["pilot_trees"], "max_depth": sampling["pilot_max_depth"]}, pn)
                scores, _ = self.fit_model(pt, self.y[pilot_rows], pv, pp, backend,
                    {"pilot_train": _array_sha256(pilot_rows), "eligible": _array_sha256(train_rows)})
            train_rows = train_rows[sampling_positions(self.y[train_rows], method, ratio, seed, scores)]
        train, valid, names = self.matrix_pair(train_rows, valid_rows, plan, test)
        parameters = _resolved_parameters(parameters, names)
        if entry.get("feature_count"):
            columns = self.select_columns(train_rows, plan, parameters, 20260930 + selector_index, entry["feature_count"])
            train, valid = train[:, columns], valid[:, columns]
            names = [names[i] for i in columns]
            parameters["max_features"] = 128 / 295
        if backend == "categorical":
            parameters["categorical_feature_indices"] = [names.index(n) for n in suite["categorical_features"]]
        identity = {"training_rows": _array_sha256(train_rows), "eligible_rows": _array_sha256(eligible),
                    "validation_rows": _array_sha256(valid_rows), "test": test, "plan": plan,
                    "columns": names, "sampling": sampling,
                    "selector_seed": 20260930 + selector_index if entry.get("feature_count") else None}
        self.last_feature_count = len(names)
        if entry.get("ensemble"):
            p = []
            for seed in entry["ensemble"]:
                predicted, model = self.fit_model(train, self.y[train_rows], valid, {**parameters, "random_seed": seed}, backend, identity)
                p.append(predicted)
                del model
            return np.mean(p, axis=0)
        predicted, model = self.fit_model(train, self.y[train_rows], valid, parameters, backend, identity)
        del model, train, valid
        gc.collect()
        return predicted

    def nested_tree(self, entry):
        suite, plan = self.suite(entry["recipe"])
        rows, y = self.development, self.y[self.development]
        scores, thresholds = np.full(len(rows), np.nan), np.full(len(rows), np.nan)
        folds = np.zeros(len(rows), dtype=np.int8)
        details = []
        for fold, (train, valid) in enumerate(_stratified_folds(y, 3, suite["seed"]), 1):
            p = self.tree_fit(suite, plan, rows[train], rows[valid], fold, (fold-1)*3, entry)
            inner = np.full(len(train), np.nan)
            for number, (it, iv) in enumerate(_stratified_folds(y[train], 2, suite["nested_threshold"]["inner_seed"]+fold), 1):
                inner[iv] = self.tree_fit(suite, plan, rows[train[it]], rows[train[iv]],
                    3+(fold-1)*2+number, (fold-1)*3+number, entry)
            threshold = best_f1_threshold(y[train], inner).threshold
            scores[valid], thresholds[valid], folds[valid] = p, threshold, fold
            details.append({"fold": fold, "threshold": threshold, "metrics": _metrics(y[valid], p, threshold)[0]})
        from src.evaluation import average_precision, binary_log_loss, classification_metrics
        confusion, metrics = classification_metrics(y, scores >= thresholds)
        counts = vars(confusion)
        metrics.update(average_precision=average_precision(y, scores), log_loss=binary_log_loss(y, scores))
        path = self.output / "oof" / (entry["id"] + ".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, rows=rows, labels=y, probabilities=scores, thresholds=thresholds, folds=folds)
        return {"metrics": metrics, "counts": counts, "folds": details,
                "oof": path.relative_to(self.output).as_posix(), "oof_sha256": _sha256(path)}

    def calibration(self, entry):
        suite, plan = self.suite(entry["recipe"])
        rows, y = self.active, self.y[self.active]
        scores = np.full(len(rows), np.nan)
        for fold, (train, valid) in enumerate(_stratified_folds(y, 3, 20260919), 1):
            scores[valid] = self.tree_fit(suite, plan, rows[train], rows[valid], fold, 0, entry)
        threshold = best_f1_threshold(y, scores).threshold
        p = self.tree_fit(suite, plan, rows, self.test_rows, 4, 0, entry, test=True)
        from run import _read_sample_submission_ids, _write_submission
        ids = _read_sample_submission_ids(self.data_dir / "sample_submission.csv")[self.test_rows]
        path = self.output / "submissions" / (entry["id"] + ".csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_submission(path, ids, np.where(p >= threshold, 1, -1))
        oof = self.output / "oof" / (entry["id"] + ".npz")
        oof.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(oof, rows=rows, labels=y, probabilities=scores)
        return {"metrics": _metrics(y, scores, threshold)[0], "threshold": threshold,
                "features": self.last_feature_count,
                "submission": path.relative_to(self.output).as_posix(), "submission_sha256": _sha256(path),
                "positive_predictions": int(np.sum(p >= threshold)), "test_rows": len(p),
                "oof": oof.relative_to(self.output).as_posix(), "oof_sha256": _sha256(oof)}

    def configured_runner(self, entry):
        suite = read(self.root / entry["recipe"])
        output = self.output / "configured" / entry["id"]
        output.mkdir(parents=True, exist_ok=True)
        common = {"features_path": str(self.cache / "x_train_float32.npy"),
                  "labels_path": str(self.cache / "y_train_int8.npy"),
                  "split_indices_path": str(self.cache / "split.npz"),
                  "x_train_csv_path": str(self.data_dir / "x_train.csv")}
        if entry["kind"] == "mlp":
            from src import run_mlp_cv as module
            suite["data"] = common
            suite["artifact_dir"] = str(output / "artifacts")
            suite["models"] = [m for m in suite["models"] if m["name"] == entry["model"]]
            if self.smoke:
                cleaning = read(self.root / suite["cleaning_config_path"])
                for item in cleaning["variants"]:
                    if item["experiment_name"] == suite["cleaning_variant"]:
                        item["preprocessing"] = self.smoke_plan(item["preprocessing"])
                clean_path = output / "cleaning.json"
                write(clean_path, cleaning)
                suite["cleaning_config_path"] = str(clean_path)
                for m in suite["models"]:
                    m["parameters"].update(max_epochs=2, patience=1)
            path = output / "suite.json"
            write(path, suite)
            records = module.run_suite(path, self.root, output / "records", quiet=True)
        else:
            from src import run_model_cv, run_ridge_cv
            module = run_ridge_cv if entry["kind"] == "ridge" else run_model_cv
            preprocessing = read(self.root / suite["preprocessing_config_path"])
            preprocessing["data"] = common
            if self.smoke:
                for item in preprocessing["variants"]:
                    if item["experiment_name"] in suite["preprocessing_variants"]:
                        item["preprocessing"] = self.smoke_plan(item["preprocessing"])
            prep_path = output / "preprocessing.json"
            write(prep_path, preprocessing)
            suite["preprocessing_config_path"] = str(prep_path)
            if self.smoke:
                if entry["kind"] == "ridge":
                    suite["calibration"]["max_iters"] = 10
                else:
                    for m in suite["models"]:
                        m["checkpoints"] = [10]
                    suite["nested_threshold"]["checkpoints"] = [10]
            path = output / "suite.json"
            write(path, suite)
            with historical_linear_runner(module):
                if entry["kind"] == "ridge":
                    records = module.run_suite(path, self.root, output / "records", {0.0001})
                else:
                    records = module.run_suite(path, self.root, output / "records")
        record = read(records[0])
        metrics = (record["outcome"]["metrics"] if entry["kind"] == "ridge" else
                   record["outcome"]["nested_threshold_evaluation"]["pooled_metrics"])
        info = record["outcome"]["oof_predictions_artifact"]
        original = self.root / info["path"]
        if _sha256(original) != info["sha256"]:
            raise ValueError("Configured runner OOF fingerprint differs.")
        with np.load(original, allow_pickle=False) as arrays:
            values = {k: arrays[k] for k in arrays.files}
        thresholds = values.get("applied_thresholds", values.get("nested_thresholds"))
        if thresholds is None:
            thresholds = np.empty(len(values["labels"]), dtype=np.float64)
            for fold in record["outcome"]["nested_threshold_evaluation"]["folds"]:
                thresholds[values["fold_ids"] == fold["outer_fold"]] = fold["inner_threshold"]
        oof = self.output / "oof" / (entry["id"] + ".npz")
        oof.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(oof, rows=values["development_indices"], labels=values["labels"],
                            probabilities=values["probabilities"], thresholds=thresholds, folds=values["fold_ids"])
        return {"metrics": metrics,
                "oof": oof.relative_to(self.output).as_posix(), "oof_sha256": _sha256(oof),
                "record": Path(records[0]).relative_to(self.output).as_posix(), "record_sha256": _sha256(records[0])}

    def execute(self, entry):
        started = time.perf_counter()
        if entry["kind"] == "nested_tree":
            result = self.nested_tree(entry)
        elif entry["kind"] == "calibration":
            result = self.calibration(entry)
        else:
            result = self.configured_runner(entry)
        result.update(experiment=entry["id"], mode="smoke" if self.smoke else "full",
                      run_fingerprint=self.run_id, elapsed_seconds=time.perf_counter()-started,
                      historical_target=entry.get("expected"))
        if self.smoke:
            result["comparison"] = {"status": "not_applicable", "reason": "Reduced workflow check; not report metric reproduction."}
        else:
            target = entry.get("expected", {})
            differences = {k: result["metrics"][k]-v for k, v in target.items() if k in result["metrics"]}
            result["comparison"] = {"status": ("not_compared" if not differences else
                                    "matched" if all(abs(v) <= 1e-6 for v in differences.values()) else "different"),
                                    "absolute_tolerance": 1e-6, "differences": differences}
            if entry.get("expected_threshold") is not None:
                result["comparison"]["threshold_difference"] = result["threshold"]-entry["expected_threshold"]
                if abs(result["comparison"]["threshold_difference"]) > 1e-6:
                    result["comparison"]["status"] = "different"
            if entry.get("expected_features") is not None and result.get("features") != entry["expected_features"]:
                result["comparison"]["status"] = "different"
                result["comparison"]["feature_count_identical"] = False
            if entry.get("expected_submission"):
                expected = entry["expected_submission"]
                if expected.get("sha256") is not None:
                    result["comparison"]["submission_byte_identical"] = result["submission_sha256"] == expected["sha256"]
                if expected.get("positive_predictions") is not None:
                    result["comparison"]["positive_count_identical"] = result["positive_predictions"] == expected["positive_predictions"]
                if result["comparison"].get("submission_byte_identical") is False or result["comparison"].get("positive_count_identical") is False:
                    result["comparison"]["status"] = "different"
        write(self.output / "records" / (entry["id"] + ".json"), result)
        return result

    def data_claims(self):
        rows = self.active
        missing = sum(int(np.isnan(self.x[rows[start:start+4096]]).sum()) for start in range(0, len(rows), 4096))
        suite, plan = self.suite("configs/experiments/final_phase14_validation.json")
        processor = FeaturePreprocessor(self.names, self.metadata, plan)
        metadata = {m["name"]: m for m in self.metadata}
        nonresponse, zero, retained = 0, 0, 0
        for name in processor._retained_names():
            entry = metadata[name]
            if entry.get("semantic_type") not in {"binary", "continuous", "count", "ordinal", "categorical"}:
                continue
            retained += 1
            raw = self.x[self.development, self.names.index(name)]
            clean, _ = processor._clean_values(raw, processor._missing_codes(name, entry), processor._zero_codes(name, entry))
            nonresponse += int(np.sum(np.isfinite(raw) & ~np.isfinite(clean)))
            zero += int(np.sum(np.isfinite(raw) & (raw != 0) & (clean == 0)))
        harehab = self.x[self.development, self.names.index("HAREHAB1")]
        observed = np.isfinite(harehab)
        result = {"training_rows": len(rows), "test_rows": len(self.test_rows), "raw_features": len(self.names),
                  "positives": int(self.y[rows].sum()), "positive_fraction": float(np.mean(self.y[rows])),
                  "missing_cells": missing, "missing_fraction": missing/(len(rows)*len(self.names)),
                  "development_rows": len(self.development), "cleaned_columns": retained,
                  "finite_nonresponses_to_nan": nonresponse, "documented_zero_codes_to_zero": zero,
                  "harehab_raw_responses": int(observed.sum()), "harehab_positive_responses": int(self.y[self.development][observed].sum()),
                  "harehab_valid_responses": int(np.isin(harehab, [1, 2]).sum()),
                  "run_fingerprint": self.run_id, "mode": "smoke" if self.smoke else "full"}
        expected = {"training_rows": 328135, "test_rows": 109379, "raw_features": 321, "positives": 28975,
                    "missing_cells": 47175779, "development_rows": 262508, "cleaned_columns": 295,
                    "finite_nonresponses_to_nan": 1095421, "documented_zero_codes_to_zero": 899687,
                    "harehab_raw_responses": 620, "harehab_positive_responses": 620, "harehab_valid_responses": 607}
        result["comparison"] = {"status": "not_applicable" if self.smoke else
            "matched" if all(result[k] == v for k, v in expected.items()) else "different", "expected": expected}
        write(self.output / "data_claims.json", result)
        return result


def paired_comparisons(output):
    from src.summarize_phase19_transfer import _paired_interval
    output = Path(output)
    results = {}
    for name, baseline, candidate in (("cleaning", "raw321", "phase14"), ("split_search", "phase14", "phase34")):
        paths = [output / "records" / (n + ".json") for n in (baseline, candidate)]
        if not all(p.is_file() for p in paths):
            continue
        records = [read(p) for p in paths]
        if records[0]["run_fingerprint"] != records[1]["run_fingerprint"]:
            raise ValueError("Paired comparisons belong to different executions.")
        arrays = []
        for record in records:
            path = output / record["oof"]
            if _sha256(path) != record["oof_sha256"]:
                raise ValueError("Paired comparison OOF fingerprint differs.")
            with np.load(path, allow_pickle=False) as saved:
                arrays.append({k: saved[k] for k in saved.files})
        b, c = arrays
        for k in ("rows", "labels", "folds"):
            np.testing.assert_array_equal(b[k], c[k])
        bp, cp = b["probabilities"] >= b["thresholds"], c["probabilities"] >= c["thresholds"]
        interval = _paired_interval(b["labels"], bp, cp)
        if name == "cleaning":
            # The historical bootstrap was drawn as raw minus clean. Preserve
            # its exact random draws, then reverse the reported contrast.
            reverse = _paired_interval(b["labels"], cp, bp)
            interval = [-reverse[1], -reverse[0]]
        results[name] = {"baseline": baseline, "candidate": candidate,
            "conditional_f1_difference_interval": interval,
            "f1_difference": records[1]["metrics"]["f1"]-records[0]["metrics"]["f1"],
            "confusion_count_differences": {k: records[1]["counts"][k]-v for k, v in records[0]["counts"].items()},
            "scope": "Paired stratified row bootstrap, fixed fits and thresholds; 20000 repetitions, seed20260929."}
        target = ([-0.0012257231537289115, 0.0026348459934233044] if name == "cleaning" else
                  [-0.0030049692460550823, 0.00031912702088536326])
        expected_counts = ({"true_positives": -2, "false_positives": -99, "true_negatives": 99, "false_negatives": 2}
                           if name == "cleaning" else
                           {"true_positives": 206, "false_positives": 907, "true_negatives": -907, "false_negatives": -206})
        results[name]["comparison"] = {"status": "not_applicable" if records[0]["mode"] == "smoke" else
            "matched" if np.allclose(interval, target, atol=1e-6, rtol=0) and results[name]["confusion_count_differences"] == expected_counts else "different",
            "expected_interval": target, "expected_count_differences": expected_counts}
    write(output / "paired_comparisons.json", results)
    return results
