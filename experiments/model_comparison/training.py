"""가공 자료부터 학습·VALID 선택·공통 TEST·SHAP까지 재현하는 CPU 실행 모듈."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier


SEED, THREADS, THRESHOLD = 20261030, 2, 0.5
KEY, TARGET = "AUDIT_SAMPLE_KEY", "TARGET_EVENT_RECORDED"
BASE_FEATURES = (
    [f"FUSION_{day:02d}D_MM" for day in range(1, 16)]
    + ["SLOPE_MEAN_SLOPE_DEG", "SLOPE_GE_20DEG_FRACTION",
       "SLOPE_GE_30DEG_FRACTION", "SLOPE_GE_40DEG_FRACTION"]
    + [f"LANDCOVER_L1_{code}_FRACTION_ADMIN" for code in (100, 200, 300, 400, 500, 600, 700)]
)
TM_FEATURES = [f"TM_DEM_{i:03d}" for i in range(384)]
MODEL_NAMES = ("xgboost", "catboost", "terramind4_rf", "terramind16_rf", "terramind384_rf")
RF_PARAMETERS = dict(n_estimators=300, min_samples_leaf=2, max_features="sqrt",
                     class_weight=None, n_jobs=THREADS, random_state=SEED)
XGB_PARAMETERS = dict(n_estimators=300, max_depth=5, learning_rate=0.05,
    tree_method="hist", device="cpu", objective="binary:logistic", eval_metric="logloss",
    n_jobs=THREADS, verbosity=0, random_state=SEED)
CATBOOST_PARAMETERS = dict(iterations=300, depth=6, learning_rate=0.05,
    loss_function="Logloss", random_seed=SEED, thread_count=THREADS,
    task_type="CPU", allow_writing_files=False, verbose=False)
COMPRESSOR_PARAMETERS = dict(hidden_dim=32, fusion_hidden_dim=32, dropout=0.1,
    batch_size=64, max_epochs=200, patience=20, learning_rate=0.001, weight_decay=0.0001)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


@contextmanager
def run_lock(path):
    # 같은 모델·조합을 두 커널에서 동시에 덮어쓰지 못하게 합니다.
    with path.open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def positive_probability(estimator, frame):
    require(set(estimator.classes_) == {0, 1}, "분류기의 타깃이 0/1이 아닙니다.")
    positive = int(np.flatnonzero(estimator.classes_ == 1)[0])
    probability = np.asarray(estimator.predict_proba(frame))[:, positive]
    require(probability.shape == (len(frame),) and np.isfinite(probability).all()
            and ((0 <= probability) & (probability <= 1)).all(), "예측 확률 오류")
    return probability


def metrics_for(y, probability):
    predicted = probability >= THRESHOLD
    tn, fp, fn, tp = [int(v) for v in confusion_matrix(y, predicted, labels=[0, 1]).ravel()]
    return dict(AP=float(average_precision_score(y, probability)), ROC_AUC=float(roc_auc_score(y, probability)),
        ACCURACY=float(accuracy_score(y, predicted)), PRECISION=float(precision_score(y, predicted, zero_division=0)),
        RECALL=float(recall_score(y, predicted)), F1=float(f1_score(y, predicted)),
        BALANCED_ACCURACY=float(balanced_accuracy_score(y, predicted)),
        TN=tn, FP=fp, FN=fn, TP=tp, ROWS=len(y), POSITIVE_ROWS=int(np.asarray(y).sum()), THRESHOLD=THRESHOLD)


def load_data(data_dir, case):
    """원 분할을 유지하고 COMMON_TEST_INCLUDED=True인 TEST 묶음만 사용합니다."""
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    hashes = {name: file_sha(data_dir / name) for name in ("samples.xlsx", "dem_embeddings.parquet")}
    for name, value in hashes.items():
        require(value == manifest["files"][name], f"배포 데이터 해시 불일치: {name}")
    samples = pd.read_excel(data_dir / "samples.xlsx", sheet_name="Samples", engine="openpyxl")
    samples = samples.sort_values(KEY).reset_index(drop=True)
    needed = [KEY, "TRIPLET_ID", "SAMPLE_CLASS", "SPLIT", TARGET,
              "SPATIAL_UNIT_ID", "COMMON_TEST_INCLUDED"] + BASE_FEATURES
    require(set(needed) <= set(samples), "Samples 시트의 필수 열이 없습니다.")
    require(samples[KEY].notna().all() and samples[KEY].is_unique, "표본키 누락·중복")
    require(samples.SAMPLE_CLASS.isin(list("ABC")).all()
            and samples.SPLIT.isin(["TRAIN", "VALID", "TEST", "HOLD"]).all(), "표본 종류·분할 오류")
    groups = samples.groupby("TRIPLET_ID", dropna=False)
    require(samples.TRIPLET_ID.notna().all() and groups.size().eq(3).all()
            and groups.SAMPLE_CLASS.nunique().eq(3).all() and groups.SPLIT.nunique().eq(1).all(), "ABC 묶음·분할 오류")
    require(np.array_equal(samples[TARGET], samples.SAMPLE_CLASS.eq("A").astype(int)), "A=1, B/C=0 타깃 정의 오류")
    original_test = samples.loc[samples.SPLIT.eq("TEST")]
    require(original_test.groupby("TRIPLET_ID").COMMON_TEST_INCLUDED.nunique(dropna=False).eq(1).all(),
            "같은 TEST 묶음의 일부 표본만 제외되었습니다.")
    embeddings = pd.read_parquet(data_dir / "dem_embeddings.parquet")
    require(set(["SPATIAL_UNIT_ID"] + TM_FEATURES) <= set(embeddings)
            and embeddings.SPATIAL_UNIT_ID.notna().all() and embeddings.SPATIAL_UNIT_ID.is_unique, "DEM 특징 키·열 오류")
    selected = samples.loc[samples.SPLIT.isin(["TRAIN", "VALID"])
        | (samples.SPLIT.eq("TEST") & samples.COMMON_TEST_INCLUDED.eq(True))].copy()
    selected = selected.merge(embeddings[["SPATIAL_UNIT_ID"] + TM_FEATURES], on="SPATIAL_UNIT_ID",
                              how="left", validate="many_to_one", sort=False)
    require(np.isfinite(selected[BASE_FEATURES + TM_FEATURES].to_numpy(float)).all(),
            "사용 표본에 결측·무한값이 있습니다. 자동 대체하지 않습니다.")
    if "INPUT_NUMERIC_QC" in selected:
        require(selected.INPUT_NUMERIC_QC.eq("PASS").all(), "입력 수치 QC 검토 필요")
    fractions = BASE_FEATURES[16:19] + BASE_FEATURES[19:]
    require(selected[fractions].ge(0).all().all() and selected[fractions].le(1 + 1e-8).all().all(), "비율은 0~1이어야 합니다.")
    require(selected[BASE_FEATURES[:15]].ge(0).all().all()
            and np.diff(selected[BASE_FEATURES[:15]].to_numpy(float), axis=1).min() >= -1e-8,
            "누적강우량의 음수·기간별 단조성 오류")
    window = ["SPATIAL_UNIT_ID", "FEATURE_WINDOW_START", "FEATURE_WINDOW_END_EXCLUSIVE"]
    if set(window) <= set(samples):
        require(samples.loc[samples.SPLIT.ne("HOLD")].groupby(window, dropna=False).SPLIT.nunique().le(1).all(),
                "같은 구역·강우창이 서로 다른 분할에 있습니다.")
    frames = {}
    for split, expected in (("TRAIN", 966), ("VALID", 322), ("TEST", 321)):
        full = selected.loc[selected.SPLIT.eq(split)]
        require(full.SAMPLE_CLASS.value_counts().to_dict() == {"A": expected, "B": expected, "C": expected},
                f"기존 {split} 표본 수와 다릅니다.")
        frames[split] = full.loc[full.SAMPLE_CLASS.isin(list(case))].sort_values(KEY).reset_index(drop=True)
    require(set(frames["TRAIN"][KEY]).isdisjoint(frames["VALID"][KEY])
            and set(frames["TRAIN"][KEY]).isdisjoint(frames["TEST"][KEY])
            and set(frames["VALID"][KEY]).isdisjoint(frames["TEST"][KEY]), "분할 간 표본 중복")
    return frames, hashes


class DEMCompressor(nn.Module):
    """DEM384→32→k; 물리26과 연결한 보조 분류 head로 TRAIN에서 압축을 학습합니다."""

    def __init__(self, code_dim):
        super().__init__()
        self.dem_encoder = nn.Sequential(nn.Linear(384, 32), nn.ReLU(inplace=False), nn.Dropout(0.1),
            nn.Linear(32, code_dim), nn.ReLU(inplace=False), nn.Dropout(0.1))
        self.fusion_head = nn.Sequential(nn.Linear(26 + code_dim, 32), nn.ReLU(inplace=False),
                                        nn.Dropout(0.1), nn.Linear(32, 1))

    def forward(self, physical, dem):
        return self.fusion_head(torch.cat([physical, self.dem_encoder(dem)], dim=1))


def input_tensors(frame, scaling):
    tensors = []
    for branch, columns in (("physical", BASE_FEATURES), ("dem", TM_FEATURES)):
        stats = scaling[branch]
        values = ((frame[columns].to_numpy(np.float64) - np.asarray(stats["mean"]))
                  / np.asarray(stats["scale"])).astype(np.float32)
        require(np.isfinite(values).all(), "표준화 후 결측·무한값")
        tensors.append(torch.from_numpy(values))
    return tuple(tensors)


def fit_compressor(train, valid, code_dim, report):
    # 초기화·배치 순서·Dropout의 난수 순서를 기존 실험과 같게 유지합니다.
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    scaling = {}
    for branch, columns in (("physical", BASE_FEATURES), ("dem", TM_FEATURES)):
        scaler = StandardScaler().fit(train[columns])
        scaling[branch] = dict(mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist())
    xtr, xva = input_tensors(train, scaling), input_tensors(valid, scaling)
    ytr = torch.tensor(train[TARGET].to_numpy(), dtype=torch.float32).reshape(-1, 1)
    yva = torch.tensor(valid[TARGET].to_numpy(), dtype=torch.float32).reshape(-1, 1)
    loader = DataLoader(TensorDataset(*xtr, ytr), batch_size=64, shuffle=True, drop_last=False,
                        num_workers=0, generator=torch.Generator().manual_seed(SEED))
    model = DEMCompressor(code_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    loss_fn = nn.BCEWithLogitsLoss()
    best_loss, best_epoch, best_state, waiting = float("inf"), 0, None, 0
    history = []
    for epoch in range(1, 201):
        model.train()
        total = 0.0
        for physical, dem, target in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(physical, dem), target)
            require(torch.isfinite(loss).item(), "압축기 TRAIN 손실 오류")
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(target)
        model.eval()
        with torch.no_grad():
            valid_loss = float(loss_fn(model(*xva), yva))
        require(np.isfinite(valid_loss), "압축기 VALID 손실 오류")
        improved = valid_loss < best_loss
        if improved:
            best_loss, best_epoch, waiting = valid_loss, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            waiting += 1
        history.append(dict(EPOCH=epoch, TRAIN_BCE=total / len(train), VALID_BCE=valid_loss, NEW_BEST=improved))
        if epoch == 1 or epoch % 10 == 0 or waiting >= 20:
            report(f"압축기 epoch {epoch}/200(최대 기준 {epoch / 2:.1f}%) | VALID BCE {valid_loss:.5f} | best {best_epoch}")
        if waiting >= 20:
            break
    require(best_state is not None, "최적 압축기 가중치 없음")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    with torch.no_grad():
        require(abs(float(loss_fn(model(*xva), yva)) - best_loss) <= 1e-7, "최적 epoch 복원 실패")
    return scaling, {key: value.numpy().copy() for key, value in best_state.items()}, pd.DataFrame(history), dict(
        best_epoch=best_epoch, best_valid_bce=best_loss, epochs_ran=epoch,
        parameters=sum(p.numel() for p in model.parameters()))


def feature_frame(frame, bundle):
    """RF의 물리 변수는 원단위입니다. 384개 원특징 RF도 표준화하지 않습니다."""
    code_dim = bundle["code_dim"]
    if code_dim is None:
        return frame[bundle["features"]].copy()
    model = DEMCompressor(code_dim)
    model.load_state_dict({key: torch.from_numpy(value) for key, value in bundle["compressor_state"].items()}, strict=True)
    model.eval().requires_grad_(False)
    with torch.inference_mode():
        codes = model.dem_encoder(input_tensors(frame, bundle["scaling"])[1]).numpy().copy()
    require(codes.shape == (len(frame), code_dim) and np.isfinite(codes).all(), "압축 특징 크기·유효값 오류")
    return pd.concat([frame[BASE_FEATURES].reset_index(drop=True),
        pd.DataFrame(codes, columns=[f"DEM_CODE_{i:02d}" for i in range(code_dim)])], axis=1)


def explain_probability(estimator, frame, background, report):
    """TRAIN100 기준 확률 SHAP. RF 보정은 explainer 사본에만 적용합니다."""
    probability = positive_probability(estimator, frame)
    background_probability = positive_probability(estimator, background)
    is_rf = isinstance(estimator, RandomForestClassifier)
    x = frame.to_numpy(np.float32 if is_rf else np.float64)
    bg = background.to_numpy(np.float32 if is_rf else np.float64)
    explainer = shap.TreeExplainer(estimator, data=bg, feature_perturbation="interventional", model_output="probability")
    positive = int(np.flatnonzero(estimator.classes_ == 1)[0])
    checks = dict(method="TreeSHAP_interventional", unit="probability", background_rows=len(bg))
    if is_rf:
        original = explainer.model.thresholds.copy()
        require(np.isfinite(original).all() and not np.any(explainer.model.threshold_types), "수치형 RF 임계값만 지원합니다.")
        rounded = original.astype(np.float32)
        floor = np.where(rounded.astype(float) > original, np.nextafter(rounded, np.float32(-np.inf)), rounded)
        require((floor.astype(float) <= original).all()
            and (np.nextafter(floor, np.float32(np.inf)).astype(float) > original).all(), "RF 분기 경계 동치 검증 실패")
        explainer.model.thresholds[:] = floor.astype(float)
        model_error = float(np.max(np.abs(explainer.model.predict(x)[:, positive] - probability)))
        bg_error = float(np.max(np.abs(explainer.model.predict(bg)[:, positive] - background_probability)))
        require(max(model_error, bg_error) < 1e-12, "RF와 explainer 사본의 예측 불일치")
        expected = float(np.asarray(explainer.expected_value)[positive])
        checks.update(method="TreeSHAP_interventional_float32_floor", explainer_prediction_max_error=model_error,
                      background_prediction_max_error=bg_error)
    else:
        expected = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    require(abs(expected - background_probability.mean()) < 1e-5, "TRAIN100 기준값 불일치")
    values, bases = [], []
    for start in range(0, len(x), 128):
        explanation = explainer(x[start:start + 128], check_additivity=True)
        value, base = np.asarray(explanation.values), np.asarray(explanation.base_values)
        if value.ndim == 3:
            value = value[:, :, positive]
            base = base[:, positive] if base.ndim == 2 else base[positive]
        base = np.broadcast_to(base, (len(value),)).copy()
        require(value.shape == (len(value), frame.shape[1]) and np.isfinite(value).all()
                and np.isfinite(base).all(), "SHAP 차원·유효값 오류")
        values.append(value)
        bases.append(base)
        done = min(start + 128, len(x))
        report(f"TEST SHAP {done:,}/{len(x):,} ({done / len(x):.1%})")
    values, bases = np.concatenate(values), np.concatenate(bases)
    error = float(np.max(np.abs(bases + values.sum(axis=1) - probability)))
    require(error < (1e-6 if is_rf else 1e-5), f"SHAP 합산 검증 실패: {error}")
    checks["max_additivity_error"] = error
    return values, bases, checks


def plot_results(test, probability, scores, output, title):
    fpr, tpr, _ = roc_curve(test[TARGET], probability)
    precision, recall, _ = precision_recall_curve(test[TARGET], probability)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(fpr, tpr, label=f"ROC-AUC {scores['ROC_AUC']:.4f}")
    axes[0].plot([0, 1], [0, 1], color="gray", ls=":")
    axes[0].set(xlabel="False Positive Rate", ylabel="True Positive Rate (Recall)", title="TEST ROC")
    axes[1].step(recall[::-1], precision[::-1], where="post", label=f"AP {scores['AP']:.4f}")
    axes[1].axhline(test[TARGET].mean(), color="gray", ls=":")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="TEST PR")
    for ax in axes:
        ax.set(xlim=(0, 1), ylim=(0, 1.02))
        ax.legend()
        ax.grid(alpha=0.2)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output / "roc_pr.png", dpi=160)
    plt.close(fig)


def run_experiment(model_name, case="ABC", *, data_dir, output_root, log_root, run_shap=True):
    """저장된 분할에서 모델을 새 학습하고 VALID·공통 TEST를 평가합니다.

    출력은 output_root/<model>/<case>, 실행·QC·오류는 log_root/<model>/<case>에 둡니다.
    다른 조건의 기존 결과는 덮어쓰지 않으며 완료 실행은 해시 검증 후 재사용합니다.
    """
    require(model_name in MODEL_NAMES and case in ("AB", "AC", "ABC"), "모델명·표본 조합 오류")
    data_dir = Path(data_dir).resolve()
    output = Path(output_root).resolve() / model_name / case
    log = Path(log_root).resolve() / model_name / case
    require(output != log and output not in log.parents and log not in output.parents, "분석 출력과 로그 폴더를 분리하세요.")
    output.mkdir(parents=True, exist_ok=True)
    log.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(THREADS)
    state = None
    with run_lock(log / "run.lock"), (log / "execution.log").open("a", encoding="utf-8") as execution:
        def report(message):
            line = f"[{model_name}/{case}] {message}"
            print(line, flush=True)
            execution.write(line + "\n")
            execution.flush()

        try:
            old = json.loads((log / "run.json").read_text(encoding="utf-8")) if (log / "run.json").exists() else None
            report("[1/5 · 0%] 입력·분할 검증")
            frames, source_hashes = load_data(data_dir, case)
            source_hashes["manifest.json"] = file_sha(data_dir / "manifest.json")
            code_sha = file_sha(Path(__file__))
            versions = {name: metadata.version(name) for name in
                ("numpy", "pandas", "scikit-learn", "torch", "xgboost", "catboost", "shap", "joblib",
                 "pyarrow", "matplotlib", "threadpoolctl", "openpyxl")}
            code_dim = {"terramind4_rf": 4, "terramind16_rf": 16}.get(model_name)
            features = BASE_FEATURES + ([f"DEM_CODE_{i:02d}" for i in range(code_dim)] if code_dim else
                TM_FEATURES if model_name == "terramind384_rf" else [])
            recipe = dict(model=model_name, case=case, source_hashes=source_hashes, code_sha256=code_sha,
                packages=versions, python=sys.version.split()[0], seed=SEED, threads=THREADS, threshold=THRESHOLD, features=features,
                code_dim=code_dim, rf_parameters=RF_PARAMETERS if "rf" in model_name else None,
                compressor_parameters=COMPRESSOR_PARAMETERS if code_dim else None,
                run_shap=bool(run_shap), split_key_sha256={s: digest(f[KEY].tolist()) for s, f in frames.items()},
                epoch_selection="minimum VALID BCE" if code_dim else "none",
                test_tuning=False, backbone_finetuned=False)
            recipe["classifier_parameters"] = XGB_PARAMETERS if model_name == "xgboost" else (
                CATBOOST_PARAMETERS if model_name == "catboost" else RF_PARAMETERS)
            fingerprint = digest(recipe)
            if old:
                require(old.get("fingerprint") == fingerprint, "기존 실행의 입력·코드·환경·설정과 다릅니다. 새 output/log 경로를 사용하세요.")
                if old["status"] == "COMPLETE":
                    for name, expected in old["artifacts"].items():
                        require(file_sha(output / name) == expected, f"완료 산출물 변경: {name}")
                    report("완료 결과 해시 검증 · 재사용 100%")
                    return old
            else:
                require(not any(output.iterdir()), "기록 없는 기존 결과가 있습니다. 새 output/log 경로를 사용하세요.")
            state = dict(status="RUNNING", fingerprint=fingerprint, recipe=recipe,
                         started=datetime.now(timezone.utc).isoformat())
            write_json(log / "run.json", state)
            train, valid, test = (frames[name] for name in ("TRAIN", "VALID", "TEST"))
            report(f"[2/5 · 20%] TRAIN {len(train):,} · VALID {len(valid):,} · TEST {len(test):,} | 학습 시작")
            bundle = dict(features=features, code_dim=code_dim, threshold=THRESHOLD, recipe=recipe)
            with threadpool_limits(limits=THREADS):
                if code_dim:
                    scaling, compressor_state, curve, training = fit_compressor(train, valid, code_dim, report)
                    bundle.update(scaling=scaling, compressor_state=compressor_state, training=training)
                    curve.to_csv(output / "learning_curve.csv", index=False)
                x_train, x_valid = feature_frame(train, bundle), feature_frame(valid, bundle)
                if model_name == "xgboost":
                    estimator = XGBClassifier(**XGB_PARAMETERS)
                elif model_name == "catboost":
                    estimator = CatBoostClassifier(**CATBOOST_PARAMETERS)
                else:
                    estimator = RandomForestClassifier(**RF_PARAMETERS)
                estimator.fit(x_train, train[TARGET])
                valid_probability = positive_probability(estimator, x_valid)
            bundle["estimator"] = estimator
            temporary = output / "model.joblib.tmp"
            joblib.dump(bundle, temporary, compress=3)
            restored = joblib.load(temporary)
            restored_valid = feature_frame(valid, restored)
            require(np.array_equal(restored_valid.to_numpy(), x_valid.to_numpy()), "모델 재로드 입력 불일치")
            require(np.max(np.abs(positive_probability(restored["estimator"], restored_valid) - valid_probability)) < 1e-12,
                    "모델 재로드 예측 불일치")
            os.replace(temporary, output / "model.joblib")
            model_file_sha = file_sha(output / "model.joblib")
            estimator = restored["estimator"]
            report("[3/5 · 40%] 모델·epoch 고정 완료 | 공통 TEST 평가")
            x_test = feature_frame(test, restored)
            test_probability = positive_probability(estimator, x_test)
            scores = {"VALID": metrics_for(valid[TARGET], valid_probability),
                      "TEST": metrics_for(test[TARGET], test_probability)}
            result_frames = {}
            for split, frame, probability in (("VALID", valid, valid_probability), ("TEST", test, test_probability)):
                result = frame[[KEY, "TRIPLET_ID", "SAMPLE_CLASS", "SPLIT", TARGET]].copy()
                result["P_EVENT_RECORDED"] = probability
                result["PREDICTED_LABEL"] = (probability >= THRESHOLD).astype(int)
                y, pred = result[TARGET].to_numpy(), result.PREDICTED_LABEL.to_numpy()
                result["ERROR_TYPE"] = np.select([(y == 1) & (pred == 1), (y == 1) & (pred == 0),
                                                  (y == 0) & (pred == 1)], ["TP", "FN", "FP"], default="TN")
                result_frames[split] = result
            report("[4/5 · 60%] TRAIN100 배경 TEST SHAP" if run_shap else "[4/5 · 60%] 설정에 따라 SHAP 생략")
            shap_checks = None
            if run_shap:
                background = train.sample(n=100, random_state=SEED).sort_values(KEY)
                background_x = feature_frame(background, restored)
                before = joblib.hash(estimator)
                with threadpool_limits(limits=THREADS):
                    values, bases, shap_checks = explain_probability(estimator, x_test, background_x, report)
                require(joblib.hash(estimator) == before, "SHAP 중 분류기 가중치 변경")
                require(np.max(np.abs(positive_probability(estimator, x_test) - test_probability)) < 1e-12,
                        "SHAP 후 TEST 예측 변경")
                result_frames["TEST"]["BASE_VALUE"] = bases
                result_frames["TEST"] = pd.concat([result_frames["TEST"],
                    pd.DataFrame(values, columns=["SHAP__" + feature for feature in features])], axis=1)
                family = lambda feature: "RAIN" if feature in BASE_FEATURES[:15] else "SLOPE" if feature in BASE_FEATURES[15:19] else "LANDCOVER" if feature in BASE_FEATURES[19:] else "TM_DEM"
                importance = pd.DataFrame(dict(FEATURE=features, MEAN_ABS_SHAP=np.abs(values).mean(axis=0),
                    FAMILY=[family(feature) for feature in features])).sort_values("MEAN_ABS_SHAP", ascending=False)
                importance.to_csv(output / "importance.csv", index=False)
                top = importance.head(20).iloc[::-1]
                fig, ax = plt.subplots(figsize=(9, 6))
                ax.barh(top.FEATURE, top.MEAN_ABS_SHAP)
                ax.set(xlabel="Mean absolute SHAP (probability)", title=f"{model_name} / {case} / TEST")
                fig.tight_layout()
                fig.savefig(output / "shapbar.png", dpi=160)
                plt.close(fig)
                state["background_keys_sha256"] = digest(background[KEY].tolist())
            for split, result in result_frames.items():
                result.to_parquet(output / f"{split.lower()}_predictions.parquet", index=False)
                saved = pd.read_parquet(output / f"{split.lower()}_predictions.parquet")
                require(saved[KEY].tolist() == frames[split][KEY].tolist()
                    and np.array_equal(saved[TARGET], frames[split][TARGET]), "저장 표본키·정답 불일치")
                require(metrics_for(saved[TARGET], saved.P_EVENT_RECORDED.to_numpy()) == scores[split], "저장 지표 재계산 불일치")
            write_json(output / "metrics.json", scores)
            plot_results(test, test_probability, scores["TEST"], output, f"{model_name} / {case}")
            report("[5/5 · 80%] 원본·가중치·저장 결과 검증")
            require(file_sha(output / "model.joblib") == model_file_sha, "학습 후 모델 파일 변경")
            require(all(file_sha(data_dir / name) == expected for name, expected in source_hashes.items()), "실행 중 입력 파일 변경")
            require(file_sha(Path(__file__)) == code_sha, "실행 중 코드 변경")
            state.update(status="COMPLETE", finished=datetime.now(timezone.utc).isoformat(),
                metrics=scores, shap=shap_checks, training=bundle.get("training"),
                artifacts={p.name: file_sha(p) for p in output.iterdir() if p.is_file()})
            write_json(log / "run.json", state)
            report(f"완료 100% | TEST AP {scores['TEST']['AP']:.4f} · ROC-AUC {scores['TEST']['ROC_AUC']:.4f}")
            return state
        except Exception:
            error = traceback.format_exc()
            for path, label in ((data_dir, "<data_dir>"), (output, "<output_dir>"),
                                (log, "<log_dir>"), (Path(__file__).resolve(), "training.py")):
                error = error.replace(str(path), label)
            with (log / "error.log").open("a", encoding="utf-8") as stream:
                stream.write(error + "\n")
            if state is not None:
                state.update(status="FAILED", error=error)
                write_json(log / "run.json", state)
            report("실패 | log의 error.log를 확인하세요.")
            raise


if __name__ == "__main__":
    package = Path(__file__).resolve().parent
    project = package.parent.parent
    parser = argparse.ArgumentParser(description="고정 분할의 산사태 분류 학습·TEST·SHAP")
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--case", choices=("AB", "AC", "ABC"), default="ABC")
    parser.add_argument("--data-dir", type=Path, default=package / "data/processed")
    parser.add_argument("--output-root", type=Path,
                        default=project / "output/model_comparison")
    parser.add_argument("--log-root", type=Path,
                        default=project / "log/model_comparison")
    parser.add_argument("--no-shap", action="store_true")
    args = parser.parse_args()
    run_experiment(args.model, args.case, data_dir=args.data_dir, output_root=args.output_root,
                   log_root=args.log_root, run_shap=not args.no_shap)
