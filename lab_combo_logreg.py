import argparse
import io
import itertools
import math
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from tqdm import tqdm

import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

RANDOM_STATE = 42
TOP_MODE = "fraction"    # "fraction" или "count"
TOP_FRACTION = 0.10      # используется при TOP_MODE="fraction"
TOP_COUNT = 10_000

CHUNK_MODELS = 100_000   # сколько моделей считается между обновлениями top-буфера
BUFFER_FACTOR = 1.5      # запас буфера над ожидаемым размером top при --top-mode fraction

TEST_SIZE = 0.2
# Мин. число строк ПОСЛЕ балансировки 1:1. При меньшем размере accuracy
# на test-выборке статистически почти ничего не значит.
MIN_SAMPLES = 60
WINDOW_DAYS = 30          # длина скользящего окна (в днях), "месяц" = WINDOW_DAYS дней

# Случайные состояния для train/test-разбиений внутри fit_combo_accuracy_only()/
# fit_combo_model(). Модель обучается и тестируется на нескольких разных
# разбиениях одного и того же сбалансированного датасета, итоговая
# accuracy - среднее по ним (устойчивее к случайности сплита).
SPLIT_RANDOM_STATES: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

# Тесты, содержащие эту подстроку в test_type_code, исключаются из
# кандидатов на комбинирование (аналог старого фильтра "RET" по столбцам).
EXCLUDE_TEST_SUBSTR = "RET"


@dataclass
class ModelRecord:
    """Одна обученная модель: конкретный набор из n_features колонок (по
    одной статистике на каждый тест из test_combo) на одном месяце.

    accuracy       - СРЕДНЯЯ accuracy по нескольким (SPLIT_RANDOM_STATES)
                     train/test-разбиениям одного и того же сбалансированного
                     датасета.
    accuracy_std   - выборочное std этих accuracy по разбиениям (0, если
                      разбиение было только одно).
    accuracies     - accuracy по каждому отдельному разбиению (для контроля).
    model          - финальная модель, дообученная на ВСЕХ сбалансированных
                     данных (без hold-out) - для последующего инференса;
                     accuracy для ранжирования при этом берётся из
                     hold-out разбиений, а не с этой модели.
    """
    month: int
    n_features: int
    combo: Tuple[str, ...]          # тесты (test_type_code) в комбинации
    feature_cols: Tuple[str, ...]   # ровно n_features колонок - по одной статистике на каждый тест из combo
    accuracy: float
    n_train: int
    n_test: int
    model: LogisticRegression
    random_state: int = RANDOM_STATE
    accuracy_std: float = 0.0
    accuracies: Tuple[float, ...] = field(default_factory=tuple)

    @property
    def name(self) -> str:
        return "+".join(self.feature_cols)


# --------------------------------------------------------------------------
# 1. Загрузка и общая предобработка сырых данных (1 строка = 1 измерение)
# --------------------------------------------------------------------------

def load_raw_dataset(path) -> pd.DataFrame:
    """Загружает один сырой .tsv/.csv файл с измерениями (длинный формат)."""
    path = Path(path)
    sep = "\t" if path.suffix == ".tsv" else ","
    df = pd.read_csv(path, sep=sep)
    if df.shape[1] == 1:
        # разделитель определился неверно - пробуем автоопределение
        df = pd.read_csv(path, sep=None, engine="python")
    return df


def timing(df: pd.DataFrame) -> pd.DataFrame:
    """Разбирает даты, пересчитывает target по death_date, зануляет
    нулевые test_result и кодирует пол. Работает на сыром "длинном" df,
    применяется один раз ко всему датасету."""
    df = df.copy()
    df["death_date"] = pd.to_datetime(df["death_date"], errors="coerce")
    df["test_date"] = pd.to_datetime(df["test_date"], errors="coerce")
    df["birth_date"] = pd.to_datetime(df["birth_date"], errors="coerce")
    df["days_to_death"] = (df["death_date"] - df["test_date"]).dt.days
    df["target"] = np.where(df["death_date"].notna(), 1, 0)
    df["test_result"] = df["test_result"].replace(0, np.nan)
    df["gender"] = df["gender"].replace({"Мужской": 1, "Женский": 0})
    return df


def add_global_percentiles(
    df: pd.DataFrame,
    value_col: str = "test_result",
    group_col: str = "test_type_code",
    out_col: str = "test_result_percentile",
) -> pd.DataFrame:
    """
    Добавляет в ДЛИННУЮ таблицу (1 измерение = 1 строка) колонку out_col -
    перцентильный ранг (0..1) значения value_col ОТНОСИТЕЛЬНО ВСЕГО
    распределения измерений ЭТОГО ЖЕ теста (group_col) по всей популяции
    и по всему времени наблюдения - в т.ч. учитываются измерения, которые
    для конкретного пациента могли бы считаться "будущими" (никакого
    ограничения по index_date/дате пациента здесь больше нет).

    Считается ОДИН РАЗ для всего датасета (а не отдельно для каждой
    комбинации тестов и месяца, как раньше) - результат затем просто
    агрегируется (mean/q25/median/q75 за окно) наравне с абсолютным
    значением теста, см. feature_engineering().

    NaN в value_col (после timing() это, например, обнулённые измерения)
    остаются NaN и в out_col - как и абсолютное значение, они будут
    считаться пропуском при построении признаков.
    """
    df = df.copy()
    df[out_col] = df.groupby(group_col)[value_col].rank(pct=True, method="average")
    return df


def load_allowed_tests_file(path: Optional[str]) -> Optional[List[str]]:
    """Необязательный список test_type_code-кандидатов (по одному на
    строку) - аналог старого общего features.txt."""
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Файл со списком тестов не найден: {path}")
    with open(p, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def list_candidate_tests(
    df: pd.DataFrame,
    exclude_substr: str = EXCLUDE_TEST_SUBSTR,
    allowed_tests: Optional[Sequence[str]] = None,
) -> List[str]:
    """Список test_type_code - кандидатов для комбинирования. allowed_tests
    (если задан) сужает список; exclude_substr всегда отфильтровывает
    совпадающие тесты (например, "RET")."""
    all_tests = sorted(df["test_type_code"].dropna().unique().tolist())
    if allowed_tests:
        allowed_set = set(allowed_tests)
        all_tests = [t for t in all_tests if t in allowed_set]
    if exclude_substr:
        all_tests = [t for t in all_tests if exclude_substr.lower() not in str(t).lower()]
    return all_tests


# --------------------------------------------------------------------------
# 2. Feature engineering для ОДНОЙ комбинации тестов и ОДНОГО месяца
# --------------------------------------------------------------------------

def _window_stats(g) -> pd.DataFrame:
    """mean/q25/median/q75 по группам SeriesGroupBy - векторно
    (groupby.quantile сразу для списка квантилей), без Python-лямбды на
    каждую группу (пациент, тест). Используется одинаково и для
    абсолютного значения теста, и для его глобального перцентиля."""
    q = g.quantile([0.25, 0.75]).unstack()
    return pd.DataFrame({"mean": g.mean(), "q25": q[0.25], "median": g.median(), "q75": q[0.75]})


def feature_engineering(
    df: pd.DataFrame,
    n_month: int,
    window_days: int = WINDOW_DAYS,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """
    df уже отфильтрован по нужной комбинации test_type_code (df_loc) и уже
    содержит колонку test_result_percentile (см. add_global_percentiles(),
    вызывается один раз для всего датасета до цикла по комбинациям/месяцам).

    "Живым" (target==0) пациентам сопоставляет случайное окно длиной
    window_days где-то между первым тестом + 30 дней и (последний тест -
    window_days*n_month), т.е. имитирует снэпшот "как будто до конца
    наблюдения осталось n_month месяцев". Случайность берётся из rng -
    передавайте генератор с seed, иначе результат невоспроизводим.

    "Умершим" (target==1) пациентам сопоставляет измерения, попавшие в
    окно [window_days*n_month, window_days*(n_month+1)) дней до смерти.

    Для каждого теста строит:
      - mean/q25/median/q75_<тест>_<Nd>     - статистики АБСОЛЮТНОГО значения за окно
      - mean/q25/median/q75_pct_<тест>_<Nd> - статистики ГЛОБАЛЬНОГО ПЕРЦЕНТИЛЯ за окно
    Итого 8 кандидатов-столбцов на каждый тест (см. test_feature_candidates()).

    Пропуски (пациент без данных по какому-то из тестов combo в этом окне)
    здесь НЕ отбрасываются - widetable сохраняет NaN как есть; какие
    именно строки годны для конкретной модели, решает
    fit_combo_accuracy_only()/fit_combo_model() по своему набору feature_cols.
    """
    if rng is None:
        rng = np.random.default_rng()

    patients = (
        df[df["target"] == 0]
        .groupby("patient_id")
        .agg(min_start_date=("test_date", "min"), last_test_date=("test_date", "max"))
    )
    patients["max_end_date"] = patients["last_test_date"] - pd.Timedelta(days=window_days * n_month)

    min_end_date = patients["min_start_date"] + pd.Timedelta(days=window_days)
    span = (patients["max_end_date"] - min_end_date).dt.total_seconds()

    offsets = pd.to_timedelta(
        rng.random(len(patients)) * span.clip(lower=0).to_numpy(),
        unit="s",
    )
    patients["random_end_date"] = (min_end_date + offsets).where(span >= 0)
    patients["random_start_date"] = patients["random_end_date"] - pd.Timedelta(days=window_days)

    df = df.merge(patients[["random_start_date", "random_end_date"]], on="patient_id", how="left")

    mask_dead = (
        (df["target"] == 1)
        & (df["days_to_death"] >= window_days * n_month)
        & (df["days_to_death"] < window_days * (n_month + 1))
    )
    mask_alive = (
        (df["target"] == 0)
        & (df["test_date"] >= df["random_start_date"])
        & (df["test_date"] <= df["random_end_date"])
    )

    df_window = pd.concat([df[mask_dead], df[mask_alive]]).copy()
    if df_window.empty:
        return df_window.assign(patient_id=[], target=[])

    df_age = df_window.groupby("patient_id").agg(target=("target", "first"))
    suffix = window_days * n_month

    df_agg_raw = (
        df_window.groupby(["patient_id", "test_type_code"])["test_result"].pipe(_window_stats).unstack()
    )
    df_agg_raw.columns = [f"{stat}_{test}_{suffix}d" for stat, test in df_agg_raw.columns]

    df_agg_pct = (
        df_window.groupby(["patient_id", "test_type_code"])["test_result_percentile"].pipe(_window_stats).unstack()
    )
    df_agg_pct.columns = [f"{stat}_pct_{test}_{suffix}d" for stat, test in df_agg_pct.columns]

    df_features = pd.concat([df_age, df_agg_raw, df_agg_pct], axis=1).reset_index()
    return df_features


def build_feature_table(
    df_timed: pd.DataFrame,
    combo: Tuple[str, ...],
    n_month: int,
    window_days: int = WINDOW_DAYS,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """Препроцессинг ДЛЯ ОДНОЙ комбинации тестов и ОДНОГО месяца:
    фильтрация по тестам -> feature_engineering(). Глобальные перцентили
    уже посчитаны заранее (один раз для всего датасета, до этого вызова -
    см. add_global_percentiles()) и просто агрегируются здесь наравне с
    абсолютными значениями."""
    df_loc = df_timed[df_timed["test_type_code"].isin(combo)]
    return feature_engineering(df_loc, n_month, window_days=window_days, rng=rng)


def test_feature_candidates(test_name: str, suffix_days: int) -> List[str]:
    """Список из 8 колонок-кандидатов, которые feature_engineering строит
    ДЛЯ ОДНОГО теста (4 по абсолютному значению: mean/q25/median/q75
    за окно, и 4 те же статистики, но по глобальному перцентилю теста).
    Именно из этого списка для каждого теста в test_combo выбирается
    РОВНО ОДНА статистика при переборе моделей."""
    return [
        f"mean_{test_name}_{suffix_days}d",
        f"q25_{test_name}_{suffix_days}d",
        f"median_{test_name}_{suffix_days}d",
        f"q75_{test_name}_{suffix_days}d",
        f"mean_pct_{test_name}_{suffix_days}d",
        f"q25_pct_{test_name}_{suffix_days}d",
        f"median_pct_{test_name}_{suffix_days}d",
        f"q75_pct_{test_name}_{suffix_days}d",
    ]


# --------------------------------------------------------------------------
# 3. Балансировка 1:1, обучение и оценка одного набора колонок
# --------------------------------------------------------------------------

def _balance_1to1(df: pd.DataFrame, target_col: str, random_state: int) -> pd.DataFrame:
    """Undersampling: оставляет одинаковое число строк каждого класса
    (по размеру меньшего класса), независимо от исходного дисбаланса."""
    counts = df[target_col].value_counts()
    if len(counts) < 2:
        return df.iloc[0:0]
    n = counts.min()
    parts = [df[df[target_col] == cls].sample(n=n, random_state=random_state) for cls in counts.index]
    return pd.concat(parts, ignore_index=True)


def fit_combo_accuracy_only(
    df_features: pd.DataFrame,
    feature_cols: Tuple[str, ...],
    target_col: str = "target",
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    min_samples: int = MIN_SAMPLES,
    split_random_states: Sequence[int] = SPLIT_RANDOM_STATES,
) -> Optional[Tuple[float, float, Tuple[float, ...], int, int]]:
    """Balanced hold-out оценка ОДНОГО набора feature_cols - БЕЗ
    дообучения финальной модели на всех данных и БЕЗ возврата
    сбалансированного датасета.

    Возвращает (accuracy, accuracy_std, accuracies, n_train, n_test) или
    None, если этот набор колонок непригоден (нет данных, < 2 классов,
    меньше min_samples строк после балансировки и т.п.).
    """
    if df_features.empty:
        return None
    missing = [c for c in feature_cols if c not in df_features.columns]
    if missing:
        return None

    needed_cols = list(feature_cols) + [target_col]
    sub = df_features[needed_cols].dropna(subset=needed_cols)
    if sub[target_col].nunique() < 2:
        return None

    balanced = _balance_1to1(sub, target_col, random_state)
    if len(balanced) < min_samples:
        return None

    accuracies: List[float] = []
    train_sizes: List[int] = []
    test_sizes: List[int] = []
    for split_rs in split_random_states:
        try:
            train_df, test_df = train_test_split(
                balanced, test_size=test_size, stratify=balanced[target_col], random_state=split_rs
            )
        except ValueError:
            continue
        if train_df[target_col].nunique() < 2 or len(test_df) == 0:
            continue

        split_model = LogisticRegression(max_iter=1000, random_state=random_state)
        split_model.fit(train_df[list(feature_cols)], train_df[target_col])
        y_pred = split_model.predict(test_df[list(feature_cols)])
        accuracies.append(accuracy_score(test_df[target_col], y_pred))
        train_sizes.append(len(train_df))
        test_sizes.append(len(test_df))

    if not accuracies:
        return None

    mean_acc = float(np.mean(accuracies))
    acc_std = float(np.std(accuracies, ddof=1)) if len(accuracies) > 1 else 0.0
    return mean_acc, acc_std, tuple(accuracies), int(round(np.mean(train_sizes))), int(round(np.mean(test_sizes)))


def _logreg_l2_newton(
    X: np.ndarray,
    y: np.ndarray,
    C: float = 1.0,
    max_iter: int = 100,
    tol: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """Пакетная L2-логрегрессия методом Ньютона: B независимых задач за раз.

    X: (B, n, f) признаки, y: (B, n) метки 0/1. Минимизирует ту же функцию,
    что LogisticRegression(C=C) (lbfgs) из sklearn:
        C * sum(log(1 + exp(z)) - y*z) + 0.5 * ||w||^2,  z = X w + b,
    интерсепт b не штрафуется. Отличие от sklearn только в оптимизаторе:
    lbfgs останавливается при tol=1e-4, Ньютон доходит до точного оптимума.

    Возвращает (coef_и_интерсепт (B, f+1), converged (B,)).
    """
    B, n, f = X.shape
    Xa = np.concatenate([X, np.ones((B, n, 1))], axis=2)
    d = f + 1
    reg = np.ones(d)
    reg[-1] = 0.0
    # крошечная добавка к диагонали гессиана: меняет только шаг, а не точку
    # оптимума (там градиент = 0), и страхует solve() от вырожденной матрицы
    jitter = np.diag(reg + 1e-12)

    def objective(w, idx):
        z = np.einsum("bnd,bd->bn", Xa[idx], w)
        return C * np.sum(np.logaddexp(0.0, z) - y[idx] * z, axis=1) + 0.5 * np.sum(reg * w * w, axis=1)

    w = np.zeros((B, d))
    all_idx = np.arange(B)
    f_cur = objective(w, all_idx)
    converged = np.zeros(B, dtype=bool)

    for _ in range(max_iter):
        idx = np.flatnonzero(~converged)
        if idx.size == 0:
            break
        Xi = Xa[idx]
        wi = w[idx]
        # сигмоида через tanh: без переполнения exp() при больших |z|
        # (scipy.special.expit не передаётся в воркеры joblib из __main__)
        p = 0.5 * (1.0 + np.tanh(0.5 * np.einsum("bnd,bd->bn", Xi, wi)))
        g = C * np.einsum("bnd,bn->bd", Xi, p - y[idx]) + reg * wi
        H = C * np.einsum("bni,bn,bnj->bij", Xi, p * (1.0 - p), Xi) + jitter
        step = np.linalg.solve(H, g[..., None])[..., 0]
        # квадрат ньютоновского декремента ~ 2 * (f(w) - f(оптимум))
        lam2 = np.einsum("bd,bd->b", g, step)

        done = lam2 <= 2.0 * tol
        converged[idx[done]] = True
        keep = ~done
        if not keep.any():
            break
        idx, wi, step, lam2 = idx[keep], wi[keep], step[keep], lam2[keep]

        # backtracking (Армихо): шаг уменьшается вдвое, пока функция не
        # уменьшится; допуск 1e-12*|f| - на ошибки округления у оптимума
        t = np.ones(idx.size)
        for _ in range(50):
            w_try = wi - t[:, None] * step
            f_try = objective(w_try, idx)
            ok = f_try <= f_cur[idx] - 1e-4 * t * lam2 + 1e-12 * np.abs(f_cur[idx])
            if ok.all():
                break
            t = np.where(ok, t, 0.5 * t)
        w[idx] = w_try
        f_cur[idx] = f_try

    return w, converged


def fit_feature_sets_accuracy_batched(
    df_features: pd.DataFrame,
    feature_sets: Sequence[Tuple[str, ...]],
    target_col: str = "target",
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    min_samples: int = MIN_SAMPLES,
    split_random_states: Sequence[int] = SPLIT_RANDOM_STATES,
    max_batch_elems: int = 2_000_000,
) -> List[Optional[Tuple[float, float, Tuple[float, ...], int, int]]]:
    """То же, что fit_combo_accuracy_only() для каждого набора из
    feature_sets (тот же результат на позиции набора, None - если набор
    непригоден), но в разы быстрее:

      1. Наборы группируются по маске строк, оставшихся после dropna (внутри
         одной комбинации тестов она обычно одна на все наборы). Балансировка
         1:1 и train/test-разбиения зависят только от этих строк и target,
         поэтому считаются ОДИН раз на группу - теми же _balance_1to1() и
         train_test_split() с теми же random_state, т.е. строки и сплиты
         в точности те же, что в fit_combo_accuracy_only().
      2. Логрегрессии всех наборов группы на одном сплите обучаются разом
         (_logreg_l2_newton). Задачи, где Ньютон не сошёлся, пересчитываются
         обычной sklearn LogisticRegression.
    """
    results: List[Optional[Tuple[float, float, Tuple[float, ...], int, int]]] = [None] * len(feature_sets)
    if df_features.empty or target_col not in df_features.columns:
        return results

    cols = sorted({c for fc in feature_sets for c in fc if c in df_features.columns})
    col_pos = {c: i for i, c in enumerate(cols)}
    notna = df_features[cols].notna().to_numpy()
    target_ok = df_features[target_col].notna().to_numpy()

    groups: Dict[bytes, Tuple[np.ndarray, List[int]]] = {}
    for i, fc in enumerate(feature_sets):
        if any(c not in col_pos for c in fc):
            continue
        mask = target_ok & notna[:, [col_pos[c] for c in fc]].all(axis=1)
        groups.setdefault(mask.tobytes(), (mask, []))[1].append(i)

    for mask, set_idx in groups.values():
        sub = df_features.loc[mask, cols + [target_col]]
        if sub[target_col].nunique() < 2:
            continue
        balanced = _balance_1to1(sub, target_col, random_state)
        if len(balanced) < min_samples:
            continue

        y = balanced[target_col].to_numpy()
        splits = []
        for split_rs in split_random_states:
            try:
                tr, te = train_test_split(
                    np.arange(len(balanced)), test_size=test_size, stratify=y, random_state=split_rs
                )
            except ValueError:
                continue
            if len(np.unique(y[tr])) < 2 or len(te) == 0:
                continue
            splits.append((tr, te))
        if not splits:
            continue

        # sklearn.predict() отдаёт classes_[1] при z > 0, classes_ отсортированы
        y01 = (y == np.unique(y)[-1]).astype(float)
        A = balanced[cols].to_numpy(dtype=float)
        F = np.array([[col_pos[c] for c in feature_sets[i]] for i in set_idx])  # (S, f)
        S, nf = F.shape
        acc = np.empty((S, len(splits)))

        for k, (tr, te) in enumerate(splits):
            chunk = max(1, max_batch_elems // (len(tr) * (nf + 1)))
            for s0 in range(0, S, chunk):
                Fc = F[s0:s0 + chunk]
                X_tr = A[tr][:, Fc].transpose(1, 0, 2)  # (s, n_tr, f)
                X_te = A[te][:, Fc].transpose(1, 0, 2)
                w, conv = _logreg_l2_newton(X_tr, np.broadcast_to(y01[tr], X_tr.shape[:2]))
                z_te = np.einsum("bnf,bf->bn", X_te, w[:, :-1]) + w[:, -1:]
                acc_c = ((z_te > 0) == (y01[te] > 0.5)).mean(axis=1)
                for b in np.flatnonzero(~conv):
                    model = LogisticRegression(max_iter=1000, random_state=random_state)
                    model.fit(X_tr[b], y[tr])
                    acc_c[b] = np.mean(model.predict(X_te[b]) == y[te])
                acc[s0:s0 + chunk, k] = acc_c

        n_train = int(round(np.mean([len(tr) for tr, _ in splits])))
        n_test = int(round(np.mean([len(te) for _, te in splits])))
        for row, i in enumerate(set_idx):
            accs = acc[row]
            acc_std = float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0
            results[i] = (float(np.mean(accs)), acc_std, tuple(float(a) for a in accs), n_train, n_test)

    return results


def fit_combo_model(
    df_features: pd.DataFrame,
    test_combo: Tuple[str, ...],
    feature_cols: Tuple[str, ...],
    month: int,
    target_col: str = "target",
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    min_samples: int = MIN_SAMPLES,
    split_random_states: Sequence[int] = SPLIT_RANDOM_STATES,
    stat: Optional[Tuple[float, float, Tuple[float, ...], int, int]] = None,
) -> Optional[Tuple[ModelRecord, pd.DataFrame]]:
    """Полная версия оценки для отобранной модели: делает то же самое,
    что и fit_combo_accuracy_only(), а дополнительно дообучает финальную
    модель на ВСЕХ сбалансированных данных (без hold-out, для инференса)
    и возвращает сам сбалансированный датасет - вызывается ТОЛЬКО для
    наборов колонок, вошедших в top (см. build_top_models()).

    stat - уже посчитанные при переборе (accuracy, accuracy_std,
    accuracies, n_train, n_test): тогда hold-out сплиты повторно не считаются.

    Возвращает (ModelRecord, сбалансированный_датасет) или None.
    """
    if stat is None:
        stat = fit_combo_accuracy_only(
            df_features, feature_cols, target_col, test_size, random_state, min_samples, split_random_states
        )
    if stat is None:
        return None
    accuracy, accuracy_std, accuracies, n_train, n_test = stat

    needed_cols = list(feature_cols) + [target_col]
    sub = df_features[needed_cols].dropna(subset=needed_cols)
    balanced = _balance_1to1(sub, target_col, random_state)

    final_model = LogisticRegression(max_iter=1000, random_state=random_state)
    final_model.fit(balanced[list(feature_cols)], balanced[target_col])

    record = ModelRecord(
        month=month,
        n_features=len(test_combo),
        combo=tuple(test_combo),
        feature_cols=tuple(feature_cols),
        accuracy=accuracy,
        n_train=n_train,
        n_test=n_test,
        model=final_model,
        random_state=random_state,
        accuracy_std=accuracy_std,
        accuracies=accuracies,
    )
    return record, balanced[needed_cols]


# --------------------------------------------------------------------------
# 4. Перебор: accuracy-only оценка комбинации + потоковый отбор top
# --------------------------------------------------------------------------

# Колонки сырого df, которые нужны препроцессингу. Воркерам отправляются
# только они и только строки тестов своей комбинации - меньше pickle-трафика.
WORKER_COLS = [
    "patient_id", "test_type_code", "test_date",
    "test_result", "test_result_percentile", "target", "days_to_death",
]

# Компактная запись одной модели (~130 байт). feature_cols не хранятся: они
# однозначно восстанавливаются из (combo_index, set_index) - set_index это
# позиция набора статистик в itertools.product(*кандидаты_по_тестам), см.
# _feature_cols_from_index(). accuracies - accuracy по каждому разбиению
# SPLIT_RANDOM_STATES (хвост - NaN, если часть разбиений не удалась;
# фактическая длина - n_splits).
N_SPLITS_MAX = len(SPLIT_RANDOM_STATES)
LITE_DTYPE = np.dtype([
    ("key", "f8"),            # accuracy, округлённая до ACC_KEY_DECIMALS (ключ ранжирования)
    ("combo_index", "i8"),
    ("set_index", "i8"),
    ("accuracy", "f8"),       # точная средняя accuracy
    ("accuracy_std", "f8"),
    ("n_train", "i4"),
    ("n_test", "i4"),
    ("n_splits", "u1"),
    ("accuracies", "f8", (N_SPLITS_MAX,)),
])

# accuracy = среднее долей k/n_test по разбиениям, т.е. рациональное число;
# одно и то же значение, посчитанное с разным порядком суммирования, может
# отличаться в последнем бите. Округление делает такие модели честной
# ничьей (дальше ничьи разбиваются детерминированно по индексам).
ACC_KEY_DECIMALS = 10

# Гистограмма распределения accuracy: если различных значений станет
# больше HIST_MAX_DISTINCT, значения огрубляются до HIST_COARSE_DECIMALS
# знаков (ограничивает память; в сводке это видно по hist_decimals).
HIST_MAX_DISTINCT = 5_000_000
HIST_COARSE_DECIMALS = 6


def _feature_cols_from_index(
    per_test_candidates: Sequence[Sequence[str]],
    set_index: int,
) -> Tuple[str, ...]:
    """Обратное к позиции в itertools.product(*per_test_candidates):
    последний тест меняется быстрее всех (C-порядок)."""
    digits = np.unravel_index(int(set_index), tuple(len(c) for c in per_test_candidates))
    return tuple(cands[int(d)] for cands, d in zip(per_test_candidates, digits))


def _combo_task_lite(
    df_parts: List[pd.DataFrame],
    month: int,
    test_combo: Tuple[str, ...],
    combo_index: int,
    seed: np.random.SeedSequence,
    test_size: float,
    random_state: int,
    min_samples: int,
    window_days: int,
    min_key: float = -np.inf,
) -> Tuple[int, int, int, int, Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """Одна задача перебора для joblib: строит таблицу признаков комбинации
    и оценивает accuracy всех наборов статистик.

      - n_valid    - число валидных моделей (stat is not None);
      - (vals, counts) - гистограмма accuracy-ключей ВСЕХ валидных моделей
                     комбинации (для распределения accuracy);
      - recs       - LITE_DTYPE-массив только тех моделей, чей ключ
                     >= min_key (текущий порог top-буфера на момент
                     отправки задачи; модели ниже порога уже не могут
                     попасть в итоговый top, поэтому не пересылаются).
    """
    df_loc = pd.concat(df_parts, ignore_index=True)
    empty_hist = (np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64))
    empty_recs = np.empty(0, dtype=LITE_DTYPE)

    rng = np.random.default_rng(seed)

    df_features = build_feature_table(
        df_loc,
        test_combo,
        month,
        window_days=window_days,
        rng=rng,
    )

    if df_features.empty:
        return month, len(test_combo), 0, 0, empty_hist, empty_recs

    per_test_candidates = [
        test_feature_candidates(t, window_days * month)
        for t in test_combo
    ]

    feature_sets = list(itertools.product(*per_test_candidates))
    n_tried = len(feature_sets)

    stats = fit_feature_sets_accuracy_batched(
        df_features,
        feature_sets,
        test_size=test_size,
        random_state=random_state,
        min_samples=min_samples,
    )

    valid = [i for i, s in enumerate(stats) if s is not None]
    n_valid = len(valid)
    if n_valid == 0:
        return month, len(test_combo), n_tried, 0, empty_hist, empty_recs

    sel = [stats[i] for i in valid]
    recs = np.zeros(n_valid, dtype=LITE_DTYPE)
    recs["combo_index"] = combo_index
    recs["set_index"] = valid
    recs["accuracy"] = [s[0] for s in sel]
    recs["accuracy_std"] = [s[1] for s in sel]
    recs["n_train"] = [s[3] for s in sel]
    recs["n_test"] = [s[4] for s in sel]
    recs["n_splits"] = [len(s[2]) for s in sel]
    accs = np.full((n_valid, N_SPLITS_MAX), np.nan)
    for r, s in enumerate(sel):
        accs[r, :len(s[2])] = s[2]
    recs["accuracies"] = accs
    recs["key"] = np.round(recs["accuracy"], ACC_KEY_DECIMALS)

    hist = np.unique(recs["key"], return_counts=True)
    recs = recs[recs["key"] >= min_key]

    return month, len(test_combo), n_tried, n_valid, hist, recs


def update_top(best: np.ndarray, new: np.ndarray, capacity: int) -> np.ndarray:
    """Оставляет capacity лучших записей среди best + new за O(n):
    записи с key выше порога берутся все, ничьи на пороге добираются
    детерминированно по (combo_index, set_index). Порядок внутри
    результата не определён."""
    allr = np.concatenate([best, new])
    if allr.size <= capacity:
        return allr
    pos = allr.size - capacity
    kth = np.partition(allr["key"], pos)[pos]
    above = allr[allr["key"] > kth]
    ties = allr[allr["key"] == kth]
    ties = ties[np.lexsort((ties["set_index"], ties["combo_index"]))][: capacity - above.size]
    return np.concatenate([above, ties])


class AccuracyHistogram:
    """Точное распределение accuracy-ключей всех валидных моделей:
    отсортированные различные значения + число моделей с каждым значением.
    Если различных значений больше max_distinct, значения огрубляются до
    coarse_decimals знаков (память остаётся ограниченной)."""

    def __init__(
        self,
        decimals: int = ACC_KEY_DECIMALS,
        max_distinct: int = HIST_MAX_DISTINCT,
        coarse_decimals: int = HIST_COARSE_DECIMALS,
    ):
        self.decimals = decimals
        self.max_distinct = max_distinct
        self.coarse_decimals = coarse_decimals
        self.values = np.empty(0, dtype=np.float64)
        self.counts = np.empty(0, dtype=np.int64)
        self.total = 0
        self._pv: List[np.ndarray] = []
        self._pc: List[np.ndarray] = []
        self._n_pending = 0

    def add(self, values: np.ndarray, counts: np.ndarray) -> None:
        if len(values) == 0:
            return
        self._pv.append(values)
        self._pc.append(counts.astype(np.int64, copy=False))
        self._n_pending += len(values)
        self.total += int(counts.sum())
        if self._n_pending > max(1_000_000, self.values.size):
            self._compact()

    def _reduce(self, v: np.ndarray, c: np.ndarray) -> None:
        v = np.round(v, self.decimals)
        uniq, inv = np.unique(v, return_inverse=True)
        self.values = uniq
        self.counts = np.bincount(inv, weights=c, minlength=uniq.size).round().astype(np.int64)

    def _compact(self) -> None:
        if not self._pv:
            return
        v = np.concatenate([self.values] + self._pv)
        c = np.concatenate([self.counts] + self._pc)
        self._pv, self._pc, self._n_pending = [], [], 0
        self._reduce(v, c)
        if self.values.size > self.max_distinct and self.decimals > self.coarse_decimals:
            self.decimals = self.coarse_decimals
            self._reduce(self.values, self.counts)

    def finalize(self) -> Tuple[np.ndarray, np.ndarray]:
        self._compact()
        return self.values, self.counts

    def rank_of(self, x: float) -> Dict[str, float]:
        """Положение значения x в распределении: число моделей строго ниже,
        равных (ничьи) и перцентильные ранги в процентах."""
        values, counts = self.finalize()
        xk = round(float(x), self.decimals)
        below = int(counts[values < xk].sum())
        equal = int(counts[values == xk].sum())
        total = max(1, self.total)
        return {
            "n_below": below,
            "n_tied": equal,
            "pct_rank_strict": 100.0 * below / total,              # доля строго ниже порога
            "pct_rank_mean": 100.0 * (below + 0.5 * equal) / total,  # как scipy percentileofscore(kind="mean")
            "pct_rank_weak": 100.0 * (below + equal) / total,       # доля не выше порога
        }


def _top_tag(
    top_mode: str,
    top_fraction: float,
    top_count: int,
) -> str:
    if top_mode == "count":
        return f"top{top_count}"

    if top_mode == "fraction":
        return f"top{top_fraction * 100:g}pct"

    raise ValueError(
        f"Неизвестный TOP_MODE={top_mode!r}. "
        f"Используйте 'fraction' или 'count'."
    )

# --------------------------------------------------------------------------
# 5. Хранение победителей: модель + accuracy + датасет - в одной записи
# --------------------------------------------------------------------------

def _dump_model_with_dataset(out_f, record: ModelRecord, balanced_df: pd.DataFrame) -> None:
    """Дописывает ОДНУ запись (модель + её accuracy + её сбалансированный
    датасет) в уже открытый бинарный файл через pickle.dump(). Сам .pkl
    файл группы (месяц, n_features) представляет собой не один большой
    список, а ПОСЛЕДОВАТЕЛЬНОСТЬ независимо запиклленных словарей (по
    одному на модель): это позволяет писать и читать победителей по
    одному, не держа в памяти сразу всех сразу.

    Сбалансированный (1:1) датасет для этой конкретной модели хранится
    ВНУТРИ той же записи как TSV-текст (df.to_csv(sep="\t")) под ключом
    "balanced_dataset_tsv" (см. load_one_model_dataset() для обратного
    преобразования в pd.DataFrame)."""
    pickle.dump(_plain_model_record(record, balanced_df), out_f)


def _plain_model_record(record: ModelRecord, balanced_df: pd.DataFrame) -> dict:
    """Словарь одной записи .pkl файла (см. _dump_model_with_dataset)."""
    return {
        "month": record.month,
        "n_features": record.n_features,
        "name": record.name,
        "combo": tuple(record.combo),
        "feature_cols": tuple(record.feature_cols),
        "accuracy": float(record.accuracy),
        "accuracy_std": float(record.accuracy_std),
        "accuracies": tuple(float(a) for a in record.accuracies),
        "n_train": int(record.n_train),
        "n_test": int(record.n_test),
        "model": record.model,
        "random_state": int(record.random_state),
        "balanced_dataset_tsv": balanced_df.to_csv(sep="\t", index=False),
    }


def _combo_task_refit(
    df_parts: List[pd.DataFrame],
    month: int,
    test_combo: Tuple[str, ...],
    winners: List[Tuple[Tuple[str, ...], Tuple[float, float, Tuple[float, ...], int, int]]],
    seed: np.random.SeedSequence,
    test_size: float,
    random_state: int,
    min_samples: int,
    window_days: int,
) -> Tuple[List[bytes], List[float]]:
    """Одна задача дообучения для joblib: перестраивает таблицу признаков
    комбинации (тем же seed, что и при переборе) и дообучает всех её
    победителей - winners = [(feature_cols, stat)], сплиты заново не
    считаются. Возвращает уже запиклленные записи (главный процесс только
    дописывает их байты в .pkl) и их accuracy."""
    df_loc = pd.concat(df_parts, ignore_index=True)

    rng = np.random.default_rng(seed)

    df_features = build_feature_table(
        df_loc,
        test_combo,
        month,
        window_days=window_days,
        rng=rng,
    )

    blobs: List[bytes] = []
    accuracies: List[float] = []

    for feature_cols, stat in winners:

        result = fit_combo_model(
            df_features,
            test_combo,
            feature_cols,
            month,
            test_size=test_size,
            random_state=random_state,
            min_samples=min_samples,
            stat=stat,
        )

        if result is None:
            continue

        record, balanced_df = result

        blobs.append(
            pickle.dumps(_plain_model_record(record, balanced_df))
        )
        accuracies.append(record.accuracy)

    return blobs, accuracies


def load_top_models_plain(path) -> List[dict]:
    """Загружает ВСЕ записи группы (месяц, n_features), сохранённые
    build_top_models(). Файл - последовательность независимо
    запиклленных словарей (см. _dump_model_with_dataset), поэтому
    читаем pickle.load() в цикле до EOFError, а не одним большим
    pickle.load(). Чтобы получить предсказание из записи, нужно подать
    модели датафрейм с колонками РОВНО В ПОРЯДКЕ record["feature_cols"]:
        model.predict_proba(new_df[list(record["feature_cols"])])
    """
    records: List[dict] = []
    with open(path, "rb") as f:
        while True:
            try:
                records.append(pickle.load(f))
            except EOFError:
                break
    return records


def iter_top_models_plain(path):
    """Ленивый (генераторный) аналог load_top_models_plain() - отдаёт
    записи ПО ОДНОЙ, не собирая их в список целиком; полезно, если
    победителей всё ещё много и держать их все в памяти одновременно
    не хочется (например, при последующем инференсе по всем моделям)."""
    with open(path, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


def load_one_model_dataset(record: dict) -> pd.DataFrame:
    """Восстанавливает сбалансированный датасет одной записи (см.
    _dump_model_with_dataset) обратно в pd.DataFrame."""
    return pd.read_csv(io.StringIO(record["balanced_dataset_tsv"]), sep="\t")


# --------------------------------------------------------------------------
# 6. Отбор top-N моделей на пару (месяц, n_features)
# --------------------------------------------------------------------------

def _winner_tasks(
    winners: np.ndarray,
    combos: Sequence[Tuple[str, ...]],
    suffix_days: int,
):
    """Группирует победителей (LITE_DTYPE) по combo_index и ЛЕНИВО отдаёт
    (combo_index, [(feature_cols, stat), ...]) - Python-объекты создаются
    только для комбинации, которая сейчас отправляется на дообучение.
    Внутри комбинации - от лучшей модели к худшей. stat - ровно тот же
    кортеж (accuracy, accuracy_std, accuracies, n_train, n_test), что
    вернула fit_feature_sets_accuracy_batched()."""
    order = np.lexsort((winners["set_index"], -winners["key"], winners["combo_index"]))
    w = winners[order]
    combo_ids, starts = np.unique(w["combo_index"], return_index=True)
    ends = list(starts[1:]) + [len(w)]
    for combo_index, s, e in zip(combo_ids, starts, ends):
        combo = combos[int(combo_index)]
        per_test_candidates = [test_feature_candidates(t, suffix_days) for t in combo]
        combo_winners = []
        for r in w[s:e]:
            n = int(r["n_splits"])
            stat = (
                float(r["accuracy"]),
                float(r["accuracy_std"]),
                tuple(float(a) for a in r["accuracies"][:n]),
                int(r["n_train"]),
                int(r["n_test"]),
            )
            combo_winners.append((_feature_cols_from_index(per_test_candidates, r["set_index"]), stat))
        yield int(combo_index), combo_winners


def build_top_models(
    df_timed: pd.DataFrame,
    months_list: Sequence[int],
    n_features_list: Sequence[int],
    candidate_tests: Sequence[str],
    out_dir: Path,
    top_mode: str = TOP_MODE,
    top_fraction: float = TOP_FRACTION,
    top_count: int = TOP_COUNT,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    min_samples: int = MIN_SAMPLES,
    window_days: int = WINDOW_DAYS,
    chunk_models: int = CHUNK_MODELS,
    n_jobs: int = -1,
    verbose: bool = True,
) -> Tuple[Dict[Tuple[int, int], Path], Dict[Tuple[int, int], List[float]]]:

    if top_mode not in ("count", "fraction"):
        raise ValueError(f"Неизвестный top_mode={top_mode!r}. Используйте 'count' или 'fraction'.")
    if top_mode == "count" and top_count <= 0:
        raise ValueError("top_count должен быть > 0.")
    if top_mode == "fraction" and not (0 < top_fraction <= 1):
        raise ValueError("top_fraction должен быть в диапазоне (0, 1].")

    candidate_set = set(candidate_tests)
    by_test = {
        t: g[WORKER_COLS]
        for t, g in df_timed.groupby("test_type_code")
        if t in candidate_set
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    tag = _top_tag(top_mode, top_fraction, top_count)

    output_paths: Dict[Tuple[int, int], Path] = {}
    accuracies_by_group: Dict[Tuple[int, int], List[float]] = {}
    summary_rows: List[dict] = []

    n_stat_candidates = len(test_feature_candidates("x", 0))

    for month in months_list:
        for nf in n_features_list:

            combos = list(itertools.combinations(candidate_tests, nf))
            n_sets = n_stat_candidates ** nf
            n_models_upper = len(combos) * n_sets
            chunk_combos = max(1, chunk_models // n_sets)

            accuracies_by_group[(month, nf)] = []

            if verbose:
                print(
                    f"[месяц {month}] n_features={nf}: "
                    f"{len(candidate_tests)} тестов-кандидатов, "
                    f"{len(combos)} комбинаций x {n_sets} наборов статистик = "
                    f"{n_models_upper} моделей (верхняя граница)"
                )

            best = np.empty(0, dtype=LITE_DTYPE)
            threshold = -np.inf
            capacity = top_count if top_mode == "count" else 1
            min_capacity = np.inf
            peak_records = 0
            hist = AccuracyHistogram()
            n_valid_total = 0
            n_combos_done = 0

            with Parallel(n_jobs=n_jobs, return_as="generator_unordered", max_nbytes=None) as parallel, \
                    tqdm(total=len(combos), desc=f"Month {month}, n_features={nf}", unit="combo") as pbar:

                for start in range(0, len(combos), chunk_combos):
                    ids = range(start, min(start + chunk_combos, len(combos)))

                    jobs = (
                        delayed(_combo_task_lite)(
                            [by_test[t] for t in combos[i]],
                            month,
                            combos[i],
                            i,
                            np.random.SeedSequence([random_state, month, nf, i]),
                            test_size,
                            random_state,
                            min_samples,
                            window_days,
                            threshold,
                        )
                        for i in ids
                    )

                    chunk_recs = []
                    for _m, _nf, _tried, n_valid, (h_vals, h_counts), recs in parallel(jobs):
                        n_valid_total += n_valid
                        hist.add(h_vals, h_counts)
                        chunk_recs.append(recs)
                        pbar.update(1)
                    n_combos_done += len(ids)

                    if top_mode == "fraction":
                        remaining = (len(combos) - n_combos_done) * n_sets
                        est_total = n_valid_total / (n_combos_done * n_sets) * n_models_upper
                        capacity = max(1, min(
                            math.ceil(BUFFER_FACTOR * top_fraction * est_total),
                            math.ceil(top_fraction * (n_valid_total + remaining)),
                        ))
                    min_capacity = min(min_capacity, capacity)

                    new = np.concatenate(chunk_recs)
                    peak_records = max(peak_records, best.size + new.size)
                    best = update_top(best, new, capacity)
                    if best.size >= capacity:
                        threshold = float(best["key"].min())

            if n_valid_total == 0:
                if verbose:
                    print(
                        f"  [месяц {month}] ни одной валидной комбинации для "
                        f"n_features={nf} не найдено (проверьте min_samples / плотность измерений)"
                    )
                continue

            if top_mode == "fraction":
                n_select = max(1, int(round(n_valid_total * top_fraction)))
            else:
                n_select = min(top_count, n_valid_total)
            n_select = min(n_select, best.size)

            if n_select > min_capacity:
                print(
                    f"  ВНИМАНИЕ [месяц {month}, n_features={nf}]: ёмкость буфера опускалась до "
                    f"{min_capacity:.0f} < {n_select} - top может быть неточным, "
                    f"увеличьте BUFFER_FACTOR или CHUNK_MODELS"
                )

            order = np.lexsort((best["set_index"], best["combo_index"], -best["key"]))
            winners = best[order[:n_select]]

            threshold_acc = float(winners["key"][-1])
            rank = hist.rank_of(threshold_acc)
            h_values, h_counts = hist.finalize()
            n_tied_selected = int((winners["key"] == round(threshold_acc, ACC_KEY_DECIMALS)).sum())

            dist_path = out_dir / f"month_{month}__n_features_{nf}__accuracy_distribution.tsv"
            pd.DataFrame({"accuracy": h_values, "n_models": h_counts}).to_csv(
                dist_path, sep="\t", index=False
            )

            summary_rows.append({
                "month": month,
                "n_features": nf,
                "top_mode": top_mode,
                "top_fraction": top_fraction if top_mode == "fraction" else np.nan,
                "top_count": top_count if top_mode == "count" else np.nan,
                "n_models_upper_bound": n_models_upper,
                "n_valid_models": n_valid_total,
                "n_selected": n_select,
                "selected_share_pct": 100.0 * n_select / n_valid_total,
                "accuracy_threshold": threshold_acc,
                "accuracy_best": float(winners["key"][0]),
                "accuracy_median_all": float(h_values[np.searchsorted(np.cumsum(h_counts), (hist.total + 1) // 2)]),
                "n_models_below_threshold": rank["n_below"],
                "n_models_tied_at_threshold": rank["n_tied"],
                "n_tied_selected": n_tied_selected,
                "pct_rank_strict": rank["pct_rank_strict"],
                "pct_rank_mean": rank["pct_rank_mean"],
                "pct_rank_weak": rank["pct_rank_weak"],
                "hist_distinct_values": int(h_values.size),
                "hist_decimals": hist.decimals,
                "buffer_peak_records": peak_records,
                "buffer_peak_mb": peak_records * LITE_DTYPE.itemsize / 2**20,
            })

            if verbose:
                print(
                    f"  [месяц {month}] n_features={nf}: валидных моделей: {n_valid_total}, "
                    f"отобрано {n_select} ({100.0 * n_select / n_valid_total:.2f}%); "
                    f"порог accuracy = {threshold_acc:.6f} (лучшая {float(winners['key'][0]):.6f}); "
                    f"перцентильный ранг порога: {rank['pct_rank_mean']:.2f}; "
                    f"пик буфера: ~{peak_records * LITE_DTYPE.itemsize / 2**20:.1f} МБ"
                )

            # Дообучение только отобранных моделей и запись .pkl
            n_winner_combos = int(np.unique(winners["combo_index"]).size)
            out_path = out_dir / f"month_{month}__n_features_{nf}__{tag}.pkl"
            output_paths[(month, nf)] = out_path

            refit_jobs = (
                delayed(_combo_task_refit)(
                    [by_test[t] for t in combos[combo_index]],
                    month,
                    combos[combo_index],
                    combo_winners,
                    np.random.SeedSequence([random_state, month, nf, combo_index]),
                    test_size,
                    random_state,
                    min_samples,
                    window_days,
                )
                for combo_index, combo_winners in _winner_tasks(winners, combos, window_days * month)
            )

            n_saved = 0
            with open(out_path, "wb") as out_f:
                for blobs, accs in tqdm(
                    Parallel(n_jobs=n_jobs, return_as="generator", max_nbytes=None)(refit_jobs),
                    total=n_winner_combos,
                    desc=f"Month {month}, n_features={nf} (save)",
                    unit="combo",
                ):
                    for blob in blobs:
                        out_f.write(blob)
                    accuracies_by_group[(month, nf)].extend(accs)
                    n_saved += len(blobs)

            del winners, best

            if verbose:
                print(f"  [месяц {month}] n_features={nf}: сохранено моделей: {n_saved} -> {out_path}")

    if summary_rows:
        summary_path = out_dir / f"selection_summary__{tag}.tsv"
        pd.DataFrame(summary_rows).to_csv(summary_path, sep="\t", index=False)
        if verbose:
            print(f"Сводка отбора: {summary_path}")

    return output_paths, accuracies_by_group


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Поиск логрегрессий по комбинациям лабораторных тестов на СЫРОМ "
                     "(длинном) датасете; глобальные перцентили считаются один раз на "
                     "исходной длинной таблице, отбор моделей - потоковый top (доля или "
                     "число), датасеты хранятся внутри .pkl вместе с моделью и accuracy."
    )
    parser.add_argument("--data-path", required=True, help="Путь к сырому .tsv/.csv файлу с измерениями")
    parser.add_argument("--target-col", default="target")
    parser.add_argument("--months", type=str, default="1,2,3,4,5,6",
                         help="Через запятую номера месяцев (окон), например '1,2,3,4,5,6'")
    parser.add_argument("--n-features", type=str, default="1,2",
                         help="Через запятую: сколько лабораторных тестов комбинировать, например '1,2,3'")
    parser.add_argument("--tests-file", default=None,
                         help="Необязательный файл со списком test_type_code-кандидатов (по одному на строку)")
    parser.add_argument("--exclude-test-substr", default=EXCLUDE_TEST_SUBSTR,
                         help="Тесты с этой подстрокой в test_type_code исключаются из кандидатов")
    parser.add_argument(
        "--top-mode",
        choices=("count", "fraction"),
        default=TOP_MODE,
        help="Режим отбора: count = фиксированное число лучших моделей; "
            "fraction = доля лучших моделей."
    )
    parser.add_argument(
        "--top-count",
        type=int,
        default=TOP_COUNT,
        help="Число лучших моделей при --top-mode count."
    )
    parser.add_argument(
        "--top-fraction",
        type=float,
        default=TOP_FRACTION,
        help="Доля лучших моделей при --top-mode fraction."
    )
    parser.add_argument("--chunk-models", type=int, default=CHUNK_MODELS,
                         help="Сколько моделей считать между обновлениями top-буфера")
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--min-samples", type=int, default=MIN_SAMPLES,
                         help="Мин. число строк ПОСЛЕ балансировки 1:1, чтобы комбинация считалась валидной")
    parser.add_argument("--window-days", type=int, default=WINDOW_DAYS,
                         help="Длина скользящего окна в днях (по умолчанию 30 = 1 'месяц')")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--n-jobs", type=int, default=-1,
                         help="Число параллельных процессов joblib (-1 = все ядра, 1 = без параллелизма)")
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()

    # dict.fromkeys - убрать повторы ("1,1"), сохранив порядок: иначе одни и
    # те же модели попали бы в отбор дважды
    months_list = list(dict.fromkeys(int(x) for x in args.months.split(",") if x.strip()))
    n_features_list = list(dict.fromkeys(int(x) for x in args.n_features.split(",") if x.strip()))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Загрузка сырого датасета: {args.data_path}")
    df_raw = load_raw_dataset(args.data_path)
    df_timed = timing(df_raw)

    print("Расчёт глобальных перцентилей (по всему распределению каждого теста, "
          "включая измерения, которые могли бы считаться 'будущими' для пациента)...")
    df_timed = add_global_percentiles(df_timed)

    allowed_tests = load_allowed_tests_file(args.tests_file)
    candidate_tests = list_candidate_tests(
        df_timed, exclude_substr=args.exclude_test_substr, allowed_tests=allowed_tests
    )
    print(f"Тестов-кандидатов для комбинирования: {len(candidate_tests)}")
    if allowed_tests:
        print(f"  (список сужен через --tests-file: {args.tests_file})")

    output_paths, accuracies_by_group = build_top_models(
        df_timed,
        months_list,
        n_features_list,
        candidate_tests,
        out_dir=out_dir,
        top_mode=args.top_mode,
        top_fraction=args.top_fraction,
        top_count=args.top_count,
        test_size=args.test_size,
        random_state=args.random_state,
        min_samples=args.min_samples,
        window_days=args.window_days,
        chunk_models=args.chunk_models,
        n_jobs=args.n_jobs,
    )

    for (month, nf), path in sorted(output_paths.items()):
        n_models = len(accuracies_by_group.get((month, nf), []))
        print(f"  Готово: {path} ({n_models} моделей; accuracy и сбалансированный "
              f"датасет каждой модели хранятся вместе, внутри этого .pkl)")

    print("\nГотово.")
    sys.exit(0)


if __name__ == "__main__":
    main()