"""
Поиск логистических регрессий по комбинациям ЛАБОРАТОРНЫХ ТЕСТОВ на
ОДНОМ сыром ("длинном") датасете.

На вход подаётся ОДИН сырой файл вида:

    index  test_id  patient_id  test_type_code  test_date  test_result
    value_unit_code  birth_date  gender  death_date  death_reason
    max_date  diag_code  morph_group_name  diag_metastases_localization
    days_to_death  target

(1 строка = 1 измерение одного лабораторного теста у одного пациента).

ЧТО СЧИТАЕТСЯ "КОМБИНАЦИЕЙ ПРИЗНАКОВ" (n_features):
n_features - это количество РАЗНЫХ ЛАБОРАТОРНЫХ ТЕСТОВ (test_type_code),
участвующих в модели. Каждый признак модели ОБЯЗАТЕЛЬНО берётся из
СВОЕГО, ОТДЕЛЬНОГО теста.

ГЛОБАЛЬНЫЕ ПЕРЦЕНТИЛИ (важное отличие от предыдущей версии):
Раньше перцентильный ранг считался ПОСЛЕ агрегации в widetable, отдельно
для каждой пары (test_combo, месяц), и только относительно истории
измерений ДО index_date пациента. Теперь перцентиль считается ОДИН РАЗ
для всего сырого ("длинного") датасета - см. add_global_percentiles():
каждому измерению (1 строка = 1 измерение) сопоставляется его
перцентильный ранг (0..1) относительно ВСЕГО распределения измерений
ЭТОГО ЖЕ теста по всей популяции и по ВСЕМУ времени наблюдения (в том
числе относительно измерений, случившихся позже, "в будущем" - никакого
ограничения по index_date больше нет). Эта колонка (test_result_percentile)
затем агрегируется в окне 30 дней ТАК ЖЕ, как и абсолютное значение теста
(test_result): "last" и mean/q25/median/q75 за окно - см.
feature_engineering(). Итого на каждый тест комбинации приходится
10 кандидатов-столбцов (5 по абсолютному значению + 5 по перцентилю),
а не 8, как раньше.

Для каждой комбинации тестов (test_combo) размера n_features и каждого
"месяца" (n_month = 1..6, окно наблюдения) препроцессинг строится ЗАНОВО
из сырых данных:
  1. timing()               - парсинг дат, пересчёт target по death_date,
                               чистка нулевых test_result, кодирование пола.
  2. add_global_percentiles() - см. выше; делается ОДИН РАЗ для всего
                               датасета, а не заново для каждой комбинации.
  3. feature_engineering()   - для тестов из test_combo строит widetable
                               (1 строка = 1 пациент) с "_last" и
                               window-статистиками по test_result и по
                               test_result_percentile. Пропуски (NaN)
                               сохраняются как есть - нужный dropna()
                               делается позже, в fit_combo_model()/
                               fit_combo_accuracy_only(), СТРОГО по тем
                               колонкам, которые реально входят в
                               конкретный набор feature_cols модели.

Дальше для этой же widetable перебираются ВСЕ способы выбрать РОВНО ПО
ОДНОЙ конкретной статистике из каждого теста комбинации (декартово
произведение кандидатов по тестам, 10^n_features вариантов). Для каждого
такого набора из n_features колонок отдельно:
  - dropna() СТРОГО по этим n_features колонкам (+ target);
  - балансировка классов 1:1 (undersampling);
  - обучение и оценка на нескольких РАЗНЫХ train/test-разбиениях одного и
    того же сбалансированного датасета (SPLIT_RANDOM_STATES); итоговая
    accuracy модели - среднее accuracy по этим разбиениям (плюс std и
    отдельные accuracy по каждому разбиению - для контроля устойчивости);
  - комбинации с недостаточным числом строк ПОСЛЕ балансировки (см.
    MIN_SAMPLES) отбрасываются как статистически ненадёжные.

ОТБОР И ХРАНЕНИЕ РЕЗУЛЬТАТОВ (потоковый top-N%, одна проходка, без диска):
Сохраняется top_fraction лучших моделей (по умолчанию 10%) на пару
(месяц, n_features) - либо фиксированный top_count при --top-mode count.
При n_features=3-4 число моделей (комбинации тестов x 10^n_features
наборов статистик) огромно, поэтому отбор устроен в ДВЕ ФАЗЫ (см.
build_top_models()):

  Phase 1 ("lite"): для каждой (месяц, test_combo) строится widetable
    один раз, и для каждого набора статистик считается ТОЛЬКО accuracy
    (среднее по SPLIT_RANDOM_STATES - эти 10 разбиений НЕ являются
    отдельными моделями). Воркер возвращает не список словарей, а
    КОМПАКТНЫЙ numpy-массив записей (LITE_DTYPE, ~130 байт на модель:
    индекс комбинации, индекс набора статистик, accuracy, std, accuracy
    по разбиениям, n_train, n_test) - имена колонок не хранятся, а
    восстанавливаются по индексу набора (порядок itertools.product).

    Записи проходят через StreamingTopK - ограниченный буфер top-K:
      * ёмкость K - верхняя граница числа победителей: при fraction
        K = ceil(top_fraction * C(T, n_features) * 10^n_features), т.е.
        от КОМБИНАТОРНОГО числа моделей (все модели валидны в худшем
        случае); по ходу перебора K уменьшается до
        ceil(top_fraction * (валидных_уже + ещё_не_посчитанных)) - так
        невалидные (min_samples и т.п.) модели не раздувают буфер;
      * когда буфер переполнен (K + запас), он сжимается векторно
        (np.lexsort) до K лучших ПЛЮС все ничьи с K-й моделью, и порог
        threshold = accuracy K-й модели; записи с accuracy < threshold
        дальше отбрасываются сразу - в т.ч. ещё в воркере (порог
        передаётся в новые задачи), чтобы не гонять их через
        межпроцессный pickle;
      * ничьи: accuracy округляется до ACC_KEY_DECIMALS знаков (чтобы
        одинаковые доли k/n_test, посчитанные с разным порядком
        суммирования, были ничьей), и модели с accuracy, равной порогу,
        берутся ВСЕ - как в старом скрипте; отбор не зависит от порядка
        тестов (ничьи не отрезаются по алфавиту), от --n-jobs и от
        порядка прихода задач.
    Итог: после Phase 1 буфер ГАРАНТИРОВАННО содержит все модели с
    accuracy >= accuracy N-й лучшей, где N = max(1, round(top_fraction *
    число_валидных_моделей)) (или min(top_count, число_валидных)); из-за
    ничьих на пороге отобранных может быть немного больше N. Никакого
    временного файла и повторного чтения, в памяти не больше ~2.5*K
    компактных записей (+ ничьи на пороге).

    Распределение accuracy: параллельно ведётся ТОЧНАЯ гистограмма
    (значение accuracy -> число моделей) по ВСЕМ валидным моделям
    (AccuracyHistogram). accuracy - среднее долей k/n_test, поэтому
    различных значений немного (обычно тысячи-сотни тысяч), и
    гистограмма меньше, чем даже выборка "каждой 10-й модели"; если
    различных значений всё же станет больше HIST_MAX_DISTINCT, они
    огрубляются до HIST_COARSE_DECIMALS знаков. По ней для порога
    top-N считается перцентильный ранг (доля моделей строго ниже порога,
    с учётом половины ничьих, и не выше порога) и число ничьих на пороге.
    Гистограмма и сводка пишутся рядом с .pkl:
      month_<m>__n_features_<nf>__accuracy_distribution.tsv
      selection_summary__<tag>.tsv

  Phase 2 ("refit"): победители группируются по индексу комбинации
    тестов, widetable перестраивается (по тому же seed - воспроизводимо)
    ТОЛЬКО для комбинаций тестов, у которых есть хотя бы один
    "победитель", и дообучение с финальной моделью (fit_combo_model())
    выполняется ТОЛЬКО для наборов признаков, прошедших отбор. Каждая
    финальная запись дописывается ПО ОДНОЙ в .pkl файл группы (см.
    _dump_model_with_dataset) - формат .pkl не изменился.

Пары (месяц, test_combo) внутри Phase 1 независимы, поэтому считаются
параллельно (joblib, --n-jobs). Случайные окна "живых" пациентов берут
seed из (random_state, месяц, n_features, номер комбинации), поэтому
результат не зависит ни от --n-jobs, ни от порядка выполнения задач, и
Phase 2 воспроизводит ТОЧНО ТЕ ЖЕ widetable, что и Phase 1.

ХРАНЕНИЕ ДАТАСЕТОВ (важное отличие от предыдущей версии): раньше
сбалансированный (1:1) датасет каждой top-K модели сохранялся отдельным
.tsv файлом в отдельной папке. Теперь он хранится ВНУТРИ той же записи
.pkl файла, где лежит сама модель и её accuracy (ключ
"balanced_dataset_tsv" - TSV-текст; см. load_one_model_dataset()).

В конце строится график: по оси X - месяцы, по оси Y - среднее accuracy
отобранных (top_fraction) моделей месяца с усами 95% ДИ, отдельная линия
на каждое значение n_features.
"""

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

import matplotlib
# Неинтерактивный backend - ДО импорта pyplot, иначе на машинах без
# дисплея plt.show()/savefig может ждать GUI и не отдать управление баш-сессии.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from scipy import stats as sp_stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

RANDOM_STATE = 42
TOP_MODE = "fraction"    # "fraction" или "count"
TOP_FRACTION = 0.10      # используется при TOP_MODE="fraction"
TOP_COUNT = 10_000

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
    агрегируется (last/mean/q25/median/q75 за окно) наравне с абсолютным
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
    window_days где-то между первым тестом и (последний тест -
    window_days*n_month), т.е. имитирует снэпшот "как будто до конца
    наблюдения осталось n_month месяцев". Случайность берётся из rng -
    передавайте генератор с seed, иначе результат невоспроизводим.

    "Умершим" (target==1) пациентам сопоставляет измерения, попавшие в
    окно [window_days*n_month, window_days*(n_month+1)) дней до смерти.

    Для каждого теста строит:
      - "<тест>_last"                       - последнее АБСОЛЮТНОЕ значение в окне
      - mean/q25/median/q75_<тест>_<Nd>     - статистики АБСОЛЮТНОГО значения за окно
      - "pct_<тест>_last"                   - последний ГЛОБАЛЬНЫЙ ПЕРЦЕНТИЛЬ в окне
      - mean/q25/median/q75_pct_<тест>_<Nd> - статистики ГЛОБАЛЬНОГО ПЕРЦЕНТИЛЯ за окно
    Итого 10 кандидатов-столбцов на каждый тест (см. test_feature_candidates()).

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

    span = (patients["max_end_date"] - patients["min_start_date"]).dt.total_seconds()
    offsets = pd.to_timedelta(rng.random(len(patients)) * span.clip(lower=0).to_numpy(), unit="s")
    patients["random_end_date"] = (patients["min_start_date"] + offsets.to_numpy()).where(span > 0)
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
    """Список из 10 колонок-кандидатов, которые feature_engineering строит
    ДЛЯ ОДНОГО теста (5 по абсолютному значению: last/mean/q25/median/q75
    за окно, и 5 те же статистики, но по глобальному перцентилю теста).
    Именно из этого списка для каждого теста в test_combo выбирается
    РОВНО ОДНА статистика при переборе моделей."""
    return [
        f"{test_name}_last",
        f"mean_{test_name}_{suffix_days}d",
        f"q25_{test_name}_{suffix_days}d",
        f"median_{test_name}_{suffix_days}d",
        f"q75_{test_name}_{suffix_days}d",
        f"pct_{test_name}_last",
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
    сбалансированного датасета. Используется в Phase 1 ("lite") отбора
    top_fraction моделей (см. build_top_models()) - там нужны только
    числа (accuracy), а не сами модели/датафреймы, которые при большом
    числе комбинаций заняли бы всю оперативную память.

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
    """Полная (Phase 2 / "победитель") версия оценки: делает то же самое,
    что и fit_combo_accuracy_only(), а дополнительно дообучает финальную
    модель на ВСЕХ сбалансированных данных (без hold-out, для инференса)
    и возвращает сам сбалансированный датасет - вызывается ТОЛЬКО для
    наборов колонок, уже прошедших порог top_fraction в Phase 1 (см.
    build_top_models()).

    stat - уже посчитанные в Phase 1 (accuracy, accuracy_std, accuracies,
    n_train, n_test): тогда hold-out сплиты повторно не считаются.

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
# 4. Phase 1 ("lite"): accuracy-only перебор + потоковый отбор top-K
# --------------------------------------------------------------------------

# Колонки сырого df, которые нужны препроцессингу. Воркерам отправляются
# только они и только строки тестов своей комбинации - меньше pickle-трафика.
WORKER_COLS = [
    "patient_id", "test_type_code", "test_date",
    "test_result", "test_result_percentile", "target", "days_to_death",
]

# Компактная lite-запись одной модели Phase 1 (вместо dict ~1-1.5 КБ -
# ~130 байт). feature_cols не хранятся: они однозначно восстанавливаются
# из (combo_index, set_index) - set_index это позиция набора статистик в
# itertools.product(*кандидаты_по_тестам), см. _feature_cols_from_index().
# accuracies - accuracy по каждому разбиению SPLIT_RANDOM_STATES (хвост -
# NaN, если часть разбиений не удалась; фактическая длина - n_splits).
N_SPLITS_MAX = len(SPLIT_RANDOM_STATES)
LITE_DTYPE = np.dtype([
    ("key", "f8"),            # accuracy, округлённая до ACC_KEY_DECIMALS (ключ ранжирования)
    ("combo_index", "i8"),
    ("set_index", "i8"),
    ("accuracy", "f8"),       # точная средняя accuracy (как раньше)
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
    df_loc: pd.DataFrame,
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
    """Одна задача Phase 1 для joblib.

    Препроцессинг и оценка моделей - без изменений (тот же
    build_feature_table() и fit_feature_sets_accuracy_batched()).
    Меняется только упаковка результата:

      - n_valid    - число валидных моделей (stat is not None);
      - (vals, counts) - гистограмма accuracy-ключей ВСЕХ валидных моделей
                     комбинации (для распределения accuracy);
      - recs       - LITE_DTYPE-массив только тех моделей, чей ключ
                     >= min_key (текущий порог StreamingTopK на момент
                     отправки задачи; модели ниже порога уже не могут
                     попасть в итоговый top-N, поэтому не пересылаются).
    """
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


class StreamingTopK:
    """Потоковый отбор top-K компактных lite-записей (LITE_DTYPE) с
    ограниченной памятью - векторная замена heapq. Ничьи на пороге
    (одинаковый key) не разбиваются: берутся все.

    Порядок вывода (лучшие первыми): key по убыванию, затем combo_index и
    set_index по возрастанию - детерминированно при любом порядке прихода.

    Инвариант: после каждого push() объединение kept+pending содержит
    все поданные до сих пор записи с key >= key capacity-й лучшей из них.
    Отсюда: запись с key < threshold (key capacity-й лучшей записи на
    момент последнего сжатия) уже не может попасть ни в какой top-k,
    k <= capacity, и отбрасывается.

    Память: kept (<= capacity + ничьи на пороге) + pending (<= slack +
    одна пачка), при сжатии - временная копия; итого до ~2.5*capacity
    записей.
    """

    def __init__(self, capacity: int, dtype: np.dtype = LITE_DTYPE, min_slack: int = 100_000):
        self.capacity = max(1, int(capacity))
        self.min_slack = int(min_slack)
        self.threshold = -np.inf
        self.peak_records = 0
        self.n_compactions = 0
        self._kept = np.empty(0, dtype=dtype)
        self._pending: List[np.ndarray] = []
        self._n_pending = 0

    def set_capacity(self, capacity: int) -> None:
        """Ёмкость может только уменьшаться (верхняя граница числа
        победителей уточняется по ходу перебора)."""
        self.capacity = max(1, min(self.capacity, int(capacity)))

    def _slack(self) -> int:
        return max(self.min_slack, self.capacity // 4)

    @staticmethod
    def _order(recs: np.ndarray) -> np.ndarray:
        return np.lexsort((recs["set_index"], recs["combo_index"], -recs["key"]))

    def push(self, recs: np.ndarray) -> None:
        if recs.size == 0:
            return
        if self.threshold > -np.inf:
            recs = recs[recs["key"] >= self.threshold]
            if recs.size == 0:
                return
        self._pending.append(recs)
        self._n_pending += recs.size
        size = self._kept.size + self._n_pending
        self.peak_records = max(self.peak_records, size)
        # kept может быть больше capacity за счёт ничьих на пороге - тогда
        # сжимаем, когда накопится slack новых записей
        if size > max(self.capacity, self._kept.size) + self._slack():
            self._compact()

    def _compact(self) -> None:
        if self._pending:
            allr = np.concatenate([self._kept] + self._pending)
            self._pending = []
            self._n_pending = 0
        else:
            allr = self._kept
        if allr.size > self.capacity:
            order = self._order(allr)
            keys = allr["key"][order]
            self.threshold = float(keys[self.capacity - 1])
            # capacity лучших + все ничьи с capacity-й (keys по убыванию)
            n_keep = int(np.searchsorted(-keys, -self.threshold, side="right"))
            allr = allr[order[:n_keep]]
            self.n_compactions += 1
        self._kept = allr

    def result(self, k: int) -> np.ndarray:
        """Точный top-k (k <= capacity) плюс все ничьи с k-й записью,
        отсортированный от лучшей к худшей."""
        if k > self.capacity:
            raise ValueError(f"k={k} больше ёмкости буфера {self.capacity}: точный top-k не гарантирован")
        self._compact()
        kept = self._kept[self._order(self._kept)]
        if kept.size <= k:
            return kept
        return kept[kept["key"] >= kept["key"][k - 1]]


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
    одному, не держа в памяти сразу всех сразу (важно, когда top_fraction
    от гигантского числа моделей всё равно остаётся большим числом).

    Сбалансированный (1:1) датасет для этой конкретной модели хранится
    ВНУТРИ той же записи как TSV-текст (df.to_csv(sep="\t")) - раньше он
    писался отдельным .tsv файлом рядом с .pkl, теперь - здесь же, под
    ключом "balanced_dataset_tsv" (см. load_one_model_dataset() для
    обратного преобразования в pd.DataFrame)."""
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
    df_loc: pd.DataFrame,
    month: int,
    test_combo: Tuple[str, ...],
    winners: List[Tuple[Tuple[str, ...], Tuple[float, float, Tuple[float, ...], int, int]]],
    seed: np.random.SeedSequence,
    test_size: float,
    random_state: int,
    min_samples: int,
    window_days: int,
) -> Tuple[List[bytes], List[float]]:
    """Одна задача Phase 2 для joblib: перестраивает widetable комбинации
    (тем же seed, что и в Phase 1) и дообучает всех её "победителей" -
    winners = [(feature_cols, stat из Phase 1)], сплиты заново не считаются.
    Возвращает уже запиклленные записи (главный процесс только дописывает
    их байты в .pkl - это эквивалентно pickle.dump по одной) и их accuracy."""

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
# 6. Двухфазный отбор top-N моделей на пару (месяц, n_features)
# --------------------------------------------------------------------------

def _winner_tasks(
    winners: np.ndarray,
    combos: Sequence[Tuple[str, ...]],
    suffix_days: int,
):
    """Группирует победителей (LITE_DTYPE) по combo_index и ЛЕНИВО отдаёт
    (combo_index, [(feature_cols, stat), ...]) - Python-объекты создаются
    только для комбинации, которая сейчас отправляется в Phase 2.
    Внутри комбинации - от лучшей модели к худшей. stat - ровно тот же
    кортеж (accuracy, accuracy_std, accuracies, n_train, n_test), что
    вернула fit_feature_sets_accuracy_batched() в Phase 1."""
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
    tmp_dir: Optional[Path] = None,
    top_mode: str = TOP_MODE,
    top_fraction: float = TOP_FRACTION,
    top_count: int = TOP_COUNT,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    min_samples: int = MIN_SAMPLES,
    window_days: int = WINDOW_DAYS,
    n_jobs: int = -1,
    verbose: bool = True,
) -> Tuple[Dict[Tuple[int, int], Path], Dict[Tuple[int, int], List[float]]]:
    """Phase 1 (потоковый отбор top-N, см. StreamingTopK) + Phase 2
    (дообучение победителей и запись .pkl). tmp_dir больше не нужен
    (временных файлов нет) и оставлен только для совместимости вызова.

    Кроме .pkl, для каждой группы пишется гистограмма распределения
    accuracy, а в out_dir/selection_summary__<tag>.tsv - сводка: порог
    accuracy отобранного top-N, его перцентильный ранг в распределении
    accuracy всех валидных моделей, число ничьих на пороге и т.д.
    """

    if top_mode not in ("count", "fraction"):
        raise ValueError(
            f"Неизвестный top_mode={top_mode!r}. "
            f"Используйте 'count' или 'fraction'."
        )

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

    n_stat_candidates = len(
        test_feature_candidates("x", 0)
    )

    for month in months_list:
        for nf in n_features_list:

            combos = list(
                itertools.combinations(candidate_tests, nf)
            )
            n_sets = n_stat_candidates ** nf
            # комбинаторное число моделей группы (10 random states внутри
            # одной модели - НЕ отдельные модели): верхняя граница числа
            # валидных моделей
            n_models_upper = len(combos) * n_sets

            accuracies_by_group[(month, nf)] = []

            if top_mode == "fraction":
                capacity = max(1, math.ceil(top_fraction * n_models_upper))
            else:
                capacity = top_count

            if verbose:
                print(
                    f"[месяц {month}] n_features={nf}: "
                    f"{len(candidate_tests)} тестов-кандидатов, "
                    f"{len(combos)} комбинаций тестов x {n_sets} наборов "
                    f"статистик = {n_models_upper} моделей (верхняя граница); "
                    f"ёмкость буфера top-N: {capacity} "
                    f"(Phase 1: только accuracy)"
                )

            # ==============================================================
            # Phase 1: потоковый отбор top-N + гистограмма accuracy
            # ==============================================================

            topk = StreamingTopK(capacity)
            hist = AccuracyHistogram()
            n_valid_total = 0
            n_combos_done = 0

            # Генератор читается joblib лениво (pre_dispatch), поэтому
            # topk.threshold подставляется в КАЖДУЮ задачу на момент её
            # отправки - воркер сам отбрасывает заведомо проигравшие модели.
            jobs = (
                delayed(_combo_task_lite)(
                    pd.concat(
                        [by_test[t] for t in combo],
                        ignore_index=True,
                    ),
                    month,
                    combo,
                    i,
                    np.random.SeedSequence(
                        [random_state, month, nf, i]
                    ),
                    test_size,
                    random_state,
                    min_samples,
                    window_days,
                    topk.threshold,
                )
                for i, combo in enumerate(combos)
            )

            parallel = Parallel(
                n_jobs=n_jobs,
                return_as="generator_unordered",
            )

            for _m, _nf, _tried, n_valid, (h_vals, h_counts), recs in tqdm(
                parallel(jobs),
                total=len(combos),
                desc=f"Month {month}, n_features={nf}",
                unit="combo",
            ):
                n_valid_total += n_valid
                n_combos_done += 1
                hist.add(h_vals, h_counts)

                if top_mode == "fraction":
                    # уточнённая верхняя граница числа победителей:
                    # валидные уже + все ещё не посчитанные модели
                    remaining = (len(combos) - n_combos_done) * n_sets
                    topk.set_capacity(math.ceil(top_fraction * (n_valid_total + remaining)))

                topk.push(recs)

            if n_valid_total == 0:
                if verbose:
                    print(
                        f"  [месяц {month}] ни одной валидной "
                        f"комбинации для n_features={nf} не найдено "
                        f"(проверьте min_samples / плотность измерений)"
                    )
                continue

            if top_mode == "fraction":
                n_select = max(1, int(round(n_valid_total * top_fraction)))
            else:
                n_select = min(top_count, n_valid_total)

            # top-N + все ничьи с N-й моделью, поэтому может быть > n_select
            winners = topk.result(n_select)
            n_winners = len(winners)
            assert n_winners >= n_select, (n_winners, n_select)

            threshold = float(winners["key"][-1])
            rank = hist.rank_of(threshold)
            h_values, h_counts = hist.finalize()
            n_tied_selected = int((winners["key"] == round(threshold, ACC_KEY_DECIMALS)).sum())

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
                "n_target": n_select,
                "n_selected": n_winners,
                "selected_share_pct": 100.0 * n_winners / n_valid_total,
                "accuracy_threshold": threshold,
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
                "buffer_peak_records": topk.peak_records,
                "buffer_peak_mb": topk.peak_records * LITE_DTYPE.itemsize / 2**20,
            })

            if verbose:
                print(
                    f"  [месяц {month}] n_features={nf}: валидных моделей: "
                    f"{n_valid_total}, отобрано {n_winners} "
                    f"(N={n_select} + ничьи на пороге; "
                    f"{100.0 * n_winners / n_valid_total:.2f}%); "
                    f"порог accuracy = {threshold:.6f} "
                    f"(лучшая {float(winners['key'][0]):.6f}); "
                    f"перцентильный ранг порога: {rank['pct_rank_mean']:.2f} "
                    f"[строго ниже {rank['pct_rank_strict']:.2f}, "
                    f"не выше {rank['pct_rank_weak']:.2f}]; "
                    f"ничьих на пороге: {rank['n_tied']} "
                    f"(из них отобрано {n_tied_selected}); "
                    f"пик буфера: {topk.peak_records} записей "
                    f"(~{topk.peak_records * LITE_DTYPE.itemsize / 2**20:.1f} МБ)"
                )

            # ==============================================================
            # Phase 2 - без изменений: дообучение победителей, запись .pkl
            # ==============================================================

            n_winner_combos = int(np.unique(winners["combo_index"]).size)

            out_path = (
                out_dir
                / (
                    f"month_{month}"
                    f"__n_features_{nf}"
                    f"__{tag}"
                    f".pkl"
                )
            )

            output_paths[(month, nf)] = out_path

            n_saved = 0

            refit_jobs = (
                delayed(_combo_task_refit)(
                    pd.concat(
                        [by_test[t] for t in combos[combo_index]],
                        ignore_index=True,
                    ),
                    month,
                    combos[combo_index],
                    combo_winners,
                    np.random.SeedSequence(
                        [random_state, month, nf, combo_index]
                    ),
                    test_size,
                    random_state,
                    min_samples,
                    window_days,
                )
                for combo_index, combo_winners in _winner_tasks(
                    winners, combos, window_days * month
                )
            )

            # "generator" (упорядоченный) - чтобы порядок записей в .pkl
            # не зависел от --n-jobs
            refit_parallel = Parallel(
                n_jobs=n_jobs,
                return_as="generator",
            )

            with open(out_path, "wb") as out_f:

                for blobs, accs in tqdm(
                    refit_parallel(refit_jobs),
                    total=n_winner_combos,
                    desc=f"Month {month}, n_features={nf} (Phase 2)",
                    unit="combo",
                ):

                    for blob in blobs:
                        out_f.write(blob)

                    accuracies_by_group[(month, nf)].extend(accs)
                    n_saved += len(blobs)

            del winners, topk

            if verbose:
                print(
                    f"  [месяц {month}] n_features={nf}: "
                    f"победителей найдено {n_winners}, "
                    f"сохранено моделей: {n_saved} -> {out_path}"
                )

    if summary_rows:
        summary_path = out_dir / f"selection_summary__{tag}.tsv"
        pd.DataFrame(summary_rows).to_csv(summary_path, sep="\t", index=False)
        if verbose:
            print(f"Сводка отбора (порог, перцентильный ранг, ничьи): {summary_path}")

    return output_paths, accuracies_by_group

# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Поиск логрегрессий по комбинациям лабораторных тестов на СЫРОМ "
                     "(длинном) датасете; глобальные перцентили считаются один раз на "
                     "исходной длинной таблице, отбор моделей - top_fraction (не top-K), "
                     "датасеты хранятся внутри .pkl вместе с моделью и accuracy."
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
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--min-samples", type=int, default=MIN_SAMPLES,
                         help="Мин. число строк ПОСЛЕ балансировки 1:1, чтобы комбинация считалась валидной")
    parser.add_argument("--window-days", type=int, default=WINDOW_DAYS,
                         help="Длина скользящего окна в днях (по умолчанию 30 = 1 'месяц')")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--n-jobs", type=int, default=-1,
                         help="Число параллельных процессов joblib (-1 = все ядра, 1 = без параллелизма)")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--tmp-dir", default=None,
                         help="Не используется (временных файлов больше нет); оставлен "
                              "для совместимости командной строки")
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
        n_jobs=args.n_jobs,
    )

    for (month, nf), path in sorted(output_paths.items()):
        n_models = len(accuracies_by_group.get((month, nf), []))
        print(f"  Готово: {path} ({n_models} моделей; accuracy и сбалансированный "
              f"датасет каждой модели хранятся вместе, внутри этого .pkl)")


    print("\nГотово.")
    # Явный выход с кодом 0 - гарантирует, что процесс (и вызвавшая его
    # баш-сессия) завершится сразу после окончания работы.
    sys.exit(0)


if __name__ == "__main__":
    main()