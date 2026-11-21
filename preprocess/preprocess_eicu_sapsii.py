"""Derive first-day SAPS-II component scores for the project eICU cohort.

The output deliberately follows the CSV contract used by the MIMIC-III/IV
SAPS-II queries::

    subject_id, hadm_id, icustay_id, sapsii, sapsii_prob, *_score

For eICU these identifiers map to ``uniquepid``,
``patienthealthsystemstayid`` and ``patientunitstayid`` respectively.  The
SAPS-II thresholds and mortality equation are the same as the MIT-LCP MIMIC
query.  Time-varying measurements are restricted to the first 24 hours.

eICU does not contain an official SAPS-II table.  The only approximation in
this implementation is admission type: elective operative stays are treated
as scheduled surgery, other operative stays as unscheduled surgery, and all
remaining stays as medical admissions.  Operative stays are identified from
``admissionDx`` and ICU admission source.  A summary of this classification
and component missingness is printed after extraction.

The large source tables are read in chunks.  APACHE first-day variables fill
gaps in the raw measurements; GCS, ventilation status, its paired arterial
blood gas and the validated 24-hour urine total use the APACHE variables when
available.  APACHE scores or predicted outcomes are never used.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

try:  # Support both ``python -m preprocess...`` and direct script execution.
    from .preprocess_eicu import classify_intake_output
except ImportError:  # pragma: no cover - exercised by direct command-line use
    from preprocess_eicu import classify_intake_output


DEFAULT_RAW_DIR = Path(
    r"E:\zyq\mimic\eicu-collaborative-research-database-2.0"
)
DEFAULT_COHORT_CSV = Path("data/eicu/patient_split.csv")
DEFAULT_OUTPUT = Path("data/eicu/eicu_sapsii.csv")

OBSERVATION_MINUTES = 24 * 60

SCORE_COLUMNS = [
    "age_score",
    "hr_score",
    "sysbp_score",
    "temp_score",
    "pao2fio2_score",
    "uo_score",
    "bun_score",
    "wbc_score",
    "potassium_score",
    "sodium_score",
    "bicarbonate_score",
    "bilirubin_score",
    "gcs_score",
    "comorbidity_score",
    "admissiontype_score",
]

OUTPUT_COLUMNS = [
    "subject_id",
    "hadm_id",
    "icustay_id",
    "sapsii",
    "sapsii_prob",
    *SCORE_COLUMNS,
]

LAB_KEYS = {
    "bun": "bun",
    "wbc x 1000": "wbc",
    "potassium": "potassium",
    "sodium": "sodium",
    "bicarbonate": "bicarbonate",
    "hco3": "bicarbonate",
    "total bilirubin": "bilirubin",
    "pao2": "pao2",
    "fio2": "fio2",
}

LAB_BOUNDS = {
    "bun": (0.0, 275.0),
    "wbc": (0.0, 1100.0),
    "potassium": (0.0, 15.0),
    "sodium": (0.0, 250.0),
    "bicarbonate": (0.0, 66.0),
    "bilirubin": (0.0, 66.0),
    "pao2": (0.0, 770.0),
    "fio2": (0.2, 1.0),
}


def _normalise_text(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.lower()
    )


def _numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").astype(float)
    extracted = series.astype("string").str.extract(
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)", expand=False
    )
    return pd.to_numeric(extracted, errors="coerce").astype(float)


def _parse_age(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    text = text.mask(text.str.fullmatch(r">\s*89", na=False), "90")
    return pd.to_numeric(text, errors="coerce")


def _require_tables(raw_dir: Path, names: Iterable[str]) -> None:
    missing = [name for name in names if not (raw_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required eICU source tables under {raw_dir}: {missing}"
        )


def _read_chunks(
    path: Path,
    usecols: Sequence[str],
    chunk_size: int,
    max_chunks: Optional[int],
):
    reader = pd.read_csv(
        path, usecols=list(usecols), chunksize=chunk_size, low_memory=False
    )
    for index, chunk in enumerate(reader):
        if max_chunks is not None and index >= max_chunks:
            break
        yield chunk


def _filter_first_day(
    chunk: pd.DataFrame,
    id_column: str,
    offset_column: str,
    cohort_ids: set[int],
    observation_minutes: int,
) -> pd.DataFrame:
    ids = pd.to_numeric(chunk[id_column], errors="coerce")
    offsets = pd.to_numeric(chunk[offset_column], errors="coerce")
    mask = (
        ids.isin(cohort_ids)
        & offsets.ge(0)
        & offsets.le(observation_minutes)
    )
    result = chunk.loc[mask].copy()
    result[id_column] = ids.loc[mask].astype(np.int64)
    result[offset_column] = offsets.loc[mask].astype(float)
    return result


def _merge_extrema(
    accumulated: Optional[pd.DataFrame], current: pd.DataFrame
) -> pd.DataFrame:
    if current.empty:
        return accumulated if accumulated is not None else current
    if accumulated is None or accumulated.empty:
        return current
    combined = pd.concat([accumulated, current], axis=0, sort=False)
    aggregation = {
        column: ("min" if column.endswith("_min") else "max")
        for column in combined.columns
    }
    return combined.groupby(level=0, sort=False).agg(aggregation)


def load_cohort(
    patient_path: Path,
    cohort_csv: Optional[Path],
    include_all_stays: bool,
) -> pd.DataFrame:
    columns = [
        "patientunitstayid",
        "patienthealthsystemstayid",
        "uniquepid",
        "age",
        "unitadmitsource",
        "hospitaladmitsource",
        "unitstaytype",
        "apacheadmissiondx",
    ]
    patient = pd.read_csv(patient_path, usecols=columns, low_memory=False)
    patient["patientunitstayid"] = pd.to_numeric(
        patient["patientunitstayid"], errors="coerce"
    )
    patient["age_numeric"] = _parse_age(patient["age"])
    patient = patient.dropna(subset=["patientunitstayid", "age_numeric", "uniquepid"])
    patient = patient[patient["age_numeric"].ge(18)].copy()
    patient["patientunitstayid"] = patient["patientunitstayid"].astype(np.int64)

    if not include_all_stays:
        if cohort_csv is None or not cohort_csv.exists():
            raise FileNotFoundError(
                "The project cohort file is required unless --all-stays is used: "
                f"{cohort_csv}"
            )
        cohort_ids = pd.read_csv(cohort_csv, usecols=["ts_id"])["ts_id"]
        cohort_ids = set(pd.to_numeric(cohort_ids, errors="coerce").dropna().astype(int))
        patient = patient[patient["patientunitstayid"].isin(cohort_ids)].copy()

    patient = patient.drop_duplicates("patientunitstayid", keep="first")
    if patient.empty:
        raise RuntimeError("No adult eICU stays remain after cohort filtering")
    return patient


def load_apache_inputs(raw_dir: Path, cohort_ids: set[int]) -> pd.DataFrame:
    aps_columns = [
        "patientunitstayid",
        "intubated",
        "vent",
        "eyes",
        "motor",
        "verbal",
        "urine",
        "wbc",
        "temperature",
        "sodium",
        "heartrate",
        "pao2",
        "bun",
        "bilirubin",
        "fio2",
    ]
    aps = pd.read_csv(raw_dir / "apacheApsVar.csv.gz", usecols=aps_columns)
    aps = aps[aps["patientunitstayid"].isin(cohort_ids)].copy()
    aps = aps.drop_duplicates("patientunitstayid", keep="first")
    aps = aps.rename(
        columns={column: f"aps_{column}" for column in aps_columns[1:]}
    ).set_index("patientunitstayid")

    pred_columns = [
        "patientunitstayid",
        "aids",
        "lymphoma",
        "metastaticcancer",
        "leukemia",
        "electivesurgery",
        "ventday1",
        "oobventday1",
        "oobintubday1",
        "eyes",
        "motor",
        "verbal",
        "day1eyes",
        "day1motor",
        "day1verbal",
    ]
    pred = pd.read_csv(raw_dir / "apachePredVar.csv.gz", usecols=pred_columns)
    pred = pred[pred["patientunitstayid"].isin(cohort_ids)].copy()
    pred = pred.drop_duplicates("patientunitstayid", keep="first")
    pred = pred.rename(
        columns={column: f"pred_{column}" for column in pred_columns[1:]}
    ).set_index("patientunitstayid")
    return aps.join(pred, how="outer")


def extract_vitals(
    raw_dir: Path,
    cohort_ids: set[int],
    observation_minutes: int,
    chunk_size: int,
    max_chunks: Optional[int],
) -> pd.DataFrame:
    accumulated: Optional[pd.DataFrame] = None
    specs = [
        (
            raw_dir / "vitalPeriodic.csv.gz",
            "observationoffset",
            {
                "heartrate": ("hr", 0.0, 390.0),
                "systemicsystolic": ("sysbp", 0.0, 375.0),
                "temperature": ("temp", 14.2, 47.0),
            },
        ),
        (
            raw_dir / "vitalAperiodic.csv.gz",
            "observationoffset",
            {"noninvasivesystolic": ("sysbp", 0.0, 375.0)},
        ),
    ]

    for path, offset_column, mapping in specs:
        usecols = ["patientunitstayid", offset_column, *mapping]
        print(f"Reading {path.name} for SAPS-II vital signs...")
        for chunk in _read_chunks(path, usecols, chunk_size, max_chunks):
            chunk = _filter_first_day(
                chunk,
                "patientunitstayid",
                offset_column,
                cohort_ids,
                observation_minutes,
            )
            if chunk.empty:
                continue
            values = pd.DataFrame({"patientunitstayid": chunk["patientunitstayid"]})
            for source, (target, lower, upper) in mapping.items():
                numeric = _numeric(chunk[source])
                if target == "temp":
                    fahrenheit = numeric.gt(60)
                    numeric.loc[fahrenheit] = (numeric.loc[fahrenheit] - 32.0) * 5.0 / 9.0
                values[target] = numeric.where(numeric.between(lower, upper))
            grouped = values.groupby("patientunitstayid", sort=False).agg(
                **{
                    f"{target}_min": (target, "min")
                    for target, _, _ in mapping.values()
                },
                **{
                    f"{target}_max": (target, "max")
                    for target, _, _ in mapping.values()
                },
            )
            accumulated = _merge_extrema(accumulated, grouped)

    return accumulated if accumulated is not None else pd.DataFrame()


def extract_labs(
    raw_dir: Path,
    cohort_ids: set[int],
    observation_minutes: int,
    chunk_size: int,
    max_chunks: Optional[int],
) -> tuple[pd.DataFrame, pd.Series]:
    accumulated: Optional[pd.DataFrame] = None
    blood_gas_parts: list[pd.DataFrame] = []
    columns = ["patientunitstayid", "labresultoffset", "labname", "labresult"]
    path = raw_dir / "lab.csv.gz"
    print(f"Reading {path.name} for SAPS-II laboratory values...")

    for chunk in _read_chunks(path, columns, chunk_size, max_chunks):
        chunk = _filter_first_day(
            chunk,
            "patientunitstayid",
            "labresultoffset",
            cohort_ids,
            observation_minutes,
        )
        if chunk.empty:
            continue
        chunk["key"] = _normalise_text(chunk["labname"]).map(LAB_KEYS)
        chunk = chunk[chunk["key"].notna()].copy()
        if chunk.empty:
            continue
        chunk["value"] = _numeric(chunk["labresult"])
        fio2 = chunk["key"].eq("fio2") & chunk["value"].gt(1)
        chunk.loc[fio2, "value"] = chunk.loc[fio2, "value"] / 100.0
        lower = chunk["key"].map(lambda key: LAB_BOUNDS[key][0])
        upper = chunk["key"].map(lambda key: LAB_BOUNDS[key][1])
        chunk = chunk[chunk["value"].between(lower, upper, inclusive="both")]
        if chunk.empty:
            continue

        blood_gas_parts.append(
            chunk.loc[
                chunk["key"].isin(["pao2", "fio2"]),
                ["patientunitstayid", "labresultoffset", "key", "value"],
            ].copy()
        )
        non_bg = chunk[~chunk["key"].isin(["pao2", "fio2"])]
        if non_bg.empty:
            continue
        grouped = non_bg.groupby(["patientunitstayid", "key"], sort=False)[
            "value"
        ].agg(["min", "max"])
        grouped = grouped.unstack("key")
        grouped.columns = [f"{key}_{stat}" for stat, key in grouped.columns]
        accumulated = _merge_extrema(accumulated, grouped)

    lab_summary = accumulated if accumulated is not None else pd.DataFrame()
    if not blood_gas_parts:
        return lab_summary, pd.Series(dtype=float, name="pao2fio2_min")

    blood_gas = pd.concat(blood_gas_parts, ignore_index=True)
    paired = blood_gas.pivot_table(
        index=["patientunitstayid", "labresultoffset"],
        columns="key",
        values="value",
        aggfunc="mean",
    )
    if not {"pao2", "fio2"}.issubset(paired.columns):
        return lab_summary, pd.Series(dtype=float, name="pao2fio2_min")
    paired = paired.dropna(subset=["pao2", "fio2"])
    paired = paired[paired["fio2"].gt(0)]
    ratio = (paired["pao2"] / paired["fio2"]).groupby(level=0).min()
    ratio.name = "pao2fio2_min"
    return lab_summary, ratio


def extract_urine_output(
    raw_dir: Path,
    cohort_ids: set[int],
    observation_minutes: int,
    chunk_size: int,
    max_chunks: Optional[int],
) -> pd.Series:
    columns = [
        "patientunitstayid",
        "intakeoutputoffset",
        "cellpath",
        "celllabel",
        "cellvaluenumeric",
    ]
    path = raw_dir / "intakeOutput.csv.gz"
    print(f"Reading {path.name} for SAPS-II urine output...")
    accumulated: Optional[pd.Series] = None
    for chunk in _read_chunks(path, columns, chunk_size, max_chunks):
        chunk = _filter_first_day(
            chunk,
            "patientunitstayid",
            "intakeoutputoffset",
            cohort_ids,
            observation_minutes,
        )
        if chunk.empty:
            continue
        variable = classify_intake_output(chunk["celllabel"], chunk["cellpath"])
        chunk = chunk[variable.eq("Urine")].copy()
        if chunk.empty:
            continue
        chunk["value"] = _numeric(chunk["cellvaluenumeric"])
        chunk = chunk[chunk["value"].between(0, 2500, inclusive="both")]
        grouped = chunk.groupby("patientunitstayid", sort=False)["value"].sum()
        accumulated = grouped if accumulated is None else accumulated.add(grouped, fill_value=0)
    if accumulated is None:
        accumulated = pd.Series(dtype=float)
    accumulated.name = "urineoutput"
    return accumulated


def extract_surgical_flags(
    raw_dir: Path, cohort_ids: set[int]
) -> pd.Series:
    columns = ["patientunitstayid", "admitdxpath", "admitdxname", "admitdxtext"]
    frame = pd.read_csv(raw_dir / "admissionDx.csv.gz", usecols=columns, low_memory=False)
    frame = frame[frame["patientunitstayid"].isin(cohort_ids)].copy()
    path = _normalise_text(frame["admitdxpath"])
    name = _normalise_text(frame["admitdxname"])
    text = _normalise_text(frame["admitdxtext"])
    operative_path = path.str.contains(r"\|operative(?:\||$)", regex=True, na=False)
    or_question = path.str.contains(
        r"admitted from the o\.r\.|went to the o\.r\.", regex=True, na=False
    ) & (name.eq("yes") | text.eq("yes"))
    frame["surgical"] = operative_path | or_question
    result = frame.groupby("patientunitstayid")["surgical"].max()
    result.name = "surgical_dx"
    return result


def _valid(series: pd.Series, lower: float, upper: float) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.where(numeric.between(lower, upper, inclusive="both"))


def _fallback(primary: pd.Series, secondary: pd.Series) -> pd.Series:
    return primary.combine_first(secondary.reindex(primary.index))


def assemble_raw_features(
    patient: pd.DataFrame,
    apache: pd.DataFrame,
    vitals: pd.DataFrame,
    labs: pd.DataFrame,
    raw_pao2fio2: pd.Series,
    raw_urine: pd.Series,
    surgical_dx: pd.Series,
) -> pd.DataFrame:
    frame = patient.set_index("patientunitstayid").copy()
    frame = frame.join(apache, how="left")
    frame = frame.join(vitals, how="left")
    frame = frame.join(labs, how="left")
    frame = frame.join(raw_pao2fio2, how="left")
    frame = frame.join(raw_urine, how="left")
    frame = frame.join(surgical_dx, how="left")

    index = frame.index

    def column(name: str) -> pd.Series:
        if name in frame:
            return pd.to_numeric(frame[name], errors="coerce")
        return pd.Series(np.nan, index=index, dtype=float)

    # Raw, timestamped observations are primary for these extrema. APACHE
    # variables fill gaps when the event tables have no valid first-day value.
    frame["hr_min"] = _fallback(column("hr_min"), _valid(column("aps_heartrate"), 0, 390))
    frame["hr_max"] = _fallback(column("hr_max"), _valid(column("aps_heartrate"), 0, 390))
    frame["temp_min"] = _fallback(column("temp_min"), _valid(column("aps_temperature"), 14.2, 47))
    frame["temp_max"] = _fallback(column("temp_max"), _valid(column("aps_temperature"), 14.2, 47))

    fallback_specs = {
        "bun": ("aps_bun", 0, 275),
        "wbc": ("aps_wbc", 0, 1100),
        "sodium": ("aps_sodium", 0, 250),
        "bilirubin": ("aps_bilirubin", 0, 66),
    }
    for key, (aps_column, lower, upper) in fallback_specs.items():
        fallback = _valid(column(aps_column), lower, upper)
        frame[f"{key}_min"] = _fallback(column(f"{key}_min"), fallback)
        frame[f"{key}_max"] = _fallback(column(f"{key}_max"), fallback)

    # The APACHE table stores FiO2 as a percentage and the PaO2/FiO2 values
    # belong to the same worst ABG set.
    aps_pao2 = _valid(column("aps_pao2"), 0, 770)
    aps_fio2 = _valid(column("aps_fio2"), 20, 100) / 100.0
    aps_ratio = aps_pao2 / aps_fio2
    ratios = pd.concat(
        [column("pao2fio2_min").rename("raw"), aps_ratio.rename("aps")], axis=1
    )
    frame["pao2fio2_min"] = ratios.min(axis=1, skipna=True)

    vent_columns = [
        "aps_vent",
        "aps_intubated",
        "pred_ventday1",
        "pred_oobventday1",
        "pred_oobintubday1",
    ]
    vent_values = pd.concat([column(name) for name in vent_columns], axis=1)
    frame["vent_day1"] = vent_values.eq(1).any(axis=1)
    frame.loc[~frame["vent_day1"], "pao2fio2_min"] = np.nan

    aps_urine = _valid(column("aps_urine"), 0, np.inf)
    raw_urine = column("urineoutput")
    frame["urineoutput"] = aps_urine.where(aps_urine.notna(), raw_urine)

    def valid_gcs(prefix: str) -> pd.Series:
        eyes = _valid(column(f"{prefix}eyes"), 1, 4)
        motor = _valid(column(f"{prefix}motor"), 1, 6)
        verbal = _valid(column(f"{prefix}verbal"), 1, 5)
        complete = eyes.notna() & motor.notna() & verbal.notna()
        return (eyes + motor + verbal).where(complete)

    gcs = valid_gcs("aps_")
    gcs = gcs.combine_first(valid_gcs("pred_day1"))
    gcs = gcs.combine_first(valid_gcs("pred_"))
    frame["mingcs"] = gcs

    aids = column("pred_aids").eq(1)
    hematologic = column("pred_lymphoma").eq(1) | column("pred_leukemia").eq(1)
    metastatic = column("pred_metastaticcancer").eq(1)
    frame["aids"] = aids
    frame["hematologic_malignancy"] = hematologic
    frame["metastatic_cancer"] = metastatic

    source = (
        _normalise_text(frame["unitadmitsource"])
        .fillna("")
        .str.cat(_normalise_text(frame["hospitaladmitsource"]).fillna(""), sep=" ")
    )
    surgical_source = source.str.contains(
        r"operating room|recovery room|\bpacu\b|post.?anesthesia",
        regex=True,
        na=False,
    )
    elective = column("pred_electivesurgery").eq(1)
    surgical = frame["surgical_dx"].fillna(False).astype(bool) | surgical_source | elective
    frame["admissiontype"] = np.select(
        [
            (surgical & elective).fillna(False).to_numpy(dtype=bool),
            surgical.fillna(False).to_numpy(dtype=bool),
        ],
        ["ScheduledSurgical", "UnscheduledSurgical"],
        default="Medical",
    )
    return frame


def calculate_sapsii_scores(features: pd.DataFrame) -> pd.DataFrame:
    """Apply the MIT-LCP MIMIC SAPS-II component thresholds."""

    result = pd.DataFrame(index=features.index)

    def values(name: str) -> np.ndarray:
        return pd.to_numeric(features[name], errors="coerce").to_numpy(dtype=float)

    age = values("age_numeric")
    result["age_score"] = np.select(
        [age < 40, age < 60, age < 70, age < 75, age < 80, age >= 80],
        [0, 7, 12, 15, 16, 18],
        default=np.nan,
    )

    hr_min, hr_max = values("hr_min"), values("hr_max")
    result["hr_score"] = np.select(
        [hr_min < 40, hr_max >= 160, hr_max >= 120, hr_min < 70,
         (hr_max >= 70) & (hr_max < 120) & (hr_min >= 70) & (hr_min < 120)],
        [11, 7, 4, 2, 0],
        default=np.nan,
    )

    sbp_min, sbp_max = values("sysbp_min"), values("sysbp_max")
    result["sysbp_score"] = np.select(
        [sbp_min < 70, sbp_min < 100, sbp_max >= 200,
         (sbp_max >= 100) & (sbp_max < 200) & (sbp_min >= 100) & (sbp_min < 200)],
        [13, 5, 2, 0],
        default=np.nan,
    )

    temp_min, temp_max = values("temp_min"), values("temp_max")
    result["temp_score"] = np.select(
        [temp_max >= 39, temp_min < 39], [3, 0], default=np.nan
    )

    ratio = values("pao2fio2_min")
    result["pao2fio2_score"] = np.select(
        [ratio < 100, ratio < 200, ratio >= 200], [11, 9, 6], default=np.nan
    )

    urine = values("urineoutput")
    result["uo_score"] = np.select(
        [urine < 500, urine < 1000, urine >= 1000], [11, 4, 0], default=np.nan
    )

    bun = values("bun_max")
    result["bun_score"] = np.select(
        [bun < 28, bun < 84, bun >= 84], [0, 6, 10], default=np.nan
    )

    wbc_min, wbc_max = values("wbc_min"), values("wbc_max")
    result["wbc_score"] = np.select(
        [wbc_min < 1, wbc_max >= 20,
         (wbc_max >= 1) & (wbc_max < 20) & (wbc_min >= 1) & (wbc_min < 20)],
        [12, 3, 0],
        default=np.nan,
    )

    potassium_min, potassium_max = values("potassium_min"), values("potassium_max")
    result["potassium_score"] = np.select(
        [potassium_min < 3, potassium_max >= 5,
         (potassium_max >= 3) & (potassium_max < 5)
         & (potassium_min >= 3) & (potassium_min < 5)],
        [3, 3, 0],
        default=np.nan,
    )

    sodium_min, sodium_max = values("sodium_min"), values("sodium_max")
    result["sodium_score"] = np.select(
        [sodium_min < 125, sodium_max >= 145,
         (sodium_max >= 125) & (sodium_max < 145)
         & (sodium_min >= 125) & (sodium_min < 145)],
        [5, 1, 0],
        default=np.nan,
    )

    bicarbonate_min, bicarbonate_max = values("bicarbonate_min"), values("bicarbonate_max")
    result["bicarbonate_score"] = np.select(
        [bicarbonate_min < 15, bicarbonate_min < 20,
         (bicarbonate_max >= 20) & (bicarbonate_min >= 20)],
        [6, 3, 0],
        default=np.nan,
    )

    bilirubin = values("bilirubin_max")
    result["bilirubin_score"] = np.select(
        [bilirubin < 4, bilirubin < 6, bilirubin >= 6], [0, 4, 9], default=np.nan
    )

    gcs = values("mingcs")
    result["gcs_score"] = np.select(
        [(gcs >= 3) & (gcs < 6), (gcs >= 6) & (gcs < 9),
         (gcs >= 9) & (gcs < 11), (gcs >= 11) & (gcs < 14),
         (gcs >= 14) & (gcs <= 15)],
        [26, 13, 7, 5, 0],
        default=np.nan,
    )

    result["comorbidity_score"] = np.select(
        [features["aids"].astype(bool),
         features["hematologic_malignancy"].astype(bool),
         features["metastatic_cancer"].astype(bool)],
        [17, 10, 9],
        default=0,
    )
    result["admissiontype_score"] = features["admissiontype"].map(
        {"ScheduledSurgical": 0, "Medical": 6, "UnscheduledSurgical": 8}
    )

    for name in SCORE_COLUMNS:
        result[name] = pd.array(result[name], dtype="Int64")
    result["sapsii"] = result[SCORE_COLUMNS].fillna(0).sum(axis=1).astype(int)
    score = result["sapsii"].astype(float)
    logit = -7.7631 + 0.0737 * score + 0.9971 * np.log(score + 1.0)
    result["sapsii_prob"] = 1.0 / (1.0 + np.exp(-logit))
    return result


def build_eicu_sapsii(
    raw_dir: Path,
    output_path: Path,
    cohort_csv: Optional[Path] = DEFAULT_COHORT_CSV,
    include_all_stays: bool = False,
    observation_minutes: int = OBSERVATION_MINUTES,
    chunk_size: int = 1_000_000,
    max_chunks: Optional[int] = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    output_path = Path(output_path)
    cohort_csv = Path(cohort_csv) if cohort_csv is not None else None
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace it: {output_path}")
    if observation_minutes <= 0 or chunk_size <= 0:
        raise ValueError("observation_minutes and chunk_size must be positive")

    required = [
        "patient.csv.gz",
        "apacheApsVar.csv.gz",
        "apachePredVar.csv.gz",
        "admissionDx.csv.gz",
        "vitalPeriodic.csv.gz",
        "vitalAperiodic.csv.gz",
        "lab.csv.gz",
        "intakeOutput.csv.gz",
    ]
    _require_tables(raw_dir, required)

    patient = load_cohort(
        raw_dir / "patient.csv.gz", cohort_csv, include_all_stays
    )
    cohort_ids = set(patient["patientunitstayid"].astype(int))
    print(f"SAPS-II cohort: {len(cohort_ids):,} adult ICU stays")

    apache = load_apache_inputs(raw_dir, cohort_ids)
    vitals = extract_vitals(
        raw_dir, cohort_ids, observation_minutes, chunk_size, max_chunks
    )
    labs, pao2fio2 = extract_labs(
        raw_dir, cohort_ids, observation_minutes, chunk_size, max_chunks
    )
    urine = extract_urine_output(
        raw_dir, cohort_ids, observation_minutes, chunk_size, max_chunks
    )
    surgical = extract_surgical_flags(raw_dir, cohort_ids)

    features = assemble_raw_features(
        patient, apache, vitals, labs, pao2fio2, urine, surgical
    )
    scores = calculate_sapsii_scores(features)

    output = pd.DataFrame(index=features.index)
    output["subject_id"] = features["uniquepid"]
    output["hadm_id"] = pd.to_numeric(
        features["patienthealthsystemstayid"], errors="coerce"
    ).astype("Int64")
    output["icustay_id"] = output.index.astype(np.int64)
    output["sapsii"] = scores["sapsii"]
    output["sapsii_prob"] = scores["sapsii_prob"]
    for name in SCORE_COLUMNS:
        output[name] = scores[name]
    output = output.reset_index(drop=True)[OUTPUT_COLUMNS].sort_values("icustay_id")

    if output["icustay_id"].duplicated().any():
        raise AssertionError("SAPS-II output contains duplicate ICU-stay IDs")
    if len(output) != len(patient):
        raise AssertionError("SAPS-II output lost cohort rows")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    print(f"Wrote {len(output):,} rows to {output_path}")
    print("Component availability:")
    for name in SCORE_COLUMNS:
        count = int(output[name].notna().sum())
        print(f"  {name:24s} {count:8,d} ({count / len(output):6.2%})")
    print("Admission type approximation:")
    for label, count in features["admissiontype"].value_counts().items():
        print(f"  {label:24s} {int(count):8,d} ({count / len(features):6.2%})")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract an eICU SAPS-II table compatible with MIMIC outputs."
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cohort-csv", type=Path, default=DEFAULT_COHORT_CSV)
    parser.add_argument(
        "--all-stays",
        action="store_true",
        help="Process every adult eICU stay instead of data/eicu/patient_split.csv.",
    )
    parser.add_argument("--observation-minutes", type=int, default=OBSERVATION_MINUTES)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Debug-only limit applied independently to each large table.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_eicu_sapsii(
        raw_dir=args.raw_dir,
        output_path=args.output,
        cohort_csv=None if args.all_stays else args.cohort_csv,
        include_all_stays=args.all_stays,
        observation_minutes=args.observation_minutes,
        chunk_size=args.chunk_size,
        max_chunks=args.max_chunks,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
