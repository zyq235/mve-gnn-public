"""Build an eICU-CRD cohort compatible with ``ICULargeGraphDataset``.

The generated pickle has exactly the same five-object contract as
``preprocess_mimic_iv.py``::

    [events, outcomes, train_ids, valid_ids, test_ids]

``events`` contains ``ts_id, minute, variable, value, TABLE`` and
``outcomes`` contains ``ts_id, in_hospital_mortality``.  Splits are made at
the patient level (``uniquepid``), never at the ICU-stay level.

Only information observed in the first observation window is used as model
input.  Hospital discharge status is used exclusively as the prediction
target.  APACHE predictions, discharge location, and post-window events are
not model inputs.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


DEFAULT_RAW_DATA_PATH = Path(
    r"E:\zyq\mimic\eicu-collaborative-research-database-2.0"
)
DEFAULT_OUTPUT_DIR = Path("data/eicu")
DEFAULT_REFERENCE_VAR2IND = Path("data/mimic-iv/var2ind.json")
DEFAULT_REFERENCE_META = Path("data/mimic-iv/var_mean_std.json")

EVENT_COLUMNS = ["ts_id", "minute", "variable", "value", "TABLE"]
DEMOGRAPHIC_VARIABLES = {"Age", "Gender"}


# The limits intentionally follow preprocess_mimic_iv.py where an equivalent
# feature exists.  They remove obvious unit errors and physiologically
# impossible values, rather than winsorising valid extreme measurements.
FEATURE_BOUNDS: Dict[str, Tuple[float, float]] = {
    "HR": (0, 390),
    "O2 Saturation": (0, 100),
    "RR": (0, 330),
    "DBP": (0, 375),
    "MBP": (0, 375),
    "SBP": (0, 375),
    "Temperature": (14.2, 47),
    "Weight": (0, 300),
    "Height": (0, 275),
    "FiO2": (0.2, 1),
    "Intubated": (0, 1),
    "GCS_eye": (1, 4),
    "GCS_motor": (1, 6),
    "GCS_verbal": (1, 5),
    "Glucose (Blood)": (0, 2200),
    "Glucose (Serum)": (0, 2200),
    "Sodium": (0, 250),
    "Potassium": (0, 15),
    "Magnesium": (0, 22),
    "Phosphate": (0, 22),
    "Calcium Total": (0, 40),
    "Calcium Free": (0, 10),
    "WBC": (0, 1100),
    "Hct": (0, 100),
    "Hgb": (0, 30),
    "Chloride": (0, 200),
    "Bicarbonate": (0, 66),
    "ALT": (0, 11000),
    "ALP": (0, 4000),
    "AST": (0, 22000),
    "Albumin": (0, 10),
    "Lactate": (0, 33),
    "LDH": (0, 35000),
    "SG Urine": (0, 2),
    "pH Urine": (0, 14),
    "pH Blood": (0, 14),
    "PO2": (0, 770),
    "PCO2": (0, 220),
    "Total CO2": (0, 65),
    "Base Excess": (-31, 28),
    "Basophils": (0, 100),
    "Eoisinophils": (0, 100),  # Keep the existing MIMIC-IV vocabulary spelling.
    "Lymphocytes": (0, 100),
    "Monocytes": (0, 100),
    "Neutrophils": (0, 100),
    "PT": (0, 150),
    "PTT": (0, 150),
    "INR": (0, 150),
    "Anion Gap": (0, 55),
    "BUN": (0, 275),
    "Creatinine Blood": (0, 66),
    "Creatinine Urine": (0, 650),
    "MCH": (0, 50),
    "MCHC": (0, 50),
    "MCV": (0, 150),
    "RDW": (0, 37),
    "Platelet Count": (0, 2200),
    "RBC": (0, 14),
    "Bilirubin (Total)": (0, 66),
    "Bilirubin (Direct)": (0, 66),
    "Bilirubin (Indirect)": (0, 66),
    "Urine": (0, 2500),
    "Ultrafiltrate": (0, 7000),
    "Stool": (0, 4000),
    "Chest Tube": (0, 2500),
    "Gastric": (0, 4000),
    "EBL": (0, 10000),
    "Emesis": (0, 2000),
    "Jackson-Pratt": (0, 2000),
    "Residual": (0, 1050),
    "Pre-admission Output": (0, 13000),
    "Lactated Ringers": (0, 17000),
    "PO intake": (0, 11000),
    "D5W": (0, 11000),
    "Dextrose Other": (0, 4000),
    "Normal Saline": (0, 11000),
    "Half Normal Saline": (0, 2000),
    "Sterile Water": (0, 10000),
    "Solution": (0, 1500),
    "GT Flush": (0, 2100),
    "Piggyback": (0, 1000),
    "Packed RBC": (0, 17250),
    "Albumin 25%": (0, 750),
    "Albumin 5%": (0, 1300),
    "Fresh Frozen Plasma": (0, 33000),
    "Colloid": (0, 20000),
    "TPN": (0, 1600),
    "Fiber": (0, 1600),
    # Unit-specific infusion-rate limits.  Unknown-concentration ml/hr rows
    # are rejected before these limits are applied.
    "Norepinephrine": (0, 1000),       # mcg/min
    "Epinephrine": (0, 1000),          # mcg/min
    "Neosynephrine": (0, 10000),       # mcg/min
    "Vasopressin": (0, 10),            # units/min
    "Dopamine": (0, 100),              # mcg/kg/min
    "Milrinone": (0, 10),              # mcg/kg/min
    "Propofol": (0, 500),              # mcg/kg/min
    "Fentanyl": (0, 10000),            # mcg/hr
    "Midazolam": (0, 100),             # mg/hr
    "Amiodarone": (0, 100),            # mg/min
    "Nitroglycerine": (0, 10000),       # mcg/min
    "Nitroprusside": (0, 100),          # mcg/kg/min
    "Insulin Regular": (0, 1000),       # units/hr
    "Heparin": (0, 100000),             # units/hr
    "Lorazepam": (0, 300),              # mg/hr in eICU
    "Furosemide": (0, 250),              # mg/hr in eICU
    "Hydromorphone": (0, 125),           # mg/hr in eICU
    "Pantoprazole": (0, 100),            # mg/hr in eICU
    "Magnesium Sulphate": (0, 125),      # converted from mg/hr to g/hr
    "Calcium Gluconate": (0, 200),       # converted from mg/hr to g/hr
}


VITAL_PERIODIC_MAP = {
    "temperature": "Temperature",
    "sao2": "O2 Saturation",
    "heartrate": "HR",
    "respiration": "RR",
    "systemicsystolic": "SBP",
    "systemicdiastolic": "DBP",
    "systemicmean": "MBP",
}

VITAL_APERIODIC_MAP = {
    "noninvasivesystolic": "SBP",
    "noninvasivediastolic": "DBP",
    "noninvasivemean": "MBP",
}


# Keys are lower-cased and whitespace-normalised eICU labname values.
LAB_MAP = {
    # eICU labtypeid=4 is bedside/point-of-care whole-blood glucose.
    "bedside glucose": "Glucose (Whole Blood)",
    "glucose": "Glucose (Serum)",
    "sodium": "Sodium",
    "potassium": "Potassium",
    "magnesium": "Magnesium",
    "phosphate": "Phosphate",
    "calcium": "Calcium Total",
    "ionized calcium": "Calcium Free",
    "wbc x 1000": "WBC",
    "hct": "Hct",
    "hgb": "Hgb",
    "chloride": "Chloride",
    "bicarbonate": "Bicarbonate",
    "hco3": "Bicarbonate",
    "alt (sgpt)": "ALT",
    "alkaline phos.": "ALP",
    "ast (sgot)": "AST",
    "albumin": "Albumin",
    "lactate": "Lactate",
    "ldh": "LDH",
    "urinary specific gravity": "SG Urine",
    "urinary ph": "pH Urine",
    "ph": "pH Blood",
    "pao2": "PO2",
    "paco2": "PCO2",
    "total co2": "Total CO2",
    "base excess": "Base Excess",
    "-basos": "Basophils",
    "-eos": "Eoisinophils",
    "-lymphs": "Lymphocytes",
    "-monos": "Monocytes",
    "-polys": "Neutrophils",
    "pt": "PT",
    "ptt": "PTT",
    "pt - inr": "INR",
    "anion gap": "Anion Gap",
    "bun": "BUN",
    "creatinine": "Creatinine Blood",
    "urinary creatinine": "Creatinine Urine",
    "mch": "MCH",
    "mchc": "MCHC",
    "mcv": "MCV",
    "rdw": "RDW",
    "platelets x 1000": "Platelet Count",
    "rbc": "RBC",
    "total bilirubin": "Bilirubin (Total)",
    "direct bilirubin": "Bilirubin (Direct)",
    "fio2": "FiO2",
    "o2 sat (%)": "O2 Saturation",
    "temperature": "Temperature",
}


# Only rows whose drug name explicitly states the comparable rate unit are
# included.  In particular, ml/hr rows cannot be converted without the bag
# concentration and are intentionally omitted.
INFUSION_SPECS: Sequence[Tuple[re.Pattern[str], str]] = (
    (re.compile(r"^.*(?:norepinephrine|levophed).*\(mcg/min\)$", re.I), "Norepinephrine"),
    (re.compile(r"^.*epinephrine.*\(mcg/min\)$", re.I), "Epinephrine"),
    (re.compile(r"^.*(?:phenylephrine|neosynephrine).*\(mcg/min\)$", re.I), "Neosynephrine"),
    (re.compile(r"^.*vasopressin.*\(units/min\)$", re.I), "Vasopressin"),
    (re.compile(r"^.*dopamine.*\(mcg/kg/min\)$", re.I), "Dopamine"),
    (re.compile(r"^.*milrinone.*\(mcg/kg/min\)$", re.I), "Milrinone"),
    (re.compile(r"^.*propofol.*\(mcg/kg/min\)$", re.I), "Propofol"),
    (re.compile(r"^.*fentanyl.*\(mcg/hr\)$", re.I), "Fentanyl"),
    (re.compile(r"^.*midazolam.*\(mg/hr\)$", re.I), "Midazolam"),
    (re.compile(r"^.*amiodarone.*\(mg/min\)$", re.I), "Amiodarone"),
    (re.compile(r"^.*nitroglycerin(?:e)?.*\(mcg/min\)$", re.I), "Nitroglycerine"),
    (re.compile(r"^.*nitroprusside.*\(mcg/kg/min\)$", re.I), "Nitroprusside"),
    (re.compile(r"^insulin(?:-? scale [a-z])? \(units/hr\)$", re.I), "Insulin Regular"),
    (re.compile(r"^.*heparin.*\(units/hr\)$", re.I), "Heparin"),
    (re.compile(r"^.*(?:lorazepam|ativan).*\(mg/hr\)$", re.I), "Lorazepam"),
    (re.compile(r"^.*(?:furosemide|lasix).*\(mg/hr\)$", re.I), "Furosemide"),
    (re.compile(r"^.*(?:hydromorphone|dilaudid).*\(mg/hr\)$", re.I), "Hydromorphone"),
    (re.compile(r"^.*(?:pantoprazole|protonix).*\(mg/hr\)(?: \(mg/hr\))?$", re.I), "Pantoprazole"),
    (re.compile(r"^.*magnesium(?: sulfate)?.*\(mg/hr\)$", re.I), "Magnesium Sulphate"),
    (re.compile(r"^.*calcium gluconate.*\(mg/hr\)$", re.I), "Calcium Gluconate"),
)


INTAKE_OUTPUT_VARIABLES = {
    "Urine",
    "Ultrafiltrate",
    "Stool",
    "Chest Tube",
    "Gastric",
    "EBL",
    "Emesis",
    "Jackson-Pratt",
    "Residual",
    "Pre-admission Output",
    "Lactated Ringers",
    "PO intake",
    "D5W",
    "Dextrose Other",
    "Normal Saline",
    "Half Normal Saline",
    "Sterile Water",
    "Solution",
    "GT Flush",
    "Piggyback",
    "Packed RBC",
    "Albumin 25%",
    "Albumin 5%",
    "Fresh Frozen Plasma",
    "Colloid",
    "TPN",
    "Fiber",
}


FALLBACK_SEMANTICS = {
    "Age": ("Demographics & Status", "Patient Demographics"),
    "Gender": ("Demographics & Status", "Patient Demographics"),
    "HR": ("Vital Signs", "Vital Signs"),
    "O2 Saturation": ("Vital Signs", "Vital Signs"),
    "RR": ("Vital Signs", "Vital Signs"),
    "DBP": ("Vital Signs", "Vital Signs"),
    "MBP": ("Vital Signs", "Vital Signs"),
    "SBP": ("Vital Signs", "Vital Signs"),
    "Temperature": ("Vital Signs", "Vital Signs"),
    "Weight": ("Vital Signs", "Anthropometry"),
    "Height": ("Vital Signs", "Anthropometry"),
}


def _normalise_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.replace(r"\s+", " ", regex=True).str.lower()


def _numeric(series: pd.Series) -> pd.Series:
    """Coerce eICU values such as ``'50 %'`` to numeric."""

    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
    else:
        extracted = series.astype("string").str.extract(
            r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)", expand=False
        )
        numeric = pd.to_numeric(extracted, errors="coerce")

    # pandas may infer nullable Int64 when a chunk happens to contain only
    # integral-looking values (for example FiO2 values such as "50 %").
    # Several unit conversions below create fractional values and therefore
    # must not assign back into an integer extension array.
    return numeric.astype("Float64")


def _parse_age(series: pd.Series) -> pd.Series:
    # eICU masks ages above 89 as the literal string "> 89".  The official
    # convention used by many eICU benchmarks is to encode it as 90.
    text = series.astype("string").str.strip()
    text = text.mask(text.str.fullmatch(r">\s*89", na=False), "90")
    return pd.to_numeric(text, errors="coerce")


def _read_chunks(
    path: Path,
    usecols: Sequence[str],
    chunk_size: int,
    max_chunks: Optional[int],
) -> Iterator[pd.DataFrame]:
    if not path.exists():
        print(f"[warning] Optional source table not found; skipping: {path.name}")
        return
    reader = pd.read_csv(path, usecols=list(usecols), chunksize=chunk_size, low_memory=False)
    for index, chunk in enumerate(reader):
        if max_chunks is not None and index >= max_chunks:
            break
        yield chunk


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame(columns=EVENT_COLUMNS)


def _compact_events(frame: pd.DataFrame, table: str) -> pd.DataFrame:
    if frame.empty:
        return _empty_events()
    result = frame[["ts_id", "minute", "variable", "value"]].copy()
    result["ts_id"] = pd.to_numeric(result["ts_id"], errors="coerce")
    result["minute"] = pd.to_numeric(result["minute"], errors="coerce")
    result["value"] = pd.to_numeric(result["value"], errors="coerce")
    result = result.dropna(subset=["ts_id", "minute", "variable", "value"])
    if result.empty:
        return _empty_events()

    result["ts_id"] = result["ts_id"].astype(np.int64)
    result["minute"] = np.floor(result["minute"]).astype(np.int32)
    result["value"] = result["value"].astype(np.float32)
    result["variable"] = result["variable"].astype("string")

    lower = result["variable"].map(lambda name: FEATURE_BOUNDS.get(name, (-np.inf, np.inf))[0])
    upper = result["variable"].map(lambda name: FEATURE_BOUNDS.get(name, (-np.inf, np.inf))[1])
    result = result[result["value"].between(lower, upper, inclusive="both")]
    if result.empty:
        return _empty_events()

    result["TABLE"] = table
    # Aggregating every chunk controls memory and removes duplicate readings
    # produced by synonymous columns within a source table.
    result = (
        result.groupby(["ts_id", "minute", "variable"], observed=True, sort=False)
        .agg(value=("value", "mean"), TABLE=("TABLE", "first"))
        .reset_index()
    )
    return result[EVENT_COLUMNS]


def _filter_window(
    frame: pd.DataFrame,
    id_column: str,
    offset_column: str,
    cohort_ids: set,
    observation_minutes: int,
) -> pd.DataFrame:
    ids = pd.to_numeric(frame[id_column], errors="coerce")
    offsets = pd.to_numeric(frame[offset_column], errors="coerce")
    mask = ids.isin(cohort_ids) & offsets.ge(0) & offsets.lt(observation_minutes)
    frame = frame.loc[mask].copy()
    frame[id_column] = ids.loc[mask].astype(np.int64)
    frame[offset_column] = np.floor(offsets.loc[mask]).astype(np.int32)
    return frame


def build_candidate_cohort(patient_path: Path, observation_minutes: int) -> pd.DataFrame:
    columns = [
        "patientunitstayid",
        "uniquepid",
        "gender",
        "age",
        "admissionheight",
        "admissionweight",
        "unitdischargeoffset",
        "hospitaldischargeoffset",
        "hospitaldischargestatus",
    ]
    patient = pd.read_csv(patient_path, usecols=columns, low_memory=False)
    patient = patient.rename(
        columns={
            "patientunitstayid": "ts_id",
            "uniquepid": "patient_id",
        }
    )
    patient["ts_id"] = pd.to_numeric(patient["ts_id"], errors="coerce")
    patient["age_numeric"] = _parse_age(patient["age"])
    patient["unitdischargeoffset"] = pd.to_numeric(
        patient["unitdischargeoffset"], errors="coerce"
    )
    patient["hospitaldischargeoffset"] = pd.to_numeric(
        patient["hospitaldischargeoffset"], errors="coerce"
    )

    status = patient["hospitaldischargestatus"].astype("string").str.strip().str.lower()
    gender = patient["gender"].astype("string").str.strip().str.lower()
    patient["in_hospital_mortality"] = status.map({"alive": 0, "expired": 1})
    patient["gender_numeric"] = gender.map({"male": 0, "female": 1})

    valid_status = patient["in_hospital_mortality"].notna()
    valid_gender = patient["gender_numeric"].notna()
    adult = patient["age_numeric"].ge(18)
    complete_window = patient["unitdischargeoffset"].ge(observation_minutes)
    # A positive target must not occur inside the predictor window.
    no_early_death = (
        patient["in_hospital_mortality"].eq(0)
        | patient["hospitaldischargeoffset"].ge(observation_minutes)
    )
    has_patient_key = patient["patient_id"].notna()

    cohort = patient.loc[
        valid_status & valid_gender & adult & complete_window & no_early_death & has_patient_key
    ].copy()
    cohort["ts_id"] = cohort["ts_id"].astype(np.int64)
    cohort["in_hospital_mortality"] = cohort["in_hospital_mortality"].astype(np.int8)
    cohort["gender_numeric"] = cohort["gender_numeric"].astype(np.int8)
    cohort = cohort.drop_duplicates("ts_id", keep="first")
    return cohort


def extract_static_events(cohort: pd.DataFrame) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for source, variable in (
        ("age_numeric", "Age"),
        ("gender_numeric", "Gender"),
        ("admissionheight", "Height"),
        ("admissionweight", "Weight"),
    ):
        frame = cohort[["ts_id", source]].rename(columns={source: "value"}).copy()
        frame["minute"] = 0
        frame["variable"] = variable
        frames.append(_compact_events(frame, "N/A"))
    return pd.concat(frames, ignore_index=True) if frames else _empty_events()


def extract_wide_vitals(
    path: Path,
    mapping: Mapping[str, str],
    id_column: str,
    offset_column: str,
    table_name: str,
    cohort_ids: set,
    observation_minutes: int,
    bin_minutes: int,
    chunk_size: int,
    max_chunks: Optional[int],
) -> pd.DataFrame:
    usecols = [id_column, offset_column, *mapping.keys()]
    collected: List[pd.DataFrame] = []
    iterator = _read_chunks(path, usecols, chunk_size, max_chunks)
    for chunk in tqdm(iterator, desc=table_name):
        chunk = _filter_window(
            chunk, id_column, offset_column, cohort_ids, observation_minutes
        )
        if chunk.empty:
            continue
        chunk = chunk.rename(columns={id_column: "ts_id", offset_column: "minute"})
        if bin_minutes > 1:
            chunk["minute"] = (chunk["minute"] // bin_minutes) * bin_minutes
        long = chunk.melt(
            id_vars=["ts_id", "minute"],
            value_vars=list(mapping),
            var_name="source_variable",
            value_name="value",
        )
        long["variable"] = long["source_variable"].map(mapping)
        long["value"] = _numeric(long["value"])
        collected.append(_compact_events(long, table_name))
    return pd.concat(collected, ignore_index=True) if collected else _empty_events()


def extract_labs(
    path: Path,
    cohort_ids: set,
    observation_minutes: int,
    chunk_size: int,
    max_chunks: Optional[int],
) -> pd.DataFrame:
    columns = [
        "patientunitstayid",
        "labresultoffset",
        "labname",
        "labresult",
    ]
    collected: List[pd.DataFrame] = []
    iterator = _read_chunks(path, columns, chunk_size, max_chunks)
    for chunk in tqdm(iterator, desc="lab"):
        chunk = _filter_window(
            chunk,
            "patientunitstayid",
            "labresultoffset",
            cohort_ids,
            observation_minutes,
        )
        if chunk.empty:
            continue
        chunk["lab_key"] = _normalise_text(chunk["labname"])
        chunk["variable"] = chunk["lab_key"].map(LAB_MAP)
        chunk = chunk[chunk["variable"].notna()].copy()
        if chunk.empty:
            continue
        chunk["value"] = _numeric(chunk["labresult"])

        # eICU stores some values as percent/Fahrenheit depending on hospital.
        fio2 = chunk["variable"].eq("FiO2") & chunk["value"].gt(1)
        chunk.loc[fio2, "value"] = chunk.loc[fio2, "value"] / 100.0
        temp_f = chunk["variable"].eq("Temperature") & chunk["value"].gt(60)
        chunk.loc[temp_f, "value"] = (chunk.loc[temp_f, "value"] - 32.0) * 5.0 / 9.0

        chunk = chunk.rename(
            columns={"patientunitstayid": "ts_id", "labresultoffset": "minute"}
        )
        collected.append(_compact_events(chunk, "lab"))
    if not collected:
        return _empty_events()

    laboratory_events = pd.concat(collected, ignore_index=True)
    # eICU records total and direct bilirubin but has no separate indirect
    # bilirubin field.  When both were measured at the same minute, the
    # clinically defined difference is equivalent to MIMIC's indirect value.
    bilirubin = laboratory_events[
        laboratory_events["variable"].isin(
            ["Bilirubin (Total)", "Bilirubin (Direct)"]
        )
    ]
    if not bilirubin.empty:
        paired = bilirubin.pivot_table(
            index=["ts_id", "minute"],
            columns="variable",
            values="value",
            aggfunc="mean",
        )
        required = {"Bilirubin (Total)", "Bilirubin (Direct)"}
        if required.issubset(paired.columns):
            paired = paired.dropna(subset=list(required)).reset_index()
            paired["value"] = (
                paired["Bilirubin (Total)"] - paired["Bilirubin (Direct)"]
            )
            paired["variable"] = "Bilirubin (Indirect)"
            derived = _compact_events(paired, "lab_derived")
            laboratory_events = pd.concat(
                [laboratory_events, derived], ignore_index=True
            )
    return laboratory_events


def extract_nurse_charting(
    path: Path,
    cohort_ids: set,
    observation_minutes: int,
    chunk_size: int,
    max_chunks: Optional[int],
) -> pd.DataFrame:
    columns = [
        "patientunitstayid",
        "nursingchartoffset",
        "nursingchartcelltypevallabel",
        "nursingchartcelltypevalname",
        "nursingchartvalue",
    ]
    collected: List[pd.DataFrame] = []
    iterator = _read_chunks(path, columns, chunk_size, max_chunks)
    for chunk in tqdm(iterator, desc="nurseCharting"):
        chunk = _filter_window(
            chunk,
            "patientunitstayid",
            "nursingchartoffset",
            cohort_ids,
            observation_minutes,
        )
        if chunk.empty:
            continue
        label = _normalise_text(chunk["nursingchartcelltypevallabel"])
        name = _normalise_text(chunk["nursingchartcelltypevalname"])
        chunk["variable"] = pd.Series(pd.NA, index=chunk.index, dtype="string")

        gcs = label.eq("glasgow coma score")
        chunk.loc[gcs & name.eq("eyes"), "variable"] = "GCS_eye"
        chunk.loc[gcs & name.eq("motor"), "variable"] = "GCS_motor"
        chunk.loc[gcs & name.eq("verbal"), "variable"] = "GCS_verbal"

        # Nurse-charted temperature/O2 fills gaps at hospitals whose periodic
        # interface is incomplete.  Duplicate same-minute values are averaged
        # during final aggregation.
        chunk.loc[name.eq("o2 saturation"), "variable"] = "O2 Saturation"
        chunk.loc[name.eq("temperature (c)"), "variable"] = "Temperature"
        chunk.loc[name.eq("temperature (f)"), "variable"] = "Temperature"

        chunk = chunk[chunk["variable"].notna()].copy()
        if chunk.empty:
            continue
        chunk["value"] = _numeric(chunk["nursingchartvalue"])
        is_fahrenheit = _normalise_text(chunk["nursingchartcelltypevalname"]).eq(
            "temperature (f)"
        )
        chunk.loc[is_fahrenheit, "value"] = (
            chunk.loc[is_fahrenheit, "value"] - 32.0
        ) * 5.0 / 9.0
        chunk = chunk.rename(
            columns={"patientunitstayid": "ts_id", "nursingchartoffset": "minute"}
        )
        collected.append(_compact_events(chunk, "nurseCharting"))
    return pd.concat(collected, ignore_index=True) if collected else _empty_events()


def extract_respiratory_charting(
    path: Path,
    cohort_ids: set,
    observation_minutes: int,
    chunk_size: int,
    max_chunks: Optional[int],
) -> pd.DataFrame:
    columns = [
        "patientunitstayid",
        "respchartoffset",
        "respchartvaluelabel",
        "respchartvalue",
    ]
    collected: List[pd.DataFrame] = []
    iterator = _read_chunks(path, columns, chunk_size, max_chunks)
    for chunk in tqdm(iterator, desc="respiratoryCharting"):
        chunk = _filter_window(
            chunk,
            "patientunitstayid",
            "respchartoffset",
            cohort_ids,
            observation_minutes,
        )
        if chunk.empty:
            continue
        label = _normalise_text(chunk["respchartvaluelabel"])
        chunk = chunk[label.eq("fio2")].copy()
        if chunk.empty:
            continue
        chunk["variable"] = "FiO2"
        chunk["value"] = _numeric(chunk["respchartvalue"])
        percent = chunk["value"].gt(1)
        chunk.loc[percent, "value"] = chunk.loc[percent, "value"] / 100.0
        chunk = chunk.rename(
            columns={"patientunitstayid": "ts_id", "respchartoffset": "minute"}
        )
        collected.append(_compact_events(chunk, "respiratoryCharting"))
    return pd.concat(collected, ignore_index=True) if collected else _empty_events()


def classify_intake_output(
    celllabel: pd.Series, cellpath: pd.Series
) -> pd.Series:
    """Map eICU I/O labels to the MIMIC-III/IV fluid vocabulary.

    The path is used to keep input and output concepts separate.  Specific
    products are assigned before generic categories so each source row maps to
    at most one model variable.
    """

    label = _normalise_text(celllabel)
    path_text = _normalise_text(cellpath)
    variable = pd.Series(pd.NA, index=label.index, dtype="string")
    is_input = path_text.str.contains(r"\|intake \(ml\)\|", regex=True, na=False)
    is_output = path_text.str.contains(r"\|output \(ml\)\|", regex=True, na=False)

    def assign(mask: pd.Series, name: str) -> None:
        variable.loc[variable.isna() & mask] = name

    # Outputs: exclude occurrence/count-only charting from volume features.
    is_count = label.str.contains(
        r"count|occurrence|number of|incontinence", regex=True, na=False
    )
    assign(
        is_output
        & label.str.contains(r"pre.?admission.*output", regex=True, na=False),
        "Pre-admission Output",
    )
    assign(
        label.str.contains(r"residual", regex=True, na=False) & ~is_count,
        "Residual",
    )
    assign(
        is_output
        & label.str.contains(r"emesis|vomit", regex=True, na=False)
        & ~is_count,
        "Emesis",
    )
    assign(
        is_output
        & label.str.contains(r"jackson.?pratt|\bjp\b", regex=True, na=False),
        "Jackson-Pratt",
    )
    assign(
        is_output & label.str.contains(r"chest tube", regex=True, na=False),
        "Chest Tube",
    )
    assign(
        is_output & label.str.contains(r"ultrafil", regex=True, na=False),
        "Ultrafiltrate",
    )
    assign(
        is_output
        & label.str.contains(r"estimated blood loss|\bblood loss\b|\bebl\b", regex=True, na=False),
        "EBL",
    )
    assign(
        is_output
        & label.str.contains(r"stool|fecal", regex=True, na=False)
        & ~label.str.contains(r"mixed urine", regex=True, na=False)
        & ~is_count,
        "Stool",
    )
    assign(
        is_output
        & label.str.contains(r"urine|urinary|foley|nephrostomy", regex=True, na=False)
        & ~label.str.contains(r"mixed.*stool", regex=True, na=False)
        & ~is_count,
        "Urine",
    )
    assign(
        is_output
        & label.str.contains(
            r"gastric|nasogastric|orogastric|\bng/og\b|\bng tube\b|\bog tube\b",
            regex=True,
            na=False,
        ),
        "Gastric",
    )

    # Inputs: actual recorded mL, not orders.  Dextrose mixtures are assigned
    # before their carrier fluids to avoid double representation.
    is_dextrose = label.str.contains(r"\bd5|dextrose", regex=True, na=False)
    is_dextrose_mixture = is_dextrose & label.str.contains(
        r"saline|nacl|sodium chloride|\bns\b|\bkcl\b|potassium|ringer|bicarb|d10|10 %",
        regex=True,
        na=False,
    )
    assign(is_input & is_dextrose_mixture, "Dextrose Other")
    assign(
        is_input
        & is_dextrose
        & ~is_dextrose_mixture
        & label.str.contains(r"\bd5w\b|dextrose 5", regex=True, na=False),
        "D5W",
    )
    assign(
        is_input
        & label.str.contains(r"lactated ringer|\blr\b", regex=True, na=False),
        "Lactated Ringers",
    )
    assign(
        is_input
        & label.str.contains(r"albumin", regex=True, na=False)
        & label.str.contains(r"25\s*%", regex=True, na=False),
        "Albumin 25%",
    )
    assign(
        is_input
        & label.str.contains(r"albumin", regex=True, na=False)
        & label.str.contains(r"5\s*%", regex=True, na=False),
        "Albumin 5%",
    )
    assign(
        is_input
        & label.str.contains(r"packed rbc|prbc", regex=True, na=False),
        "Packed RBC",
    )
    assign(
        is_input
        & label.str.contains(r"fresh frozen plasma|\bffp\b|transfuse plasma", regex=True, na=False),
        "Fresh Frozen Plasma",
    )
    assign(
        is_input & label.str.contains(r"\btpn\b|parenteral nutrition", regex=True, na=False),
        "TPN",
    )
    assign(
        is_input & label.str.contains(r"fiber|fibresource", regex=True, na=False),
        "Fiber",
    )
    assign(
        is_input
        & label.str.contains(r"tube.*flush|enteral.*flush|feeding.*flush|water bolus", regex=True, na=False),
        "GT Flush",
    )
    assign(
        is_input
        & label.str.contains(
            r"^po(?:\s|:|$)|^oral(?:\s|$)|oral intake|intake, oral|po fluids?|po liquids?",
            regex=True,
            na=False,
        )
        & ~label.str.contains(r"tube|gastric|orogastric", regex=True, na=False),
        "PO intake",
    )
    half_saline = label.str.contains(
        r"0\.45\s*%|\.45\s*%|half normal|1/2\s*(?:normal\s*)?(?:saline|ns|nss)",
        regex=True,
        na=False,
    )
    assign(is_input & half_saline & ~is_dextrose, "Half Normal Saline")
    normal_saline = label.str.contains(
        r"normal saline|0\.9\s*%\s*(?:sodium chloride|nacl|ns|saline)|sodium chloride.*0\.9\s*%",
        regex=True,
        na=False,
    )
    assign(
        is_input
        & normal_saline
        & ~half_saline
        & ~is_dextrose
        & ~label.str.contains(r"\bkcl\b|potassium|bicarb|hypertonic|3\s*%", regex=True, na=False),
        "Normal Saline",
    )
    assign(
        is_input & label.str.contains(r"sterile water", regex=True, na=False),
        "Sterile Water",
    )
    assign(
        is_input & label.str.contains(r"\bcolloids?\b", regex=True, na=False),
        "Colloid",
    )
    assign(
        is_input & label.str.contains(r"piggy.?back|\bivpb\b", regex=True, na=False),
        "Piggyback",
    )
    assign(
        is_input & label.str.contains(r"iv solution volume", regex=True, na=False),
        "Solution",
    )
    return variable


def extract_intake_output(
    path: Path,
    cohort_ids: set,
    observation_minutes: int,
    chunk_size: int,
    max_chunks: Optional[int],
) -> pd.DataFrame:
    columns = [
        "patientunitstayid",
        "intakeoutputoffset",
        "cellpath",
        "celllabel",
        "cellvaluenumeric",
    ]
    collected: List[pd.DataFrame] = []
    iterator = _read_chunks(path, columns, chunk_size, max_chunks)
    for chunk in tqdm(iterator, desc="intakeOutput"):
        chunk = _filter_window(
            chunk,
            "patientunitstayid",
            "intakeoutputoffset",
            cohort_ids,
            observation_minutes,
        )
        if chunk.empty:
            continue
        chunk["variable"] = classify_intake_output(
            chunk["celllabel"], chunk["cellpath"]
        )
        chunk = chunk[chunk["variable"].notna()].copy()
        if chunk.empty:
            continue
        chunk["value"] = _numeric(chunk["cellvaluenumeric"])
        chunk = chunk.rename(
            columns={"patientunitstayid": "ts_id", "intakeoutputoffset": "minute"}
        )
        collected.append(_compact_events(chunk, "intakeOutput"))
    return pd.concat(collected, ignore_index=True) if collected else _empty_events()


def extract_respiratory_care(
    path: Path,
    cohort_ids: set,
    observation_minutes: int,
    chunk_size: int,
    max_chunks: Optional[int],
) -> pd.DataFrame:
    columns = ["patientunitstayid", "respcarestatusoffset", "airwaytype"]
    collected: List[pd.DataFrame] = []
    iterator = _read_chunks(path, columns, chunk_size, max_chunks)
    invasive = {
        "oral ett",
        "nasal ett",
        "tracheostomy",
        "double-lumen tube",
        "cricothyrotomy",
    }
    for chunk in tqdm(iterator, desc="respiratoryCare"):
        chunk = _filter_window(
            chunk,
            "patientunitstayid",
            "respcarestatusoffset",
            cohort_ids,
            observation_minutes,
        )
        if chunk.empty:
            continue
        airway = _normalise_text(chunk["airwaytype"])
        chunk["value"] = np.nan
        chunk.loc[airway.isin(invasive), "value"] = 1.0
        chunk.loc[airway.eq("no artificial airway"), "value"] = 0.0
        chunk = chunk[chunk["value"].notna()].copy()
        if chunk.empty:
            continue
        chunk["variable"] = "Intubated"
        chunk = chunk.rename(
            columns={
                "patientunitstayid": "ts_id",
                "respcarestatusoffset": "minute",
            }
        )
        collected.append(_compact_events(chunk, "respiratoryCare"))
    return pd.concat(collected, ignore_index=True) if collected else _empty_events()


def _map_infusion_name(name: object) -> Optional[str]:
    if pd.isna(name):
        return None
    normalised = re.sub(r"\s+", " ", str(name).strip())
    for pattern, variable in INFUSION_SPECS:
        if pattern.fullmatch(normalised):
            return variable
    return None


def extract_infusions(
    path: Path,
    cohort_ids: set,
    observation_minutes: int,
    chunk_size: int,
    max_chunks: Optional[int],
) -> pd.DataFrame:
    columns = ["patientunitstayid", "infusionoffset", "drugname", "drugrate"]
    collected: List[pd.DataFrame] = []
    iterator = _read_chunks(path, columns, chunk_size, max_chunks)
    for chunk in tqdm(iterator, desc="infusionDrug"):
        chunk = _filter_window(
            chunk,
            "patientunitstayid",
            "infusionoffset",
            cohort_ids,
            observation_minutes,
        )
        if chunk.empty:
            continue
        chunk["variable"] = chunk["drugname"].map(_map_infusion_name)
        chunk = chunk[chunk["variable"].notna()].copy()
        if chunk.empty:
            continue
        chunk["value"] = _numeric(chunk["drugrate"])
        # MIMIC stores magnesium/calcium gluconate in grams.  Their selected
        # eICU rows explicitly report mg/hr, so convert to g/hr before the
        # eICU training-set standardisation is computed.
        convert_to_grams = chunk["variable"].isin(
            ["Magnesium Sulphate", "Calcium Gluconate"]
        )
        chunk.loc[convert_to_grams, "value"] = (
            chunk.loc[convert_to_grams, "value"] / 1000.0
        )
        chunk = chunk.rename(
            columns={"patientunitstayid": "ts_id", "infusionoffset": "minute"}
        )
        collected.append(_compact_events(chunk, "infusionDrug"))
    return pd.concat(collected, ignore_index=True) if collected else _empty_events()


def aggregate_events(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return _empty_events()

    def combine_sources(values: pd.Series) -> str:
        return ",".join(sorted(set(str(value) for value in values)))

    events["variable"] = events["variable"].astype("category")
    events = (
        events.groupby(["ts_id", "minute", "variable"], observed=True, sort=False)
        .agg(value=("value", "mean"), TABLE=("TABLE", combine_sources))
        .reset_index()
    )
    events["ts_id"] = events["ts_id"].astype(np.int64)
    events["minute"] = events["minute"].astype(np.int32)
    events["value"] = events["value"].astype(np.float32)
    events["variable"] = events["variable"].astype("category")
    events["TABLE"] = events["TABLE"].astype("category")
    return events.sort_values(["ts_id", "minute", "variable"]).reset_index(drop=True)


def patient_level_split(
    cohort: pd.DataFrame, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    subjects = cohort["patient_id"].drop_duplicates().to_numpy(copy=True)
    rng = np.random.RandomState(seed)
    rng.shuffle(subjects)
    count = len(subjects)
    first = int(0.64 * count)
    second = int(0.80 * count)
    subject_split = {
        patient_id: split
        for split, ids in (
            ("train", subjects[:first]),
            ("valid", subjects[first:second]),
            ("test", subjects[second:]),
        )
        for patient_id in ids
    }
    split_frame = cohort[
        ["ts_id", "patient_id", "in_hospital_mortality"]
    ].copy()
    split_frame["split"] = split_frame["patient_id"].map(subject_split)
    split_frame = split_frame.sort_values("ts_id").reset_index(drop=True)
    train_ids = split_frame.loc[split_frame["split"].eq("train"), "ts_id"].to_numpy(
        dtype=np.int64
    )
    valid_ids = split_frame.loc[split_frame["split"].eq("valid"), "ts_id"].to_numpy(
        dtype=np.int64
    )
    test_ids = split_frame.loc[split_frame["split"].eq("test"), "ts_id"].to_numpy(
        dtype=np.int64
    )
    return train_ids, valid_ids, test_ids, split_frame


def build_metadata(
    events: pd.DataFrame,
    train_ids: np.ndarray,
    reference_var2ind_path: Path,
    reference_meta_path: Path,
    preserve_reference_vocabulary: bool = False,
) -> Tuple[Dict[str, int], Dict[str, dict]]:
    reference_order: Dict[str, int] = {}
    reference_meta: Dict[str, dict] = {}
    if reference_var2ind_path.exists():
        with reference_var2ind_path.open("r", encoding="utf-8") as handle:
            reference_order = json.load(handle)
    if reference_meta_path.exists():
        with reference_meta_path.open("r", encoding="utf-8") as handle:
            reference_meta = json.load(handle)

    observed = set(events["variable"].astype(str).unique())
    model_variables = observed - DEMOGRAPHIC_VARIABLES
    if preserve_reference_vocabulary:
        if not reference_order or not reference_meta:
            raise FileNotFoundError(
                "--vocabulary-mode mimic-iv requires both reference JSON files"
            )
        unknown = model_variables - set(reference_order)
        if unknown:
            raise KeyError(
                "Observed variables missing from the reference vocabulary: "
                + ", ".join(sorted(unknown))
            )
        # Preserve both size and numeric IDs for checkpoint-compatible variable
        # embeddings.  JSON insertion order is not relied upon here.
        var2ind = dict(sorted(reference_order.items(), key=lambda item: item[1]))
        metadata_variables = set(var2ind) | DEMOGRAPHIC_VARIABLES
    else:
        ordered = sorted(
            model_variables,
            key=lambda name: (reference_order.get(name, 10**9), name),
        )
        var2ind = {name: index for index, name in enumerate(ordered)}
        metadata_variables = observed

    train = events[events["ts_id"].isin(set(train_ids.tolist()))]
    statistics = train.groupby("variable", observed=True)["value"].agg(["mean", "std"])
    metadata: Dict[str, dict] = {}
    for variable in sorted(metadata_variables):
        if variable in statistics.index:
            mean = float(statistics.loc[variable, "mean"])
            std = float(statistics.loc[variable, "std"])
        elif preserve_reference_vocabulary and variable in reference_meta:
            # No eICU training observation exists.  Keeping the MIMIC-IV
            # value preserves a usable, leakage-free metadata row; no eICU
            # event is synthesised.
            mean = float(reference_meta[variable]["mean"])
            std = float(reference_meta[variable]["std"])
        else:
            raise ValueError(f"No training observations available for variable {variable!r}")
        if not np.isfinite(std) or std == 0:
            std = 1.0
        if variable in reference_meta:
            class_name = reference_meta[variable]["class"]
            subclass_name = reference_meta[variable]["sub_class"]
        elif variable in FALLBACK_SEMANTICS:
            class_name, subclass_name = FALLBACK_SEMANTICS[variable]
        else:
            raise KeyError(
                f"Variable {variable!r} has no semantic class in {reference_meta_path}"
            )
        metadata[variable] = {
            "mean": mean,
            "std": std,
            "class": class_name,
            "sub_class": subclass_name,
        }
    return var2ind, metadata


def validate_contract(
    events: pd.DataFrame,
    outcomes: pd.DataFrame,
    split_frame: pd.DataFrame,
    train_ids: np.ndarray,
    valid_ids: np.ndarray,
    test_ids: np.ndarray,
    var2ind: Mapping[str, int],
    metadata: Mapping[str, dict],
    observation_minutes: int,
) -> None:
    if list(events.columns) != EVENT_COLUMNS:
        raise AssertionError(f"Unexpected event schema: {list(events.columns)}")
    if list(outcomes.columns) != ["ts_id", "in_hospital_mortality"]:
        raise AssertionError(f"Unexpected outcome schema: {list(outcomes.columns)}")
    if events.empty or outcomes.empty:
        raise AssertionError("The final cohort is empty")
    if events[["ts_id", "minute", "variable", "value"]].isna().any().any():
        raise AssertionError("Events contain missing required values")
    if not np.isfinite(events["value"].to_numpy()).all():
        raise AssertionError("Events contain non-finite values")
    if not events["minute"].between(0, observation_minutes - 1).all():
        raise AssertionError("Events fall outside the observation window")
    if outcomes["ts_id"].duplicated().any():
        raise AssertionError("Outcomes contain duplicate ICU stays")
    if not set(outcomes["in_hospital_mortality"].unique()).issubset({0, 1}):
        raise AssertionError("Mortality target is not binary")

    expected = set(outcomes["ts_id"].tolist())
    if set(events["ts_id"].unique()) != expected:
        raise AssertionError("Event and outcome stay IDs differ")
    split_sets = [set(train_ids.tolist()), set(valid_ids.tolist()), set(test_ids.tolist())]
    if split_sets[0] & split_sets[1] or split_sets[0] & split_sets[2] or split_sets[1] & split_sets[2]:
        raise AssertionError("ICU-stay split overlap detected")
    if set.union(*split_sets) != expected:
        raise AssertionError("Splits do not cover the final cohort")
    subject_sets = [
        set(split_frame.loc[split_frame["split"].eq(name), "patient_id"].tolist())
        for name in ("train", "valid", "test")
    ]
    if subject_sets[0] & subject_sets[1] or subject_sets[0] & subject_sets[2] or subject_sets[1] & subject_sets[2]:
        raise AssertionError("Patient leakage across splits detected")

    counts = events[events["variable"].isin(DEMOGRAPHIC_VARIABLES)].groupby(
        ["ts_id", "variable"], observed=True
    ).size()
    for stay_id in expected:
        for variable in DEMOGRAPHIC_VARIABLES:
            if counts.get((stay_id, variable), 0) != 1:
                raise AssertionError(
                    f"Stay {stay_id} does not have exactly one {variable} event"
                )
    model_variables = set(events["variable"].astype(str)) - DEMOGRAPHIC_VARIABLES
    if not model_variables.issubset(set(var2ind)):
        raise AssertionError("var2ind does not cover every non-demographic variable")
    if not set(events["variable"].astype(str)).issubset(set(metadata)):
        raise AssertionError("var_mean_std does not cover every event variable")


def build_audit(
    events: pd.DataFrame,
    outcomes: pd.DataFrame,
    split_frame: pd.DataFrame,
    candidate_count: int,
    observation_minutes: int,
    periodic_bin_minutes: int,
    seed: int,
    heldout_only_variables: Sequence[str],
    vocabulary_mode: str,
    vocabulary_size: int,
) -> dict:
    splits = {}
    for name in ("train", "valid", "test"):
        rows = split_frame[split_frame["split"].eq(name)]
        splits[name] = {
            "patients": int(rows["patient_id"].nunique()),
            "icu_stays": int(len(rows)),
            "deaths": int(rows["in_hospital_mortality"].sum()),
            "mortality_rate": float(rows["in_hospital_mortality"].mean()),
        }
    variable_counts = {
        str(name): int(count)
        for name, count in events["variable"].value_counts().sort_index().items()
    }
    source_counts = {
        str(name): int(count)
        for name, count in events["TABLE"].astype(str).value_counts().sort_index().items()
    }
    return {
        "protocol": {
            "prediction_target": "in_hospital_mortality",
            "observation_window_minutes": observation_minutes,
            "minimum_icu_los_minutes": observation_minutes,
            "adult_age_minimum": 18,
            "age_over_89_encoded_as": 90,
            "split_unit": "uniquepid",
            "split_ratio": {"train": 0.64, "valid": 0.16, "test": 0.20},
            "random_seed": seed,
            "vital_periodic_bin_minutes": periodic_bin_minutes,
            "normalization_statistics": (
                "eICU training split; MIMIC-IV reference fallback when a "
                "vocabulary variable has no eICU training observation"
            ),
            "vocabulary_mode": vocabulary_mode,
        },
        "candidate_icu_stays_before_event_filter": candidate_count,
        "final_patients": int(split_frame["patient_id"].nunique()),
        "final_icu_stays": int(len(outcomes)),
        "deaths": int(outcomes["in_hospital_mortality"].sum()),
        "mortality_rate": float(outcomes["in_hospital_mortality"].mean()),
        "event_rows": int(len(events)),
        "variables": int(events["variable"].nunique()),
        "var2ind_size": vocabulary_size,
        "variables_absent_from_eicu_training": list(heldout_only_variables),
        "absent_training_variable_policy": (
            "MIMIC-IV reference statistics; events retained"
            if vocabulary_mode == "mimic-iv"
            else "variable events dropped"
        ),
        "splits": splits,
        "event_count_by_variable": variable_counts,
        "event_count_by_source": source_counts,
        "checks": {
            "patient_overlap_across_splits": 0,
            "stay_overlap_across_splits": 0,
            "events_outside_observation_window": 0,
            "missing_event_values": 0,
            "duplicate_outcome_rows": 0,
        },
    }


def write_mapping_report(path: Path) -> None:
    rows = []
    for source, target in VITAL_PERIODIC_MAP.items():
        rows.append(("vitalPeriodic", source, target, "hourly mean by default"))
    for source, target in VITAL_APERIODIC_MAP.items():
        rows.append(("vitalAperiodic", source, target, "exact minute"))
    for source, target in LAB_MAP.items():
        rows.append(("lab", source, target, "exact labname mapping"))
    rows.extend(
        [
            ("nurseCharting", "Eyes/Motor/Verbal", "GCS_*", "explicit GCS component only"),
            ("nurseCharting", "Temperature (C/F)", "Temperature", "F converted to C"),
            ("nurseCharting", "O2 Saturation", "O2 Saturation", "numeric values"),
            ("respiratoryCharting", "FiO2", "FiO2", "percent converted to fraction"),
            (
                "intakeOutput",
                "I/O label and input/output path",
                ", ".join(sorted(INTAKE_OUTPUT_VARIABLES)),
                "actual mL rows; occurrence/count rows excluded",
            ),
            (
                "respiratoryCare",
                "airwaytype",
                "Intubated",
                "invasive airway=1; explicit no artificial airway=0",
            ),
            ("infusionDrug", "unit-specific drugrate", "mapped medication", "ml/hr excluded"),
        ]
    )
    pd.DataFrame(rows, columns=["source_table", "source_field", "variable", "rule"]).to_csv(
        path, index=False
    )


def _atomic_pickle(value: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def _atomic_json(value: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def preprocess_eicu(args: argparse.Namespace) -> None:
    raw = Path(args.raw_data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pickle = output_dir / args.output_name
    produced = [
        output_pickle,
        output_dir / "var2ind.json",
        output_dir / "var_mean_std.json",
        output_dir / "cohort_summary.json",
        output_dir / "patient_split.csv",
        output_dir / "feature_mapping.csv",
    ]
    existing = [str(path) for path in produced if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Output files already exist. Use --overwrite to replace them:\n"
            + "\n".join(existing)
        )

    patient_path = raw / "patient.csv.gz"
    if not patient_path.exists():
        raise FileNotFoundError(f"eICU patient table not found: {patient_path}")

    print("Step 1/8: building the adult 24-hour candidate cohort")
    cohort = build_candidate_cohort(patient_path, args.observation_minutes)
    candidate_count = len(cohort)
    cohort_ids = set(cohort["ts_id"].tolist())
    print(f"Candidate ICU stays: {candidate_count:,}")

    print("Step 2/8: extracting periodic and aperiodic vital signs")
    event_frames = [
        extract_wide_vitals(
            raw / "vitalPeriodic.csv.gz",
            VITAL_PERIODIC_MAP,
            "patientunitstayid",
            "observationoffset",
            "vitalPeriodic",
            cohort_ids,
            args.observation_minutes,
            args.periodic_bin_minutes,
            args.chunk_size,
            args.max_chunks_per_table,
        ),
        extract_wide_vitals(
            raw / "vitalAperiodic.csv.gz",
            VITAL_APERIODIC_MAP,
            "patientunitstayid",
            "observationoffset",
            "vitalAperiodic",
            cohort_ids,
            args.observation_minutes,
            1,
            args.chunk_size,
            args.max_chunks_per_table,
        ),
    ]

    print("Step 3/8: extracting laboratory measurements")
    event_frames.append(
        extract_labs(
            raw / "lab.csv.gz",
            cohort_ids,
            args.observation_minutes,
            args.chunk_size,
            args.max_chunks_per_table,
        )
    )

    print("Step 4/8: extracting nursing, respiratory, urine, and infusion events")
    event_frames.extend(
        [
            extract_nurse_charting(
                raw / "nurseCharting.csv.gz",
                cohort_ids,
                args.observation_minutes,
                args.chunk_size,
                args.max_chunks_per_table,
            ),
            extract_respiratory_charting(
                raw / "respiratoryCharting.csv.gz",
                cohort_ids,
                args.observation_minutes,
                args.chunk_size,
                args.max_chunks_per_table,
            ),
            extract_respiratory_care(
                raw / "respiratoryCare.csv.gz",
                cohort_ids,
                args.observation_minutes,
                args.chunk_size,
                args.max_chunks_per_table,
            ),
            extract_intake_output(
                raw / "intakeOutput.csv.gz",
                cohort_ids,
                args.observation_minutes,
                args.chunk_size,
                args.max_chunks_per_table,
            ),
            extract_infusions(
                raw / "infusionDrug.csv.gz",
                cohort_ids,
                args.observation_minutes,
                args.chunk_size,
                args.max_chunks_per_table,
            ),
        ]
    )

    print("Step 5/8: removing stays without dynamic first-window observations")
    dynamic = pd.concat(event_frames, ignore_index=True)
    del event_frames
    observed_ids = set(dynamic["ts_id"].unique().tolist())
    cohort = cohort[cohort["ts_id"].isin(observed_ids)].copy()
    if cohort.empty:
        raise RuntimeError("No eligible ICU stays have events in the observation window")
    final_ids = set(cohort["ts_id"].tolist())
    dynamic = dynamic[dynamic["ts_id"].isin(final_ids)]
    static = extract_static_events(cohort)
    events = aggregate_events(pd.concat([static, dynamic], ignore_index=True))
    del static, dynamic
    print(f"Final cohort: {len(cohort):,} ICU stays; {len(events):,} event rows")

    print("Step 6/8: creating leakage-safe patient-level 64/16/20 splits")
    train_ids, valid_ids, test_ids, split_frame = patient_level_split(cohort, args.seed)
    outcomes = (
        cohort[["ts_id", "in_hospital_mortality"]]
        .sort_values("ts_id")
        .reset_index(drop=True)
    )

    # The feature vocabulary is defined from training data only.  A feature
    # occurring exclusively in validation/test cannot have an eICU training
    # mean/std; dropping such a very rare feature avoids either leakage or a
    # silent fallback to test-distribution statistics.
    train_variables = set(
        events.loc[events["ts_id"].isin(train_ids), "variable"].astype(str).unique()
    )
    all_variables = set(events["variable"].astype(str).unique())
    heldout_only_variables = sorted(all_variables - train_variables)
    if heldout_only_variables and args.vocabulary_mode == "observed":
        print(
            "Dropping variables absent from the training split: "
            + ", ".join(heldout_only_variables)
        )
        events = events[
            ~events["variable"].astype(str).isin(heldout_only_variables)
        ].copy()
        events["variable"] = events["variable"].cat.remove_unused_categories()
    elif heldout_only_variables:
        print(
            "Using MIMIC-IV reference statistics for variables absent from "
            "the eICU training split: " + ", ".join(heldout_only_variables)
        )

    print("Step 7/8: computing training-only normalization metadata and validating")
    var2ind, metadata = build_metadata(
        events,
        train_ids,
        Path(args.reference_var2ind),
        Path(args.reference_meta),
        preserve_reference_vocabulary=args.vocabulary_mode == "mimic-iv",
    )
    validate_contract(
        events,
        outcomes,
        split_frame,
        train_ids,
        valid_ids,
        test_ids,
        var2ind,
        metadata,
        args.observation_minutes,
    )
    audit = build_audit(
        events,
        outcomes,
        split_frame,
        candidate_count,
        args.observation_minutes,
        args.periodic_bin_minutes,
        args.seed,
        heldout_only_variables,
        args.vocabulary_mode,
        len(var2ind),
    )

    print("Step 8/8: writing model-ready outputs")
    _atomic_pickle(
        [events, outcomes, train_ids, valid_ids, test_ids], output_pickle
    )
    _atomic_json(var2ind, output_dir / "var2ind.json")
    _atomic_json(metadata, output_dir / "var_mean_std.json")
    _atomic_json(audit, output_dir / "cohort_summary.json")
    split_frame.to_csv(output_dir / "patient_split.csv", index=False)
    write_mapping_report(output_dir / "feature_mapping.csv")
    print(f"Done. Main dataset: {output_pickle}")
    print(json.dumps(audit["splits"], indent=2, ensure_ascii=False))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess eICU-CRD into the MIMIC-IV five-object pickle contract."
    )
    parser.add_argument("--raw-data-path", type=Path, default=DEFAULT_RAW_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", default="eicu.pkl")
    parser.add_argument("--reference-var2ind", type=Path, default=DEFAULT_REFERENCE_VAR2IND)
    parser.add_argument("--reference-meta", type=Path, default=DEFAULT_REFERENCE_META)
    parser.add_argument(
        "--vocabulary-mode",
        choices=("mimic-iv", "observed"),
        default="mimic-iv",
        help=(
            "mimic-iv preserves the full 124-variable IDs for checkpoint "
            "compatibility; observed creates a compact eICU-only vocabulary."
        ),
    )
    parser.add_argument("--observation-minutes", type=int, default=24 * 60)
    parser.add_argument(
        "--periodic-bin-minutes",
        type=int,
        default=60,
        help=(
            "Aggregate dense vitalPeriodic rows to this many minutes (default: 60). "
            "Use 1 to preserve every recorded minute."
        ),
    )
    parser.add_argument("--chunk-size", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--max-chunks-per-table",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    if args.observation_minutes <= 0:
        parser.error("--observation-minutes must be positive")
    if args.periodic_bin_minutes <= 0:
        parser.error("--periodic-bin-minutes must be positive")
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")
    return args


if __name__ == "__main__":
    preprocess_eicu(parse_args())
