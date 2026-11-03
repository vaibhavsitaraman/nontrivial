import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src, so nets/envs/paths import

import json
import os
import time
import numpy as np
import pandas as pd
import requests

from paths import DATA

FACILITY_DATASET_ID = "anag-cw7u"
BASE_URL = f"https://healthdata.gov/resource/{FACILITY_DATASET_ID}.csv"

STATE_FILTER = "NJ"
DATE_START = "2021-07-01T00:00:00"
DATE_END = "2022-05-01T00:00:00"

# try these column names in order, the real schema is a bit slippery
FIELD_CANDIDATES = {
    "facility_name": ["hospital_name", "facility_name", "hospital_pk"],
    "city": ["city", "hospital_city"],
    "date": ["collection_week", "date", "week_ending"],
    "inpatient_beds": ["inpatient_beds_7_day_avg", "inpatient_beds", "total_beds_7_day_avg"],
    "inpatient_beds_used": ["inpatient_beds_used_7_day_avg", "inpatient_beds_used"],
    "inpatient_beds_used_covid": ["inpatient_bed_covid_utilization", "inpatient_beds_used_covid_7_day_avg",
                                    "total_adult_patients_hospitalized_confirmed_covid_7_day_avg"],
    "adult_icu_beds": ["staffed_adult_icu_bed_occupancy_7_day_avg", "total_staffed_adult_icu_beds_7_day_avg",
                         "staffed_icu_adult_patients_confirmed_covid_7_day_avg"],
    "adult_icu_beds_used_covid": ["staffed_icu_adult_patients_confirmed_covid_7_day_avg",
                                    "icu_beds_used_covid_7_day_avg"],
    "pediatric_covid": ["total_pediatric_patients_hospitalized_confirmed_covid_7_day_avg",
                          "previous_day_admission_pediatric_covid_confirmed_7_day_sum"],
}

NORTH_NJ_CITIES = ["newark", "jersey city", "paterson", "hackensack", "morristown",
                    "livingston", "montclair", "bloomfield", "elizabeth", "clifton"]
SOUTH_NJ_CITIES = ["camden", "cherry hill", "vineland", "voorhees", "willingboro",
                    "mount holly", "sewell", "washington township", "glassboro", "millville"]


def find_field(columns, candidates, required_name):
    for c in candidates:
        if c in columns:
            return c
    print(f"  WARNING: none of {candidates} found for '{required_name}'. columns: {list(columns)}")
    return None


def main():
    print("grabbing a tiny sample so we can see the real column names")
    sample_resp = requests.get(BASE_URL, params={"$where": f"state='{STATE_FILTER}'", "$limit": 5})
    sample_resp.raise_for_status()
    schema_path = os.path.join(DATA, "schema.csv")
    with open(schema_path, "wb") as f:
        f.write(sample_resp.content)
    sample_df = pd.read_csv(schema_path)
    print(f"columns ({len(sample_df.columns)}):")
    print(list(sample_df.columns))
    if "hospital_name" in sample_df.columns:
        cols = ["hospital_name"] + (["city"] if "city" in sample_df.columns else [])
        print(sample_df[cols].to_string())

    columns = set(sample_df.columns)
    resolved = {key: find_field(columns, candidates, key) for key, candidates in FIELD_CANDIDATES.items()}
    print(f"resolved: {resolved}")
    missing = [k for k, v in resolved.items() if v is None]
    if missing:
        print(f"missing {missing}. tweak FIELD_CANDIDATES and rerun. continuing with what we have.")

    print(f"\nfetching nj rows from {DATE_START} to {DATE_END}")
    select_fields = [v for v in resolved.values() if v is not None]
    select_fields = list(dict.fromkeys(select_fields))

    date_field = resolved["date"]
    where_clause = f"state='{STATE_FILTER}'"
    if date_field:
        where_clause += f" AND {date_field} between '{DATE_START}' and '{DATE_END}'"

    params = {
        "$select": ",".join(select_fields + ["state"]),
        "$where": where_clause,
        "$limit": 50000,
    }
    t0 = time.time()
    resp = requests.get(BASE_URL, params=params)
    resp.raise_for_status()
    raw_path = os.path.join(DATA, "nj_raw.csv")
    with open(raw_path, "wb") as f:
        f.write(resp.content)
    print(f"fetched in {time.time()-t0:.1f}s -> {raw_path}")

    df = pd.read_csv(raw_path)
    n_fac = df[resolved["facility_name"]].nunique() if resolved["facility_name"] else "?"
    print(f"{len(df)} rows, {n_fac} facilities")

    city_col = resolved.get("city")
    if city_col is None or city_col not in df.columns:
        raise RuntimeError("no city column, can't split north vs south")

    # substring match cities into hub (north) and spoke (south)
    df["_city_lower"] = df[city_col].astype(str).str.lower()
    df["_region"] = None
    df.loc[df["_city_lower"].apply(lambda c: any(n in c for n in NORTH_NJ_CITIES)), "_region"] = "H"
    df.loc[df["_city_lower"].apply(lambda c: any(s in c for s in SOUTH_NJ_CITIES)), "_region"] = "S"

    n_matched = df["_region"].notna().sum()
    print(f"matched {n_matched}/{len(df)} rows")
    print(f"  north: {(df['_region']=='H').sum()}  south: {(df['_region']=='S').sum()}")
    unmatched_cities = df.loc[df["_region"].isna(), "_city_lower"].unique()
    print("unmatched cities:", sorted(unmatched_cities)[:30])

    df = df[df["_region"].notna()].copy()
    if len(df) == 0:
        raise RuntimeError("nothing matched. extend the city lists.")

    numeric_fields = [resolved[k] for k in ["inpatient_beds", "inpatient_beds_used", "inpatient_beds_used_covid",
                                              "adult_icu_beds", "adult_icu_beds_used_covid", "pediatric_covid"]
                       if resolved[k] is not None and resolved[k] in df.columns]
    for col in numeric_fields:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # one number per week per region
    agg = df.groupby([date_field, "_region"])[numeric_fields].mean().reset_index()
    pivot = agg.pivot(index=date_field, columns="_region", values=numeric_fields)
    pivot = pivot.sort_index().dropna()
    print(f"{len(pivot)} aligned weeks")

    calibration = {"n_weeks": len(pivot), "date_range": [DATE_START, DATE_END],
                   "resolved_fields": resolved, "north_nj_cities": NORTH_NJ_CITIES,
                   "south_nj_cities": SOUTH_NJ_CITIES}

    inp_used = resolved.get("inpatient_beds_used_covid")
    inp_beds = resolved.get("inpatient_beds")
    if inp_used and inp_beds:
        for region in ["H", "S"]:
            util = pivot[(inp_used, region)] / pivot[(inp_beds, region)] * 100
            calibration[f"inpatient_covid_util_pct_{region}"] = dict(
                mean=float(util.mean()), std=float(util.std()),
                min=float(util.min()), max=float(util.max()),
                quartiles=[float(q) for q in util.quantile([0.25, 0.5, 0.75]).values],
            )
        util_H = pivot[(inp_used, "H")] / pivot[(inp_beds, "H")]
        util_S = pivot[(inp_used, "S")] / pivot[(inp_beds, "S")]
        calibration["cross_region_correlation_inpatient"] = float(np.corrcoef(util_H, util_S)[0, 1])

        lag_corrs = {}
        for lag in range(0, min(6, len(util_H) - 1)):
            if lag == 0:
                lag_corrs[str(lag)] = float(np.corrcoef(util_H, util_S)[0, 1])
            else:
                lag_corrs[str(lag)] = float(np.corrcoef(util_H[:-lag], util_S[lag:])[0, 1])
        calibration["lagged_cross_correlation_H_leads_S"] = lag_corrs

    icu_used = resolved.get("adult_icu_beds_used_covid")
    icu_beds = resolved.get("adult_icu_beds")
    if icu_used and icu_beds and icu_beds in pivot.columns.get_level_values(0):
        for region in ["H", "S"]:
            util = pivot[(icu_used, region)] / pivot[(icu_beds, region)] * 100
            calibration[f"adult_icu_covid_util_pct_{region}"] = dict(
                mean=float(util.mean()), std=float(util.std()))

    ped_field = resolved.get("pediatric_covid")
    if ped_field:
        for region in ["H", "S"]:
            series = pivot[(ped_field, region)]
            calibration[f"pediatric_covid_count_{region}"] = dict(
                mean=float(series.mean()), std=float(series.std()))
        ped_H = pivot[(ped_field, "H")]
        ped_S = pivot[(ped_field, "S")]
        calibration["cross_region_correlation_pediatric"] = float(np.corrcoef(ped_H, ped_S)[0, 1])

    out = os.path.join(DATA, "nj_calib.json")
    with open(out, "w") as f:
        json.dump(calibration, f, indent=2)
    print(f"saved {out}")
    print(json.dumps(calibration, indent=2))


if __name__ == "__main__":
    main()
