
import argparse
import json
import time
from typing import List

import pandas as pd
import requests


FEATURE_COLS = [
    "has_appointment",
    "has_observation",
    "has_encounter",
    "has_btg_access",
    "has_care_access",
    "num_btg_events",
    "num_care_events",
    "avg_time_between_events",
]

def coerce_features(df: pd.DataFrame) -> pd.DataFrame:
    
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Mangler kolonner i CSV: {missing}")

    for col in ["has_appointment", "has_observation", "has_encounter", "has_btg_access", "has_care_access"]:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)
        elif df[col].dtype == object:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({"true": 1, "false": 0, "1": 1, "0": 0})
            ).fillna(0).astype(int)
        else:
            df[col] = df[col].fillna(0).astype(int)

    df["num_btg_events"] = df["num_btg_events"].fillna(0).astype(int)
    df["num_care_events"] = df["num_care_events"].fillna(0).astype(int)
    df["avg_time_between_events"] = df["avg_time_between_events"].fillna(0.0).astype(float)
    return df[FEATURE_COLS]


def df_to_payload(df: pd.DataFrame) -> List[dict]:
    
    return df[FEATURE_COLS].to_dict(orient="records")


def main():
    ap = argparse.ArgumentParser(description="Stream CSV rows to FastAPI /predict for real-time demo")
    ap.add_argument("--csv", required=True, help="Sti til CSV med events")
    ap.add_argument("--api", default="http://127.0.0.1:8000", help="Base-URL til API (default: %(default)s)")
    ap.add_argument("--batch", type=int, default=128, help="Batch-størrelse per request (default: %(default)s)")
    ap.add_argument("--sleep", type=float, default=1.0, help="Pause mellem batches i sek. (default: %(default)s)")
    ap.add_argument("--threshold", type=float, default=0.5, help="Klassifikations-threshold (default: %(default)s)")
    ap.add_argument("--limit", type=int, default=0, help="Max rækker at sende (0 = alle)")
    args = ap.parse_args()

    
    total = 0
    anoms = 0
    batch_ix = 0

    for chunk in pd.read_csv(args.csv, chunksize=args.batch):
        if args.limit and total >= args.limit:
            break

        
        if args.limit:
            remaining = args.limit - total
            if remaining < len(chunk):
                chunk = chunk.iloc[:remaining, :]

        try:
            X = coerce_features(chunk.copy())
            payload = df_to_payload(X)
        except Exception as e:
            print(f"[skip] Kan ikke forberede batch: {e}")
            continue

        try:
            r = requests.post(
                f"{args.api}/predict",
                params={"threshold": args.threshold},
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=30,
            )
            r.raise_for_status()
            out = r.json()
        except Exception as e:
            print(f"[error] Request fejlede: {e}")
            time.sleep(args.sleep)
            continue

      
        batch_ix += 1
        total += out.get("n", 0)
        hits = [
            i for i in out.get("results", [])
            if i.get("pred_label") == "Anomaly"
        ]
        anoms += len(hits)

        
        print(f"batch={len(payload)} anomalies={len(hits):02d} "
              f"progress={total} threshold={args.threshold}")

        
        if hits:
           
            scored = [h for h in hits if "pred_prob_anomaly" in h]
            scored.sort(key=lambda x: x.get("pred_prob_anomaly", 0), reverse=True)
            top = scored[:3]
            for i, t in enumerate(top, 1):
                p = t.get("pred_prob_anomaly", 0.0)
                print(f"  [top#{i}] label={t['pred_label']} prob={p:.6f}")

        time.sleep(args.sleep)

    print(f"\nDONE  rows={total} anomalies={anoms} share={ (anoms/total if total else 0):.4f}")
    

if __name__ == "__main__":
    main()
