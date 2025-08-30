import json, math

path = "csv_ingest_results.log"
total = 0
anom = 0
top = []

with open(path) as f:
    for line in f:
        total += 1
        rec = json.loads(line)
        p = float(rec.get("pred_prob_anomaly", 0.0))
        lbl = rec.get("pred_label")
        if lbl == "Anomaly":
            anom += 1
            top.append(p)

top_sorted = sorted(top, reverse=True)[:10]
share = (anom/total) if total else 0.0

print(f"total={total}")
print(f"anomalies={anom}")
print(f"share_anomaly={share:.4f}")
print("top10_probs=" + ",".join(f"{x:.6f}" for x in top_sorted))
