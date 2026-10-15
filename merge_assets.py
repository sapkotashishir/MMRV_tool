"""
Merge all per-farmer GEE assets into ONE FeatureCollection with a unique field_id.

Run once (VS Code terminal or Colab):   python merge_assets.py
Before running: edit PROJECT and ASSET_OUT below.

Output asset properties:
  field_id : anonymous, unique  e.g. F07_Pasture 3   (use this in the public app)
  farm     : farmer/asset name  (admin use only - keep out of the public app)
  plot     : original 'Name' value
Also writes fields_lookup.csv locally (field_id <-> farm <-> plot) for your records.
"""
import csv
import ee

PROJECT = "shishirbeeville"                         # <- change
ASSET_OUT = "users/Forage_precision/texas_fields"    # <- new merged asset (must not exist yet)
PLOT_PROP = "Name"
INCLUDE_RESEARCH_SITES = False   # Beeville / OK / Driscoll are research sites, not farmer fields

RESEARCH = {
    "beeville_paddocks": "users/Forage_precision/Beeville_paddocks",
    "ok_top": "users/Forage_precision/OK_top_9x9_grid",
    "ok_bottom": "users/Forage_precision/OK_bottom_9x9_grid",
    "Driscoll": "users/Forage_precision/driscoll_grids_2020",
}

FARMERS = {
    "Grant_Wilde": "Grant-Wilde", "Michael-Thompson_1": "Michael-Thompson",
}
BASE = "users/Forage_precision/"

ee.Initialize(project=PROJECT)

sources = {k: BASE + v for k, v in FARMERS.items()}
if INCLUDE_RESEARCH_SITES:
    sources.update(RESEARCH)


def tag(fc, farm, code):
    """Add field_id / farm / plot to every feature; keep only those properties."""
    def f(feat):
        feat = ee.Feature(feat)
        plot = ee.String(ee.Algorithms.If(
            feat.propertyNames().contains(PLOT_PROP),
            ee.String(feat.get(PLOT_PROP)),          # works for numbers and text
            ee.String(feat.get("system:index"))))
        return (feat.set({"farm": farm, "plot": plot,
                          "field_id": ee.String(code).cat("_").cat(plot)})
                .select(["field_id", "farm", "plot"]))
    return fc.map(f)


parts = []
for i, (farm, asset) in enumerate(sorted(sources.items()), start=1):
    parts.append(tag(ee.FeatureCollection(asset), farm, f"F{i:02d}"))

merged = ee.FeatureCollection(parts).flatten()

# ---- sanity checks (client side; ~300 rows so this is fast) ----
rows = [f["properties"] for f in
        merged.select(["field_id", "farm", "plot"], retainGeometry=False)
        .getInfo()["features"]]
ids = [r["field_id"] for r in rows]
dups = sorted({x for x in ids if ids.count(x) > 1})
print(f"{len(rows)} fields from {len(sources)} assets")
if dups:
    print("WARNING duplicate field_ids (fix plot names or add a suffix):", dups[:20])
    raise SystemExit("Fix duplicates, then re-run.")

with open("fields_lookup.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["field_id", "farm", "plot"])
    w.writeheader()
    w.writerows(sorted(rows, key=lambda r: r["field_id"]))
print("Wrote fields_lookup.csv")

task = ee.batch.Export.table.toAsset(
    collection=merged, description="merge_texas_fields", assetId=ASSET_OUT)
task.start()
print("Export started. Watch progress in the Code Editor 'Tasks' tab,"
      " or run: earthengine task list")