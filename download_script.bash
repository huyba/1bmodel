# Download a partial slice of FineWeb-Edu — enough to comfortably clear the
# 25B-token training target (see design.md), without pulling the whole dataset.
#
# Why not the full sample/100BT tier (or the whole repo)?
#   - sample/10BT  =  28.5GB  -> only ~10B tokens, undershoots the 25B target
#   - sample/100BT = 286.4GB  -> way more than needed, and won't fit alongside
#                                the 50GB packed .bin output that has to exist
#                                on disk at the same time during packing
#     (checked directly against the Hub, not estimated)
#
# Why groups 000-003 specifically:
#   sample/100BT is 140 parquet files across 14 groups of 10 (~21.5GB/group).
#   Taking the first 4 groups (40 files, ~86GB) gives ~28.6B raw tokens
#   (~14% headroom over 25B) while keeping parquet + packed-output peak disk
#   usage manageable. Delete ./fineweb_raw after packing to reclaim the ~86GB.
#
# --local-dir must stay ./fineweb_raw: prepare_data_local.py and
# prepare_data_multiprocess.py both hardcode that as their input directory.
hf download HuggingFaceFW/fineweb-edu \
  --repo-type dataset \
  --include "sample/100BT/00[0-3]_*.parquet" \
  --local-dir ./fineweb_raw
