# Cross-Seed Baseline Study - Quick Reference

## TL;DR

**Question:** Does structured communication (INTENT/REQUEST) provide a generalizable advantage over freeform (natural language) across different agent spawn positions?

**Answer:** No. Freeform wins (62.7%), and "none" (57.3%) performs nearly as well as structured (56.0%).

---

## Key Files

📊 **Results & Analysis:**
- [`FINAL_RESULTS.md`](./FINAL_RESULTS.md) - Executive summary and key findings
- [`README.md`](./README.md) - Complete experiment documentation

📈 **Visualizations:**
- [`plots/`](./plots/) - 5 publication-quality plots with tiktoken token counting
  - Success rate by strategy
  - Communication volume (messages + tokens)
  - Collision rates
  - Completion timeline
  - Success vs token cost

📁 **Data:**
- [`run_inventory.json`](./run_inventory.json) - All 45 runs with UTC timestamps
- [`aggregate_stats.json`](./aggregate_stats.json) - Computed statistics by strategy

🔧 **Reproducibility:**
- [`generate_plots_45runs.py`](./generate_plots_45runs.py) - Regenerate all plots
- `runs/` - 45 complete run directories with metrics, transcripts, episodes

---

## Final Results (45 runs)

| Strategy | Success Rate | Messages/Run | Agents Finished |
|----------|-------------|--------------|-----------------|
| **Freeform** | **62.7%** | 5.9 ± 9.1 | 47/75 |
| None | 57.3% | 0.0 | 43/75 |
| Structured | 56.0% | 8.9 ± 6.9 | 42/75 |

**Conclusion:** Freeform communication generalizes better. The structured INTENT/REQUEST protocol provides minimal benefit over no communication at all.

---

## How to Regenerate Plots

```bash
cd experiments/cross_seed_baseline_20251112T143355Z
python3 generate_plots_45runs.py
# Output: plots/*.png (5 files)
```

---

## Experiment Details

- **Model:** azure:gpt-5-mini
- **Maze:** long_corridor (30×10, obstacle seed 606, 20% extra connections)
- **Seeds:** 13-17 (5 different agent spawn positions)
- **Runs per seed:** 9 (3 strategies × 3 replicates)
- **Total:** 45 runs (5 × 9)
- **Duration:** Nov 12-14, 2025

This is the endpoint study - no further experiments planned.
