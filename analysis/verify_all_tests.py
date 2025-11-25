#!/usr/bin/env python3
"""Verify ALL 8 possible statistical tests"""
import json
from scipy.stats import mannwhitneyu

with open('analysis/raw_data.json') as f:
    data = json.load(f)

print("=== VERIFYING ALL 8 TESTS ===\n")

# Map-Sharing data
ms_global_finished = [r['finished'] for r in data['mapshare']['global']]
ms_radio_finished = [r['finished'] for r in data['mapshare']['radio_sync']]
ms_none_finished = [r['finished'] for r in data['mapshare']['none']]
ms_global_collisions = [r['collisions'] for r in data['mapshare']['global']]
ms_radio_collisions = [r['collisions'] for r in data['mapshare']['radio_sync']]
ms_none_collisions = [r['collisions'] for r in data['mapshare']['none']]

# Communication data
comm_free_finished = [r['finished'] for r in data['communication']['freeform']]
comm_struct_finished = [r['finished'] for r in data['communication']['structured']]
comm_none_finished = [r['finished'] for r in data['communication']['none']]
comm_free_collisions = [r['collisions'] for r in data['communication']['freeform']]
comm_struct_collisions = [r['collisions'] for r in data['communication']['structured']]
comm_none_collisions = [r['collisions'] for r in data['communication']['none']]

tests = []

# Map-Sharing tests
u, p = mannwhitneyu(ms_global_finished, ms_none_finished, alternative='greater')
tests.append(("Map-Sharing", "Global vs None", "Success", u, p, "one-tailed"))

u, p = mannwhitneyu(ms_radio_finished, ms_none_finished, alternative='greater')
tests.append(("Map-Sharing", "Radio vs None", "Success", u, p, "one-tailed"))

u, p = mannwhitneyu(ms_global_collisions, ms_none_collisions, alternative='two-sided')
tests.append(("Map-Sharing", "Global vs None", "Collisions", u, p, "two-tailed"))

u, p = mannwhitneyu(ms_radio_collisions, ms_none_collisions, alternative='two-sided')
tests.append(("Map-Sharing", "Radio vs None", "Collisions", u, p, "two-tailed"))

# Communication tests
u, p = mannwhitneyu(comm_free_finished, comm_none_finished, alternative='two-sided')
tests.append(("Communication", "Freeform vs None", "Success", u, p, "two-tailed"))

u, p = mannwhitneyu(comm_struct_finished, comm_none_finished, alternative='two-sided')
tests.append(("Communication", "Structured vs None", "Success", u, p, "two-tailed"))

u, p = mannwhitneyu(comm_free_collisions, comm_none_collisions, alternative='two-sided')
tests.append(("Communication", "Freeform vs None", "Collisions", u, p, "two-tailed"))

u, p = mannwhitneyu(comm_struct_collisions, comm_none_collisions, alternative='two-sided')
tests.append(("Communication", "Structured vs None", "Collisions", u, p, "two-tailed"))

print(f"{'Branch':<15} {'Comparison':<25} {'Metric':<12} {'U':<8} {'p-value':<10} {'Test Type'}")
print("=" * 95)
for branch, comp, metric, u, p, test_type in tests:
    sig = "***" if p < 0.001 else "n.s."
    print(f"{branch:<15} {comp:<25} {metric:<12} {u:<8.1f} {p:<10.6f} {test_type:<12} {sig}")

print(f"\n=== TOTAL: {len(tests)} TESTS ===")
