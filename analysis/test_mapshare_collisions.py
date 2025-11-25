#!/usr/bin/env python3
"""Run collision test for Map-Sharing that was missing"""
import json
from scipy.stats import mannwhitneyu

# Load raw data
with open('analysis/raw_data.json') as f:
    data = json.load(f)

# Extract collision data
global_collisions = [r['collisions'] for r in data['mapshare']['global']]
none_collisions = [r['collisions'] for r in data['mapshare']['none']]
radio_collisions = [r['collisions'] for r in data['mapshare']['radio_sync']]

print(f"Global collisions (N={len(global_collisions)}): {global_collisions}")
print(f"None collisions (N={len(none_collisions)}): {none_collisions}")
print(f"Radio collisions (N={len(radio_collisions)}): {radio_collisions}")

# Mann-Whitney U test (two-tailed for collisions)
stat_global, p_global = mannwhitneyu(global_collisions, none_collisions, alternative='two-sided')
stat_radio, p_radio = mannwhitneyu(radio_collisions, none_collisions, alternative='two-sided')

print(f"\n=== Map-Sharing Collision Tests ===")
print(f"Global vs None: U={stat_global}, p={p_global:.6f}")
print(f"  Global mean: {sum(global_collisions)/len(global_collisions):.2f}")
print(f"  None mean: {sum(none_collisions)/len(none_collisions):.2f}")

print(f"\nRadio vs None: U={stat_radio}, p={p_radio:.6f}")
print(f"  Radio mean: {sum(radio_collisions)/len(radio_collisions):.2f}")
print(f"  None mean: {sum(none_collisions)/len(none_collisions):.2f}")
