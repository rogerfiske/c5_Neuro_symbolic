#!/usr/bin/env python3
"""
Score a prediction against actual parts shipped.

Usage:
    python scripts/score_prediction.py --date 2026-01-28
    python scripts/score_prediction.py --date 2026-01-28 --prediction-file predictions/prediction_2026-01-28.txt
    python scripts/score_prediction.py --date 2026-01-28 --pool "13,3,5,24,35,28,10,32,31,11,2,22,33,36,18,8,26,21,30,27,38,29,23,25,37,20,16,1,34,14"
"""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


def load_actuals(csv_path: Path, date_str: str) -> list[int] | None:
    """Load actual parts for a given date from the CSV."""
    df = pd.read_csv(csv_path, header=0)
    df.columns = ['date', 'm_1', 'm_2', 'm_3', 'm_4', 'm_5']

    # Parse dates flexibly
    df['date'] = pd.to_datetime(df['date'], format='mixed')
    target_date = pd.to_datetime(date_str)

    row = df[df['date'] == target_date]
    if row.empty:
        return None

    parts = row[['m_1', 'm_2', 'm_3', 'm_4', 'm_5']].values[0].tolist()
    return [int(p) for p in parts]


def load_prediction_from_file(pred_path: Path) -> list[int] | None:
    """Extract predicted pool from a prediction file."""
    if not pred_path.exists():
        return None

    content = pred_path.read_text()

    # Look for the "POOL PARTS (copy-paste ready):" line
    match = re.search(r'POOL PARTS \(copy-paste ready\):\s*\n([0-9, ]+)', content)
    if match:
        parts_str = match.group(1).strip()
        return [int(p.strip()) for p in parts_str.split(',')]

    return None


def score_prediction(predicted_pool: list[int], actual_parts: list[int]) -> dict:
    """Score a prediction against actuals."""
    pool_set = set(predicted_pool)

    results = []
    for part in actual_parts:
        in_pool = part in pool_set
        rank = predicted_pool.index(part) + 1 if in_pool else None
        results.append({
            'part': part,
            'in_pool': in_pool,
            'rank': rank
        })

    covered = sum(1 for r in results if r['in_pool'])
    missed = [r['part'] for r in results if not r['in_pool']]

    # Determine tier
    if covered == 5:
        tier = 'Excellent'
    elif covered == 4:
        tier = 'Good'
    else:
        tier = 'Unacceptable'

    gob = tier in ['Excellent', 'Good']

    return {
        'covered': covered,
        'total': 5,
        'tier': tier,
        'good_or_better': gob,
        'missed_parts': missed,
        'details': results
    }


def print_score_report(date_str: str, actual_parts: list[int],
                       predicted_pool: list[int], score: dict):
    """Print a formatted score report."""
    print("=" * 60)
    print(f"PREDICTION SCORE: {date_str}")
    print("=" * 60)
    print()
    print(f"Actual Parts: {actual_parts}")
    print(f"Pool Size: K={len(predicted_pool)}")
    print()
    print("-" * 40)
    print(f"{'Part':<8} {'In Pool?':<12} {'Rank':<8}")
    print("-" * 40)

    for detail in score['details']:
        part = detail['part']
        in_pool = "YES" if detail['in_pool'] else "NO"
        rank = str(detail['rank']) if detail['rank'] else "(excluded)"
        symbol = "+" if detail['in_pool'] else "-"
        print(f"  {part:<6} {symbol} {in_pool:<10} {rank:<8}")

    print("-" * 40)
    print()
    print(f"RESULT: {score['tier'].upper()} ({score['covered']}/5)")
    print()

    if score['missed_parts']:
        print(f"Missed Parts: {score['missed_parts']}")

    if score['good_or_better']:
        print("Status: Good-or-Better")
    else:
        print("Status: UNACCEPTABLE - expedited parts likely needed")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Score prediction against actuals')
    parser.add_argument('--date', required=True, help='Date to score (YYYY-MM-DD)')
    parser.add_argument('--prediction-file', help='Path to prediction file')
    parser.add_argument('--pool', help='Comma-separated list of predicted parts')
    parser.add_argument('--csv', default='data/raw/CA5_date.csv',
                        help='Path to data CSV')
    parser.add_argument('--quiet', action='store_true', help='Only output tier')

    args = parser.parse_args()

    # Determine project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    csv_path = project_root / args.csv

    # Load actuals
    actuals = load_actuals(csv_path, args.date)
    if actuals is None:
        print(f"ERROR: No data found for date {args.date}")
        sys.exit(1)

    # Load prediction
    predicted_pool = None

    if args.pool:
        predicted_pool = [int(p.strip()) for p in args.pool.split(',')]
    elif args.prediction_file:
        pred_path = project_root / args.prediction_file
        predicted_pool = load_prediction_from_file(pred_path)
    else:
        # Try default prediction file location
        pred_path = project_root / f"predictions/prediction_{args.date}.txt"
        if pred_path.exists():
            predicted_pool = load_prediction_from_file(pred_path)

    if predicted_pool is None:
        print(f"ERROR: Could not load prediction for {args.date}")
        print("Provide --prediction-file or --pool argument")
        sys.exit(1)

    # Score
    score = score_prediction(predicted_pool, actuals)

    if args.quiet:
        print(score['tier'])
    else:
        print_score_report(args.date, actuals, predicted_pool, score)


if __name__ == '__main__':
    main()
