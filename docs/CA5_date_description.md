# CA5_date.csv - Technical Dataset Description

**Document Version**: 1.0
**Last Updated**: 2026-01-27
**Author**: C5 Neuro-Symbolic Research Team

---

## 1. Overview

The `CA5_date.csv` file contains historical daily part shipment records for a 5-machine production line (CA5). This dataset serves as the foundation for the C5 Neuro-Symbolic Predictive Maintenance project, which aims to predict next-day staged parts pools.

| Property | Value |
|----------|-------|
| **File Location** | `data/raw/CA5_date.csv` |
| **File Format** | CSV (Comma-Separated Values) |
| **Encoding** | UTF-8 |
| **Total Records** | 11,690 |
| **Date Range** | 1992-02-04 to 2026-01-26 |
| **Time Span** | 34 years (12,410 calendar days) |

---

## 2. Schema Definition

### 2.1 Column Specifications

| Column | Data Type | Description | Constraints |
|--------|-----------|-------------|-------------|
| `date` | Date (M/D/YYYY) | Calendar date of the shipment record | Non-null, unique, ascending |
| `m_1` | Integer | Part ID for machine slot 1 | Range: 1-30 |
| `m_2` | Integer | Part ID for machine slot 2 | Range: 2-35 |
| `m_3` | Integer | Part ID for machine slot 3 | Range: 3-37 |
| `m_4` | Integer | Part ID for machine slot 4 | Range: 4-38 |
| `m_5` | Integer | Part ID for machine slot 5 | Range: 9-39 |

### 2.2 Data Types After Parsing

When loaded with pandas (`pd.read_csv` with date parsing):

```
date    datetime64[ns]
m_1     int64
m_2     int64
m_3     int64
m_4     int64
m_5     int64
```

---

## 3. Data Invariants

The following invariants hold for every record in the dataset:

| Invariant | Description | Verified |
|-----------|-------------|----------|
| **5 Parts Per Day** | Each row contains exactly 5 part IDs | Yes |
| **Unique Parts** | All 5 parts within a row are distinct (no duplicates) | Yes |
| **Part ID Range** | All part IDs are integers from 1 to 39 | Yes |
| **No Null Values** | No missing or null values in any column | Yes |
| **No Duplicate Dates** | Each date appears exactly once | Yes |
| **Sorted Columns** | Within each row: m_1 < m_2 < m_3 < m_4 < m_5 | Yes |

---

## 4. Part Domain

### 4.1 Part ID Range

- **Total Unique Parts**: 39
- **Part ID Range**: 1 to 39 (inclusive)
- **All Part IDs**: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39

### 4.2 Part Frequency Distribution

The part distribution is **near-uniform** with a coefficient of variation (CV) of only **2.43%**.

| Statistic | Value |
|-----------|-------|
| Mean occurrences per part | 1,498.7 |
| Standard deviation | 36.4 |
| Coefficient of Variation | 2.43% |
| Minimum occurrences | 1,420 (Part 38) |
| Maximum occurrences | 1,573 (Part 2) |

### 4.3 Complete Part Frequency Table

| Rank | Part ID | Count | Frequency (%) |
|------|---------|-------|---------------|
| 1 | 2 | 1,573 | 13.46% |
| 2 | 13 | 1,565 | 13.39% |
| 3 | 10 | 1,564 | 13.38% |
| 4 | 11 | 1,563 | 13.37% |
| 5 | 39 | 1,558 | 13.33% |
| 6 | 30 | 1,534 | 13.12% |
| 7 | 27 | 1,532 | 13.11% |
| 8 | 6 | 1,529 | 13.08% |
| 9 | 34 | 1,525 | 13.05% |
| 10 | 3 | 1,524 | 13.04% |
| 11 | 7 | 1,521 | 13.01% |
| 12 | 5 | 1,514 | 12.95% |
| 13 | 25 | 1,510 | 12.92% |
| 14 | 9 | 1,509 | 12.91% |
| 15 | 15 | 1,506 | 12.88% |
| 16 | 24 | 1,505 | 12.87% |
| 17 | 32 | 1,501 | 12.84% |
| 18 | 23 | 1,498 | 12.81% |
| 19 | 1 | 1,498 | 12.81% |
| 20 | 22 | 1,496 | 12.80% |
| 21 | 28 | 1,492 | 12.76% |
| 22 | 8 | 1,491 | 12.75% |
| 23 | 31 | 1,491 | 12.75% |
| 24 | 35 | 1,489 | 12.74% |
| 25 | 14 | 1,483 | 12.69% |
| 26 | 36 | 1,482 | 12.68% |
| 27 | 12 | 1,481 | 12.67% |
| 28 | 26 | 1,481 | 12.67% |
| 29 | 37 | 1,480 | 12.66% |
| 30 | 18 | 1,472 | 12.59% |
| 31 | 19 | 1,471 | 12.58% |
| 32 | 29 | 1,471 | 12.58% |
| 33 | 33 | 1,470 | 12.57% |
| 34 | 21 | 1,457 | 12.46% |
| 35 | 17 | 1,450 | 12.40% |
| 36 | 20 | 1,449 | 12.40% |
| 37 | 16 | 1,448 | 12.39% |
| 38 | 4 | 1,447 | 12.38% |
| 39 | 38 | 1,420 | 12.15% |

---

## 5. Temporal Characteristics

### 5.1 Date Range and Coverage

| Property | Value |
|----------|-------|
| First Record | 1992-02-04 |
| Last Record | 2026-01-26 |
| Calendar Days Spanned | 12,410 |
| Records Present | 11,690 |
| Coverage Rate | 94.2% |

### 5.2 Gap Analysis

Records are **nearly consecutive** with occasional gaps:

| Gap Size | Count | Percentage |
|----------|-------|------------|
| 1 day | 11,221 | 96.0% |
| 2 days | 340 | 2.9% |
| 3 days | 3 | 0.03% |
| 4 days | 125 | 1.1% |

- **Mean gap**: 1.06 days
- **Median gap**: 1 day
- **Maximum gap**: 4 days

### 5.3 Day of Week Distribution

Records are distributed across all days of the week (indicates production operates 7 days/week):

| Day | Count | Percentage |
|-----|-------|------------|
| Monday | 1,648 | 14.1% |
| Tuesday | 1,772 | 15.2% |
| Wednesday | 1,538 | 13.2% |
| Thursday | 1,772 | 15.2% |
| Friday | 1,771 | 15.1% |
| Saturday | 1,545 | 13.2% |
| Sunday | 1,644 | 14.1% |

### 5.4 Monthly Distribution

Average records per month (across all years):

| Month | Avg Records/Year |
|-------|------------------|
| January | 28.4 |
| February | 25.6 |
| March | 28.1 |
| April | 27.3 |
| May | 28.1 |
| June | 27.2 |
| July | 28.4 |
| August | 28.5 |
| September | 27.7 |
| October | 28.6 |
| November | 27.6 |
| December | 28.6 |

### 5.5 Yearly Record Counts (Recent)

| Year | Records | Notes |
|------|---------|-------|
| 2017 | 365 | Full year |
| 2018 | 365 | Full year |
| 2019 | 365 | Full year |
| 2020 | 366 | Leap year |
| 2021 | 365 | Full year |
| 2022 | 365 | Full year |
| 2023 | 365 | Full year |
| 2024 | 366 | Leap year |
| 2025 | 365 | Full year |
| 2026 | 26 | Partial (through Jan 26) |

---

## 6. Machine Column Characteristics

Each machine column (m_1 through m_5) has distinct value ranges due to the sorted nature of the data:

| Column | Min | Max | Mean | Interpretation |
|--------|-----|-----|------|----------------|
| m_1 | 1 | 30 | 6.7 | Smallest part ID of the day |
| m_2 | 2 | 35 | 13.2 | Second smallest |
| m_3 | 3 | 37 | 19.9 | Middle part ID |
| m_4 | 4 | 38 | 26.6 | Second largest |
| m_5 | 9 | 39 | 33.2 | Largest part ID of the day |

**Note**: The columns are sorted in ascending order within each row, meaning `m_1 < m_2 < m_3 < m_4 < m_5` for all records.

---

## 7. Data Quality Assessment

### 7.1 Quality Metrics

| Metric | Status |
|--------|--------|
| Completeness | 100% (no null values) |
| Uniqueness | 100% (no duplicate dates) |
| Validity | 100% (all values within expected ranges) |
| Consistency | 100% (invariants hold for all records) |

### 7.2 Potential Considerations

1. **Near-Uniform Distribution**: The extremely low CV (2.43%) indicates limited predictability ceiling - any prediction model will struggle to significantly outperform frequency-based baselines.

2. **Sorted Columns**: Part IDs are sorted within rows, so column position (m_1 vs m_5) does not represent machine assignment - only relative part ID ordering.

3. **Missing Days**: Approximately 5.8% of calendar days have no records, but gaps are never longer than 4 days.

---

## 8. Sample Data

### First 5 Records

```csv
date,m_1,m_2,m_3,m_4,m_5
2/4/1992,5,8,10,30,38
2/6/1992,2,9,12,18,21
2/7/1992,1,6,17,30,35
2/11/1992,9,10,13,14,23
2/13/1992,3,15,30,34,38
```

### Last 5 Records

```csv
date,m_1,m_2,m_3,m_4,m_5
1/22/2026,6,8,11,27,28
1/23/2026,11,14,15,20,36
1/24/2026,15,18,25,33,35
1/25/2026,1,14,19,27,37
1/26/2026,1,2,5,7,20
```

---

## 9. Usage in Predictive Modeling

### 9.1 Target Variable

For the C5 predictive maintenance task:
- **Input**: Historical sequences of daily part records
- **Output**: Predict which 5 parts will be needed the next day
- **Evaluation**: Good-or-Better rate (4/5 or 5/5 correct predictions)

### 9.2 Recommended Train/Val/Test Split

Based on chronological ordering:

| Split | Date Range | Records | Purpose |
|-------|------------|---------|---------|
| Train | 1992-02-04 to 2023-07-10 | ~10,776 | Model training |
| Validation | 2023-07-11 to 2024-01-07 | ~182 | Hyperparameter tuning |
| Test | 2024-01-08 to 2026-01-26 | ~731 | Final evaluation |

### 9.3 Key Research Findings

From the C5 Neuro-Symbolic project:

1. **Baseline Performance**: Frequency-based selection achieves 65.8% Good-or-Better at K=30
2. **Neural Improvement**: Transformer model achieves 68.2% GoB (+2.4pp)
3. **Hybrid Strategy**: Neural + Baseline for Part 12 achieves 69.9% GoB (+4.1pp over baseline)
4. **Theoretical Ceiling**: Near-uniform distribution limits maximum achievable accuracy

---

## 10. File Integrity

For verification purposes:

| Property | Value |
|----------|-------|
| Expected Row Count | 11,690 (including header) |
| Expected Column Count | 6 |
| Header Row | `date,m_1,m_2,m_3,m_4,m_5` |

### Validation Code

```python
import pandas as pd

df = pd.read_csv('data/raw/CA5_date.csv')

# Basic validation
assert len(df) == 11690, "Row count mismatch"
assert list(df.columns) == ['date', 'm_1', 'm_2', 'm_3', 'm_4', 'm_5'], "Column mismatch"
assert df.isnull().sum().sum() == 0, "Null values found"
assert df['date'].duplicated().sum() == 0, "Duplicate dates found"

# Part validation
all_parts = pd.concat([df['m_1'], df['m_2'], df['m_3'], df['m_4'], df['m_5']])
assert all_parts.min() == 1, "Part ID below 1"
assert all_parts.max() == 39, "Part ID above 39"

print("All validations passed!")
```

---

## 11. Change Log

| Date | Version | Description |
|------|---------|-------------|
| 2026-01-27 | 1.0 | Initial technical description created |

---

**End of Document**
