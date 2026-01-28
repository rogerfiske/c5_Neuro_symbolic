# Production Prediction Workflow

**Workflow ID**: predict
**Purpose**: Generate next-day parts pool prediction using the hybrid strategy
**Prerequisites**: Neural model checkpoint available at `outputs/outputs/best_model/checkpoints/`

---

## Objective

Run the production hybrid inference pipeline to predict the next-day staged parts pool:
- Neural model scores for parts 1-11, 13-39
- Frequency baseline score for Part 12
- Output ranked pool of K=30 parts with probabilities
- Save prediction to `predictions/` directory

---

## Steps

### 1. Determine Target Date

- If user specifies a date, use that
- Otherwise default to the day after the most recent data in `data/raw/CA5_date.csv`

### 2. Run Production Inference

```bash
cd C:\Users\Minis\CascadeProjects\c5_neuro_symbolic
python scripts/production_inference.py --date {TARGET_DATE} --output predictions/prediction_{TARGET_DATE}.txt
```

### 3. Review Output

- Confirm neural model loaded successfully (not baseline fallback)
- Check Part 12 pool status
- Review top-ranked parts and excluded parts
- Report prediction to user

### 4. Optional: Score Against Actuals

If actual parts for the predicted date are available in the CSV:
- Compare predicted pool against actual 5 parts
- Report tier result (Excellent/Good/Unacceptable)
- Log accuracy to `predictions/accuracy_log.csv`

---

## Command Reference

```bash
# Predict for tomorrow (default)
python scripts/production_inference.py --output predictions/prediction_YYYY-MM-DD.txt

# Predict for specific date
python scripts/production_inference.py --date 2026-01-28 --output predictions/prediction_2026-01-28.txt

# Baseline-only mode (no neural model required)
python scripts/production_inference.py --baseline-only --output predictions/prediction_YYYY-MM-DD.txt
```

---

## Output Location

`predictions/prediction_{date}.txt`
