# NBA Player Prop Prediction Model

This module gives you a baseline workflow for NBA player prop picks (over/under) with **no external Python dependencies**.

## What changed

- Added a **readable daily report format** (Markdown table) in addition to CSV output.
- Refreshed `sample_upcoming_props.csv` with a **daily props slate** dated `2026-03-10`.

## Modeling approach

The model learns the average difference between actual outcomes and sportsbook lines (`actual - line`) using:

- global average edge
- player+stat average edge (example: `Luka Doncic::points`)
- opponent-level average edge

At prediction time:

`predicted = line + (0.8 * player_stat_edge + 0.2 * opponent_edge)`

Then:

- `edge = predicted - line`
- `pick = OVER` if edge >= 0, else `UNDER`

## Expected training data format

A CSV with at least these columns:

- `line`
- `actual`
- `player`
- `stat_type`
- `opponent`

See `sample_historical_props.csv` for an example schema.

## Train a model

```bash
python nba_props_model/model.py train \
  --train-csv nba_props_model/sample_historical_props.csv \
  --model-out nba_props_model/artifacts/points_model.pkl \
  --metrics-out nba_props_model/artifacts/metrics.json
```

## Predict to CSV

```bash
python nba_props_model/model.py predict \
  --model-path nba_props_model/artifacts/points_model.pkl \
  --input-csv nba_props_model/sample_upcoming_props.csv \
  --output-csv nba_props_model/artifacts/picks.csv
```

## Generate readable daily report (Markdown)

```bash
python nba_props_model/model.py report \
  --model-path nba_props_model/artifacts/points_model.pkl \
  --input-csv nba_props_model/sample_upcoming_props.csv \
  --output-md nba_props_model/artifacts/daily_picks.md
```

This creates a table sorted by largest absolute edge and also prints it in the terminal.

## Notes

- This is a baseline starting point, not financial advice.
- Replace `sample_upcoming_props.csv` each day with current lines from your sportsbook feed.
- Upgrade quality by adding injury context, expected minutes, pace, and defensive matchup stats.
- Re-train frequently and track closing-line value for model validation.
