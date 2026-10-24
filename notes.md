# Dataset notes

- Dataset: UCI Parkinsons (Oxford), 195 recordings
- Label: status (1 = Parkinson's, 0 = healthy)
- Features: 22 (MDVP Fo/Fhi/Flo, Jitter x5, Shimmer x6, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE)
- Subject ID: extracted from the name column (S01, S02, ...)
- Subjects: 32
- Recordings per subject: about 6
- Class counts (subjects): 24 Parkinson's, 8 healthy. By recordings: 147 Parkinson's, 48 healthy.

## Implications for evaluation
- Baseline: always predicting Parkinson's gives 147/195 = 75.4% accuracy, so every model must beat this
- Subject-wise test set will be tiny (about 6-7 subjects, maybe 1-2 healthy), so metrics will be noisy
- Recordings per subject: 6 to 7 (not exactly 6)


## Frontend decisions

- Stack: plain HTML/CSS/JS, no framework. One page, four <section> screens.
- Screens: Record, Processing, Results, Error (see docs/wireframe.png).
- Recording: MediaRecorder, then convert to WAV in the browser (16-bit PCM).
- Task: sustained vowel "aaah", min 5 s, max 10 s.
- API: POST /predict, multipart field "audio", response shaped like mock_response.json.
- Mock switch: USE_MOCK = true at the top of app.js.
- Chart: horizontal bar chart of top_features, using Chart.js from a CDN.
- Errors handled: mic denied, take too short, network/server failure, malformed response.
- Disclaimer: "Screening aid, not a diagnosis." Shown on Record and Results.

## MLP 
- Split: StratifiedGroupKFold by subject, 5 folds, seed 42
- Pipeline: StandardScaler inside each fold + MLPClassifier, early_stopping
- Tuned: hidden sizes, alpha, 5 seeds averaged. Best: hidden=<...>, alpha=<...>
- Single-model mean AUC: <...> (first run: 0.78 for hidden=(32,))
- alpha had almost no effect. Small layers (16) were unstable across seeds (early stopping uses only ~23 validation samples)
- Saved predictions are seed-averaged (small ensemble), so report them as an ensemble
- Config chosen on the same out-of-fold predictions, so results are slightly optimistic

## Evaluation (MLP only so far)
- Recording level: acc 0.810, precision 0.857, recall 0.898, F1 0.877, AUC 0.854
- Subject level: acc 0.844, AUC 0.885 (recordings averaged per subject)
- Baseline is 75.4% accuracy, so 0.81 is only modestly above it
- High recall partly reflects the 75% Parkinson's majority class
- TODO: add specificity and balanced accuracy to evaluate.py