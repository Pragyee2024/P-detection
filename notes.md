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
- Report subject-level results too, and ask Member 1 for grouped k-fold folds
- Recordings per subject: 6 to 7 (not exactly 6)