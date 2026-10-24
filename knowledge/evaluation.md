# Project Atlas evaluation policy

Project Atlas uses five-fold cross-validation. Every preprocessing operation, including scaling and category encoding, must be fit only on the training fold and then applied to the validation fold. Validation data must never influence fitted preprocessing parameters.

The primary classification metric is macro F1. The evaluation report also records per-class precision and recall so weak performance on a minority class is visible.# Project Atlas evaluation policy

Project Atlas uses five-fold cross-validation. Every preprocessing operation, including scaling and category encoding, must be fit only on the training fold and then applied to the validation fold. Validation data must never influence fitted preprocessing parameters.

The primary classification metric is macro F1. The evaluation report also records per-class precision and recall so weak performance on a minority class is visible.