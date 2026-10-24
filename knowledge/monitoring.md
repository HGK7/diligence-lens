# Project Atlas monitoring policy

Project Atlas compares the production feature distribution with the approved training snapshot once per day. A monitoring report is created whenever the comparison detects a material shift.

Prediction logs retain the model version and timestamp but exclude raw customer text. A reviewer investigates drift reports before a retraining run is approved.