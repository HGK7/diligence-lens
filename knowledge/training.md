# Project Atlas training policy

Project Atlas trains from a fixed, versioned dataset snapshot. Every experiment records the dataset version, feature configuration, random seed, and model parameters in its run summary.

Training stops when validation loss fails to improve for three consecutive evaluation checks. The checkpoint with the lowest validation loss is retained for final evaluation.