# task_progress_sequences
Analysis code for Tarahi et al - prefrontal task progress sequences

All analyses are done on publicly available data from Muysers et al., 2024, Nat Commun, which can be found at https://doi.org/10.5281/zenodo.10528244.

Description of scripts:

task_progress_sequences_prepare_figures.py:
Generates Figures 1&2 as well as Supplementary Figures 1-3 of the manuscript.

The code reads results files of the following separate analysis code:
detect_significant_sequences.py (sequence detection analysis)
detect_significant_sequences_random_time_points.py (random time point control analysis)
extract_trial_durations.py (basic trial parameters)
decode_time_from_sampling_onset.py (prediction of time from sampling or reward start)
decode_trial_outcome_with_sequence_info.py (prediction of trial type from sequential information)
extract_task_phase_coding.py (task phase correlation & decoding analysis)
rastermap_analysis.py (Rastermap implementation)

The path of the results files can be specified in each analysis code and needs to be matched in the figure preparation code.

Additional supporting scripts are:
task_progress_sequences_statistical_functions
task_progress_sequences_utils

All codes were used in Python 3.13.5 (except rastermap, which was run in Python 3.11.13).

Dependencies and used version main analysis (Python 3.13.5)

numpy (2.3.1)
scipy (1.16.0)
matplotlib (3.10.0)
colorcet (3.1.0)
pandas (2.3.3)
pingouin (0.5.5)
h5py (3.14.0)
sklearn (1.8.0)

Dependencies and used version rastermap analysis (Python 3.11.13)

numpy (2.2.5)
h5py (3.14.0)
matplotlib (3.10.0)
