# python check_logs.py \
# -logs_pattern "2024-05-17_train_WideBNet_2_freq_(.*).out" \
# -logs_dir logs/2024-05-17/ \
# -jobs_pattern "2024-05-17_train_WideBNet_2_freq_(.*).sh" \
# -jobs_dir jobs \
# -finish_string FinishedFinished \
# -output_file data/checks/2024-05-17_train_WideBNet_2_freq.txt \
# -resubmit

python select_hyperparameter_from_runs.py \
--results-dir-base /net/projects/willettlab/meliao/recursive-linearization/generated \
--results-file-pattern "2024-05-17_train_WideBNet_2_freq_*/eval_results.txt" \
--output-summary-fp data/hyperparam_summaries/2024-05-17_train_WideBNet_2_freq.yaml \
--field-name eval_rrmse_mean \
--verbosity-level 1