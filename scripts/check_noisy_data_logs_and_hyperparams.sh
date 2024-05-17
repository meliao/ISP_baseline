# python check_logs.py \
# -logs_pattern "2024-04-26_train_WideBNet_3_freq_noisy_(.*).out" \
# -logs_dir logs/2024-04-26/ \
# -jobs_pattern "2024-04-26_train_WideBNet_3_freq_noisy_(.*).sh" \
# -jobs_dir jobs \
# -finish_string FinishedFinished \
# -output_file data/checks/2024-04-26_train_WideBNet_3_freq_noisy.txt \
# -resubmit

python select_hyperparameter_from_runs.py \
--results-dir-base /net/projects/willettlab/meliao/recursive-linearization/generated \
--results-file-pattern "2024-04-26_train_WideBNet_3_freq_noisy_*/eval_results.txt" \
--output-summary-fp data/hyperparam_summaries/2024-04-26_train_WideBNet_3_freq_noisy.yaml \
--field-name eval_rrmse_mean \
--verbosity-level 1