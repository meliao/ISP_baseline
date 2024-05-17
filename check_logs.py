"""This file will take some command-line arguments, find log files matching
the given pattern, and check the log files for the presence of a given 
finish string. If the finish string is not found, the job will optionally
be resubmitted.
"""

from typing import List, Tuple
import logging
import os
import sys
import re
import argparse
import time
import pandas as pd


K_SLEEP_TIME_SECS = 1.0


def setup_args() -> argparse.Namespace:
    """Defines the command-line arguments."""

    parser = argparse.ArgumentParser(description="Check logs for finish string.")
    parser.add_argument(
        "-logs_pattern",
        type=str,
        help="Pattern to match log files with a regex capture group. Example: 'log_(.*).txt'",
    )
    parser.add_argument(
        "-jobs_pattern",
        type=str,
        help="Pattern to match log files with a regex capture group. Example: 'log_(.*).txt'",
    )
    parser.add_argument(
        "-logs_dir", type=str, help="Directory to search for log files."
    )
    parser.add_argument(
        "-jobs_dir", type=str, help="Directory to search for job files."
    )
    parser.add_argument(
        "-finish_string",
        type=str,
        help="String to search for in log files. Example: 'Finished'",
    )
    parser.add_argument(
        "-resubmit",
        action="store_true",
        help="If present, resubmit the job if the finish string is not found.",
    )
    parser.add_argument(
        "-output_file",
        type=str,
        default="check_logs.txt",
        help="Name of the output report file. Default: 'check_logs.txt'",
    )
    parser.add_argument("-debug", action="store_true", help="Print debug messages.")
    return parser.parse_args()


def grab_list_of_files(dir: str, re_pattern: str) -> List[Tuple[str, str]]:
    """Grab a list of files from a directory that match a given pattern.
    Parameters
    ----------
    dir : str
        Directory to search for files.
    re_pattern : str
        Regular expression pattern to match files.
    Returns
    -------
    List[Tuple[str, Tuple]]
        List of tuples with the first element being the file name and the
        second element being the regex capture group.
    """
    files = os.listdir(dir)
    files = sorted(files)
    file_list = []
    for f in files:
        match = re.match(re_pattern, f)
        if match:
            assert len(match.groups()) == 1, "Only one capture group is allowed."
            full_fp = os.path.join(dir, f)
            file_list.append((full_fp, match.groups()[0]))
    return file_list


def check_log_files(log_files: List[str], finish_string: str) -> List[bool]:
    """Check a list of log files for the presence of a finish string.
    Parameters
    ----------
    log_files : List[str]
        List of log files to check.
    finish_string : str
        String to search for in log files.
    Returns
    -------
    List[bool]
        List of booleans indicating whether the finish string was found in
        each log file.
    """
    results = []
    for log_file in log_files:
        with open(log_file, "r") as f:
            log_contents = f.read()
        if finish_string in log_contents:
            results.append(True)
        else:
            results.append(False)
    return results


def construct_and_check_job_files(
    log_groups_lst: List[str], job_dir: str, job_pattern: str
) -> List[str]:
    """Construct a list of job files and check that each job file exists.
    job_pattern is a regex pattern with a single capture group.
    """
    job_files = []
    for group in log_groups_lst:

        # Create job_fname by submitting the group into the job_pattern
        # regex capture group
        job_fname = re.sub(r"\(.*\)", group, job_pattern)
        job_file = os.path.join(job_dir, job_fname)
        if not os.path.isfile(job_file):
            logging.error("Job file does not exist: %s" % job_file)
            sys.exit(1)
        job_files.append(job_file)
    return job_files


def resubmit(job_file: str) -> None:
    """Resubmit a job file."""
    logging.debug("Resubmitting job file: %s", job_file)
    os.system("sbatch %s" % job_file)


def main(args: argparse.Namespace) -> None:
    """The main function does the following:
    1. Grabs a list of log files from the log directory.
    2. Grabs a list of job files from the job directory.
    3. Ensures the two lists contain the same regex capture groups.
    4. Checks each log file for the finish string.
    5. Writes the results to a file.
    6. Optionally resubmits the job if the finish string is not found.
    """
    log_files = grab_list_of_files(args.logs_dir, args.logs_pattern)
    job_files = grab_list_of_files(args.jobs_dir, args.jobs_pattern)

    logging.info("Found %i log files and %i job files.", len(log_files), len(job_files))

    log_groups = set([i[1] for i in log_files])
    job_groups = set([i[1] for i in job_files])
    n_jobs = len(job_groups)

    if log_groups != job_groups:
        logging.error(
            "Log and job groups do not match. There are {} log groups and {} job groups.".format(
                len(log_groups), len(job_groups)
            )
        )
        logging.error("Group difference: %s", log_groups ^ job_groups)
        sys.exit(1)

    logging.info("All log and job files match.")

    log_files_lst = [i[0] for i in log_files]
    log_groups_lst = [i[1] for i in log_files]
    job_files_lst = construct_and_check_job_files(
        log_groups_lst, args.jobs_dir, args.jobs_pattern
    )
    completed_jobs_lst = check_log_files(log_files_lst, args.finish_string)
    n_completed = sum(completed_jobs_lst)
    logging.info("Found %i completed jobs out of %i.", n_completed, n_jobs)
    logging.info("Writing results to file: %s", args.output_file)

    results_df = pd.DataFrame(
        {
            "log_file": log_files_lst,
            "job_file": job_files_lst,
            "completed": completed_jobs_lst,
            "key": log_groups_lst,
        }
    )
    results_df.to_csv(args.output_file, sep="\t", index=False)

    if args.resubmit:
        logging.info("Beginning to resubmit %i jobs.", n_jobs - n_completed)
        for i, completed in enumerate(completed_jobs_lst):
            if not completed:
                resubmit(job_files_lst[i])
                time.sleep(K_SLEEP_TIME_SECS)

    logging.info("Finished")


if __name__ == "__main__":
    a = setup_args()

    for name, logger in logging.root.manager.loggerDict.items():
        logging.getLogger(name).setLevel(logging.WARNING)

    if a.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    main(a)
