"""
Runner script to perform fine-tuning of a CNN scenario
and handle the location of the output artifacts.

We made this because we need to run fine-tuning 10 times
to replicate the work that Weimann and Conrad did in
their paper.

Setup:
1. Mount the google drive
2. Clone the project repo
3. cd to the repo root directory
4. Ensure --job-base-dir looks like /content/drive/MyDrive/.../jobs
"""

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights-type",
        choices=("random", "20", "10"),
        required=True,
        help="Weight type",
    )
    parser.add_argument(
        "--job-base-dir",
        type=Path,
        required=True,
        help="Base directory where we will create a job output directory.",
    )
    parser.add_argument(
        "--job-prefix",
        help="If provided, the output job directory will begin with this. No need to put type of weight or seed here.",
    )
    parser.add_argument(
        "--train", type=Path, required=True, help="Path to the train file."
    )
    # parser.add_argument('--val', type=Path, help='Path to the validation file.\n'
    #                                              'Overrides --val-size.')
    parser.add_argument(
        "--test", type=Path, required=True, help="Path to the test file."
    )
    parser.add_argument(
        "--weights-file",
        type=Path,
        help="Path to pretrained weights or a checkpoint of the model.",
    )
    # --train is 80% of full data, and 6.25% of 80% will give us 5% of the full data set for validation.
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.0625,
        help="Size of the validation set or proportion of the train set.",
    )
    parser.add_argument(
        "--arch",
        default="resnet18",
        help="Network architecture: " "`resnet18`, `resnet34` or `resnet50`.",
    )
    # parser.add_argument('--subset', type=float, default=None, help='Size of a subset of the train set '
    #                                                                'or proportion of the train set.')
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    # parser.add_argument('--val-metric', default='loss',
    #                     help='Validation metric used to find the best model at each epoch. Supported metrics are:'
    #                          '`loss`, `acc`, `f1`, `auc`.')
    # parser.add_argument('--channel', type=int, default=None, help='Use only the selected channel. '
    #                                                               'By default use all available channels.')
    parser.add_argument("--epochs", type=int, required=True, help="Number of epochs.")
    parser.add_argument("--seed", type=int, required=True, help="Random state.")
    parser.add_argument("--verbose", action="store_true", help="Show debug messages.")
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Print the command that would be run and exit.",
    )
    args = parser.parse_args()

    job_name = args.job_prefix or "finetune"
    job_name += f"__{args.weights_type}_seed{args.seed}"

    job_dir = args.job_base_dir / job_name
    print("=" * 40)
    print(f"Finetuning output job dir: {job_dir}")
    print("=" * 40)

    command = [
        "python",
        "-u",  # unbuffered output, according to https://stackoverflow.com/a/28319191
        "-m",
        "finetuning.trainer",
        "--job-dir",
        str(job_dir),
        "--train",
        str(args.train),
        "--test",
        str(args.test),
        "--val-size",
        str(args.val_size),
        "--val-metric",
        'f1',
        "--arch",
        args.arch,
        "--batch-size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
    ]

    if args.weights_file:
        command.extend(["--weights-file", str(args.weights_file)])

    if args.verbose:
        command.append("--verbose")

    print(f'Configured command:\n{" ".join(command)}')
    print("=" * 40)

    if args.dryrun:
        print("Dryrun -- Exiting.")
        sys.exit(0)

    # Source: https://stackoverflow.com/a/28319191
    with subprocess.Popen(
        command, stdout=subprocess.PIPE, bufsize=1, universal_newlines=True
    ) as p:
        for line in p.stdout:
            print(line, end="")

    if p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, p.args)


if __name__ == "__main__":
    main()
