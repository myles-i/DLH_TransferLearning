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
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights-type",
        choices=("random", "50", "20", "10", "1", "100"),
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

    # allow for no arguments to be passsed to weights-file
    parser.add_argument(
        "--weights-file",
        nargs="?",
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
        help="Network architecture: " "`resnet18`, `resnet34`, `resnet50`, 'resnet18_2d.",
    )
    # parser.add_argument('--subset', type=float, default=None, help='Size of a subset of the train set '
    #                                                                'or proportion of the train set.')
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument('--val-metric', default='f1',
                        help='Validation metric used to find the best model at each epoch. Supported metrics are:'
                             '`loss`, `acc`, `f1`, `auc`.')
    # parser.add_argument('--channel', type=int, default=None, help='Use only the selected channel. '
    #                                                               'By default use all available channels.')
    parser.add_argument("--epochs", type=int, required=True, help="Number of epochs.")
    parser.add_argument("--seed", type=int, required=True, help="Random state.")
    # dryrun argument can be True False or {}. If argument is not present, it is False. If it is present and empty, it is True
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser.add_argument(
        "--dryrun",
        nargs="?",
        const=True,
        default=False,
        type=str2bool,
        help="If present, the script will not run the command."
    )
    args = parser.parse_args()

    job_name = args.job_prefix or "finetune"
    job_name += f"__{args.weights_type}_seed{args.seed}"

    job_dir = args.job_base_dir / job_name
    # print("=" * 40)
    # print(f"Finetuning output job dir: {job_dir}")
    # print("=" * 40)

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
        str(args.val_metric),
        "--arch",
        args.arch,
        "--batch-size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--seed",
        str(args.seed),
    ]

    if args.weights_file:
        command.extend(["--weights-file", str(args.weights_file)])

    print(f'Configured command:\n{" ".join(command)}')
    print("=" * 40)

    if args.dryrun:
        # print("Dryrun -- Exiting.")
        sys.exit(0)

    start = time.time()

    # Source: https://stackoverflow.com/a/28319191
    with subprocess.Popen(
        command, stdout=subprocess.PIPE, bufsize=1, universal_newlines=True
    ) as p:
        for line in p.stdout:
            print(line, end="")

    if p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, p.args)

    end = time.time() - start

    with open(job_dir / "time.txt", "w") as f:
        f.write(f"Total time for weight type {args.weights_type}, seed {args.seed}: {end} seconds.")


if __name__ == "__main__":
    main()
