#!/usr/bin/env python3
"""CLI entrypoint for generating RGC-style Monte Carlo training data."""

import argparse
import logging
import os
import sys
import time

from signal_generator_rgc import RGCSignalGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate RGC deuteron training spectra from empirical fit ranges."
    )
    parser.add_argument("--job_id", help="Job identifier for the output filename")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples")
    parser.add_argument(
        "--add_noise", type=int, choices=[0, 1], default=0, help="Set to 1 to add noise"
    )
    parser.add_argument(
        "--noise_level",
        type=float,
        default=2.7e-5,
        help="Gaussian noise standard deviation",
    )
    parser.add_argument(
        "--output_dir",
        default="Training_Data_RGC",
        help="Directory for output Parquet files",
    )
    parser.add_argument("--seed", type=int, default=None, help="RNG seed (optional)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("cli_rgc")

    generator = RGCSignalGenerator(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        add_noise=bool(args.add_noise),
        noise_level=args.noise_level,
        seed=args.seed,
    )
    logger.info("Generating RGC signal data...")

    start = time.time()
    try:
        path = generator.generate_samples(args.job_id)
    except Exception as exc:  # noqa: BLE001
        logger.error("Error during signal generation: %s", exc, exc_info=True)
        logger.error(
            "Input parameters | num_samples=%s | output_dir=%s | add_noise=%s",
            args.num_samples,
            args.output_dir,
            args.add_noise,
        )
        logger.error(
            "Output directory exists=%s writable=%s",
            os.path.exists(args.output_dir),
            os.access(args.output_dir, os.W_OK) if os.path.exists(args.output_dir) else False,
        )
        return 1

    logger.info("Wrote %s in %.2f seconds", path, time.time() - start)
    return 0


if __name__ == "__main__":
    sys.exit(main())
