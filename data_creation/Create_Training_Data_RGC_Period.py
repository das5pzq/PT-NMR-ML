#!/usr/bin/env python3
"""CLI for one Monte Carlo batch per shared RGC event period.

Dulya parameters come from dulya_fits_highest.yaml and baseline parameters
from baseline_fits_data_d.yaml. A sample always pairs the two fits from the
same data file. Polarization is resampled in a signed 10%–60% window, and
Cknob is resampled within ±|C|/100 of that period's fitted value.
"""

import argparse
import logging
import os
import sys
import time

from rgc_ranges import SyncedPeriod, load_synced_periods, sample_synced_period_params
from signal_generator_rgc import RGCSignalGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate RGC deuteron training spectra, one batch per event period, "
            "with Dulya and baseline fits kept in sync."
        )
    )
    parser.add_argument(
        "--period_index",
        type=int,
        default=None,
        help="1-based period index. Omit to write every shared period.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples in each period batch",
    )
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
        default="Training_Data_RGC_Period",
        help="Directory for output Parquet files",
    )
    parser.add_argument("--seed", type=int, default=None, help="RNG seed (optional)")
    return parser.parse_args()


def select_periods(
    periods: tuple[SyncedPeriod, ...], period_index: int | None
) -> tuple[SyncedPeriod, ...]:
    if period_index is None:
        return periods
    match = next((period for period in periods if period.index == period_index), None)
    if match is None:
        raise ValueError(
            f"--period_index must be between 1 and {len(periods)}, got {period_index}"
        )
    return (match,)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("cli_rgc_period")

    if args.num_samples < 1:
        logger.error("--num_samples must be >= 1, got %s", args.num_samples)
        return 1

    try:
        periods = load_synced_periods()
        selected = select_periods(periods, args.period_index)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    if not periods:
        logger.error("No shared event periods in the Dulya and baseline fit YAMLs")
        return 1

    generator = RGCSignalGenerator(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        add_noise=bool(args.add_noise),
        noise_level=args.noise_level,
        seed=args.seed,
    )
    logger.info(
        "Shared periods=%d | writing=%d | samples per period=%d",
        len(periods),
        len(selected),
        args.num_samples,
    )

    start = time.time()
    written: list[str] = []
    try:
        for period in selected:
            logger.info(
                "Period %d/%d | %s | dulya templates=%d",
                period.index,
                len(periods),
                period.filename,
                len(period.templates),
            )
            rows = [
                sample_synced_period_params(period, generator.rng)
                for _ in range(args.num_samples)
            ]
            written.append(generator.generate_from_params(rows, job_id=str(period.index)))
    except Exception as exc:  # noqa: BLE001
        logger.error("Error during signal generation: %s", exc, exc_info=True)
        logger.error(
            "Input parameters | num_samples=%s | output_dir=%s | add_noise=%s | period_index=%s",
            args.num_samples,
            args.output_dir,
            args.add_noise,
            args.period_index,
        )
        logger.error(
            "Output directory exists=%s writable=%s",
            os.path.exists(args.output_dir),
            os.access(args.output_dir, os.W_OK) if os.path.exists(args.output_dir) else False,
        )
        return 1

    logger.info("Wrote %d file(s) in %.2f seconds", len(written), time.time() - start)
    for path in written:
        logger.info("Wrote %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
