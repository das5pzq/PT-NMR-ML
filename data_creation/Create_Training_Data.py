#!/usr/bin/env python3
"""CLI entrypoint for generating training data."""

import argparse
import logging
import os
import sys
import time

from signal_generator import BaselineConfig, NoiseConfig, OversamplingConfig, SignalGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate signal data for training.")

    parser.add_argument("--job_id", help="Job identifier for the output filename")
    parser.add_argument("--mode", choices=["deuteron", "proton"], default="deuteron", help="Specimen type")
    parser.add_argument(
        "--polarization_type", type=str, choices=["vector", "tensor"], default="vector", help="Polarization type"
    )
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--add_noise", type=int, choices=[0, 1], default=0, help="Set to 1 to add noise")
    parser.add_argument("--oversampling", type=int, choices=[0, 1], default=0, help="Set to 1 to enable oversampling")
    parser.add_argument("--shifting", type=int, choices=[0, 1], default=0, help="Set to 1 to enable shifting")
    parser.add_argument("--oversampled_value", type=float, default=0.0005, help="Value to oversample around")
    parser.add_argument("--oversampling_upper_bound", type=float, default=0.0006, help="Upper bound for oversampling")
    parser.add_argument("--oversampling_lower_bound", type=float, default=0.0004, help="Lower bound for oversampling")
    parser.add_argument("--upper_bound", type=float, default=0.6, help="Upper bound of P value (non-oversampled)")
    parser.add_argument("--lower_bound", type=float, default=0.0005, help="Lower bound of P value (non-oversampled)")
    parser.add_argument("--p_max", type=float, default=0.6, help="Maximum polarization value")
    parser.add_argument("--alpha", type=float, default=2.0, help="Decay rate for power law distribution")
    parser.add_argument("--baseline", type=int, choices=[0, 1], default=1, help="Whether to add a baseline")
    parser.add_argument("--noise_level", type=float, default=10*2.690506959957014e-05, help="Standard deviation of Gaussian noise")
    parser.add_argument("--output_dir", default="Training_Data", help="Directory to save output Parquet files")
    parser.add_argument("--bound", type=float, default=0.08, help="Bound of the shift when shifting is enabled")
    parser.add_argument("--tensor-domain", type=str, choices=["None", "phase", "time"], default="None", help="Domain for tensor polarization. Choose between 'phase' or 'time'.")
    return parser.parse_args()


def build_generator(args: argparse.Namespace) -> SignalGenerator:
    oversampling_cfg = OversamplingConfig(
        enabled=bool(args.oversampling),
        value=args.oversampled_value,
        lower=args.oversampling_lower_bound,
        upper=args.oversampling_upper_bound,
        p_max=args.p_max,
        alpha=args.alpha,
        uniform_lower=args.lower_bound,
        uniform_upper=args.upper_bound,
    )
    noise_cfg = NoiseConfig(enabled=bool(args.add_noise), level=args.noise_level)
    baseline_cfg = BaselineConfig(enabled=bool(args.baseline))
    return SignalGenerator(
        mode=args.mode,
        polarization_type=args.polarization_type,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        oversampling=oversampling_cfg,
        noise=noise_cfg,
        baseline=baseline_cfg,
        shifting=bool(args.shifting),
        bound=args.bound,
        tensor_domain=args.tensor_domain,
    )


def main() -> int:
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("cli")

    generator = build_generator(args)
    logger.info("Generating signal data...")

    start_time = time.time()
    try:
        generator.generate_samples(args.job_id)
    except Exception as exc:  # noqa: BLE001
        logger.error("Error during signal generation: %s", exc, exc_info=True)
        logger.error(
            "Input parameters | mode=%s | pol=%s | tensor_domain=%s | num_samples=%s | output_dir=%s",
            args.mode,
            args.polarization_type,
            args.tensor_domain,
            args.num_samples,
            args.output_dir,
        )
        logger.error("Output directory exists=%s writable=%s", os.path.exists(args.output_dir), os.access(args.output_dir, os.W_OK))
        return 1

    duration = time.time() - start_time
    logger.info("Signal generation complete in %.2f seconds", duration)
    return 0


if __name__ == "__main__":
    sys.exit(main())