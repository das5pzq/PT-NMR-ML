#!/usr/bin/env python3
"""
High-level signal generation utilities for training data creation.

This module keeps the main CLI thin while grouping the generation logic,
sampling helpers, and noise handling in one place. The goal is to keep the
code readable, testable, and easy to extend for additional modes or
polarization types.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tqdm
from scipy.integrate import trapezoid
from scipy.signal import hilbert

from Lineshape import (
    Baseline,
    GenerateVectorLineshape,
    Lineshape,
    SamplingVectorLineshape,
    generate_proton_signal,
)

LOGGER_NAME = "SignalGenerator"
# Calibrates integrated proton lineshape area (∫y dx) to polarization: P = area * PROTON_CC.
PROTON_CC = 2.377120

CC = 1.0


@dataclass(frozen=True)
class OversamplingConfig:
    enabled: bool = False
    value: float = 0.0005
    lower: float = 0.0004
    upper: float = 0.0006
    p_max: float = 0.6
    alpha: float = 2.0
    uniform_lower: float = 0.1
    uniform_upper: float = 0.6

    @property
    def uniform_range(self) -> Tuple[float, float]:
        return (self.uniform_lower, self.uniform_upper)


@dataclass(frozen=True)
class NoiseConfig:
    enabled: bool = False
    level: float = 0.02


@dataclass(frozen=True)
class BaselineConfig:
    enabled: bool = True
    U: float = 10.43
    # U: float = 1.0
    eta: float = 1.04e-2
    g: float = 0.05
    Cstray: float = 10 ** (-20)
    shift: float = 0.0
    cable: float = 0.5
    Cknob: float = 0.204
    phi: float = 2*np.pi
    # Cknob: float = 0.0647
    vary_baseline: bool = True
    U_range: Tuple[float, float] = (5.0, 10.0)
    Cknob_range: Tuple[float, float] = (0.15, 0.3)
    # Cknob_range: Tuple[float, float] = (0.0477, 0.0647)
    phase_range: Tuple[float, float] = (2*np.pi * .80, 2 * np.pi * 1.20)

class SignalGenerator:

    def __init__(
        self,
        mode: str = "deuteron",
        polarization_type: str = "vector",
        output_dir: str = "Training_Data",
        num_samples: int = 10,
        oversampling: OversamplingConfig = OversamplingConfig(),
        noise: NoiseConfig = NoiseConfig(),
        baseline: BaselineConfig = BaselineConfig(),
        shifting: bool = False,
        bound: float = 0.08,
        tensor_domain: str = "None",
    ) -> None:
        self.mode = mode.lower()
        self.polarization_type = polarization_type.lower()
        self.output_dir = output_dir
        self.num_samples = num_samples
        self.oversampling_cfg = oversampling
        self.noise_cfg = noise
        self.baseline_cfg = baseline
        self.shifting = shifting
        self.bound = bound
        self.tensor_domain = tensor_domain
        self.phi = 0.0
        # Tensor phi precomputation
        if self.polarization_type == "tensor":
            if self.tensor_domain == "phase":
                self.phi_values = np.linspace(0, 180, 500)
                self.phi_rad = np.deg2rad(self.phi_values)
                self.sin_phi = np.sin(self.phi_rad)
                self.cos_phi = np.cos(self.phi_rad)
                self.phi = None
            elif self.tensor_domain == "time":
                self.phi = np.pi / 2 # 90 degree
                self.phi_values = None
                self.cos_phi = self.sin_phi = None
            elif self.tensor_domain == "None":
                self.phi_values = None
                self.sin_phi = self.cos_phi = None
                self.phi = 2*np.pi # 360 degrees
            else:
                raise ValueError(f"Invalid tensor domain: {self.tensor_domain}. Choose 'None', 'phase' or 'time'.")

        # Mode-specific constants
        if self.mode == "deuteron":
            self.center_freq = 32.68
        elif self.mode == "proton":
            self.center_freq = 213
            # Proton-specific overrides
            self.baseline_cfg = BaselineConfig(
                enabled=self.baseline_cfg.enabled,
                U=self.baseline_cfg.U,
                eta=self.baseline_cfg.eta,
                g=self.baseline_cfg.g,
                Cstray=self.baseline_cfg.Cstray,
                shift=self.baseline_cfg.shift,
                cable= 22 /2 ,
                Cknob=0.0647,
            )
        else:
            raise ValueError(f"Invalid mode: {mode}. Choose 'deuteron' or 'proton'.")

        self.bigy = np.sqrt(3 - self.baseline_cfg.eta * np.cos(2 * (self.phi or 0)))

        # Cache for tensor pre-computations
        self._lineshape_cache: Optional[dict[str, np.ndarray]] = None
        self._baseline_cache: Optional[np.ndarray] = None

        os.makedirs(self.output_dir, exist_ok=True)
        self.logger = logging.getLogger(LOGGER_NAME)

    # ------------------------------------------------------------------ #
    # Sampling helpers
    # ------------------------------------------------------------------ #
    def _sample_p_values(self) -> np.ndarray:
        if self.oversampling_cfg.enabled:
            self.logger.info(
                "Oversampling around %.4f in [%.4f, %.4f]",
                self.oversampling_cfg.value,
                self.oversampling_cfg.lower,
                self.oversampling_cfg.upper,
            )
            oversample_p = np.random.uniform(
                self.oversampling_cfg.lower, self.oversampling_cfg.upper, self.num_samples
            )

            def sample_exponential_with_cutoff(scale: float, p_min: float, p_max: float, size: int) -> np.ndarray:
                samples: List[float] = []
                while len(samples) < size:
                    new_samples = p_min + np.random.exponential(scale=scale, size=size)
                    filtered = new_samples[new_samples <= p_max]
                    samples.extend(filtered.tolist())
                return np.array(samples[:size])

            p_exp = sample_exponential_with_cutoff(
                scale=self.oversampling_cfg.alpha,
                p_min=self.oversampling_cfg.upper,
                p_max=self.oversampling_cfg.p_max,
                size=self.num_samples,
            )
            return np.concatenate([oversample_p, p_exp])

        lower, upper = self.oversampling_cfg.uniform_range
        self.logger.info("Uniformly sampling P in [%.4f, %.4f]", lower, upper)
        return np.random.uniform(lower, upper, self.num_samples)

    # ------------------------------------------------------------------ #
    # Tensor precomputation
    # ------------------------------------------------------------------ #
    def _sample_baseline_params(self) -> Tuple[float, float, float]:
        """Sample baseline parameters if variation is enabled."""
        if self.baseline_cfg.vary_baseline:
            U = np.random.uniform(*self.baseline_cfg.U_range)
            Cknob = np.random.uniform(*self.baseline_cfg.Cknob_range)
            phase = np.random.uniform(*self.baseline_cfg.phase_range)
            return U, Cknob, phase
        return self.baseline_cfg.U, self.baseline_cfg.Cknob, (self.phi or 0)

    def _precompute_tensor_components(self) -> None:
        if self._lineshape_cache is not None and not self.baseline_cfg.vary_baseline:
            return

        X = np.linspace(-6, 6, 500)
        yvals_absorp1 = Lineshape(X, 1, self.baseline_cfg.eta, self.phi or 0, self.baseline_cfg.g)
        yvals_absorp2 = Lineshape(-X, 1, self.baseline_cfg.eta, self.phi or 0, self.baseline_cfg.g)
        yvals_disp1 = np.imag(hilbert(yvals_absorp1))
        yvals_disp2 = np.imag(hilbert(yvals_absorp2))

        self._lineshape_cache = {
            "X": X,
            "yvals_absorp1": yvals_absorp1,
            "yvals_absorp2": yvals_absorp2,
            "yvals_disp1": yvals_disp1,
            "yvals_disp2": yvals_disp2,
        }

        if self.baseline_cfg.enabled:
            if self.baseline_cfg.vary_baseline:
                # For varied baseline, we'll compute baseline per sample
                self._baseline_cache = None
            elif self.tensor_domain == "phase":
                baseline_all_phi = np.zeros((500, 500))
                for i, phi_deg in enumerate(self.phi_values):
                    baseline_all_phi[:, i] = Baseline(
                        X,
                        self.baseline_cfg.U,
                        self.baseline_cfg.Cknob,
                        self.baseline_cfg.eta,
                        self.baseline_cfg.cable,
                        self.baseline_cfg.Cstray,
                        phi_deg,
                        self.baseline_cfg.shift,
                        self.mode,
                    )
                self._baseline_cache = baseline_all_phi
            else:
                phase_deg = np.rad2deg(self.phi or 0)
                baseline = Baseline(
                    X,
                    self.baseline_cfg.U,
                    self.baseline_cfg.Cknob,
                    self.baseline_cfg.eta,
                    self.baseline_cfg.cable,
                    self.baseline_cfg.Cstray,
                    phase_deg,
                    self.baseline_cfg.shift,
                    self.mode,
                )
                self._baseline_cache = baseline
        else:
            self._baseline_cache = np.zeros((500, 500)) if self.tensor_domain == "phase" else np.zeros(500)

    # ------------------------------------------------------------------ #
    # Signal generation
    # ------------------------------------------------------------------ #
    def _generate_tensor_batch(self, p_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self._precompute_tensor_components()

        yvals_absorp1 = self._lineshape_cache["yvals_absorp1"]
        yvals_absorp2 = self._lineshape_cache["yvals_absorp2"]
        yvals_disp1 = self._lineshape_cache["yvals_disp1"]
        yvals_disp2 = self._lineshape_cache["yvals_disp2"]
        X = self._lineshape_cache["X"]

        r_values = (np.sqrt(4 - 3 * p_values ** 2) + p_values) / (2 - 2 * p_values)
        num_samples = len(p_values)
        all_signals = np.zeros((num_samples, 500, 500))
        all_lineshapes = np.zeros((num_samples, 500, 500))

        batch_size = min(50, num_samples)
        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            batch_r = r_values[batch_start:batch_end]

            yvals_absorp1_3d = yvals_absorp1[np.newaxis, :, np.newaxis]
            yvals_absorp2_3d = yvals_absorp2[np.newaxis, :, np.newaxis]
            yvals_disp1_3d = yvals_disp1[np.newaxis, :, np.newaxis]
            yvals_disp2_3d = yvals_disp2[np.newaxis, :, np.newaxis]
            sin_phi_2d = self.sin_phi[np.newaxis, np.newaxis, :]
            cos_phi_2d = self.cos_phi[np.newaxis, np.newaxis, :]
            r_3d = batch_r[:, np.newaxis, np.newaxis]

            Iplus = r_3d * (yvals_absorp1_3d * sin_phi_2d + yvals_disp1_3d * cos_phi_2d)
            Iminus = yvals_absorp2_3d * sin_phi_2d + yvals_disp2_3d * cos_phi_2d
            lineshape = Iplus + Iminus

            if self.baseline_cfg.enabled:
                if self.baseline_cfg.vary_baseline:
                    # Generate baseline for each sample in batch
                    baseline_batch = np.zeros((batch_end - batch_start, 500, 500))
                    for i in range(batch_end - batch_start):
                        U, Cknob, phase = self._sample_baseline_params()
                        phase_deg = np.rad2deg(phase)
                        # Use sampled phase for all phi values in this sample
                        for j, _ in enumerate(self.phi_values):
                            baseline_batch[i, :, j] = Baseline(
                                X,
                                U,
                                Cknob,
                                self.baseline_cfg.eta,
                                self.baseline_cfg.cable,
                                self.baseline_cfg.Cstray,
                                phase_deg,
                                self.baseline_cfg.shift,
                                self.mode 
                            )
                    total_signal = lineshape + baseline_batch
                else:
                    baseline_3d = self._baseline_cache[np.newaxis, :, :]
                    total_signal = lineshape + baseline_3d
            else:
                total_signal = lineshape

            all_signals[batch_start:batch_end] = total_signal
            all_lineshapes[batch_start:batch_end] = lineshape

        return all_signals, all_lineshapes

    def _generate_tensor_single(self, P: float) -> np.ndarray:
        self._precompute_tensor_components()

        yvals_absorp1 = self._lineshape_cache["yvals_absorp1"][:, np.newaxis]
        yvals_absorp2 = self._lineshape_cache["yvals_absorp2"][:, np.newaxis]
        yvals_disp1 = self._lineshape_cache["yvals_disp1"][:, np.newaxis]
        yvals_disp2 = self._lineshape_cache["yvals_disp2"][:, np.newaxis]
        sin_phi_2d = self.sin_phi[np.newaxis, :]
        cos_phi_2d = self.cos_phi[np.newaxis, :]
        X = self._lineshape_cache["X"]

        r = (np.sqrt(4 - 3 * P ** 2) + P) / (2 - 2 * P)
        Iplus = r * (yvals_absorp1 * sin_phi_2d + yvals_disp1 * cos_phi_2d)
        Iminus = yvals_absorp2 * sin_phi_2d + yvals_disp2 * cos_phi_2d
        signal = Iplus + Iminus

        if self.baseline_cfg.enabled:
            if self.baseline_cfg.vary_baseline:
                U, Cknob, phase = self._sample_baseline_params()
                phase_deg = np.rad2deg(phase)
                baseline = np.zeros((500, 500))
                for j, _ in enumerate(self.phi_values):
                    baseline[:, j] = Baseline(
                        X,
                        U,
                        Cknob,
                        self.baseline_cfg.eta,
                        self.baseline_cfg.cable,
                        self.baseline_cfg.Cstray,
                        phase_deg,
                        self.baseline_cfg.shift,
                        self.mode 
                    )
                return signal + baseline
            return signal + self._baseline_cache
        return signal

    def _generate_tensor_time_series(self, P: float, time_steps: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        self._precompute_tensor_components()

        yvals_absorp1 = self._lineshape_cache["yvals_absorp1"]
        yvals_absorp2 = self._lineshape_cache["yvals_absorp2"]
        yvals_disp1 = self._lineshape_cache["yvals_disp1"]
        yvals_disp2 = self._lineshape_cache["yvals_disp2"]

        r = (np.sqrt(4 - 3 * P ** 2) + P) / (2 - 2 * P)
        sin_phi = np.sin(self.phi)
        cos_phi = np.cos(self.phi)
        Iplus = r * (yvals_absorp1 * sin_phi + yvals_disp1 * cos_phi)
        Iminus = yvals_absorp2 * sin_phi + yvals_disp2 * cos_phi
        lineshape = Iplus + Iminus

        if self.baseline_cfg.enabled:
            if self.baseline_cfg.vary_baseline:
                U, Cknob, phase = self._sample_baseline_params()
                phase_deg = np.rad2deg(phase)
                baseline = Baseline(
                    self._lineshape_cache["X"],
                    U,
                    Cknob,
                    self.baseline_cfg.eta,
                    self.baseline_cfg.cable,
                    self.baseline_cfg.Cstray,
                    phase_deg,
                    self.baseline_cfg.shift,
                    self.mode,
                )
            else:
                baseline = self._baseline_cache
            base_signal = lineshape + baseline
        else:
            base_signal = lineshape

        signal = np.zeros((base_signal.shape[0], time_steps))
        snr_values: List[float] = []
        for t in range(time_steps):
            noise, _ = self._generate_noise(lineshape)
            signal[:, t] = base_signal + noise
            snr = self._calculate_snr(lineshape, noise)
            if snr is not None:
                snr_values.append(snr)

        snr_avg = float(np.mean(snr_values)) if snr_values else None
        return signal, lineshape, snr_avg

    def _qmeter_baseline_1d(
        self, X: np.ndarray, U: float, Cknob: float, phase_rad: Optional[float]
    ) -> np.ndarray:
        """Q-meter (cable/trim) background vs frequency (MHz) for 1D vector spectra."""
        pr = 2*np.pi if phase_rad is None else float(phase_rad)
        # phase_deg = np.rad2deg(pr)
        phase_deg = pr
        return Baseline(
            X,
            U,
            Cknob,
            self.baseline_cfg.eta,
            self.baseline_cfg.cable,
            self.baseline_cfg.Cstray,
            phase_deg,
            self.baseline_cfg.shift,
            self.mode,
        )

    def _generate_vector_signal(self, P: float) -> Tuple[np.ndarray, float, np.ndarray]:
        X = np.linspace(self.center_freq - 6, self.center_freq + 6, 500)
        R = np.linspace(-6, 6, 500)

        # Sample baseline parameters once if varying
        if self.baseline_cfg.vary_baseline:
            U, Cknob, phase = self._sample_baseline_params()
            phi_to_use = phase
        else:
            U = self.baseline_cfg.U
            Cknob = self.baseline_cfg.Cknob
            phi_to_use = self.phi

        if self.shifting:
            lineshape = SamplingVectorLineshape(P, R, self.bound, 3.528*CC, self.baseline_cfg.eta, phi_to_use, self.baseline_cfg.g)
            area = trapezoid(lineshape, R)
        else:
            lineshape, _, _ = GenerateVectorLineshape(P, R, 3.528*CC, self.baseline_cfg.eta, phi_to_use, self.baseline_cfg.g)
            area = trapezoid(lineshape, R)

        if self.baseline_cfg.enabled:
            # Same phase in radians as used for the lineshape (U, Cknob from the same draw when varying).
            signal = lineshape + self._qmeter_baseline_1d(X, U, Cknob, phi_to_use)
        else:
            signal = lineshape

        return signal, area, lineshape

    # ------------------------------------------------------------------ #
    # Noise helpers
    # ------------------------------------------------------------------ #
    def _generate_noise(self, lineshape: np.ndarray) -> Tuple[np.ndarray, Optional[float]]:
        """Gaussian noise with std ``noise_cfg.level`` (``--noise_level`` / NOISE_LEVEL)."""
        noise = np.zeros_like(lineshape)
        if not self.noise_cfg.enabled:
            return noise, None

        std = float(self.noise_cfg.level)
        if std <= 0:
            return noise, None

        noise = np.random.normal(0, std, size=lineshape.shape)
        return noise, None

    @staticmethod
    def _calculate_snr(lineshape: np.ndarray, noise: np.ndarray) -> Optional[float]:
        if np.all(noise == 0):
            return None
        lineshape_max = np.max(np.abs(lineshape))
        noise_std = np.max(np.abs(noise))
        if noise_std <= 0:
            return None
        return lineshape_max / noise_std

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def generate_samples(self, job_id: Optional[str] = None) -> str:
        self.logger.info(
            "Generating %s samples (%s, %s)",
            self.num_samples,
            self.mode,
            self.polarization_type,
        )

        p_values = self._sample_p_values()
        signal_arr: List[np.ndarray] = []
        snr_arr: List[Optional[float]] = []
        area_arr: List[float] = []

        if self.polarization_type == "tensor" and self.mode == "deuteron":
            if self.tensor_domain == "phase":
                signals, lineshapes = self._generate_tensor_batch(p_values)
                for i, (signal, lineshape) in enumerate(zip(signals, lineshapes)):
                    noise, _ = self._generate_noise(lineshape)
                    signal += noise
                    signal_arr.append(signal.reshape(500, 500, 1))
                    snr_arr.append(self._calculate_snr(lineshape, noise))
                    if (i + 1) % 100 == 0:
                        self.logger.info("Processed %d/%d samples", i + 1, len(p_values))
            elif self.tensor_domain == "time":
                time_steps = 500
                for i, P in tqdm.tqdm(enumerate(p_values), total=len(p_values), desc="Generating samples"):
                    # For each P in the selected range, generate repeated time steps
                    # with fresh noise applied at every step.
                    signal, lineshape, snr = self._generate_tensor_time_series(P, time_steps=time_steps)
                    signal_arr.append(signal.reshape(500, time_steps, 1))
                    snr_arr.append(snr)
                    if (i + 1) % 100 == 0:
                        self.logger.info("Generated %d/%d samples", i + 1, len(p_values))
            else:
                for i, P in tqdm.tqdm(enumerate(p_values), total=len(p_values), desc="Generating samples"):
                    signal = self._generate_tensor_single(P)
                    noise, _ = self._generate_noise(signal)
                    signal += noise
                    signal_arr.append(signal.reshape(500, 500, 1))
                    snr_arr.append(self._calculate_snr(signal, noise))
                    if (i + 1) % 100 == 0:
                        self.logger.info("Generated %d/%d samples", i + 1, len(p_values))
            p_column = np.asarray(p_values, dtype=np.float64)
        else:
            p_list: List[float] = []
            for i, P in tqdm.tqdm(enumerate(p_values), total=len(p_values), desc="Generating samples"):
                if self.mode == "deuteron":
                    signal, area, lineshape = (
                        self._generate_vector_signal(P)
                        if self.polarization_type == "vector"
                        else self._generate_tensor_single(P)
                    )
                    p_list.append(float(P))
                else:
                    # x must be in the same units as generate_proton_signal's Voigt center (~213 MHz);
                    # linspace(-3, 3) put the window 210 MHz away from the peak.
                    x = np.linspace(self.center_freq - 3, self.center_freq + 3, 500)
                    target_area = float(P) / PROTON_CC
                    lineshape, area = generate_proton_signal(x, target_area=target_area)
                    p_list.append(float(P))

                    if self.baseline_cfg.enabled:
                        if self.baseline_cfg.vary_baseline:
                            U, Cknob, phase = self._sample_baseline_params()
                            phi_baseline = phase
                        else:
                            U = self.baseline_cfg.U
                            Cknob = self.baseline_cfg.Cknob
                            phi_baseline = self.phi
                        signal = lineshape + self._qmeter_baseline_1d(x, U, Cknob, phi_baseline)
                    else:
                        signal = lineshape

                noise, _ = self._generate_noise(lineshape)
                signal += noise

                if self.polarization_type == "tensor":
                    signal_arr.append(signal.reshape(500, 500, 1))
                else:
                    signal_arr.append(signal)
                    if area is not None:
                        area_arr.append(area)

                snr_arr.append(self._calculate_snr(lineshape, noise))
                if (i + 1) % 100 == 0:
                    self.logger.info("Generated %d/%d samples", i + 1, len(p_values))

            p_column = np.array(p_list, dtype=np.float64)

        df = self._build_dataframe(signal_arr, p_column, snr_arr, area_arr)
        file_path = self._persist(df, job_id)
        return file_path

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def _build_dataframe(
        self,
        signals: Sequence[np.ndarray],
        p_values: np.ndarray,
        snr_arr: Sequence[Optional[float]],
        area_arr: Sequence[float],
    ) -> pd.DataFrame:
        if self.polarization_type == "tensor":
            df = pd.DataFrame(
                {
                    "signal": [sig.flatten() for sig in signals],
                    "P": p_values,
                    "SNR": snr_arr,
                }
            )
            self.logger.info("Generated %d tensor signals", len(signals))
        else:
            df = pd.DataFrame(signals)
            if len(p_values) > 0:
                df["P"] = p_values
            if len(snr_arr) > 0:
                df["SNR"] = snr_arr
            if len(area_arr) > 0:
                df["Area"] = area_arr
        return df

    def _persist(self, df: pd.DataFrame, job_id: Optional[str]) -> str:
        filename = f"Sample_{self.polarization_type}"
        if job_id is not None:
            filename += f"_{job_id}"
        filename += ".parquet"
        file_path = os.path.join(self.output_dir, filename)

        time_steps = 500 if self.polarization_type == "tensor" and self.tensor_domain == "time" else None
        metadata = {
            "polarization_type": self.polarization_type,
            "mode": self.mode,
            "tensor_domain": self.tensor_domain,
            "frequency_bins": 500,
            "phi_bins": 500
            if self.polarization_type == "tensor" and self.tensor_domain == "phase"
            else 1,
            "signal_shape": (500, time_steps, 1)
            if time_steps is not None
            else (500, 500, 1)
            if self.polarization_type == "tensor"
            else (500,),
            "is_flattened": self.polarization_type == "tensor",
            "frequency_range": (self.center_freq - 6, self.center_freq + 6)
            if self.polarization_type == "vector"
            else (-3, 3),
            "phi_range": (0, 180)
            if self.polarization_type == "tensor" and self.tensor_domain == "phase"
            else None,
            "time_steps": time_steps,
            "num_samples": len(df),
        }

        df.attrs["metadata"] = str(metadata)
        df.to_parquet(file_path, engine="pyarrow", compression="snappy")
        self.logger.info("Parquet saved to %s (shape=%s)", file_path, df.shape)
        return file_path

