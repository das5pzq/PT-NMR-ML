"""RGC deuteron vector signal generator using empirical fit ranges."""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Sequence

import numpy as np
import tqdm

from physics.Lineshape import DulyaFit, QmeterGain
from physics.Modified_Baseline import Baseline
from rgc_ranges import (
    BASELINE_SAMPLE_KEYS,
    CENTER_MHZ,
    DULYA_SAMPLE_KEYS,
    RGC_N_BINS,
    rgc_frequency_mhz,
    sample_rgc_params,
)

LOGGER_NAME = "RGCSignalGenerator"


class RGCSignalGenerator:
    """Monte Carlo deuteron spectra matching RGC Dulya + baseline fits."""

    def __init__(
        self,
        output_dir: str = "Training_Data_RGC",
        num_samples: int = 10,
        add_noise: bool = False,
        noise_level: float = 2.7e-5,
        seed: Optional[int] = None,
    ) -> None:
        self.output_dir = output_dir
        self.num_samples = int(num_samples)
        self.add_noise = bool(add_noise)
        self.noise_level = float(noise_level)
        self.rng = np.random.default_rng(seed)
        self.freq_mhz = rgc_frequency_mhz()
        self.logger = logging.getLogger(LOGGER_NAME)
        os.makedirs(self.output_dir, exist_ok=True)

    def _lineshape(self, params: dict[str, float]) -> tuple[np.ndarray, float]:
        p = float(params["P"])
        cc = float(params["cc"])
        half_width = float(params["half_width_mhz"])
        center = float(params.get("center_mhz", CENTER_MHZ))
        x_eff = self.freq_mhz - center
        x = x_eff / half_width
        shape = DulyaFit(
            x,
            p,
            1.0 / cc,
            float(params["eta"]),
            float(params["phi"]),
            float(params["g"]),
            g1_amp=float(params["g1_amp"]),
            g1_loc=float(params["g1_loc"]),
            g1_wid=float(params["g1_wid"]),
            g2_amp=float(params["g2_amp"]),
            g2_loc=float(params["g2_loc"]),
            g2_wid=float(params["g2_wid"]),
            powder_average=True,
        )
        area = float(np.sum(shape))
        gain = QmeterGain(x_eff, half_width, float(params["xi"]))
        return shape * gain, area

    def _baseline(self, params: dict[str, float]) -> np.ndarray:
        return Baseline(
            self.freq_mhz,
            float(params["U"]),
            float(params["Cknob"]),
            float(params["baseline_eta"]),
            float(params["trim"]),
            float(params["Cstray"]),
            float(params["phi_const"]),
            float(params["DC_offset"]),
            "deuteron",
            L0=float(params["L0"]),
            Rcoil=float(params["Rcoil"]),
            R=float(params["R"]),
            R1=float(params["R1"]),
            r=float(params["r"]),
            alpha=float(params["alpha"]),
            beta1=float(params["beta1"]),
            Z_cable=float(params["Z_cable"]),
            D=float(params["D"]),
            M=float(params["M"]),
            delta_C=float(params["delta_C"]),
            delta_phi=float(params["delta_phi"]),
            delta_phase=float(params["delta_phase"]),
            delta_l=float(params["delta_l"]),
        )

    def _noise(self, size: int) -> np.ndarray:
        if not self.add_noise or self.noise_level <= 0.0:
            return np.zeros(size, dtype=np.float64)
        return self.rng.normal(0.0, self.noise_level, size=size)

    @staticmethod
    def _snr(lineshape: np.ndarray, noise: np.ndarray) -> Optional[float]:
        if np.all(noise == 0):
            return None
        noise_std = float(np.std(noise))
        if noise_std <= 0.0:
            return None
        return float(np.max(np.abs(lineshape)) / noise_std)

    def generate_one(self, params: Optional[dict[str, float]] = None) -> dict:
        if params is None:
            params = sample_rgc_params(self.rng)
        lineshape, area = self._lineshape(params)
        baseline = self._baseline(params)
        noise = self._noise(len(lineshape))
        signal = lineshape + baseline + noise
        return {
            "params": params,
            "signal": signal,
            "lineshape": lineshape,
            "baseline": baseline,
            "noise": noise,
            "area": area,
            "snr": self._snr(lineshape, noise),
        }

    def generate_samples(self, job_id: Optional[str] = None) -> str:
        self.logger.info("Generating %d RGC deuteron samples", self.num_samples)

        signals: List[np.ndarray] = []
        p_values: List[float] = []
        cc_values: List[float] = []
        areas: List[float] = []
        snrs: List[Optional[float]] = []
        meta_rows: List[dict[str, float]] = []

        for i in tqdm.tqdm(range(self.num_samples), desc="Generating RGC samples"):
            sample = self.generate_one()
            params = sample["params"]
            signals.append(sample["signal"])
            p_values.append(float(params["P"]))
            cc_values.append(float(params["cc"]))
            areas.append(sample["area"])
            snrs.append(sample["snr"])
            meta_rows.append(
                {
                    key: params[key]
                    for key in (*DULYA_SAMPLE_KEYS, *BASELINE_SAMPLE_KEYS)
                    if key not in ("P", "cc")
                }
            )
            if (i + 1) % 10000 == 0:
                self.logger.info("Generated %d/%d samples", i + 1, self.num_samples)

        columns = self._build_columns(signals, p_values, cc_values, snrs, areas, meta_rows)
        return self._persist(columns, job_id)

    def _build_columns(
        self,
        signals: Sequence[np.ndarray],
        p_values: Sequence[float],
        cc_values: Sequence[float],
        snrs: Sequence[Optional[float]],
        areas: Sequence[float],
        meta_rows: Sequence[dict[str, float]],
    ) -> dict[str, np.ndarray]:
        sig = np.asarray(signals, dtype=np.float64)
        columns: dict[str, np.ndarray] = {
            str(i): sig[:, i] for i in range(sig.shape[1])
        }
        columns["P"] = np.asarray(p_values, dtype=np.float64)
        columns["cc"] = np.asarray(cc_values, dtype=np.float64)
        columns["SNR"] = np.asarray(
            [np.nan if s is None else float(s) for s in snrs], dtype=np.float64
        )
        columns["Area"] = np.asarray(areas, dtype=np.float64)
        for key in meta_rows[0]:
            columns[key] = np.asarray([row[key] for row in meta_rows], dtype=np.float64)
        return columns

    def _persist(self, columns: dict[str, np.ndarray], job_id: Optional[str]) -> str:
        filename = "Sample_vector"
        if job_id is not None:
            filename += f"_{job_id}"
        filename += ".parquet"
        file_path = os.path.join(self.output_dir, filename)

        try:
            import pandas as pd

            df = pd.DataFrame(columns)
            df.to_parquet(file_path, compression="snappy", index=False)
        except ImportError:
            import pyarrow as pa
            import pyarrow.parquet as pq

            table = pa.Table.from_pydict(columns)
            pq.write_table(table, file_path, compression="snappy")

        self.logger.info("Wrote %s (%d samples)", file_path, len(next(iter(columns.values()))))
        return file_path
