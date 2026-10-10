"""GPU energy, wall-clock, and token metering.

Ollama runs as a separate server process from whatever Python script issues
the calls, so there is no simple before/after energy reading tied to a single
process. Instead this polls nvidia-ml-py's instantaneous power reading on a
background thread at a fixed interval across the timed window and integrates
power over time into watt-hours. This assumes no other process is using the
GPU concurrently during a timed run, reasonable on a single-user machine but
worth stating as a caveat in the paper.

Package note: `nvidia-ml-py` (not the deprecated `pynvml` package) installs
under the `pynvml` module name; that is not a typo.
"""

import logging
import threading
import time
from dataclasses import dataclass, field

import pynvml

logger = logging.getLogger(__name__)


@dataclass
class MeterResult:
    wall_clock_s: float
    gpu_energy_wh: float
    gpu_power_samples: int
    tokens_in: int = 0
    tokens_out: int = 0


class GPUEnergyMeter:
    """Background-thread poller. One instance per GPU device index.

    Usage:
        meter = GPUEnergyMeter()
        with meter.measure() as m:
            ... do the timed work ...
        m.gpu_energy_wh, m.wall_clock_s  # filled in after the `with` block exits
    """

    def __init__(self, device_index: int = 0, poll_interval_s: float = 0.1):
        self.device_index = device_index
        self.poll_interval_s = poll_interval_s
        self._handle = None

    def _ensure_init(self):
        if self._handle is None:
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)

    def measure(self) -> "_MeasurementContext":
        self._ensure_init()
        return _MeasurementContext(self._handle, self.poll_interval_s)


class _MeasurementContext:
    def __init__(self, handle, poll_interval_s: float):
        self._handle = handle
        self._poll_interval_s = poll_interval_s
        self._stop_event = threading.Event()
        self._thread = None
        self._energy_wh = 0.0
        self._sample_count = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.wall_clock_s = 0.0
        self.gpu_energy_wh = 0.0
        self.gpu_power_samples = 0

    def _poll_loop(self):
        last_t = time.monotonic()
        while not self._stop_event.is_set():
            power_w = pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0  # mW -> W
            now = time.monotonic()
            dt_h = (now - last_t) / 3600.0
            self._energy_wh += power_w * dt_h
            self._sample_count += 1
            last_t = now
            self._stop_event.wait(self._poll_interval_s)

    def __enter__(self) -> "_MeasurementContext":
        self._start_wall = time.monotonic()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop_event.set()
        self._thread.join(timeout=self._poll_interval_s * 5)
        self.wall_clock_s = time.monotonic() - self._start_wall
        self.gpu_energy_wh = self._energy_wh
        self.gpu_power_samples = self._sample_count
        return False

    def result(self) -> MeterResult:
        return MeterResult(
            wall_clock_s=self.wall_clock_s,
            gpu_energy_wh=self.gpu_energy_wh,
            gpu_power_samples=self.gpu_power_samples,
            tokens_in=self.tokens_in,
            tokens_out=self.tokens_out,
        )
