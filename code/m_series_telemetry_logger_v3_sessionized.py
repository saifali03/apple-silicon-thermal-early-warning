#!/usr/bin/env python3
"""
Telemetry logger for Apple Silicon (M-series) Macs, with best-effort Linux support.

Adjusted for project data management:
- Removes requested columns from the output schema.
- Uses file naming convention: {machine_id}_{date}_{run_number}.csv
- Writes data under a structured folder tree.
- Maintains a session_registry.csv with one row per recording session.

Platform behavior:
- macOS (primary target): full telemetry via `powermetrics` (P/E-cluster
  activity/frequency, GPU, CPU/GPU/ANE power) plus optional `macmon` for
  die temperature.
- Linux (best-effort): CPU temperature via psutil sensors, CPU package power
  via the Linux RAPL/powercap energy counters (may require root depending on
  distro), GPU stats via `nvidia-smi` if present. Apple's P/E-cluster split
  has no generic Linux equivalent, so those columns are left NaN.

Suggested usage (run from the repository root):
sudo python3 code/m_series_telemetry_logger_v3_sessionized.py \
  --interval 5 \
  --machine-id mac_m5 \
  --base-dir data/raw
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import platform
import plistlib
import re
import select
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import psutil

POWERMETRICS_BIN = "/usr/bin/powermetrics"
PLIST_DELIMITER = b"\x00"
MACOS_PAGE_SIZE_BYTES = 16384

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("m_series_telemetry")

CSV_COLUMNS = [
    "timestamp_utc",
    "os_family",
    "thermal_pressure_level",
    "thermal_pressure_code",
    "cpu_die_temp_c",
    "gpu_die_temp_c",
    "cpu_total_active_pct",
    "cpu_ecluster_active_pct",
    "cpu_pcluster_active_pct",
    "cpu_ecluster_freq_mhz",
    "cpu_pcluster_freq_mhz",
    "gpu_active_pct",
    "gpu_freq_mhz",
    "cpu_power_mw",
    "gpu_power_mw",
    "ane_power_mw",
    "combined_power_mw",
    "cpu_percent_psutil",
    "loadavg_1m",
    "loadavg_5m",
    "loadavg_15m",
    "ram_total_gb",
    "ram_used_gb",
    "ram_available_gb",
    "ram_percent",
    "mem_pressure_pct",
    "swap_total_gb",
    "swap_used_gb",
    "swap_percent",
    "mem_compressed_gb",
    "battery_percent",
]

REGISTRY_COLUMNS = [
    "session_id",
    "machine_id",
    "date",
    "run_number",
    "file_path",
    "file_name",
    "start_utc",
    "end_utc",
    "duration_seconds",
    "n_rows",
    "interval_seconds",
    "logger_version",
    "os_family",
    "notes",
]

THERMAL_PRESSURE_ORDINAL = {
    "nominal": 0,
    "fair": 1,
    "moderate": 1,
    "serious": 2,
    "heavy": 2,
    "critical": 3,
    "trapping": 3,
    "sleeping": -1,
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat()


def nan() -> float:
    return float("nan")


def safe_get(d: Any, *path: Any, default: Any = None) -> Any:
    cur = d
    for key in path:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        elif isinstance(cur, (list, tuple)) and isinstance(key, int) and -len(cur) <= key < len(cur):
            cur = cur[key]
        else:
            return default
    return cur


def sanitize_machine_id(machine_id: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", machine_id.strip()).strip("_").lower()
    if not cleaned:
        raise ValueError("machine_id must contain at least one alphanumeric character")
    return cleaned


def current_utc_date_str() -> str:
    return utc_now().strftime("%Y-%m-%d")


def current_utc_date_compact() -> str:
    return utc_now().strftime("%Y%m%d")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def init_csv_if_missing(path: Path, columns: list[str]) -> None:
    ensure_parent(path)
    if not path.exists():
        with path.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=columns).writeheader()


def count_existing_runs(day_dir: Path, machine_id: str, date_compact: str) -> int:
    pattern = f"{machine_id}_{date_compact}_*.csv"
    return len(list(day_dir.glob(pattern)))


def resolve_paths(base_dir: Path, registry_path: Optional[Path], machine_id: str) -> dict[str, Path]:
    date_folder = current_utc_date_str()
    date_compact = current_utc_date_compact()
    day_dir = base_dir / machine_id / date_folder
    day_dir.mkdir(parents=True, exist_ok=True)

    run_number = count_existing_runs(day_dir, machine_id, date_compact) + 1
    file_name = f"{machine_id}_{date_compact}_{run_number:03d}.csv"
    data_path = day_dir / file_name

    if registry_path is None:
        registry_path = base_dir.parent / "interim" / "session_registry.csv"

    return {
        "day_dir": day_dir,
        "data_path": data_path,
        "registry_path": registry_path,
        "date_folder": Path(date_folder),
        "run_number": Path(f"{run_number:03d}"),
    }


def combine_power(*values: float) -> float:
    present = [v for v in values if not math.isnan(v)]
    return sum(present) if present else nan()


# ---------------------------------------------------------------------------
# macOS backend — powermetrics
# ---------------------------------------------------------------------------

class PowermetricsStream:
    SAMPLERS = "cpu_power,gpu_power,thermal"

    def __init__(self, interval_ms: int = 5000):
        self.interval_ms = interval_ms
        self.proc: Optional[subprocess.Popen] = None
        self._buffer = b""

    def _command(self) -> list[str]:
        base = [
            POWERMETRICS_BIN,
            "--samplers", self.SAMPLERS,
            "-i", str(self.interval_ms),
            "--format", "plist",
        ]
        return base if os.geteuid() == 0 else ["/usr/bin/sudo", "-n"] + base

    def start(self) -> None:
        if not Path(POWERMETRICS_BIN).exists():
            raise FileNotFoundError(f"{POWERMETRICS_BIN} not found")
        cmd = self._command()
        log.info("Launching powermetrics: %s", " ".join(cmd))
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self._buffer = b""

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def read_sample(self, timeout: float) -> Optional[dict]:
        if self.proc is None or self.proc.stdout is None:
            return None
        deadline = time.monotonic() + timeout
        fd = self.proc.stdout.fileno()

        while PLIST_DELIMITER not in self._buffer:
            if self.proc.poll() is not None:
                stderr = self.proc.stderr.read().decode("utf-8", "ignore") if self.proc.stderr else ""
                log.error("powermetrics exited (code %s): %s", self.proc.returncode, stderr.strip())
                return None
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                log.warning("Timed out waiting for a powermetrics sample")
                return None
            ready, _, _ = select.select([fd], [], [], min(remaining, 1.0))
            if not ready:
                continue
            chunk = os.read(fd, 65536)
            if not chunk:
                return None
            self._buffer += chunk

        raw, _, self._buffer = self._buffer.partition(PLIST_DELIMITER)
        raw = raw.strip(b"\x00")
        if not raw:
            return None
        try:
            return plistlib.loads(raw)
        except Exception as exc:
            log.warning("Failed to parse powermetrics plist sample (%d bytes): %s", len(raw), exc)
            return None


def encode_thermal_pressure(raw: Optional[str]) -> tuple[Optional[str], Optional[int]]:
    if not raw:
        return None, None
    return raw, THERMAL_PRESSURE_ORDINAL.get(raw.strip().lower())


def parse_cpu_clusters(processor: dict) -> dict:
    buckets: dict[str, dict[str, list[float]]] = {
        "E": {"active": [], "freq": []},
        "P": {"active": [], "freq": []},
    }
    clusters = safe_get(processor, "clusters", default=[]) or []
    for cluster in clusters:
        name = safe_get(cluster, "name", default="") or ""
        family = "E" if name.startswith("E") else "P" if name.startswith("P") else None
        if family is None:
            continue
        idle_ratio = safe_get(cluster, "idle_ratio")
        freq_hz = safe_get(cluster, "freq_hz")
        if idle_ratio is not None:
            buckets[family]["active"].append((1.0 - idle_ratio) * 100.0)
        if freq_hz is not None:
            buckets[family]["freq"].append(freq_hz / 1e6)

    def avg(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else nan()

    all_active = buckets["E"]["active"] + buckets["P"]["active"]
    return {
        "cpu_total_active_pct": avg(all_active),
        "cpu_ecluster_active_pct": avg(buckets["E"]["active"]),
        "cpu_pcluster_active_pct": avg(buckets["P"]["active"]),
        "cpu_ecluster_freq_mhz": avg(buckets["E"]["freq"]),
        "cpu_pcluster_freq_mhz": avg(buckets["P"]["freq"]),
    }


def parse_gpu(sample: dict) -> dict:
    gpu = safe_get(sample, "gpu", default={}) or {}
    idle = safe_get(gpu, "idle_ratio")
    freq_hz = safe_get(gpu, "freq_hz")
    return {
        "gpu_active_pct": (1.0 - idle) * 100.0 if idle is not None else nan(),
        "gpu_freq_mhz": freq_hz / 1e6 if freq_hz is not None else nan(),
    }


def parse_power(sample: dict, processor: dict, interval_s: float) -> dict:
    def resolve(power_key: str, energy_key: str) -> float:
        direct = safe_get(sample, power_key, default=safe_get(processor, power_key))
        if direct is not None:
            return float(direct)
        energy_mj = safe_get(processor, energy_key)
        if energy_mj is not None and interval_s > 0:
            return float(energy_mj) / interval_s
        return nan()

    combined = safe_get(sample, "combined_power", default=safe_get(processor, "combined_power"))
    return {
        "cpu_power_mw": resolve("cpu_power", "cpu_energy"),
        "gpu_power_mw": resolve("gpu_power", "gpu_energy"),
        "ane_power_mw": resolve("ane_power", "ane_energy"),
        "combined_power_mw": float(combined) if combined is not None else nan(),
    }


def read_optional_numeric_temperature() -> dict:
    result = {"cpu_die_temp_c": nan(), "gpu_die_temp_c": nan()}
    macmon_bin = shutil.which("macmon")
    if not macmon_bin:
        return result
    try:
        out = subprocess.run([macmon_bin, "pipe", "-s", "1"], capture_output=True, timeout=3, text=True)
        data = json.loads(out.stdout.strip().splitlines()[-1])
        result["cpu_die_temp_c"] = float(safe_get(data, "temp", "cpu_temp_avg", default=nan()))
        result["gpu_die_temp_c"] = float(safe_get(data, "temp", "gpu_temp_avg", default=nan()))
    except Exception as exc:
        log.debug("Optional macmon temperature read failed (non-fatal): %s", exc)
    return result


class MacTelemetryCollector:
    """Wraps powermetrics; provides Apple Silicon P/E-cluster, GPU, and power detail."""

    os_family = "macos"

    def __init__(self, interval_ms: int):
        self._stream = PowermetricsStream(interval_ms=interval_ms)

    def start(self) -> None:
        self._stream.start()

    def stop(self) -> None:
        self._stream.stop()

    def wait_for_sample(self, timeout: float) -> Optional[dict]:
        return self._stream.read_sample(timeout=timeout)

    def is_alive(self) -> bool:
        return self._stream.proc is not None and self._stream.proc.poll() is None

    def restart(self) -> None:
        self._stream.start()

    def parse_sample(self, sample: Optional[dict], interval_s: float, cpu_pct: float) -> dict:
        sample = sample or {}
        processor = safe_get(sample, "processor", default={}) or {}
        level, code = encode_thermal_pressure(safe_get(sample, "thermal_pressure"))
        out = {"thermal_pressure_level": level, "thermal_pressure_code": code}
        out.update(parse_cpu_clusters(processor))
        out.update(parse_gpu(sample))
        out.update(parse_power(sample, processor, interval_s))
        out.update(read_optional_numeric_temperature())
        return out


# ---------------------------------------------------------------------------
# Linux backend — psutil sensors + RAPL/powercap + optional nvidia-smi
# ---------------------------------------------------------------------------

_LINUX_CPU_TEMP_LABELS = ("coretemp", "k10temp", "cpu_thermal", "acpitz")


def _cpu_temp_entries(temps: dict) -> list:
    for label in _LINUX_CPU_TEMP_LABELS:
        entries = temps.get(label)
        if entries:
            return entries
    return []


def read_linux_temperature_and_pressure(temps: dict) -> tuple[float, Optional[str], Optional[int]]:
    """Reads CPU die temperature and derives a synthetic nominal/fair/serious/critical
    pressure level from the same sensor's high/critical trip points, if the kernel
    reports them. There is no direct Linux equivalent of macOS thermal_pressure."""
    entries = _cpu_temp_entries(temps)
    currents = [e.current for e in entries if e.current is not None]
    if not currents:
        return nan(), None, None

    current = sum(currents) / len(currents)
    highs = [e.high for e in entries if e.high]
    crits = [e.critical for e in entries if e.critical]
    high = min(highs) if highs else None
    critical = min(crits) if crits else None

    if critical is not None and current >= critical:
        level, code = "critical", 3
    elif high is not None and current >= high:
        level, code = "serious", 2
    elif high is not None and current >= 0.85 * high:
        level, code = "fair", 1
    else:
        level, code = "nominal", 0
    return current, level, code


def read_linux_gpu() -> dict:
    """Best-effort NVIDIA GPU stats via nvidia-smi. NaN on any other GPU vendor."""
    result = {"gpu_active_pct": nan(), "gpu_freq_mhz": nan(), "gpu_die_temp_c": nan(), "gpu_power_mw": nan()}
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return result
    try:
        out = subprocess.run(
            [nvidia_smi, "--query-gpu=utilization.gpu,clocks.sm,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, timeout=3, text=True, check=True,
        ).stdout.strip().splitlines()[0]
        util, clock, temp, power = (p.strip() for p in out.split(","))
        result.update(
            gpu_active_pct=float(util),
            gpu_freq_mhz=float(clock),
            gpu_die_temp_c=float(temp),
            gpu_power_mw=float(power) * 1000.0,
        )
    except Exception as exc:
        log.debug("nvidia-smi read failed (non-fatal): %s", exc)
    return result


def _find_rapl_energy_path() -> Optional[Path]:
    base = Path("/sys/class/powercap")
    if not base.exists():
        return None
    for zone in sorted(base.glob("*/energy_uj")):
        return zone
    return None


class LinuxPowerReader:
    """Best-effort CPU package power (mW) from Linux RAPL/powercap energy counters."""

    # RAPL counters occasionally reset outside of normal overflow (e.g. package
    # C-state transitions), which the overflow-recovery math below would otherwise
    # misread as a multi-kilowatt spike. No consumer/prosumer CPU package sustains
    # anywhere near this, so treat readings above it as a bad sample (NaN) instead.
    _SANITY_MAX_POWER_MW = 500_000  # 500 W

    def __init__(self):
        self._energy_path = _find_rapl_energy_path()
        self._max_energy_uj = (
            self._read_int(self._energy_path.parent / "max_energy_range_uj") if self._energy_path else None
        )
        self._prev_uj: Optional[int] = None
        self._prev_t: Optional[float] = None
        if self._energy_path:
            log.info("Linux CPU power source: %s", self._energy_path)
        else:
            log.warning("No RAPL/powercap energy counter found — cpu_power_mw will be NaN.")

    @staticmethod
    def _read_int(path: Path) -> Optional[int]:
        try:
            return int(path.read_text().strip())
        except Exception:
            return None

    def read_cpu_power_mw(self) -> float:
        if self._energy_path is None:
            return nan()
        uj = self._read_int(self._energy_path)
        now = time.monotonic()
        power_mw = nan()
        if uj is not None and self._prev_uj is not None:
            dt = now - self._prev_t
            duj = uj - self._prev_uj
            if duj < 0 and self._max_energy_uj:  # counter wrapped
                duj += self._max_energy_uj
            if dt > 0 and duj >= 0:
                candidate = duj / dt / 1000.0  # uJ/s == uW; /1000 -> mW
                if candidate <= self._SANITY_MAX_POWER_MW:
                    power_mw = candidate
                else:
                    log.debug("Discarding implausible RAPL power reading: %.0f mW", candidate)
        if uj is not None:
            self._prev_uj = uj
            self._prev_t = now
        return power_mw


class LinuxTelemetryCollector:
    """Direct psutil/sysfs polling. No subprocess stream is needed, so pacing
    happens via a plain sleep instead of a blocking read like the mac backend."""

    os_family = "linux"

    def __init__(self, interval_s: float):
        self.interval_s = interval_s
        self._power = LinuxPowerReader()

    def start(self) -> None:
        if os.geteuid() != 0:
            log.warning("Not running as root — some Linux sensors (RAPL power) may be unreadable.")

    def stop(self) -> None:
        pass

    def wait_for_sample(self, timeout: float) -> dict:
        time.sleep(self.interval_s)
        return {}

    def is_alive(self) -> bool:
        return True

    def restart(self) -> None:
        pass

    def parse_sample(self, sample: dict, interval_s: float, cpu_pct: float) -> dict:
        try:
            temps = psutil.sensors_temperatures()
        except Exception:
            temps = {}
        cpu_temp, level, code = read_linux_temperature_and_pressure(temps)
        gpu = read_linux_gpu()
        cpu_power = self._power.read_cpu_power_mw()
        return {
            "thermal_pressure_level": level,
            "thermal_pressure_code": code,
            "cpu_die_temp_c": cpu_temp,
            "gpu_die_temp_c": gpu["gpu_die_temp_c"],
            "cpu_total_active_pct": cpu_pct,
            "cpu_ecluster_active_pct": nan(),
            "cpu_pcluster_active_pct": nan(),
            "cpu_ecluster_freq_mhz": nan(),
            "cpu_pcluster_freq_mhz": nan(),
            "gpu_active_pct": gpu["gpu_active_pct"],
            "gpu_freq_mhz": gpu["gpu_freq_mhz"],
            "cpu_power_mw": cpu_power,
            "gpu_power_mw": gpu["gpu_power_mw"],
            "ane_power_mw": nan(),
            "combined_power_mw": combine_power(cpu_power, gpu["gpu_power_mw"]),
        }


# ---------------------------------------------------------------------------
# Shared metrics (identical on both platforms)
# ---------------------------------------------------------------------------

def read_memory_metrics() -> dict:
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    mem_pressure_pct = (1.0 - vm.available / vm.total) * 100.0 if vm.total else nan()

    mem_compressed_gb = nan()
    if platform.system() == "Darwin":
        try:
            vmstat_out = subprocess.run(["/usr/bin/vm_stat"], capture_output=True, timeout=3, text=True, check=True).stdout
            match = re.search(r"Pages occupied by compressor:\s+(\d+)", vmstat_out)
            if match:
                mem_compressed_gb = (int(match.group(1)) * MACOS_PAGE_SIZE_BYTES) / (1024 ** 3)
        except Exception as exc:
            log.debug("vm_stat compressor read failed (non-fatal): %s", exc)
    else:
        try:
            match = re.search(r"^Zswapped:\s+(\d+)\s+kB", Path("/proc/meminfo").read_text(), re.MULTILINE)
            if match:
                mem_compressed_gb = int(match.group(1)) / (1024 ** 2)
        except Exception as exc:
            log.debug("/proc/meminfo zswap read failed (non-fatal): %s", exc)

    return {
        "ram_total_gb": vm.total / (1024 ** 3),
        "ram_used_gb": vm.used / (1024 ** 3),
        "ram_available_gb": vm.available / (1024 ** 3),
        "ram_percent": vm.percent,
        "mem_pressure_pct": mem_pressure_pct,
        "swap_total_gb": swap.total / (1024 ** 3),
        "swap_used_gb": swap.used / (1024 ** 3),
        "swap_percent": swap.percent,
        "mem_compressed_gb": mem_compressed_gb,
    }


def read_load_averages() -> dict:
    try:
        load1, load5, load15 = os.getloadavg()
        return {
            "loadavg_1m": float(load1),
            "loadavg_5m": float(load5),
            "loadavg_15m": float(load15),
        }
    except (AttributeError, OSError):
        return {"loadavg_1m": nan(), "loadavg_5m": nan(), "loadavg_15m": nan()}


def read_battery_pct() -> float:
    try:
        batt = psutil.sensors_battery()
        return float(batt.percent) if batt else nan()
    except Exception:
        return nan()


def check_environment() -> str:
    system = platform.system()
    if system == "Darwin":
        if platform.machine() != "arm64":
            log.warning("platform.machine() = '%s', not 'arm64'. This script targets Apple Silicon.", platform.machine())
        if os.geteuid() != 0 and shutil.which("sudo") is None:
            sys.exit("powermetrics requires root and `sudo` was not found on PATH.")
        return "darwin"
    if system == "Linux":
        if os.geteuid() != 0:
            log.warning("Not running as root — CPU power (RAPL) readings may be unavailable on some systems.")
        return "linux"
    sys.exit(f"Unsupported platform: {system}. This script supports macOS and Linux.")


def append_row(path: Path, row: dict) -> None:
    with path.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_COLUMNS).writerow({col: row.get(col, nan()) for col in CSV_COLUMNS})


def collect_one_row(collector: Any, sample: Optional[dict], interval_s: float) -> dict:
    row: dict[str, Any] = {
        "timestamp_utc": utc_now_iso(),
        "os_family": collector.os_family,
    }
    cpu_pct = psutil.cpu_percent(interval=None)
    row.update(collector.parse_sample(sample, interval_s, cpu_pct))
    row.update(read_memory_metrics())
    row.update(read_load_averages())
    row["cpu_percent_psutil"] = cpu_pct
    row["battery_percent"] = read_battery_pct()
    return row


def append_registry_row(registry_path: Path, row: dict) -> None:
    init_csv_if_missing(registry_path, REGISTRY_COLUMNS)
    with registry_path.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=REGISTRY_COLUMNS).writerow(row)


def update_registry_end(
    registry_path: Path,
    session_id: str,
    end_utc: str,
    duration_seconds: float,
    n_rows: int,
) -> None:
    if not registry_path.exists():
        log.warning("Registry file not found at %s — skipping end-of-session update.", registry_path)
        return

    with registry_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        # Guard 1: fieldnames must exist and be valid
        if reader.fieldnames is None:
            log.warning("Registry file has no header — skipping update.")
            return
        rows = [row for row in reader if any(row.values())]  # Guard 2: skip blank/None rows

    updated = False
    for row in rows:
        if row.get("session_id") == session_id:
            row["end_utc"]          = end_utc
            row["duration_seconds"] = f"{duration_seconds:.3f}"
            row["n_rows"]           = str(n_rows)
            updated = True
            break

    if not updated:
        log.warning("Session ID '%s' not found in registry — end metadata not written.", session_id)
        return

    with registry_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REGISTRY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    log.info("Registry updated for session %s.", session_id)


def _display(val: float) -> float:
    """Substitutes -1 for NaN so the log line stays readable with %f formats."""
    return -1 if isinstance(val, float) and math.isnan(val) else val


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--interval", type=float, default=5.0, help="Polling interval in seconds (default: 5)")
    parser.add_argument("--machine-id", required=True, help="Machine identifier used in filenames, e.g. mac_m5")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("data/raw"),
        help="Base raw-data directory. Data stored as base_dir/machine_id/YYYY-MM-DD/{machine_id}_{date}_{run_number}.csv",
    )
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=None,
        help="Optional explicit path for session_registry.csv. Default: sibling interim/session_registry.csv",
    )
    parser.add_argument("--notes", type=str, default="", help="Optional free-text note stored in session_registry.csv")
    parser.add_argument("--dump-raw-sample", action="store_true", help="Print one raw/parsed sample as JSON and exit")
    args = parser.parse_args()

    platform_name = check_environment()
    machine_id = sanitize_machine_id(args.machine_id)
    resolved = resolve_paths(args.base_dir, args.registry_path, machine_id)
    data_path = resolved["data_path"]
    registry_path = resolved["registry_path"]
    date_folder = str(resolved["date_folder"])
    run_number = str(resolved["run_number"])
    date_compact = current_utc_date_compact()
    session_id = f"{machine_id}_{date_compact}_{run_number}"

    init_csv_if_missing(data_path, CSV_COLUMNS)
    init_csv_if_missing(registry_path, REGISTRY_COLUMNS)

    collector = (
        MacTelemetryCollector(interval_ms=int(args.interval * 1000))
        if platform_name == "darwin"
        else LinuxTelemetryCollector(interval_s=args.interval)
    )
    collector.start()

    if args.dump_raw_sample:
        sample = collector.wait_for_sample(timeout=args.interval + 10.0)
        payload = sample if platform_name == "darwin" else collector.parse_sample(sample, args.interval, psutil.cpu_percent(interval=None))
        collector.stop()
        print(json.dumps(payload, indent=2, default=str))
        return

    psutil.cpu_percent(interval=None)
    start_dt = utc_now()
    start_utc = start_dt.isoformat()

    registry_row = {
        "session_id": session_id,
        "machine_id": machine_id,
        "date": date_folder,
        "run_number": run_number,
        "file_path": str(data_path.resolve()),
        "file_name": data_path.name,
        "start_utc": start_utc,
        "end_utc": "",
        "duration_seconds": "",
        "n_rows": "0",
        "interval_seconds": str(args.interval),
        "logger_version": "v3_sessionized",
        "os_family": collector.os_family,
        "notes": args.notes,
    }
    append_registry_row(registry_path, registry_row)

    stop_requested = False
    rows_written = 0

    def _handle_signal(signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        log.info("Received signal %s — shutting down cleanly...", signum)
        stop_requested = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    log.info("Logging to: %s", data_path.resolve())
    log.info("Registry: %s", registry_path.resolve())
    log.info("Session ID: %s", session_id)

    try:
        while not stop_requested:
            sample = collector.wait_for_sample(timeout=args.interval + 5.0)
            row = collect_one_row(collector, sample, interval_s=args.interval)
            append_row(data_path, row)
            rows_written += 1

            log.info(
                "logged | thermal=%-8s cpu_active=%5.1f%% p_cores=%5.1f%% combined_power=%6.0fmW ram=%4.1f%% loadavg_5m=%.2f",
                row["thermal_pressure_level"] or "n/a",
                row["cpu_total_active_pct"],
                _display(row["cpu_pcluster_active_pct"]),
                _display(row.get("combined_power_mw", nan())),
                row["ram_percent"],
                _display(row.get("loadavg_5m", nan())),
            )

            if sample is None and not collector.is_alive():
                log.warning("Telemetry source died — restarting.")
                collector.restart()
    finally:
        collector.stop()
        end_dt = utc_now()
        duration_seconds = (end_dt - start_dt).total_seconds()
        update_registry_end(
            registry_path=registry_path,
            session_id=session_id,
            end_utc=end_dt.isoformat(),
            duration_seconds=duration_seconds,
            n_rows=rows_written,
        )
        log.info("Done. Data written to: %s", data_path.resolve())
        log.info("Rows written: %d", rows_written)


if __name__ == "__main__":
    main()
