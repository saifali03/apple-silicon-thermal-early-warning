#!/usr/bin/env python3
import argparse
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime

STOP = False
CHILD = None


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def handle_signal(signum, frame):
    global STOP, CHILD
    STOP = True
    log(f"Received signal {signum}; stopping...")
    if CHILD and CHILD.poll() is None:
        try:
            CHILD.terminate()
            CHILD.wait(timeout=10)
        except Exception:
            try:
                CHILD.kill()
            except Exception:
                pass


def have_stress_ng() -> str | None:
    return shutil.which('stress-ng')


def _fallback_vm_megabytes(vm_bytes: str, default: int = 256, minimum: int = 64) -> int:
    """Parses e.g. '256M' -> 256 for the no-stress-ng memory-touch fallback.
    Falls back to `default` on any unrecognized/invalid format instead of raising."""
    if vm_bytes.endswith('M'):
        try:
            return max(minimum, int(vm_bytes[:-1]))
        except ValueError:
            pass
    return default


def build_stress_cmd(mode: str, cpu_workers: int, cpu_load: int, vm_workers: int, vm_bytes: str, duration_s: int):
    stress_ng = have_stress_ng()
    if stress_ng:
        base = [stress_ng, '--timeout', f'{duration_s}s', '--metrics-brief']
        if mode == 'cpu':
            return base + ['--cpu', str(cpu_workers), '--cpu-load', str(cpu_load)]
        if mode == 'cpu-vm':
            return base + ['--cpu', str(cpu_workers), '--cpu-load', str(cpu_load), '--vm', str(vm_workers), '--vm-bytes', vm_bytes]
        if mode == 'matrix':
            return base + ['--cpu', str(cpu_workers), '--cpu-method', 'matrixprod', '--cpu-load', str(cpu_load)]

    # No stress-ng: fall back to spawning `yes` processes for CPU load, plus a
    # small Python one-liner that allocates memory when a vm/matrix mode was requested.
    n = max(1, cpu_workers)
    parts = ['yes > /dev/null &' for _ in range(n)]
    if mode in ('cpu-vm', 'matrix'):
        mb = _fallback_vm_megabytes(vm_bytes)
        touch_mem = f'python3 -c "import time; x=[b\'x\' * 1024 * 1024 for _ in range({mb})]; time.sleep({duration_s})" &'
        parts.append(touch_mem)
    cmd = 'trap "kill 0" EXIT; ' + ' '.join(parts) + f' sleep {duration_s}'
    return ['/bin/bash', '-lc', cmd]


def run_once(cmd):
    global CHILD
    log('Starting stress block: ' + ' '.join(cmd))
    CHILD = subprocess.Popen(cmd)
    rc = None
    try:
        rc = CHILD.wait()
    finally:
        CHILD = None
    log(f'Stress block finished with exit code {rc}')


def main():
    parser = argparse.ArgumentParser(description='Automated intermittent stress runner for long telemetry sessions on macOS.')
    parser.add_argument('--hours', type=float, default=4.0, help='Total runtime in hours (default: 4)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--mode', choices=['cpu', 'cpu-vm', 'matrix'], default='cpu', help='Stress style')
    parser.add_argument('--cpu-workers', type=int, default=max(1, (os.cpu_count() or 4) // 3), help='CPU workers per burst')
    parser.add_argument('--cpu-load', type=int, default=65, help='Target CPU load percent for stress-ng (default: 65)')
    parser.add_argument('--vm-workers', type=int, default=1, help='VM workers when mode=cpu-vm')
    parser.add_argument('--vm-bytes', type=str, default='256M', help='Memory to touch when mode=cpu-vm, e.g. 256M')
    parser.add_argument('--burst-min-sec', type=int, default=120, help='Minimum stress burst length in seconds (default: 120)')
    parser.add_argument('--burst-max-sec', type=int, default=180, help='Maximum stress burst length in seconds (default: 180)')
    parser.add_argument('--cool-min-sec', type=int, default=120, help='Minimum cool-down after a burst in seconds (default: 120)')
    parser.add_argument('--cool-max-sec', type=int, default=180, help='Maximum cool-down after a burst in seconds (default: 180)')
    parser.add_argument('--gap-min-sec', type=int, default=1800, help='Minimum time between block starts in seconds (default: 1800 = 30 min)')
    parser.add_argument('--gap-max-sec', type=int, default=3600, help='Maximum time between block starts in seconds (default: 3600 = 60 min)')
    parser.add_argument('--cycles-min', type=int, default=2, help='Minimum bursts per block (default: 2)')
    parser.add_argument('--cycles-max', type=int, default=4, help='Maximum bursts per block (default: 4)')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if args.burst_min_sec > args.burst_max_sec or args.cool_min_sec > args.cool_max_sec or args.gap_min_sec > args.gap_max_sec or args.cycles_min > args.cycles_max:
        sys.exit('Invalid min/max argument pair.')

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    total_seconds = int(args.hours * 3600)
    start = time.time()
    end = start + total_seconds

    if have_stress_ng():
        log('stress-ng detected; will use it for bursts.')
    else:
        log('stress-ng not found; using built-in fallback (yes + optional Python memory touch).')

    log(f'Run window: {args.hours:.2f} hours')
    next_block = start

    while not STOP and time.time() < end:
        now = time.time()
        if now < next_block:
            sleep_for = min(30, next_block - now, end - now)
            if sleep_for > 0:
                time.sleep(sleep_for)
            continue

        cycles = random.randint(args.cycles_min, args.cycles_max)
        log(f'Starting stress block with {cycles} burst(s).')

        for i in range(cycles):
            if STOP or time.time() >= end:
                break
            burst = random.randint(args.burst_min_sec, args.burst_max_sec)
            cmd = build_stress_cmd(args.mode, args.cpu_workers, args.cpu_load, args.vm_workers, args.vm_bytes, burst)
            run_once(cmd)
            if i != cycles - 1 and not STOP and time.time() < end:
                cool = random.randint(args.cool_min_sec, args.cool_max_sec)
                log(f'Cooling down for {cool} seconds.')
                until = time.time() + cool
                while not STOP and time.time() < until and time.time() < end:
                    time.sleep(min(5, until - time.time(), end - time.time()))

        if STOP or time.time() >= end:
            break
        gap = random.randint(args.gap_min_sec, args.gap_max_sec)
        next_block = time.time() + gap
        log(f'Next block scheduled in {gap//60}m {gap%60}s.')

    log('Finished intermittent stress run.')


if __name__ == '__main__':
    main()
