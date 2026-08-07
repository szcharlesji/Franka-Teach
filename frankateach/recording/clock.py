"""NTP-style monotonic-clock fitting used for camera and robot alignment."""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ClockSample:
    local_send_ns: int
    remote_recv_ns: int
    remote_send_ns: int
    local_recv_ns: int

    @property
    def rtt_ns(self):
        remote_work = max(0, self.remote_send_ns - self.remote_recv_ns)
        return max(0, self.local_recv_ns - self.local_send_ns - remote_work)

    @property
    def remote_mid_ns(self):
        return (self.remote_recv_ns + self.remote_send_ns) / 2.0

    @property
    def local_mid_ns(self):
        return (self.local_send_ns + self.local_recv_ns) / 2.0

    def to_dict(self):
        out = asdict(self)
        out["rtt_ns"] = self.rtt_ns
        return out


@dataclass(frozen=True)
class ClockFit:
    """Affine map from a remote monotonic clock to Discovery monotonic ns."""

    scale: float
    remote_origin_ns: int
    local_origin_ns: int
    uncertainty_ns: int
    min_rtt_ns: int
    sample_count: int

    def map_ns(self, remote_ns):
        return int(round(self.local_origin_ns + self.scale * (remote_ns - self.remote_origin_ns)))

    def to_dict(self):
        return asdict(self)


def camera_sample(local_send_ns, capture_clock_ns, local_recv_ns):
    """Represent CameraAPI's single cheap /clock timestamp as an NTP sample."""
    remote = int(capture_clock_ns)
    return ClockSample(
        local_send_ns=int(local_send_ns),
        remote_recv_ns=remote,
        remote_send_ns=remote,
        local_recv_ns=int(local_recv_ns),
    )


def fit_clock(samples, fastest_fraction=0.35, minimum_fast_samples=5):
    """Fit remote->local time using the least-delayed samples.

    Network-delay asymmetry, not round-trip delay itself, is the unknown. Keeping
    the fastest samples bounds how much asymmetry can remain. The reported
    uncertainty is half the best RTT plus the largest residual among samples used.
    """
    samples = list(samples)
    if len(samples) < 2:
        raise ValueError("clock fit needs at least two samples")
    ordered = sorted(samples, key=lambda sample: sample.rtt_ns)
    keep = min(
        len(ordered),
        max(minimum_fast_samples, int(np.ceil(len(ordered) * fastest_fraction))),
    )
    chosen = ordered[:keep]

    remote = np.asarray([s.remote_mid_ns for s in chosen], dtype=np.float64)
    local = np.asarray([s.local_mid_ns for s in chosen], dtype=np.float64)
    remote_origin = int(round(remote[0]))
    local_origin = int(round(local[0]))
    x = remote - remote_origin
    y = local - local_origin
    # A short burst cannot distinguish oscillator drift from transport jitter.
    # Use a pure offset within one second; pre+post bursts span the episode and
    # therefore have enough baseline for a drift fit.
    if np.ptp(x) < 1e9:
        scale = 1.0
        offset = float(np.median(y - x))
    else:
        scale, offset = np.polyfit(x, y, 1)
    predicted = local_origin + offset + scale * x
    residual = np.abs(local - predicted)
    residual_bound = float(np.max(residual)) if len(residual) else 0.0
    uncertainty = int(round(ordered[0].rtt_ns / 2.0 + residual_bound))
    return ClockFit(
        scale=float(scale),
        remote_origin_ns=remote_origin,
        local_origin_ns=int(round(local_origin + offset)),
        uncertainty_ns=uncertainty,
        min_rtt_ns=int(ordered[0].rtt_ns),
        sample_count=len(samples),
    )


def percentile_ns(values, percentile):
    values = list(values)
    if not values:
        return None
    return int(round(float(np.percentile(np.asarray(values, dtype=np.float64), percentile))))
