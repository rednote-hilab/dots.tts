"""Pure waveform helpers shared by training noise augmentation and Studio."""

from __future__ import annotations

import torch


def mix_region_at_snr(
    clean: torch.Tensor,
    noise: torch.Tensor,
    *,
    snr_db: float,
    eps: float = 1.0e-8,
) -> tuple[torch.Tensor, float]:
    clean_float = clean.float()
    noise_float = noise.float()
    noise_float = noise_float - noise_float.mean()
    clean_rms = clean_float.pow(2).mean().clamp_min(float(eps)).sqrt()
    noise_rms = noise_float.pow(2).mean().clamp_min(float(eps)).sqrt()
    gain = clean_rms / (noise_rms * (10.0 ** (float(snr_db) / 20.0)))
    mixed = clean_float + gain * noise_float
    return mixed.to(dtype=clean.dtype), float(gain.item())


def limit_peak(
    waveform: torch.Tensor,
    *,
    effective_length: int,
    peak_limit: float = 0.99,
) -> tuple[torch.Tensor, float]:
    effective_length = min(int(effective_length), int(waveform.numel()))
    if effective_length <= 0:
        return waveform.contiguous(), 1.0
    peak = float(waveform[:effective_length].abs().max().item())
    scale = 1.0 if peak <= float(peak_limit) else float(peak_limit) / peak
    if scale < 1.0:
        waveform = waveform.clone()
        waveform[:effective_length] *= scale
    return waveform.contiguous(), scale


def tile_noise_with_crossfade(
    noise: torch.Tensor,
    *,
    minimum_length: int,
    sample_rate: int,
    fade_ms: float = 10.0,
    preserve_tail: bool = False,
) -> torch.Tensor:
    noise_length = int(noise.numel())
    if noise_length <= 0:
        raise ValueError("Noise waveform is empty.")
    minimum_length = int(minimum_length)
    overlap = int(round(int(sample_rate) * float(fade_ms) / 1000.0))
    overlap = min(overlap, noise_length // 2)
    if overlap <= 0:
        repeat_count = (minimum_length + noise_length - 1) // noise_length
        result = noise.repeat(repeat_count)
        return result.contiguous() if preserve_tail else result[:minimum_length].contiguous()

    fade_in = torch.linspace(
        0.0,
        1.0,
        overlap + 2,
        dtype=noise.dtype,
        device=noise.device,
    )[1:-1]
    fade_out = 1.0 - fade_in
    stride = noise_length - overlap
    repeat_count = max(
        1,
        (max(0, minimum_length - overlap) + stride - 1) // stride,
    )
    if repeat_count == 1:
        result = noise
    else:
        crossfade = noise[-overlap:] * fade_out + noise[:overlap] * fade_in
        middle = noise[overlap:-overlap]
        pieces = [noise[:-overlap]]
        for repeat_index in range(1, repeat_count):
            pieces.append(crossfade)
            pieces.append(
                noise[overlap:]
                if repeat_index == repeat_count - 1
                else middle
            )
        result = torch.cat(pieces, dim=0)
    return result.contiguous() if preserve_tail else result[:minimum_length].contiguous()


def fit_noise_segment(
    noise: torch.Tensor,
    *,
    length: int,
    sample_rate: int,
    crop_fraction: float,
    fade_ms: float = 10.0,
    random_crop_after_tiling: bool = False,
) -> tuple[torch.Tensor, dict[str, int | bool]]:
    length = int(length)
    available = int(noise.numel())
    if length <= 0:
        return noise.new_zeros((0,)), {
            "requested_samples": 0,
            "available_samples": available,
            "crop_start": 0,
            "tiled": False,
        }
    tiled = available < length
    candidate = (
        tile_noise_with_crossfade(
            noise,
            minimum_length=length,
            sample_rate=sample_rate,
            fade_ms=fade_ms,
            preserve_tail=random_crop_after_tiling,
        )
        if tiled
        else noise
    )
    max_start = max(0, int(candidate.numel()) - length)
    fraction = min(1.0, max(0.0, float(crop_fraction)))
    start = min(max_start, int(fraction * float(max_start + 1)))
    return candidate[start : start + length].contiguous(), {
        "requested_samples": length,
        "available_samples": available,
        "crop_start": start,
        "tiled": tiled,
    }
