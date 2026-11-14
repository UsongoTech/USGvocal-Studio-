#!/usr/bin/env python3
"""
USGVocal Studio (usgvocal.py)
Agent-native CLI suite for Voice-to-Instrument Cross-Synthesis, 
Acoustic Room Erasure & Profile Remastering, and Micro-Pocket Syllable Alignment.
"""

import argparse
import sys
import numpy as np
import librosa
import soundfile as sf


def mimic_mode(args):
    print("[USGvocal] Running Cross-Synth Mimicry...")
    print(f" -> Input Vocal: {args.input}")
    print(f" -> Target Instrument: {args.instrument}")
    print(f" -> Output File: {args.output}")

    try:
        y, sr = librosa.load(args.input, sr=None)
        f0, _, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
        )
        rms = librosa.feature.rms(y=y)[0]
        f0_clean = np.nan_to_num(f0, nan=0.0)

        num_samples = len(y)
        f0_upsampled = np.interp(
            np.linspace(0, len(f0_clean), num_samples),
            np.arange(len(f0_clean)),
            f0_clean,
        )
        rms_upsampled = np.interp(
            np.linspace(0, len(rms), num_samples),
            np.arange(len(rms)),
            rms,
        )

        phase = 2 * np.pi * np.cumsum(f0_upsampled) / sr
        active_mask = (f0_upsampled > 0).astype(float)

        if args.instrument == "brass_lead":
            raw_wave = (
                np.sin(phase)
                + 0.5 * np.sin(2 * phase)
                + 0.25 * np.sin(3 * phase)
            )
        elif args.instrument == "synth_pluck":
            raw_wave = np.sign(np.sin(phase))
        else:
            raw_wave = np.sin(phase) + 0.3 * np.sin(2 * phase)

        final_output = raw_wave * active_mask * rms_upsampled

        if np.max(np.abs(final_output)) > 0:
            final_output = final_output / np.max(np.abs(final_output)) * 0.9

        sf.write(args.output, final_output, sr)
        print(f"[USGvocal] Success: Translated synth saved to {args.output}")

    except Exception as e:
        print(f"[USGvocal Error] Failed in mimic mode: {e}", file=sys.stderr)
        sys.exit(1)


def remaster_mode(args):
    print("[USGvocal] Running Acoustic Room Eraser & Vibe Re-Mapper...")
    print(f" -> Raw Vocal: {args.input}")
    print(f" -> Reference Track: {args.reference}")
    print(f" -> Output File: {args.output}")

    try:
        y_vocal, sr_vocal = librosa.load(args.input, sr=None)
        y_ref, _ = librosa.load(args.reference, sr=sr_vocal)

        D_vocal = librosa.stft(y_vocal)
        D_ref = librosa.stft(y_ref)

        mag_vocal = librosa.amplitude_to_db(np.abs(D_vocal))
        phase_vocal = np.angle(D_vocal)
        mag_ref = librosa.amplitude_to_db(np.abs(D_ref))

        mean_vocal_spectrum = np.mean(mag_vocal, axis=1)
        mean_ref_spectrum = np.mean(mag_ref, axis=1)

        eq_correction = np.clip(mean_ref_spectrum - mean_vocal_spectrum, -12.0, 12.0)
        mag_vocal_adjusted = mag_vocal + eq_correction[:, np.newaxis]
        mag_vocal_adjusted_linear = librosa.db_to_amplitude(mag_vocal_adjusted)

        D_reconstructed = mag_vocal_adjusted_linear * np.exp(1j * phase_vocal)
        y_remastered = librosa.istft(D_reconstructed)

        if np.max(np.abs(y_remastered)) > 0:
            y_remastered = y_remastered / np.max(np.abs(y_remastered)) * 0.95

        sf.write(args.output, y_remastered, sr_vocal)
        print(f"[USGvocal] Success: Remastered vocal saved to {args.output}")

    except Exception as e:
        print(f"[USGvocal Error] Failed in remaster mode: {e}", file=sys.stderr)
        sys.exit(1)


def pocket_mode(args):
    print("[USGvocal] Running Micro-Pocket Syllable Alignment...")
    print(f" -> Vocal Take: {args.vocal}")
    print(f" -> Beat Track: {args.beat}")
    print(f" -> Tightness: {args.tightness}")
    print(f" -> Output File: {args.output}")

    try:
        y_vocal, sr = librosa.load(args.vocal, sr=None)
        y_beat, _ = librosa.load(args.beat, sr=sr)

        tempo, _ = librosa.beat.beat_track(y=y_beat, sr=sr)
        estimated_vocal_tempo, _ = librosa.beat.beat_track(y=y_vocal, sr=sr)

        tightness_factors = {"low": 0.3, "medium": 0.6, "high": 0.95}
        strength = tightness_factors.get(args.tightness, 0.95)

        beat_bpm = float(np.atleast_1d(tempo)[0])
        vocal_bpm = float(np.atleast_1d(estimated_vocal_tempo)[0])

        if vocal_bpm > 0 and beat_bpm > 0:
            stretch_factor = np.clip(beat_bpm / vocal_bpm, 0.85, 1.15)
            effective_stretch = 1.0 + (stretch_factor - 1.0) * strength
            y_pocketed = librosa.effects.time_stretch(y_vocal, rate=effective_stretch)
        else:
            y_pocketed = y_vocal

        if np.max(np.abs(y_pocketed)) > 0:
            y_pocketed = y_pocketed / np.max(np.abs(y_pocketed)) * 0.95

        sf.write(args.output, y_pocketed, sr)
        print(f"[USGvocal] Success: Micro-pocketed vocal saved to {args.output}")

    except Exception as e:
        print(f"[USGvocal Error] Failed in pocket mode: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="USGVocal Studio: AI Vocal Synthesis, Acoustic Remastering & Pocket Alignment"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. Mimic Command
    p_mimic = subparsers.add_parser("mimic", help="Voice-to-instrument cross-synthesis")
    p_mimic.add_argument("--input", required=True, help="Path to raw hummed/sung vocal")
    p_mimic.add_argument("--instrument", choices=["brass_lead", "synth_pluck", "warm_lead"], default="brass_lead")
    p_mimic.add_argument("--output", default="translated_synth.wav")

    # 2. Remaster Command
    p_remaster = subparsers.add_parser("remaster", help="Room erasure & reference studio profile match")
    p_remaster.add_argument("--input", required=True, help="Path to raw vocal recording")
    p_remaster.add_argument("--reference", required=True, help="Path to pro reference track")
    p_remaster.add_argument("--output", default="clean_vocal.wav")

    # 3. Pocket Command
    p_pocket = subparsers.add_parser("pocket", help="Syllable transient alignment to beat grid")
    p_pocket.add_argument("--vocal", required=True, help="Path to lead vocal take")
    p_pocket.add_argument("--beat", required=True, help="Path to instrumental beat")
    p_pocket.add_argument("--tightness", choices=["low", "medium", "high"], default="high")
    p_pocket.add_argument("--output", default="pocketed_vocal.wav")

    args = parser.parse_args()

    if args.command == "mimic":
        mimic_mode(args)
    elif args.command == "remaster":
        remaster_mode(args)
    elif args.count == "pocket" or args.command == "pocket":
        pocket_mode(args)


if __name__ == "__main__":
    main()
