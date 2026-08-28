// SPDX-FileCopyrightText: 2026 Pier Luigi Fiorini
// SPDX-License-Identifier: MIT

#pragma once

#include <juce_audio_basics/juce_audio_basics.h>

namespace DSP {

struct AlignmentResult
{
    float delaySamples; // signed; positive = B lags A, negative = B leads A
    bool invertPolarity;
    float correlationScore; // 0..1
};

/**
 * Analyses two impulse responses and returns the delay and polarity correction
 * needed to align IR B to IR A.
 *
 * Uses FFT-based normalized cross-correlation with parabolic sub-sample
 * refinement. Runs entirely on the calling (non-RT) thread.
 */
class TimeAligner
{
public:
    /**
     * @param irA            Reference IR (IR A).
     * @param irB            IR to align (IR B).
     * @param sampleRate     Processing sample rate of both buffers.
     * @param searchWindowMs Maximum lag searched (ms). Default 25 ms covers
     *                       all practical mic-placement scenarios.
     */
    [[nodiscard]] static AlignmentResult analyse(
        const juce::AudioBuffer<float> &irA,
        const juce::AudioBuffer<float> &irB,
        double sampleRate,
        float searchWindowMs = 25.0f);

    /**
     * Returns a copy of `ir` shifted in time and optionally polarity-inverted —
     * the correction half of analyse().
     *
     * Sub-sample fractions are applied with a 16-tap Blackman-windowed sinc
     * fractional-delay FIR. A positive delay prepends silence (lengthening the
     * buffer); a negative delay trims the onset.
     *
     * Allocates and runs entirely on the calling thread — non-RT only.
     *
     * @param ir             Source IR. Returned unchanged if empty.
     * @param sampleRate     Rate at which `delayMs` is interpreted.
     * @param delayMs        Delay in milliseconds. Positive = lag, negative = lead.
     * @param invertPolarity Negate all samples when true.
     */
    [[nodiscard]] static juce::AudioBuffer<float> applyAlignment(
        const juce::AudioBuffer<float> &ir, double sampleRate, float delayMs, bool invertPolarity);
};

} // namespace DSP
