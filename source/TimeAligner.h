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
};

} // namespace DSP
