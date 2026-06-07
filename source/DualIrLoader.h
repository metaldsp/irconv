// SPDX-FileCopyrightText: 2026 Pier Luigi Fiorini
// SPDX-License-Identifier: MIT

#pragma once

#include "IrLoader.h"

#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_dsp/juce_dsp.h>

#include <atomic>

namespace DSP {

/**
 * Runs the input signal through two IRs and routes them according to the active
 * StereoMode.
 *
 * Blend mode (default):
 *   out = (1 - blend) * wet_A + blend * wet_B   (applied to every channel)
 *   blend == 0.0 → IR A only; 1.0 → IR B only; 0.5 → equal mix.
 *   The crossfade uses two SIMD ramp passes (applyGainRamp + addFromWithRamp)
 *   to avoid zipper noise.
 *
 * Stereo Split mode:
 *   IR A is applied exclusively to channel 0 (left).
 *   IR B is applied exclusively to channel 1 (right).
 *
 * IR loading follows the same threading contract as DSP::IrLoader: non-RT
 * thread only.
 */
class DualIrLoader
{
public:
    enum class StereoMode { Blend = 0, StereoSplit = 1 };

    DualIrLoader() = default;
    ~DualIrLoader() = default;

    //==============================================================================
    // JUCE DSP module interface

    void prepare(const juce::dsp::ProcessSpec &spec);
    void reset();
    void process(juce::dsp::ProcessContextReplacing<float> &context);

    //==============================================================================
    // IR loading — non-audio thread only

    bool loadImpulseResponseA(const juce::File &irFile);
    bool loadImpulseResponseB(const juce::File &irFile);
    bool loadImpulseResponseA(const juce::AudioBuffer<float> &ir, double sourceSampleRate);
    bool loadImpulseResponseB(const juce::AudioBuffer<float> &ir, double sourceSampleRate);

    void clearImpulseResponseA() noexcept;
    void clearImpulseResponseB() noexcept;

    /**
     * Enables or disables peak normalisation applied to both IR A and IR B
     * when loading. Takes effect on the next loadImpulseResponse() call;
     * does not retroactively alter already-loaded IRs. Message-thread only.
     */
    void setNormalise(bool normalise) noexcept;

    /**
     * Applies a time-alignment shift and optional polarity inversion to IR A.
     * Re-derives a shifted/inverted copy from the stored raw IR A buffer and
     * reloads it into the convolver. Message-thread only.
     *
     * @param delayMs        Delay in milliseconds. Positive = A lags reference
     *                       (prepend silence). Negative = A leads (trim onset).
     * @param invertPolarity Negate all samples when true.
     */
    void applyAlignmentToIrA(float delayMs, bool invertPolarity);

    /**
     * Applies a time-alignment shift and optional polarity inversion to IR B.
     * Re-derives a shifted/inverted copy from the stored raw IR B buffer and
     * reloads it into the convolver. Message-thread only.
     *
     * @param delayMs        Delay in milliseconds. Positive = B lags A
     *                       (prepend silence). Negative = B leads A (trim onset).
     * @param invertPolarity Negate all samples when true.
     */
    void applyAlignmentToIrB(float delayMs, bool invertPolarity);

    // Message-thread only — not RT-safe.
    [[nodiscard]] const juce::AudioBuffer<float> &getRawIrABuffer() const noexcept
    {
        return m_rawIrABuffer;
    }
    [[nodiscard]] const juce::AudioBuffer<float> &getRawIrBBuffer() const noexcept
    {
        return m_rawIrBBuffer;
    }
    [[nodiscard]] double getIrASourceRate() const noexcept { return m_irASourceRate; }
    [[nodiscard]] double getIrBSourceRate() const noexcept { return m_irBSourceRate; }

    //==============================================================================
    // Audio-thread safe. Stores the target value atomically; the smoother
    // ramps to it inside the next process() call.
    void setBlend(float blend01) noexcept;

    // Audio-thread safe. Switches between Blend and StereoSplit routing.
    void setMode(StereoMode mode) noexcept;
    [[nodiscard]] StereoMode getMode() const noexcept;

    //==============================================================================
    // Message-thread only — not RT-safe.
    [[nodiscard]] const juce::File &getImpulseResponseFileA() const noexcept { return m_irAFile; }
    [[nodiscard]] const juce::File &getImpulseResponseFileB() const noexcept { return m_irBFile; }

private:
    IrLoader m_irA;
    IrLoader m_irB;

    juce::AudioBuffer<float> m_scratch;               // sized in prepare(), reused in process()
    juce::LinearSmoothedValue<float> m_blendSmoother; // audio-thread only
    std::atomic<float> m_blendTarget{0.5f};           // any thread → audio
    std::atomic<int> m_mode{static_cast<int>(StereoMode::Blend)}; // any thread → audio
    juce::HeapBlock<float *> m_blockChannelPtrs;                  // sized in prepare()

    juce::File m_irAFile;
    juce::File m_irBFile;

    // Raw IR buffers and metadata for alignment re-application.
    juce::AudioBuffer<float> m_rawIrABuffer;
    double m_irASourceRate = 0.0;
    juce::AudioBuffer<float> m_rawIrBBuffer;
    double m_irBSourceRate = 0.0;
    double m_processingRate = 0.0; // cached in prepare()

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(DualIrLoader)
};

} // namespace DSP
