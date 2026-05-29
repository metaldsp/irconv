//======================================================================================
// Copyright (c) 2026 Pier Luigi Fiorini
// All rights reserved.
//======================================================================================

#pragma once

#include "IrLoader.h"

#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_dsp/juce_dsp.h>

#include <atomic>

namespace DSP {

/**
 * Runs the input signal through two IRs in parallel and linearly crossfades
 * their fully convolved wet outputs in the time domain.
 *
 *   out = (1 - blend) * wet_A + blend * wet_B
 *
 * blend == 0.0  → IR A only
 * blend == 1.0  → IR B only
 * blend == 0.5  → equal-weight mix
 *
 * The crossfade is performed in place via two SIMD-accelerated ramp passes per
 * channel (juce::AudioBuffer::applyGainRamp + addFromWithRamp). The blend
 * value is smoothed by a juce::LinearSmoothedValue that advances once per
 * block to avoid zipper noise without introducing a per-sample dependency.
 *
 * IR loading follows the same threading contract as DSP::IrLoader: non-RT
 * thread only.
 */
class DualIrLoader
{
public:
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

    //==============================================================================
    // Audio-thread safe. Stores the target value atomically; the smoother
    // ramps to it inside the next process() call.
    void setBlend(float blend01) noexcept;

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
    juce::HeapBlock<float *> m_blockChannelPtrs;      // sized in prepare()

    juce::File m_irAFile;
    juce::File m_irBFile;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(DualIrLoader)
};

} // namespace DSP
