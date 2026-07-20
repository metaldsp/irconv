// SPDX-FileCopyrightText: 2026 Pier Luigi Fiorini
// SPDX-License-Identifier: MIT

#pragma once

#include <juce_dsp/juce_dsp.h>

#include <atomic>

namespace DSP {

/** Realtime-safe, smoothed 12 dB/octave HPF followed by a 12 dB/octave LPF. */
class IrFilter
{
public:
    static constexpr float defaultHighPassFrequency = 10.0f;
    static constexpr float defaultLowPassFrequency = 22000.0f;

    IrFilter() noexcept;

    void prepare(const juce::dsp::ProcessSpec &spec);
    /** Clears filter state. Call from the audio thread while processing is stopped. */
    void reset() noexcept;

    /** Publishes a new target from any thread without blocking. */
    void setHighPassFrequency(float frequency) noexcept;
    /** Publishes a new target from any thread without blocking. */
    void setLowPassFrequency(float frequency) noexcept;

    [[nodiscard]] float getHighPassFrequency() const noexcept;
    [[nodiscard]] float getLowPassFrequency() const noexcept;

    /** Clamps a requested cutoff to the stable range for a processing sample rate. */
    [[nodiscard]] static float clampFrequency(float frequency, double sampleRate) noexcept;

    /** Processes in place on the realtime audio thread without allocation or locking. */
    void process(juce::dsp::ProcessContextReplacing<float> &context) noexcept;

private:
    [[nodiscard]] float clampFrequency(float frequency) const noexcept
    {
        return clampFrequency(frequency, m_sampleRate);
    }

    juce::dsp::StateVariableTPTFilter<float> m_highPass;
    juce::dsp::StateVariableTPTFilter<float> m_lowPass;
    std::atomic<float> m_highPassTarget{defaultHighPassFrequency};
    std::atomic<float> m_lowPassTarget{defaultLowPassFrequency};
    juce::LinearSmoothedValue<float> m_smoothedHighPass;
    juce::LinearSmoothedValue<float> m_smoothedLowPass;
    double m_sampleRate = 44100.0;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(IrFilter)
};

} // namespace DSP
