// SPDX-FileCopyrightText: 2026 Pier Luigi Fiorini
// SPDX-License-Identifier: MIT

#include "IrFilter.h"

namespace DSP {

namespace {
constexpr double kSmoothingTimeSeconds = 0.02;
constexpr float kMinimumCutoff = 1.0f;
constexpr float kNyquistGuard = 0.99f;
} // namespace

IrFilter::IrFilter() noexcept
{
    m_highPass.setType(juce::dsp::StateVariableTPTFilterType::highpass);
    m_lowPass.setType(juce::dsp::StateVariableTPTFilterType::lowpass);
}

void IrFilter::prepare(const juce::dsp::ProcessSpec &spec)
{
    jassert(spec.sampleRate > 0.0);
    m_sampleRate = spec.sampleRate > 0.0 ? spec.sampleRate : 44100.0;

    m_highPass.prepare(spec);
    m_lowPass.prepare(spec);
    m_smoothedHighPass.reset(m_sampleRate, kSmoothingTimeSeconds);
    m_smoothedLowPass.reset(m_sampleRate, kSmoothingTimeSeconds);

    const float highPass = clampFrequency(m_highPassTarget.load(std::memory_order_acquire));
    const float lowPass = clampFrequency(m_lowPassTarget.load(std::memory_order_acquire));
    m_smoothedHighPass.setCurrentAndTargetValue(highPass);
    m_smoothedLowPass.setCurrentAndTargetValue(lowPass);
    m_highPass.setCutoffFrequency(highPass);
    m_lowPass.setCutoffFrequency(lowPass);
    reset();
}

void IrFilter::reset() noexcept
{
    const float highPass = clampFrequency(m_highPassTarget.load(std::memory_order_acquire));
    const float lowPass = clampFrequency(m_lowPassTarget.load(std::memory_order_acquire));
    m_smoothedHighPass.setCurrentAndTargetValue(highPass);
    m_smoothedLowPass.setCurrentAndTargetValue(lowPass);
    m_highPass.setCutoffFrequency(highPass);
    m_lowPass.setCutoffFrequency(lowPass);
    m_highPass.reset();
    m_lowPass.reset();
}

void IrFilter::setHighPassFrequency(float frequency) noexcept
{
    m_highPassTarget.store(frequency, std::memory_order_release);
}

void IrFilter::setLowPassFrequency(float frequency) noexcept
{
    m_lowPassTarget.store(frequency, std::memory_order_release);
}

float IrFilter::getHighPassFrequency() const noexcept
{
    return m_highPassTarget.load(std::memory_order_acquire);
}

float IrFilter::getLowPassFrequency() const noexcept
{
    return m_lowPassTarget.load(std::memory_order_acquire);
}

void IrFilter::process(juce::dsp::ProcessContextReplacing<float> &context) noexcept
{
    if (context.isBypassed)
        return;

    auto &block = context.getOutputBlock();
    const auto numChannels = block.getNumChannels();
    const auto numSamples = block.getNumSamples();

    m_smoothedHighPass.setTargetValue(
        clampFrequency(m_highPassTarget.load(std::memory_order_acquire)));
    m_smoothedLowPass.setTargetValue(
        clampFrequency(m_lowPassTarget.load(std::memory_order_acquire)));

    // Update the cutoff every sample (not once per block) so automation/knob sweeps
    // ramp smoothly instead of stepping once per block, which produces zipper noise.
    for (size_t sample = 0; sample < numSamples; ++sample) {
        m_highPass.setCutoffFrequency(m_smoothedHighPass.getNextValue());
        m_lowPass.setCutoffFrequency(m_smoothedLowPass.getNextValue());
        for (size_t channel = 0; channel < numChannels; ++channel) {
            const float highPassed
                = m_highPass
                      .processSample(static_cast<int>(channel), block.getSample(channel, sample));
            block.setSample(
                channel, sample, m_lowPass.processSample(static_cast<int>(channel), highPassed));
        }
    }
}

float IrFilter::clampFrequency(float frequency, double sampleRate) noexcept
{
    const double validSampleRate = sampleRate > 0.0 ? sampleRate : 44100.0;
    const float maximum = static_cast<float>(validSampleRate * 0.5) * kNyquistGuard;
    return juce::jlimit(kMinimumCutoff, maximum, frequency);
}

} // namespace DSP
