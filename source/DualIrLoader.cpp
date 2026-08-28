// SPDX-FileCopyrightText: 2026 Pier Luigi Fiorini
// SPDX-License-Identifier: MIT

#include "DualIrLoader.h"

#include "TimeAligner.h"

namespace DSP {

namespace {

constexpr double kBlendRampSeconds = 0.02; // 20 ms — fast enough to feel
                                           // instant, slow enough to silence
                                           // zipper noise on knob sweeps.

} // namespace

DualIrLoader::DualIrLoader()
{
    // juce::dsp::Gain wraps a linear juce::SmoothedValue, whose default value is 0 —
    // i.e. silence. Seed both slots at unity so a host that never touches the gain
    // still hears the wet signal. prepare() propagates this target to the smoother's
    // current value, so a setGainA/B() call made before prepare() is still honoured.
    m_gainA.setGainLinear(1.0f);
    m_gainB.setGainLinear(1.0f);
}

void DualIrLoader::prepare(const juce::dsp::ProcessSpec &spec)
{
    m_irA.prepare(spec);
    m_irB.prepare(spec);
    m_filterA.prepare(spec);
    m_filterB.prepare(spec);
    m_gainA.prepare(spec);
    m_gainA.setRampDurationSeconds(kBlendRampSeconds);
    m_gainB.prepare(spec);
    m_gainB.setRampDurationSeconds(kBlendRampSeconds);

    m_scratch.setSize(
        static_cast<int>(spec.numChannels),
        static_cast<int>(spec.maximumBlockSize),
        false,
        false,
        true);
    m_scratch.clear();

    m_blockChannelPtrs.allocate(spec.numChannels, false);

    m_processingRate = spec.sampleRate;
    m_blendSmoother.reset(spec.sampleRate, kBlendRampSeconds);
    m_blendSmoother.setCurrentAndTargetValue(m_blendTarget.load(std::memory_order_relaxed));
}

void DualIrLoader::reset()
{
    m_irA.reset();
    m_irB.reset();
    m_filterA.reset();
    m_filterB.reset();
    m_gainA.reset();
    m_gainB.reset();
    m_scratch.clear();
    m_blendSmoother.setCurrentAndTargetValue(m_blendSmoother.getTargetValue());
}

void DualIrLoader::process(juce::dsp::ProcessContextReplacing<float> &context)
{
    if (context.isBypassed)
        return;

    auto &block = context.getOutputBlock();
    const auto numSamples = block.getNumSamples();
    const auto numCh = block.getNumChannels();

    // Clamp to what we provisioned in prepare(). If the host hands us a buffer
    // with more channels than spec.numChannels, we mix only the channels we
    // have scratch + channel-pointer storage for; extra channels are left
    // untouched rather than indexed out of bounds.
    const auto channels = std::min(numCh, static_cast<size_t>(m_scratch.getNumChannels()));

    // Advance the smoother unconditionally so its state tracks real time even
    // in the single-IR branches; wStart/wEnd are only used in the both-loaded path.
    m_blendSmoother.setTargetValue(m_blendTarget.load(std::memory_order_relaxed));
    const float wStart = m_blendSmoother.getCurrentValue();
    const float wEnd = m_blendSmoother.skip(static_cast<int>(numSamples));

    const bool aLoaded = m_irA.isLoaded();
    const bool bLoaded = m_irB.isLoaded();

    if (!aLoaded && !bLoaded)
        return;

    // Stereo Split: IR A → left channel only, IR B → right channel only.
    const auto mode = static_cast<StereoMode>(m_mode.load(std::memory_order_relaxed));
    if (mode == StereoMode::StereoSplit) {
        if (aLoaded && numCh >= 1) {
            auto leftBlock = block.getSubsetChannelBlock(0, 1);
            juce::dsp::ProcessContextReplacing<float> ctxA{leftBlock};
            m_irA.process(ctxA);
            m_filterA.process(ctxA);
            m_gainA.process(ctxA);
        }
        if (bLoaded && numCh >= 2) {
            auto rightBlock = block.getSubsetChannelBlock(1, 1);
            juce::dsp::ProcessContextReplacing<float> ctxB{rightBlock};
            m_irB.process(ctxB);
            m_filterB.process(ctxB);
            m_gainB.process(ctxB);
        }
        return;
    }

    // Blend mode: route through whichever single IR is loaded, or crossfade both.
    // Clamped to `channels` for the same reason as the both-loaded path below: a host
    // buffer wider than what prepare() provisioned must not be indexed out of bounds.
    if (aLoaded && !bLoaded) {
        auto soloBlockA = block.getSubsetChannelBlock(0, channels);
        juce::dsp::ProcessContextReplacing<float> ctxSoloA{soloBlockA};
        m_irA.process(ctxSoloA);
        m_filterA.process(ctxSoloA);
        m_gainA.process(ctxSoloA);
        return;
    }

    if (!aLoaded && bLoaded) {
        auto soloBlockB = block.getSubsetChannelBlock(0, channels);
        juce::dsp::ProcessContextReplacing<float> ctxSoloB{soloBlockB};
        m_irB.process(ctxSoloB);
        m_filterB.process(ctxSoloB);
        m_gainB.process(ctxSoloB);
        return;
    }

    // Both loaded — snapshot dry input into scratch so IR B can consume it
    // after IR A overwrites the main block.
    juce::dsp::AudioBlock<float> scratchBlock{m_scratch};
    auto scratch = scratchBlock.getSubsetChannelBlock(0, channels).getSubBlock(0, numSamples);
    scratch.copyFrom(block.getSubsetChannelBlock(0, channels));

    // Wet A in place on the output block.
    auto blockA = block.getSubsetChannelBlock(0, channels);
    juce::dsp::ProcessContextReplacing<float> ctxA{blockA};
    m_irA.process(ctxA);
    m_filterA.process(ctxA);
    m_gainA.process(ctxA);

    // Wet B in place on scratch.
    juce::dsp::ProcessContextReplacing<float> ctxB{scratch};
    m_irB.process(ctxB);
    m_filterB.process(ctxB);
    m_gainB.process(ctxB);

    // In-place linear crossfade: out[i] = wetA[i] * (1 - w[i]) + wetB[i] * w[i].
    // Two SIMD ramp passes per channel, zero extra copies.
    for (size_t ch = 0; ch < channels; ++ch)
        m_blockChannelPtrs[ch] = block.getChannelPointer(ch);

    juce::AudioBuffer<float> outBuf(
        m_blockChannelPtrs.getData(), static_cast<int>(channels), static_cast<int>(numSamples));

    for (int ch = 0; ch < static_cast<int>(channels); ++ch) {
        outBuf.applyGainRamp(ch, 0, static_cast<int>(numSamples), 1.0f - wStart, 1.0f - wEnd);
        outBuf.addFromWithRamp(
            ch, 0, m_scratch.getReadPointer(ch), static_cast<int>(numSamples), wStart, wEnd);
    }
}

//==============================================================================

bool DualIrLoader::loadImpulseResponseA(const juce::File &irFile)
{
    if (!m_irA.loadImpulseResponse(irFile))
        return false;
    m_rawIrABuffer = m_irA.getRawIrBuffer();
    m_irASourceRate = m_irA.getSourceSampleRate();
    m_irAFile = irFile;
    return true;
}

bool DualIrLoader::loadImpulseResponseB(const juce::File &irFile)
{
    if (!m_irB.loadImpulseResponse(irFile))
        return false;
    m_rawIrBBuffer = m_irB.getRawIrBuffer();
    m_irBSourceRate = m_irB.getSourceSampleRate();
    m_irBFile = irFile;
    return true;
}

bool DualIrLoader::loadImpulseResponseA(const juce::AudioBuffer<float> &ir, double sourceSampleRate)
{
    if (!m_irA.loadImpulseResponse(ir, sourceSampleRate))
        return false;
    m_rawIrABuffer = m_irA.getRawIrBuffer();
    m_irASourceRate = m_irA.getSourceSampleRate();
    return true;
}

bool DualIrLoader::loadImpulseResponseB(const juce::AudioBuffer<float> &ir, double sourceSampleRate)
{
    if (!m_irB.loadImpulseResponse(ir, sourceSampleRate))
        return false;
    m_rawIrBBuffer = m_irB.getRawIrBuffer();
    m_irBSourceRate = m_irB.getSourceSampleRate();
    return true;
}

void DualIrLoader::setBlend(float blend01) noexcept
{
    m_blendTarget.store(juce::jlimit(0.0f, 1.0f, blend01), std::memory_order_relaxed);
}

void DualIrLoader::setHighPassFrequencyA(float frequency) noexcept
{
    m_filterA.setHighPassFrequency(frequency);
}

void DualIrLoader::setLowPassFrequencyA(float frequency) noexcept
{
    m_filterA.setLowPassFrequency(frequency);
}

void DualIrLoader::setHighPassFrequencyB(float frequency) noexcept
{
    m_filterB.setHighPassFrequency(frequency);
}

void DualIrLoader::setLowPassFrequencyB(float frequency) noexcept
{
    m_filterB.setLowPassFrequency(frequency);
}

void DualIrLoader::setGainA(float decibels) noexcept
{
    m_gainA.setGainDecibels(decibels);
}

void DualIrLoader::setGainB(float decibels) noexcept
{
    m_gainB.setGainDecibels(decibels);
}

void DualIrLoader::setMode(StereoMode mode) noexcept
{
    m_mode.store(static_cast<int>(mode), std::memory_order_relaxed);
}

DualIrLoader::StereoMode DualIrLoader::getMode() const noexcept
{
    return static_cast<StereoMode>(m_mode.load(std::memory_order_relaxed));
}

//==============================================================================

double DualIrLoader::getTailLengthSeconds() const noexcept
{
    double tail = 0.0;
    if (m_irASourceRate > 0.0 && m_rawIrABuffer.getNumSamples() > 0)
        tail = std::max(tail, m_rawIrABuffer.getNumSamples() / m_irASourceRate);
    if (m_irBSourceRate > 0.0 && m_rawIrBBuffer.getNumSamples() > 0)
        tail = std::max(tail, m_rawIrBBuffer.getNumSamples() / m_irBSourceRate);
    return tail;
}

void DualIrLoader::clearImpulseResponseA() noexcept
{
    m_irA.clearImpulseResponse();
    m_irAFile = juce::File{};
    m_rawIrABuffer.setSize(0, 0);
    m_irASourceRate = 0.0;
}

void DualIrLoader::clearImpulseResponseB() noexcept
{
    m_irB.clearImpulseResponse();
    m_irBFile = juce::File{};
    m_rawIrBBuffer.setSize(0, 0);
    m_irBSourceRate = 0.0;
}

void DualIrLoader::setNormalise(bool normalise) noexcept
{
    m_irA.setNormalise(normalise);
    m_irB.setNormalise(normalise);
}

void DualIrLoader::applyAlignmentToIrA(float delayMs, bool invertPolarity)
{
    if (m_rawIrABuffer.getNumSamples() == 0)
        return;

    // Work in source-rate samples; IrLoader::loadImpulseResponse handles resampling.
    const double workRate = (m_irASourceRate > 0.0)
                                ? m_irASourceRate
                                : (m_processingRate > 0.0 ? m_processingRate : 44100.0);

    const auto working
        = TimeAligner::applyAlignment(m_rawIrABuffer, workRate, delayMs, invertPolarity);

    // m_rawIrABuffer is already normalised, so re-normalising here would scale twice.
    m_irA.setNormalise(false);
    m_irA.loadImpulseResponse(working, workRate);
    m_irA.setNormalise(true);
}

void DualIrLoader::applyAlignmentToIrB(float delayMs, bool invertPolarity)
{
    if (m_rawIrBBuffer.getNumSamples() == 0)
        return;

    // Work in source-rate samples; IrLoader::loadImpulseResponse handles resampling.
    const double workRate = (m_irBSourceRate > 0.0)
                                ? m_irBSourceRate
                                : (m_processingRate > 0.0 ? m_processingRate : 44100.0);

    const auto working
        = TimeAligner::applyAlignment(m_rawIrBBuffer, workRate, delayMs, invertPolarity);

    // m_rawIrBBuffer is already normalised, so re-normalising here would scale twice.
    m_irB.setNormalise(false);
    m_irB.loadImpulseResponse(working, workRate);
    m_irB.setNormalise(true);
}

} // namespace DSP
