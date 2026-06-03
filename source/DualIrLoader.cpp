//======================================================================================
// Copyright (c) 2026 Pier Luigi Fiorini
// All rights reserved.
//======================================================================================

#include "DualIrLoader.h"

namespace DSP {

namespace {
constexpr double kBlendRampSeconds = 0.02; // 20 ms — fast enough to feel
                                           // instant, slow enough to silence
                                           // zipper noise on knob sweeps.
} // namespace

void DualIrLoader::prepare(const juce::dsp::ProcessSpec &spec)
{
    m_irA.prepare(spec);
    m_irB.prepare(spec);

    m_scratch.setSize(
        static_cast<int>(spec.numChannels),
        static_cast<int>(spec.maximumBlockSize),
        false,
        false,
        true);
    m_scratch.clear();

    m_blockChannelPtrs.allocate(spec.numChannels, false);

    m_blendSmoother.reset(spec.sampleRate, kBlendRampSeconds);
    m_blendSmoother.setCurrentAndTargetValue(m_blendTarget.load(std::memory_order_relaxed));
}

void DualIrLoader::reset()
{
    m_irA.reset();
    m_irB.reset();
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

    // Advance the smoother once for the whole block (O(1)).
    m_blendSmoother.setTargetValue(m_blendTarget.load(std::memory_order_relaxed));
    const float wStart = m_blendSmoother.getCurrentValue();
    const float wEnd = m_blendSmoother.skip(static_cast<int>(numSamples));

    // Snapshot dry input into scratch so IR B can consume it after IR A
    // overwrites the main block.
    juce::dsp::AudioBlock<float> scratchBlock{m_scratch};
    auto scratch = scratchBlock.getSubsetChannelBlock(0, channels).getSubBlock(0, numSamples);
    scratch.copyFrom(block.getSubsetChannelBlock(0, channels));

    // Wet A in place on the output block (only the channels we have scratch for).
    auto blockA = block.getSubsetChannelBlock(0, channels);
    juce::dsp::ProcessContextReplacing<float> ctxA{blockA};
    m_irA.process(ctxA);

    // Wet B in place on scratch.
    juce::dsp::ProcessContextReplacing<float> ctxB{scratch};
    m_irB.process(ctxB);

    // In-place linear crossfade — two SIMD passes per channel, zero extra
    // copies. out[i] := wetA[i] * (1 - w[i]) + wetB[i] * w[i].
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
    m_irAFile = irFile;
    return true;
}

bool DualIrLoader::loadImpulseResponseB(const juce::File &irFile)
{
    if (!m_irB.loadImpulseResponse(irFile))
        return false;
    m_irBFile = irFile;
    return true;
}

bool DualIrLoader::loadImpulseResponseA(const juce::AudioBuffer<float> &ir, double sourceSampleRate)
{
    return m_irA.loadImpulseResponse(ir, sourceSampleRate);
}

bool DualIrLoader::loadImpulseResponseB(const juce::AudioBuffer<float> &ir, double sourceSampleRate)
{
    return m_irB.loadImpulseResponse(ir, sourceSampleRate);
}

void DualIrLoader::setBlend(float blend01) noexcept
{
    m_blendTarget.store(juce::jlimit(0.0f, 1.0f, blend01), std::memory_order_relaxed);
}

//==============================================================================

void DualIrLoader::clearImpulseResponseA() noexcept
{
    m_irA.clearImpulseResponse();
    m_irAFile = juce::File{};
}

void DualIrLoader::clearImpulseResponseB() noexcept
{
    m_irB.clearImpulseResponse();
    m_irBFile = juce::File{};
}

void DualIrLoader::setNormalise(bool normalise) noexcept
{
    m_irA.setNormalise(normalise);
    m_irB.setNormalise(normalise);
}

} // namespace DSP
