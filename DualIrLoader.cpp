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

// Build a fractional-delay FIR (N-tap windowed sinc, Blackman window).
// frac is in [0, 1). Returns a kernel of length N.
std::vector<float> fractionalDelaySinc(float frac, int N = 16)
{
    std::vector<float> h(static_cast<size_t>(N));
    const int center = N / 2;
    for (int i = 0; i < N; ++i) {
        const float x = static_cast<float>(i - center) - frac;
        float sinc = (std::abs(x) < 1e-6f) ? 1.0f
                                           : std::sin(juce::MathConstants<float>::pi * x)
                                                 / (juce::MathConstants<float>::pi * x);
        // Blackman window
        const float w = 0.42f - 0.5f * std::cos(2.0f * juce::MathConstants<float>::pi * i / (N - 1))
                        + 0.08f * std::cos(4.0f * juce::MathConstants<float>::pi * i / (N - 1));
        h[static_cast<size_t>(i)] = sinc * w;
    }
    return h;
}

// Convolve src with a short kernel into dst (same length as src, causal).
juce::AudioBuffer<float> applyFir(
    const juce::AudioBuffer<float> &src, const std::vector<float> &kernel)
{
    const int nCh = src.getNumChannels();
    const int nSamp = src.getNumSamples();
    const int kLen = static_cast<int>(kernel.size());
    juce::AudioBuffer<float> dst(nCh, nSamp);
    dst.clear();
    for (int ch = 0; ch < nCh; ++ch) {
        const float *in = src.getReadPointer(ch);
        float *out = dst.getWritePointer(ch);
        for (int i = 0; i < nSamp; ++i) {
            float acc = 0.0f;
            for (int k = 0; k < kLen; ++k) {
                const int j = i - k;
                if (j >= 0)
                    acc += in[j] * kernel[static_cast<size_t>(k)];
            }
            out[i] = acc;
        }
    }
    return dst;
}

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

    m_processingRate = spec.sampleRate;
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
        }
        if (bLoaded && numCh >= 2) {
            auto rightBlock = block.getSubsetChannelBlock(1, 1);
            juce::dsp::ProcessContextReplacing<float> ctxB{rightBlock};
            m_irB.process(ctxB);
        }
        return;
    }

    // Blend mode: route through whichever single IR is loaded, or crossfade both.
    if (aLoaded && !bLoaded) {
        m_irA.process(context);
        return;
    }

    if (!aLoaded && bLoaded) {
        m_irB.process(context);
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

    // Wet B in place on scratch.
    juce::dsp::ProcessContextReplacing<float> ctxB{scratch};
    m_irB.process(ctxB);

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

void DualIrLoader::setMode(StereoMode mode) noexcept
{
    m_mode.store(static_cast<int>(mode), std::memory_order_relaxed);
}

DualIrLoader::StereoMode DualIrLoader::getMode() const noexcept
{
    return static_cast<StereoMode>(m_mode.load(std::memory_order_relaxed));
}

//==============================================================================

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

    const double workRate = (m_irASourceRate > 0.0)
                                ? m_irASourceRate
                                : (m_processingRate > 0.0 ? m_processingRate : 44100.0);

    juce::AudioBuffer<float> working = m_rawIrABuffer;

    if (invertPolarity)
        working.applyGain(-1.0f);

    const double delaySamples = delayMs * 0.001 * workRate;
    const int intDelay = juce::roundToInt(static_cast<float>(delaySamples));
    const float fracDelay = static_cast<float>(delaySamples - intDelay);

    if (std::abs(fracDelay) > 1e-3f)
        working = applyFir(working, fractionalDelaySinc(fracDelay));

    if (intDelay > 0) {
        const int nCh = working.getNumChannels();
        const int nSamp = working.getNumSamples();
        juce::AudioBuffer<float> delayed(nCh, nSamp + intDelay);
        delayed.clear();
        for (int ch = 0; ch < nCh; ++ch)
            delayed.copyFrom(ch, intDelay, working, ch, 0, nSamp);
        working = std::move(delayed);
    } else if (intDelay < 0) {
        const int trim = std::min(-intDelay, working.getNumSamples() - 1);
        const int nCh = working.getNumChannels();
        const int newLen = working.getNumSamples() - trim;
        juce::AudioBuffer<float> trimmed(nCh, newLen);
        for (int ch = 0; ch < nCh; ++ch)
            trimmed.copyFrom(ch, 0, working, ch, trim, newLen);
        working = std::move(trimmed);
    }

    m_irA.loadImpulseResponse(working, workRate);
}

void DualIrLoader::applyAlignmentToIrB(float delayMs, bool invertPolarity)
{
    if (m_rawIrBBuffer.getNumSamples() == 0)
        return;

    // Work in source-rate samples; IrLoader::loadImpulseResponse handles resampling.
    const double workRate = (m_irBSourceRate > 0.0)
                                ? m_irBSourceRate
                                : (m_processingRate > 0.0 ? m_processingRate : 44100.0);

    juce::AudioBuffer<float> working = m_rawIrBBuffer;

    if (invertPolarity)
        working.applyGain(-1.0f);

    const double delaySamples = delayMs * 0.001 * workRate;
    const int intDelay = juce::roundToInt(static_cast<float>(delaySamples));
    const float fracDelay = static_cast<float>(delaySamples - intDelay);

    if (std::abs(fracDelay) > 1e-3f)
        working = applyFir(working, fractionalDelaySinc(fracDelay));

    if (intDelay > 0) {
        const int nCh = working.getNumChannels();
        const int nSamp = working.getNumSamples();
        juce::AudioBuffer<float> delayed(nCh, nSamp + intDelay);
        delayed.clear();
        for (int ch = 0; ch < nCh; ++ch)
            delayed.copyFrom(ch, intDelay, working, ch, 0, nSamp);
        working = std::move(delayed);
    } else if (intDelay < 0) {
        const int trim = std::min(-intDelay, working.getNumSamples() - 1);
        const int nCh = working.getNumChannels();
        const int newLen = working.getNumSamples() - trim;
        juce::AudioBuffer<float> trimmed(nCh, newLen);
        for (int ch = 0; ch < nCh; ++ch)
            trimmed.copyFrom(ch, 0, working, ch, trim, newLen);
        working = std::move(trimmed);
    }

    // Pass source rate so IrLoader resamples to processing rate automatically.
    m_irB.loadImpulseResponse(working, workRate);
}

} // namespace DSP
