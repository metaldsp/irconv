//======================================================================================
// Copyright (c) 2026 Pier Luigi Fiorini
// All rights reserved.
//======================================================================================

#include "IrLoader.h"

namespace DSP {

IrLoader::IrLoader()
{
    m_formatManager.registerBasicFormats();
}

//==============================================================================

void IrLoader::prepare(const juce::dsp::ProcessSpec &spec)
{
    m_spec = spec;
    m_prepared.store(true, std::memory_order_release);

    if (m_irStored.load(std::memory_order_acquire)) {
        if (!juce::approximatelyEqual(m_sourceSampleRate, spec.sampleRate))
            initConvolvers(resampleIR(m_rawIrBuffer, m_sourceSampleRate, spec.sampleRate));
        else
            initConvolvers(m_rawIrBuffer);
    }
}

void IrLoader::reset()
{
    const int active = m_activeSet.load(std::memory_order_acquire);
    if (active < 0)
        return;
    for (auto &conv : m_sets[active].convs)
        if (conv)
            conv->resetState();
}

void IrLoader::process(juce::dsp::ProcessContextReplacing<float> &context)
{
    if (context.isBypassed)
        return;

    // Snapshot the active slot index once. The message thread may update
    // m_activeSet between callbacks, but never modifies the slot the audio
    // thread is currently reading — it always builds into the other slot first.
    const int active = m_activeSet.load(std::memory_order_acquire);
    if (active < 0)
        return;

    auto &block = context.getOutputBlock();
    const auto numSamples = block.getNumSamples();
    const auto &convs = m_sets[active].convs;
    const auto numChannels = std::min(block.getNumChannels(), convs.size());

    for (size_t ch = 0; ch < numChannels; ++ch) {
        auto *data = block.getChannelPointer(ch);
        convs[ch]->process(data, data, numSamples);
    }
}

//==============================================================================

bool IrLoader::loadImpulseResponse(const juce::File &irFile)
{
    std::unique_ptr<juce::AudioFormatReader> reader(m_formatManager.createReaderFor(irFile));
    if (reader == nullptr)
        return false;

    const int numChannels = static_cast<int>(reader->numChannels);
    const int numSamples = static_cast<int>(reader->lengthInSamples);

    juce::AudioBuffer<float> buffer(numChannels, numSamples);
    if (!reader->read(&buffer, 0, numSamples, 0, true, true))
        return false;

    return loadImpulseResponse(buffer, reader->sampleRate);
}

bool IrLoader::loadImpulseResponse(const juce::AudioBuffer<float> &ir, double sourceSampleRate)
{
    if (ir.getNumSamples() == 0 || ir.getNumChannels() == 0)
        return false;

    m_rawIrBuffer = ir;
    m_sourceSampleRate = sourceSampleRate;
    m_irStored.store(true, std::memory_order_release);

    if (m_prepared.load(std::memory_order_acquire)) {
        if (!juce::approximatelyEqual(sourceSampleRate, m_spec.sampleRate))
            initConvolvers(resampleIR(m_rawIrBuffer, sourceSampleRate, m_spec.sampleRate));
        else
            initConvolvers(m_rawIrBuffer);
    }

    return true;
}

//==============================================================================

void IrLoader::initConvolvers(const juce::AudioBuffer<float> &ir)
{
    const int numIrChannels = ir.getNumChannels();
    const auto irLen = static_cast<size_t>(ir.getNumSamples());
    const auto blockSize = static_cast<size_t>(m_spec.maximumBlockSize);

    // Build into the slot that is not currently active so the audio thread
    // always reads from a stable, fully-initialised set.
    const int current = m_activeSet.load(std::memory_order_relaxed);
    const int build = (current == 0) ? 1 : 0;

    auto &set = m_sets[build];
    set.convs.resize(m_spec.numChannels);

    for (size_t ch = 0; ch < m_spec.numChannels; ++ch) {
        if (!set.convs[ch])
            set.convs[ch] = std::make_unique<fftconvolver::FFTConvolver>();

        const int irCh = static_cast<int>(ch) % numIrChannels;
        set.convs[ch]->init(blockSize, ir.getReadPointer(irCh), irLen);
    }

    // Publish: the audio thread will pick up the new slot on its next callback.
    m_activeSet.store(build, std::memory_order_release);
}

juce::AudioBuffer<float> IrLoader::resampleIR(
    const juce::AudioBuffer<float> &ir, double sourceSampleRate, double targetSampleRate)
{
    const double ratio = sourceSampleRate / targetSampleRate;
    const int numChannels = ir.getNumChannels();
    const int newLength
        = std::max(1, juce::roundToInt(static_cast<double>(ir.getNumSamples()) / ratio));

    juce::AudioBuffer<float> output(numChannels, newLength);

    for (int ch = 0; ch < numChannels; ++ch) {
        juce::LagrangeInterpolator interpolator;
        interpolator.reset();
        interpolator.process(ratio, ir.getReadPointer(ch), output.getWritePointer(ch), newLength);
    }

    return output;
}

} // namespace DSP
