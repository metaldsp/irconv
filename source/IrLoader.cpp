//======================================================================================
// Copyright (c) 2026 Pier Luigi Fiorini
// All rights reserved.
//======================================================================================

#include "IrLoader.h"

namespace DSP {

namespace {

// IRs longer than this threshold use TwoStageFFTConvolver (background tail thread).
// Below it, the simpler single-stage FFTConvolver is sufficient.
static constexpr size_t kTwoStageThreshold = 16384;

//==============================================================================
// Background-threaded TwoStageFFTConvolver driven by a pair of WaitableEvents.
//
// startBackgroundProcessing() fires once per process() call (from the audio thread);
// run() wakes, executes doBackgroundProcessing(), then signals completion.
// waitForBackgroundProcessing() blocks until that signal arrives.
// Because juce::WaitableEvent auto-resets on wait(), the protocol is race-free
// as long as the caller always pairs start with wait before the next process().
class JuceTwoStageConvolver : public fftconvolver::TwoStageFFTConvolver, private juce::Thread
{
public:
    JuceTwoStageConvolver()
        : juce::Thread("IR Convolver Tail")
    {
        startThread(juce::Thread::Priority::high);
    }

    ~JuceTwoStageConvolver() override
    {
        signalThreadShouldExit();
        m_startEvent.signal(); // unblock run() if it is waiting
        stopThread(500);
        reset();
    }

protected:
    void startBackgroundProcessing() override { m_startEvent.signal(); }

    void waitForBackgroundProcessing() override { m_doneEvent.wait(); }

private:
    void run() override
    {
        while (!threadShouldExit()) {
            m_startEvent.wait();
            if (threadShouldExit())
                break;
            doBackgroundProcessing();
            m_doneEvent.signal();
        }
    }

    juce::WaitableEvent m_startEvent; // auto-reset, initially unsignalled
    juce::WaitableEvent m_doneEvent;  // auto-reset, initially unsignalled
};

//==============================================================================
struct SingleConvChannel : IrLoader::ConvChannel
{
    fftconvolver::FFTConvolver conv;

    void process(const float *in, float *out, size_t len) override { conv.process(in, out, len); }

    void resetState() override { conv.resetState(); }
};

struct TwoStageConvChannel : IrLoader::ConvChannel
{
    JuceTwoStageConvolver conv;

    void process(const float *in, float *out, size_t len) override { conv.process(in, out, len); }

    void resetState() override { conv.resetState(); }
};

//==============================================================================
// Returns the smallest power of two that is >= value.
inline size_t nextPowerOfTwo(size_t value) noexcept
{
    size_t p = 1;
    while (p < value)
        p *= 2;
    return p;
}

} // namespace

//==============================================================================

IrLoader::IrLoader(bool normalise)
    : m_normalise(normalise)
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
    if (reader == nullptr) {
        DBG("IrLoader: unable to open " + irFile.getFullPathName());
        return false;
    }

    const int numChannels = static_cast<int>(reader->numChannels);
    const int numSamples = static_cast<int>(reader->lengthInSamples);

    juce::AudioBuffer<float> buffer(numChannels, numSamples);
    if (!reader->read(&buffer, 0, numSamples, 0, true, true)) {
        DBG("IrLoader: error reading " + irFile.getFullPathName());
        return false;
    }

    return loadImpulseResponse(buffer, reader->sampleRate);
}

bool IrLoader::loadImpulseResponse(const juce::AudioBuffer<float> &ir, double sourceSampleRate)
{
    if (ir.getNumSamples() == 0 || ir.getNumChannels() == 0)
        return false;

    m_rawIrBuffer = ir;

    if (m_normalise) {
        // Step 1: peak-normalize so that the loudest sample reaches 0.8.
        float peak = 0.0f;
        for (int ch = 0; ch < m_rawIrBuffer.getNumChannels(); ++ch)
            peak = std::max(peak, m_rawIrBuffer.getMagnitude(ch, 0, m_rawIrBuffer.getNumSamples()));

        if (peak > 0.0f) {
            m_rawIrBuffer.applyGain(0.8f / peak);

            // Step 2: scale by 1/energy so the sum-of-squares equals 1.
            double sumSq = 0.0;
            for (int ch = 0; ch < m_rawIrBuffer.getNumChannels(); ++ch) {
                const float *data = m_rawIrBuffer.getReadPointer(ch);
                for (int i = 0; i < m_rawIrBuffer.getNumSamples(); ++i)
                    sumSq += static_cast<double>(data[i]) * data[i];
            }
            if (sumSq > 0.0)
                m_rawIrBuffer.applyGain(static_cast<float>(1.0 / sumSq));
        }
    }

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

void IrLoader::clearImpulseResponse() noexcept
{
    m_activeSet.store(-1, std::memory_order_release);
    m_irStored.store(false, std::memory_order_release);
}

//==============================================================================

void IrLoader::initConvolvers(const juce::AudioBuffer<float> &ir)
{
    const int numIrChannels = ir.getNumChannels();
    const auto irLen = static_cast<size_t>(ir.getNumSamples());
    const size_t headBlockSize = nextPowerOfTwo(static_cast<size_t>(m_spec.maximumBlockSize));

    // Build into the slot that is not currently active so the audio thread
    // always reads from a stable, fully-initialised set.
    const int current = m_activeSet.load(std::memory_order_relaxed);
    const int build = (current == 0) ? 1 : 0;

    auto &set = m_sets[build];
    set.convs.clear();
    set.convs.resize(m_spec.numChannels);

    if (irLen > kTwoStageThreshold) {
        // Large IR: use TwoStageFFTConvolver so the tail runs on a background thread.
        const size_t tailBlockSize = std::max(headBlockSize, size_t{8192});

        for (size_t ch = 0; ch < m_spec.numChannels; ++ch) {
            const int irCh = static_cast<int>(ch) % numIrChannels;
            auto channel = std::make_unique<TwoStageConvChannel>();
            channel->conv.init(headBlockSize, tailBlockSize, ir.getReadPointer(irCh), irLen);
            set.convs[ch] = std::move(channel);
        }
    } else {
        // Small IR: use single-stage FFTConvolver, no background thread needed.
        for (size_t ch = 0; ch < m_spec.numChannels; ++ch) {
            const int irCh = static_cast<int>(ch) % numIrChannels;
            auto channel = std::make_unique<SingleConvChannel>();
            channel->conv.init(headBlockSize, ir.getReadPointer(irCh), irLen);
            set.convs[ch] = std::move(channel);
        }
    }

    // Publish: the audio thread will pick up the new slot on its next callback.
    m_activeSet.store(build, std::memory_order_release);
}

juce::AudioBuffer<float> IrLoader::resampleIR(
    const juce::AudioBuffer<float> &ir, double sourceSampleRate, double targetSampleRate)
{
    const double ratio = sourceSampleRate / targetSampleRate;
    const int newLen = std::max(1, juce::roundToInt(ir.getNumSamples() / ratio));
    const int nCh = ir.getNumChannels();

    // MemoryAudioSource requires a non-const buffer reference; copy to satisfy that.
    juce::AudioBuffer<float> mutableCopy(ir);
    juce::MemoryAudioSource memSource(mutableCopy, /*copyMemory=*/false, /*shouldLoop=*/false);

    juce::ResamplingAudioSource resampler(&memSource, /*deleteSourceWhenDeleted=*/false, nCh);
    resampler.setResamplingRatio(ratio);
    resampler.prepareToPlay(newLen, targetSampleRate);

    juce::AudioBuffer<float> output(nCh, newLen);
    juce::AudioSourceChannelInfo info(&output, 0, newLen);
    resampler.getNextAudioBlock(info);

    return output;
}

} // namespace DSP
