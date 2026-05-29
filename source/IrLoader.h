//======================================================================================
// Copyright (c) 2026 Pier Luigi Fiorini
// All rights reserved.
//======================================================================================

#pragma once

#include <FFTConvolver.h>

#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_dsp/juce_dsp.h>

#include <atomic>
#include <memory>
#include <vector>

namespace DSP {

/**
 * JUCE-style DSP wrapper around fftconvolver::FFTConvolver.
 *
 * Follows the juce::dsp module contract: prepare() / reset() / process().
 *
 * IR loading is NOT real-time safe and must be called from a non-audio thread
 * (e.g. the message thread or a background thread). While a new IR is being
 * loaded the audio thread passes the signal through unmodified.
 *
 * Channel mapping:
 *   - Mono IR   → same IR channel applied to every audio channel.
 *   - Stereo IR → audio channel n uses IR channel (n % irChannels).
 */
class IrLoader
{
public:
    IrLoader();
    ~IrLoader() = default;

    //==============================================================================
    // JUCE DSP module interface

    /** Allocates and initialises all internal state. Call before any processing. */
    void prepare(const juce::dsp::ProcessSpec &spec);

    /** Clears convolver delay lines while retaining the loaded IR. */
    void reset();

    /** Convolves the block in-place. No-op if no IR has been loaded yet. */
    void process(juce::dsp::ProcessContextReplacing<float> &context);

    //==============================================================================
    // IR loading — non-audio thread only

    /**
     * Loads an impulse response from a file.
     * Supports any format registered by juce::AudioFormatManager::registerBasicFormats().
     * @return true on success.
     */
    bool loadImpulseResponse(const juce::File &irFile);

    /**
     * Loads an impulse response from an AudioBuffer.
     * The buffer is resampled automatically if sourceSampleRate differs from the
     * current processing sample rate.
     * @return true on success.
     */
    bool loadImpulseResponse(const juce::AudioBuffer<float> &ir, double sourceSampleRate);

private:
    //==============================================================================
    void initConvolvers(const juce::AudioBuffer<float> &ir);

    [[nodiscard]] static juce::AudioBuffer<float> resampleIR(
        const juce::AudioBuffer<float> &ir, double sourceSampleRate, double targetSampleRate);

    //==============================================================================
    juce::dsp::ProcessSpec m_spec{};
    std::atomic<bool> m_prepared{false};
    juce::AudioFormatManager m_formatManager;

    // Double-buffered convolver sets. The message thread builds into the non-active
    // slot and atomically publishes it; the audio thread always reads the active slot.
    // This makes IR swaps lock-free and race-free for the audio thread.
    struct ConvolverSet
    {
        std::vector<std::unique_ptr<fftconvolver::FFTConvolver>> convs;
    };
    ConvolverSet m_sets[2];
    std::atomic<int> m_activeSet{-1}; // -1 = no IR ready; 0/1 = active slot index

    juce::AudioBuffer<float> m_rawIrBuffer; // stored at m_sourceSampleRate
    double m_sourceSampleRate = 0.0;
    std::atomic<bool> m_irStored{false}; // true once m_rawIrBuffer has been populated

    //==============================================================================
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(IrLoader)
};

} // namespace DSP
