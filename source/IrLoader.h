// SPDX-FileCopyrightText: 2026 Pier Luigi Fiorini
// SPDX-License-Identifier: MIT

#pragma once

#include <FFTConvolver.h>
#include <TwoStageFFTConvolver.h>

#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_dsp/juce_dsp.h>

#include <atomic>
#include <memory>
#include <vector>

namespace DSP {

/**
 * JUCE-style DSP wrapper for partitioned FFT convolution.
 *
 * Follows the juce::dsp module contract: prepare() / reset() / process().
 *
 * Convolver selection is automatic and based on the loaded IR length:
 *   - Short IR (≤ 16 384 samples) — fftconvolver::FFTConvolver, processed entirely
 *     on the audio thread with a uniform partition size.
 *   - Long IR  (> 16 384 samples) — fftconvolver::TwoStageFFTConvolver, where the
 *     head is processed on the audio thread and the tail runs on a dedicated
 *     high-priority background thread via juce::Thread + juce::WaitableEvent.
 *
 * IR loading is NOT real-time safe and must be called from a non-audio thread
 * (e.g. the message thread or a background thread). While a new IR is being
 * loaded the audio thread passes the signal through unmodified.
 *
 * Normalisation (when enabled) is a two-step process applied at load time:
 *   1. Peak-normalise so the loudest sample reaches 0.8.
 *   2. Scale by 1 / sum-of-squares so the total energy equals 1.
 *
 * Channel mapping:
 *   - Mono IR   → same IR channel applied to every audio channel.
 *   - Stereo IR → audio channel n uses IR channel (n % irChannels).
 */
class IrLoader
{
public:
    explicit IrLoader(bool normalise = true);
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

    /** Unloads the current IR. The audio thread will pass signal through unmodified. */
    void clearImpulseResponse() noexcept;

    /** RT-safe. Returns true once a convolver set has been built and published. */
    [[nodiscard]] bool isLoaded() const noexcept;

    /**
     * Enables or disables peak normalisation applied when loading an IR.
     * Takes effect on the next loadImpulseResponse() call; does not retroactively
     * alter an already-loaded IR. Message-thread only.
     */
    void setNormalise(bool normalise) noexcept { m_normalise = normalise; }

    /** Message-thread only. Returns the raw IR buffer at its original source rate. */
    [[nodiscard]] const juce::AudioBuffer<float> &getRawIrBuffer() const noexcept
    {
        return m_rawIrBuffer;
    }

    /** Message-thread only. Returns the source sample rate of the stored raw IR. */
    [[nodiscard]] double getSourceSampleRate() const noexcept { return m_sourceSampleRate; }

    [[nodiscard]] static juce::AudioBuffer<float> resampleIR(
        const juce::AudioBuffer<float> &ir, double sourceSampleRate, double targetSampleRate);

    // Type-erased per-channel convolver. Concrete implementations live in IrLoader.cpp.
    // Choosing single-stage vs two-stage at IR load time avoids runtime branching in process().
    struct ConvChannel
    {
        virtual ~ConvChannel() = default;
        virtual void process(const float *in, float *out, size_t len) = 0;
        virtual void resetState() = 0;
    };

private:
    //==============================================================================
    void initConvolvers(const juce::AudioBuffer<float> &ir);

    //==============================================================================
    bool m_normalise;

    juce::dsp::ProcessSpec m_spec{};
    std::atomic<bool> m_prepared{false};
    juce::AudioFormatManager m_formatManager;

    // Double-buffered convolver sets. The message thread builds into the non-active
    // slot and atomically publishes it; the audio thread always reads the active slot.
    // This makes IR swaps lock-free and race-free for the audio thread.
    struct ConvolverSet
    {
        std::vector<std::unique_ptr<ConvChannel>> convs;
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
