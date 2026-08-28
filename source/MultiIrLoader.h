// SPDX-FileCopyrightText: 2026 Pier Luigi Fiorini
// SPDX-License-Identifier: MIT

#pragma once

#include "IrFilter.h"
#include "IrLoader.h"

#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_dsp/juce_dsp.h>

#include <atomic>
#include <span>
#include <vector>

namespace DSP {

/**
 * Runs the input signal through a configurable number of impulse responses in
 * parallel and sums the results with per-slot weights:
 *
 *     out = sum_i ( w_i * wet_i )        with  sum_i w_i == 1
 *
 * The slot count is fixed at construction, so the same class serves a three-way
 * blend or an eight-corner morph pad. Each slot owns an IrLoader, an IrFilter
 * (HPF + LPF) and an output Gain, so slots can be voiced independently before
 * they are mixed.
 *
 * Weights are supplied as relative, non-negative values on any scale and are
 * normalised internally, so the effective percentages always total 100. The
 * normalisation runs over the *loaded* slots only: an empty slot does not
 * consume a share of the mix, and loading or clearing a slot renormalises the
 * others. If every supplied weight is zero the output is silent — the one case
 * where the percentages total 0 rather than 100.
 *
 * setWeights() publishes every weight in one atomic update, which is what a
 * positional control (a point dragged inside a polygon, one IR per corner)
 * should call: one publish per drag frame rather than N racy stores.
 *
 * IR loading follows the same threading contract as DSP::IrLoader: non-RT
 * thread only.
 *
 * Cost model: any IR longer than 16384 samples makes IrLoader select a
 * two-stage convolver with one background tail thread *per audio channel*, each
 * with its own signal/wait handshake. process() therefore performs one
 * round-trip per channel of every active slot, serially — eight slots in stereo
 * is up to sixteen tail threads. Slots whose weight is zero across the whole
 * block are skipped entirely, which is what keeps the realistic case (two or
 * three audible slots) affordable.
 */
class MultiIrLoader
{
public:
    /** Slot count used when none is given — eight corners of a morph pad. */
    static constexpr int defaultNumSlots = 8;

    /** Upper bound on the slot count. A sanity cap, not a design limit. */
    static constexpr int maxSlots = 64;

    /**
     * @param numSlots Number of IR slots, clamped to [1, maxSlots]. Fixed for the
     *                 lifetime of the object, so every allocation that scales with
     *                 the slot count happens here. prepare() still sizes the
     *                 channel scratch buffers; process() allocates nothing.
     */
    explicit MultiIrLoader(int numSlots = defaultNumSlots);
    ~MultiIrLoader() = default;

    /** Any thread. The slot count this instance was constructed with. */
    [[nodiscard]] int getNumSlots() const noexcept;

    //==============================================================================
    // JUCE DSP module interface

    /** Allocates and initialises all slot state. Call before any processing. */
    void prepare(const juce::dsp::ProcessSpec &spec);

    /** Clears convolver, filter, gain and weight-smoother state in every slot. */
    void reset();

    /**
     * Audio-thread safe. Mixes the active slots in place.
     *
     * If no slot has an IR loaded the block is left untouched (dry passthrough,
     * matching DSP::DualIrLoader). If at least one slot is loaded but every
     * weight is zero the block is silenced.
     *
     * Slots whose weight is zero at both the start and the end of the block are
     * skipped: no convolution, no filtering, no gain. Their convolver tails do
     * not advance while they are muted, so a slot re-entering from exactly zero
     * starts with an empty tail — inaudible, because it fades up from silence.
     * The per-slot weight smoothers are advanced unconditionally, so their state
     * always tracks real time.
     */
    void process(juce::dsp::ProcessContextReplacing<float> &context);

    //==============================================================================
    // IR loading — non-audio thread only. Out-of-range slots are no-ops.

    /** @return false if `slot` is out of range or the file could not be read. */
    bool loadImpulseResponse(int slot, const juce::File &irFile);

    /** @return false if `slot` is out of range or the buffer is empty. */
    bool loadImpulseResponse(int slot, const juce::AudioBuffer<float> &ir, double sourceSampleRate);

    /** Unloads one slot and renormalises the weights of those that remain. */
    void clearImpulseResponse(int slot) noexcept;

    /** Unloads every slot. The requested weights are left untouched. */
    void clearAllImpulseResponses() noexcept;

    /**
     * Enables or disables peak normalisation applied to every slot when loading.
     * Takes effect on the next loadImpulseResponse() call; does not retroactively
     * alter already-loaded IRs. Message-thread only.
     */
    void setNormalise(bool normalise) noexcept;

    /**
     * Applies a time-alignment shift and optional polarity inversion to one slot.
     * Re-derives a shifted copy from that slot's stored raw IR, so repeated calls
     * start from the original rather than compounding. Message-thread only; a
     * no-op if `slot` is out of range or holds no IR.
     *
     * @param delayMs        Delay in milliseconds. Positive = this slot lags the
     *                       reference (prepend silence). Negative = it leads
     *                       (trim onset).
     * @param invertPolarity Negate all samples when true.
     */
    void applyAlignment(int slot, float delayMs, bool invertPolarity);

    //==============================================================================
    // Mix weights. Any thread, but callers must serialise these setters among
    // themselves — the published weight set has a single writer.

    /**
     * Publishes every weight as one atomic update. Values are relative and may
     * use any non-negative scale; negative and non-finite entries are treated as
     * zero. A span shorter than getNumSlots() leaves the remaining slots at zero;
     * a longer one is truncated.
     *
     * This is the entry point a positional control should use, so that one drag
     * frame produces exactly one publish and process() can never observe a
     * half-updated mix.
     */
    void setWeights(std::span<const float> weights) noexcept;

    /**
     * Publishes one slot's relative weight, leaving the others as they are and
     * renormalising the whole set.
     *
     * `percent` is a relative weight, NOT a guaranteed share of the output:
     * getWeightPercent(slot) reads back `percent` only when this slot is loaded
     * and the requested weights of the other *loaded* slots happen to total
     * (100 - percent). Prefer setWeights() when changing more than one slot.
     * A no-op if `slot` is out of range.
     */
    void setWeightPercent(int slot, float percent) noexcept;

    /**
     * Any thread. The effective, normalised contribution of a slot, in 0..100.
     * Across all slots these total 100, except when every requested weight is
     * zero (or nothing is loaded), in which case all are 0 and the output is
     * silent. Returns 0 if `slot` is out of range.
     */
    [[nodiscard]] float getWeightPercent(int slot) const noexcept;

    /** Message-thread only — allocates. Snapshot of every effective percentage. */
    [[nodiscard]] std::vector<float> getWeightPercentages() const noexcept;

    //==============================================================================
    // Per-slot voicing. Any thread — each publishes a target without blocking.
    // Out-of-range slots are no-ops.

    /** Publishes a slot's high-pass cutoff target. */
    void setHighPassFrequency(int slot, float frequency) noexcept;
    /** Publishes a slot's low-pass cutoff target. */
    void setLowPassFrequency(int slot, float frequency) noexcept;
    /** Publishes a slot's output gain target, in decibels. */
    void setGain(int slot, float decibels) noexcept;

    [[nodiscard]] float getHighPassFrequency(int slot) const noexcept;
    [[nodiscard]] float getLowPassFrequency(int slot) const noexcept;

    //==============================================================================
    // Audio-thread safe queries.

    /**
     * True once this slot has a convolver set built and published. Note that
     * IrLoader only builds convolvers once prepared, so a slot loaded before
     * prepare() reports false until prepare() runs.
     */
    [[nodiscard]] bool isSlotLoaded(int slot) const noexcept;

    /** Number of slots currently reporting isSlotLoaded(). */
    [[nodiscard]] int getNumLoadedSlots() const noexcept;

    //==============================================================================
    // Message-thread only — not RT-safe. Out-of-range slots return empty objects.

    [[nodiscard]] const juce::File &getImpulseResponseFile(int slot) const noexcept;

    /** The stored raw IR, at its original source rate and already normalised. */
    [[nodiscard]] const juce::AudioBuffer<float> &getRawIrBuffer(int slot) const noexcept;

    [[nodiscard]] double getIrSourceRate(int slot) const noexcept;

    /** Length of the longest IR across all loaded slots, in seconds. */
    [[nodiscard]] double getTailLengthSeconds() const noexcept;

private:
    //==============================================================================
    [[nodiscard]] bool isValidSlot(int slot) const noexcept;

    /** Renormalises the requested weights and publishes them. Non-RT, single writer. */
    void publishWeights() noexcept;

    //==============================================================================
    struct Slot
    {
        IrLoader ir;
        IrFilter filter;
        juce::dsp::Gain<float> gain;

        juce::LinearSmoothedValue<float> weightSmoother; // audio thread only

        juce::File file;                // message thread only
        juce::AudioBuffer<float> rawIr; // message thread only, post-normalisation
        double sourceRate = 0.0;        // message thread only
    };

    const int m_numSlots;

    // Never resized after construction: Slot is non-movable (IrLoader and IrFilter
    // are JUCE_DECLARE_NON_COPYABLE), so any push_back/resize is a compile error
    // rather than undefined behaviour.
    std::vector<Slot> m_slots;

    // Scratch, sized in prepare() and reused. m_dry holds the untouched input for
    // the second and later slots; m_wet is the per-slot wet buffer, reused
    // serially, so scratch memory does not scale with the slot count.
    juce::AudioBuffer<float> m_dry;
    juce::AudioBuffer<float> m_wet;
    juce::HeapBlock<float *> m_blockChannelPtrs; // sized in prepare()

    // Per-block scratch, allocated in the constructor so process() never does.
    juce::HeapBlock<float> m_weightStart;
    juce::HeapBlock<float> m_weightEnd;
    juce::HeapBlock<bool> m_slotActive;

    // Double-buffered normalised weights. The writer fills the inactive row and
    // publishes the index, mirroring IrLoader's m_sets / m_activeSet. This makes a
    // bulk setWeights() one atomic update instead of N independent stores, which
    // would let process() observe a half-updated mix during a drag.
    std::vector<float> m_weightSets[2];
    std::atomic<int> m_activeWeightSet{0}; // writer → audio

    std::vector<float> m_requestedWeights; // message thread only, pre-normalisation

    bool m_normalise = true;       // message thread only
    double m_processingRate = 0.0; // cached in prepare()

    // Returned by the reference-getters for an out-of-range slot. Instance members
    // rather than namespace-scope statics: a static juce::AudioBuffer in a unity
    // build interacts badly with the JUCE leak detector's destruction order.
    const juce::File m_emptyFile{};
    const juce::AudioBuffer<float> m_emptyIr{};

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MultiIrLoader)
};

} // namespace DSP
