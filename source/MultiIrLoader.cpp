// SPDX-FileCopyrightText: 2026 Pier Luigi Fiorini
// SPDX-License-Identifier: MIT

#include "MultiIrLoader.h"

#include "TimeAligner.h"

#include <cmath>

namespace DSP {

namespace {

constexpr double kWeightRampSeconds = 0.02; // 20 ms — matches DualIrLoader's blend
                                            // ramp: instant to the ear, slow enough
                                            // to silence zipper noise on a drag.

} // namespace

MultiIrLoader::MultiIrLoader(int numSlots)
    : m_numSlots(juce::jlimit(1, maxSlots, numSlots))
    , m_slots(static_cast<size_t>(m_numSlots))
{
    const auto count = static_cast<size_t>(m_numSlots);

    m_weightSets[0].assign(count, 0.0f);
    m_weightSets[1].assign(count, 0.0f);

    // Start with an equal share of every slot, so a freshly loaded IR is audible
    // without the caller having to touch the weights first.
    m_requestedWeights.assign(count, 1.0f);

    m_weightStart.allocate(count, true);
    m_weightEnd.allocate(count, true);
    m_slotActive.allocate(count, true);

    for (auto &slot : m_slots) {
        // juce::dsp::Gain wraps a linear juce::SmoothedValue, whose default value
        // is 0 — i.e. silence. Seed every slot at unity.
        slot.gain.setGainLinear(1.0f);
    }

    publishWeights();
}

int MultiIrLoader::getNumSlots() const noexcept
{
    return m_numSlots;
}

bool MultiIrLoader::isValidSlot(int slot) const noexcept
{
    return slot >= 0 && slot < m_numSlots;
}

//==============================================================================

void MultiIrLoader::prepare(const juce::dsp::ProcessSpec &spec)
{
    m_processingRate = spec.sampleRate;

    for (auto &slot : m_slots) {
        slot.ir.prepare(spec);
        slot.filter.prepare(spec);
        slot.gain.prepare(spec);
        slot.gain.setRampDurationSeconds(kWeightRampSeconds);
    }

    // IrLoader defers building its convolvers until it has been prepared, so an IR
    // loaded before this call only becomes visible to isLoaded() just now. Weights
    // normalise over the loaded slots, so republish here — otherwise a caller that
    // loads before preparing would be left with the all-zero set published while
    // every slot still looked empty, and process() would output silence forever.
    publishWeights();

    const int ws = m_activeWeightSet.load(std::memory_order_acquire);

    for (size_t i = 0; i < m_slots.size(); ++i) {
        // Seat the smoother on the published weight so that weights set before
        // prepare() take effect immediately instead of ramping up from zero.
        m_slots[i].weightSmoother.reset(spec.sampleRate, kWeightRampSeconds);
        m_slots[i].weightSmoother.setCurrentAndTargetValue(m_weightSets[ws][i]);
    }

    m_dry.setSize(
        static_cast<int>(spec.numChannels),
        static_cast<int>(spec.maximumBlockSize),
        false,
        false,
        true);
    m_dry.clear();

    m_wet.setSize(
        static_cast<int>(spec.numChannels),
        static_cast<int>(spec.maximumBlockSize),
        false,
        false,
        true);
    m_wet.clear();

    m_blockChannelPtrs.allocate(spec.numChannels, false);
}

void MultiIrLoader::reset()
{
    for (auto &slot : m_slots) {
        slot.ir.reset();
        slot.filter.reset();
        slot.gain.reset();
        slot.weightSmoother.setCurrentAndTargetValue(slot.weightSmoother.getTargetValue());
    }

    m_dry.clear();
    m_wet.clear();
}

void MultiIrLoader::process(juce::dsp::ProcessContextReplacing<float> &context)
{
    if (context.isBypassed)
        return;

    auto &block = context.getOutputBlock();
    const auto numSamples = block.getNumSamples();
    const auto numCh = block.getNumChannels();

    // Clamp to what prepare() provisioned. A host handing us more channels than
    // spec.numChannels gets its extra channels left untouched rather than indexed
    // out of bounds — same guard as DualIrLoader.
    const auto channels = std::min(numCh, static_cast<size_t>(m_dry.getNumChannels()));
    if (channels == 0)
        return;

    jassert(numSamples <= static_cast<size_t>(m_dry.getNumSamples()));

    // Snapshot the published weights once. The writer fills the inactive row
    // before flipping the index, so a single publish can never be observed
    // half-applied. Two publishes while this block is in flight would reach the
    // row being read here, which the intended one-publish-per-drag-frame usage
    // does not do; the worst case is a mix one frame stale, never a torn one.
    const int ws = m_activeWeightSet.load(std::memory_order_acquire);
    const auto &weights = m_weightSets[ws];

    // Advance every smoother, including those belonging to skipped slots, so that
    // smoother state always tracks real time.
    for (size_t i = 0; i < m_slots.size(); ++i) {
        auto &smoother = m_slots[i].weightSmoother;
        smoother.setTargetValue(weights[i]);
        m_weightStart[i] = smoother.getCurrentValue();
        m_weightEnd[i] = smoother.skip(static_cast<int>(numSamples));
    }

    bool anyLoaded = false;
    int activeCount = 0;

    for (size_t i = 0; i < m_slots.size(); ++i) {
        const bool loaded = m_slots[i].ir.isLoaded();
        anyLoaded = anyLoaded || loaded;

        // A slot contributes nothing when its weight is zero across the entire
        // block. Skipping it is safe against IrLoader's two-stage convolver: the
        // background handshake uses auto-reset WaitableEvents, so a skipped block
        // simply leaves the done event signalled for the next process() call to
        // consume. Do not "fix" this by force-processing muted slots.
        const bool silent = juce::exactlyEqual(m_weightStart[i], 0.0f)
                            && juce::exactlyEqual(m_weightEnd[i], 0.0f);
        m_slotActive[i] = loaded && !silent;
        if (m_slotActive[i])
            ++activeCount;
    }

    // Nothing loaded at all: pass the dry signal through untouched.
    if (!anyLoaded)
        return;

    auto outBlock = block.getSubsetChannelBlock(0, channels).getSubBlock(0, numSamples);

    // Loaded, but every slot is muted: the weighted sum is silence.
    if (activeCount == 0) {
        outBlock.clear();
        return;
    }

    juce::dsp::AudioBlock<float> dryStorage{m_dry};
    auto dry = dryStorage.getSubsetChannelBlock(0, channels).getSubBlock(0, numSamples);

    // Only needed when a second slot has to consume the input after the first
    // slot has overwritten the block in place.
    if (activeCount > 1)
        dry.copyFrom(outBlock);

    juce::dsp::AudioBlock<float> wetStorage{m_wet};
    auto wet = wetStorage.getSubsetChannelBlock(0, channels).getSubBlock(0, numSamples);

    // Non-owning view over the output block, so the SIMD ramp helpers can be used
    // without allocating on the audio thread.
    for (size_t ch = 0; ch < channels; ++ch)
        m_blockChannelPtrs[ch] = block.getChannelPointer(ch);

    juce::AudioBuffer<float> outBuf(
        m_blockChannelPtrs.getData(), static_cast<int>(channels), static_cast<int>(numSamples));

    const auto intChannels = static_cast<int>(channels);
    const auto intSamples = static_cast<int>(numSamples);

    bool first = true;

    for (size_t i = 0; i < m_slots.size(); ++i) {
        if (!m_slotActive[i])
            continue;

        auto &slot = m_slots[i];

        if (first) {
            // The block still holds the dry input, so the first active slot runs
            // in place and its weight becomes a single gain ramp — no copy, and
            // no need to clear the block first.
            juce::dsp::ProcessContextReplacing<float> ctx{outBlock};
            slot.ir.process(ctx);
            slot.filter.process(ctx);
            slot.gain.process(ctx);

            for (int ch = 0; ch < intChannels; ++ch)
                outBuf.applyGainRamp(ch, 0, intSamples, m_weightStart[i], m_weightEnd[i]);

            first = false;
        } else {
            wet.copyFrom(dry);

            juce::dsp::ProcessContextReplacing<float> ctx{wet};
            slot.ir.process(ctx);
            slot.filter.process(ctx);
            slot.gain.process(ctx);

            for (int ch = 0; ch < intChannels; ++ch)
                outBuf.addFromWithRamp(
                    ch, 0, m_wet.getReadPointer(ch), intSamples, m_weightStart[i], m_weightEnd[i]);
        }
    }
}

//==============================================================================

bool MultiIrLoader::loadImpulseResponse(int slot, const juce::File &irFile)
{
    if (!isValidSlot(slot))
        return false;

    auto &s = m_slots[static_cast<size_t>(slot)];
    if (!s.ir.loadImpulseResponse(irFile))
        return false;

    s.rawIr = s.ir.getRawIrBuffer();
    s.sourceRate = s.ir.getSourceSampleRate();
    s.file = irFile;

    // Weights normalise over the loaded slots, so load state changes the mix.
    publishWeights();
    return true;
}

bool MultiIrLoader::loadImpulseResponse(
    int slot, const juce::AudioBuffer<float> &ir, double sourceSampleRate)
{
    if (!isValidSlot(slot))
        return false;

    auto &s = m_slots[static_cast<size_t>(slot)];
    if (!s.ir.loadImpulseResponse(ir, sourceSampleRate))
        return false;

    s.rawIr = s.ir.getRawIrBuffer();
    s.sourceRate = s.ir.getSourceSampleRate();

    publishWeights();
    return true;
}

void MultiIrLoader::clearImpulseResponse(int slot) noexcept
{
    if (!isValidSlot(slot))
        return;

    auto &s = m_slots[static_cast<size_t>(slot)];
    s.ir.clearImpulseResponse();
    s.file = juce::File{};
    s.rawIr.setSize(0, 0);
    s.sourceRate = 0.0;

    publishWeights();
}

void MultiIrLoader::clearAllImpulseResponses() noexcept
{
    for (auto &slot : m_slots) {
        slot.ir.clearImpulseResponse();
        slot.file = juce::File{};
        slot.rawIr.setSize(0, 0);
        slot.sourceRate = 0.0;
    }

    publishWeights();
}

void MultiIrLoader::setNormalise(bool normalise) noexcept
{
    m_normalise = normalise;
    for (auto &slot : m_slots)
        slot.ir.setNormalise(normalise);
}

void MultiIrLoader::applyAlignment(int slot, float delayMs, bool invertPolarity)
{
    if (!isValidSlot(slot))
        return;

    auto &s = m_slots[static_cast<size_t>(slot)];
    if (s.rawIr.getNumSamples() == 0)
        return;

    // Work in source-rate samples; IrLoader::loadImpulseResponse handles resampling.
    const double workRate = (s.sourceRate > 0.0)
                                ? s.sourceRate
                                : (m_processingRate > 0.0 ? m_processingRate : 44100.0);

    const auto working = TimeAligner::applyAlignment(s.rawIr, workRate, delayMs, invertPolarity);

    // s.rawIr is already normalised, so re-normalising here would scale it twice.
    // Restore the configured setting rather than assuming it was enabled.
    s.ir.setNormalise(false);
    s.ir.loadImpulseResponse(working, workRate);
    s.ir.setNormalise(m_normalise);
}

//==============================================================================

void MultiIrLoader::publishWeights() noexcept
{
    const int current = m_activeWeightSet.load(std::memory_order_relaxed);
    const int build = (current == 0) ? 1 : 0;
    auto &target = m_weightSets[build];

    double sum = 0.0;

    for (size_t i = 0; i < m_requestedWeights.size(); ++i) {
        float weight = m_requestedWeights[i];

        if (!std::isfinite(weight) || weight < 0.0f)
            weight = 0.0f;
        m_requestedWeights[i] = weight;

        // An empty slot must not consume a share of the mix, otherwise loading
        // two IRs out of eight would leave the output at a quarter level.
        if (!m_slots[i].ir.isLoaded())
            weight = 0.0f;

        target[i] = weight;
        sum += weight;
    }

    if (sum > 0.0) {
        const auto scale = static_cast<float>(1.0 / sum);
        for (auto &weight : target)
            weight *= scale;
    } else {
        // Nothing loaded, or every requested weight is zero: silence.
        for (auto &weight : target)
            weight = 0.0f;
    }

    m_activeWeightSet.store(build, std::memory_order_release);
}

void MultiIrLoader::setWeights(std::span<const float> weights) noexcept
{
    const auto count = std::min(weights.size(), m_requestedWeights.size());

    for (size_t i = 0; i < count; ++i)
        m_requestedWeights[i] = weights[i];

    // A short span zeroes the slots it does not cover, so setWeights() always
    // replaces the whole mix rather than merging into the previous one.
    for (size_t i = count; i < m_requestedWeights.size(); ++i)
        m_requestedWeights[i] = 0.0f;

    publishWeights();
}

void MultiIrLoader::setWeightPercent(int slot, float percent) noexcept
{
    if (!isValidSlot(slot))
        return;

    m_requestedWeights[static_cast<size_t>(slot)] = percent;
    publishWeights();
}

float MultiIrLoader::getWeightPercent(int slot) const noexcept
{
    if (!isValidSlot(slot))
        return 0.0f;

    const int ws = m_activeWeightSet.load(std::memory_order_acquire);
    return m_weightSets[ws][static_cast<size_t>(slot)] * 100.0f;
}

std::vector<float> MultiIrLoader::getWeightPercentages() const noexcept
{
    const int ws = m_activeWeightSet.load(std::memory_order_acquire);

    std::vector<float> out(m_weightSets[ws].size());
    for (size_t i = 0; i < out.size(); ++i)
        out[i] = m_weightSets[ws][i] * 100.0f;
    return out;
}

//==============================================================================

void MultiIrLoader::setHighPassFrequency(int slot, float frequency) noexcept
{
    if (isValidSlot(slot))
        m_slots[static_cast<size_t>(slot)].filter.setHighPassFrequency(frequency);
}

void MultiIrLoader::setLowPassFrequency(int slot, float frequency) noexcept
{
    if (isValidSlot(slot))
        m_slots[static_cast<size_t>(slot)].filter.setLowPassFrequency(frequency);
}

void MultiIrLoader::setGain(int slot, float decibels) noexcept
{
    if (isValidSlot(slot))
        m_slots[static_cast<size_t>(slot)].gain.setGainDecibels(decibels);
}

float MultiIrLoader::getHighPassFrequency(int slot) const noexcept
{
    if (!isValidSlot(slot))
        return 0.0f;
    return m_slots[static_cast<size_t>(slot)].filter.getHighPassFrequency();
}

float MultiIrLoader::getLowPassFrequency(int slot) const noexcept
{
    if (!isValidSlot(slot))
        return 0.0f;
    return m_slots[static_cast<size_t>(slot)].filter.getLowPassFrequency();
}

//==============================================================================

bool MultiIrLoader::isSlotLoaded(int slot) const noexcept
{
    if (!isValidSlot(slot))
        return false;
    return m_slots[static_cast<size_t>(slot)].ir.isLoaded();
}

int MultiIrLoader::getNumLoadedSlots() const noexcept
{
    int loaded = 0;
    for (const auto &slot : m_slots)
        if (slot.ir.isLoaded())
            ++loaded;
    return loaded;
}

//==============================================================================

const juce::File &MultiIrLoader::getImpulseResponseFile(int slot) const noexcept
{
    if (!isValidSlot(slot))
        return m_emptyFile;
    return m_slots[static_cast<size_t>(slot)].file;
}

const juce::AudioBuffer<float> &MultiIrLoader::getRawIrBuffer(int slot) const noexcept
{
    if (!isValidSlot(slot))
        return m_emptyIr;
    return m_slots[static_cast<size_t>(slot)].rawIr;
}

double MultiIrLoader::getIrSourceRate(int slot) const noexcept
{
    if (!isValidSlot(slot))
        return 0.0;
    return m_slots[static_cast<size_t>(slot)].sourceRate;
}

double MultiIrLoader::getTailLengthSeconds() const noexcept
{
    double tail = 0.0;
    for (const auto &slot : m_slots)
        if (slot.sourceRate > 0.0 && slot.rawIr.getNumSamples() > 0)
            tail = std::max(tail, slot.rawIr.getNumSamples() / slot.sourceRate);
    return tail;
}

} // namespace DSP
