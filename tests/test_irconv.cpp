// SPDX-FileCopyrightText: 2026 Pier Luigi Fiorini
// SPDX-License-Identifier: MIT

#include <irconv/irconv.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

// ---------------------------------------------------------------------------
// Minimal test harness
// ---------------------------------------------------------------------------

static int g_failures = 0;

#define EXPECT(cond) \
    do { \
        if (!(cond)) { \
            std::fprintf(stderr, "  FAIL: %s  (%s:%d)\n", #cond, __FILE__, __LINE__); \
            ++g_failures; \
        } \
    } while (false)

#define EXPECT_NEAR(a, b, eps) EXPECT(std::fabs((a) - (b)) <= (eps))
#define EXPECT_TRUE(x) EXPECT(x)
#define EXPECT_FALSE(x) EXPECT(!(x))
#define EXPECT_GT(a, b) EXPECT((a) > (b))

static void beginTest(const char *name)
{
    std::printf("  %s\n", name);
}

// ---------------------------------------------------------------------------
// TimeAligner tests
// ---------------------------------------------------------------------------

static void test_TimeAligner_ZeroDelayAlignment()
{
    beginTest("TimeAligner / ZeroDelayAlignment");

    juce::AudioBuffer<float> ir(1, 1024);
    ir.clear();
    ir.setSample(0, 64, 1.0f);

    const auto result = DSP::TimeAligner::analyse(ir, ir, 44100.0);
    EXPECT_NEAR(result.delaySamples, 0.0f, 0.5f);
    EXPECT_FALSE(result.invertPolarity);
    EXPECT_GT(result.correlationScore, 0.9f);
}

static void test_TimeAligner_KnownDelayAlignment()
{
    beginTest("TimeAligner / KnownDelayAlignment");

    juce::AudioBuffer<float> irA(1, 1024), irB(1, 1024);
    irA.clear();
    irB.clear();
    irA.setSample(0, 50, 1.0f);
    irB.setSample(0, 60, 1.0f); // 10 samples later

    const auto result = DSP::TimeAligner::analyse(irA, irB, 44100.0);
    EXPECT_NEAR(result.delaySamples, 10.0f, 0.5f);
    EXPECT_FALSE(result.invertPolarity);
}

static void test_TimeAligner_PolarityInversionDetected()
{
    beginTest("TimeAligner / PolarityInversionDetected");

    juce::AudioBuffer<float> irA(1, 1024), irB(1, 1024);
    irA.clear();
    irB.clear();
    irA.setSample(0, 50, 1.0f);
    irB.setSample(0, 50, -1.0f); // same position, inverted

    const auto result = DSP::TimeAligner::analyse(irA, irB, 44100.0);
    EXPECT_NEAR(result.delaySamples, 0.0f, 0.5f);
    EXPECT_TRUE(result.invertPolarity);
}

// ---------------------------------------------------------------------------
// IrLoader tests
// ---------------------------------------------------------------------------

static void test_IrLoader_LoadFromBuffer()
{
    beginTest("IrLoader / LoadFromBuffer");

    DSP::IrLoader loader(/*normalise=*/false);

    juce::dsp::ProcessSpec spec{44100.0, 512, 1};
    loader.prepare(spec);

    juce::AudioBuffer<float> ir(1, 2048);
    ir.clear();
    ir.setSample(0, 0, 1.0f);

    EXPECT_TRUE(loader.loadImpulseResponse(ir, 44100.0));
    EXPECT_TRUE(loader.isLoaded());
}

static void test_IrLoader_ClearUnloadsConvolver()
{
    beginTest("IrLoader / ClearUnloadsConvolver");

    DSP::IrLoader loader(false);

    juce::dsp::ProcessSpec spec{44100.0, 512, 1};
    loader.prepare(spec);

    juce::AudioBuffer<float> ir(1, 512);
    ir.clear();
    ir.setSample(0, 0, 1.0f);
    loader.loadImpulseResponse(ir, 44100.0);
    EXPECT(loader.isLoaded());

    loader.clearImpulseResponse();
    EXPECT_FALSE(loader.isLoaded());
}

// ---------------------------------------------------------------------------
// DualIrLoader tests
// ---------------------------------------------------------------------------

static void test_DualIrLoader_AlignmentDoesNotChangeLevel()
{
    beginTest("DualIrLoader / AlignmentDoesNotChangeLevel");

    DSP::DualIrLoader dl;

    juce::dsp::ProcessSpec spec{44100.0, 512, 2};
    dl.prepare(spec);

    juce::AudioBuffer<float> ir(1, 1024);
    ir.clear();
    ir.setSample(0, 0, 1.0f);
    dl.loadImpulseResponseA(ir, 44100.0);

    const auto &rawBefore = dl.getRawIrABuffer();
    const float peakBefore = rawBefore.getMagnitude(0, 0, rawBefore.getNumSamples());

    // Negative delay: trim IR onset — most likely to expose a double-normalise bug.
    dl.applyAlignmentToIrA(-5.0f /*ms*/, false);

    const auto &rawAfter = dl.getRawIrABuffer();
    const float peakAfter = rawAfter.getMagnitude(0, 0, rawAfter.getNumSamples());
    EXPECT_NEAR(peakBefore, peakAfter, 1e-3f);
}

static juce::AudioBuffer<float> makeIdentityIr()
{
    juce::AudioBuffer<float> ir(1, 128);
    ir.clear();
    ir.setSample(0, 0, 1.0f);
    return ir;
}

/**
 * Renders 24 blocks of 512 samples of a sine through any loader exposing the
 * juce::dsp process() contract, and returns the per-channel RMS of the final
 * block — long enough that every smoother and ramp has settled.
 */
template<typename LoaderType>
static std::array<float, 2> renderLoaderSine(
    LoaderType &loader, int numChannels, float frequency, double sampleRate)
{
    constexpr int blockSize = 512;
    constexpr int numBlocks = 24;
    juce::AudioBuffer<float> buffer(numChannels, blockSize);
    std::array<float, 2> levels{};

    for (int blockIndex = 0; blockIndex < numBlocks; ++blockIndex) {
        for (int sample = 0; sample < blockSize; ++sample) {
            const auto absoluteSample = blockIndex * blockSize + sample;
            const float value = static_cast<float>(std::sin(
                juce::MathConstants<double>::twoPi * frequency * absoluteSample / sampleRate));
            for (int channel = 0; channel < numChannels; ++channel)
                buffer.setSample(channel, sample, value);
        }

        juce::dsp::AudioBlock<float> audioBlock(buffer);
        juce::dsp::ProcessContextReplacing<float> context(audioBlock);
        loader.process(context);
    }

    for (int channel = 0; channel < numChannels; ++channel)
        levels[static_cast<size_t>(channel)] = buffer.getRMSLevel(channel, 0, blockSize);
    return levels;
}

static void test_DualIrLoader_NoIrPreservesDrySignal()
{
    beginTest("DualIrLoader / NoIrPreservesDrySignal");
    DSP::DualIrLoader loader;
    loader.prepare({48000.0, 16, 2});

    juce::AudioBuffer<float> buffer(2, 16);
    for (int channel = 0; channel < buffer.getNumChannels(); ++channel)
        for (int sample = 0; sample < buffer.getNumSamples(); ++sample)
            buffer.setSample(channel, sample, static_cast<float>(channel * 100 + sample));
    const juce::AudioBuffer<float> expected(buffer);

    juce::dsp::AudioBlock<float> block(buffer);
    juce::dsp::ProcessContextReplacing<float> context(block);
    loader.process(context);

    for (int channel = 0; channel < buffer.getNumChannels(); ++channel)
        for (int sample = 0; sample < buffer.getNumSamples(); ++sample)
            EXPECT_NEAR(buffer.getSample(channel, sample), expected.getSample(channel, sample), 0.0f);
}

static float renderSingleSlot(bool loadSlotA)
{
    constexpr double sampleRate = 48000.0;
    DSP::DualIrLoader loader;
    loader.setNormalise(false);
    loader.setHighPassFrequencyA(400.0f);
    loader.setLowPassFrequencyA(22000.0f);
    loader.setHighPassFrequencyB(10.0f);
    loader.setLowPassFrequencyB(22000.0f);
    loader.prepare({sampleRate, 512, 1});

    const auto identity = makeIdentityIr();
    const bool loaded = loadSlotA ? loader.loadImpulseResponseA(identity, sampleRate)
                                  : loader.loadImpulseResponseB(identity, sampleRate);
    EXPECT_TRUE(loaded);
    return renderLoaderSine(loader, 1, 40.0f, sampleRate)[0];
}

static void test_DualIrLoader_SingleLoadedSlotUsesItsOwnFilter()
{
    beginTest("DualIrLoader / SingleLoadedSlotUsesItsOwnFilter");
    const float filteredA = renderSingleSlot(true);
    const float openB = renderSingleSlot(false);
    EXPECT_GT(openB, filteredA * 8.0f);
}

static float renderBlendEndpoint(float blend)
{
    constexpr double sampleRate = 48000.0;
    DSP::DualIrLoader loader;
    loader.setNormalise(false);
    loader.setBlend(blend);
    loader.setHighPassFrequencyA(10.0f);
    loader.setLowPassFrequencyA(1000.0f);
    loader.setHighPassFrequencyB(10.0f);
    loader.setLowPassFrequencyB(22000.0f);
    loader.prepare({sampleRate, 512, 1});

    const auto identity = makeIdentityIr();
    EXPECT_TRUE(loader.loadImpulseResponseA(identity, sampleRate));
    EXPECT_TRUE(loader.loadImpulseResponseB(identity, sampleRate));
    return renderLoaderSine(loader, 1, 8000.0f, sampleRate)[0];
}

static void test_DualIrLoader_BlendFiltersSlotsIndependently()
{
    beginTest("DualIrLoader / BlendFiltersSlotsIndependently");
    const float filteredA = renderBlendEndpoint(0.0f);
    const float openB = renderBlendEndpoint(1.0f);
    EXPECT_GT(openB, filteredA * 8.0f);
}

static void test_DualIrLoader_StereoSplitFiltersSlotsIndependently()
{
    beginTest("DualIrLoader / StereoSplitFiltersSlotsIndependently");
    constexpr double sampleRate = 48000.0;
    DSP::DualIrLoader loader;
    loader.setNormalise(false);
    loader.setMode(DSP::DualIrLoader::StereoMode::StereoSplit);
    loader.setHighPassFrequencyA(400.0f);
    loader.setLowPassFrequencyA(22000.0f);
    loader.setHighPassFrequencyB(10.0f);
    loader.setLowPassFrequencyB(22000.0f);
    loader.prepare({sampleRate, 512, 2});

    const auto identity = makeIdentityIr();
    EXPECT_TRUE(loader.loadImpulseResponseA(identity, sampleRate));
    EXPECT_TRUE(loader.loadImpulseResponseB(identity, sampleRate));
    const auto levels = renderLoaderSine(loader, 2, 40.0f, sampleRate);
    EXPECT_GT(levels[1], levels[0] * 8.0f);
}

static float processSine(DSP::IrFilter &filter, float frequency, double sampleRate)
{
    constexpr int numSamples = 8192;
    juce::AudioBuffer<float> buffer(1, numSamples);
    for (int sample = 0; sample < numSamples; ++sample)
        buffer.setSample(
            0,
            sample,
            static_cast<float>(
                std::sin(juce::MathConstants<double>::twoPi * frequency * sample / sampleRate)));

    juce::dsp::AudioBlock<float> block(buffer);
    juce::dsp::ProcessContextReplacing<float> context(block);
    filter.process(context);
    return buffer.getRMSLevel(0, numSamples / 2, numSamples / 2);
}

static void test_IrFilter_PassesMidBandAndRejectsExtremes()
{
    beginTest("IrFilter / PassesMidBandAndRejectsExtremes");
    constexpr double sampleRate = 48000.0;
    DSP::IrFilter filter;
    filter.setHighPassFrequency(400.0f);
    filter.setLowPassFrequency(6000.0f);
    filter.prepare({sampleRate, 8192, 1});

    const float low = processSine(filter, 40.0f, sampleRate);
    filter.reset();
    const float mid = processSine(filter, 1000.0f, sampleRate);
    filter.reset();
    const float high = processSine(filter, 16000.0f, sampleRate);
    EXPECT_GT(mid, low * 5.0f);
    EXPECT_GT(mid, high * 3.0f);
}

static void test_IrFilter_ClampsCutoffBelowNyquist()
{
    beginTest("IrFilter / ClampsCutoffBelowNyquist");
    DSP::IrFilter filter;
    filter.setLowPassFrequency(22000.0f);
    filter.prepare({16000.0, 256, 1});
    const float level = processSine(filter, 1000.0f, 16000.0);
    EXPECT_GT(level, 0.5f);
}

// ---------------------------------------------------------------------------
// MultiIrLoader tests
// ---------------------------------------------------------------------------

// A sine whose period divides the 512-sample render block exactly, so the RMS of
// the final block is 1/sqrt(2) for a full-scale input.
static constexpr float kMultiTestFrequency = 1500.0f;
static constexpr float kFullScaleSineRms = 0.70710678f;

static juce::AudioBuffer<float> makeInvertedIdentityIr()
{
    auto ir = makeIdentityIr();
    ir.applyGain(-1.0f);
    return ir;
}

static void test_MultiIrLoader_ConstructorClampsSlotCount()
{
    beginTest("MultiIrLoader / ConstructorClampsSlotCount");

    EXPECT(DSP::MultiIrLoader{}.getNumSlots() == DSP::MultiIrLoader::defaultNumSlots);
    EXPECT(DSP::MultiIrLoader{3}.getNumSlots() == 3);
    EXPECT(DSP::MultiIrLoader{1}.getNumSlots() == 1);

    // Out-of-range counts clamp to [1, maxSlots] rather than allocating nothing.
    EXPECT(DSP::MultiIrLoader{0}.getNumSlots() == 1);
    EXPECT(DSP::MultiIrLoader{-7}.getNumSlots() == 1);
    EXPECT(DSP::MultiIrLoader{100000}.getNumSlots() == DSP::MultiIrLoader::maxSlots);

    // The weight snapshot is always one entry per slot.
    DSP::MultiIrLoader loader(3);
    EXPECT(loader.getWeightPercentages().size() == 3u);
}

static void test_MultiIrLoader_WeightsNormaliseToOneHundred()
{
    beginTest("MultiIrLoader / WeightsNormaliseToOneHundred");
    constexpr double sampleRate = 48000.0;

    DSP::MultiIrLoader loader(2);
    loader.prepare({sampleRate, 512, 1});

    const auto identity = makeIdentityIr();
    EXPECT_TRUE(loader.loadImpulseResponse(0, identity, sampleRate));
    EXPECT_TRUE(loader.loadImpulseResponse(1, identity, sampleRate));

    const float requested[2] = {1.0f, 3.0f};
    loader.setWeights(requested);
    EXPECT_NEAR(loader.getWeightPercent(0), 25.0f, 1e-3f);
    EXPECT_NEAR(loader.getWeightPercent(1), 75.0f, 1e-3f);

    // Relative, so any scale of the same ratio gives the same percentages.
    const float scaled[2] = {20.0f, 60.0f};
    loader.setWeights(scaled);
    EXPECT_NEAR(loader.getWeightPercent(0), 25.0f, 1e-3f);
    EXPECT_NEAR(loader.getWeightPercent(1), 75.0f, 1e-3f);

    const auto percentages = loader.getWeightPercentages();
    EXPECT(percentages.size() == 2u);
    EXPECT_NEAR(percentages[0] + percentages[1], 100.0f, 1e-3f);
}

static void test_MultiIrLoader_NegativeAndNonFiniteWeightsAreZero()
{
    beginTest("MultiIrLoader / NegativeAndNonFiniteWeightsAreZero");
    constexpr double sampleRate = 48000.0;

    DSP::MultiIrLoader loader(3);
    loader.prepare({sampleRate, 512, 1});

    const auto identity = makeIdentityIr();
    for (int slot = 0; slot < 3; ++slot)
        EXPECT_TRUE(loader.loadImpulseResponse(slot, identity, sampleRate));

    const float requested[3] = {-1.0f, std::numeric_limits<float>::quiet_NaN(), 2.0f};
    loader.setWeights(requested);

    EXPECT_NEAR(loader.getWeightPercent(0), 0.0f, 1e-3f);
    EXPECT_NEAR(loader.getWeightPercent(1), 0.0f, 1e-3f);
    EXPECT_NEAR(loader.getWeightPercent(2), 100.0f, 1e-3f);
}

static void test_MultiIrLoader_NormalisationCoversLoadedSlotsOnly()
{
    beginTest("MultiIrLoader / NormalisationCoversLoadedSlotsOnly");
    constexpr double sampleRate = 48000.0;

    DSP::MultiIrLoader loader; // eight slots, all requesting an equal share
    loader.prepare({sampleRate, 512, 1});

    EXPECT(loader.getNumLoadedSlots() == 0);
    for (int slot = 0; slot < loader.getNumSlots(); ++slot)
        EXPECT_NEAR(loader.getWeightPercent(slot), 0.0f, 1e-3f);

    const auto identity = makeIdentityIr();
    EXPECT_TRUE(loader.loadImpulseResponse(2, identity, sampleRate));
    EXPECT_TRUE(loader.loadImpulseResponse(5, identity, sampleRate));
    EXPECT(loader.getNumLoadedSlots() == 2);

    // Two of eight loaded at equal request: 50/50, not 12.5 each.
    EXPECT_NEAR(loader.getWeightPercent(2), 50.0f, 1e-3f);
    EXPECT_NEAR(loader.getWeightPercent(5), 50.0f, 1e-3f);

    float total = 0.0f;
    for (int slot = 0; slot < loader.getNumSlots(); ++slot) {
        if (slot != 2 && slot != 5)
            EXPECT_NEAR(loader.getWeightPercent(slot), 0.0f, 1e-3f);
        total += loader.getWeightPercent(slot);
    }
    EXPECT_NEAR(total, 100.0f, 1e-3f);
}

static void test_MultiIrLoader_NoIrPreservesDrySignal()
{
    beginTest("MultiIrLoader / NoIrPreservesDrySignal");
    DSP::MultiIrLoader loader;
    loader.prepare({48000.0, 16, 2});

    juce::AudioBuffer<float> buffer(2, 16);
    for (int channel = 0; channel < buffer.getNumChannels(); ++channel)
        for (int sample = 0; sample < buffer.getNumSamples(); ++sample)
            buffer.setSample(channel, sample, static_cast<float>(channel * 100 + sample));
    const juce::AudioBuffer<float> expected(buffer);

    juce::dsp::AudioBlock<float> block(buffer);
    juce::dsp::ProcessContextReplacing<float> context(block);
    loader.process(context);

    for (int channel = 0; channel < buffer.getNumChannels(); ++channel)
        for (int sample = 0; sample < buffer.getNumSamples(); ++sample)
            EXPECT_NEAR(buffer.getSample(channel, sample), expected.getSample(channel, sample), 0.0f);
}

static void test_MultiIrLoader_SingleSlotIsUnityGain()
{
    beginTest("MultiIrLoader / SingleSlotIsUnityGain");
    constexpr double sampleRate = 48000.0;

    DSP::MultiIrLoader loader(1);
    loader.setNormalise(false);
    loader.prepare({sampleRate, 512, 1});

    EXPECT_TRUE(loader.loadImpulseResponse(0, makeIdentityIr(), sampleRate));
    EXPECT_NEAR(loader.getWeightPercent(0), 100.0f, 1e-3f);

    const float level = renderLoaderSine(loader, 1, kMultiTestFrequency, sampleRate)[0];
    EXPECT_NEAR(level, kFullScaleSineRms, 0.01f);
}

static void test_MultiIrLoader_EqualWeightsPreserveLevel()
{
    beginTest("MultiIrLoader / EqualWeightsPreserveLevel");
    constexpr double sampleRate = 48000.0;

    DSP::MultiIrLoader loader(4);
    loader.setNormalise(false);
    loader.prepare({sampleRate, 512, 1});

    const auto identity = makeIdentityIr();
    for (int slot = 0; slot < 4; ++slot)
        EXPECT_TRUE(loader.loadImpulseResponse(slot, identity, sampleRate));

    for (int slot = 0; slot < 4; ++slot)
        EXPECT_NEAR(loader.getWeightPercent(slot), 25.0f, 1e-3f);

    // Four copies of the same IR at 25% each still sum to unity.
    const float level = renderLoaderSine(loader, 1, kMultiTestFrequency, sampleRate)[0];
    EXPECT_NEAR(level, kFullScaleSineRms, 0.01f);
}

static void test_MultiIrLoader_InvertedSlotCancels()
{
    beginTest("MultiIrLoader / InvertedSlotCancels");
    constexpr double sampleRate = 48000.0;

    DSP::MultiIrLoader loader(2);
    loader.setNormalise(false);
    loader.prepare({sampleRate, 512, 1});

    EXPECT_TRUE(loader.loadImpulseResponse(0, makeIdentityIr(), sampleRate));
    EXPECT_TRUE(loader.loadImpulseResponse(1, makeInvertedIdentityIr(), sampleRate));
    EXPECT_NEAR(loader.getWeightPercent(0), 50.0f, 1e-3f);
    EXPECT_NEAR(loader.getWeightPercent(1), 50.0f, 1e-3f);

    // Equal weights on opposite polarities must sum to silence.
    const float level = renderLoaderSine(loader, 1, kMultiTestFrequency, sampleRate)[0];
    EXPECT_NEAR(level, 0.0f, 1e-5f);
}

static void test_MultiIrLoader_ZeroWeightSlotIsExcluded()
{
    beginTest("MultiIrLoader / ZeroWeightSlotIsExcluded");
    constexpr double sampleRate = 48000.0;

    DSP::MultiIrLoader loader(2);
    loader.setNormalise(false);
    const float requested[2] = {1.0f, 0.0f};
    loader.setWeights(requested);
    loader.prepare({sampleRate, 512, 1});

    EXPECT_TRUE(loader.loadImpulseResponse(0, makeIdentityIr(), sampleRate));
    EXPECT_TRUE(loader.loadImpulseResponse(1, makeInvertedIdentityIr(), sampleRate));
    EXPECT_NEAR(loader.getWeightPercent(0), 100.0f, 1e-3f);
    EXPECT_NEAR(loader.getWeightPercent(1), 0.0f, 1e-3f);

    // The muted inverted slot must not cancel the audible one.
    const float level = renderLoaderSine(loader, 1, kMultiTestFrequency, sampleRate)[0];
    EXPECT_NEAR(level, kFullScaleSineRms, 0.01f);
}

static void test_MultiIrLoader_AllZeroWeightsSilenceOutput()
{
    beginTest("MultiIrLoader / AllZeroWeightsSilenceOutput");
    constexpr double sampleRate = 48000.0;

    DSP::MultiIrLoader loader(2);
    loader.setNormalise(false);
    loader.prepare({sampleRate, 512, 1});

    const auto identity = makeIdentityIr();
    EXPECT_TRUE(loader.loadImpulseResponse(0, identity, sampleRate));
    EXPECT_TRUE(loader.loadImpulseResponse(1, identity, sampleRate));

    const float requested[2] = {0.0f, 0.0f};
    loader.setWeights(requested);
    EXPECT_NEAR(loader.getWeightPercent(0), 0.0f, 1e-3f);
    EXPECT_NEAR(loader.getWeightPercent(1), 0.0f, 1e-3f);

    // Loaded but fully muted is silence, not dry passthrough.
    const float level = renderLoaderSine(loader, 1, kMultiTestFrequency, sampleRate)[0];
    EXPECT_NEAR(level, 0.0f, 1e-6f);
}

static float renderMultiFilteredSlot(int activeSlot)
{
    constexpr double sampleRate = 48000.0;

    DSP::MultiIrLoader loader(3);
    loader.setNormalise(false);
    loader.setHighPassFrequency(0, 400.0f);
    loader.setLowPassFrequency(0, 22000.0f);
    loader.setHighPassFrequency(2, 10.0f);
    loader.setLowPassFrequency(2, 22000.0f);

    const float requested[3] = {activeSlot == 0 ? 1.0f : 0.0f, 0.0f, activeSlot == 2 ? 1.0f : 0.0f};
    loader.setWeights(requested);
    loader.prepare({sampleRate, 512, 1});

    const auto identity = makeIdentityIr();
    EXPECT_TRUE(loader.loadImpulseResponse(0, identity, sampleRate));
    EXPECT_TRUE(loader.loadImpulseResponse(2, identity, sampleRate));

    return renderLoaderSine(loader, 1, 40.0f, sampleRate)[0];
}

static void test_MultiIrLoader_FiltersSlotsIndependently()
{
    beginTest("MultiIrLoader / FiltersSlotsIndependently");
    EXPECT_NEAR(DSP::MultiIrLoader{3}.getHighPassFrequency(1), 10.0f, 1e-3f);

    const float filtered = renderMultiFilteredSlot(0); // 40 Hz into a 400 Hz HPF
    const float open = renderMultiFilteredSlot(2);     // 40 Hz into a 10 Hz HPF
    EXPECT_GT(open, filtered * 8.0f);
}

static void test_MultiIrLoader_OutOfRangeSlotsAreNoOps()
{
    beginTest("MultiIrLoader / OutOfRangeSlotsAreNoOps");
    constexpr double sampleRate = 48000.0;

    DSP::MultiIrLoader loader(2);
    loader.setNormalise(false);
    loader.prepare({sampleRate, 512, 1});

    const auto identity = makeIdentityIr();
    EXPECT_TRUE(loader.loadImpulseResponse(0, identity, sampleRate));

    constexpr int below = -1;
    constexpr int above = 2; // == getNumSlots()

    // Loading.
    EXPECT_FALSE(loader.loadImpulseResponse(below, identity, sampleRate));
    EXPECT_FALSE(loader.loadImpulseResponse(above, identity, sampleRate));
    EXPECT_FALSE(loader.loadImpulseResponse(below, juce::File{}));
    EXPECT_FALSE(loader.loadImpulseResponse(above, juce::File{}));

    // Queries.
    EXPECT_FALSE(loader.isSlotLoaded(below));
    EXPECT_FALSE(loader.isSlotLoaded(above));
    EXPECT_NEAR(loader.getWeightPercent(below), 0.0f, 0.0f);
    EXPECT_NEAR(loader.getWeightPercent(above), 0.0f, 0.0f);
    EXPECT_NEAR(loader.getIrSourceRate(below), 0.0, 0.0);
    EXPECT_NEAR(loader.getIrSourceRate(above), 0.0, 0.0);
    EXPECT(loader.getRawIrBuffer(below).getNumSamples() == 0);
    EXPECT(loader.getRawIrBuffer(above).getNumSamples() == 0);
    EXPECT_TRUE(loader.getImpulseResponseFile(below) == juce::File{});
    EXPECT_TRUE(loader.getImpulseResponseFile(above) == juce::File{});
    EXPECT_NEAR(loader.getHighPassFrequency(below), 0.0f, 0.0f);
    EXPECT_NEAR(loader.getLowPassFrequency(above), 0.0f, 0.0f);

    // Setters must neither crash nor disturb the valid slots.
    loader.setHighPassFrequency(below, 5000.0f);
    loader.setHighPassFrequency(above, 5000.0f);
    loader.setLowPassFrequency(below, 100.0f);
    loader.setLowPassFrequency(above, 100.0f);
    loader.setGain(below, -60.0f);
    loader.setGain(above, -60.0f);
    loader.setWeightPercent(below, 90.0f);
    loader.setWeightPercent(above, 90.0f);
    loader.applyAlignment(below, 1.0f, true);
    loader.applyAlignment(above, 1.0f, true);
    loader.clearImpulseResponse(below);
    loader.clearImpulseResponse(above);

    EXPECT_TRUE(loader.isSlotLoaded(0));
    EXPECT(loader.getNumLoadedSlots() == 1);
    EXPECT_NEAR(loader.getWeightPercent(0), 100.0f, 1e-3f);
    EXPECT_NEAR(loader.getHighPassFrequency(0), 10.0f, 1e-3f);
    EXPECT_NEAR(loader.getLowPassFrequency(0), 22000.0f, 1e-3f);

    const float level = renderLoaderSine(loader, 1, kMultiTestFrequency, sampleRate)[0];
    EXPECT_NEAR(level, kFullScaleSineRms, 0.01f);
}

static void test_MultiIrLoader_AlignmentDoesNotChangeLevel()
{
    beginTest("MultiIrLoader / AlignmentDoesNotChangeLevel");
    constexpr double sampleRate = 48000.0;

    DSP::MultiIrLoader loader(4); // slot 2: a non-zero index catches indexing bugs
    loader.prepare({sampleRate, 512, 1});

    juce::AudioBuffer<float> ir(1, 1024);
    ir.clear();
    ir.setSample(0, 200, 1.0f);
    EXPECT_TRUE(loader.loadImpulseResponse(2, ir, sampleRate));

    const auto &rawBefore = loader.getRawIrBuffer(2);
    const float peakBefore = rawBefore.getMagnitude(0, 0, rawBefore.getNumSamples());
    const float levelBefore = renderLoaderSine(loader, 1, kMultiTestFrequency, sampleRate)[0];
    EXPECT_GT(levelBefore, 0.0f);

    // Negative delay: trim IR onset — most likely to expose a double-normalise bug.
    loader.applyAlignment(2, -1.0f /*ms*/, false);

    const auto &rawAfter = loader.getRawIrBuffer(2);
    const float peakAfter = rawAfter.getMagnitude(0, 0, rawAfter.getNumSamples());
    EXPECT_NEAR(peakBefore, peakAfter, 1e-3f);

    const float levelAfter = renderLoaderSine(loader, 1, kMultiTestFrequency, sampleRate)[0];
    EXPECT_NEAR(levelBefore, levelAfter, 1e-3f);

    // Repeated calls re-derive from the stored raw IR instead of compounding.
    loader.applyAlignment(2, -1.0f /*ms*/, false);
    const float levelAgain = renderLoaderSine(loader, 1, kMultiTestFrequency, sampleRate)[0];
    EXPECT_NEAR(levelBefore, levelAgain, 1e-3f);
}

static void test_MultiIrLoader_SetWeightsReplacesPreviousMix()
{
    beginTest("MultiIrLoader / SetWeightsReplacesPreviousMix");
    constexpr double sampleRate = 48000.0;

    DSP::MultiIrLoader loader(4);
    loader.prepare({sampleRate, 512, 1});

    const auto identity = makeIdentityIr();
    for (int slot = 0; slot < 3; ++slot)
        EXPECT_TRUE(loader.loadImpulseResponse(slot, identity, sampleRate));

    loader.setWeightPercent(2, 50.0f);
    EXPECT_GT(loader.getWeightPercent(2), 90.0f);

    // A span shorter than the slot count zeroes everything it does not cover.
    const std::vector<float> requested{1.0f, 1.0f};
    loader.setWeights(requested);

    EXPECT_NEAR(loader.getWeightPercent(0), 50.0f, 1e-3f);
    EXPECT_NEAR(loader.getWeightPercent(1), 50.0f, 1e-3f);
    EXPECT_NEAR(loader.getWeightPercent(2), 0.0f, 1e-3f);
    EXPECT_NEAR(loader.getWeightPercent(3), 0.0f, 1e-3f);

    // A span longer than the slot count is truncated, not an overrun.
    const std::vector<float> tooMany{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    loader.setWeights(tooMany);
    EXPECT_NEAR(loader.getWeightPercent(0), 100.0f / 3.0f, 1e-3f);
    EXPECT_NEAR(loader.getWeightPercent(3), 0.0f, 1e-3f); // loaded gates the share
}

static void test_MultiIrLoader_TailLengthIsLongestLoadedIr()
{
    beginTest("MultiIrLoader / TailLengthIsLongestLoadedIr");
    constexpr double sampleRate = 48000.0;

    DSP::MultiIrLoader loader(3);
    loader.prepare({sampleRate, 512, 1});
    EXPECT_NEAR(loader.getTailLengthSeconds(), 0.0, 0.0);

    juce::AudioBuffer<float> shortIr(1, 1024);
    shortIr.clear();
    shortIr.setSample(0, 0, 1.0f);
    EXPECT_TRUE(loader.loadImpulseResponse(0, shortIr, sampleRate));
    EXPECT_NEAR(loader.getTailLengthSeconds(), 1024.0 / sampleRate, 1e-9);

    juce::AudioBuffer<float> longIr(1, 4800);
    longIr.clear();
    longIr.setSample(0, 0, 1.0f);
    EXPECT_TRUE(loader.loadImpulseResponse(2, longIr, sampleRate));
    EXPECT_NEAR(loader.getTailLengthSeconds(), 0.1, 1e-9);

    // Dropping the longest IR shortens the reported tail again.
    loader.clearImpulseResponse(2);
    EXPECT_NEAR(loader.getTailLengthSeconds(), 1024.0 / sampleRate, 1e-9);
}

static void test_MultiIrLoader_ClearRenormalisesRemainingSlots()
{
    beginTest("MultiIrLoader / ClearRenormalisesRemainingSlots");
    constexpr double sampleRate = 48000.0;

    DSP::MultiIrLoader loader(4);
    loader.setNormalise(false);
    loader.prepare({sampleRate, 512, 1});

    const auto identity = makeIdentityIr();
    EXPECT_TRUE(loader.loadImpulseResponse(0, identity, sampleRate));
    EXPECT_TRUE(loader.loadImpulseResponse(1, identity, sampleRate));
    EXPECT_NEAR(loader.getWeightPercent(0), 50.0f, 1e-3f);
    EXPECT_NEAR(loader.getWeightPercent(1), 50.0f, 1e-3f);

    loader.clearImpulseResponse(0);

    EXPECT_FALSE(loader.isSlotLoaded(0));
    EXPECT(loader.getNumLoadedSlots() == 1);
    EXPECT(loader.getRawIrBuffer(0).getNumSamples() == 0);
    EXPECT_NEAR(loader.getIrSourceRate(0), 0.0, 0.0);
    EXPECT_NEAR(loader.getWeightPercent(0), 0.0f, 1e-3f);
    EXPECT_NEAR(loader.getWeightPercent(1), 100.0f, 1e-3f);

    // The survivor now carries the whole mix, at unity.
    const float level = renderLoaderSine(loader, 1, kMultiTestFrequency, sampleRate)[0];
    EXPECT_NEAR(level, kFullScaleSineRms, 0.01f);
}

static void test_MultiIrLoader_ClearAllRestoresDryPassthrough()
{
    beginTest("MultiIrLoader / ClearAllRestoresDryPassthrough");
    constexpr double sampleRate = 48000.0;

    DSP::MultiIrLoader loader(3);
    loader.setNormalise(false);
    loader.prepare({sampleRate, 16, 1});

    const auto identity = makeIdentityIr();
    EXPECT_TRUE(loader.loadImpulseResponse(0, identity, sampleRate));
    EXPECT_TRUE(loader.loadImpulseResponse(1, identity, sampleRate));

    loader.clearAllImpulseResponses();
    EXPECT(loader.getNumLoadedSlots() == 0);
    for (int slot = 0; slot < loader.getNumSlots(); ++slot)
        EXPECT_NEAR(loader.getWeightPercent(slot), 0.0f, 1e-3f);

    juce::AudioBuffer<float> buffer(1, 16);
    for (int sample = 0; sample < buffer.getNumSamples(); ++sample)
        buffer.setSample(0, sample, static_cast<float>(sample + 1));
    const juce::AudioBuffer<float> expected(buffer);

    juce::dsp::AudioBlock<float> block(buffer);
    juce::dsp::ProcessContextReplacing<float> context(block);
    loader.process(context);

    for (int sample = 0; sample < buffer.getNumSamples(); ++sample)
        EXPECT_NEAR(buffer.getSample(0, sample), expected.getSample(0, sample), 0.0f);
}

static void test_MultiIrLoader_LoadBeforePrepareIsReportedOnlyAfterPrepare()
{
    beginTest("MultiIrLoader / LoadBeforePrepareIsReportedOnlyAfterPrepare");
    constexpr double sampleRate = 48000.0;

    DSP::MultiIrLoader loader(2);
    loader.setNormalise(false);

    // IrLoader only builds convolvers once prepared, so the slot is stored but
    // not yet reported as loaded.
    EXPECT_TRUE(loader.loadImpulseResponse(0, makeIdentityIr(), sampleRate));
    EXPECT_FALSE(loader.isSlotLoaded(0));
    EXPECT(loader.getNumLoadedSlots() == 0);
    EXPECT_NEAR(loader.getTailLengthSeconds(), 128.0 / sampleRate, 1e-9);

    loader.prepare({sampleRate, 512, 1});
    EXPECT_TRUE(loader.isSlotLoaded(0));
    EXPECT(loader.getNumLoadedSlots() == 1);

    // prepare() is what makes the slot visible to isLoaded(), so it must also
    // republish the weights. Without that, the all-zero set published while the
    // slot still looked empty would survive and hard-mute the output forever.
    EXPECT_NEAR(loader.getWeightPercent(0), 100.0f, 1e-3f);
    EXPECT_NEAR(
        renderLoaderSine(loader, 1, kMultiTestFrequency, sampleRate)[0], kFullScaleSineRms, 0.01f);
}

// ---------------------------------------------------------------------------

int main()
{
    std::printf("=== irconv tests ===\n");

    test_TimeAligner_ZeroDelayAlignment();
    test_TimeAligner_KnownDelayAlignment();
    test_TimeAligner_PolarityInversionDetected();
    test_IrLoader_LoadFromBuffer();
    test_IrLoader_ClearUnloadsConvolver();
    test_DualIrLoader_AlignmentDoesNotChangeLevel();
    test_DualIrLoader_NoIrPreservesDrySignal();
    test_DualIrLoader_SingleLoadedSlotUsesItsOwnFilter();
    test_DualIrLoader_BlendFiltersSlotsIndependently();
    test_DualIrLoader_StereoSplitFiltersSlotsIndependently();
    test_IrFilter_PassesMidBandAndRejectsExtremes();
    test_IrFilter_ClampsCutoffBelowNyquist();
    test_MultiIrLoader_ConstructorClampsSlotCount();
    test_MultiIrLoader_WeightsNormaliseToOneHundred();
    test_MultiIrLoader_NegativeAndNonFiniteWeightsAreZero();
    test_MultiIrLoader_NormalisationCoversLoadedSlotsOnly();
    test_MultiIrLoader_NoIrPreservesDrySignal();
    test_MultiIrLoader_SingleSlotIsUnityGain();
    test_MultiIrLoader_EqualWeightsPreserveLevel();
    test_MultiIrLoader_InvertedSlotCancels();
    test_MultiIrLoader_ZeroWeightSlotIsExcluded();
    test_MultiIrLoader_AllZeroWeightsSilenceOutput();
    test_MultiIrLoader_FiltersSlotsIndependently();
    test_MultiIrLoader_OutOfRangeSlotsAreNoOps();
    test_MultiIrLoader_AlignmentDoesNotChangeLevel();
    test_MultiIrLoader_SetWeightsReplacesPreviousMix();
    test_MultiIrLoader_TailLengthIsLongestLoadedIr();
    test_MultiIrLoader_ClearRenormalisesRemainingSlots();
    test_MultiIrLoader_ClearAllRestoresDryPassthrough();
    test_MultiIrLoader_LoadBeforePrepareIsReportedOnlyAfterPrepare();

    if (g_failures == 0)
        std::printf("All tests passed.\n");
    else
        std::fprintf(stderr, "%d test(s) FAILED.\n", g_failures);

    return g_failures == 0 ? 0 : 1;
}
