// SPDX-FileCopyrightText: 2026 Pier Luigi Fiorini
// SPDX-License-Identifier: MIT

#include <irconv/irconv.h>

#include <array>
#include <cmath>
#include <cstdio>

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

static std::array<float, 2> renderDualSine(
    DSP::DualIrLoader &loader,
    int numChannels,
    float frequency,
    double sampleRate)
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
    return renderDualSine(loader, 1, 40.0f, sampleRate)[0];
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
    return renderDualSine(loader, 1, 8000.0f, sampleRate)[0];
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
    const auto levels = renderDualSine(loader, 2, 40.0f, sampleRate);
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

    if (g_failures == 0)
        std::printf("All tests passed.\n");
    else
        std::fprintf(stderr, "%d test(s) FAILED.\n", g_failures);

    return g_failures == 0 ? 0 : 1;
}
