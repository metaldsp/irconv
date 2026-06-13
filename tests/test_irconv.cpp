// SPDX-FileCopyrightText: 2026 Pier Luigi Fiorini
// SPDX-License-Identifier: MIT

#include <irconv/irconv.h>

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

    if (g_failures == 0)
        std::printf("All tests passed.\n");
    else
        std::fprintf(stderr, "%d test(s) FAILED.\n", g_failures);

    return g_failures == 0 ? 0 : 1;
}
