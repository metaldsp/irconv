// SPDX-FileCopyrightText: 2026 Pier Luigi Fiorini
// SPDX-License-Identifier: MIT

#include <irconv/irconv.h>

#include <cstdio>

#define EXPECT(cond) \
    do { \
        if (!(cond)) { \
            std::fprintf(stderr, "FAIL: %s\n", #cond); \
            return 1; \
        } \
    } while (false)

int main()
{
    // AudioBuffer from juce_audio_basics (transitive via irconv)
    juce::AudioBuffer<float> buffer(2, 512);
    buffer.clear();

    EXPECT(buffer.getNumChannels() == 2);
    EXPECT(buffer.getNumSamples() == 512);

    for (int ch = 0; ch < buffer.getNumChannels(); ++ch)
        for (int s = 0; s < buffer.getNumSamples(); ++s)
            EXPECT(buffer.getSample(ch, s) == 0.0f);

    // ProcessSpec from juce_dsp (direct dependency of irconv)
    juce::dsp::ProcessSpec spec{};
    spec.sampleRate = 44100.0;
    spec.maximumBlockSize = 512;
    spec.numChannels = 2;

    EXPECT(spec.sampleRate == 44100.0);
    EXPECT(spec.maximumBlockSize == 512u);
    EXPECT(spec.numChannels == 2u);

    std::printf("All tests passed.\n");
    return 0;
}
