// SPDX-FileCopyrightText: 2026 Pier Luigi Fiorini
// SPDX-License-Identifier: MIT

#include "TimeAligner.h"

#include <juce_dsp/juce_dsp.h>

namespace DSP {

namespace {

// Mono-mix an AudioBuffer into a std::vector<float> of length `len`,
// using only the first channel (sufficient for cab IR onset detection).
std::vector<float> monoChannel(const juce::AudioBuffer<float> &buf, int len)
{
    std::vector<float> out(static_cast<size_t>(len), 0.0f);
    const int nCh = buf.getNumChannels();
    const int nSamp = std::min(buf.getNumSamples(), len);
    if (nCh == 0 || nSamp == 0)
        return out;
    const float *src = buf.getReadPointer(0);
    for (int i = 0; i < nSamp; ++i)
        out[static_cast<size_t>(i)] = src[i];
    return out;
}

// Normalize a vector to unit RMS. Returns false if signal is silent.
bool normaliseRms(std::vector<float> &v)
{
    double sumSq = 0.0;
    for (float x : v)
        sumSq += static_cast<double>(x) * x;
    if (sumSq < 1e-20)
        return false;
    const float scale = static_cast<float>(1.0 / std::sqrt(sumSq));
    for (float &x : v)
        x *= scale;
    return true;
}

// Smallest power of two >= value.
int nextPow2(int v) noexcept
{
    int p = 1;
    while (p < v)
        p <<= 1;
    return p;
}

} // namespace

AlignmentResult TimeAligner::analyse(
    const juce::AudioBuffer<float> &irA,
    const juce::AudioBuffer<float> &irB,
    double sampleRate,
    float searchWindowMs)
{
    const AlignmentResult noChange{0.0f, false, 0.0f};

    const int lagMax = juce::roundToInt(searchWindowMs * 0.001 * sampleRate);
    // Limit analysis window to avoid heavy FFTs on long IRs.
    const int analysisLen = std::min({irA.getNumSamples(), irB.getNumSamples(), 4096});
    if (analysisLen < 4 || lagMax < 1)
        return noChange;

    auto a = monoChannel(irA, analysisLen);
    auto b = monoChannel(irB, analysisLen);

    if (!normaliseRms(a) || !normaliseRms(b))
        return noChange;

    // FFT size: zero-pad for linear (not circular) correlation.
    const int fftSize = nextPow2(2 * analysisLen);
    juce::dsp::FFT fft(static_cast<int>(std::log2(fftSize)));

    // JUCE performRealOnlyForwardTransform works in-place on a 2*fftSize buffer
    // (interleaved complex: re0, im0, re1, im1, ...).
    std::vector<float> fa(static_cast<size_t>(fftSize * 2), 0.0f);
    std::vector<float> fb(static_cast<size_t>(fftSize * 2), 0.0f);

    std::copy(a.begin(), a.end(), fa.begin());
    std::copy(b.begin(), b.end(), fb.begin());

    fft.performRealOnlyForwardTransform(fa.data(), true);
    fft.performRealOnlyForwardTransform(fb.data(), true);

    // Cross-spectrum: R[k] = conj(Fa[k]) * Fb[k] (stored interleaved re/im).
    const int numBins = fftSize / 2 + 1;
    std::vector<float> r(static_cast<size_t>(fftSize * 2), 0.0f);
    for (int k = 0; k < numBins; ++k) {
        const float re_a = fa[static_cast<size_t>(2 * k)];
        const float im_a = fa[static_cast<size_t>(2 * k + 1)];
        const float re_b = fb[static_cast<size_t>(2 * k)];
        const float im_b = fb[static_cast<size_t>(2 * k + 1)];
        // conj(Fa) * Fb
        r[static_cast<size_t>(2 * k)] = re_a * re_b + im_a * im_b;
        r[static_cast<size_t>(2 * k + 1)] = re_a * im_b - im_a * re_b;
    }

    fft.performRealOnlyInverseTransform(r.data());

    // r[0..fftSize-1] now holds the normalized cross-correlation.
    // Search within [0, lagMax] (B lags A) and [fftSize-lagMax, fftSize-1] (B leads A).
    int peakIdx = 0;
    float peakAbs = -1.0f;

    auto check = [&](int idx) {
        const float v = r[static_cast<size_t>(idx)];
        if (std::abs(v) > peakAbs) {
            peakAbs = std::abs(v);
            peakIdx = idx;
        }
    };

    for (int k = 0; k <= lagMax; ++k)
        check(k);
    for (int k = fftSize - lagMax; k < fftSize; ++k)
        check(k);

    if (peakAbs < 1e-6f)
        return noChange;

    // Map circular index → signed delay.
    float delayInSamples;
    if (peakIdx <= lagMax)
        delayInSamples = static_cast<float>(peakIdx);
    else
        delayInSamples = static_cast<float>(peakIdx - fftSize);

    // Parabolic sub-sample refinement.
    {
        const int k = peakIdx;
        const int km1 = (k - 1 + fftSize) % fftSize;
        const int kp1 = (k + 1) % fftSize;
        const float ym1 = r[static_cast<size_t>(km1)];
        const float y0 = r[static_cast<size_t>(k)];
        const float yp1 = r[static_cast<size_t>(kp1)];
        const float denom = ym1 - 2.0f * y0 + yp1;
        if (std::abs(denom) > 1e-6f)
            delayInSamples += 0.5f * (ym1 - yp1) / denom;
    }

    const float peakVal = r[static_cast<size_t>(peakIdx)];
    const bool invertPolarity = (peakVal < 0.0f);
    const float score = std::min(peakAbs, 1.0f);

    return {delayInSamples, invertPolarity, score};
}

} // namespace DSP
