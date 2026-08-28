# DSP::MultiIrLoader

`source/MultiIrLoader.h` — an N-slot parallel convolver with weighted mixing.

## Overview

`MultiIrLoader` runs the input signal through a configurable number of impulse
responses in parallel and sums the wet results with per-slot weights:

```
out = Σ wᵢ · wetᵢ            with  Σ wᵢ == 1
```

The slot count is a constructor argument, fixed for the lifetime of the object:

```cpp
explicit MultiIrLoader(int numSlots = defaultNumSlots);   // clamped to [1, maxSlots]
```

| Constant | Value | Meaning |
|---|---|---|
| `MultiIrLoader::defaultNumSlots` | `8` | Used when no count is given — eight corners of a morph pad |
| `MultiIrLoader::maxSlots` | `64` | Sanity cap on the constructor argument, not a design limit |

Every per-slot object is allocated in the constructor, so the slot vector is
never resized afterwards.

### When to use it instead of `DualIrLoader`

`MultiIrLoader` does not replace [`DSP::DualIrLoader`](../source/DualIrLoader.h);
the two solve different problems.

| | `DualIrLoader` | `MultiIrLoader` |
|---|---|---|
| IRs | Exactly two (A and B) | 1 … 64, chosen at construction |
| Mixing | A single `blend` scalar, `0` = A, `1` = B | N relative weights, normalised over the loaded slots |
| Extra routing | `StereoMode::StereoSplit` — IR A hard-left, IR B hard-right | None; every slot is applied to every channel |
| Typical control | One knob or slider | A 2D pad, a matrix, an automated weight set |

Reach for `DualIrLoader` when you want a classic two-mic blend (or a hard
stereo split). Reach for `MultiIrLoader` when you need three or more IRs mixed
at once, or when the control surface produces a whole weight vector at a time —
a point dragged inside a polygon with one IR per corner being the motivating
case.

## Signal flow

Each slot owns its own `IrLoader`, `IrFilter` (HPF → LPF) and
`juce::dsp::Gain<float>`, so slots can be voiced independently *before* they are
mixed. The weight is applied last, as a per-block gain ramp.

```
                ┌─ slot 0 ─ IrLoader ─ IrFilter ─ Gain ─┐× w₀ ┐
                │                                       │     │
input block ────┼─ slot 1 ─ IrLoader ─ IrFilter ─ Gain ─┤× w₁ ┼──▶ output block
                │                                       │     │
                └─ slot n ─ IrLoader ─ IrFilter ─ Gain ─┘× wₙ ┘
                                                          (Σ)
```

Internally the summation is arranged so nothing is allocated and no extra clear
is needed:

- The **first active slot** processes the output block in place and its weight
  becomes a single `applyGainRamp()` pass.
- Every **later active slot** convolves a copy of the untouched dry input in a
  reusable scratch buffer and is accumulated with `addFromWithRamp()`.

Scratch memory is therefore two buffers (`dry` and `wet`) regardless of the slot
count — it does not scale with N.

Weights and per-slot gains ramp over **20 ms**, matching `DualIrLoader`'s blend
ramp: instant to the ear, slow enough to remove zipper noise during a drag.

## Weight semantics

This is the subtle part of the class; the rules are worth reading in full.

### Weights are relative, not absolute

Weights are supplied as non-negative values on **any scale** and are normalised
internally, so the effective percentages returned by `getWeightPercent()` always
total 100. `{1, 1, 2}`, `{25, 25, 50}` and `{0.5f, 0.5f, 1.0f}` all describe the
same mix. Negative and non-finite entries are treated as zero.

The constructor seeds every slot's requested weight to `1`, i.e. an equal share.
A freshly loaded IR is therefore audible without the caller touching the weights
first.

### Normalisation runs over loaded slots only

An empty slot does not consume a share of the mix. With eight slots and two IRs
loaded, those two IRs are mixed 50/50 at full level — not at a quarter level.

The corollary is that **load state changes the mix**: `loadImpulseResponse()`,
`clearImpulseResponse()` and `clearAllImpulseResponses()` each renormalise and
republish the weights, so the effective percentages of the *other* slots change
without any weight setter having been called. The requested (pre-normalisation)
values are never altered by loading or clearing, so clearing a slot and loading
it again restores the previous balance.

### `setWeightPercent()` sets a relative weight, not a guaranteed share

```cpp
void setWeightPercent(int slot, float percent) noexcept;
```

Despite the name, `percent` joins the same relative pool as every other weight.
`getWeightPercent(slot)` reads back `percent` only if the slot is loaded and the
requested weights of the other *loaded* slots happen to total `100 - percent`.

Worked example — two slots loaded, both at the default requested weight of `1`:

```cpp
DSP::MultiIrLoader loader{2};
// ... prepare(), then load slots 0 and 1 ...
loader.getWeightPercent(0);        // 50.0f
loader.getWeightPercent(1);        // 50.0f

loader.setWeightPercent(0, 75.0f); // requested set is now { 75, 1 }
loader.getWeightPercent(0);        // ≈ 98.7f, NOT 75.0f  (75 / 76)
loader.getWeightPercent(1);        // ≈  1.3f             ( 1 / 76)
```

To actually obtain a 75/25 split, publish the whole set:

```cpp
const std::array<float, 2> w{75.0f, 25.0f};
loader.setWeights(w);
loader.getWeightPercent(0);        // 75.0f
```

`setWeightPercent()` is convenient for "nudge one slot" interactions; prefer
`setWeights()` whenever more than one slot is changing.

### All-zero weights mean silence

If every requested weight is zero — or nothing is loaded at all — the published
weights are all zero and `getWeightPercent()` returns 0 for every slot. This is
the one case where the percentages total 0 rather than 100. With at least one IR
loaded, `process()` then silences the block; with nothing loaded, it passes the
dry signal through untouched.

### Why `setWeights()` exists

```cpp
void setWeights(std::span<const float> weights) noexcept;
```

`setWeights()` publishes the entire weight set as **one atomic update**. The
normalised weights live in a lock-free double buffer (`m_weightSets[2]` plus an
atomic index, mirroring `IrLoader`'s convolver-set publish): the writer fills the
inactive row, then flips the index with a release store, and `process()` reads
the active row with an acquire load.

A positional UI **must** use it. Calling `setWeightPercent()` N times per drag
frame publishes N times, and each intermediate publish is a fully normalised,
observable mix — `process()` can land on one of them and render a block from a
half-applied drag frame. One `setWeights()` call per drag frame gives one
publish, so the audio thread only ever sees complete mixes.

A span shorter than `getNumSlots()` leaves the remaining slots at zero; a longer
one is truncated. `setWeights()` always *replaces* the mix, it never merges into
the previous one.

## Threading contract

Grouped as the header documents them.

### Audio-thread safe

| Method | Notes |
|---|---|
| `void process(juce::dsp::ProcessContextReplacing<float>&)` | In-place; honours `context.isBypassed` |
| `void reset()` | Clears convolver, filter, gain and smoother state; allocation-free |
| `bool isSlotLoaded(int slot) const noexcept` | Atomic read of the slot's convolver-set index |
| `int getNumLoadedSlots() const noexcept` | Loops the slots, atomic reads only |

### Any thread

Each of these publishes a target without blocking or allocating.

| Method | Notes |
|---|---|
| `int getNumSlots() const noexcept` | Immutable after construction |
| `void setWeights(std::span<const float>) noexcept` | One atomic publish |
| `void setWeightPercent(int slot, float percent) noexcept` | One atomic publish |
| `float getWeightPercent(int slot) const noexcept` | Effective, normalised, 0…100 |
| `void setHighPassFrequency(int slot, float frequency) noexcept` | Per-slot HPF target |
| `void setLowPassFrequency(int slot, float frequency) noexcept` | Per-slot LPF target |
| `void setGain(int slot, float decibels) noexcept` | Per-slot output gain target |
| `float getHighPassFrequency(int slot) const noexcept` | |
| `float getLowPassFrequency(int slot) const noexcept` | |

**Single-writer requirement.** The published weight set has exactly one writer.
`setWeights()`, `setWeightPercent()` and *the loading calls below* all go through
the same publish path, so callers must serialise them among themselves — a UI
thread dragging a pad while a loader thread swaps an IR is a data race on the
requested-weight array. Keep all of them on one thread (typically the message
thread), or guard them with your own mutex.

Out-of-range slot indices are no-ops for setters and return `0` / empty objects
for getters, on every method that takes a slot.

### Message-thread only — not RT-safe

| Method | Notes |
|---|---|
| `void prepare(const juce::dsp::ProcessSpec&)` | Allocates and resizes scratch; call before any processing |
| `void setNormalise(bool) noexcept` | Applies to the next load, not retroactively |
| `void applyAlignment(int slot, float delayMs, bool invertPolarity)` | Allocates and reloads the slot |
| `std::vector<float> getWeightPercentages() const noexcept` | Allocates the returned vector |
| `const juce::File& getImpulseResponseFile(int slot) const noexcept` | |
| `const juce::AudioBuffer<float>& getRawIrBuffer(int slot) const noexcept` | |
| `double getIrSourceRate(int slot) const noexcept` | |
| `double getTailLengthSeconds() const noexcept` | Longest *stored* raw IR, in seconds at its own source rate — so it also counts a slot loaded before `prepare()` |

`reset()` follows the usual JUCE contract: it allocates nothing and is callable
from any thread once `prepare()` has run, but it must not race with `process()`.

### IR loading — non-RT thread only

| Method | Returns |
|---|---|
| `bool loadImpulseResponse(int slot, const juce::File&)` | `false` if the slot is out of range or the file could not be read |
| `bool loadImpulseResponse(int slot, const juce::AudioBuffer<float>&, double sourceSampleRate)` | `false` if the slot is out of range or the buffer is empty |
| `void clearImpulseResponse(int slot) noexcept` | — |
| `void clearAllImpulseResponses() noexcept` | — |

These follow `DSP::IrLoader`'s contract exactly: they read files, allocate,
resample and build convolvers, so they belong on the message thread or a
background loader thread — never on the audio thread. While a slot is being
loaded the audio thread keeps running; the new convolver set becomes visible
atomically when it is ready.

## Performance and cost model

### The zero-weight skip

A slot whose weight is exactly zero at **both** the start and the end of the
block is skipped entirely: no convolution, no filtering, no gain. This is what
keeps the realistic case affordable — an eight-slot pad with two or three
audible corners costs two or three convolutions per block, not eight.

Because a skipped slot's convolver does not advance, a slot re-entering the mix
from exactly zero starts with an **empty tail**. This is inaudible in practice:
the slot's weight smoother ramps it up from silence over 20 ms, so the missing
tail energy would have been faded out anyway.

Skipping is safe against `IrLoader`'s two-stage background handshake. The
handshake uses auto-reset `juce::WaitableEvent`s: a skipped block simply leaves
the "done" event signalled for the next `process()` call to consume, and the
start/wait pairing is preserved. Do not "fix" this by force-processing muted
slots.

The per-slot weight smoothers are advanced unconditionally, including for
skipped slots, so smoother state always tracks real time.

### Tail threads

`IrLoader` selects its convolver from the IR length, measured after any
resampling to the processing rate:

| IR length | Convolver | Threads |
|---|---|---|
| ≤ 16384 samples | `fftconvolver::FFTConvolver` | None — runs entirely on the audio thread |
| > 16384 samples | `fftconvolver::TwoStageFFTConvolver` | One high-priority background tail thread **per audio channel** |

The thread count multiplies by the slot count. Eight slots of long IRs in stereo
is up to **sixteen** tail threads, and `process()` performs one signal/wait
round-trip per active slot (one per channel within each slot), serially. A full
set of long stereo IRs is genuinely expensive; size the slot count to what the
product needs rather than defaulting to eight out of habit.

Two further notes inherited from `IrLoader`:

- Convolver sets are double-buffered, and the previous set is not torn down
  until the next load into that slot. A slot that has been loaded more than once
  can therefore hold two sets' worth of tail threads.
- Reloading a slot (including via `applyAlignment()`) rebuilds its convolvers and
  their threads.

## Usage example

```cpp
#include <irconv/irconv.h>

class Processor
{
public:
    void prepareToPlay(double sampleRate, int blockSize)
    {
        const juce::dsp::ProcessSpec spec{
            sampleRate, static_cast<juce::uint32>(blockSize), 2};

        // prepare() first: IrLoader only builds convolvers once prepared, so a
        // slot loaded before prepare() reports isSlotLoaded() == false.
        m_loader.prepare(spec);

        m_loader.loadImpulseResponse(0, juce::File{"/irs/close.wav"});
        m_loader.loadImpulseResponse(1, juce::File{"/irs/room.wav"});
        m_loader.loadImpulseResponse(2, juce::File{"/irs/far.wav"});

        // Optional per-slot voicing, applied before the mix.
        m_loader.setHighPassFrequency(2, 80.0f);
        m_loader.setGain(2, -3.0f);

        // Relative weights: slot 0 half the mix, slots 1 and 2 a quarter each.
        const std::array<float, 4> weights{2.0f, 1.0f, 1.0f, 0.0f};
        m_loader.setWeights(weights);

        jassert(m_loader.getNumLoadedSlots() == 3);
    }

    void processBlock(juce::AudioBuffer<float> &buffer)
    {
        juce::dsp::AudioBlock<float> block{buffer};
        juce::dsp::ProcessContextReplacing<float> context{block};
        m_loader.process(context);
    }

    void releaseResources() { m_loader.reset(); }

private:
    DSP::MultiIrLoader m_loader{4}; // slot count fixed at construction
};
```

## Driving it from a 2D positional control

The motivating use case: a point dragged inside a polygon, one IR per corner.
The point's position is mapped to one weight per vertex and published with a
single `setWeights()` call per drag frame.

> **Illustration only.** No UI code and no geometry code ships in this
> repository. The snippet below is not part of the module — it exists to show
> the shape of the integration, and you are expected to write your own mapping
> to suit your control surface.

```cpp
// NOT part of irconv — illustrative host-side code.
struct Point { float x, y; };

// Inverse-distance weighting: the closer the point is to a vertex, the louder
// that vertex's IR. At the centre of a regular polygon every distance is equal,
// so the result is an equal mix of all corners.
void updateWeightsFromPad(DSP::MultiIrLoader &loader,
                          std::span<const Point> vertices,
                          Point position)
{
    constexpr float epsilon = 1.0e-4f; // avoids a division by zero on a vertex

    std::vector<float> weights(vertices.size(), 0.0f);

    for (size_t i = 0; i < vertices.size(); ++i) {
        const float dx = position.x - vertices[i].x;
        const float dy = position.y - vertices[i].y;
        const float distance = std::sqrt(dx * dx + dy * dy);

        weights[i] = 1.0f / std::max(distance, epsilon);
    }

    // No need to normalise here — MultiIrLoader does it, over the loaded slots.
    // One publish per drag frame, so process() never sees a partial mix.
    loader.setWeights(weights);
}
```

Points to carry over into a real implementation:

- Build the whole weight vector, then publish it once. Never loop over
  `setWeightPercent()`.
- Leave normalisation to the class. Feeding it raw `1/d` values is fine, and it
  keeps empty slots from stealing a share.
- Do not special-case empty corners in the mapping; an unloaded slot is already
  excluded from the normalisation.
- Drive the pad from one thread only (see the single-writer requirement above).

## Caveats

- **`isSlotLoaded()` only becomes true after `prepare()`.** `IrLoader` defers
  building its convolvers until it has been prepared, so a slot loaded before
  `prepare()` reports `isSlotLoaded() == false` until that call. Either ordering
  works: `prepare()` republishes the weights once the convolvers exist, so a slot
  loaded beforehand takes its correct share and is audible immediately. Note that
  the weights *are* zero for such a slot in the window between loading and
  `prepare()`, which matters only if you read `getWeightPercent()` there.
- **`getRawIrBuffer(slot)` returns the post-normalisation buffer.** It is the IR
  at its original source sample rate, but with `IrLoader`'s peak-and-energy
  normalisation already applied when `setNormalise(true)` was in effect at load
  time. It is not a byte-for-byte copy of the file.
- **`applyAlignment()` does not compound.** It re-derives a shifted copy from the
  slot's stored raw IR each time, so calling it repeatedly starts from the
  original rather than stacking shifts. It also disables normalisation for the
  internal reload (the stored IR is already normalised, so re-normalising would
  scale it twice) and then restores the configured setting. It is a no-op if the
  slot is out of range or holds no stored IR, and it neither changes the slot's
  reported source rate nor republishes the weights.
- **Block-size contract.** Scratch buffers are sized to `spec.maximumBlockSize`
  and `spec.numChannels` in `prepare()`. Handing `process()` a longer block trips
  an assertion in debug builds; call `prepare()` again whenever the host changes
  the block size or sample rate. Extra channels beyond `spec.numChannels` are
  left untouched rather than indexed out of bounds.
- **One publish per block is what the double buffer guarantees.** The audio
  thread snapshots the active weight row once per `process()` call. That row is
  stable as long as the writer publishes at most once while a block is in
  flight — the normal case for a UI publishing one weight set per drag frame. A
  burst of publishes inside a single audio callback can, in principle, land on the
  row being read; one `setWeights()` per frame avoids the question entirely.
- **Loading changes the mix.** Any load or clear renormalises the published
  weights. If your UI mirrors the effective percentages, refresh them from
  `getWeightPercentages()` after every load or clear, not only after a weight
  change.
- **Filters and gains are always in the chain.** They default to
  `IrFilter::defaultHighPassFrequency` (10 Hz), `IrFilter::defaultLowPassFrequency`
  (22 kHz) and unity gain, so an untouched slot is effectively flat, but the
  filter cost is paid on every active slot.
