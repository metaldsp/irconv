# JUCE DSP Best Practices

## Thread safety contracts

Every API must clearly belong to one of three thread domains — and that domain must be documented:

- **Audio thread** (`process()`, `reset()`): real-time safe. No allocations, no locks, no blocking I/O, no `new`/`delete`, no JUCE `String` construction, no `DBG()`.
- **Message thread** (IR loading, UI callbacks): may allocate and block. Owns all mutable non-atomic state that the audio thread never touches.
- **Any thread**: only allowed when access is guarded by `std::atomic` with explicit memory orders.

Use `// Audio-thread safe.` or `// Message-thread only — not RT-safe.` comments on methods that are not obvious from name alone.

## Real-time safety in `process()`

- Never allocate in `process()`. Size scratch buffers in `prepare()` and reuse them (`juce::HeapBlock`, `juce::AudioBuffer` resized once in `prepare()`).
- Never take a lock. Use lock-free double-buffering (e.g. the `m_sets[2]` / `m_activeSet` pattern in `IrLoader`) for IR swaps.
- Snapshot atomics once per callback — do not re-read mid-block. Use `std::memory_order_acquire` on the snapshot, `std::memory_order_release` on the publish.
- Prefer `std::atomic<int>` index over `std::atomic<std::shared_ptr<>>` to avoid hidden allocations.

## DSP module contract

All DSP classes must implement the three-method JUCE DSP contract in declaration order:

```cpp
void prepare(const juce::dsp::ProcessSpec& spec);
void reset();
void process(juce::dsp::ProcessContextReplacing<float>& context);
```

- `prepare()`: allocate, resize, and initialize all state. Cache `spec` for later use.
- `reset()`: clear delay lines and smoothers; do not deallocate. Must be callable from any thread after `prepare()`.
- `process()`: check `context.isBypassed` first; return immediately if true. Use `context.getOutputBlock()` for in-place processing.

## `juce::dsp::ProcessSpec` usage

- Store the spec in `prepare()` (`m_spec = spec;`).
- Use `spec.numChannels`, `spec.maximumBlockSize`, and `spec.sampleRate` — never hard-code channel counts or block sizes.
- Pass `spec` by const reference in `prepare()`.

## Memory and ownership

- Use `JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ClassName)` in the `private:` section of every non-trivially-copyable DSP class.
- Use `std::unique_ptr` for owned heap objects. Avoid raw `new`/`delete`.
- Use `juce::HeapBlock<T>` for plain arrays sized at runtime (faster than `std::vector` for reuse in `prepare()`).
- Use `juce::AudioBuffer<float>` for multi-channel audio data.

## JUCE threading primitives

- Use `juce::Thread` for background audio-tail threads. Always call `startThread()` in the constructor and `stopThread(timeoutMs)` in the destructor.
- Use `juce::WaitableEvent` (auto-reset, initially unsignalled) for producer/consumer signalling between audio and background threads.
- In `juce::Thread::run()`, always check `threadShouldExit()` at the top of the loop and after every `wait()`.
- Signal the start event before calling `stopThread()` to unblock a waiting `run()`.

## Smoothed values

- Use `juce::LinearSmoothedValue<float>` for parameters that are written from non-RT threads and read on the audio thread.
- Pair with a `std::atomic<float>` target: non-RT code writes the atomic; `process()` calls `smoother.setTargetValue(m_target.load())` each block to avoid zipper noise.
- Call `smoother.reset(sampleRate, rampLengthSeconds)` in `prepare()`.

## Format and I/O

- Always use `juce::AudioFormatManager::registerBasicFormats()` in the constructor; create readers via `m_formatManager.createReaderFor(file)`.
- Check the returned `unique_ptr` for null before use.
- Use `DBG()` (never `std::cout`) for debug messages. Prefix messages with the class name: `DBG("IrLoader: ...")`.
- Use `juce::approximatelyEqual()` for floating-point sample-rate comparisons.
- Cast `reader->numChannels` / `reader->lengthInSamples` (which are `uint32`/`int64`) to `int` explicitly.

## Float arithmetic conventions

- Use `float` for per-sample audio math. Use `double` for accumulations (e.g. energy sums) or sample-rate ratios.
- Prefer `juce::roundToInt()` over `static_cast<int>(std::round(...))`.
- Use `juce::AudioBuffer::applyGain()`, `addFromWithRamp()`, `applyGainRamp()` for SIMD-friendly buffer operations instead of manual loops.

## Resampling

- Use `juce::ResamplingAudioSource` wrapped around a `juce::MemoryAudioSource`. Call `prepareToPlay(newLen, targetRate)` before reading.
- Compute the ratio as `sourceSampleRate / targetSampleRate` and output length as `juce::roundToInt(inputLen / ratio)`.

## Atomic memory ordering

- Publish: `store(..., std::memory_order_release)`.
- Consume: `load(..., std::memory_order_acquire)`.
- Internal (same-thread) reads where ordering does not matter: `std::memory_order_relaxed`.
- Never use `memory_order_seq_cst` without a documented reason.
