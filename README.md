# irconv

A [JUCE module](https://juce.com) that provides impulse response convolution for audio applications and plugins.

## Requirements

| Requirement | Version |
|---|---|
| JUCE | 8.0.12 or later |
| C++ | 23 |
| CMake | 3.28 or later |

## Adding to your project

Add this repository as a git submodule **inside a folder named `irconv`** — JUCE derives the module ID from the directory name:

```sh
git submodule add <url> modules/irconv
git submodule update --init --recursive
```

Then register the module and link it in your `CMakeLists.txt`:

```cmake
juce_add_module(${CMAKE_CURRENT_SOURCE_DIR}/modules/irconv)

target_link_libraries(MyTarget
    PRIVATE
        irconv
)
```

JUCE must already be available before `juce_add_module` is called. The module declares the following dependencies, which JUCE will pull in automatically:

- `juce_dsp`
- `juce_audio_formats`

## Repository layout

```
irconv/
├── irconv.h        # JUCE module declaration and public header umbrella
├── irconv.cpp      # Module unity-build entry point
├── docs/           # Per-class documentation
└── source/         # Implementation (headers and .cpp files)
```

The `source/` directory provides `DSP::IrLoader` (partitioned FFT convolution),
`DSP::IrFilter` (per-IR HPF/LPF voicing), `DSP::TimeAligner` (delay and polarity
analysis), `DSP::DualIrLoader` (two IRs blended or split across the stereo field)
and `DSP::MultiIrLoader` (N IRs convolved in parallel and mixed with per-slot
weights, for morph-pad style controls).

## Documentation

- [`DSP::MultiIrLoader`](docs/MultiIrLoader.md) — N-slot parallel convolution
  with weighted mixing.

## License

MIT. See source file headers for details.
