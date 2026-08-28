// SPDX-FileCopyrightText: 2026 Pier Luigi Fiorini
// SPDX-License-Identifier: MIT

#pragma once

/*******************************************************************************
 BEGIN_JUCE_MODULE_DECLARATION

  ID:               irconv
  vendor:           metaldsp
  version:          0.1.0
  name:             IR Convolution
  description:      Impulse response convolution library for JUCE
  website:
  license:          MIT
  minimumCppStandard: 23
  dependencies:     juce_dsp juce_audio_formats

 END_JUCE_MODULE_DECLARATION
*******************************************************************************/

#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_dsp/juce_dsp.h>

#include "source/DualIrLoader.h"
#include "source/IrFilter.h"
#include "source/IrLoader.h"
#include "source/MultiIrLoader.h"
#include "source/TimeAligner.h"
