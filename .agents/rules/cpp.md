# C++ Coding Standards

## File headers

Every `.cpp` and `.h` file must start with SPDX copyright and license lines:

```cpp
// SPDX-FileCopyrightText: <year> Pier Luigi Fiorini
// SPDX-License-Identifier: MIT
```

Use `#pragma once` (not include guards) in all headers.

## Namespace

All DSP classes live in the `DSP` namespace. Close with `} // namespace DSP`.

Anonymous namespaces in `.cpp` files contain file-local helpers (concrete `ConvChannel` subclasses, free functions). Static constants inside anonymous namespaces use `k` prefix + PascalCase (e.g. `kTwoStageThreshold`).

## Class member naming

Private members use `m_` prefix + camelCase (e.g. `m_activeSet`, `m_spec`). No Hungarian type encoding beyond the `m_` prefix.
