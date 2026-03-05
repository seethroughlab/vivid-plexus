# vivid-plexus

`vivid-plexus` is a Vivid package library that provides:

- `Plexus` (GPU)
- `PlexusSynth` (audio)

## Contents

- `src/plexus.cpp`
- `src/plexus_synth.cpp`
- `graphs/plexus_demo.json`
- `tests/test_package_manifest.cpp`
- `vivid-package.json`

## Local development

From vivid-core:

```bash
./build/vivid link ../vivid-plexus
./build/vivid rebuild vivid-plexus
```

## CI smoke coverage

The package CI workflow:

1. Clones and builds vivid-core (`test_demo_graphs` + core operators).
2. Builds package operators and package tests.
3. Runs package tests.
4. Runs graph smoke tests against this package's `graphs/` directory.

## License

MIT (see `LICENSE`).
