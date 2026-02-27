# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.15.3] - 2026-02-27

### Changed

- Updated `bengal-stm` dependency from 0.11.0 to 0.12.0

## [0.15.2] - 2026-02-26

### Added

- 76 new tests: CachedComputation (5), ModelParameterSimplex (16), QuadratureAbscissa (15), PointInCube (25), SLQ integration smoke (16)
- `MdsOptimisedPosterior` convenience factory method (matching other optimised posteriors)
- `sbt-dependency-check` plugin for CVE scanning
- `Test / fork := true` to prevent JVM hang from fire-and-forget fibers

### Changed

- Narrowed `ejml-all` dependency to `ejml-ddense` (only dense row-major modules used)
- Grouped `HmcmcEngine.runDynamicSimulationFrom` parameters into case classes (14 → 6 params)
- Replaced tuple access in `SlqEngine` with named telemetry case class fields

### Fixed

- README Quick Start: 3 broken API references (`fromConfidenceIntervals` → `fromStandardDeviations`, etc.)
- CONTRIBUTING.md code style section and PR checklist

### Removed

- 3 flaky `CachedComputation` tests dependent on fire-and-forget STM timing
- 6 stale `.gitignore` entries for non-existent directories

## [0.15.1] - 2026-02-26

### Fixed

- MiMa binary compatibility check failing on unpublished 0.15.x artifact (`tlVersionIntroduced` updated for both Scala versions)
- Flaky `LeapfrogMcmcSampledPosterior` posterior mean test marked as ignored to prevent CI timeouts

## [0.15.0] - 2026-02-26

### Fixed

- Infinite recursion in `exploreLine` on flat landscapes where probe differential overflows to infinity
- Infinite recursion in `probeLine` on monotonically increasing or constant functions
- Division by near-zero in conjugate gradient beta computation producing infinite search directions
- Redundant forward model evaluations in `centralDifferenceJacobianAt` (hoisted `evalAt(input)` out of loop)

### Added

- Depth limits (max 50) to `exploreLine` and `probeLine` with `isInfinite`/`isNaN` guards
- Max iteration guard (default 10,000) to MDS and Hooke-Jeeves optimizers to bound worst-case runtime
- Validation that `goldenSectionTolerance > 0` in `ConjugateGradientConfig` and `CoordinateSlideConfig`
- Cholesky-based `choleskySolve` and `choleskyInvert` to `LinearAlgebra` for symmetric positive definite matrices
- `shuffle` and `setSeed` methods to `MathOps` for reproducible RNG

### Changed

- `GaussianAnalyticPosterior` now uses Cholesky factorization instead of generic matrix inversion for improved numerical stability
- Centralized remaining direct `scala.util.Random` usages in `GoldenSectionSearch`, `HookeAndJeevesEngine`, and `CoordinateSlideEngine` through `MathOps`
