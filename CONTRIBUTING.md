# Contributing to Thylacine

Thank you for your interest in contributing to Thylacine! This guide explains how to get involved.

## Reporting Bugs

Open a [GitHub Issue](https://github.com/Entrolution/thylacine/issues/new?template=bug_report.md) with:

- A clear description of the problem
- Steps to reproduce
- Expected vs actual behaviour
- Scala version, Java version, and OS

## Feature Requests

Open a [GitHub Issue](https://github.com/Entrolution/thylacine/issues/new?template=feature_request.md) describing the motivation and proposed approach.

## Development Setup

### Prerequisites

- **Java 21+** (Temurin recommended)
- **sbt** (latest)

### Build and Test

```bash
# Compile for all Scala versions
sbt +compile

# Run tests for all Scala versions
sbt +test

# Check formatting
sbt scalafmtCheckAll scalafmtSbtCheck

# Auto-format
sbt scalafmtAll scalafmtSbt

# Check headers
sbt headerCheckAll

# Check binary compatibility
sbt +mimaReportBinaryIssues
```

## Submitting Changes

1. Fork the repository
2. Create a feature branch from `main`
3. Make your changes
4. Ensure `sbt +test` passes
5. Ensure `sbt scalafmtAll` has been run
6. Ensure `sbt headerCheckAll` passes (license headers)
7. Open a pull request against `main`

### PR Expectations

- One logical change per PR
- Tests required for new functionality
- CI must pass (both Scala 2.13 and 3)
- Keep the PR description clear and concise

## Code Style

This project uses [scalafmt](https://scalameta.org/scalafmt/) for code formatting. Configuration is in `.scalafmt.conf`. Run `sbt scalafmtAll` before committing to ensure consistency. Import ordering rules are configured in `.scalafix.conf`.

## License

By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).
