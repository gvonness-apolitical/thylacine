# Thylacine

[![CI](https://github.com/Entrolution/thylacine/actions/workflows/ci.yml/badge.svg)](https://github.com/Entrolution/thylacine/actions/workflows/ci.yml)
[![Maven Central](https://img.shields.io/maven-central/v/ai.entrolution/thylacine_2.13)](https://central.sonatype.com/artifact/ai.entrolution/thylacine_2.13)
![Scala 2.13](https://img.shields.io/badge/Scala-2.13-red?logo=scala)
![Scala 3](https://img.shields.io/badge/Scala-3-red?logo=scala)
![Java 21+](https://img.shields.io/badge/Java-21%2B-blue?logo=openjdk)

Thylacine is under active development. The core inference engine, algorithms, and API are functional and published to Maven Central, but the framework continues to evolve.

Thylacine is a FP (Functional Programming) Bayesian inference framework that facilitates sampling, integration and
subsequent statistical analysis on posterior distributions born out of a Bayesian inference.

A few aspects differentiate Thylacine from other Bayesian inference frameworks:

* Fully FP - Designed from the ground-up to fully leverage the virtues of FP in internal computations, while providing
  generic FP API.
* Framework and design largely de-coupled from Bayesian graphical representations of the problem (more details in the
  FAQ below)
* Designed to be multi-threaded from the ground up using a high-performance Software Transactional Memory (STM)
  implementation to facilitate efficient concurrency across the differing computational profiles for sampling,
  integration, visualisation, etc.
* Analytic gradient calculations can be specified on a component level that enables automatic differentiation to be used
  in gradient calculations. This is essential if aiming to perform high-dimension inference using gradient information.
* A growing list of advanced algorithms/concepts have already been implemented:
    * Gaussian analytic posteriors
    * Optimisation
        * Hooke & Jeeves
        * Coordinate line search (Coordinate Slide)
        * Multi-direction search (MDS)
        * Nonlinear conjugate gradient
    * Sampling
        * Hamiltonian MCMC
        * Leapfrog MCMC
    * Integration/Sampling
        * An advanced version of Nested Sampling: Stochastic Lebesgue Quadrature (SLQ)
    * General
        * Component-wise automatic differentiation that falls back to finite-differences when analytic gradient
          calculations are not available for a particular component
        * Likelihoods that support in-memory caching of evaluation and Jacobian calculations
        * Support for Cauchy distributions (in addition to standard Gaussian and uniform distributions)

---

## Theory

The theory of Bayesian inference is too vast to cover here. However, two references I highly recommend are:

* [Information Theory, Inference, and Learning Algorithms](https://www.inference.org.uk/itprnn/book.pdf) - An amazing
  book that shows how deeply connected Bayesian inference, coding theory, information theory, ML models and learning
  really are (i.e. all different facets of the same concept). What's even better is that David MacKay has made the book
  freely available to the public (definitely worth buying if you like physical books though).
* [Data Analysis: A Bayesian Tutorial](https://blackwells.co.uk/bookshop/product/Data-Analysis-by-D-S-Sivia-J-Skilling/9780198568322) -
  While many references can get bogged down in theory that can be cumbersome to penetrate for pragmatic approaches, this
  book cuts right to the chase and focuses on experimental data analysis from the start. This is an excellent reference
  for people who want to get up and running fast with Bayesian methods. The book is also a much faster read at ~250
  pages than most other references out there.

---

## Quick Start

To use Thylacine in an existing SBT project with Scala 2.13 or 3, add the following dependency to your
`build.sbt`:

```scala
libraryDependencies += "ai.entrolution" %% "thylacine" % VERSION
```

See the Maven badge above for the latest version.

### Example: 1D Bayesian Update

A Gaussian prior combined with a Gaussian likelihood yields an analytic posterior.
This example infers a single parameter `x` with a `N(0, 1)` prior and an observation
`y = 2` with measurement variance `1`, giving a posterior of `N(1, 0.5)`.

```scala
import ai.entrolution.thylacine.model.components.likelihood.GaussianLinearLikelihood
import ai.entrolution.thylacine.model.components.posterior.GaussianAnalyticPosterior
import ai.entrolution.thylacine.model.components.prior.GaussianPrior
import bengal.stm.STM
import cats.effect.{IO, IOApp}

object BayesianUpdateExample extends IOApp.Simple {
  def run: IO[Unit] =
    STM.runtime[IO].flatMap { implicit stm =>
      val prior = GaussianPrior.fromConfidenceIntervals[IO](
        label               = "x",
        values              = Vector(0.0),
        confidenceIntervals = Vector(2.0) // 95% CI width of 2 => variance = 1
      )
      for {
        likelihood <- GaussianLinearLikelihood.of[IO](
                        coefficients   = Vector(Vector(1.0)),
                        measurements   = Vector(2.0),
                        uncertainties  = Vector(2.0), // 95% CI width of 2 => variance = 1
                        priorLabel     = "x",
                        evalCacheDepth = None
                      )
        posterior = GaussianAnalyticPosterior[IO](
                      priors      = Set(prior),
                      likelihoods = Set(likelihood)
                    )
        _ <- posterior.init
        _ <- IO.println(s"Posterior mean: ${posterior.mean}")
        _ <- IO.println(s"Posterior covariance: ${posterior.covarianceStridedVector}")
      } yield ()
    }
}
// Posterior mean: Map(x -> Vector(1.0))
// Posterior covariance: Vector(0.5)
```

---

## Framework Documentation

API documentation is generated during CI for each Scala version. For questions and discussion, see [GitHub Discussions](https://github.com/Entrolution/thylacine/discussions).

---
---

## PR FAQ

### What is a PR FAQ?

Take a look
at [the Medium article on PR FAQs](https://medium.com/agileinsider/press-releases-for-product-managers-everything-you-need-to-know-942485961e31)
for a good overview of the concept. I have taken some liberties with the formatting, but I generally like the concept of
a living FAQ to help introduce products.

### Why another Bayesian framework?

I became involved with Bayesian techniques a long time ago when I was working as a plasma physicist. In that time most
of the Bayesian data analysis I did was via a bespoke framework written in Java. Over the course of my research, I saw
many avenues of improvement for this framework that mostly centred around data modelling and performance optimisations.
However, the nature of research did not afford me to the time to work on tooling improvement over getting research
results.

After I left academia, I wanted to implement these improvements but lacked a good problem to test my ideas against.
Also, extending the framework I was using wasn't really feasible, as the project was closed sourced with commercial
aspirations. Eventually, I came across an interesting problem of inferring mass density distributions of Katana when I
started training in Battojutsu. As I continued to flesh out details in the forward model for this problem, the
supporting inference code became more complex; and it became clear it was time to extract out the underpinning framework
from the application code. Hence, Thylacine was born.

### What about Graphical models?

The framework is decoupled from any concepts in Bayesian graphical models by design. Indeed, I found previously that
trying to integrate graphical concepts into a low-level Bayesian framework led to unneeded coupling within the data
model, when just needing to perform posterior analysis. Indeed, priors and likelihoods can be formulated in any desired
context and then fed into this framework.

More abstractly, the concept of probabilistic graphical models is not intrinsically linked to Bayesian analysis. Indeed,
frequentist methods can also be used to process these graphs representations. Given this, I decided it did not make
sense to artificially couple two concepts within this framework that are not intrinsically linked together (for
more-or-less standard software engineering reasons not to introduce unneeded coupling in one's code).

### How is the framework tested?

The project has a comprehensive test suite covering analytic posteriors, numerical gradient computation, likelihood
evaluation, MCMC sampling, and optimisation algorithms. Tests run against both Scala 2.13 and Scala 3 in CI.

### Why 'Thylacine'?

Tasmanian tigers have been one of my favourite animals since I was a kid. I really hope we can bring them back someday.
I named this framework after them, as this framework is about meaningfully merging data from a heterogenous collection
of measurements, and I see thylacines as a bit of a chimera with respect to other animals: they are marsupials,
carnivorous, have stripes, and exhibit both feline and canine qualities. I.e. both are about combining different parts
to get a better whole. If that's too much of a stretch, then let's just say it's artistic license :).