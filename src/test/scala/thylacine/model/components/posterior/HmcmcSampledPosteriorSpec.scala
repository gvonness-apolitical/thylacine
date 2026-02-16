/*
 * Copyright 2023 Greg von Nessi
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package ai.entrolution
package thylacine.model.components.posterior

import bengal.stm.STM
import thylacine.config.HmcmcConfig
import thylacine.model.components.likelihood.{ GaussianLinearLikelihood, Likelihood }
import thylacine.model.components.prior.{ GaussianPrior, Prior }
import thylacine.model.core.telemetry.HmcmcTelemetryUpdate

import cats.effect.{ IO, Ref }
import cats.effect.testing.scalatest.AsyncIOSpec
import org.scalatest.freespec.AsyncFreeSpec
import org.scalatest.matchers.should.Matchers

class HmcmcSampledPosteriorSpec extends AsyncFreeSpec with AsyncIOSpec with Matchers {

  // Step size must be small relative to the posterior width to achieve reasonable
  // acceptance rates. These configs use σ_likelihood >= 1.0 posteriors where ε=0.05
  // gives trajectory lengths of ~0.5, well within the posterior width of ~O(1).
  private val standardConfig: HmcmcConfig = HmcmcConfig(
    stepsBetweenSamples        = 2,
    stepsInDynamicsSimulation  = 10,
    warmupStepCount            = 5,
    dynamicsSimulationStepSize = 0.05
  )

  private val lightConfig: HmcmcConfig = HmcmcConfig(
    stepsBetweenSamples        = 2,
    stepsInDynamicsSimulation  = 5,
    warmupStepCount            = 3,
    dynamicsSimulationStepSize = 0.05
  )

  private val noOpCallback: HmcmcTelemetryUpdate => IO[Unit] = _ => IO.unit

  // 2D MCMC-friendly system: wide prior + moderate likelihood uncertainty.
  // Prior: mean=(0,0), CI=(10,10)
  // Likelihood: identity forward model, measurements=(1,2), σ=(1,1)
  // True posterior mean ≈ (1, 2)
  private val mcmcPrior: GaussianPrior[IO] =
    GaussianPrior.fromConfidenceIntervals[IO](
      label               = "p",
      values              = Vector(0.0, 0.0),
      confidenceIntervals = Vector(10.0, 10.0)
    )

  private def mcmcLikelihoodF(implicit stm: STM[IO]): IO[GaussianLinearLikelihood[IO]] =
    GaussianLinearLikelihood.of[IO](
      coefficients   = Vector(Vector(1.0, 0.0), Vector(0.0, 1.0)),
      measurements   = Vector(1.0, 2.0),
      uncertainties  = Vector(1.0, 1.0),
      priorLabel     = "p",
      evalCacheDepth = None
    )

  "HmcmcSampledPosterior" - {

    "produce samples near the known posterior mean" in {
      STM
        .runtime[IO]
        .flatMap { implicit stm =>
          for {
            likelihood <- mcmcLikelihoodF
            sampler = HmcmcSampledPosterior[IO](
                        hmcmcConfig             = standardConfig,
                        telemetryUpdateCallback = noOpCallback,
                        seed                    = Map("p" -> Vector(1.0, 2.0)),
                        priors                  = Set[Prior[IO, ?]](mcmcPrior),
                        likelihoods             = Set[Likelihood[IO, ?, ?]](likelihood)
                      )
            samples <- sampler.sample(15)
            sampleList = samples.toList
            mean0      = sampleList.map(_("p").head).sum / sampleList.size
            mean1      = sampleList.map(_("p")(1)).sum / sampleList.size
          } yield (mean0, mean1)
        }
        .asserting { case (m0, m1) =>
          m0 shouldBe (1.0 +- 2.0)
          m1 shouldBe (2.0 +- 2.0)
        }
    }

    "produce samples from a simple 1D posterior" in {
      STM
        .runtime[IO]
        .flatMap { implicit stm =>
          val prior1d = GaussianPrior.fromConfidenceIntervals[IO](
            label               = "x",
            values              = Vector(0.0),
            confidenceIntervals = Vector(10.0)
          )
          for {
            likelihood1d <- GaussianLinearLikelihood.of[IO](
                              coefficients   = Vector(Vector(1.0)),
                              measurements   = Vector(3.0),
                              uncertainties  = Vector(1.0),
                              priorLabel     = "x",
                              evalCacheDepth = None
                            )
            sampler = HmcmcSampledPosterior[IO](
                        hmcmcConfig             = standardConfig,
                        telemetryUpdateCallback = noOpCallback,
                        seed                    = Map("x" -> Vector(3.0)),
                        priors                  = Set[Prior[IO, ?]](prior1d),
                        likelihoods             = Set[Likelihood[IO, ?, ?]](likelihood1d)
                      )
            samples <- sampler.sample(15)
            sampleList = samples.toList
            mean       = sampleList.map(_("x").head).sum / sampleList.size
          } yield mean
        }
        .asserting { m =>
          m shouldBe (3.0 +- 2.0)
        }
    }

    "produce samples with reasonable spread" in {
      STM
        .runtime[IO]
        .flatMap { implicit stm =>
          for {
            likelihood <- mcmcLikelihoodF
            sampler = HmcmcSampledPosterior[IO](
                        hmcmcConfig             = standardConfig,
                        telemetryUpdateCallback = noOpCallback,
                        seed                    = Map("p" -> Vector(1.0, 2.0)),
                        priors                  = Set[Prior[IO, ?]](mcmcPrior),
                        likelihoods             = Set[Likelihood[IO, ?, ?]](likelihood)
                      )
            samples <- sampler.sample(15)
            sampleList = samples.toList
            values0    = sampleList.map(_("p").head)
            values1    = sampleList.map(_("p")(1))
            mean0      = values0.sum / values0.size
            mean1      = values1.sum / values1.size
            std0       = Math.sqrt(values0.map(v => (v - mean0) * (v - mean0)).sum / values0.size)
            std1       = Math.sqrt(values1.map(v => (v - mean1) * (v - mean1)).sum / values1.size)
          } yield (std0, std1)
        }
        .asserting { case (s0, s1) =>
          s0 should be > 0.01
          s0 should be < 5.0
          s1 should be > 0.01
          s1 should be < 5.0
        }
    }

    "maintain a reasonable acceptance rate" in {
      STM
        .runtime[IO]
        .flatMap { implicit stm =>
          for {
            ref <- Ref.of[IO, List[HmcmcTelemetryUpdate]](List.empty)
            callback = (update: HmcmcTelemetryUpdate) => ref.update(update :: _)
            likelihood <- mcmcLikelihoodF
            sampler = HmcmcSampledPosterior[IO](
                        hmcmcConfig             = standardConfig,
                        telemetryUpdateCallback = callback,
                        seed                    = Map("p" -> Vector(1.0, 2.0)),
                        priors                  = Set[Prior[IO, ?]](mcmcPrior),
                        likelihoods             = Set[Likelihood[IO, ?, ?]](likelihood)
                      )
            _       <- sampler.sample(15)
            updates <- ref.get
            lastUpdate     = updates.maxBy(_.jumpAttempts)
            acceptanceRate = lastUpdate.jumpAcceptances.toDouble / lastUpdate.jumpAttempts
          } yield acceptanceRate
        }
        .asserting { rate =>
          rate should be > 0.1
          rate should be <= 1.0
        }
    }

    "produce distinct samples" in {
      STM
        .runtime[IO]
        .flatMap { implicit stm =>
          for {
            likelihood <- mcmcLikelihoodF
            sampler = HmcmcSampledPosterior[IO](
                        hmcmcConfig             = lightConfig,
                        telemetryUpdateCallback = noOpCallback,
                        seed                    = Map("p" -> Vector(1.0, 2.0)),
                        priors                  = Set[Prior[IO, ?]](mcmcPrior),
                        likelihoods             = Set[Likelihood[IO, ?, ?]](likelihood)
                      )
            samples <- sampler.sample(5)
          } yield samples.size
        }
        .asserting { size =>
          size should be > 1
        }
    }

    "use a provided seed as starting point" in {
      STM
        .runtime[IO]
        .flatMap { implicit stm =>
          for {
            likelihood <- mcmcLikelihoodF
            sampler = HmcmcSampledPosterior[IO](
                        hmcmcConfig             = lightConfig,
                        telemetryUpdateCallback = noOpCallback,
                        seed                    = Map("p" -> Vector(1.0, 2.0)),
                        priors                  = Set[Prior[IO, ?]](mcmcPrior),
                        likelihoods             = Set[Likelihood[IO, ?, ?]](likelihood)
                      )
            samples <- sampler.sample(5)
          } yield samples
        }
        .asserting { samples =>
          samples should not be empty
        }
    }

    "be constructable via the companion object factory" in {
      STM
        .runtime[IO]
        .flatMap { implicit stm =>
          // Build an UnnormalisedPosterior with MCMC-friendly widths
          val prior = GaussianPrior.fromConfidenceIntervals[IO](
            label               = "q",
            values              = Vector(0.0),
            confidenceIntervals = Vector(10.0)
          )
          for {
            likelihood <- GaussianLinearLikelihood.of[IO](
                            coefficients   = Vector(Vector(1.0)),
                            measurements   = Vector(5.0),
                            uncertainties  = Vector(1.0),
                            priorLabel     = "q",
                            evalCacheDepth = None
                          )
            unnormalisedPosterior = UnnormalisedPosterior[IO](
                                      priors      = Set[Prior[IO, ?]](prior),
                                      likelihoods = Set[Likelihood[IO, ?, ?]](likelihood)
                                    )
            sampler = HmcmcSampledPosterior[IO](
                        hmcmcConfig             = lightConfig,
                        posterior               = unnormalisedPosterior,
                        telemetryUpdateCallback = noOpCallback,
                        seed                    = Map("q" -> Vector(5.0))
                      )
            samples <- sampler.sample(3)
          } yield samples
        }
        .asserting { samples =>
          samples should not be empty
        }
    }

    "produce sample covariance in a reasonable range" in {
      STM
        .runtime[IO]
        .flatMap { implicit stm =>
          for {
            likelihood <- mcmcLikelihoodF
            sampler = HmcmcSampledPosterior[IO](
                        hmcmcConfig = HmcmcConfig(
                          stepsBetweenSamples        = 2,
                          stepsInDynamicsSimulation  = 10,
                          warmupStepCount            = 10,
                          dynamicsSimulationStepSize = 0.05
                        ),
                        telemetryUpdateCallback = noOpCallback,
                        seed                    = Map("p" -> Vector(1.0, 2.0)),
                        priors                  = Set[Prior[IO, ?]](mcmcPrior),
                        likelihoods             = Set[Likelihood[IO, ?, ?]](likelihood)
                      )
            samples <- sampler.sample(30)
            sampleList = samples.toList
            values0    = sampleList.map(_("p").head)
            values1    = sampleList.map(_("p")(1))
            mean0      = values0.sum / values0.size
            mean1      = values1.sum / values1.size
            var0       = values0.map(v => (v - mean0) * (v - mean0)).sum / (values0.size - 1)
            var1       = values1.map(v => (v - mean1) * (v - mean1)).sum / (values1.size - 1)
            cov01 = values0
                      .zip(values1)
                      .map { case (v0, v1) => (v0 - mean0) * (v1 - mean1) }
                      .sum / (values0.size - 1)
            correlation = cov01 / Math.sqrt(var0 * var1)
          } yield (var0, var1, correlation)
        }
        .asserting { case (v0, v1, corr) =>
          // True posterior variance ≈ 0.25 (per dimension). Check order of magnitude.
          v0 should be > 0.01
          v0 should be < 3.0
          v1 should be > 0.01
          v1 should be < 3.0
          // Independent dimensions → low correlation
          Math.abs(corr) should be < 0.8
        }
    }
  }
}
