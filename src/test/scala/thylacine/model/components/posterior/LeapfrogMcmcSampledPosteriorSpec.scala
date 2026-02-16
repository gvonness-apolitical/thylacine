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
import thylacine.config.LeapfrogMcmcConfig
import thylacine.model.components.likelihood.{ GaussianLinearLikelihood, Likelihood }
import thylacine.model.components.prior.{ GaussianPrior, Prior }

import cats.effect.{ IO, Ref }
import cats.effect.testing.scalatest.AsyncIOSpec
import org.scalatest.freespec.AsyncFreeSpec
import org.scalatest.matchers.should.Matchers

import scala.concurrent.duration.*

class LeapfrogMcmcSampledPosteriorSpec extends AsyncFreeSpec with AsyncIOSpec with Matchers {

  private val euclidean: (Vector[Double], Vector[Double]) => Double =
    (a, b) => Math.sqrt(a.zip(b).map { case (x, y) => (x - y) * (x - y) }.sum)

  private val manhattan: (Vector[Double], Vector[Double]) => Double =
    (a, b) => a.zip(b).map { case (x, y) => Math.abs(x - y) }.sum

  private val noOpSetCallback: Int => IO[Unit]    = _ => IO.unit
  private val noOpUpdateCallback: Int => IO[Unit] = _ => IO.unit

  private val lightConfig: LeapfrogMcmcConfig = LeapfrogMcmcConfig(
    stepsBetweenSamples = 2,
    warmupStepCount     = 5,
    samplePoolSize      = 3
  )

  private val largePoolConfig: LeapfrogMcmcConfig = LeapfrogMcmcConfig(
    stepsBetweenSamples = 2,
    warmupStepCount     = 5,
    samplePoolSize      = 5
  )

  // 2D MCMC-friendly system: wide prior + moderate likelihood uncertainty.
  // Prior: mean=(0,0), CI=(10,10)
  // Likelihood: identity forward model, measurements=(1,2), sigma=(1,1)
  // True posterior mean ~ (1, 2)
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

  private def build2dSampler(
    config: LeapfrogMcmcConfig,
    distFn: (Vector[Double], Vector[Double]) => Double,
    setCallback: Int => IO[Unit],
    updateCallback: Int => IO[Unit]
  )(implicit
    stm: STM[IO]
  ): IO[LeapfrogMcmcSampledPosterior[IO]] =
    for {
      likelihood <- mcmcLikelihoodF
      posterior = UnnormalisedPosterior[IO](
                    priors      = Set[Prior[IO, ?]](mcmcPrior),
                    likelihoods = Set[Likelihood[IO, ?, ?]](likelihood)
                  )
      sampler <- LeapfrogMcmcSampledPosterior.of[IO](
                   leapfrogMcmcConfig          = config,
                   distanceCalculation         = distFn,
                   posterior                   = posterior,
                   sampleRequestSetCallback    = setCallback,
                   sampleRequestUpdateCallback = updateCallback,
                   seed                        = Map("p" -> Vector(1.0, 2.0))
                 )
    } yield sampler

  "LeapfrogMcmcSampledPosterior" - {

    "produce samples near the known posterior mean" in {
      STM
        .runtime[IO]
        .flatMap { implicit stm =>
          for {
            sampler <- build2dSampler(lightConfig, euclidean, noOpSetCallback, noOpUpdateCallback)
            samples <- sampler.sample(5).timeout(120.seconds)
            sampleList = samples.toList
            mean0      = sampleList.map(_("p").head).sum / sampleList.size
            mean1      = sampleList.map(_("p")(1)).sum / sampleList.size
          } yield (mean0, mean1)
        }
        .asserting { case (m0, m1) =>
          m0 shouldBe (1.0 +- 5.0)
          m1 shouldBe (2.0 +- 5.0)
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
            posterior1d = UnnormalisedPosterior[IO](
                            priors      = Set[Prior[IO, ?]](prior1d),
                            likelihoods = Set[Likelihood[IO, ?, ?]](likelihood1d)
                          )
            sampler <- LeapfrogMcmcSampledPosterior.of[IO](
                         leapfrogMcmcConfig          = lightConfig,
                         distanceCalculation         = euclidean,
                         posterior                   = posterior1d,
                         sampleRequestSetCallback    = noOpSetCallback,
                         sampleRequestUpdateCallback = noOpUpdateCallback,
                         seed                        = Map("x" -> Vector(3.0))
                       )
            samples <- sampler.sample(5).timeout(120.seconds)
            sampleList = samples.toList
            mean       = sampleList.map(_("x").head).sum / sampleList.size
          } yield mean
        }
        .asserting { m =>
          m shouldBe (3.0 +- 5.0)
        }
    }

    "produce samples from a 3D posterior" in {
      STM
        .runtime[IO]
        .flatMap { implicit stm =>
          val prior3d = GaussianPrior.fromConfidenceIntervals[IO](
            label               = "v",
            values              = Vector.fill(3)(0.0),
            confidenceIntervals = Vector.fill(3)(10.0)
          )
          for {
            likelihood3d <- GaussianLinearLikelihood.of[IO](
                              coefficients =
                                (0 until 3).map(i => (0 until 3).map(j => if (i == j) 1.0 else 0.0).toVector).toVector,
                              measurements   = Vector(1.0, 2.0, 3.0),
                              uncertainties  = Vector.fill(3)(1.0),
                              priorLabel     = "v",
                              evalCacheDepth = None
                            )
            posterior3d = UnnormalisedPosterior[IO](
                            priors      = Set[Prior[IO, ?]](prior3d),
                            likelihoods = Set[Likelihood[IO, ?, ?]](likelihood3d)
                          )
            sampler <- LeapfrogMcmcSampledPosterior.of[IO](
                         leapfrogMcmcConfig          = lightConfig,
                         distanceCalculation         = euclidean,
                         posterior                   = posterior3d,
                         sampleRequestSetCallback    = noOpSetCallback,
                         sampleRequestUpdateCallback = noOpUpdateCallback,
                         seed                        = Map("v" -> Vector(1.0, 2.0, 3.0))
                       )
            samples <- sampler.sample(5).timeout(120.seconds)
          } yield samples
        }
        .asserting { samples =>
          samples should not be empty
          // Each sample should have a 3-element vector for label "v"
          samples.foreach(s => s("v") should have size 3)
          succeed
        }
    }

    "produce samples with reasonable spread" in {
      STM
        .runtime[IO]
        .flatMap { implicit stm =>
          for {
            sampler <- build2dSampler(lightConfig, euclidean, noOpSetCallback, noOpUpdateCallback)
            samples <- sampler.sample(5).timeout(120.seconds)
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
          s0 should be < 10.0
          s1 should be > 0.01
          s1 should be < 10.0
        }
    }

    "produce distinct samples" in {
      STM
        .runtime[IO]
        .flatMap { implicit stm =>
          for {
            sampler <- build2dSampler(lightConfig, euclidean, noOpSetCallback, noOpUpdateCallback)
            samples <- sampler.sample(5).timeout(120.seconds)
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
            sampler <- build2dSampler(lightConfig, euclidean, noOpSetCallback, noOpUpdateCallback)
            samples <- sampler.sample(5).timeout(120.seconds)
          } yield samples
        }
        .asserting { samples =>
          samples should not be empty
        }
    }

    "invoke sample request callbacks during sampling" in {
      STM
        .runtime[IO]
        .flatMap { implicit stm =>
          for {
            ref <- Ref.of[IO, List[Int]](List.empty)
            setCallback = (n: Int) => ref.update(n :: _)
            sampler <- build2dSampler(lightConfig, euclidean, setCallback, noOpUpdateCallback)
            _       <- sampler.sample(5).timeout(120.seconds)
            calls   <- ref.get
          } yield calls
        }
        .asserting { calls =>
          calls should not be empty
        }
    }

    "work with Manhattan distance function" in {
      STM
        .runtime[IO]
        .flatMap { implicit stm =>
          for {
            sampler <- build2dSampler(lightConfig, manhattan, noOpSetCallback, noOpUpdateCallback)
            samples <- sampler.sample(5).timeout(120.seconds)
          } yield samples
        }
        .asserting { samples =>
          samples should not be empty
        }
    }

    "work with a larger sample pool" in {
      STM
        .runtime[IO]
        .flatMap { implicit stm =>
          for {
            sampler <- build2dSampler(largePoolConfig, euclidean, noOpSetCallback, noOpUpdateCallback)
            samples <- sampler.sample(5).timeout(120.seconds)
          } yield samples
        }
        .asserting { samples =>
          samples should not be empty
        }
    }
  }
}
