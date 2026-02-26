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
package thylacine.model.integration.slq

import bengal.stm.STM
import thylacine.config.SlqConfig
import thylacine.model.components.likelihood.{ GaussianLinearLikelihood, Likelihood }
import thylacine.model.components.posterior.{ SlqIntegratedPosterior, UnnormalisedPosterior }
import thylacine.model.components.prior.{ GaussianPrior, Prior }
import thylacine.model.core.telemetry.SlqTelemetryUpdate

import cats.effect.IO
import cats.effect.testing.scalatest.AsyncIOSpec
import org.scalatest.freespec.AsyncFreeSpec
import org.scalatest.matchers.should.Matchers

class SlqIntegrationSmokeSpec extends AsyncFreeSpec with AsyncIOSpec with Matchers {

  private val smokeSlqConfig: SlqConfig = SlqConfig(
    poolSize                    = 3,
    abscissaNumber              = 2,
    domainScalingIncrement      = 0.01,
    targetAcceptanceProbability = 0.5,
    sampleParallelism           = 1,
    maxIterationCount           = 20,
    minIterationCount           = 5
  )

  private val noOpTelemetry: SlqTelemetryUpdate => IO[Unit] = _ => IO.unit
  private val noOpCallback: Unit => IO[Unit]                = _ => IO.unit

  private val prior1d: GaussianPrior[IO] =
    GaussianPrior.fromStandardDeviations[IO](
      label              = "x",
      values             = Vector(0.0),
      standardDeviations = Vector(5.0)
    )

  private def likelihood1dF(implicit stm: STM[IO]): IO[GaussianLinearLikelihood[IO]] =
    GaussianLinearLikelihood.of[IO](
      coefficients       = Vector(Vector(1.0)),
      measurements       = Vector(3.0),
      standardDeviations = Vector(1.0),
      priorLabel         = "x",
      evalCacheDepth     = None
    )

  "SlqIntegratedPosterior" - {

    "be constructable from an UnnormalisedPosterior" ignore {
      STM
        .runtime[IO]
        .flatMap { implicit stm =>
          for {
            likelihood <- likelihood1dF
            unnormalisedPosterior = UnnormalisedPosterior[IO](
                                      priors      = Set[Prior[IO, ?]](prior1d),
                                      likelihoods = Set[Likelihood[IO, ?, ?]](likelihood)
                                    )
            slqPosterior <- SlqIntegratedPosterior.of[IO](
                              slqConfig                   = smokeSlqConfig,
                              posterior                   = unnormalisedPosterior,
                              slqTelemetryUpdateCallback  = noOpTelemetry,
                              domainRebuildStartCallback  = noOpCallback,
                              domainRebuildFinishCallback = noOpCallback,
                              seedsSpec                   = IO.pure(Set(Map("x" -> Vector(3.0))))
                            )
          } yield slqPosterior
        }
        .asserting { posterior =>
          posterior.priors should not be empty
          posterior.likelihoods should not be empty
        }
    }

    "build and complete a sample simulation for a 1D Gaussian" ignore {
      STM
        .runtime[IO]
        .flatMap { implicit stm =>
          for {
            likelihood <- likelihood1dF
            unnormalisedPosterior = UnnormalisedPosterior[IO](
                                      priors      = Set[Prior[IO, ?]](prior1d),
                                      likelihoods = Set[Likelihood[IO, ?, ?]](likelihood)
                                    )
            slqPosterior <- SlqIntegratedPosterior.of[IO](
                              slqConfig                   = smokeSlqConfig,
                              posterior                   = unnormalisedPosterior,
                              slqTelemetryUpdateCallback  = noOpTelemetry,
                              domainRebuildStartCallback  = noOpCallback,
                              domainRebuildFinishCallback = noOpCallback,
                              seedsSpec                   = IO.pure(Set(Map("x" -> Vector(3.0))))
                            )
            _       <- slqPosterior.buildSampleSimulation
            _       <- slqPosterior.waitForSimulationConstruction
            samples <- slqPosterior.sample(5)
          } yield samples
        }
        .asserting { samples =>
          samples should not be empty
          all(samples.map(_.keys)) should contain("x")
        }
    }

  }

  "QuadratureIntegrator" - {

    "compute evidence stats for synthetic data" in {
      IO {
        val logPdfs     = Vector(-1.0, -0.5, -0.1, -0.5, -1.0)
        val quadratures = Vector(Vector(0.2, 0.2, 0.2, 0.2, 0.2))
        val integrator  = QuadratureIntegrator(logPdfs, quadratures)
        val evidence    = integrator.evidenceStats
        evidence should not be empty
        evidence.foreach { e =>
          e should be > BigDecimal(0)
        }
        succeed
      }
    }

    "compute negative entropy stats for synthetic data" in {
      IO {
        val logPdfs     = Vector(-1.0, -0.5, -0.1, -0.5, -1.0)
        val quadratures = Vector(Vector(0.2, 0.2, 0.2, 0.2, 0.2))
        val integrator  = QuadratureIntegrator(logPdfs, quadratures)
        val negEntropy  = integrator.negativeEntropyStats
        negEntropy should not be empty
        negEntropy.foreach { ne =>
          ne.toDouble should not be Double.NaN
        }
        succeed
      }
    }

    "integrate a constant function over synthetic data" in {
      IO {
        val logPdfs     = Vector(-1.0, -0.5, -0.1, -0.5, -1.0)
        val quadratures = Vector(Vector(0.2, 0.2, 0.2, 0.2, 0.2))
        val integrator  = QuadratureIntegrator(logPdfs, quadratures)
        val result      = integrator.getIntegrationStats(x => x * BigDecimal(0) + BigDecimal(1))
        result should not be empty
      }
    }

    "empty integrator has empty stats" in {
      IO {
        val emptyIntegrator = QuadratureIntegrator.empty
        emptyIntegrator.logPdfs shouldBe Vector.empty
        emptyIntegrator.quadratures shouldBe Vector.empty
      }
    }
  }

  "QuadratureDomainTelemetry" - {

    "track acceptances and rejections" in {
      IO {
        val telemetry = QuadratureDomainTelemetry.init
        val updated   = telemetry.addAcceptance.addRejection.addRejection
        updated.acceptances shouldBe 1
        updated.rejections shouldBe 2
      }
    }

    "report not converged initially" in {
      IO {
        QuadratureDomainTelemetry.init.isConverged shouldBe false
      }
    }

    "track rejection streak" in {
      IO {
        val telemetry = QuadratureDomainTelemetry.init
        val afterRejects = (1 to 5).foldLeft(telemetry) { (t, _) =>
          t.addRejection
        }
        afterRejects.rejectionStreak shouldBe 5
      }
    }

    "reset rejection streak on acceptance" in {
      IO {
        val telemetry = QuadratureDomainTelemetry.init
        val afterRejects = (1 to 10).foldLeft(telemetry) { (t, _) =>
          t.addRejection
        }
        val afterAccept = afterRejects.addAcceptance
        afterAccept.rejectionStreak shouldBe 0
      }
    }

    "reset for rebuild" in {
      IO {
        val telemetry = QuadratureDomainTelemetry.init.addAcceptance.addAcceptance
        val reset     = telemetry.resetForRebuild
        reset.acceptances shouldBe 0
        reset.rejections shouldBe 0
        reset.currentScaleFactor shouldBe 1.0
      }
    }

    "maintain scale factor at 1.0 when rescaling is disabled" in {
      IO {
        val telemetry = QuadratureDomainTelemetry.init
        val updated = (1 to 100).foldLeft(telemetry) { (t, _) =>
          t.addRejection
        }
        updated.currentScaleFactor shouldBe 1.0
      }
    }
  }

  "SamplingSimulation" - {

    "deconstructed simulation reports not constructed" in {
      IO {
        SamplingSimulation.empty.isConstructed shouldBe false
      }
    }

    "deconstructed simulation throws on getSample" in {
      IO {
        assertThrows[IllegalStateException] {
          SamplingSimulation.empty.getSample
        }
      }
    }

    "constructed simulation reports constructed" in {
      IO {
        import thylacine.model.core.values.IndexedVectorCollection
        val logPdfResults = Vector(
          (-1.0, IndexedVectorCollection(Map("x" -> Vector(1.0)))),
          (-0.5, IndexedVectorCollection(Map("x" -> Vector(2.0)))),
          (-0.1, IndexedVectorCollection(Map("x" -> Vector(3.0))))
        )
        val abscissas = Vector(Vector(0.3, 0.6, 0.9))
        val sim       = SamplingSimulation.SamplingSimulationConstructed(logPdfResults, abscissas)
        sim.isConstructed shouldBe true
      }
    }

    "constructed simulation produces a sample" in {
      IO {
        import thylacine.model.core.values.IndexedVectorCollection
        val logPdfResults = Vector(
          (-1.0, IndexedVectorCollection(Map("x" -> Vector(1.0)))),
          (-0.5, IndexedVectorCollection(Map("x" -> Vector(2.0)))),
          (-0.1, IndexedVectorCollection(Map("x" -> Vector(3.0))))
        )
        val abscissas = Vector(Vector(0.3, 0.6, 0.9))
        val sim       = SamplingSimulation.SamplingSimulationConstructed(logPdfResults, abscissas)
        val sample    = sim.getSample
        sample.genericScalaRepresentation should contain key "x"
      }
    }
  }
}
