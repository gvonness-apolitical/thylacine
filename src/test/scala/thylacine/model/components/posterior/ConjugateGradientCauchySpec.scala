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
import thylacine.TestUtils.maxIndexVectorDiff
import thylacine.config.ConjugateGradientConfig
import thylacine.model.components.likelihood.{ GaussianLinearLikelihood, Likelihood }
import thylacine.model.components.prior.{ CauchyPrior, Prior }

import cats.effect.IO
import cats.effect.testing.scalatest.AsyncIOSpec
import org.scalatest.freespec.AsyncFreeSpec
import org.scalatest.matchers.should.Matchers

class ConjugateGradientCauchySpec extends AsyncFreeSpec with AsyncIOSpec with Matchers {

  private val cgConfig = ConjugateGradientConfig(
    convergenceThreshold      = 1e-15,
    goldenSectionTolerance    = 1e-10,
    lineProbeExpansionFactor  = 2.0,
    minimumNumberOfIterations = 100
  )

  // CauchyPrior: center=0, CI=2 (variance=1)
  // GaussianLinearLikelihood: identity, measurement=3, uncertainty=0.01 (variance=0.000025)
  // The tight likelihood should dominate → optimizer converges near x≈3
  "ConjugateGradientOptimisedPosterior with Cauchy prior" - {

    "converge to the correct value after gradient fix" in {
      STM
        .runtime[IO]
        .flatMap { implicit stm =>
          val cauchyPrior = CauchyPrior[IO](
            label           = "x",
            values          = Vector(0.0),
            scaleParameters = Vector(1.0)
          )
          for {
            likelihood <- GaussianLinearLikelihood.of[IO](
                            coefficients       = Vector(Vector(1.0)),
                            measurements       = Vector(3.0),
                            standardDeviations = Vector(0.005),
                            priorLabel         = "x",
                            evalCacheDepth     = None
                          )
            unnormalisedPosterior = UnnormalisedPosterior[IO](
                                      priors      = Set[Prior[IO, ?]](cauchyPrior),
                                      likelihoods = Set[Likelihood[IO, ?, ?]](likelihood)
                                    )
            optimizer = ConjugateGradientOptimisedPosterior[IO](
                          conjugateGradientConfig = cgConfig,
                          posterior               = unnormalisedPosterior,
                          iterationUpdateCallback = _ => IO.unit,
                          isConvergedCallback     = _ => IO.unit
                        )
            result <- optimizer.findMaximumLogPdf(Map("x" -> Vector(0.0)))
          } yield maxIndexVectorDiff(result._2, Map("x" -> Vector(3.0)))
        }
        .asserting(_ shouldBe (0.0 +- 0.1))
    }
  }
}
