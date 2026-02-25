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
import thylacine.TestUtils.*
import thylacine.model.components.ComponentFixture.*
import thylacine.model.components.likelihood.GaussianLinearLikelihood
import thylacine.model.components.prior.GaussianPrior

import cats.effect.IO
import cats.effect.testing.scalatest.AsyncIOSpec
import org.scalatest.freespec.AsyncFreeSpec
import org.scalatest.matchers.should.Matchers

class GaussianAnalyticPosteriorSpec extends AsyncFreeSpec with AsyncIOSpec with Matchers {
  "GaussianAnalyticPosterior" - {

    // Unfortunately, the analytic solution has a number of matrix inversions
    // that chip away at the numerical accuracy of the result
    "generate the correct mean for the inference" in {
      STM
        .runtime[IO]
        .flatMap { implicit stm =>
          for {
            posterior <- analyticPosteriorF
          } yield maxIndexVectorDiff(posterior.mean, Map("foo" -> Vector(1, 2), "bar" -> Vector(5)))
        }
        .asserting(_ shouldBe (0d +- .1))
    }

    // Not a great test, but the prior and likelihood align
    // on the inference for the parameters (i.e. all uncertainties should
    // be very small)
    "generate the correct covariance" in {
      STM
        .runtime[IO]
        .flatMap { implicit stm =>
          for {
            posterior <- analyticPosteriorF
          } yield maxVectorDiff(posterior.covarianceStridedVector, Vector.fill(9)(0d))
        }
        .asserting(_ shouldBe (0.0 +- .01))
    }

    // 1D Bayesian update: N(0,1) prior + y=x with y_obs=2, σ_obs²=1
    // σ_post² = 1/(1+1) = 0.5, μ_post = 0.5*(0+2) = 1.0
    "compute correct 1D Bayesian update" in {
      STM
        .runtime[IO]
        .flatMap { implicit stm =>
          val prior = GaussianPrior.fromStandardDeviations[IO](
            label              = "x",
            values             = Vector(0.0),
            standardDeviations = Vector(1.0) // variance = 1
          )
          for {
            likelihood <- GaussianLinearLikelihood.of[IO](
                            coefficients       = Vector(Vector(1.0)),
                            measurements       = Vector(2.0),
                            standardDeviations = Vector(1.0), // variance = 1
                            priorLabel         = "x",
                            evalCacheDepth     = None
                          )
            posterior = GaussianAnalyticPosterior[IO](
                          priors      = Set(prior),
                          likelihoods = Set(likelihood)
                        )
            _ <- posterior.init
          } yield (posterior.mean, posterior.covarianceStridedVector)
        }
        .asserting { case (mean, cov) =>
          maxIndexVectorDiff(mean, Map("x" -> Vector(1.0))) shouldBe (0.0 +- 1e-6)
          maxVectorDiff(cov, Vector(0.5)) shouldBe (0.0 +- 1e-6)
        }
    }
  }
}
