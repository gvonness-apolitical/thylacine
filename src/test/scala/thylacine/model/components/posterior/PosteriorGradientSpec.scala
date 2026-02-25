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
import thylacine.model.components.likelihood.{ GaussianLinearLikelihood, Likelihood }
import thylacine.model.components.prior.{ GaussianPrior, Prior }
import thylacine.model.core.values.IndexedVectorCollection

import cats.effect.IO
import cats.effect.testing.scalatest.AsyncIOSpec
import org.scalatest.freespec.AsyncFreeSpec
import org.scalatest.matchers.should.Matchers

class PosteriorGradientSpec extends AsyncFreeSpec with AsyncIOSpec with Matchers {

  // Prior "x": mean=0, σ=1 → σ²=1
  // Prior "y": mean=1, σ=2 → σ²=4
  // Likelihood for x: identity, data=2, σ=1 → σ²_obs=1
  // Likelihood for y: identity, data=3, σ=2 → σ²_obs=4
  private val priorX = GaussianPrior.fromStandardDeviations[IO](
    label              = "x",
    values             = Vector(0.0),
    standardDeviations = Vector(1.0)
  )

  private val priorY = GaussianPrior.fromStandardDeviations[IO](
    label              = "y",
    values             = Vector(1.0),
    standardDeviations = Vector(2.0)
  )

  private def posteriorF(implicit stm: STM[IO]): IO[UnnormalisedPosterior[IO]] =
    for {
      likelihoodX <- GaussianLinearLikelihood.of[IO](
                       coefficients       = Vector(Vector(1.0)),
                       measurements       = Vector(2.0),
                       standardDeviations = Vector(1.0),
                       priorLabel         = "x",
                       evalCacheDepth     = None
                     )
      likelihoodY <- GaussianLinearLikelihood.of[IO](
                       coefficients       = Vector(Vector(1.0)),
                       measurements       = Vector(3.0),
                       standardDeviations = Vector(2.0),
                       priorLabel         = "y",
                       evalCacheDepth     = None
                     )
    } yield UnnormalisedPosterior[IO](
      priors      = Set[Prior[IO, ?]](priorX, priorY),
      likelihoods = Set[Likelihood[IO, ?, ?]](likelihoodX, likelihoodY)
    )

  "UnnormalisedPosterior gradient" - {

    // At θ=(x=0, y=0):
    // Prior x gradient: 1*(0-0) = 0
    // Prior y gradient: (1/4)*(1-0) = 0.25
    // Likelihood x: 1*1*(2-0) = 2
    // Likelihood y: 1*(1/4)*(3-0) = 0.75
    // Total: x → 0+2=2, y → 0.25+0.75=1.0
    "equal sum of prior and likelihood gradients" in {
      STM
        .runtime[IO]
        .flatMap { implicit stm =>
          for {
            posterior <- posteriorF
            grad <- posterior.logPdfGradientAt(IndexedVectorCollection(Map("x" -> Vector(0.0), "y" -> Vector(0.0))))
          } yield grad.genericScalaRepresentation
        }
        .asserting { g =>
          maxIndexVectorDiff(g, Map("x" -> Vector(2.0), "y" -> Vector(1.0))) shouldBe (0.0 +- 1e-8)
        }
    }

    "match gradient via finite differences" in {
      val testPoint = Map("x" -> Vector(0.5), "y" -> Vector(1.5))
      val eps       = 1e-7
      STM
        .runtime[IO]
        .flatMap { implicit stm =>
          for {
            posterior <- posteriorF
            grad      <- posterior.logPdfGradientAt(IndexedVectorCollection(testPoint))
            logPdf0   <- posterior.logPdfAt(IndexedVectorCollection(testPoint))
            logPdfNx <- posterior.logPdfAt(
                          IndexedVectorCollection(Map("x" -> Vector(0.5 + eps), "y" -> Vector(1.5)))
                        )
            logPdfNy <- posterior.logPdfAt(
                          IndexedVectorCollection(Map("x" -> Vector(0.5), "y" -> Vector(1.5 + eps)))
                        )
          } yield {
            val fdGrad = Map("x" -> Vector((logPdfNx - logPdf0) / eps), "y" -> Vector((logPdfNy - logPdf0) / eps))
            maxIndexVectorDiff(grad.genericScalaRepresentation, fdGrad)
          }
        }
        .asserting(_ shouldBe (0.0 +- 1e-5))
    }
  }
}
