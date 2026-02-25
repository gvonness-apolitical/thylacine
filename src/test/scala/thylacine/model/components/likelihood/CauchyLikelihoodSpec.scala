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
package thylacine.model.components.likelihood

import bengal.stm.STM
import thylacine.TestUtils.maxIndexVectorDiff
import thylacine.model.core.values.IndexedVectorCollection

import cats.effect.IO
import cats.effect.testing.scalatest.AsyncIOSpec
import org.scalatest.freespec.AsyncFreeSpec
import org.scalatest.matchers.should.Matchers

class CauchyLikelihoodSpec extends AsyncFreeSpec with AsyncIOSpec with Matchers {

  // 1D identity forward model, measurement=3, scale=1 (variance=1)
  private def cauchyLikelihoodF(implicit stm: STM[IO]): IO[CauchyLikelihood[IO]] =
    CauchyLikelihood.of[IO](
      coefficients    = Vector(Vector(1.0)),
      measurements    = Vector(3.0),
      scaleParameters = Vector(1.0),
      priorLabel      = "x",
      evalCacheDepth  = None
    )

  "CauchyLikelihood" - {

    "generate zero gradient at the likelihood maximum" in {
      STM
        .runtime[IO]
        .flatMap { implicit stm =>
          for {
            likelihood <- cauchyLikelihoodF
            result     <- likelihood.logPdfGradientAt(IndexedVectorCollection(Map("x" -> Vector(3.0))))
          } yield result.genericScalaRepresentation
        }
        .asserting(_ shouldBe Map("x" -> Vector(0.0)))
    }

    "generate correct gradient at a non-trivial point" in {
      // At θ=4: f(θ)=4, diff=4-3=1, Q=1/1=1
      // measGrad = -(1+1)/(1+1) * 1 = -1.0
      // gradient = J^T * measGrad = 1 * (-1.0) = -1.0
      STM
        .runtime[IO]
        .flatMap { implicit stm =>
          for {
            likelihood <- cauchyLikelihoodF
            result     <- likelihood.logPdfGradientAt(IndexedVectorCollection(Map("x" -> Vector(4.0))))
          } yield result.genericScalaRepresentation
        }
        .asserting(r => maxIndexVectorDiff(r, Map("x" -> Vector(-1.0))) shouldBe (0.0 +- 1e-8))
    }

    "match gradient against finite differences" in {
      val testPoint = Map("x" -> Vector(4.5))
      val eps       = 1e-7
      STM
        .runtime[IO]
        .flatMap { implicit stm =>
          for {
            likelihood <- cauchyLikelihoodF
            grad       <- likelihood.logPdfGradientAt(IndexedVectorCollection(testPoint))
            logPdf0    <- likelihood.logPdfAt(IndexedVectorCollection(testPoint))
            logPdfN    <- likelihood.logPdfAt(IndexedVectorCollection(Map("x" -> Vector(4.5 + eps))))
          } yield {
            val fdGrad = Map("x" -> Vector((logPdfN - logPdf0) / eps))
            maxIndexVectorDiff(grad.genericScalaRepresentation, fdGrad)
          }
        }
        .asserting(_ shouldBe (0.0 +- 1e-5))
    }
  }
}
