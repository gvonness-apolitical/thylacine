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
import thylacine.model.components.ComponentFixture.fooNonAnalyticLikelihoodF
import thylacine.model.core.values.IndexedVectorCollection

import cats.effect.IO
import cats.effect.testing.scalatest.AsyncIOSpec
import org.scalatest.freespec.AsyncFreeSpec
import org.scalatest.matchers.should.Matchers

class GaussianLikelihoodSpec extends AsyncFreeSpec with AsyncIOSpec with Matchers {
  "GaussianLikelihood" - {

    "generate the a zero gradient at the likelihood maximum" in {
      STM
        .runtime[IO]
        .flatMap { implicit stm =>
          for {
            likelihood <- fooNonAnalyticLikelihoodF
            result     <- likelihood.logPdfGradientAt(IndexedVectorCollection(Map("foo" -> Vector(1d, 2d))))
          } yield result.genericScalaRepresentation
        }
        .asserting(_ shouldBe Map("foo" -> Vector(0d, 0d)))
    }

    "generate the correct gradient of the logPdf for a given point" in {
      STM
        .runtime[IO]
        .flatMap { implicit stm =>
          for {
            likelihood <- fooNonAnalyticLikelihoodF
            result     <- likelihood.logPdfGradientAt(IndexedVectorCollection(Map("foo" -> Vector(3d, 2d))))
          } yield result.genericScalaRepresentation
        }
        .asserting(result => maxIndexVectorDiff(result, Map("foo" -> Vector(-4e5, -88e4))) shouldBe (0d +- 1e-4))
    }

    // End-to-end: chain-rule gradient vs full finite-difference gradient of likelihood logPdf
    "match gradient against finite differences" in {
      val testPoint = Map("foo" -> Vector(1.5, 2.5))
      val eps       = 1e-7
      STM
        .runtime[IO]
        .flatMap { implicit stm =>
          for {
            likelihood <- fooNonAnalyticLikelihoodF
            grad       <- likelihood.logPdfGradientAt(IndexedVectorCollection(testPoint))
            logPdf0    <- likelihood.logPdfAt(IndexedVectorCollection(testPoint))
            logPdfN0 <- likelihood.logPdfAt(
                          IndexedVectorCollection(Map("foo" -> Vector(1.5 + eps, 2.5)))
                        )
            logPdfN1 <- likelihood.logPdfAt(
                          IndexedVectorCollection(Map("foo" -> Vector(1.5, 2.5 + eps)))
                        )
          } yield {
            val fdGrad = Map("foo" -> Vector((logPdfN0 - logPdf0) / eps, (logPdfN1 - logPdf0) / eps))
            maxIndexVectorDiff(grad.genericScalaRepresentation, fdGrad)
          }
        }
        .asserting(_ shouldBe (0.0 +- 0.1))
    }
  }
}
