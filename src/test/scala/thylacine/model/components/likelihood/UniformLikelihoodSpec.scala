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
import thylacine.model.components.forwardmodel.LinearForwardModel
import thylacine.model.core.values.IndexedVectorCollection

import cats.effect.IO
import cats.effect.testing.scalatest.AsyncIOSpec
import org.scalatest.freespec.AsyncFreeSpec
import org.scalatest.matchers.should.Matchers

class UniformLikelihoodSpec extends AsyncFreeSpec with AsyncIOSpec with Matchers {

  // 1D identity forward model with uniform observation bounds [0, 5]
  private def uniformLikelihoodF(implicit stm: STM[IO]): IO[UniformLikelihood[IO, LinearForwardModel[IO]]] =
    for {
      fm <- LinearForwardModel.of[IO](
              label          = "x",
              values         = Vector(Vector(1.0)),
              evalCacheDepth = None
            )
    } yield UniformLikelihood[IO, LinearForwardModel[IO]](
      forwardModel = fm,
      upperBounds  = Vector(5.0),
      lowerBounds  = Vector(0.0)
    )

  "UniformLikelihood" - {

    "return finite logPdf when forward model output is inside bounds" in {
      STM
        .runtime[IO]
        .flatMap { implicit stm =>
          for {
            likelihood <- uniformLikelihoodF
            result     <- likelihood.logPdfAt(IndexedVectorCollection(Map("x" -> Vector(2.0))))
          } yield result
        }
        .asserting { r =>
          r shouldBe (-Math.log(5.0) +- 1e-8)
        }
    }

    "return negative infinity logPdf when forward model output is outside bounds" in {
      STM
        .runtime[IO]
        .flatMap { implicit stm =>
          for {
            likelihood <- uniformLikelihoodF
            result     <- likelihood.logPdfAt(IndexedVectorCollection(Map("x" -> Vector(6.0))))
          } yield result
        }
        .asserting(_ shouldBe Double.NegativeInfinity)
    }

    "return zero gradient inside bounds" in {
      STM
        .runtime[IO]
        .flatMap { implicit stm =>
          for {
            likelihood <- uniformLikelihoodF
            result     <- likelihood.logPdfGradientAt(IndexedVectorCollection(Map("x" -> Vector(2.0))))
          } yield result.genericScalaRepresentation
        }
        .asserting(_ shouldBe Map("x" -> Vector(0.0)))
    }
  }
}
