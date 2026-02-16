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
import thylacine.model.components.ComponentFixture.fooLikelihoodF
import thylacine.model.core.values.IndexedVectorCollection

import cats.effect.IO
import cats.effect.testing.scalatest.AsyncIOSpec
import org.scalatest.freespec.AsyncFreeSpec
import org.scalatest.matchers.should.Matchers

class GaussianLinearLikelihoodSpec extends AsyncFreeSpec with AsyncIOSpec with Matchers {
  "GaussianLinearLikelihood" - {

    "generate the a zero gradient at the likelihood maximum" in {
      STM
        .runtime[IO]
        .flatMap { implicit stm =>
          for {
            likelihood <- fooLikelihoodF
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
            likelihood <- fooLikelihoodF
            result     <- likelihood.logPdfGradientAt(IndexedVectorCollection(Map("foo" -> Vector(3d, 2d))))
          } yield result.genericScalaRepresentation
        }
        .asserting(_ shouldBe Map("foo" -> Vector(-4e5, -88e4)))
    }

    // J^T * Σ_obs^(-1) * (data - J*θ) at θ=(0,0)
    // J=[[1,3],[2,4]], Σ_obs^(-1)=diag(40000,40000), data=(7,10)
    // = [[1,2],[3,4]] * (280000, 400000) = (1080000, 2440000)
    "generate the correct gradient at a second non-trivial point" in {
      STM
        .runtime[IO]
        .flatMap { implicit stm =>
          for {
            likelihood <- fooLikelihoodF
            result     <- likelihood.logPdfGradientAt(IndexedVectorCollection(Map("foo" -> Vector(0d, 0d))))
          } yield result.genericScalaRepresentation
        }
        .asserting(_ shouldBe Map("foo" -> Vector(108e4, 244e4)))
    }
  }
}
