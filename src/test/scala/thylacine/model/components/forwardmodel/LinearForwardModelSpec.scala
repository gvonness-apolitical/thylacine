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
package thylacine.model.components.forwardmodel

import bengal.stm.STM
import thylacine.TestUtils.*
import thylacine.model.core.GenericIdentifier.*
import thylacine.model.core.values.{ IndexedMatrixCollection, IndexedVectorCollection, VectorContainer }

import cats.effect.IO
import cats.effect.testing.scalatest.AsyncIOSpec
import org.scalatest.freespec.AsyncFreeSpec
import org.scalatest.matchers.should.Matchers

class LinearForwardModelSpec extends AsyncFreeSpec with AsyncIOSpec with Matchers {
  "LinearForwardModel" - {

    "evaluate A*x correctly" in {
      STM
        .runtime[IO]
        .flatMap { implicit stm =>
          for {
            model <- LinearForwardModel.of[IO](
                       label          = "p",
                       values         = Vector(Vector(1.0, 3.0), Vector(2.0, 4.0)),
                       evalCacheDepth = None
                     )
            result <- model.evalAt(IndexedVectorCollection(Map("p" -> Vector(1.0, 2.0))))
          } yield maxVectorDiff(result.scalaVector, Vector(7.0, 10.0))
        }
        .asserting(_ shouldBe (0.0 +- 1e-10))
    }

    "evaluate A*x + b correctly with offset" in {
      STM
        .runtime[IO]
        .flatMap { implicit stm =>
          for {
            model <- LinearForwardModel.of[IO](
                       transform = IndexedMatrixCollection(
                         ModelParameterIdentifier("p"),
                         Vector(Vector(1.0, 0.0), Vector(0.0, 1.0))
                       ),
                       vectorOffset   = Some(VectorContainer(Vector(1.0, 2.0))),
                       evalCacheDepth = None
                     )
            result <- model.evalAt(IndexedVectorCollection(Map("p" -> Vector(3.0, 4.0))))
          } yield maxVectorDiff(result.scalaVector, Vector(4.0, 6.0))
        }
        .asserting(_ shouldBe (0.0 +- 1e-10))
    }

    "return the transform matrix as the Jacobian" in {
      STM
        .runtime[IO]
        .flatMap { implicit stm =>
          for {
            model <- LinearForwardModel.of[IO](
                       label          = "p",
                       values         = Vector(Vector(1.0, 3.0), Vector(2.0, 4.0)),
                       evalCacheDepth = None
                     )
            jac <- model.jacobianAt(IndexedVectorCollection(Map("p" -> Vector(99.0, -42.0))))
          } yield jac.genericScalaRepresentation
        }
        .asserting(_ shouldBe Map("p" -> Vector(Vector(1.0, 3.0), Vector(2.0, 4.0))))
    }

    "handle multi-block parameters" in {
      STM
        .runtime[IO]
        .flatMap { implicit stm =>
          for {
            model <- LinearForwardModel.of[IO](
                       transform = IndexedMatrixCollection(
                         Map(
                           ModelParameterIdentifier("a") -> thylacine.model.core.values
                             .MatrixContainer(Vector(Vector(1.0), Vector(0.0))),
                           ModelParameterIdentifier("b") -> thylacine.model.core.values
                             .MatrixContainer(Vector(Vector(0.0), Vector(1.0)))
                         )
                       ),
                       vectorOffset   = None,
                       evalCacheDepth = None
                     )
            result <- model.evalAt(IndexedVectorCollection(Map("a" -> Vector(3.0), "b" -> Vector(4.0))))
            jac    <- model.jacobianAt(IndexedVectorCollection(Map("a" -> Vector(3.0), "b" -> Vector(4.0))))
          } yield (result.scalaVector, jac.genericScalaRepresentation)
        }
        .asserting { case (eval, jac) =>
          maxVectorDiff(eval, Vector(3.0, 4.0)) shouldBe (0.0 +- 1e-10)
          jac("a") shouldBe Vector(Vector(1.0), Vector(0.0))
          jac("b") shouldBe Vector(Vector(0.0), Vector(1.0))
        }
    }
  }
}
