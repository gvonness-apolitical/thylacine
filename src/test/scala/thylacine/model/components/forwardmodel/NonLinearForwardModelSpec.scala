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
import thylacine.model.core.computation.DifferencingScheme
import thylacine.model.core.values.IndexedVectorCollection

import cats.effect.IO
import cats.effect.testing.scalatest.AsyncIOSpec
import org.scalatest.freespec.AsyncFreeSpec
import org.scalatest.matchers.should.Matchers

class NonLinearForwardModelSpec extends AsyncFreeSpec with AsyncIOSpec with Matchers {

  // f(x1, x2) = [x1^2, x1*x2]
  private val nonlinearEval: Map[String, Vector[Double]] => Vector[Double] = { input =>
    val x = input("x")
    Vector(x(0) * x(0), x(0) * x(1))
  }

  // J(x1, x2) = [[2*x1, 0], [x2, x1]]
  private val analyticJacobian: Map[String, Vector[Double]] => Map[String, Vector[Vector[Double]]] = { input =>
    val x = input("x")
    Map("x" -> Vector(Vector(2 * x(0), 0.0), Vector(x(1), x(0))))
  }

  "NonLinearForwardModel" - {

    "evaluate a nonlinear function correctly" in {
      STM
        .runtime[IO]
        .flatMap { implicit stm =>
          for {
            model <- NonLinearForwardModel.of[IO](
                       evaluation         = nonlinearEval,
                       differential       = 1e-7,
                       domainDimensions   = Map("x" -> 2),
                       rangeDimension     = 2,
                       evalCacheDepth     = None,
                       jacobianCacheDepth = None
                     )
            result <- model.evalAt(IndexedVectorCollection(Map("x" -> Vector(2.0, 3.0))))
          } yield maxVectorDiff(result.scalaVector, Vector(4.0, 6.0))
        }
        .asserting(_ shouldBe (0.0 +- 1e-10))
    }

    "compute finite-difference Jacobian close to analytical" in {
      STM
        .runtime[IO]
        .flatMap { implicit stm =>
          for {
            model <- NonLinearForwardModel.of[IO](
                       evaluation         = nonlinearEval,
                       differential       = 1e-7,
                       domainDimensions   = Map("x" -> 2),
                       rangeDimension     = 2,
                       evalCacheDepth     = None,
                       jacobianCacheDepth = None
                     )
            jac <- model.jacobianAt(IndexedVectorCollection(Map("x" -> Vector(2.0, 3.0))))
          } yield {
            val jacMatrix = jac.genericScalaRepresentation("x")
            // Expected: [[4, 0], [3, 2]]
            val expected = Vector(Vector(4.0, 0.0), Vector(3.0, 2.0))
            maxMatrixDiff(jacMatrix, expected)
          }
        }
        .asserting(_ shouldBe (0.0 +- 1e-3))
    }

    "compute central-difference Jacobian with tighter accuracy" in {
      STM
        .runtime[IO]
        .flatMap { implicit stm =>
          for {
            modelForward <- NonLinearForwardModel.of[IO](
                              evaluation         = nonlinearEval,
                              differential       = 1e-5,
                              domainDimensions   = Map("x" -> 2),
                              rangeDimension     = 2,
                              evalCacheDepth     = None,
                              jacobianCacheDepth = None,
                              differencingScheme = DifferencingScheme.Forward
                            )
            modelCentral <- NonLinearForwardModel.of[IO](
                              evaluation         = nonlinearEval,
                              differential       = 1e-5,
                              domainDimensions   = Map("x" -> 2),
                              rangeDimension     = 2,
                              evalCacheDepth     = None,
                              jacobianCacheDepth = None,
                              differencingScheme = DifferencingScheme.Central
                            )
            jacForward <- modelForward.jacobianAt(IndexedVectorCollection(Map("x" -> Vector(2.0, 3.0))))
            jacCentral <- modelCentral.jacobianAt(IndexedVectorCollection(Map("x" -> Vector(2.0, 3.0))))
          } yield {
            val expected     = Vector(Vector(4.0, 0.0), Vector(3.0, 2.0))
            val forwardError = maxMatrixDiff(jacForward.genericScalaRepresentation("x"), expected)
            val centralError = maxMatrixDiff(jacCentral.genericScalaRepresentation("x"), expected)
            (forwardError, centralError)
          }
        }
        .asserting { case (fe, ce) =>
          // Central differences should be more accurate than forward at the same step size
          ce should be < fe
          // Central should achieve ~1e-10 accuracy with h=1e-5 (O(h^2) error)
          ce shouldBe (0.0 +- 1e-8)
        }
    }

    "use user-provided analytical Jacobian when given" in {
      STM
        .runtime[IO]
        .flatMap { implicit stm =>
          for {
            model <- NonLinearForwardModel.of[IO](
                       evaluation         = nonlinearEval,
                       jacobian           = analyticJacobian,
                       domainDimensions   = Map("x" -> 2),
                       rangeDimension     = 2,
                       evalCacheDepth     = None,
                       jacobianCacheDepth = None
                     )
            jac <- model.jacobianAt(IndexedVectorCollection(Map("x" -> Vector(2.0, 3.0))))
          } yield jac.genericScalaRepresentation("x")
        }
        .asserting(_ shouldBe Vector(Vector(4.0, 0.0), Vector(3.0, 2.0)))
    }
  }
}
