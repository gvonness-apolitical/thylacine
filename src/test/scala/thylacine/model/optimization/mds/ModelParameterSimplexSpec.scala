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
package thylacine.model.optimization.mds

import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.should.Matchers

class ModelParameterSimplexSpec extends AnyFreeSpec with Matchers {

  private val tol = 1e-10

  // A simple 2D triangle (equilateral is not required for non-regular simplexes)
  private val triangle = ModelParameterSimplex(
    vertices = Map(
      0 -> Vector(0.0, 0.0),
      1 -> Vector(4.0, 0.0),
      2 -> Vector(2.0, 3.0)
    ),
    validated = true
  )

  // A simple 1D simplex (line segment)
  private val lineSegment = ModelParameterSimplex(
    vertices = Map(
      0 -> Vector(1.0),
      1 -> Vector(5.0)
    ),
    validated = true
  )

  "ModelParameterSimplex" - {

    "reflectAbout" - {

      "reflect all vertices through the specified vertex" in {
        // Reflecting about vertex 0 at (0,0):
        //   v0 -> 2*(0,0) - (0,0) = (0,0)
        //   v1 -> 2*(0,0) - (4,0) = (-4,0)
        //   v2 -> 2*(0,0) - (2,3) = (-2,-3)
        val reflected = triangle.reflectAbout(0)
        reflected.vertices(0)(0) shouldBe (0.0 +- tol)
        reflected.vertices(0)(1) shouldBe (0.0 +- tol)
        reflected.vertices(1)(0) shouldBe (-4.0 +- tol)
        reflected.vertices(1)(1) shouldBe (0.0 +- tol)
        reflected.vertices(2)(0) shouldBe (-2.0 +- tol)
        reflected.vertices(2)(1) shouldBe (-3.0 +- tol)
      }

      "reflect about a non-origin vertex" in {
        // Reflecting about vertex 1 at (4,0):
        //   v0 -> 2*(4,0) - (0,0) = (8,0)
        //   v1 -> 2*(4,0) - (4,0) = (4,0)
        //   v2 -> 2*(4,0) - (2,3) = (6,-3)
        val reflected = triangle.reflectAbout(1)
        reflected.vertices(0)(0) shouldBe (8.0 +- tol)
        reflected.vertices(0)(1) shouldBe (0.0 +- tol)
        reflected.vertices(1)(0) shouldBe (4.0 +- tol)
        reflected.vertices(1)(1) shouldBe (0.0 +- tol)
        reflected.vertices(2)(0) shouldBe (6.0 +- tol)
        reflected.vertices(2)(1) shouldBe (-3.0 +- tol)
      }

      "work for a 1D simplex" in {
        // Reflecting about vertex 0 at (1):
        //   v0 -> 2*1 - 1 = 1
        //   v1 -> 2*1 - 5 = -3
        val reflected = lineSegment.reflectAbout(0)
        reflected.vertices(0)(0) shouldBe (1.0 +- tol)
        reflected.vertices(1)(0) shouldBe (-3.0 +- tol)
      }
    }

    "expandAbout" - {

      "expand vertices away from the specified vertex" in {
        val mu = 2.0
        // expandAbout(0, 2.0): baseVertex = (0,0) * (1-2) = (0,0)
        //   v0 -> (0,0) + 2*(0,0) = (0,0)
        //   v1 -> (0,0) + 2*(4,0) = (8,0)
        //   v2 -> (0,0) + 2*(2,3) = (4,6)
        val expanded = triangle.expandAbout(0, mu)
        expanded.vertices(0)(0) shouldBe (0.0 +- tol)
        expanded.vertices(0)(1) shouldBe (0.0 +- tol)
        expanded.vertices(1)(0) shouldBe (8.0 +- tol)
        expanded.vertices(1)(1) shouldBe (0.0 +- tol)
        expanded.vertices(2)(0) shouldBe (4.0 +- tol)
        expanded.vertices(2)(1) shouldBe (6.0 +- tol)
      }

      "contract vertices toward the specified vertex when mu < 1" in {
        val mu = 0.5
        // expandAbout(1, 0.5): baseVertex = (4,0) * (1-0.5) = (2,0)
        //   v0 -> (2,0) + 0.5*(0,0) = (2,0)
        //   v1 -> (2,0) + 0.5*(4,0) = (4,0)
        //   v2 -> (2,0) + 0.5*(2,3) = (3,1.5)
        val expanded = triangle.expandAbout(1, mu)
        expanded.vertices(0)(0) shouldBe (2.0 +- tol)
        expanded.vertices(0)(1) shouldBe (0.0 +- tol)
        expanded.vertices(1)(0) shouldBe (4.0 +- tol)
        expanded.vertices(1)(1) shouldBe (0.0 +- tol)
        expanded.vertices(2)(0) shouldBe (3.0 +- tol)
        expanded.vertices(2)(1) shouldBe (1.5 +- tol)
      }

      "leave the simplex unchanged when mu is 1" in {
        val expanded = triangle.expandAbout(0, 1.0)
        for ((idx, vertex) <- triangle.vertices) {
          expanded.vertices(idx).zip(vertex).foreach { case (actual, expected) =>
            actual shouldBe (expected +- tol)
          }
        }
      }
    }

    "contractAbout" - {

      "contract vertices toward the specified vertex" in {
        val theta = 0.5
        // contractAbout(0, 0.5): baseVertex = (0,0) * 1.5 = (0,0)
        //   v0 -> (0,0) - 0.5*(0,0) = (0,0)
        //   v1 -> (0,0) - 0.5*(4,0) = (-2,0)
        //   v2 -> (0,0) - 0.5*(2,3) = (-1,-1.5)
        val contracted = triangle.contractAbout(0, theta)
        contracted.vertices(0)(0) shouldBe (0.0 +- tol)
        contracted.vertices(0)(1) shouldBe (0.0 +- tol)
        contracted.vertices(1)(0) shouldBe (-2.0 +- tol)
        contracted.vertices(1)(1) shouldBe (0.0 +- tol)
        contracted.vertices(2)(0) shouldBe (-1.0 +- tol)
        contracted.vertices(2)(1) shouldBe (-1.5 +- tol)
      }

      "contract about a non-origin vertex" in {
        val theta = 0.5
        // contractAbout(2, 0.5): baseVertex = (2,3) * 1.5 = (3,4.5)
        //   v0 -> (3,4.5) - 0.5*(0,0) = (3,4.5)
        //   v1 -> (3,4.5) - 0.5*(4,0) = (1,4.5)
        //   v2 -> (3,4.5) - 0.5*(2,3) = (2,3)
        val contracted = triangle.contractAbout(2, theta)
        contracted.vertices(0)(0) shouldBe (3.0 +- tol)
        contracted.vertices(0)(1) shouldBe (4.5 +- tol)
        contracted.vertices(1)(0) shouldBe (1.0 +- tol)
        contracted.vertices(1)(1) shouldBe (4.5 +- tol)
        contracted.vertices(2)(0) shouldBe (2.0 +- tol)
        contracted.vertices(2)(1) shouldBe (3.0 +- tol)
      }

      "leave the pivot vertex unchanged" in {
        val theta      = 0.5
        val contracted = triangle.contractAbout(1, theta)
        contracted.vertices(1)(0) shouldBe (4.0 +- tol)
        contracted.vertices(1)(1) shouldBe (0.0 +- tol)
      }
    }

    "maxAdjacentEdgeLength" - {

      "compute the maximum distance from a vertex to all others" in {
        // From vertex 0 at (0,0):
        //   to vertex 1 at (4,0): distance = 4.0
        //   to vertex 2 at (2,3): distance = sqrt(4+9) = sqrt(13) ~ 3.606
        // Max = 4.0
        triangle.maxAdjacentEdgeLength(0) shouldBe (4.0 +- tol)
      }

      "compute correctly for different pivot vertices" in {
        // From vertex 2 at (2,3):
        //   to vertex 0 at (0,0): distance = sqrt(4+9) = sqrt(13) ~ 3.606
        //   to vertex 1 at (4,0): distance = sqrt(4+9) = sqrt(13) ~ 3.606
        // Max = sqrt(13)
        triangle.maxAdjacentEdgeLength(2) shouldBe (Math.sqrt(13.0) +- tol)
      }

      "compute correctly for a 1D simplex" in {
        // From vertex 0 at (1): to vertex 1 at (5): distance = 4.0
        lineSegment.maxAdjacentEdgeLength(0) shouldBe (4.0 +- tol)
      }
    }

    "regularity validation" - {

      "accept a regular simplex with isRegular flag" in {
        // Equilateral triangle with side length 2
        val side   = 2.0
        val height = side * Math.sqrt(3.0) / 2.0
        val regular = ModelParameterSimplex(
          vertices = Map(
            0 -> Vector(0.0, 0.0),
            1 -> Vector(side, 0.0),
            2 -> Vector(side / 2.0, height)
          ),
          isRegular = true
        )
        regular.vertices should have size 3
      }

      "reject a non-regular simplex with isRegular flag" in {
        an[IllegalArgumentException] should be thrownBy {
          ModelParameterSimplex(
            vertices = Map(
              0 -> Vector(0.0, 0.0),
              1 -> Vector(4.0, 0.0),
              2 -> Vector(2.0, 3.0)
            ),
            isRegular = true
          )
        }
      }

      "accept a non-regular simplex without isRegular flag" in {
        val simplex = ModelParameterSimplex(
          vertices = Map(
            0 -> Vector(0.0, 0.0),
            1 -> Vector(4.0, 0.0),
            2 -> Vector(2.0, 3.0)
          )
        )
        simplex.vertices should have size 3
      }
    }

    "maxAdjacentEdgeLength for regular simplex" - {

      "use the cached edge length" in {
        val side   = 2.0
        val height = side * Math.sqrt(3.0) / 2.0
        val regular = ModelParameterSimplex(
          vertices = Map(
            0 -> Vector(0.0, 0.0),
            1 -> Vector(side, 0.0),
            2 -> Vector(side / 2.0, height)
          ),
          isRegular = true
        )
        // For a regular simplex, maxAdjacentEdgeLength should return the
        // cached randomAdjacentEdgeLength (distance between first two vertices)
        regular.maxAdjacentEdgeLength(0) shouldBe (side +- 1e-3)
        regular.maxAdjacentEdgeLength(1) shouldBe (side +- 1e-3)
        regular.maxAdjacentEdgeLength(2) shouldBe (side +- 1e-3)
      }
    }
  }
}
