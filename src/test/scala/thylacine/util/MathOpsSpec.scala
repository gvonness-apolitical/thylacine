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
package thylacine.util

import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.should.Matchers

class MathOpsSpec extends AnyFreeSpec with Matchers {

  private val tol = 1e-10

  "vectorCdfStaircase" - {

    "produce equal-width bins for uniform input" in {
      val result = MathOps.vectorCdfStaircase(Vector(1.0, 1.0, 1.0))
      result should have size 3
      result.foreach { case (lo, hi) =>
        (hi - lo) shouldBe ((1.0 / 3.0) +- tol)
      }
    }

    "produce proportional bins for non-uniform input" in {
      val result = MathOps.vectorCdfStaircase(Vector(1.0, 2.0, 3.0))
      result should have size 3
      result(0)._1 shouldBe (0.0 +- tol)
      result(0)._2 shouldBe ((1.0 / 6.0) +- tol)
      result(1)._1 shouldBe ((1.0 / 6.0) +- tol)
      result(1)._2 shouldBe ((3.0 / 6.0) +- tol)
      result(2)._1 shouldBe ((3.0 / 6.0) +- tol)
      result(2)._2 shouldBe (1.0 +- tol)
    }

    "produce a single bin spanning [0, 1] for single-element input" in {
      val result = MathOps.vectorCdfStaircase(Vector(5.0))
      result shouldBe Vector((0.0, 1.0))
    }

    "produce contiguous bins covering [0, 1]" in {
      val result = MathOps.vectorCdfStaircase(Vector(3.0, 1.0, 4.0, 1.0, 5.0))
      result.head._1 shouldBe (0.0 +- tol)
      result.last._2 shouldBe (1.0 +- tol)
      result.sliding(2).foreach { pair =>
        val bins = pair.toVector
        bins(0)._2 shouldBe (bins(1)._1 +- tol)
      }
    }

    "assign zero-width bins for zero-valued entries" in {
      val result = MathOps.vectorCdfStaircase(Vector(1.0, 0.0, 1.0))
      result should have size 3
      result(1)._1 shouldBe (result(1)._2 +- tol)
    }
  }
}
