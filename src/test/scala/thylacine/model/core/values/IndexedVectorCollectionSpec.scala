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
package thylacine.model.core.values

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should

class IndexedVectorCollectionSpec extends AnyFlatSpec with should.Matchers {

  private val tol = 1e-10

  "IndexedVectorCollection" should "construct from labeled maps" in {
    val ivc = IndexedVectorCollection(Map("foo" -> Vector(1.0, 2.0), "bar" -> Vector(3.0)))
    ivc.genericScalaRepresentation shouldBe Map("foo" -> Vector(1.0, 2.0), "bar" -> Vector(3.0))
  }

  it should "compute total dimension" in {
    val ivc = IndexedVectorCollection(Map("foo" -> Vector(1.0, 2.0), "bar" -> Vector(3.0)))
    ivc.totalDimension shouldBe 3
  }

  it should "compute magnitude" in {
    val ivc = IndexedVectorCollection(Map("a" -> Vector(3.0, 4.0)))
    ivc.magnitude shouldBe (5.0 +- tol)
  }

  it should "merge disjoint collections" in {
    val ivc1   = IndexedVectorCollection(Map("foo" -> Vector(1.0, 2.0)))
    val ivc2   = IndexedVectorCollection(Map("bar" -> Vector(3.0)))
    val merged = ivc1.rawMergeWith(ivc2)
    merged.genericScalaRepresentation shouldBe Map("foo" -> Vector(1.0, 2.0), "bar" -> Vector(3.0))
  }

  it should "throw on merge with overlapping keys" in {
    val ivc1 = IndexedVectorCollection(Map("foo" -> Vector(1.0)))
    val ivc2 = IndexedVectorCollection(Map("foo" -> Vector(2.0)))
    an[IllegalArgumentException] should be thrownBy ivc1.rawMergeWith(ivc2)
  }

  it should "sum two collections" in {
    val ivc1   = IndexedVectorCollection(Map("foo" -> Vector(1.0, 2.0)))
    val ivc2   = IndexedVectorCollection(Map("foo" -> Vector(3.0, 4.0)))
    val result = ivc1.rawSumWith(ivc2)
    result.genericScalaRepresentation shouldBe Map("foo" -> Vector(4.0, 6.0))
  }

  it should "sum with disjoint keys" in {
    val ivc1   = IndexedVectorCollection(Map("foo" -> Vector(1.0)))
    val ivc2   = IndexedVectorCollection(Map("bar" -> Vector(2.0)))
    val result = ivc1.rawSumWith(ivc2)
    result.genericScalaRepresentation shouldBe Map("foo" -> Vector(1.0), "bar" -> Vector(2.0))
  }

  it should "scalar multiply" in {
    val ivc    = IndexedVectorCollection(Map("foo" -> Vector(1.0, 2.0), "bar" -> Vector(3.0)))
    val result = ivc.rawScalarMultiplyWith(2.0)
    result.genericScalaRepresentation shouldBe Map("foo" -> Vector(2.0, 4.0), "bar" -> Vector(6.0))
  }

  it should "subtract collections" in {
    val ivc1   = IndexedVectorCollection(Map("foo" -> Vector(5.0, 6.0)))
    val ivc2   = IndexedVectorCollection(Map("foo" -> Vector(1.0, 2.0)))
    val result = ivc1.rawSubtract(ivc2)
    result.genericScalaRepresentation shouldBe Map("foo" -> Vector(4.0, 4.0))
  }

  it should "compute distance between collections" in {
    val ivc1 = IndexedVectorCollection(Map("a" -> Vector(0.0, 0.0)))
    val ivc2 = IndexedVectorCollection(Map("a" -> Vector(3.0, 4.0)))
    ivc1.distanceTo(ivc2) shouldBe (5.0 +- tol)
  }

  it should "handle empty collection" in {
    val emptyCol = IndexedVectorCollection.empty
    emptyCol.index shouldBe Map.empty
    emptyCol.totalDimension shouldBe 0
  }
}
