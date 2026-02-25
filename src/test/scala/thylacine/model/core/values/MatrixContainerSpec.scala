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

class MatrixContainerSpec extends AnyFlatSpec with should.Matchers {

  "MatrixContainer" should "construct from Vector[Vector[Double]]" in {
    val mc = MatrixContainer(Vector(Vector(1.0, 2.0), Vector(3.0, 4.0)))
    mc.rowTotalNumber shouldBe 2
    mc.columnTotalNumber shouldBe 2
  }

  it should "convert to raw EJML matrix and back" in {
    val mc  = MatrixContainer(Vector(Vector(1.0, 2.0), Vector(3.0, 4.0)))
    val raw = mc.rawMatrix
    raw.get(0, 0) shouldBe 1.0
    raw.get(0, 1) shouldBe 2.0
    raw.get(1, 0) shouldBe 3.0
    raw.get(1, 1) shouldBe 4.0
  }

  it should "convert to generic Scala representation" in {
    val mc = MatrixContainer(Vector(Vector(1.0, 2.0), Vector(3.0, 4.0)))
    mc.genericScalaRepresentation shouldBe Vector(Vector(1.0, 2.0), Vector(3.0, 4.0))
  }

  it should "detect square matrices" in {
    val square    = MatrixContainer(Vector(Vector(1.0, 2.0), Vector(3.0, 4.0)))
    val nonSquare = MatrixContainer(Vector(Vector(1.0, 2.0, 3.0), Vector(4.0, 5.0, 6.0)))
    square.isSquare shouldBe true
    nonSquare.isSquare shouldBe false
  }

  it should "create zeros matrix" in {
    val mc = MatrixContainer.zeros(2, 3)
    mc.rowTotalNumber shouldBe 2
    mc.columnTotalNumber shouldBe 3
    mc.values shouldBe empty
    mc.rawMatrix.get(0, 0) shouldBe 0.0
  }

  it should "create identity matrix" in {
    val mc = MatrixContainer.squareIdentity(3)
    mc.rowTotalNumber shouldBe 3
    mc.columnTotalNumber shouldBe 3
    mc.rawMatrix.get(0, 0) shouldBe 1.0
    mc.rawMatrix.get(1, 1) shouldBe 1.0
    mc.rawMatrix.get(2, 2) shouldBe 1.0
    mc.rawMatrix.get(0, 1) shouldBe 0.0
    mc.rawMatrix.get(1, 0) shouldBe 0.0
  }

  it should "column merge two matrices" in {
    val m1     = MatrixContainer(Vector(Vector(1.0, 2.0), Vector(3.0, 4.0)))
    val m2     = MatrixContainer(Vector(Vector(5.0), Vector(6.0)))
    val result = m1.columnMergeWith(m2)
    result.rowTotalNumber shouldBe 2
    result.columnTotalNumber shouldBe 3
    result.genericScalaRepresentation shouldBe Vector(Vector(1.0, 2.0, 5.0), Vector(3.0, 4.0, 6.0))
  }

  it should "row merge two matrices" in {
    val m1     = MatrixContainer(Vector(Vector(1.0, 2.0), Vector(3.0, 4.0)))
    val m2     = MatrixContainer(Vector(Vector(5.0, 6.0)))
    val result = m1.rowMergeWith(m2)
    result.rowTotalNumber shouldBe 3
    result.columnTotalNumber shouldBe 2
    result.genericScalaRepresentation shouldBe Vector(Vector(1.0, 2.0), Vector(3.0, 4.0), Vector(5.0, 6.0))
  }

  it should "diagonal merge two matrices" in {
    val m1     = MatrixContainer(Vector(Vector(1.0, 2.0), Vector(3.0, 4.0)))
    val m2     = MatrixContainer(Vector(Vector(5.0)))
    val result = m1.diagonalMergeWith(m2)
    result.rowTotalNumber shouldBe 3
    result.columnTotalNumber shouldBe 3
    result.genericScalaRepresentation shouldBe Vector(
      Vector(1.0, 2.0, 0.0),
      Vector(3.0, 4.0, 0.0),
      Vector(0.0, 0.0, 5.0)
    )
  }

  it should "diagonal merge non-square matrices" in {
    val m1     = MatrixContainer(Vector(Vector(1.0, 2.0, 3.0), Vector(4.0, 5.0, 6.0)))
    val m2     = MatrixContainer(Vector(Vector(7.0, 8.0), Vector(9.0, 10.0), Vector(11.0, 12.0)))
    val result = m1.diagonalMergeWith(m2)
    result.rowTotalNumber shouldBe 5
    result.columnTotalNumber shouldBe 5
    result.genericScalaRepresentation shouldBe Vector(
      Vector(1.0, 2.0, 3.0, 0.0, 0.0),
      Vector(4.0, 5.0, 6.0, 0.0, 0.0),
      Vector(0.0, 0.0, 0.0, 7.0, 8.0),
      Vector(0.0, 0.0, 0.0, 9.0, 10.0),
      Vector(0.0, 0.0, 0.0, 11.0, 12.0)
    )
  }

  it should "handle sparse values correctly" in {
    val mc        = MatrixContainer(Vector(Vector(0.0, 1.0), Vector(2.0, 0.0)))
    val validated = mc.getValidated
    validated.values should not contain key((1, 1))
    validated.values should not contain key((2, 2))
    validated.rawMatrix.get(0, 1) shouldBe 1.0
    validated.rawMatrix.get(1, 0) shouldBe 2.0
  }
}
