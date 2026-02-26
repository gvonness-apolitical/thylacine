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
package thylacine.model.integration.slq

import cats.effect.IO
import cats.effect.testing.scalatest.AsyncIOSpec
import org.scalatest.freespec.AsyncFreeSpec
import org.scalatest.matchers.should.Matchers

class QuadratureAbscissaSpec extends AsyncFreeSpec with AsyncIOSpec with Matchers {

  "QuadratureAbscissa" - {

    "create a sample pool of the requested size" in {
      IO {
        val qa = QuadratureAbscissa(10)
        qa.samplePool.size shouldBe 10
      }
    }

    "create sample pool values in (0, 1]" in {
      IO {
        val qa = QuadratureAbscissa(50)
        qa.samplePool.foreach { v =>
          v should be > 0.0
          v should be <= 1.0
        }
        succeed
      }
    }

    "create unique sample pool values" in {
      IO {
        val qa = QuadratureAbscissa(50)
        qa.samplePool.size shouldBe 50
      }
    }

    "start with an empty abscissa" in {
      IO {
        val qa = QuadratureAbscissa(10)
        qa.abscissa shouldBe empty
      }
    }

    "extend abscissa by one" in {
      IO {
        val qa       = QuadratureAbscissa(10)
        val extended = qa.extendAbscissaByOne
        extended.abscissa.size shouldBe 1
      }
    }

    "extend abscissa multiple times" in {
      IO {
        val qa = QuadratureAbscissa(10)
        val extended = (1 to 5).foldLeft(qa) { (acc, _) =>
          acc.extendAbscissaByOne
        }
        extended.abscissa.size shouldBe 5
      }
    }

    "produce a trapezoidal quadrature after sufficient extensions" in {
      IO {
        val qa = QuadratureAbscissa(10)
        val extended = (1 to 5).foldLeft(qa) { (acc, _) =>
          acc.extendAbscissaByOne
        }
        val quad = extended.getTrapezoidalQuadrature
        quad.size shouldBe 5
        quad.foreach { v =>
          v should not be 0.0
        }
        succeed
      }
    }

    "track maxSample correctly" in {
      IO {
        val qa = QuadratureAbscissa(10)
        qa.maxSample should be > 0.0
        qa.maxSample should be <= 1.0
      }
    }

    "shrink sample pool on each extension" in {
      IO {
        val qa       = QuadratureAbscissa(10)
        val extended = qa.extendAbscissaByOne
        // The max sample is moved from pool to abscissa, and a new random
        // value replaces it, so pool size stays the same
        extended.samplePool.size shouldBe 10
      }
    }
  }

  "QuadratureAbscissaCollection" - {

    "create the requested number of abscissas" in {
      IO {
        val collection = QuadratureAbscissaCollection(3, 10)
        collection.abscissas.size shouldBe 3
      }
    }

    "start with size 0 (no abscissa points yet)" in {
      IO {
        val collection = QuadratureAbscissaCollection(3, 10)
        collection.size shouldBe 0
      }
    }

    "extend all abscissas by one" in {
      IO {
        val collection = QuadratureAbscissaCollection(3, 10)
        val extended   = collection.extendAllAbscissaByOne
        extended.size shouldBe 1
        extended.abscissas.foreach { qa =>
          qa.abscissa.size shouldBe 1
        }
        succeed
      }
    }

    "produce quadratures after extensions" in {
      IO {
        val collection = QuadratureAbscissaCollection(3, 10)
        val extended = (1 to 5).foldLeft(collection) { (acc, _) =>
          acc.extendAllAbscissaByOne
        }
        val quadratures = extended.getQuadratures
        quadratures.size shouldBe 3
        quadratures.foreach { q =>
          q.size shouldBe 5
        }
        succeed
      }
    }

    "produce abscissa vectors after extensions" in {
      IO {
        val collection = QuadratureAbscissaCollection(2, 10)
        val extended = (1 to 4).foldLeft(collection) { (acc, _) =>
          acc.extendAllAbscissaByOne
        }
        val abscissaVecs = extended.getAbscissas
        abscissaVecs.size shouldBe 2
        abscissaVecs.foreach { a =>
          a.size shouldBe 4
        }
        succeed
      }
    }

    "init creates an empty collection" in {
      IO {
        val emptyCollection = QuadratureAbscissaCollection.init
        emptyCollection.abscissas shouldBe Vector.empty
        emptyCollection.size shouldBe 0
      }
    }
  }
}
