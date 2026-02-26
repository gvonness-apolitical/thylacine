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

class PointInCubeSpec extends AsyncFreeSpec with AsyncIOSpec with Matchers {

  "PointInInterval" - {

    "create a valid interval with point inside bounds" in {
      IO {
        val pii = PointInInterval(point = 0.5, lowerBound = 0.0, upperBound = 1.0)
        pii.point shouldBe 0.5
        pii.lowerBound shouldBe 0.0
        pii.upperBound shouldBe 1.0
      }
    }

    "calculate interval length" in {
      IO {
        val pii = PointInInterval(point = 0.5, lowerBound = 0.0, upperBound = 1.0)
        pii.intervalLength shouldBe 1.0
      }
    }

    "create a placeholder interval from a single point" in {
      IO {
        val pii = PointInInterval(3.0)
        pii.point shouldBe 3.0
        pii.lowerBound shouldBe 2.0
        pii.upperBound shouldBe 4.0
      }
    }

    "symmetrize an asymmetric interval" in {
      IO {
        val pii       = PointInInterval(point = 0.5, lowerBound = 0.0, upperBound = 2.0, validated = true)
        val symm      = pii.symmetrize
        val lowerDist = symm.point - symm.lowerBound
        val upperDist = symm.upperBound - symm.point
        lowerDist shouldBe (upperDist +- 1e-12)
      }
    }

    "detect intersecting intervals" in {
      IO {
        val pii1 = PointInInterval(point = 0.5, lowerBound = 0.0, upperBound = 1.0, validated = true)
        val pii2 = PointInInterval(point = 0.8, lowerBound = 0.5, upperBound = 1.5, validated = true)
        pii1.isIntersectingWith(pii2) shouldBe true
      }
    }

    "detect non-intersecting intervals" in {
      IO {
        val pii1 = PointInInterval(point = 0.5, lowerBound = 0.0, upperBound = 1.0, validated = true)
        val pii2 = PointInInterval(point = 2.0, lowerBound = 1.5, upperBound = 3.0, validated = true)
        pii1.isIntersectingWith(pii2) shouldBe false
      }
    }

    "compute distance squared between two intervals" in {
      IO {
        val pii1 = PointInInterval(point = 1.0, lowerBound = 0.0, upperBound = 2.0, validated = true)
        val pii2 = PointInInterval(point = 4.0, lowerBound = 3.0, upperBound = 5.0, validated = true)
        pii1.distanceSquaredFrom(pii2) shouldBe 9.0
      }
    }

    "find disjoint boundary between two intervals" in {
      IO {
        val pii1     = PointInInterval(point = 1.0, lowerBound = 0.0, upperBound = 3.0, validated = true)
        val pii2     = PointInInterval(point = 2.0, lowerBound = 0.5, upperBound = 4.0, validated = true)
        val (d1, d2) = PointInInterval.findDisjointBoundary(pii1, pii2)
        d1.isIntersectingWith(d2) shouldBe false
      }
    }

    "generate a sample within a scaled interval" in {
      IO {
        val pii     = PointInInterval(point = 5.0, lowerBound = 3.0, upperBound = 7.0, validated = true)
        val samples = (1 to 100).map(_ => pii.getSample(1.0))
        samples.foreach { s =>
          s should be >= (5.0 - 2.0) // point - 0.5 * scale * length
          s should be <= (5.0 + 2.0)
        }
        succeed
      }
    }

    "reject invalid interval where upper <= lower" in {
      IO {
        assertThrows[IllegalArgumentException] {
          PointInInterval(point = 0.5, lowerBound = 1.0, upperBound = 0.0)
        }
      }
    }
  }

  "PointInCube" - {

    "report correct dimension" in {
      IO {
        val pic = PointInCube(
          Vector(PointInInterval(1.0), PointInInterval(2.0), PointInInterval(3.0)),
          validated = true
        )
        pic.dimension shouldBe 3
      }
    }

    "compute cube volume as sum of interval lengths" in {
      IO {
        val pic = PointInCube(
          Vector(
            PointInInterval(point = 0.0, lowerBound = -1.0, upperBound = 1.0, validated = true),
            PointInInterval(point = 0.0, lowerBound = -2.0, upperBound = 2.0, validated = true)
          ),
          validated = true
        )
        // cubeVolume is sum of interval lengths (not product)
        pic.cubeVolume shouldBe BigDecimal(6.0)
      }
    }

    "extract point as VectorContainer" in {
      IO {
        val pic = PointInCube(
          Vector(PointInInterval(1.0), PointInInterval(2.0)),
          validated = true
        )
        pic.point.scalaVector shouldBe Vector(1.0, 2.0)
      }
    }

    "symmetrize all intervals" in {
      IO {
        val pic = PointInCube(
          Vector(
            PointInInterval(point = 0.5, lowerBound = 0.0, upperBound = 2.0, validated = true),
            PointInInterval(point = 1.0, lowerBound = 0.0, upperBound = 1.5, validated = true)
          ),
          validated = true
        )
        val symm = pic.symmetrize
        symm.pointInIntervals.foreach { pii =>
          val lower = pii.point - pii.lowerBound
          val upper = pii.upperBound - pii.point
          lower shouldBe (upper +- 1e-12)
        }
        succeed
      }
    }

    "detect intersection between two cubes" in {
      IO {
        val pic1 = PointInCube(
          Vector(
            PointInInterval(point = 0.0, lowerBound = -1.0, upperBound = 1.0, validated = true)
          ),
          validated = true
        )
        val pic2 = PointInCube(
          Vector(
            PointInInterval(point = 0.5, lowerBound = -0.5, upperBound = 1.5, validated = true)
          ),
          validated = true
        )
        pic1.isIntersectingWith(pic2) shouldBe true
      }
    }

    "make two cubes disjoint" in {
      IO {
        val pic1 = PointInCube(
          Vector(PointInInterval(point = 1.0, lowerBound = 0.0, upperBound = 3.0, validated = true)),
          validated = true
        )
        val pic2 = PointInCube(
          Vector(PointInInterval(point = 2.0, lowerBound = 0.5, upperBound = 4.0, validated = true)),
          validated = true
        )
        val (d1, d2) = PointInCube.makeDisjoint(pic1, pic2)
        d1.isIntersectingWith(d2) shouldBe false
      }
    }

    "make a vector of cubes disjoint" in {
      IO {
        val cubes = Vector(
          PointInCube(Vector(PointInInterval(1.0)), validated = true),
          PointInCube(Vector(PointInInterval(1.5)), validated = true),
          PointInCube(Vector(PointInInterval(2.0)), validated = true)
        )
        val disjoint = PointInCube.makeDisjoint(cubes)
        disjoint.size shouldBe 3
        // Check pairwise disjointness
        for {
          i <- disjoint.indices
          j <- (i + 1) until disjoint.size
        } {
          disjoint(i).isIntersectingWith(disjoint(j)) shouldBe false
        }
        succeed
      }
    }

    "generate a sample from a cube" in {
      IO {
        val pic = PointInCube(
          Vector(
            PointInInterval(point = 0.0, lowerBound = -1.0, upperBound = 1.0, validated = true),
            PointInInterval(point = 0.0, lowerBound = -1.0, upperBound = 1.0, validated = true)
          ),
          validated = true
        )
        val sample = pic.getSample(1.0)
        sample.dimension shouldBe 2
      }
    }

    "retrieve and replace dimension index" in {
      IO {
        val pic = PointInCube(
          Vector(PointInInterval(1.0), PointInInterval(2.0)),
          validated = true
        )
        val retrieved = pic.retrieveIndex(1)
        retrieved.point shouldBe 1.0

        val replacement = PointInInterval(point = 5.0, lowerBound = 4.0, upperBound = 6.0)
        val replaced    = pic.replaceIndex(1, replacement)
        replaced.retrieveIndex(1).point shouldBe 5.0
        replaced.retrieveIndex(2).point shouldBe 2.0
      }
    }
  }

  "PointInCubeCollection" - {

    "prepare a collection for sampling via readyForSampling" in {
      IO {
        val pics = Vector(
          PointInCube(Vector(PointInInterval(1.0), PointInInterval(2.0)), validated = true),
          PointInCube(Vector(PointInInterval(3.0), PointInInterval(4.0)), validated = true)
        )
        val collection = PointInCubeCollection(pics)
        val ready      = collection.readyForSampling
        ready.pointsInCube.size shouldBe 2
        ready.dimension shouldBe 2
      }
    }

    "generate samples from a ready collection" in {
      IO {
        val pics = Vector(
          PointInCube(Vector(PointInInterval(1.0), PointInInterval(2.0)), validated = true),
          PointInCube(Vector(PointInInterval(3.0), PointInInterval(4.0)), validated = true)
        )
        val collection = PointInCubeCollection(pics)
        val ready      = collection.readyForSampling
        val sample     = ready.getSample(1.0)
        sample.dimension shouldBe 2
      }
    }

    "extract points only" in {
      IO {
        val pics = Vector(
          PointInCube(Vector(PointInInterval(1.0), PointInInterval(2.0)), validated = true),
          PointInCube(Vector(PointInInterval(3.0), PointInInterval(4.0)), validated = true)
        )
        val collection = PointInCubeCollection(pics, validated = true)
        val points     = collection.pointsOnly
        points.size shouldBe 2
        points.head.scalaVector shouldBe Vector(1.0, 2.0)
        points(1).scalaVector shouldBe Vector(3.0, 4.0)
      }
    }

    "require at least 2 points" in {
      IO {
        assertThrows[IllegalArgumentException] {
          PointInCubeCollection(
            Vector(PointInCube(Vector(PointInInterval(1.0)), validated = true))
          )
        }
      }
    }

    "empty collection has dimension 0" in {
      IO {
        PointInCubeCollection.empty.dimension shouldBe 0
      }
    }
  }
}
