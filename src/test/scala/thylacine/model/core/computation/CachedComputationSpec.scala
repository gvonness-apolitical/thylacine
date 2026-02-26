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
package thylacine.model.core.computation

import bengal.stm.STM
import thylacine.model.core.values.IndexedVectorCollection
import thylacine.model.core.values.IndexedVectorCollection.ModelParameterCollection

import cats.effect.IO
import cats.effect.testing.scalatest.AsyncIOSpec
import cats.syntax.all.*
import org.scalatest.freespec.AsyncFreeSpec
import org.scalatest.matchers.should.Matchers

import java.util.concurrent.atomic.AtomicInteger

class CachedComputationSpec extends AsyncFreeSpec with AsyncIOSpec with Matchers {

  private def mkInput(label: String, values: Vector[Double]): ModelParameterCollection =
    IndexedVectorCollection(Map(label -> values))

  private val inputA: ModelParameterCollection = mkInput("p", Vector(1.0, 2.0))
  private val inputB: ModelParameterCollection = mkInput("p", Vector(3.0, 4.0))

  "CachedComputation" - {

    "return the correct computation result" in {
      STM
        .runtime[IO]
        .flatMap { implicit stm =>
          val computation: ModelParameterCollection => Double = { input =>
            input.genericScalaRepresentation("p").sum
          }
          for {
            cached <- CachedComputation.of[IO, Double](computation, cacheDepth = Some(5))
            result <- cached.performComputation(inputA)
          } yield result
        }
        .asserting(_ shouldBe 3.0)
    }

    "cache miss - new input triggers computation" in {
      STM
        .runtime[IO]
        .flatMap { implicit stm =>
          val callCount = new AtomicInteger(0)
          val computation: ModelParameterCollection => Double = { input =>
            callCount.incrementAndGet()
            input.genericScalaRepresentation("p").sum
          }
          for {
            cached  <- CachedComputation.of[IO, Double](computation, cacheDepth = Some(5))
            result1 <- cached.performComputation(inputA)
            result2 <- cached.performComputation(inputB)
          } yield (result1, result2, callCount.get())
        }
        .asserting { case (r1, r2, count) =>
          r1 shouldBe 3.0
          r2 shouldBe 7.0
          count shouldBe 2
        }
    }

    "None cache depth - caching disabled, every call triggers computation" in {
      STM
        .runtime[IO]
        .flatMap { implicit stm =>
          val callCount = new AtomicInteger(0)
          val computation: ModelParameterCollection => Double = { input =>
            callCount.incrementAndGet()
            input.genericScalaRepresentation("p").sum
          }
          for {
            cached  <- CachedComputation.of[IO, Double](computation, cacheDepth = None)
            result1 <- cached.performComputation(inputA)
            result2 <- cached.performComputation(inputA)
            result3 <- cached.performComputation(inputA)
          } yield (result1, result2, result3, callCount.get())
        }
        .asserting { case (r1, r2, r3, count) =>
          r1 shouldBe 3.0
          r2 shouldBe 3.0
          r3 shouldBe 3.0
          count shouldBe 3
        }
    }

    "cache depth of zero - behaves like caching disabled" in {
      STM
        .runtime[IO]
        .flatMap { implicit stm =>
          val callCount = new AtomicInteger(0)
          val computation: ModelParameterCollection => Double = { input =>
            callCount.incrementAndGet()
            input.genericScalaRepresentation("p").sum
          }
          for {
            cached  <- CachedComputation.of[IO, Double](computation, cacheDepth = Some(0))
            result1 <- cached.performComputation(inputA)
            result2 <- cached.performComputation(inputA)
          } yield (result1, result2, callCount.get())
        }
        .asserting { case (r1, r2, count) =>
          r1 shouldBe 3.0
          r2 shouldBe 3.0
          count shouldBe 2
        }
    }

    "concurrent access - multiple fibers get correct results" in {
      STM
        .runtime[IO]
        .flatMap { implicit stm =>
          val callCount = new AtomicInteger(0)
          val computation: ModelParameterCollection => Double = { input =>
            callCount.incrementAndGet()
            input.genericScalaRepresentation("p").sum
          }
          for {
            cached <- CachedComputation.of[IO, Double](computation, cacheDepth = Some(10))
            // Launch 20 fibers, each computing one of 5 distinct inputs
            inputs = (0 until 20).toList.map(i => mkInput("p", Vector(i % 5.toDouble, 0.0)))
            results <- inputs.parTraverse(cached.performComputation)
          } yield results
        }
        .asserting { results =>
          // Each result should match the expected sum for its input
          results.zipWithIndex.foreach { case (r, i) =>
            r shouldBe (i % 5).toDouble
          }
          succeed
        }
    }

  }
}
