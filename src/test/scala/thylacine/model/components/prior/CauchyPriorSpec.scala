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
package thylacine.model.components.prior

import thylacine.model.core.values.VectorContainer
import thylacine.model.distributions.CauchyDistribution

import cats.effect.IO
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should

class CauchyPriorSpec extends AnyFlatSpec with should.Matchers {

  private val tol = 1e-8

  // scale=1 → variance=1
  private val prior = CauchyPrior[IO](
    label           = "x",
    values          = Vector(0.0),
    scaleParameters = Vector(1.0)
  )

  private val dist = CauchyDistribution(prior.priorData)

  "CauchyPrior" should "compute correct gradient at a non-mean point" in {
    // μ=0, σ²=1, x=1: Q=1, gradient = -(1+1)/(1+1) * 1 * 1 = -1.0
    val grad = prior.rawLogPdfGradientAt(Vector(1.0))
    grad(0) shouldBe (-1.0 +- tol)
  }

  it should "compute zero gradient at the mean" in {
    val grad = prior.rawLogPdfGradientAt(Vector(0.0))
    grad(0) shouldBe (0.0 +- tol)
  }

  it should "match gradient via finite differences" in {
    val x           = Vector(1.5)
    val eps         = 1e-7
    val grad        = prior.rawLogPdfGradientAt(x)
    val logPdfBase  = dist.logPdfAt(VectorContainer(x))
    val logPdfNudge = dist.logPdfAt(VectorContainer(Vector(x(0) + eps)))
    val fdGrad      = (logPdfNudge - logPdfBase) / eps
    grad(0) shouldBe (fdGrad +- 1e-5)
  }
}
