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
package thylacine.model.distributions

import thylacine.model.core.values.{ MatrixContainer, VectorContainer }
import thylacine.model.core.{ CanValidate, RecordedData }
import thylacine.util.LinearAlgebra

import smile.stat.distribution.MultivariateGaussianDistribution

private[thylacine] case class GaussianDistribution(
  mean: VectorContainer,
  covariance: MatrixContainer,
  validated: Boolean = false
) extends Distribution
    with CanValidate[GaussianDistribution] {
  if (!validated) {
    require(covariance.rowTotalNumber == covariance.columnTotalNumber, "Covariance matrix must be square")
    require(covariance.rowTotalNumber == mean.dimension, "Covariance dimension must match mean dimension")
  }

  override private[thylacine] lazy val getValidated: GaussianDistribution =
    if (validated) {
      this
    } else {
      GaussianDistribution(mean.getValidated, covariance.getValidated, validated = true)
    }

  override val domainDimension: Int = mean.dimension

  // Low-level API
  private[thylacine] lazy val rawDistribution: MultivariateGaussianDistribution =
    new MultivariateGaussianDistribution(mean.rawVector, covariance.genericScalaRepresentation.map(_.toArray).toArray)

  private lazy val rawInverseCovariance =
    LinearAlgebra.invert(covariance.rawMatrix)

  override private[thylacine] def logPdfAt(
    input: VectorContainer
  ): Double =
    rawDistribution.logp(input.rawVector)

  override private[thylacine] def logPdfGradientAt(
    input: VectorContainer
  ): VectorContainer = {
    val diff = mean.rawVector.zip(input.rawVector).map { case (m, i) => m - i }
    VectorContainer(LinearAlgebra.multiplyMV(rawInverseCovariance, diff))
  }

}

private[thylacine] object GaussianDistribution {

  private[thylacine] def apply(input: RecordedData): GaussianDistribution = {
    val validatedData = input.getValidated

    GaussianDistribution(
      validatedData.data,
      validatedData.covariance,
      validated = true
    )
  }
}
