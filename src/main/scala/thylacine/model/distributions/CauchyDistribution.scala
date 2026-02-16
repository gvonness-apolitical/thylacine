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

import smile.math.special.Gamma.gamma
import smile.stat.distribution.{ ChiSquareDistribution, MultivariateGaussianDistribution }

private[thylacine] case class CauchyDistribution(
  mean: VectorContainer,
  covariance: MatrixContainer,
  validated: Boolean = false
) extends Distribution
    with CanValidate[CauchyDistribution] {
  if (!validated) {
    require(covariance.rowTotalNumber == covariance.columnTotalNumber, "Covariance matrix must be square")
    require(covariance.rowTotalNumber == mean.dimension, "Covariance dimension must match mean dimension")
  }

  override private[thylacine] lazy val getValidated: CauchyDistribution =
    if (validated) {
      this
    } else {
      CauchyDistribution(mean.getValidated, covariance.getValidated, validated = true)
    }

  private lazy val logMultiplier = Math.log(gamma((1 + domainDimension) / 2.0)) - Math.log(
    gamma(0.5)
  ) - domainDimension / 2.0 * Math.log(Math.PI) - Math.log(LinearAlgebra.determinant(covariance.rawMatrix)) / 2.0

  override val domainDimension: Int = mean.dimension

  private lazy val rawInverseCovariance =
    LinearAlgebra.invert(covariance.rawMatrix)

  override private[thylacine] def logPdfAt(
    input: VectorContainer
  ): Double = {
    val differentialFromMean = input.rawVector.zip(mean.rawVector).map { case (i, m) => i - m }
    val quadForm             = LinearAlgebra.quadraticForm(differentialFromMean, rawInverseCovariance)

    logMultiplier - (1.0 + domainDimension) / 2.0 * Math.log(1 + quadForm)
  }

  override private[thylacine] def logPdfGradientAt(
    input: VectorContainer
  ): VectorContainer = {
    val differentialFromMean = input.rawVector.zip(mean.rawVector).map { case (i, m) => i - m }
    val quadForm             = LinearAlgebra.quadraticForm(differentialFromMean, rawInverseCovariance)
    val multiplierResult     = (1 + domainDimension) / (1 + quadForm)
    val vectorResult         = LinearAlgebra.multiplyMV(rawInverseCovariance, differentialFromMean)

    VectorContainer(vectorResult.map(_ * multiplierResult))
  }

  private lazy val chiSquared = new ChiSquareDistribution(1)

  // Leveraging connection to Gamma and Gaussian distributions
  private[thylacine] def getRawSample: VectorContainer = {
    val scaledCovariance = LinearAlgebra.divide(covariance.rawMatrix, chiSquared.rand())
    val scaledCovArray   = LinearAlgebra.toArray2D(scaledCovariance)
    val mvn              = new MultivariateGaussianDistribution(mean.rawVector, scaledCovArray)
    VectorContainer(mvn.rand())
  }

}

private[thylacine] object CauchyDistribution {

  private[thylacine] def apply(input: RecordedData): CauchyDistribution = {
    val validatedData = input.getValidated

    CauchyDistribution(
      validatedData.data,
      validatedData.covariance,
      validated = true
    )
  }
}
