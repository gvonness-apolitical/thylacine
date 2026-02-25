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

import thylacine.model.core.GenericIdentifier.*
import thylacine.model.core.*
import thylacine.model.core.values.{ MatrixContainer, VectorContainer }
import thylacine.model.distributions.GaussianDistribution

import cats.effect.kernel.Async
import smile.stat.distribution.MultivariateGaussianDistribution

case class GaussianPrior[F[_]: Async](
  override private[thylacine] val identifier: ModelParameterIdentifier,
  private[thylacine] val priorData: RecordedData,
  override private[thylacine] val validated: Boolean = false
) extends AsyncImplicits[F]
    with Prior[F, GaussianDistribution] {

  override protected lazy val priorDistribution: GaussianDistribution =
    GaussianDistribution(priorData)

  private lazy val rawDistribution: MultivariateGaussianDistribution =
    priorDistribution.rawDistribution

  override private[thylacine] lazy val getValidated: GaussianPrior[F] =
    if (validated) this
    else this.copy(priorData = priorData.getValidated, validated = true)

  override protected def rawSampleModelParameters: F[VectorContainer] =
    Async[F].delay(VectorContainer(rawDistribution.rand()))

  // Testing
  private[thylacine] lazy val mean: Vector[Double] =
    rawDistribution.mean.toVector

  private[thylacine] lazy val covariance: Vector[Double] =
    rawDistribution.cov.toArray.flatMap(_.toArray).toVector

  lazy val entropy: Double = rawDistribution.entropy()
}

object GaussianPrior {

  def fromStandardDeviations[F[_]: Async](
    label: String,
    values: Vector[Double],
    standardDeviations: Vector[Double]
  ): GaussianPrior[F] = {
    require(values.size == standardDeviations.size, "Values and standard deviations must have the same size")
    GaussianPrior(
      identifier = ModelParameterIdentifier(label),
      priorData = RecordedData(
        values             = VectorContainer(values),
        standardDeviations = VectorContainer(standardDeviations)
      )
    )
  }

  def fromCovarianceMatrix[F[_]: Async](
    label: String,
    values: Vector[Double],
    covarianceMatrix: Vector[Vector[Double]]
  ): GaussianPrior[F] = {
    val covarianceContainer = MatrixContainer(covarianceMatrix)
    val valueContainer      = VectorContainer(values)
    require(
      covarianceContainer.isSquare && valueContainer.dimension == covarianceContainer.rowTotalNumber,
      "Covariance must be square and match value dimension"
    )
    GaussianPrior(
      identifier = ModelParameterIdentifier(label),
      priorData = RecordedData(
        data       = valueContainer,
        covariance = covarianceContainer
      )
    )
  }
}
