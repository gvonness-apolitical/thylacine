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
package thylacine.model.components.posterior

import thylacine.model.components.likelihood.*
import thylacine.model.components.posterior.GaussianAnalyticPosterior.*
import thylacine.model.components.prior.*
import thylacine.model.core.GenericIdentifier.*
import thylacine.model.core.*
import thylacine.model.core.values.IndexedVectorCollection.ModelParameterCollection
import thylacine.model.core.values.{ MatrixContainer, VectorContainer }
import thylacine.model.sampling.ModelParameterSampler
import thylacine.util.LinearAlgebra

import cats.effect.kernel.Async
import cats.syntax.all.*
import smile.math.matrix.Matrix
import smile.stat.distribution.MultivariateGaussianDistribution

import scala.Vector as ScalaVector

case class GaussianAnalyticPosterior[F[_]: Async](
  override private[thylacine] val priors: Set[GaussianPrior[F]],
  override private[thylacine] val likelihoods: Set[GaussianLinearLikelihood[F]],
  override private[thylacine] val validated: Boolean
) extends AsyncImplicits[F]
    with Posterior[F, GaussianPrior[F], GaussianLinearLikelihood[F]]
    with ModelParameterSampler[F]
    with CanValidate[GaussianAnalyticPosterior[F]] {
  if (!validated) {
    require(priors.size == priors.map(_.identifier).size, "Prior identifiers must be unique")
    require(
      likelihoods.size == likelihoods.map(_.posteriorTermIdentifier).size,
      "Likelihood identifiers must be unique"
    )
    require(
      priors
        .map((i: GaussianPrior[F]) => i.posteriorTermIdentifier)
        .intersect(likelihoods.map(_.posteriorTermIdentifier))
        .isEmpty,
      "Prior and likelihood identifiers must not overlap"
    )
  }

  override private[thylacine] lazy val getValidated: GaussianAnalyticPosterior[F] =
    if (validated) {
      this
    } else {
      GaussianAnalyticPosterior(
        priors      = priors.map(_.getValidated),
        likelihoods = likelihoods.map(_.getValidated),
        validated   = true
      )
    }

  lazy val entropy: Double = rawDistribution.entropy()

  private lazy val rawDistribution: MultivariateGaussianDistribution = {
    val priorsAdded: AnalyticPosteriorAccumulation[F] =
      priors.toVector
        .foldLeft(
          AnalyticPosteriorAccumulation[F](
            orderedParameterIdentifiersWithDimension = orderedParameterIdentifiersWithDimension
          )
        )((acc, p) => acc.add(p))

    val allAdded: AnalyticPosteriorAccumulation[F] =
      likelihoods.toVector
        .foldLeft(priorsAdded)((acc, l) => acc.add(l))

    allAdded.gRawDistribution
  }

  lazy val mean: Map[String, ScalaVector[Double]] =
    rawVectorToModelParameterCollection(rawDistribution.mean).genericScalaRepresentation

  // For testing only
  lazy val covarianceStridedVector: ScalaVector[Double] =
    rawDistribution.cov.toArray.flatMap(_.toArray).toVector

  override private[thylacine] def logPdfAt(
    input: ModelParameterCollection
  ): F[Double] =
    Async[F].delay(rawDistribution.logp(modelParameterCollectionToRawVector(input)))

  final override protected def rawSampleModelParameters: F[VectorContainer] =
    Async[F].delay(VectorContainer(rawDistribution.rand()))

  private def sampleModelParameters: F[ModelParameterCollection] =
    rawSampleModelParameters.map(s => rawVectorToModelParameterCollection(s.rawVector))

  final override protected def sampleModelParameters(numberOfSamples: Int): F[Vector[ModelParameterCollection]] =
    (1 to numberOfSamples).toList.traverse(_ => sampleModelParameters).map(_.toVector)

  def init: F[Unit] =
    sample(1).void
}

object GaussianAnalyticPosterior {

  def apply[F[_]: Async](
    priors: Set[GaussianPrior[F]],
    likelihoods: Set[GaussianLinearLikelihood[F]]
  ): GaussianAnalyticPosterior[F] =
    GaussianAnalyticPosterior(priors = priors, likelihoods = likelihoods, validated = false)

  // Contains the logic for merging and ordered set of
  // Gaussian Priors and Linear Likelihoods to produce a single
  // Multivariate Gaussian distribution that represents the
  // Posterior distribution
  private[thylacine] case class AnalyticPosteriorAccumulation[F[_]: Async](
    priorMean: Option[VectorContainer]                 = None,
    priorCovariance: Option[MatrixContainer]           = None,
    data: Option[VectorContainer]                      = None,
    likelihoodCovariance: Option[MatrixContainer]      = None,
    likelihoodTransformations: Option[MatrixContainer] = None,
    orderedParameterIdentifiersWithDimension: ScalaVector[(ModelParameterIdentifier, Int)]
  ) {

    private[thylacine] lazy val gRawDistribution: MultivariateGaussianDistribution =
      (for {
        pmContainer <- priorMean
        pcContainer <- priorCovariance
        dContainer  <- data
        lcContainer <- likelihoodCovariance
        tmContainer <- likelihoodTransformations
      } yield {
        val newInversePriorCovariance = LinearAlgebra.invert(pcContainer.rawMatrix)

        // tmContainer.rawMatrix.t * (lcContainer.rawMatrix \ tmContainer.rawMatrix)
        val tmTranspose = LinearAlgebra.transpose(tmContainer.rawMatrix)
        val lcSolveTm   = LinearAlgebra.solve(lcContainer.rawMatrix, tmContainer.rawMatrix)
        val fisherInfo  = LinearAlgebra.multiply(tmTranspose, lcSolveTm)

        val newInverseCovariance = LinearAlgebra.add(newInversePriorCovariance, fisherInfo)
        val newCovariance        = LinearAlgebra.invert(newInverseCovariance)

        // In reality, this suffers from some pretty serious rounding errors
        // with all the multiple matrix inversions that need to happen
        // newInverseCovariance \ (pcContainer.rawMatrix \ pmContainer.rawVector +
        //   tmContainer.rawMatrix.t * (lcContainer.rawMatrix \ dContainer.rawVector))
        val pcSolvePm                = LinearAlgebra.solve(pcContainer.rawMatrix, pmContainer.rawVector)
        val lcSolveD                 = LinearAlgebra.solve(lcContainer.rawMatrix, dContainer.rawVector)
        val tmTransposeTimesLcSolveD = LinearAlgebra.multiplyMV(tmTranspose, lcSolveD)
        val rhs                      = pcSolvePm.zip(tmTransposeTimesLcSolveD).map { case (a, b) => a + b }
        val newMean                  = LinearAlgebra.solve(newInverseCovariance, rhs)

        // (newCovariance + newCovariance.t) * 0.5
        val symmetricCovariance = LinearAlgebra.symmetrize(newCovariance)

        new MultivariateGaussianDistribution(newMean, Matrix.of(LinearAlgebra.toArray2D(symmetricCovariance)))
      }).getOrElse(
        throw new IllegalStateException(
          "Cannot create posterior Gaussian distribution: required prior or likelihood term is missing"
        )
      )

    private[thylacine] def add(
      prior: GaussianPrior[F]
    ): AnalyticPosteriorAccumulation[F] = {
      val incomingPriorMean       = prior.priorData.data
      val incomingPriorCovariance = prior.priorData.covariance
      this
        .copy(
          priorMean = Some(
            this.priorMean
              .map(_.rawConcatenateWith(incomingPriorMean))
              .getOrElse(incomingPriorMean)
          ),
          priorCovariance = Some(
            this.priorCovariance
              .map(_.diagonalMergeWith(incomingPriorCovariance))
              .getOrElse(incomingPriorCovariance)
          )
        )
    }

    private[thylacine] def add(
      likelihood: GaussianLinearLikelihood[F]
    ): AnalyticPosteriorAccumulation[F] = {
      val incomingData           = likelihood.observations.data
      val incomingDataCovariance = likelihood.observations.covariance
      val incomingTransformationMatrix = orderedParameterIdentifiersWithDimension
        .map { id =>
          likelihood.forwardModel.getJacobian.index.getOrElse(
            id._1,
            MatrixContainer.zeros(likelihood.forwardModel.rangeDimension, id._2)
          )
        }
        .reduce(_.columnMergeWith(_))

      this.copy(
        data = Some(
          this.data
            .map(_.rawConcatenateWith(incomingData))
            .getOrElse(incomingData)
        ),
        likelihoodCovariance = Some(
          this.likelihoodCovariance
            .map(
              _.diagonalMergeWith(incomingDataCovariance)
            )
            .getOrElse(incomingDataCovariance)
        ),
        likelihoodTransformations = Some(
          this.likelihoodTransformations
            .map(
              _.rowMergeWith(incomingTransformationMatrix)
            )
            .getOrElse(incomingTransformationMatrix)
        )
      )
    }

  }
}
