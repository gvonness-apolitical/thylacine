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

import thylacine.config.HmcmcConfig
import thylacine.model.components.likelihood.Likelihood
import thylacine.model.components.prior.Prior
import thylacine.model.core.AsyncImplicits
import thylacine.model.core.telemetry.HmcmcTelemetryUpdate
import thylacine.model.core.values.IndexedVectorCollection
import thylacine.model.core.values.IndexedVectorCollection.ModelParameterCollection
import thylacine.model.sampling.hmcmc.HmcmcEngine

import cats.effect.kernel.{ Async, Deferred, Ref }
import cats.syntax.all.*

case class HmcmcSampledPosterior[F[_]: Async] private[thylacine] (
  private[thylacine] val hmcmcConfig: HmcmcConfig,
  override protected val telemetryUpdateCallback: HmcmcTelemetryUpdate => F[Unit],
  private[thylacine] val seed: Map[String, Vector[Double]],
  override private[thylacine] val priors: Set[Prior[F, ?]],
  override private[thylacine] val likelihoods: Set[Likelihood[F, ?, ?]],
  override protected val burnInStarted: Ref[F, Boolean],
  override protected val burnInResult: Deferred[F, ModelParameterCollection],
  override protected val epsilonRef: Ref[F, Double],
  override protected val daLogEpsilonBarRef: Ref[F, Double]
) extends AsyncImplicits[F]
    with Posterior[F, Prior[F, ?], Likelihood[F, ?, ?]]
    with HmcmcEngine[F] {

  final override protected val simulationsBetweenSamples: Int =
    hmcmcConfig.stepsBetweenSamples

  final override protected val stepsInSimulation: Int =
    hmcmcConfig.stepsInDynamicsSimulation

  final override protected val simulationEpsilon: Double =
    hmcmcConfig.dynamicsSimulationStepSize

  final override protected val warmUpSimulationCount: Int =
    hmcmcConfig.warmupStepCount

  final override protected val massMatrixDiagonal: Option[Vector[Double]] =
    hmcmcConfig.massMatrixDiagonal

  final override protected val adaptStepSize: Boolean =
    hmcmcConfig.adaptStepSize

  final override protected val targetAcceptanceRate: Double =
    hmcmcConfig.targetAcceptanceRate

  final override protected val startingPoint: F[ModelParameterCollection] =
    Async[F].delay(IndexedVectorCollection(seed))
}

object HmcmcSampledPosterior {

  def of[F[_]: Async](
    hmcmcConfig: HmcmcConfig,
    telemetryUpdateCallback: HmcmcTelemetryUpdate => F[Unit],
    seed: Map[String, Vector[Double]],
    priors: Set[Prior[F, ?]],
    likelihoods: Set[Likelihood[F, ?, ?]]
  ): F[HmcmcSampledPosterior[F]] =
    for {
      started     <- Ref.of[F, Boolean](false)
      deferred    <- Deferred[F, ModelParameterCollection]
      epsRef      <- Ref.of[F, Double](hmcmcConfig.dynamicsSimulationStepSize)
      daLogBarRef <- Ref.of[F, Double](Math.log(hmcmcConfig.dynamicsSimulationStepSize))
    } yield HmcmcSampledPosterior(
      hmcmcConfig             = hmcmcConfig,
      telemetryUpdateCallback = telemetryUpdateCallback,
      seed                    = seed,
      priors                  = priors,
      likelihoods             = likelihoods,
      burnInStarted           = started,
      burnInResult            = deferred,
      epsilonRef              = epsRef,
      daLogEpsilonBarRef      = daLogBarRef
    )

  def of[F[_]: Async](
    hmcmcConfig: HmcmcConfig,
    posterior: Posterior[F, Prior[F, ?], Likelihood[F, ?, ?]],
    telemetryUpdateCallback: HmcmcTelemetryUpdate => F[Unit],
    seed: Map[String, Vector[Double]]
  ): F[HmcmcSampledPosterior[F]] =
    of(
      hmcmcConfig             = hmcmcConfig,
      telemetryUpdateCallback = telemetryUpdateCallback,
      seed                    = seed,
      priors                  = posterior.priors,
      likelihoods             = posterior.likelihoods
    )

}
