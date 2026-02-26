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
package thylacine.model.sampling.hmcmc

import thylacine.model.components.posterior.*
import thylacine.model.components.prior.Prior
import thylacine.model.core.AsyncImplicits
import thylacine.model.core.telemetry.HmcmcTelemetryUpdate
import thylacine.model.core.values.IndexedVectorCollection.ModelParameterCollection
import thylacine.model.core.values.{ IndexedVectorCollection, VectorContainer }
import thylacine.model.sampling.ModelParameterSampler
import thylacine.util.MathOps

import cats.effect.implicits.*
import cats.effect.kernel.{ Async, Deferred, Ref }
import cats.syntax.all.*

/** Implementation of the Hamiltonian MCMC sampling algorithm
  */
private[thylacine] trait HmcmcEngine[F[_]] extends ModelParameterSampler[F] {
  this: AsyncImplicits[F] & Posterior[F, Prior[F, ?], ?] =>

  /*
   * - - -- --- ----- -------- -------------
   * Configuration
   * - - -- --- ----- -------- -------------
   */
  protected def simulationsBetweenSamples: Int
  protected def stepsInSimulation: Int
  protected def simulationEpsilon: Double
  protected def warmUpSimulationCount: Int
  protected def startingPoint: F[ModelParameterCollection]
  protected def telemetryUpdateCallback: HmcmcTelemetryUpdate => F[Unit]
  protected def massMatrixDiagonal: Option[Vector[Double]]
  protected def adaptStepSize: Boolean
  protected def targetAcceptanceRate: Double

  protected def burnInStarted: Ref[F, Boolean]
  protected def burnInResult: Deferred[F, ModelParameterCollection]
  protected def epsilonRef: Ref[F, Double]
  protected def daLogEpsilonBarRef: Ref[F, Double]

  private case class HmcmcSimulationState(
    input: ModelParameterCollection,
    logPdfOpt: Option[Double]                       = None,
    gradLogPdfOpt: Option[ModelParameterCollection] = None,
    iterationCount: Int,
    samples: Vector[ModelParameterCollection] = Vector()
  )

  private case class HmcmcAcceptanceTracking(
    jumpAcceptances: Int = 0,
    jumpAttempts: Int    = 0
  )

  private case class DualAveragingState(
    hBar: Double          = 0.0,
    logEpsilonBar: Double = 0.0,
    mu: Double            = 0.0,
    stepCount: Int        = 0
  )

  /*
   * - - -- --- ----- -------- -------------
   * HMCMC
   * - - -- --- ----- -------- -------------
   */

  // Reflect a scalar value into [lo, hi] and determine if momentum should be negated.
  // Returns (reflected_value, should_negate_momentum).
  private def reflectScalar(x: Double, lo: Double, hi: Double): (Double, Boolean) = {
    if (x >= lo && x <= hi) (x, false)
    else {
      val period = 2.0 * (hi - lo)
      val offset = ((x - lo) % period + period) % period // always non-negative
      if (offset <= (hi - lo)) (lo + offset, false)
      else (lo + period - offset, true)
    }
  }

  // Apply boundary reflection to a position + momentum pair.
  // Only bounded parameters (from UniformPrior) are reflected.
  private def applyBoundaryReflection(
    position: ModelParameterCollection,
    momentum: VectorContainer
  ): (ModelParameterCollection, VectorContainer) = {
    val bounds = collectedBounds
    if (bounds.isEmpty) (position, momentum)
    else {
      val posRaw   = modelParameterCollectionToRawVector(position)
      val momRaw   = momentum.rawVector
      val posArray = posRaw.clone()
      val momArray = momRaw.clone()

      // We need the ordered parameter layout to know which raw indices
      // correspond to which parameter identifiers.
      var offset = 0
      orderedParameterIdentifiersWithDimension.foreach { case (id, dim) =>
        bounds.get(id).foreach { case (lo, hi) =>
          val loVec = lo.rawVector
          val hiVec = hi.rawVector
          (0 until dim).foreach { j =>
            val (reflected, negate) = reflectScalar(posArray(offset + j), loVec(j), hiVec(j))
            posArray(offset + j)             = reflected
            if (negate) momArray(offset + j) = -momArray(offset + j)
          }
        }
        offset += dim
      }

      (rawVectorToModelParameterCollection(posArray), VectorContainer(momArray.toVector))
    }
  }

  // Inverse mass matrix diagonal (1/M_ii for each component).
  // None means unit mass (identity).
  private lazy val inverseMassDiag: Option[VectorContainer] =
    massMatrixDiagonal.map(m => VectorContainer(m.map(1.0 / _)))

  // Apply M^{-1} to a raw vector: element-wise multiply by inverseMassDiag
  private def applyInverseMass(v: VectorContainer): VectorContainer =
    inverseMassDiag match {
      case Some(invM) => v.rawProductWith(invM)
      case None => v
    }

  private def runLeapfrogAt(
    input: ModelParameterCollection,
    rawP: VectorContainer,
    gradNegLogPdf: ModelParameterCollection,
    epsilon: Double,
    iterationCount: Int = 1
  ): F[(ModelParameterCollection, VectorContainer, ModelParameterCollection)] =
    if (iterationCount > stepsInSimulation) {
      Async[F].pure((input, rawP, gradNegLogPdf))
    } else {
      (for {
        p <- Async[F].delay(rawVectorToModelParameterCollection(rawP.rawVector))
        pNew <-
          Async[F].delay(p.rawSumWith(gradNegLogPdf.rawScalarMultiplyWith(-epsilon / 2)))
        pNewRaw <- Async[F].delay {
                     VectorContainer(modelParameterCollectionToRawVector(pNew).toArray.toVector)
                   }
        // Position update: x += ε * M^{-1} * p, then reflect at bounds
        reflectedResult <- Async[F].delay {
                             val scaledP = applyInverseMass(pNewRaw).rawScalarProductWith(epsilon)
                             val xRaw    = input.rawSumWith(rawVectorToModelParameterCollection(scaledP.rawVector))
                             applyBoundaryReflection(xRaw, pNewRaw)
                           }
        (xNew, pReflected) = reflectedResult
        gNew <- logPdfGradientAt(xNew).map(_.rawScalarMultiplyWith(-1))
        // Second half-step of momentum: use reflected momentum
        pReflectedMpc <- Async[F].delay(rawVectorToModelParameterCollection(pReflected.rawVector))
        pNewNew <- Async[F].delay {
                     modelParameterCollectionToRawVector(
                       pReflectedMpc.rawSumWith(
                         gNew.rawScalarMultiplyWith(-epsilon / 2)
                       )
                     )
                   }
      } yield (xNew, VectorContainer(pNewNew.toArray.toVector), gNew)).flatMap { case (xNew, pNewNew, gNew) =>
        runLeapfrogAt(xNew, pNewNew, gNew, epsilon, iterationCount + 1)
      }
    }

  // K(p) = Σ p_i² / (2 * M_ii). With unit mass: p·p / 2.
  private def getHamiltonianValue(p: VectorContainer, E: Double): Double = {
    val kineticEnergy = inverseMassDiag match {
      case Some(invM) => p.rawProductWith(p).rawDotProductWith(invM) / 2.0
      case None => p.rawDotProductWith(p) / 2.0
    }
    kineticEnergy + E
  }

  /*
   * - - -- --- ----- -------- -------------
   * Dual Averaging for Adaptive Step Size
   * (Nesterov 2009, as used in Stan/NUTS)
   * - - -- --- ----- -------- -------------
   */

  // Dual averaging constants
  private val daGamma: Double = 0.05
  private val daT0: Double    = 10.0
  private val daKappa: Double = 0.75

  private def updateDualAveraging(
    daState: DualAveragingState,
    acceptProb: Double
  ): (DualAveragingState, Double) = {
    val m         = daState.stepCount.toDouble
    val newHBar   = (1.0 - 1.0 / (m + daT0)) * daState.hBar + (1.0 / (m + daT0)) * (targetAcceptanceRate - acceptProb)
    val logEps    = daState.mu - Math.sqrt(m) / daGamma * newHBar
    val newLogBar = Math.pow(m, -daKappa) * logEps + (1.0 - Math.pow(m, -daKappa)) * daState.logEpsilonBar
    (daState.copy(hBar = newHBar, logEpsilonBar = newLogBar), logEps)
  }

  private def runDynamicSimulationFrom(
    state: HmcmcSimulationState,
    maxIterations: Int,
    numberOfRequestedSamples: Int,
    burnIn: Boolean                   = false,
    tracking: HmcmcAcceptanceTracking = HmcmcAcceptanceTracking(),
    daState: DualAveragingState       = DualAveragingState()
  ): F[Vector[ModelParameterCollection]] = {
    val simulationSpec: F[Vector[ModelParameterCollection]] = (for {
      epsilon <- epsilonRef.get
      negLogPdf <- state.logPdfOpt match {
                     case Some(res) => Async[F].pure(res)
                     case _ => logPdfAt(state.input).map(_ * -1)
                   }
      gradNegLogPdf <- state.gradLogPdfOpt match {
                         case Some(res) => Async[F].pure(res)
                         case _ => logPdfGradientAt(state.input).map(_.rawScalarMultiplyWith(-1))
                       }
      // Sample momentum: p_i ~ N(0, M_ii). With unit mass: p_i ~ N(0, 1).
      p <- Async[F].delay(massMatrixDiagonal match {
             case Some(massDiag) => VectorContainer.randomWithVariances(massDiag)
             case None => VectorContainer.random(domainDimension)
           })
      hamiltonian    <- Async[F].delay(getHamiltonianValue(p, negLogPdf))
      leapfrogResult <- runLeapfrogAt(state.input, p, gradNegLogPdf, epsilon)
      (xNew, pNew, gradAtXNew) = leapfrogResult
      eNew       <- logPdfAt(xNew).map(_ * -1)
      hNew       <- Async[F].delay(getHamiltonianValue(pNew, eNew))
      dH         <- Async[F].delay(hNew - hamiltonian)
      acceptProb <- Async[F].delay(Math.min(1.0, Math.exp(-dH)))
      _ <- Async[F].ifM(Async[F].pure(burnIn || tracking.jumpAttempts <= 0))(
             Async[F].unit,
             telemetryUpdateCallback(
               HmcmcTelemetryUpdate(
                 samplesRemaining        = numberOfRequestedSamples - state.samples.size,
                 jumpAttempts            = tracking.jumpAttempts,
                 jumpAcceptances         = tracking.jumpAcceptances,
                 hamiltonianDifferential = Option(dH)
               )
             ).start.void
           )
      accepted <- Async[F].delay(dH < 0 || MathOps.nextDouble < Math.exp(-dH))
      result <- Async[F].ifM(Async[F].pure(accepted))(
                  Async[F].pure((xNew, eNew, gradAtXNew, tracking.jumpAcceptances + 1, tracking.jumpAttempts + 1)),
                  Async[F].delay(
                    (state.input, negLogPdf, gradNegLogPdf, tracking.jumpAcceptances, tracking.jumpAttempts + 1)
                  )
                )
      // Dual averaging update during burn-in
      newDaState <- if (burnIn && adaptStepSize) {
                      val stepped           = daState.copy(stepCount = daState.stepCount + 1)
                      val (updated, logEps) = updateDualAveraging(stepped, acceptProb)
                      for {
                        _ <- epsilonRef.set(Math.exp(logEps))
                        _ <- daLogEpsilonBarRef.set(updated.logEpsilonBar)
                      } yield updated
                    } else {
                      Async[F].pure(daState)
                    }
    } yield (result, newDaState)).flatMap { case ((xNew, eNew, gNew, acceptances, attempts), updatedDaState) =>
      runDynamicSimulationFrom(
        state = HmcmcSimulationState(
          input          = xNew,
          logPdfOpt      = Option(eNew),
          gradLogPdfOpt  = Option(gNew),
          iterationCount = state.iterationCount + 1,
          samples        = state.samples
        ),
        maxIterations            = maxIterations,
        numberOfRequestedSamples = numberOfRequestedSamples,
        burnIn                   = burnIn,
        tracking                 = HmcmcAcceptanceTracking(acceptances, attempts),
        daState                  = updatedDaState
      )
    }

    val sampleAppendSpec: F[Vector[ModelParameterCollection]] =
      Async[F].delay(state.samples :+ state.input).flatMap { newSamples =>
        Async[F].ifM(Async[F].pure(newSamples.size >= numberOfRequestedSamples || burnIn))(
          Async[F].pure(newSamples),
          runDynamicSimulationFrom(
            state                    = state.copy(iterationCount = 0, samples = newSamples),
            maxIterations            = maxIterations,
            numberOfRequestedSamples = numberOfRequestedSamples,
            burnIn                   = burnIn,
            tracking                 = tracking,
            daState                  = daState
          )
        )
      }

    Async[F].ifM(Async[F].delay(state.iterationCount <= maxIterations))(
      simulationSpec,
      sampleAppendSpec
    )
  }

  /*
   * - - -- --- ----- -------- -------------
   * Initialisation
   * - - -- --- ----- -------- -------------
   */

  private def runBurnIn: F[ModelParameterCollection] =
    for {
      possibleStartingPoint <- startingPoint
      priorSample <-
        if (possibleStartingPoint == IndexedVectorCollection.empty) {
          samplePriors
        } else {
          Async[F].pure(possibleStartingPoint)
        }
      // Initialize dual averaging state
      mu = Math.log(10.0 * simulationEpsilon)
      _ <- daLogEpsilonBarRef.set(Math.log(simulationEpsilon))
      results <- runDynamicSimulationFrom(
                   state = HmcmcSimulationState(
                     input          = priorSample,
                     iterationCount = 0
                   ),
                   maxIterations            = warmUpSimulationCount,
                   numberOfRequestedSamples = 1,
                   burnIn                   = true,
                   daState = DualAveragingState(
                     mu            = mu,
                     logEpsilonBar = Math.log(simulationEpsilon)
                   )
                 )
      // After burn-in with adaptation, fix epsilon to the smoothed bar value
      _ <- if (adaptStepSize) {
             daLogEpsilonBarRef.get.flatMap(logBar => epsilonRef.set(Math.exp(logBar)))
           } else {
             Async[F].unit
           }
    } yield results.head

  private def getOrRunBurnIn: F[ModelParameterCollection] =
    burnInStarted.getAndSet(true).flatMap { alreadyStarted =>
      if (!alreadyStarted) {
        runBurnIn.flatMap(result => burnInResult.complete(result).as(result))
      } else {
        burnInResult.get
      }
    }

  /*
   * - - -- --- ----- -------- -------------
   * Framework Internal Interfaces
   * - - -- --- ----- -------- -------------
   */

  override protected def sampleModelParameters(numberOfSamples: Int): F[Vector[ModelParameterCollection]] =
    for {
      startPt <- getOrRunBurnIn
      results <- runDynamicSimulationFrom(
                   state = HmcmcSimulationState(
                     input          = startPt,
                     iterationCount = 0
                   ),
                   maxIterations            = simulationsBetweenSamples,
                   numberOfRequestedSamples = numberOfSamples
                 )
    } yield results

}
