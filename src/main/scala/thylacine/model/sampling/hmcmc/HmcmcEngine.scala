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

  // Dual averaging state: (hBar, logEpsilonBar, mu, stepCount)
  // hBar: running average of (delta - alpha)
  // logEpsilonBar: smoothed log-epsilon (final value after burn-in)
  // mu: log(10 * epsilon_0)
  // stepCount: number of adaptation steps taken
  private def updateDualAveraging(
    hBar: Double,
    logEpsilonBar: Double,
    mu: Double,
    stepCount: Int,
    acceptProb: Double
  ): (Double, Double, Double) = {
    val m         = stepCount.toDouble
    val newHBar   = (1.0 - 1.0 / (m + daT0)) * hBar + (1.0 / (m + daT0)) * (targetAcceptanceRate - acceptProb)
    val logEps    = mu - Math.sqrt(m) / daGamma * newHBar
    val newLogBar = Math.pow(m, -daKappa) * logEps + (1.0 - Math.pow(m, -daKappa)) * logEpsilonBar
    (newHBar, newLogBar, logEps)
  }

  private def runDynamicSimulationFrom(
    input: ModelParameterCollection,
    maxIterations: Int,
    logPdfOpt: Option[Double]                       = None,
    gradLogPdfOpt: Option[ModelParameterCollection] = None,
    burnIn: Boolean                                 = false,
    iterationCount: Int,
    numberOfRequestedSamples: Int,
    jumpAcceptances: Int                      = 0,
    jumpAttempts: Int                         = 0,
    samples: Vector[ModelParameterCollection] = Vector(),
    // Dual averaging state (only used during burn-in with adaptStepSize=true)
    daHBar: Double          = 0.0,
    daLogEpsilonBar: Double = 0.0,
    daMu: Double            = 0.0,
    daStepCount: Int        = 0
  ): F[Vector[ModelParameterCollection]] = {
    val simulationSpec: F[Vector[ModelParameterCollection]] = (for {
      epsilon <- epsilonRef.get
      negLogPdf <- logPdfOpt match {
                     case Some(res) => Async[F].pure(res)
                     case _ => logPdfAt(input).map(_ * -1)
                   }
      gradNegLogPdf <- gradLogPdfOpt match {
                         case Some(res) => Async[F].pure(res)
                         case _ => logPdfGradientAt(input).map(_.rawScalarMultiplyWith(-1))
                       }
      // Sample momentum: p_i ~ N(0, M_ii). With unit mass: p_i ~ N(0, 1).
      p <- Async[F].delay(massMatrixDiagonal match {
             case Some(massDiag) => VectorContainer.randomWithVariances(massDiag)
             case None => VectorContainer.random(domainDimension)
           })
      hamiltonian    <- Async[F].delay(getHamiltonianValue(p, negLogPdf))
      leapfrogResult <- runLeapfrogAt(input, p, gradNegLogPdf, epsilon)
      (xNew, pNew, gradAtXNew) = leapfrogResult
      eNew       <- logPdfAt(xNew).map(_ * -1)
      hNew       <- Async[F].delay(getHamiltonianValue(pNew, eNew))
      dH         <- Async[F].delay(hNew - hamiltonian)
      acceptProb <- Async[F].delay(Math.min(1.0, Math.exp(-dH)))
      _ <- Async[F].ifM(Async[F].pure(burnIn || jumpAttempts <= 0))(
             Async[F].unit,
             telemetryUpdateCallback(
               HmcmcTelemetryUpdate(
                 samplesRemaining        = numberOfRequestedSamples - samples.size,
                 jumpAttempts            = jumpAttempts,
                 jumpAcceptances         = jumpAcceptances,
                 hamiltonianDifferential = Option(dH)
               )
             ).start.void
           )
      accepted <- Async[F].delay(dH < 0 || MathOps.nextDouble < Math.exp(-dH))
      result <- Async[F].ifM(Async[F].pure(accepted))(
                  Async[F].pure((xNew, eNew, gradAtXNew, jumpAcceptances + 1, jumpAttempts + 1)),
                  Async[F].delay((input, negLogPdf, gradNegLogPdf, jumpAcceptances, jumpAttempts + 1))
                )
      // Dual averaging update during burn-in
      daState <- if (burnIn && adaptStepSize) {
                   val newStep = daStepCount + 1
                   val (newHBar, newLogBar, logEps) =
                     updateDualAveraging(daHBar, daLogEpsilonBar, daMu, newStep, acceptProb)
                   for {
                     _ <- epsilonRef.set(Math.exp(logEps))
                     _ <- daLogEpsilonBarRef.set(newLogBar)
                   } yield (newHBar, newLogBar, daMu, newStep)
                 } else {
                   Async[F].pure((daHBar, daLogEpsilonBar, daMu, daStepCount))
                 }
    } yield (result, daState)).flatMap {
      case ((xNew, eNew, gNew, acceptances, attempts), (newHBar, newLogBar, newMu, newStepCount)) =>
        runDynamicSimulationFrom(
          input                    = xNew,
          maxIterations            = maxIterations,
          logPdfOpt                = Option(eNew),
          gradLogPdfOpt            = Option(gNew),
          burnIn                   = burnIn,
          iterationCount           = iterationCount + 1,
          numberOfRequestedSamples = numberOfRequestedSamples,
          jumpAcceptances          = acceptances,
          jumpAttempts             = attempts,
          samples                  = samples,
          daHBar                   = newHBar,
          daLogEpsilonBar          = newLogBar,
          daMu                     = newMu,
          daStepCount              = newStepCount
        )
    }

    val sampleAppendSpec: F[Vector[ModelParameterCollection]] =
      Async[F].delay(samples :+ input).flatMap { newSamples =>
        Async[F].ifM(Async[F].pure(newSamples.size >= numberOfRequestedSamples || burnIn))(
          Async[F].pure(newSamples),
          runDynamicSimulationFrom(
            input                    = input,
            maxIterations            = maxIterations,
            logPdfOpt                = logPdfOpt,
            gradLogPdfOpt            = gradLogPdfOpt,
            burnIn                   = burnIn,
            iterationCount           = 0,
            numberOfRequestedSamples = numberOfRequestedSamples,
            jumpAcceptances          = jumpAcceptances,
            jumpAttempts             = jumpAttempts,
            samples                  = newSamples,
            daHBar                   = daHBar,
            daLogEpsilonBar          = daLogEpsilonBar,
            daMu                     = daMu,
            daStepCount              = daStepCount
          )
        )
      }

    Async[F].ifM(Async[F].delay(iterationCount <= maxIterations))(
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
                   input                    = priorSample,
                   maxIterations            = warmUpSimulationCount,
                   burnIn                   = true,
                   numberOfRequestedSamples = 1,
                   iterationCount           = 0,
                   daMu                     = mu,
                   daLogEpsilonBar          = Math.log(simulationEpsilon)
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
                   input                    = startPt,
                   maxIterations            = simulationsBetweenSamples,
                   numberOfRequestedSamples = numberOfSamples,
                   iterationCount           = 0
                 )
    } yield results

}
