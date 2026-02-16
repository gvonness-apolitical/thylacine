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
package thylacine.model.core

import thylacine.model.core.values.{ MatrixContainer, VectorContainer }

private[thylacine] case class RecordedData(
  data: VectorContainer,
  covariance: MatrixContainer,
  validated: Boolean = false
) extends CanValidate[RecordedData] {
  if (!validated) {
    require(covariance.rowTotalNumber == covariance.columnTotalNumber, "Covariance matrix must be square")
    require(data.dimension == covariance.rowTotalNumber, "Data dimension must match covariance dimension")
  }

  private[thylacine] val dimension: Int = data.dimension

  override private[thylacine] lazy val getValidated: RecordedData =
    if (validated) {
      this
    } else {
      RecordedData(
        data.getValidated,
        covariance.getValidated,
        validated = true
      )
    }
}

private[thylacine] object RecordedData {

  private[thylacine] def apply(
    values: VectorContainer,
    symmetricConfidenceIntervals: VectorContainer
  ): RecordedData = {
    val validatedValues: VectorContainer = values.getValidated
    val validatedConfidenceIntervals: VectorContainer =
      symmetricConfidenceIntervals.getValidated

    require(
      validatedValues.dimension == validatedConfidenceIntervals.dimension,
      "Values and confidence intervals must have the same dimension"
    )
    require(validatedConfidenceIntervals.values.values.forall(_ > 0), "All confidence intervals must be positive")

    RecordedData(
      data = validatedValues,
      covariance = MatrixContainer(
        values            = validatedConfidenceIntervals.values.map(i => (i._1, i._1) -> Math.pow(i._2 / 2, 2)),
        rowTotalNumber    = values.dimension,
        columnTotalNumber = values.dimension,
        validated         = true
      ),
      validated = true
    )
  }

  def apply(values: Vector[Double], symmetricConfidenceIntervals: Vector[Double]): RecordedData =
    apply(VectorContainer(values), VectorContainer(symmetricConfidenceIntervals))
}
