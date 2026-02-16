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
package thylacine.util

import org.ejml.data.DMatrixRMaj
import org.ejml.dense.row.CommonOps_DDRM

private[thylacine] object LinearAlgebra {

  def zeros(rows: Int, cols: Int): DMatrixRMaj =
    new DMatrixRMaj(rows, cols)

  def fromArray2D(data: Array[Array[Double]]): DMatrixRMaj =
    new DMatrixRMaj(data)

  def toArray2D(m: DMatrixRMaj): Array[Array[Double]] = {
    val result = Array.ofDim[Double](m.numRows, m.numCols)
    for (i <- 0 until m.numRows; j <- 0 until m.numCols) {
      result(i)(j) = m.get(i, j)
    }
    result
  }

  def invert(m: DMatrixRMaj): DMatrixRMaj = {
    val result = new DMatrixRMaj(m.numRows, m.numCols)
    if (!CommonOps_DDRM.invert(m, result)) {
      throw new RuntimeException(s"Matrix inversion failed: ${m.numRows}x${m.numCols} matrix may be singular")
    }
    result
  }

  def determinant(m: DMatrixRMaj): Double =
    CommonOps_DDRM.det(m)

  def transpose(m: DMatrixRMaj): DMatrixRMaj = {
    val result = new DMatrixRMaj(m.numCols, m.numRows)
    CommonOps_DDRM.transpose(m, result)
    result
  }

  def multiply(a: DMatrixRMaj, b: DMatrixRMaj): DMatrixRMaj = {
    val result = new DMatrixRMaj(a.numRows, b.numCols)
    CommonOps_DDRM.mult(a, b, result)
    result
  }

  def multiplyMV(m: DMatrixRMaj, v: Array[Double]): Array[Double] = {
    val vMatrix = new DMatrixRMaj(v.length, 1)
    vMatrix.setData(v)
    val result = new DMatrixRMaj(m.numRows, 1)
    CommonOps_DDRM.mult(m, vMatrix, result)
    result.getData.take(m.numRows)
  }

  def multiplyMV(m: DMatrixRMaj, v: DMatrixRMaj): DMatrixRMaj = {
    val result = new DMatrixRMaj(m.numRows, 1)
    CommonOps_DDRM.mult(m, v, result)
    result
  }

  // Solves A * x = b for x (equivalent to Breeze's A \ b)
  def solve(a: DMatrixRMaj, b: Array[Double]): Array[Double] = {
    val bMatrix = new DMatrixRMaj(b.length, 1)
    bMatrix.setData(b)
    val result = new DMatrixRMaj(a.numCols, 1)
    if (!CommonOps_DDRM.solve(a, bMatrix, result)) {
      throw new RuntimeException(
        s"Linear system solve failed: ${a.numRows}x${a.numCols} system with ${b.length}-element RHS"
      )
    }
    result.getData.take(a.numCols)
  }

  def solve(a: DMatrixRMaj, b: DMatrixRMaj): DMatrixRMaj = {
    val result = new DMatrixRMaj(a.numCols, b.numCols)
    if (!CommonOps_DDRM.solve(a, b, result)) {
      throw new RuntimeException(
        s"Linear system solve failed: ${a.numRows}x${a.numCols} system with ${b.numRows}x${b.numCols} RHS"
      )
    }
    result
  }

  // Computes v^T * M * v (quadratic form)
  def quadraticForm(v: Array[Double], m: DMatrixRMaj): Double = {
    val vMatrix = new DMatrixRMaj(v.length, 1)
    vMatrix.setData(v)
    val temp = new DMatrixRMaj(m.numRows, 1)
    CommonOps_DDRM.mult(m, vMatrix, temp)
    CommonOps_DDRM.dot(vMatrix, temp)
  }

  def quadraticForm(v: DMatrixRMaj, m: DMatrixRMaj): Double = {
    val temp = new DMatrixRMaj(m.numRows, 1)
    CommonOps_DDRM.mult(m, v, temp)
    CommonOps_DDRM.dot(v, temp)
  }

  def add(a: DMatrixRMaj, b: DMatrixRMaj): DMatrixRMaj = {
    val result = new DMatrixRMaj(a.numRows, a.numCols)
    CommonOps_DDRM.add(a, b, result)
    result
  }

  def subtract(a: DMatrixRMaj, b: DMatrixRMaj): DMatrixRMaj = {
    val result = new DMatrixRMaj(a.numRows, a.numCols)
    CommonOps_DDRM.subtract(a, b, result)
    result
  }

  def scale(m: DMatrixRMaj, s: Double): DMatrixRMaj = {
    val result = new DMatrixRMaj(m.numRows, m.numCols)
    CommonOps_DDRM.scale(s, m, result)
    result
  }

  def divide(m: DMatrixRMaj, s: Double): DMatrixRMaj =
    scale(m, 1.0 / s)

  // Computes (M + M^T) / 2 to ensure symmetry
  def symmetrize(m: DMatrixRMaj): DMatrixRMaj = {
    val mt  = transpose(m)
    val sum = add(m, mt)
    scale(sum, 0.5)
  }

  def arrayToColumnVector(arr: Array[Double]): DMatrixRMaj = {
    val result = new DMatrixRMaj(arr.length, 1)
    result.setData(arr)
    result
  }

  def columnVectorToArray(v: DMatrixRMaj): Array[Double] =
    v.getData.take(v.numRows)
}
