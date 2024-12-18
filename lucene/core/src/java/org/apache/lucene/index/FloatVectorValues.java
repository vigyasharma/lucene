/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.lucene.index;

import java.io.IOException;
import java.util.List;

import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.util.ArrayUtil;
import org.apache.lucene.util.hnsw.HnswUtil;

/**
 * This class provides access to per-document floating point vector values indexed as {@link
 * KnnFloatVectorField}.
 *
 * @lucene.experimental
 */
public abstract class FloatVectorValues extends KnnVectorValues {

  /** Sole constructor */
  protected FloatVectorValues() {}

  /**
   * Returns the single specific vector value corresponding to an (ordinal, subOrdinal) pair.
   * @return vector value
   */
  public float[] vectorValue(int ordinal, int subOrdinal) throws IOException {
    float[] packedValue = allVectorValues(ordinal);
    if (packedValue.length < (subOrdinal + 1) * dimension()) {
      throw new ArrayIndexOutOfBoundsException("requested subOrdinal [" + subOrdinal + "] " +
          "for vector ordinal [" + ordinal + "] is out of range.");
    }
    // optimize for single valued vector field
    if (packedValue.length == dimension()) {
      return packedValue;
    }
    return ArrayUtil.copyOfSubArray(vectorValue(ordinal), subOrdinal, dimension());
  }

  /**
   * Returns the single specific vector value corresponding to a nodeId.
   * @return vector value
   */
  public float[] vectorValue(long nodeId) throws IOException {
    return vectorValue(HnswUtil.ordinal(nodeId), HnswUtil.subOrdinal(nodeId));
  }

  /** Returns the single vector value for subOrdinal 0 for provided ordinal */
  public float[] vectorValue(int ord) throws IOException {
    return vectorValue(ord, 0);
  }

  /**
   * Returns all vector values for a given ordinal.
   * <p>
   * Each graph nodeId is a long with ordinal and subOrdinal values packed in
   * LSB and MSB respectively. This API returns all vector values for the ordinal represented
   * by the 32 LSB of nodeId. The (subOrdinal) vector values are concatenated into a
   * single float[] array. For single valued vector fields, this API returns the same
   * value as {@link FloatVectorValues#vectorValue(int ord)}
   * @return the vector value
   */
  public abstract float[] allVectorValues(int ordinal) throws IOException;

  /** Returns all vector values for the ordinal of a given nodeId */
  public float[] allVectorValues(long nodeId) throws IOException {
    return allVectorValues(HnswUtil.ordinal(nodeId));
  }

  @Override
  public abstract FloatVectorValues copy() throws IOException;

  /**
   * Checks the Vector Encoding of a field
   *
   * @throws IllegalStateException if {@code field} has vectors, but using a different encoding
   * @lucene.internal
   * @lucene.experimental
   */
  public static void checkField(LeafReader in, String field) {
    FieldInfo fi = in.getFieldInfos().fieldInfo(field);
    if (fi != null && fi.hasVectorValues() && fi.getVectorEncoding() != VectorEncoding.FLOAT32) {
      throw new IllegalStateException(
          "Unexpected vector encoding ("
              + fi.getVectorEncoding()
              + ") for field "
              + field
              + "(expected="
              + VectorEncoding.FLOAT32
              + ")");
    }
  }

  /**
   * Return a {@link VectorScorer} for the given query vector and the current {@link
   * FloatVectorValues}.
   *
   * @param target the query vector
   * @return a {@link VectorScorer} instance or null
   */
  public VectorScorer scorer(float[] target) throws IOException {
    throw new UnsupportedOperationException();
  }

  @Override
  public VectorEncoding getEncoding() {
    return VectorEncoding.FLOAT32;
  }

  /**
   * Creates a {@link FloatVectorValues} from a list of float arrays.
   *
   * @param vectors the list of float arrays
   * @param dim the dimension of the vectors
   * @return a {@link FloatVectorValues} instance
   */
  public static FloatVectorValues fromFloats(List<float[]> vectors, int dim) {
    return new FloatVectorValues() {
      @Override
      public int size() {
        return vectors.size();
      }

      @Override
      public int dimension() {
        return dim;
      }

      @Override
      public float[] allVectorValues(int targetOrd) {
        return vectors.get(targetOrd);
      }

      @Override
      public FloatVectorValues copy() {
        return this;
      }

      @Override
      public DocIndexIterator iterator() {
        return createDenseIterator();
      }
    };
  }
}
