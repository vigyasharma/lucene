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
import org.apache.lucene.document.KnnByteVectorField;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.util.ArrayUtil;
import org.apache.lucene.util.hnsw.HnswUtil;

/**
 * This class provides access to per-document floating point vector values indexed as {@link
 * KnnByteVectorField}.
 *
 * @lucene.experimental
 */
public abstract class ByteVectorValues extends KnnVectorValues {

  /** Sole constructor */
  protected ByteVectorValues() {}

  /**
   * Returns the single specific vector value corresponding to an (ordinal, subOrdinal) pair.
   * @return vector value
   */
  public byte[] vectorValue(int ordinal, int subOrdinal) throws IOException {
    byte[] packedValue = allVectorValues(ordinal);
    if (packedValue.length < (subOrdinal + 1) * dimension()) {
      throw new ArrayIndexOutOfBoundsException("requested subOrdinal [" + subOrdinal + "] " +
          "for vector ordinal [" + ordinal + "] is out of range.");
    }
    // optimize for single valued vector fields
    if (packedValue.length == dimension()) {
      return packedValue;
    }
    return ArrayUtil.copyOfSubArray(vectorValue(ordinal), subOrdinal, dimension());
  }

  /**
   * Returns the single specific vector value corresponding to a nodeId.
   * @return vector value
   */
  public byte[] vectorValue(long nodeId) throws IOException {
    return vectorValue(HnswUtil.ordinal(nodeId), HnswUtil.subOrdinal(nodeId));
  }

  /** Returns the single vector value for subOrdinal 0 for provided ordinal */
  public byte[] vectorValue(int ord) throws IOException {
    return vectorValue(ord, 0);
  }

  /**
   * Returns all vector values for a given ordinal.
   * <p>
   * Each graph nodeId is a long with ordinal and subOrdinal values packed in
   * LSB and MSB respectively. This API returns all vector values for the ordinal represented
   * by the 32 LSB of nodeId. The (subOrdinal) vector values are concatenated into a
   * single byte[] array. For single valued vector fields, this API returns the same
   * value as {@link ByteVectorValues#allVectorValues(int ord)} 
   * @return all vector values packed in a single array
   */
  public abstract byte[] allVectorValues(int ordinal) throws IOException;

  /** Returns all vector values for the ordinal of a given nodeId */
  public byte[] allVectorValues(long nodeId) throws IOException {
    return allVectorValues(HnswUtil.ordinal(nodeId));
  }

  @Override
  public abstract ByteVectorValues copy() throws IOException;

  /**
   * Checks the Vector Encoding of a field
   *
   * @throws IllegalStateException if {@code field} has vectors, but using a different encoding
   * @lucene.internal
   * @lucene.experimental
   */
  public static void checkField(LeafReader in, String field) {
    FieldInfo fi = in.getFieldInfos().fieldInfo(field);
    if (fi != null && fi.hasVectorValues() && fi.getVectorEncoding() != VectorEncoding.BYTE) {
      throw new IllegalStateException(
          "Unexpected vector encoding ("
              + fi.getVectorEncoding()
              + ") for field "
              + field
              + "(expected="
              + VectorEncoding.BYTE
              + ")");
    }
  }

  /**
   * Return a {@link VectorScorer} for the given query vector.
   *
   * @param query the query vector
   * @return a {@link VectorScorer} instance or null
   */
  public VectorScorer scorer(byte[] query) throws IOException {
    throw new UnsupportedOperationException();
  }

  @Override
  public VectorEncoding getEncoding() {
    return VectorEncoding.BYTE;
  }

  /**
   * Creates a {@link ByteVectorValues} from a list of byte arrays.
   *
   * @param vectors the list of byte arrays
   * @param dim the dimension of the vectors
   * @return a {@link ByteVectorValues} instancec
   */
  public static ByteVectorValues fromBytes(List<byte[]> vectors, int dim) {
    return new ByteVectorValues() {
      @Override
      public int size() {
        return vectors.size();
      }

      @Override
      public int dimension() {
        return dim;
      }

      @Override
      public byte[] allVectorValues(int targetOrd) {
        return vectors.get(targetOrd);
      }

      @Override
      public ByteVectorValues copy() {
        return this;
      }

      @Override
      public DocIndexIterator iterator() {
        return createDenseIterator();
      }
    };
  }
}
