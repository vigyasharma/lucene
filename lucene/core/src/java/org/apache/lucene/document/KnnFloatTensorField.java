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

package org.apache.lucene.document;

import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.TensorSimilarityFunction;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.util.VectorUtil;

import java.util.List;
import java.util.Objects;

/**
 * A field that contains multiple (or none) floating-point numeric vectors for each document.
 * Similar to {@link KnnFloatVectorField}, vectors are dense - that is, every dimension of a vector
 * contains an explicit value, stored packed into an array (of type float[]) whose length is
 * the vector dimension. <TODO: add more info>.
 *
 * @lucene.experimental
 */
// * A field that contains a single floating-point numeric vector (or none) for each document. Vectors
// * are dense - that is, every dimension of a vector contains an explicit value, stored packed into
// * an array (of type float[]) whose length is the vector dimension. Values can be retrieved using
// * {@link FloatVectorValues}, which is a forward-only docID-based iterator and also offers
// * random-access by dense ordinal (not docId). {@link VectorSimilarityFunction} may be used to
// * compare vectors at query time (for example as part of result ranking). A {@link
// * KnnFloatTensorField} may be associated with a search similarity function defining the metric used
// * for nearest-neighbor search among vectors of that field.
// *
// * @lucene.experimental
// */
public class KnnFloatTensorField extends Field {

  private static FieldType createType(List<float[]> t, TensorSimilarityFunction similarityFunction) {
    if (t == null) {
      throw new IllegalArgumentException("tensor value must not be null");
    }
    int degree = t.size();
    if (degree == 0) {
      throw new IllegalArgumentException("cannot index an empty tensor");
    }
    if (similarityFunction == null) {
      throw new IllegalArgumentException("similarity function must not be null");
    }
    for (int i = 0; i < t.size(); i++) {
      if (t.get(i).length == 0) {
        throw new IllegalArgumentException("empty vector found at index (" + i + "). Tensor cannot have empty vectors");
      }
    }
    FieldType type = new FieldType();
    type.setTensorAttributes(degree, VectorEncoding.FLOAT32, similarityFunction);
    type.freeze();
    return type;
  }

  /**
   * A convenience method for creating a tensor field type.
   *
   * @param degree Number of vectors in each tensor
   * @param similarityFunction a function defining tensor proximity.
   * @throws IllegalArgumentException if any parameter is null, or has dimension &gt; 1024.
   */
  public static FieldType createFieldType(
      int degree, VectorSimilarityFunction similarityFunction) {
    FieldType type = new FieldType();
    type.setVectorAttributes(degree, VectorEncoding.FLOAT32, similarityFunction);
    type.freeze();
    return type;
  }

  /**
   * Create a new vector query for the provided field targeting the float vector
   *
   * @param field The field to query
   * @param queryVector The float vector target
   * @param k The number of nearest neighbors to gather
   * @return A new vector query
   */
  public static Query newVectorQuery(String field, float[] queryVector, int k) {
    return new KnnFloatVectorQuery(field, queryVector, k);
  }

  /**
   * Creates a numeric vector field. Fields are single-valued: each document has either one value or
   * no value. Vectors of a single field share the same dimension and similarity function. Note that
   * some vector similarities (like {@link VectorSimilarityFunction#DOT_PRODUCT}) require values to
   * be unit-length, which can be enforced using {@link VectorUtil#l2normalize(float[])}.
   *
   * @param name field name
   * @param vector value
   * @param similarityFunction a function defining vector proximity.
   * @throws IllegalArgumentException if any parameter is null, or the vector is empty or has
   *     dimension &gt; 1024.
   */
  public KnnFloatTensorField(
      String name, float[] vector, VectorSimilarityFunction similarityFunction) {
    super(name, createType(vector, similarityFunction));
    fieldsData = VectorUtil.checkFinite(vector); // null check done above
  }

  /**
   * Creates a numeric vector field with the default EUCLIDEAN_HNSW (L2) similarity. Fields are
   * single-valued: each document has either one value or no value. Vectors of a single field share
   * the same dimension and similarity function.
   *
   * @param name field name
   * @param vector value
   * @throws IllegalArgumentException if any parameter is null, or the vector is empty or has
   *     dimension &gt; 1024.
   */
  public KnnFloatTensorField(String name, float[] vector) {
    this(name, vector, VectorSimilarityFunction.EUCLIDEAN);
  }

  /**
   * Creates a numeric vector field. Fields are single-valued: each document has either one value or
   * no value. Vectors of a single field share the same dimension and similarity function.
   *
   * @param name field name
   * @param vector value
   * @param fieldType field type
   * @throws IllegalArgumentException if any parameter is null, or the vector is empty or has
   *     dimension &gt; 1024.
   */
  public KnnFloatTensorField(String name, float[] vector, FieldType fieldType) {
    super(name, fieldType);
    if (fieldType.vectorEncoding() != VectorEncoding.FLOAT32) {
      throw new IllegalArgumentException(
          "Attempt to create a vector for field "
              + name
              + " using float[] but the field encoding is "
              + fieldType.vectorEncoding());
    }
    Objects.requireNonNull(vector, "vector value must not be null");
    if (vector.length != fieldType.vectorDimension()) {
      throw new IllegalArgumentException(
          "The number of vector dimensions does not match the field type");
    }
    fieldsData = VectorUtil.checkFinite(vector);
  }

  /** Return the vector value of this field */
  public float[] vectorValue() {
    return (float[]) fieldsData;
  }

  /**
   * Set the vector value of this field
   *
   * @param value the value to set; must not be null, and length must match the field type
   */
  public void setVectorValue(float[] value) {
    if (value == null) {
      throw new IllegalArgumentException("value must not be null");
    }
    if (value.length != type.vectorDimension()) {
      throw new IllegalArgumentException(
          "value length " + value.length + " must match field dimension " + type.vectorDimension());
    }
    fieldsData = value;
  }
}