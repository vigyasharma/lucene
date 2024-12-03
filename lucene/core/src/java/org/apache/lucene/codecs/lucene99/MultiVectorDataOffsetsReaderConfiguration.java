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

package org.apache.lucene.codecs.lucene99;

import java.io.IOException;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.RandomAccessInput;
import org.apache.lucene.util.packed.DirectMonotonicReader;
import org.apache.lucene.util.packed.DirectMonotonicWriter;

/**
 * Configuration for {@link DirectMonotonicReader} for reading varying length
 * vector data offsets from flat vector storage. This helps read multi-vectors with varying
 * vectors per document.
 */
public class MultiVectorDataOffsetsReaderConfiguration {

  /**
   * Writes out data offsets for each multiVector value per document. Across documents, multiVectors
   * can have variable number of vectors. Data offsets written here are used to read variable sized
   * vector slices per document (ordinal).
   *
   * <p>Within outputMeta the format is as follows:
   *
   * <ul>
   *   <li>MultiVectorDataOffsets encoded by {@link DirectMonotonicWriter}
   * </ul>
   *
   * <p>Within the vectorData the format is as follows:
   *
   * <ul>
   *   <li>MultiVectorDataOffsets encoded by {@link DirectMonotonicWriter}
   * </ul>
   *
   * @param directMonotonicBlockShift block shift to use for DirectMonotonicWriter
   * @param outputMeta the outputMeta
   * @param vectorData the vector data
   * @param multiVectorDataOffsets array holding data offsets for each multiVector
   * @throws IOException thrown when writing data fails to either output
   */
  public static void writeStoredMeta(
      int directMonotonicBlockShift,
      IndexOutput outputMeta,
      IndexOutput vectorData,
      long[] multiVectorDataOffsets)
      throws IOException {
    long start = vectorData.getFilePointer();
    outputMeta.writeLong(start); // vector data offset
    outputMeta.writeVInt(directMonotonicBlockShift); // block shift
    final int numValues = multiVectorDataOffsets.length;
    outputMeta.writeVInt(numValues); // count
    final DirectMonotonicWriter dataOffsetsWriter =
        DirectMonotonicWriter.getInstance(
            outputMeta, vectorData, numValues, directMonotonicBlockShift);
    for (long offset : multiVectorDataOffsets) {
      dataOffsetsWriter.add(offset);
    }
    dataOffsetsWriter.finish();
    outputMeta.writeLong(vectorData.getFilePointer() - start); // vector data length
  }

  /**
   * Reads in the necessary fields stored in the outputMeta to configure {@link
   * DirectMonotonicReader} over MultiVectorDataOffsets
   *
   * @param inputMeta the inputMeta, previously written to via {@link #writeStoredMeta(int,
   *     IndexOutput, IndexOutput, long[])}
   * @return the configuration required to read multiVector data offsets
   * @throws IOException thrown when reading data fails
   */
  public static MultiVectorDataOffsetsReaderConfiguration fromStoredMeta(IndexInput inputMeta)
      throws IOException {
    long addressesOffset = inputMeta.readLong();
    int blockShift = inputMeta.readVInt();
    int numValues = inputMeta.readVInt();
    DirectMonotonicReader.Meta meta =
        DirectMonotonicReader.loadMeta(inputMeta, numValues, blockShift);
    long addressesLength = inputMeta.readLong();
    return new MultiVectorDataOffsetsReaderConfiguration(addressesOffset, addressesLength, meta);
  }

  final long addressesOffset, addressesLength;
  final DirectMonotonicReader.Meta meta;

  MultiVectorDataOffsetsReaderConfiguration(
      long addressesOffset, long addressesLength, DirectMonotonicReader.Meta meta) {
    this.addressesOffset = addressesOffset;
    this.addressesLength = addressesLength;
    this.meta = meta;
  }

  /**
   * @param dataIn the IndexInput to read data from
   * @return the DirectMonotonicReader stored values
   * @throws IOException thrown when reading data fails
   */
  public DirectMonotonicReader getDirectMonotonicReader(IndexInput dataIn) throws IOException {
    final RandomAccessInput addressesData =
        dataIn.randomAccessSlice(addressesOffset, addressesLength);
    return DirectMonotonicReader.getInstance(meta, addressesData);
  }
}
