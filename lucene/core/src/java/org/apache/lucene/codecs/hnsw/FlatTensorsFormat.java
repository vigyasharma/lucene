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

package org.apache.lucene.codecs.hnsw;

import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;

import java.io.IOException;

/**
 * Encodes/decodes per-document tensors and provides a scoring interface for the flat stored tensors
 *
 * @lucene.experimental
 */
public abstract class FlatTensorsFormat {

  /** Sole constructor */
  protected FlatTensorsFormat() {}

  /** Returns a {@link FlatTensorsWriter} to write the tensors to the index. */
  public abstract FlatTensorsWriter fieldsWriter(SegmentWriteState state) throws IOException;

  /** Returns a {@link FlatTensorsReader} to read the tensors from the index. */
  public abstract FlatTensorsReader fieldsReader(SegmentReadState state) throws IOException;
}
