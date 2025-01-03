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

package org.apache.lucene.util.hnsw;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import org.apache.lucene.internal.hppc.IntArrayList;
import org.apache.lucene.internal.hppc.LongArrayList;
import org.apache.lucene.internal.hppc.LongObjectHashMap;
import org.apache.lucene.util.Accountable;
import org.apache.lucene.util.ArrayUtil;
import org.apache.lucene.util.RamUsageEstimator;

/**
 * An {@link HnswGraph} where all nodes and connections are held in memory. This class is used to
 * construct the HNSW graph before it's written to the index.
 */
public final class OnHeapHnswGraph extends HnswGraph implements Accountable {

  private static final int INIT_SIZE = 128;
  private static final int INIT_LEVELS = 15;

  private final AtomicReference<EntryNode> entryNode;

  /** internal graph representation: maps graph nodeId to an array of its neighbors at each level.
   * graph.get(3)[2] returns a {@link NeighborArray} with all neighbors for node 3 on level 2
   */
  LongObjectHashMap<NeighborArray[]> graph;

  // the internal graph representation where the first dimension is node id and second dimension is
  // level
  // e.g. graph[1][2] is all the neighbours of node 1 at level 2
//  private NeighborArray[][] graph;

  // essentially another 2d map which the first dimension is level and second dimension is node id,
  // this is only
  // generated on demand when there's someone calling getNodeOnLevel on a non-zero level
  private LongArrayList[] levelToNodes;
  private int
      lastFreezeSize; // remember the size we are at last time to freeze the graph and generate levelToNodes
  private final AtomicInteger size =
      new AtomicInteger(0); // graph size, which is number of nodes in level 0
  private final AtomicInteger nonZeroLevelSize =
      new AtomicInteger(
          0); // total number of NeighborArrays created that is not on level 0, for now it
  // is only used to account memory usage
  private final AtomicLong maxNodeId = new AtomicLong(-1);
  private final int nsize; // neighbour array size at non-zero level
  private final int nsize0; // neighbour array size at zero level
  private final int maxSize; // holds max possible allowed size for the graph
  private final boolean
      noGrowth; // if an initial size is passed in, we don't expect the graph to grow itself

  // KnnGraphValues iterator members
  private int upto;
  private NeighborArray cur;

  /**
   * ctor
   *
   * @param numNodes number of nodes that will be added to this graph, passing in -1 means unbounded
   *     while passing in a non-negative value will lock the whole graph and disable the graph from
   *     growing itself (you cannot add a node with has id >= numNodes)
   */
  OnHeapHnswGraph(int M, int numNodes) {
    this.entryNode = new AtomicReference<>(new EntryNode(-1, 1));
    // Neighbours' size on upper levels (nsize) and level 0 (nsize0)
    // We allocate extra space for neighbours, but then prune them to keep allowed maximum
    this.nsize = M + 1;
    this.nsize0 = (M * 2 + 1);
    noGrowth = numNodes != -1;
    if (noGrowth == false) {
      numNodes = INIT_SIZE;
      maxSize = Integer.MAX_VALUE;
    } else {
      maxSize = numNodes;
    }
    this.graph = new LongObjectHashMap<>(numNodes);
  }

  /**
   * Returns the {@link NeighborArray} connected to the given node.
   *
   * @param level level of the graph
   * @param node the node whose neighbors are returned, represented as an ordinal on the level 0.
   */
  public NeighborArray getNeighbors(int level, long node) {
    assert graph.containsKey(node);
    assert level < graph.get(node).length;
    return graph.get(node)[level];
  }

  @Override
  public int size() {
    return size.get();
  }

  /**
   * Max node id can be different from {@link #size()} when we have multi-valued vectors,
   * or when we initialize from another graph and nodes get added out of order.
   *
   * @return max node id (inclusive)
   */
  @Override
  public long maxNodeId() {
    return maxNodeId.get();
  }

  /**
   * Add node on the given level. Nodes can be inserted out of order, but it requires that the nodes
   * preceded by the node inserted out of order are eventually added.
   *
   * <p>NOTE: You must add a node starting from the node's top level
   *
   * @param level level to add a node on
   * @param node the node to add, represented as an ordinal on the level 0.
   */
  public void addNode(int level, long node) {
    if (noGrowth && graph.size() == maxSize) {
      throw new IllegalStateException(
          "The graph already has maximum allowed nodes. It is not expect to grow when an initial size is given");
    }
    assert graph.containsKey(node) == false || graph.get(node).length > level
        : "node must be inserted from the top level";
    if (graph.containsKey(node) == false) {
      graph.put(node, new NeighborArray[level + 1]); // assumption: we always call this function from top level
      size.incrementAndGet();
    }
    if (level == 0) {
      graph.get(node)[level] = new NeighborArray(nsize0, true);
    } else {
      graph.get(node)[level] = new NeighborArray(nsize, true);
      nonZeroLevelSize.incrementAndGet();
    }
    maxNodeId.accumulateAndGet(node, Math::max);
  }

  @Override
  public void seek(int level, long targetNode) {
    cur = getNeighbors(level, targetNode);
    upto = -1;
  }

  @Override
  public long nextNeighbor() {
    if (++upto < cur.size()) {
      return cur.nodes()[upto];
    }
    return NO_MORE_DOCS;
  }

  /**
   * Returns the current number of levels in the graph
   *
   * @return the current number of levels in the graph
   */
  @Override
  public int numLevels() {
    return entryNode.get().level + 1;
  }

  /**
   * Returns the graph's current entry node on the top level shown as ordinals of the nodes on 0th
   * level
   *
   * @return the graph's current entry node on the top level
   */
  @Override
  public long entryNode() {
    return entryNode.get().node;
  }

  /**
   * Try to set the entry node if the graph does not have one
   *
   * @return True if the entry node is set to the provided node. False if the entry node already
   *     exists
   */
  public boolean trySetNewEntryNode(long node, int level) {
    EntryNode current = entryNode.get();
    if (current.node == -1) {
      return entryNode.compareAndSet(current, new EntryNode(node, level));
    }
    return false;
  }

  /**
   * Try to promote the provided node to the entry node
   *
   * @param level should be larger than expectedOldLevel
   * @param expectOldLevel is the old entry node level the caller expect to be, the actual graph
   *     level can be different due to concurrent modification
   * @return True if the entry node is set to the provided node. False if expectOldLevel is not the
   *     same as the current entry node level. Even if the provided node's level is still higher
   *     than the current entry node level, the new entry node will not be set and false will be
   *     returned.
   */
  public boolean tryPromoteNewEntryNode(long node, int level, int expectOldLevel) {
    assert level > expectOldLevel;
    EntryNode currentEntry = entryNode.get();
    if (currentEntry.level == expectOldLevel) {
      return entryNode.compareAndSet(currentEntry, new EntryNode(node, level));
    }
    return false;
  }

  /**
   * WARN: calling this method will essentially iterate through all nodes at level 0 (even if you're
   * not getting node at level 0), we have built some caching mechanism such that if graph is not
   * changed only the first non-zero level call will pay the cost. So it is highly NOT recommended
   * to call this method while the graph is still building.
   *
   * <p>NOTE: calling this method while the graph is still building is prohibited
   */
  @Override
  public NodesIterator getNodesOnLevel(int level) {
    // we cannot rely on size() == maxNodeId()+1 to indicate graph build completion
    // TODO: should we add a flag to freeze the graph after building it?
    if (level == 0) {
      return new CollectionNodesIterator(size(), graph.keys().iterator());
    } else {
      generateLevelToNodes();
      return new CollectionNodesIterator(levelToNodes[level]);
    }
  }

  @SuppressWarnings({"unchecked", "rawtypes"})
  private void generateLevelToNodes() {
    if (lastFreezeSize == size()) {
      return;
    }
    int maxLevels = numLevels();
    levelToNodes = new LongArrayList[maxLevels];
    for (int i = 1; i < maxLevels; i++) {
      levelToNodes[i] = new LongArrayList();
    }
    for (long node: graph.keys) {
      for (int i = 1; i < graph.get(node).length; i++) {
        levelToNodes[i].add(node);
      }
    }
    lastFreezeSize = size();
  }

  @Override
  public long ramBytesUsed() {
    // TODO: fix with new graph structure
    long neighborArrayBytes0 =
        (long) nsize0 * (Integer.BYTES + Float.BYTES)
            + RamUsageEstimator.NUM_BYTES_ARRAY_HEADER * 2L
            + RamUsageEstimator.NUM_BYTES_OBJECT_REF * 2L
            + Integer.BYTES * 3;
    long neighborArrayBytes =
        (long) nsize * (Integer.BYTES + Float.BYTES)
            + RamUsageEstimator.NUM_BYTES_ARRAY_HEADER * 2L
            + RamUsageEstimator.NUM_BYTES_OBJECT_REF * 2L
            + Integer.BYTES * 3;
    long total = 0;
    total +=
        size() * (neighborArrayBytes0 + RamUsageEstimator.NUM_BYTES_ARRAY_HEADER)
            + RamUsageEstimator.NUM_BYTES_ARRAY_HEADER; // for graph and level 0;
    total += nonZeroLevelSize.get() * neighborArrayBytes; // for non-zero level
    total += 4 * Integer.BYTES; // all int fields
    total += 1; // field: noGrowth
    total +=
        RamUsageEstimator.NUM_BYTES_OBJECT_REF
            + RamUsageEstimator.NUM_BYTES_OBJECT_HEADER
            + 2 * Integer.BYTES; // field: entryNode
    total += 3L * (Integer.BYTES + RamUsageEstimator.NUM_BYTES_OBJECT_HEADER); // 3 AtomicInteger
    total += RamUsageEstimator.NUM_BYTES_OBJECT_REF; // field: cur
    total += RamUsageEstimator.NUM_BYTES_ARRAY_HEADER; // field: levelToNodes
    if (levelToNodes != null) {
      total +=
          (long) (numLevels() - 1) * RamUsageEstimator.NUM_BYTES_OBJECT_REF; // no cost for level 0
      total +=
          (long) nonZeroLevelSize.get()
              * (RamUsageEstimator.NUM_BYTES_OBJECT_HEADER
                  + RamUsageEstimator.NUM_BYTES_OBJECT_HEADER
                  + Integer.BYTES);
    }
    return total;
  }

  @Override
  public String toString() {
    return "OnHeapHnswGraph(size="
        + size()
        + ", numLevels="
        + numLevels()
        + ", entryNode="
        + entryNode()
        + ")";
  }

  private record EntryNode(long node, int level) {}
}
