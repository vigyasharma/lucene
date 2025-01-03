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

import org.apache.lucene.util.LongHeap;
import org.apache.lucene.util.NumericUtils;
import org.apache.lucene.util.PriorityQueue;

import java.util.Iterator;
import java.util.Objects;

/**
 * NeighborQueue uses a {@link PriorityQueue} to store lists of arcs in an HNSW graph, represented as a
 * neighbor node id with an associated score.
 * The queue provides both fixed-size and unbounded operations via {@link #insertWithOverflow(long, float)}
 * and {@link #add(long, float)}, and provides MIN and MAX heap subclasses.
 */
public class NeighborQueue {

  /** Stores the score and nodeId for an arc in the HNSW graph. */
  private record ScoreNode (int score, long node) {
    ScoreNode(float score, long node) {
      this(NumericUtils.floatToSortableInt(score), node);
    }

    @Override
    public boolean equals(Object obj) {
      if (obj instanceof ScoreNode == false) {
        return false;
      }
      ScoreNode other = (ScoreNode) obj;
      return other.score == this.score && other.node == this.node;
    }
  }

  private final PriorityQueue<ScoreNode> heap;

  // Used to track the number of neighbors visited during a single graph traversal
  private int visitedCount;
  // Whether the search stopped early because it reached the visited nodes limit
  private boolean incomplete;

  public NeighborQueue(int initialSize, boolean maxHeap) {
    this.heap = new PriorityQueue<ScoreNode>(initialSize) {
      @Override
      protected boolean lessThan(ScoreNode a, ScoreNode b) {
        boolean minHeapComparison = (a.score < b.score || (a.score == b.score && a.node < b.node));
        if (maxHeap == false) {
          return minHeapComparison;
        }
        return !minHeapComparison;
      }
    };
  }

  /**
   * @return the number of elements in the heap
   */
  public int size() {
    return heap.size();
  }

  /**
   * Adds a new graph arc, extending the storage as needed.
   *
   * @param newNode the neighbor node id
   * @param newScore the score of the neighbor, relative to some other node
   */
  public void add(long newNode, float newScore) {
    heap.add(new ScoreNode(newScore, newNode));
  }

  /**
   * If the heap is not full (size is less than the initialSize provided to the constructor), adds a
   * new node-and-score element. If the heap is full, compares the score against the current top
   * score, and replaces the top element if newScore is better than (greater than unless the heap is
   * reversed), the current top score.
   *
   * @param newNode the neighbor node id
   * @param newScore the score of the neighbor, relative to some other node
   */
  public boolean insertWithOverflow(long newNode, float newScore) {
    ScoreNode ele = new ScoreNode(newScore, newNode);
    ScoreNode res = heap.insertWithOverflow(ele);
    return res == null || res.equals(ele) == false;  // true if ele was added successfully
  }

  /** Removes the top element and returns its node id. */
  public long pop() {
    return Objects.requireNonNull(heap.pop(), "The heap is empty").node;
  }

  public long[] nodes() {
    int size = size();
    long[] nodes = new long[size];
    Iterator<ScoreNode> it = heap.iterator();
    int i = 0;
    while (it.hasNext()) {
      nodes[i++]  = it.next().node;
    }
    return nodes;
  }

  /** Returns the top element's node id. */
  public long topNode() {
    return Objects.requireNonNull(heap.top(), "The heap is empty").node;
  }

  /**
   * Returns the top element's node score. For the min heap this is the minimum score. For the max
   * heap this is the maximum score.
   */
  public float topScore() {
    int score = Objects.requireNonNull(heap.top(), "The heap is empty").score;
    return NumericUtils.sortableIntToFloat(score);
  }

  public void clear() {
    heap.clear();
    visitedCount = 0;
    incomplete = false;
  }

  public int visitedCount() {
    return visitedCount;
  }

  public void setVisitedCount(int visitedCount) {
    this.visitedCount = visitedCount;
  }

  public boolean incomplete() {
    return incomplete;
  }

  public void markIncomplete() {
    this.incomplete = true;
  }

  @Override
  public String toString() {
    return "Neighbors[" + heap.size() + "]";
  }
}
