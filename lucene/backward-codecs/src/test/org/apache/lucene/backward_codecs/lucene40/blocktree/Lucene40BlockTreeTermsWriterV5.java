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
package org.apache.lucene.backward_codecs.lucene40.blocktree;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import org.apache.lucene.backward_codecs.store.EndiannessReverserUtil;
import org.apache.lucene.codecs.BlockTermState;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.FieldsConsumer;
import org.apache.lucene.codecs.NormsProducer;
import org.apache.lucene.codecs.PostingsWriterBase;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.Fields;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.store.ByteArrayDataOutput;
import org.apache.lucene.store.ByteBuffersDataOutput;
import org.apache.lucene.store.DataOutput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.ArrayUtil;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.BytesRefBuilder;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.IntsRefBuilder;
import org.apache.lucene.util.StringHelper;
import org.apache.lucene.util.compress.LZ4;
import org.apache.lucene.util.compress.LowercaseAsciiCompression;
import org.apache.lucene.util.fst.ByteSequenceOutputs;
import org.apache.lucene.util.fst.BytesRefFSTEnum;
import org.apache.lucene.util.fst.FST;
import org.apache.lucene.util.fst.FSTCompiler;
import org.apache.lucene.util.fst.Util;

/**
 * Writer for {@link Lucene40BlockTreeTermsReader} prior to the refactoring that separated metadata
 * to its own file.
 *
 * @see Lucene40BlockTreeTermsReader#VERSION_META_FILE
 */
public final class Lucene40BlockTreeTermsWriterV5 extends FieldsConsumer {

  /**
   * Suggested default value for the {@code minItemsInBlock} parameter to {@link
   * #Lucene40BlockTreeTermsWriterV5(SegmentWriteState,PostingsWriterBase,int,int)}.
   */
  public static final int DEFAULT_MIN_BLOCK_SIZE = 25;

  /**
   * Suggested default value for the {@code maxItemsInBlock} parameter to {@link
   * #Lucene40BlockTreeTermsWriterV5(SegmentWriteState,PostingsWriterBase,int,int)}.
   */
  public static final int DEFAULT_MAX_BLOCK_SIZE = 48;

  private static final int VERSION_CURRENT = Lucene40BlockTreeTermsReader.VERSION_META_FILE - 1;

  // public static boolean DEBUG = false;
  // public static boolean DEBUG2 = false;

  // private final static boolean SAVE_DOT_FILES = false;

  private final IndexOutput termsOut;
  private final IndexOutput indexOut;
  final int maxDoc;
  final int minItemsInBlock;
  final int maxItemsInBlock;

  final PostingsWriterBase postingsWriter;
  final FieldInfos fieldInfos;

  private static class FieldMetaData {
    public final FieldInfo fieldInfo;
    public final BytesRef rootCode;
    public final long numTerms;
    public final long indexStartFP;
    public final long sumTotalTermFreq;
    public final long sumDocFreq;
    public final int docCount;
    public final BytesRef minTerm;
    public final BytesRef maxTerm;

    public FieldMetaData(
        FieldInfo fieldInfo,
        BytesRef rootCode,
        long numTerms,
        long indexStartFP,
        long sumTotalTermFreq,
        long sumDocFreq,
        int docCount,
        BytesRef minTerm,
        BytesRef maxTerm) {
      assert numTerms > 0;
      this.fieldInfo = fieldInfo;
      assert rootCode != null : "field=" + fieldInfo.name + " numTerms=" + numTerms;
      this.rootCode = rootCode;
      this.indexStartFP = indexStartFP;
      this.numTerms = numTerms;
      this.sumTotalTermFreq = sumTotalTermFreq;
      this.sumDocFreq = sumDocFreq;
      this.docCount = docCount;
      this.minTerm = minTerm;
      this.maxTerm = maxTerm;
    }
  }

  private final List<FieldMetaData> fields = new ArrayList<>();

  /**
   * Create a new writer. The number of items (terms or sub-blocks) per block will aim to be between
   * minItemsPerBlock and maxItemsPerBlock, though in some cases the blocks may be smaller than the
   * min.
   */
  public Lucene40BlockTreeTermsWriterV5(
      SegmentWriteState state,
      PostingsWriterBase postingsWriter,
      int minItemsInBlock,
      int maxItemsInBlock)
      throws IOException {
    validateSettings(minItemsInBlock, maxItemsInBlock);

    this.minItemsInBlock = minItemsInBlock;
    this.maxItemsInBlock = maxItemsInBlock;

    this.maxDoc = state.segmentInfo.maxDoc();
    this.fieldInfos = state.fieldInfos;
    this.postingsWriter = postingsWriter;

    final String termsName =
        IndexFileNames.segmentFileName(
            state.segmentInfo.name,
            state.segmentSuffix,
            Lucene40BlockTreeTermsReader.TERMS_EXTENSION);
    termsOut = EndiannessReverserUtil.createOutput(state.directory, termsName, state.context);
    boolean success = false;
    IndexOutput indexOut = null;
    try {
      CodecUtil.writeIndexHeader(
          termsOut,
          Lucene40BlockTreeTermsReader.TERMS_CODEC_NAME,
          VERSION_CURRENT,
          state.segmentInfo.getId(),
          state.segmentSuffix);

      final String indexName =
          IndexFileNames.segmentFileName(
              state.segmentInfo.name,
              state.segmentSuffix,
              Lucene40BlockTreeTermsReader.TERMS_INDEX_EXTENSION);
      indexOut = EndiannessReverserUtil.createOutput(state.directory, indexName, state.context);
      CodecUtil.writeIndexHeader(
          indexOut,
          Lucene40BlockTreeTermsReader.TERMS_INDEX_CODEC_NAME,
          VERSION_CURRENT,
          state.segmentInfo.getId(),
          state.segmentSuffix);
      // segment = state.segmentInfo.name;

      postingsWriter.init(termsOut, state); // have consumer write its format/header

      this.indexOut = indexOut;
      success = true;
    } finally {
      if (!success) {
        IOUtils.closeWhileHandlingException(termsOut, indexOut);
      }
    }
  }

  /** Writes the terms file trailer. */
  private void writeTrailer(IndexOutput out, long dirStart) throws IOException {
    out.writeLong(dirStart);
  }

  /** Writes the index file trailer. */
  private void writeIndexTrailer(IndexOutput indexOut, long dirStart) throws IOException {
    indexOut.writeLong(dirStart);
  }

  /** Throws {@code IllegalArgumentException} if any of these settings is invalid. */
  public static void validateSettings(int minItemsInBlock, int maxItemsInBlock) {
    if (minItemsInBlock <= 1) {
      throw new IllegalArgumentException("minItemsInBlock must be >= 2; got " + minItemsInBlock);
    }
    if (minItemsInBlock > maxItemsInBlock) {
      throw new IllegalArgumentException(
          "maxItemsInBlock must be >= minItemsInBlock; got maxItemsInBlock="
              + maxItemsInBlock
              + " minItemsInBlock="
              + minItemsInBlock);
    }
    if (2 * (minItemsInBlock - 1) > maxItemsInBlock) {
      throw new IllegalArgumentException(
          "maxItemsInBlock must be at least 2*(minItemsInBlock-1); got maxItemsInBlock="
              + maxItemsInBlock
              + " minItemsInBlock="
              + minItemsInBlock);
    }
  }

  @Override
  public void write(Fields fields, NormsProducer norms) throws IOException {
    // if (DEBUG) System.out.println("\nBTTW.write seg=" + segment);

    String lastField = null;
    for (String field : fields) {
      assert lastField == null || lastField.compareTo(field) < 0;
      lastField = field;

      // if (DEBUG) System.out.println("\nBTTW.write seg=" + segment + " field=" + field);
      Terms terms = fields.terms(field);
      if (terms == null) {
        continue;
      }

      TermsEnum termsEnum = terms.iterator();
      TermsWriter termsWriter = new TermsWriter(fieldInfos.fieldInfo(field));
      while (true) {
        BytesRef term = termsEnum.next();
        // if (DEBUG) System.out.println("BTTW: next term " + term);

        if (term == null) {
          break;
        }

        // if (DEBUG) System.out.println("write field=" + fieldInfo.name + " term=" +
        // brToString(term));
        termsWriter.write(term, termsEnum, norms);
      }

      termsWriter.finish();

      // if (DEBUG) System.out.println("\nBTTW.write done seg=" + segment + " field=" + field);
    }
  }

  static long encodeOutput(long fp, boolean hasTerms, boolean isFloor) {
    assert fp < (1L << 62);
    return (fp << 2)
        | (hasTerms ? Lucene40BlockTreeTermsReader.OUTPUT_FLAG_HAS_TERMS : 0)
        | (isFloor ? Lucene40BlockTreeTermsReader.OUTPUT_FLAG_IS_FLOOR : 0);
  }

  private static class PendingEntry {
    public final boolean isTerm;

    protected PendingEntry(boolean isTerm) {
      this.isTerm = isTerm;
    }
  }

  private static final class PendingTerm extends PendingEntry {
    public final byte[] termBytes;
    // stats + metadata
    public final BlockTermState state;

    public PendingTerm(BytesRef term, BlockTermState state) {
      super(true);
      this.termBytes = new byte[term.length];
      System.arraycopy(term.bytes, term.offset, termBytes, 0, term.length);
      this.state = state;
    }

    @Override
    public String toString() {
      return "TERM: " + brToString(termBytes);
    }
  }

  // for debugging
  @SuppressWarnings("unused")
  static String brToString(BytesRef b) {
    if (b == null) {
      return "(null)";
    } else {
      try {
        return b.utf8ToString() + " " + b;
      } catch (Throwable t) {
        // If BytesRef isn't actually UTF8, or it's eg a
        // prefix of UTF8 that ends mid-unicode-char, we
        // fallback to hex:
        return b.toString();
      }
    }
  }

  // for debugging
  @SuppressWarnings("unused")
  static String brToString(byte[] b) {
    return brToString(new BytesRef(b));
  }

  private static final class PendingBlock extends PendingEntry {
    public final BytesRef prefix;
    public final long fp;
    public FST<BytesRef> index;
    public List<FST<BytesRef>> subIndices;
    public final boolean hasTerms;
    public final boolean isFloor;
    public final int floorLeadByte;

    public PendingBlock(
        BytesRef prefix,
        long fp,
        boolean hasTerms,
        boolean isFloor,
        int floorLeadByte,
        List<FST<BytesRef>> subIndices) {
      super(false);
      this.prefix = prefix;
      this.fp = fp;
      this.hasTerms = hasTerms;
      this.isFloor = isFloor;
      this.floorLeadByte = floorLeadByte;
      this.subIndices = subIndices;
    }

    @Override
    public String toString() {
      return "BLOCK: prefix=" + brToString(prefix);
    }

    public void compileIndex(
        List<PendingBlock> blocks,
        ByteBuffersDataOutput scratchBytes,
        IntsRefBuilder scratchIntsRef)
        throws IOException {

      assert (isFloor && blocks.size() > 1) || (isFloor == false && blocks.size() == 1)
          : "isFloor=" + isFloor + " blocks=" + blocks;
      assert this == blocks.get(0);

      assert scratchBytes.size() == 0;

      // TODO: try writing the leading vLong in MSB order
      // (opposite of what Lucene does today), for better
      // outputs sharing in the FST
      scratchBytes.writeVLong(encodeOutput(fp, hasTerms, isFloor));
      if (isFloor) {
        scratchBytes.writeVInt(blocks.size() - 1);
        for (int i = 1; i < blocks.size(); i++) {
          PendingBlock sub = blocks.get(i);
          assert sub.floorLeadByte != -1;
          // if (DEBUG) {
          //  System.out.println("    write floorLeadByte=" +
          // Integer.toHexString(sub.floorLeadByte&0xff));
          // }
          scratchBytes.writeByte((byte) sub.floorLeadByte);
          assert sub.fp > fp;
          scratchBytes.writeVLong((sub.fp - fp) << 1 | (sub.hasTerms ? 1 : 0));
        }
      }

      final ByteSequenceOutputs outputs = ByteSequenceOutputs.getSingleton();
      final FSTCompiler<BytesRef> fstCompiler =
          new FSTCompiler.Builder<>(FST.INPUT_TYPE.BYTE1, outputs).build();
      // if (DEBUG) {
      //  System.out.println("  compile index for prefix=" + prefix);
      // }
      // indexBuilder.DEBUG = false;
      final byte[] bytes = scratchBytes.toArrayCopy();
      assert bytes.length > 0;
      fstCompiler.add(Util.toIntsRef(prefix, scratchIntsRef), new BytesRef(bytes, 0, bytes.length));
      scratchBytes.reset();

      // Copy over index for all sub-blocks
      for (PendingBlock block : blocks) {
        if (block.subIndices != null) {
          for (FST<BytesRef> subIndex : block.subIndices) {
            append(fstCompiler, subIndex, scratchIntsRef);
          }
          block.subIndices = null;
        }
      }

      index = FST.fromFSTReader(fstCompiler.compile(), fstCompiler.getFSTReader());

      assert subIndices == null;

      /*
      Writer w = new OutputStreamWriter(new FileOutputStream("out.dot"));
      Util.toDot(index, w, false, false);
      System.out.println("SAVED to out.dot");
      w.close();
      */
    }

    // TODO: maybe we could add bulk-add method to
    // Builder?  Takes FST and unions it w/ current
    // FST.
    private void append(
        FSTCompiler<BytesRef> fstCompiler, FST<BytesRef> subIndex, IntsRefBuilder scratchIntsRef)
        throws IOException {
      final BytesRefFSTEnum<BytesRef> subIndexEnum = new BytesRefFSTEnum<>(subIndex);
      BytesRefFSTEnum.InputOutput<BytesRef> indexEnt;
      while ((indexEnt = subIndexEnum.next()) != null) {
        // if (DEBUG) {
        //  System.out.println("      add sub=" + indexEnt.input + " " + indexEnt.input + " output="
        // + indexEnt.output);
        // }
        fstCompiler.add(Util.toIntsRef(indexEnt.input, scratchIntsRef), indexEnt.output);
      }
    }
  }

  private final ByteBuffersDataOutput scratchBytes = ByteBuffersDataOutput.newResettableInstance();
  private final IntsRefBuilder scratchIntsRef = new IntsRefBuilder();

  static final BytesRef EMPTY_BYTES_REF = new BytesRef();

  private static class StatsWriter {

    private final DataOutput out;
    private final boolean hasFreqs;
    private int singletonCount;

    StatsWriter(DataOutput out, boolean hasFreqs) {
      this.out = out;
      this.hasFreqs = hasFreqs;
    }

    void add(int df, long ttf) throws IOException {
      // Singletons (DF==1, TTF==1) are run-length encoded
      if (df == 1 && (hasFreqs == false || ttf == 1)) {
        singletonCount++;
      } else {
        finish();
        out.writeVInt(df << 1);
        if (hasFreqs) {
          out.writeVLong(ttf - df);
        }
      }
    }

    void finish() throws IOException {
      if (singletonCount > 0) {
        out.writeVInt(((singletonCount - 1) << 1) | 1);
        singletonCount = 0;
      }
    }
  }

  class TermsWriter {
    private final FieldInfo fieldInfo;
    private long numTerms;
    final FixedBitSet docsSeen;
    long sumTotalTermFreq;
    long sumDocFreq;
    long indexStartFP;

    // Records index into pending where the current prefix at that
    // length "started"; for example, if current term starts with 't',
    // startsByPrefix[0] is the index into pending for the first
    // term/sub-block starting with 't'.  We use this to figure out when
    // to write a new block:
    private final BytesRefBuilder lastTerm = new BytesRefBuilder();
    private int[] prefixStarts = new int[8];

    // Pending stack of terms and blocks.  As terms arrive (in sorted order)
    // we append to this stack, and once the top of the stack has enough
    // terms starting with a common prefix, we write a new block with
    // those terms and replace those terms in the stack with a new block:
    private final List<PendingEntry> pending = new ArrayList<>();

    // Reused in writeBlocks:
    private final List<PendingBlock> newBlocks = new ArrayList<>();

    private PendingTerm firstPendingTerm;
    private PendingTerm lastPendingTerm;

    /** Writes the top count entries in pending, using prevTerm to compute the prefix. */
    void writeBlocks(int prefixLength, int count) throws IOException {

      assert count > 0;

      // if (DEBUG2) {
      //  BytesRef br = new BytesRef(lastTerm.bytes());
      //  br.length = prefixLength;
      //  System.out.println("writeBlocks: seg=" + segment + " prefix=" + brToString(br) + " count="
      // + count);
      // }

      // Root block better write all remaining pending entries:
      assert prefixLength > 0 || count == pending.size();

      int lastSuffixLeadLabel = -1;

      // True if we saw at least one term in this block (we record if a block
      // only points to sub-blocks in the terms index so we can avoid seeking
      // to it when we are looking for a term):
      boolean hasTerms = false;
      boolean hasSubBlocks = false;

      int start = pending.size() - count;
      int end = pending.size();
      int nextBlockStart = start;
      int nextFloorLeadLabel = -1;

      for (int i = start; i < end; i++) {

        PendingEntry ent = pending.get(i);

        int suffixLeadLabel;

        if (ent.isTerm) {
          PendingTerm term = (PendingTerm) ent;
          if (term.termBytes.length == prefixLength) {
            // Suffix is 0, i.e. prefix 'foo' and term is
            // 'foo' so the term has empty string suffix
            // in this block
            assert lastSuffixLeadLabel == -1
                : "i=" + i + " lastSuffixLeadLabel=" + lastSuffixLeadLabel;
            suffixLeadLabel = -1;
          } else {
            suffixLeadLabel = term.termBytes[prefixLength] & 0xff;
          }
        } else {
          PendingBlock block = (PendingBlock) ent;
          assert block.prefix.length > prefixLength;
          suffixLeadLabel = block.prefix.bytes[block.prefix.offset + prefixLength] & 0xff;
        }
        // if (DEBUG) System.out.println("  i=" + i + " ent=" + ent + " suffixLeadLabel=" +
        // suffixLeadLabel);

        if (suffixLeadLabel != lastSuffixLeadLabel) {
          int itemsInBlock = i - nextBlockStart;
          if (itemsInBlock >= minItemsInBlock && end - nextBlockStart > maxItemsInBlock) {
            // The count is too large for one block, so we must break it into "floor" blocks, where
            // we record
            // the leading label of the suffix of the first term in each floor block, so at search
            // time we can
            // jump to the right floor block.  We just use a naive greedy segmenter here: make a new
            // floor
            // block as soon as we have at least minItemsInBlock.  This is not always best: it often
            // produces
            // a too-small block as the final block:
            boolean isFloor = itemsInBlock < count;
            newBlocks.add(
                writeBlock(
                    prefixLength,
                    isFloor,
                    nextFloorLeadLabel,
                    nextBlockStart,
                    i,
                    hasTerms,
                    hasSubBlocks));

            hasTerms = false;
            hasSubBlocks = false;
            nextFloorLeadLabel = suffixLeadLabel;
            nextBlockStart = i;
          }

          lastSuffixLeadLabel = suffixLeadLabel;
        }

        if (ent.isTerm) {
          hasTerms = true;
        } else {
          hasSubBlocks = true;
        }
      }

      // Write last block, if any:
      if (nextBlockStart < end) {
        int itemsInBlock = end - nextBlockStart;
        boolean isFloor = itemsInBlock < count;
        newBlocks.add(
            writeBlock(
                prefixLength,
                isFloor,
                nextFloorLeadLabel,
                nextBlockStart,
                end,
                hasTerms,
                hasSubBlocks));
      }

      assert newBlocks.isEmpty() == false;

      PendingBlock firstBlock = newBlocks.get(0);

      assert firstBlock.isFloor || newBlocks.size() == 1;

      firstBlock.compileIndex(newBlocks, scratchBytes, scratchIntsRef);

      // Remove slice from the top of the pending stack, that we just wrote:
      pending.subList(pending.size() - count, pending.size()).clear();

      // Append new block
      pending.add(firstBlock);

      newBlocks.clear();
    }

    private boolean allEqual(byte[] b, int startOffset, int endOffset, byte value) {
      Objects.checkFromToIndex(startOffset, endOffset, b.length);
      for (int i = startOffset; i < endOffset; ++i) {
        if (b[i] != value) {
          return false;
        }
      }
      return true;
    }

    /**
     * Writes the specified slice (start is inclusive, end is exclusive) from pending stack as a new
     * block. If isFloor is true, there were too many (more than maxItemsInBlock) entries sharing
     * the same prefix, and so we broke it into multiple floor blocks where we record the starting
     * label of the suffix of each floor block.
     */
    private PendingBlock writeBlock(
        int prefixLength,
        boolean isFloor,
        int floorLeadLabel,
        int start,
        int end,
        boolean hasTerms,
        boolean hasSubBlocks)
        throws IOException {

      assert end > start;

      long startFP = termsOut.getFilePointer();

      boolean hasFloorLeadLabel = isFloor && floorLeadLabel != -1;

      final BytesRef prefix = new BytesRef(prefixLength + (hasFloorLeadLabel ? 1 : 0));
      System.arraycopy(lastTerm.get().bytes, 0, prefix.bytes, 0, prefixLength);
      prefix.length = prefixLength;

      // if (DEBUG2) System.out.println("    writeBlock field=" + fieldInfo.name + " prefix=" +
      // brToString(prefix) + " fp=" + startFP + " isFloor=" + isFloor + " isLastInFloor=" + (end ==
      // pending.size()) + " floorLeadLabel=" + floorLeadLabel + " start=" + start + " end=" + end +
      // " hasTerms=" + hasTerms + " hasSubBlocks=" + hasSubBlocks);

      // Write block header:
      int numEntries = end - start;
      int code = numEntries << 1;
      if (end == pending.size()) {
        // Last block:
        code |= 1;
      }
      termsOut.writeVInt(code);

      /*
      if (DEBUG) {
        System.out.println("  writeBlock " + (isFloor ? "(floor) " : "") + "seg=" + segment + " pending.size()=" + pending.size() + " prefixLength=" + prefixLength + " indexPrefix=" + brToString(prefix) + " entCount=" + (end-start+1) + " startFP=" + startFP + (isFloor ? (" floorLeadLabel=" + Integer.toHexString(floorLeadLabel)) : ""));
      }
      */

      // 1st pass: pack term suffix bytes into byte[] blob
      // TODO: cutover to bulk int codec... simple64?

      // We optimize the leaf block case (block has only terms), writing a more
      // compact format in this case:
      boolean isLeafBlock = hasSubBlocks == false;

      // System.out.println("  isLeaf=" + isLeafBlock);

      final List<FST<BytesRef>> subIndices;

      boolean absolute = true;

      if (isLeafBlock) {
        // Block contains only ordinary terms:
        subIndices = null;
        StatsWriter statsWriter =
            new StatsWriter(this.statsWriter, fieldInfo.getIndexOptions() != IndexOptions.DOCS);
        for (int i = start; i < end; i++) {
          PendingEntry ent = pending.get(i);
          assert ent.isTerm : "i=" + i;

          PendingTerm term = (PendingTerm) ent;

          assert StringHelper.startsWith(term.termBytes, prefix)
              : "term.term=" + new BytesRef(term.termBytes) + " prefix=" + prefix;
          BlockTermState state = term.state;
          final int suffix = term.termBytes.length - prefixLength;
          // if (DEBUG2) {
          //  BytesRef suffixBytes = new BytesRef(suffix);
          //  System.arraycopy(term.termBytes, prefixLength, suffixBytes.bytes, 0, suffix);
          //  suffixBytes.length = suffix;
          //  System.out.println("    write term suffix=" + brToString(suffixBytes));
          // }

          // For leaf block we write suffix straight
          suffixLengthsWriter.writeVInt(suffix);
          suffixWriter.append(term.termBytes, prefixLength, suffix);
          assert floorLeadLabel == -1 || (term.termBytes[prefixLength] & 0xff) >= floorLeadLabel;

          // Write term stats, to separate byte[] blob:
          statsWriter.add(state.docFreq, state.totalTermFreq);

          // Write term meta data
          postingsWriter.encodeTerm(metaWriter, fieldInfo, state, absolute);
          absolute = false;
        }
        statsWriter.finish();
      } else {
        // Block has at least one prefix term or a sub block:
        subIndices = new ArrayList<>();
        StatsWriter statsWriter =
            new StatsWriter(this.statsWriter, fieldInfo.getIndexOptions() != IndexOptions.DOCS);
        for (int i = start; i < end; i++) {
          PendingEntry ent = pending.get(i);
          if (ent.isTerm) {
            PendingTerm term = (PendingTerm) ent;

            assert StringHelper.startsWith(term.termBytes, prefix)
                : "term.term=" + new BytesRef(term.termBytes) + " prefix=" + prefix;
            BlockTermState state = term.state;
            final int suffix = term.termBytes.length - prefixLength;
            // if (DEBUG2) {
            //  BytesRef suffixBytes = new BytesRef(suffix);
            //  System.arraycopy(term.termBytes, prefixLength, suffixBytes.bytes, 0, suffix);
            //  suffixBytes.length = suffix;
            //  System.out.println("      write term suffix=" + brToString(suffixBytes));
            // }

            // For non-leaf block we borrow 1 bit to record
            // if entry is term or sub-block, and 1 bit to record if
            // it's a prefix term.  Terms cannot be larger than ~32 KB
            // so we won't run out of bits:

            suffixLengthsWriter.writeVInt(suffix << 1);
            suffixWriter.append(term.termBytes, prefixLength, suffix);

            // Write term stats, to separate byte[] blob:
            statsWriter.add(state.docFreq, state.totalTermFreq);

            // TODO: now that terms dict "sees" these longs,
            // we can explore better column-stride encodings
            // to encode all long[0]s for this block at
            // once, all long[1]s, etc., e.g. using
            // Simple64.  Alternatively, we could interleave
            // stats + meta ... no reason to have them
            // separate anymore:

            // Write term meta data
            postingsWriter.encodeTerm(metaWriter, fieldInfo, state, absolute);
            absolute = false;
          } else {
            PendingBlock block = (PendingBlock) ent;
            assert StringHelper.startsWith(block.prefix, prefix);
            final int suffix = block.prefix.length - prefixLength;
            assert StringHelper.startsWith(block.prefix, prefix);

            assert suffix > 0;

            // For non-leaf block we borrow 1 bit to record
            // if entry is term or sub-block:f
            suffixLengthsWriter.writeVInt((suffix << 1) | 1);
            suffixWriter.append(block.prefix.bytes, prefixLength, suffix);

            // if (DEBUG2) {
            //  BytesRef suffixBytes = new BytesRef(suffix);
            //  System.arraycopy(block.prefix.bytes, prefixLength, suffixBytes.bytes, 0, suffix);
            //  suffixBytes.length = suffix;
            //  System.out.println("      write sub-block suffix=" + brToString(suffixBytes) + "
            // subFP=" + block.fp + " subCode=" + (startFP-block.fp) + " floor=" + block.isFloor);
            // }

            assert floorLeadLabel == -1
                    || (block.prefix.bytes[prefixLength] & 0xff) >= floorLeadLabel
                : "floorLeadLabel="
                    + floorLeadLabel
                    + " suffixLead="
                    + (block.prefix.bytes[prefixLength] & 0xff);
            assert block.fp < startFP;

            suffixLengthsWriter.writeVLong(startFP - block.fp);
            subIndices.add(block.index);
          }
        }
        statsWriter.finish();

        assert subIndices.size() != 0;
      }

      // Write suffixes byte[] blob to terms dict output, either uncompressed, compressed with LZ4
      // or with LowercaseAsciiCompression.
      CompressionAlgorithm compressionAlg = CompressionAlgorithm.NO_COMPRESSION;
      // If there are 2 suffix bytes or less per term, then we don't bother compressing as suffix
      // are unlikely what
      // makes the terms dictionary large, and it also tends to be frequently the case for dense IDs
      // like
      // auto-increment IDs, so not compressing in that case helps not hurt ID lookups by too much.
      // We also only start compressing when the prefix length is greater than 2 since blocks whose
      // prefix length is
      // 1 or 2 always all get visited when running a fuzzy query whose max number of edits is 2.
      if (suffixWriter.length() > 2L * numEntries && prefixLength > 2) {
        // LZ4 inserts references whenever it sees duplicate strings of 4 chars or more, so only try
        // it out if the
        // average suffix length is greater than 6.
        if (suffixWriter.length() > 6L * numEntries) {
          LZ4.compress(
              suffixWriter.bytes(), 0, suffixWriter.length(), spareWriter, compressionHashTable);
          if (spareWriter.size() < suffixWriter.length() - (suffixWriter.length() >>> 2)) {
            // LZ4 saved more than 25%, go for it
            compressionAlg = CompressionAlgorithm.LZ4;
          }
        }
        if (compressionAlg == CompressionAlgorithm.NO_COMPRESSION) {
          spareWriter.reset();
          if (spareBytes.length < suffixWriter.length()) {
            spareBytes = new byte[ArrayUtil.oversize(suffixWriter.length(), 1)];
          }
          if (LowercaseAsciiCompression.compress(
              suffixWriter.bytes(), suffixWriter.length(), spareBytes, spareWriter)) {
            compressionAlg = CompressionAlgorithm.LOWERCASE_ASCII;
          }
        }
      }
      long token = ((long) suffixWriter.length()) << 3;
      if (isLeafBlock) {
        token |= 0x04;
      }
      token |= compressionAlg.code;
      termsOut.writeVLong(token);
      if (compressionAlg == CompressionAlgorithm.NO_COMPRESSION) {
        termsOut.writeBytes(suffixWriter.bytes(), suffixWriter.length());
      } else {
        spareWriter.copyTo(termsOut);
      }
      suffixWriter.setLength(0);
      spareWriter.reset();

      // Write suffix lengths
      final int numSuffixBytes = Math.toIntExact(suffixLengthsWriter.size());
      spareBytes = ArrayUtil.grow(spareBytes, numSuffixBytes);
      suffixLengthsWriter.copyTo(new ByteArrayDataOutput(spareBytes));
      suffixLengthsWriter.reset();
      if (allEqual(spareBytes, 1, numSuffixBytes, spareBytes[0])) {
        // Structured fields like IDs often have most values of the same length
        termsOut.writeVInt((numSuffixBytes << 1) | 1);
        termsOut.writeByte(spareBytes[0]);
      } else {
        termsOut.writeVInt(numSuffixBytes << 1);
        termsOut.writeBytes(spareBytes, numSuffixBytes);
      }

      // Stats
      final int numStatsBytes = Math.toIntExact(statsWriter.size());
      termsOut.writeVInt(numStatsBytes);
      statsWriter.copyTo(termsOut);
      statsWriter.reset();

      // Write term meta data byte[] blob
      termsOut.writeVInt((int) metaWriter.size());
      metaWriter.copyTo(termsOut);
      metaWriter.reset();

      // if (DEBUG) {
      //   System.out.println("      fpEnd=" + out.getFilePointer());
      // }

      if (hasFloorLeadLabel) {
        // We already allocated to length+1 above:
        prefix.bytes[prefix.length++] = (byte) floorLeadLabel;
      }

      return new PendingBlock(prefix, startFP, hasTerms, isFloor, floorLeadLabel, subIndices);
    }

    TermsWriter(FieldInfo fieldInfo) {
      this.fieldInfo = fieldInfo;
      assert fieldInfo.getIndexOptions() != IndexOptions.NONE;
      docsSeen = new FixedBitSet(maxDoc);
      postingsWriter.setField(fieldInfo);
    }

    /** Writes one term's worth of postings. */
    public void write(BytesRef text, TermsEnum termsEnum, NormsProducer norms) throws IOException {
      /*
      if (DEBUG) {
        int[] tmp = new int[lastTerm.length];
        System.arraycopy(prefixStarts, 0, tmp, 0, tmp.length);
        System.out.println("BTTW: write term=" + brToString(text) + " prefixStarts=" + Arrays.toString(tmp) + " pending.size()=" + pending.size());
      }
      */

      BlockTermState state = postingsWriter.writeTerm(text, termsEnum, docsSeen, norms);
      if (state != null) {

        assert state.docFreq != 0;
        assert fieldInfo.getIndexOptions() == IndexOptions.DOCS
                || state.totalTermFreq >= state.docFreq
            : "postingsWriter=" + postingsWriter;
        pushTerm(text);

        PendingTerm term = new PendingTerm(text, state);
        pending.add(term);
        // if (DEBUG) System.out.println("    add pending term = " + text + " pending.size()=" +
        // pending.size());

        sumDocFreq += state.docFreq;
        sumTotalTermFreq += state.totalTermFreq;
        numTerms++;
        if (firstPendingTerm == null) {
          firstPendingTerm = term;
        }
        lastPendingTerm = term;
      }
    }

    /** Pushes the new term to the top of the stack, and writes new blocks. */
    private void pushTerm(BytesRef text) throws IOException {
      // Find common prefix between last term and current term:
      int prefixLength =
          Arrays.mismatch(
              lastTerm.bytes(),
              0,
              lastTerm.length(),
              text.bytes,
              text.offset,
              text.offset + text.length);
      if (prefixLength == -1) { // Only happens for the first term, if it is empty
        assert lastTerm.length() == 0;
        prefixLength = 0;
      }

      // if (DEBUG) System.out.println("  shared=" + pos + "  lastTerm.length=" + lastTerm.length);

      // Close the "abandoned" suffix now:
      for (int i = lastTerm.length() - 1; i >= prefixLength; i--) {

        // How many items on top of the stack share the current suffix
        // we are closing:
        int prefixTopSize = pending.size() - prefixStarts[i];
        if (prefixTopSize >= minItemsInBlock) {
          // if (DEBUG) System.out.println("pushTerm i=" + i + " prefixTopSize=" + prefixTopSize + "
          // minItemsInBlock=" + minItemsInBlock);
          writeBlocks(i + 1, prefixTopSize);
          prefixStarts[i] -= prefixTopSize - 1;
        }
      }

      if (prefixStarts.length < text.length) {
        prefixStarts = ArrayUtil.grow(prefixStarts, text.length);
      }

      // Init new tail:
      for (int i = prefixLength; i < text.length; i++) {
        prefixStarts[i] = pending.size();
      }

      lastTerm.copyBytes(text);
    }

    // Finishes all terms in this field
    public void finish() throws IOException {
      if (numTerms > 0) {
        // if (DEBUG) System.out.println("BTTW: finish prefixStarts=" +
        // Arrays.toString(prefixStarts));

        // Add empty term to force closing of all final blocks:
        pushTerm(new BytesRef());

        // TODO: if pending.size() is already 1 with a non-zero prefix length
        // we can save writing a "degenerate" root block, but we have to
        // fix all the places that assume the root block's prefix is the empty string:
        pushTerm(new BytesRef());
        writeBlocks(0, pending.size());

        // We better have one final "root" block:
        assert pending.size() == 1 && !pending.get(0).isTerm
            : "pending.size()=" + pending.size() + " pending=" + pending;
        final PendingBlock root = (PendingBlock) pending.get(0);
        assert root.prefix.length == 0;
        assert root.index.getEmptyOutput() != null;

        // Write FST to index
        indexStartFP = indexOut.getFilePointer();
        root.index.save(indexOut, indexOut);
        // System.out.println("  write FST " + indexStartFP + " field=" + fieldInfo.name);

        /*
        if (DEBUG) {
          final String dotFileName = segment + "_" + fieldInfo.name + ".dot";
          Writer w = new OutputStreamWriter(new FileOutputStream(dotFileName));
          Util.toDot(root.index, w, false, false);
          System.out.println("SAVED to " + dotFileName);
          w.close();
        }
        */
        assert firstPendingTerm != null;
        BytesRef minTerm = new BytesRef(firstPendingTerm.termBytes);

        assert lastPendingTerm != null;
        BytesRef maxTerm = new BytesRef(lastPendingTerm.termBytes);

        fields.add(
            new FieldMetaData(
                fieldInfo,
                ((PendingBlock) pending.get(0)).index.getEmptyOutput(),
                numTerms,
                indexStartFP,
                sumTotalTermFreq,
                sumDocFreq,
                docsSeen.cardinality(),
                minTerm,
                maxTerm));
      } else {
        assert sumTotalTermFreq == 0
            || fieldInfo.getIndexOptions() == IndexOptions.DOCS && sumTotalTermFreq == -1;
        assert sumDocFreq == 0;
        assert docsSeen.cardinality() == 0;
      }
    }

    private final ByteBuffersDataOutput suffixLengthsWriter =
        ByteBuffersDataOutput.newResettableInstance();
    private final BytesRefBuilder suffixWriter = new BytesRefBuilder();
    private final ByteBuffersDataOutput statsWriter = ByteBuffersDataOutput.newResettableInstance();
    private final ByteBuffersDataOutput metaWriter = ByteBuffersDataOutput.newResettableInstance();
    private final ByteBuffersDataOutput spareWriter = ByteBuffersDataOutput.newResettableInstance();
    private byte[] spareBytes = BytesRef.EMPTY_BYTES;
    private final LZ4.HighCompressionHashTable compressionHashTable =
        new LZ4.HighCompressionHashTable();
  }

  private boolean closed;

  @Override
  public void close() throws IOException {
    if (closed) {
      return;
    }
    closed = true;

    boolean success = false;
    try {

      final long dirStart = termsOut.getFilePointer();
      final long indexDirStart = indexOut.getFilePointer();

      termsOut.writeVInt(fields.size());

      for (FieldMetaData field : fields) {
        // System.out.println("  field " + field.fieldInfo.name + " " + field.numTerms + " terms");
        termsOut.writeVInt(field.fieldInfo.number);
        assert field.numTerms > 0;
        termsOut.writeVLong(field.numTerms);
        termsOut.writeVInt(field.rootCode.length);
        termsOut.writeBytes(field.rootCode.bytes, field.rootCode.offset, field.rootCode.length);
        assert field.fieldInfo.getIndexOptions() != IndexOptions.NONE;
        if (field.fieldInfo.getIndexOptions() != IndexOptions.DOCS) {
          termsOut.writeVLong(field.sumTotalTermFreq);
        }
        termsOut.writeVLong(field.sumDocFreq);
        termsOut.writeVInt(field.docCount);
        indexOut.writeVLong(field.indexStartFP);
        writeBytesRef(termsOut, field.minTerm);
        writeBytesRef(termsOut, field.maxTerm);
      }
      writeTrailer(termsOut, dirStart);
      CodecUtil.writeFooter(termsOut);
      writeIndexTrailer(indexOut, indexDirStart);
      CodecUtil.writeFooter(indexOut);
      success = true;
    } finally {
      if (success) {
        IOUtils.close(termsOut, indexOut, postingsWriter);
      } else {
        IOUtils.closeWhileHandlingException(termsOut, indexOut, postingsWriter);
      }
    }
  }

  private static void writeBytesRef(IndexOutput out, BytesRef bytes) throws IOException {
    out.writeVInt(bytes.length);
    out.writeBytes(bytes.bytes, bytes.offset, bytes.length);
  }
}
