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
package org.apache.lucene.util.packed;

import com.carrotsearch.randomizedtesting.generators.RandomNumbers;
import java.io.IOException;
import java.lang.reflect.Field;
import java.nio.ByteBuffer;
import java.nio.LongBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import org.apache.lucene.store.ByteArrayDataInput;
import org.apache.lucene.store.DataInput;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.RamUsageTester;
import org.apache.lucene.tests.util.TestUtil;
import org.apache.lucene.util.ArrayUtil;
import org.apache.lucene.util.LongValues;
import org.apache.lucene.util.LongsRef;
import org.apache.lucene.util.packed.PackedInts.Reader;
import org.junit.Ignore;

public class TestPackedInts extends LuceneTestCase {

  public void testByteCount() {
    final int iters = atLeast(3);
    for (int i = 0; i < iters; ++i) {
      final int valueCount = RandomNumbers.randomIntBetween(random(), 1, Integer.MAX_VALUE);
      for (PackedInts.Format format : PackedInts.Format.values()) {
        for (int bpv = 1; bpv <= 64; ++bpv) {
          final long byteCount = format.byteCount(PackedInts.VERSION_CURRENT, valueCount, bpv);
          String msg =
              "format="
                  + format
                  + ", byteCount="
                  + byteCount
                  + ", valueCount="
                  + valueCount
                  + ", bpv="
                  + bpv;
          assertTrue(msg, byteCount * 8 >= (long) valueCount * bpv);
          if (format == PackedInts.Format.PACKED) {
            assertTrue(msg, (byteCount - 1) * 8 < (long) valueCount * bpv);
          }
        }
      }
    }
  }

  public void testBitsRequired() {
    assertEquals(61, PackedInts.bitsRequired((long) Math.pow(2, 61) - 1));
    assertEquals(61, PackedInts.bitsRequired(0x1FFFFFFFFFFFFFFFL));
    assertEquals(62, PackedInts.bitsRequired(0x3FFFFFFFFFFFFFFFL));
    assertEquals(63, PackedInts.bitsRequired(0x7FFFFFFFFFFFFFFFL));
    assertEquals(64, PackedInts.unsignedBitsRequired(-1));
    assertEquals(64, PackedInts.unsignedBitsRequired(Long.MIN_VALUE));
    assertEquals(1, PackedInts.bitsRequired(0));
  }

  public void testMaxValues() {
    assertEquals("1 bit -> max == 1", 1, PackedInts.maxValue(1));
    assertEquals("2 bit -> max == 3", 3, PackedInts.maxValue(2));
    assertEquals("8 bit -> max == 255", 255, PackedInts.maxValue(8));
    assertEquals("63 bit -> max == Long.MAX_VALUE", Long.MAX_VALUE, PackedInts.maxValue(63));
    assertEquals(
        "64 bit -> max == Long.MAX_VALUE (same as for 63 bit)",
        Long.MAX_VALUE,
        PackedInts.maxValue(64));
  }

  public void testPackedInts() throws IOException {
    int num = atLeast(3);
    for (int iter = 0; iter < num; iter++) {
      for (int nbits = 1; nbits <= 64; nbits++) {
        final long maxValue = PackedInts.maxValue(nbits);
        final int valueCount = TestUtil.nextInt(random(), 1, 600);
        final int bufferSize =
            random().nextBoolean()
                ? TestUtil.nextInt(random(), 0, 48)
                : TestUtil.nextInt(random(), 0, 4096);
        final Directory d = newDirectory();

        IndexOutput out = d.createOutput("out.bin", newIOContext(random()));
        final int mem = random().nextInt(2 * PackedInts.DEFAULT_BUFFER_SIZE);
        PackedInts.Writer w =
            PackedInts.getWriterNoHeader(out, PackedInts.Format.PACKED, valueCount, nbits, mem);
        final long startFp = out.getFilePointer();

        final int actualValueCount =
            random().nextBoolean() ? valueCount : TestUtil.nextInt(random(), 0, valueCount);
        final long[] values = new long[valueCount];
        for (int i = 0; i < actualValueCount; i++) {
          if (nbits == 64) {
            values[i] = random().nextLong();
          } else {
            values[i] = TestUtil.nextLong(random(), 0, maxValue);
          }
          w.add(values[i]);
        }
        w.finish();
        final long fp = out.getFilePointer();
        out.close();

        // ensure that finish() added the (valueCount-actualValueCount) missing values
        final long bytes =
            w.getFormat().byteCount(PackedInts.VERSION_CURRENT, valueCount, w.bitsPerValue);
        assertEquals(bytes, fp - startFp);

        { // test reader iterator next
          IndexInput in = d.openInput("out.bin", newIOContext(random()));
          PackedInts.ReaderIterator r =
              PackedInts.getReaderIteratorNoHeader(
                  in,
                  PackedInts.Format.PACKED,
                  PackedInts.VERSION_CURRENT,
                  valueCount,
                  nbits,
                  bufferSize);
          for (int i = 0; i < valueCount; i++) {
            assertEquals(
                "index="
                    + i
                    + " valueCount="
                    + valueCount
                    + " nbits="
                    + nbits
                    + " for "
                    + r.getClass().getSimpleName(),
                values[i],
                r.next());
            assertEquals(i, r.ord());
          }
          assertEquals(fp, in.getFilePointer());
          in.close();
        }

        { // test reader iterator bulk next
          IndexInput in = d.openInput("out.bin", newIOContext(random()));
          PackedInts.ReaderIterator r =
              PackedInts.getReaderIteratorNoHeader(
                  in,
                  PackedInts.Format.PACKED,
                  PackedInts.VERSION_CURRENT,
                  valueCount,
                  nbits,
                  bufferSize);
          int i = 0;
          while (i < valueCount) {
            final int count = TestUtil.nextInt(random(), 1, 95);
            final LongsRef next = r.next(count);
            for (int k = 0; k < next.length; ++k) {
              assertEquals(
                  "index="
                      + i
                      + " valueCount="
                      + valueCount
                      + " nbits="
                      + nbits
                      + " for "
                      + r.getClass().getSimpleName(),
                  values[i + k],
                  next.longs[next.offset + k]);
            }
            i += next.length;
          }
          assertEquals(fp, in.getFilePointer());
          in.close();
        }
        d.close();
      }
    }
  }

  public void testEndPointer() throws IOException {
    final Directory dir = newDirectory();
    final int valueCount = RandomNumbers.randomIntBetween(random(), 1, 1000);
    final IndexOutput out = dir.createOutput("tests.bin", newIOContext(random()));
    for (int i = 0; i < valueCount; ++i) {
      out.writeLong(0);
    }
    out.close();
    final IndexInput in = dir.openInput("tests.bin", newIOContext(random()));
    for (int version = PackedInts.VERSION_START; version <= PackedInts.VERSION_CURRENT; ++version) {
      for (int bpv = 1; bpv <= 64; ++bpv) {
        for (PackedInts.Format format : PackedInts.Format.values()) {
          if (!format.isSupported(bpv)) {
            continue;
          }
          final long byteCount = format.byteCount(version, valueCount, bpv);
          String msg =
              "format="
                  + format
                  + ",version="
                  + version
                  + ",valueCount="
                  + valueCount
                  + ",bpv="
                  + bpv;

          // test iterator
          in.seek(0L);
          final PackedInts.ReaderIterator it =
              PackedInts.getReaderIteratorNoHeader(
                  in,
                  format,
                  version,
                  valueCount,
                  bpv,
                  RandomNumbers.randomIntBetween(random(), 1, 1 << 16));
          for (int i = 0; i < valueCount; ++i) {
            it.next();
          }
          assertEquals(msg, byteCount, in.getFilePointer());
        }
      }
    }
    in.close();
    dir.close();
  }

  public void testControlledEquality() {
    final int VALUE_COUNT = 255;
    final int BITS_PER_VALUE = 8;

    List<PackedInts.Mutable> packedInts = createPackedInts(VALUE_COUNT, BITS_PER_VALUE);
    for (PackedInts.Mutable packedInt : packedInts) {
      for (int i = 0; i < packedInt.size(); i++) {
        packedInt.set(i, i + 1);
      }
    }
    assertListEquality(packedInts);
  }

  public void testRandomBulkCopy() {
    Random random = random();
    final int numIters = atLeast(random, 3);
    for (int iter = 0; iter < numIters; iter++) {
      if (VERBOSE) {
        System.out.println("\nTEST: iter=" + iter);
      }
      final int valueCount = TEST_NIGHTLY ? atLeast(random, 100000) : atLeast(random, 10000);
      int bits1 = TestUtil.nextInt(random, 1, 64);
      int bits2 = TestUtil.nextInt(random, 1, 64);
      if (bits1 > bits2) {
        int tmp = bits1;
        bits1 = bits2;
        bits2 = tmp;
      }
      if (VERBOSE) {
        System.out.println("  valueCount=" + valueCount + " bits1=" + bits1 + " bits2=" + bits2);
      }

      final PackedInts.Mutable packed1 =
          PackedInts.getMutable(valueCount, bits1, PackedInts.COMPACT);
      final PackedInts.Mutable packed2 =
          PackedInts.getMutable(valueCount, bits2, PackedInts.COMPACT);

      final long maxValue = PackedInts.maxValue(bits1);
      for (int i = 0; i < valueCount; i++) {
        final long val = TestUtil.nextLong(random, 0, maxValue);
        packed1.set(i, val);
        packed2.set(i, val);
      }

      final long[] buffer = new long[valueCount];

      // Copy random slice over, 20 times:
      for (int iter2 = 0; iter2 < 20; iter2++) {
        int start = random.nextInt(valueCount - 1);
        int len = TestUtil.nextInt(random, 1, valueCount - start);
        int offset;
        if (VERBOSE) {
          System.out.println("  copy " + len + " values @ " + start);
        }
        if (len == valueCount) {
          offset = 0;
        } else {
          offset = random.nextInt(valueCount - len);
        }
        if (random.nextBoolean()) {
          int got = packed1.get(start, buffer, offset, len);
          assertTrue(got <= len);
          int sot = packed2.set(start, buffer, offset, got);
          assertTrue(sot <= got);
        } else {
          PackedInts.copy(packed1, offset, packed2, offset, len, random.nextInt(10 * len));
        }

        /*
        for(int i=0;i<valueCount;i++) {
          assertEquals("value " + i, packed1.get(i), packed2.get(i));
        }
        */
      }

      for (int i = 0; i < valueCount; i++) {
        assertEquals("value " + i, packed1.get(i), packed2.get(i));
      }
    }
  }

  public void testRandomEquality() {
    final int numIters = TEST_NIGHTLY ? atLeast(2) : 1;
    for (int i = 0; i < numIters; ++i) {
      final int valueCount = TestUtil.nextInt(random(), 1, 300);

      for (int bitsPerValue = 1; bitsPerValue <= 64; bitsPerValue++) {
        assertRandomEquality(valueCount, bitsPerValue, random().nextLong());
      }
    }
  }

  private static void assertRandomEquality(int valueCount, int bitsPerValue, long randomSeed) {
    List<PackedInts.Mutable> packedInts = createPackedInts(valueCount, bitsPerValue);
    for (PackedInts.Mutable packedInt : packedInts) {
      try {
        fill(packedInt, bitsPerValue, randomSeed);
      } catch (Exception e) {
        e.printStackTrace(System.err);
        fail(
            String.format(
                Locale.ROOT,
                "Exception while filling %s: valueCount=%d, bitsPerValue=%s",
                packedInt.getClass().getSimpleName(),
                valueCount,
                bitsPerValue));
      }
    }
    assertListEquality(packedInts);
  }

  private static List<PackedInts.Mutable> createPackedInts(int valueCount, int bitsPerValue) {
    List<PackedInts.Mutable> packedInts = new ArrayList<>();
    packedInts.add(new Packed64(valueCount, bitsPerValue));
    for (int bpv = bitsPerValue; bpv <= Packed64SingleBlock.MAX_SUPPORTED_BITS_PER_VALUE; ++bpv) {
      if (Packed64SingleBlock.isSupported(bpv)) {
        packedInts.add(Packed64SingleBlock.create(valueCount, bpv));
      }
    }
    return packedInts;
  }

  private static void fill(PackedInts.Mutable packedInt, int bitsPerValue, long randomSeed) {
    Random rnd2 = new Random(randomSeed);
    final long maxValue = bitsPerValue == 64 ? Long.MAX_VALUE : (1L << bitsPerValue) - 1;
    for (int i = 0; i < packedInt.size(); i++) {
      long value = bitsPerValue == 64 ? random().nextLong() : TestUtil.nextLong(rnd2, 0, maxValue);
      packedInt.set(i, value);
      assertEquals(
          String.format(
              Locale.ROOT,
              "The set/get of the value at index %d should match for %s",
              i,
              packedInt.getClass().getSimpleName()),
          value,
          packedInt.get(i));
    }
  }

  private static void assertListEquality(List<? extends PackedInts.Reader> packedInts) {
    assertListEquality("", packedInts);
  }

  private static void assertListEquality(
      String message, List<? extends PackedInts.Reader> packedInts) {
    if (packedInts.size() == 0) {
      return;
    }
    PackedInts.Reader base = packedInts.get(0);
    int valueCount = base.size();
    for (PackedInts.Reader packedInt : packedInts) {
      assertEquals(
          message + ". The number of values should be the same ", valueCount, packedInt.size());
    }
    for (int i = 0; i < valueCount; i++) {
      for (int j = 1; j < packedInts.size(); j++) {
        assertEquals(
            String.format(
                Locale.ROOT,
                "%s. The value at index %d should be the same for %s and %s",
                message,
                i,
                base.getClass().getSimpleName(),
                packedInts.get(j).getClass().getSimpleName()),
            base.get(i),
            packedInts.get(j).get(i));
      }
    }
  }

  public void testSecondaryBlockChange() {
    PackedInts.Mutable mutable = new Packed64(26, 5);
    mutable.set(24, 31);
    assertEquals("The value #24 should be correct", 31, mutable.get(24));
    mutable.set(4, 16);
    assertEquals("The value #24 should remain unchanged", 31, mutable.get(24));
  }

  /*
   * Check if the structures properly handle the case where
   * index * bitsPerValue > Integer.MAX_VALUE
   *
   * NOTE: this test allocates 256 MB
   */
  @Ignore("See LUCENE-4488")
  public void testIntOverflow() {
    int INDEX = (int) Math.pow(2, 30) + 1;
    int BITS = 2;

    Packed64 p64 = null;
    try {
      p64 = new Packed64(INDEX, BITS);
    } catch (
        @SuppressWarnings("unused")
        OutOfMemoryError oome) {
      // This can easily happen: we're allocating a
      // long[] that needs 256-273 MB.  Heap is 512 MB,
      // but not all of that is available for large
      // objects ... empirical testing shows we only
      // have ~ 67 MB free.
    }
    if (p64 != null) {
      p64.set(INDEX - 1, 1);
      assertEquals(
          "The value at position " + (INDEX - 1) + " should be correct for Packed64",
          1,
          p64.get(INDEX - 1));
      p64 = null;
    }

    Packed64SingleBlock p64sb = null;
    try {
      p64sb = Packed64SingleBlock.create(INDEX, BITS);
    } catch (
        @SuppressWarnings("unused")
        OutOfMemoryError oome) {
      // Ignore: see comment above
    }
    if (p64sb != null) {
      p64sb.set(INDEX - 1, 1);
      assertEquals(
          "The value at position "
              + (INDEX - 1)
              + " should be correct for "
              + p64sb.getClass().getSimpleName(),
          1,
          p64sb.get(INDEX - 1));
    }
  }

  public void testFill() {
    final int valueCount = 1111;
    final int from = random().nextInt(valueCount + 1);
    final int to = from + random().nextInt(valueCount + 1 - from);
    for (int bpv = 1; bpv <= 64; ++bpv) {
      final long val = TestUtil.nextLong(random(), 0, PackedInts.maxValue(bpv));
      List<PackedInts.Mutable> packedInts = createPackedInts(valueCount, bpv);
      for (PackedInts.Mutable ints : packedInts) {
        String msg =
            ints.getClass().getSimpleName()
                + " bpv="
                + bpv
                + ", from="
                + from
                + ", to="
                + to
                + ", val="
                + val;
        ints.fill(0, ints.size(), 1);
        ints.fill(from, to, val);
        for (int i = 0; i < ints.size(); ++i) {
          if (i >= from && i < to) {
            assertEquals(msg + ", i=" + i, val, ints.get(i));
          } else {
            assertEquals(msg + ", i=" + i, 1, ints.get(i));
          }
        }
      }
    }
  }

  public void testPackedIntsNull() {
    // must be > 10 for the bulk reads below
    int size = TestUtil.nextInt(random(), 11, 256);
    Reader packedInts = PackedInts.NullReader.forCount(size);
    assertEquals(0, packedInts.get(TestUtil.nextInt(random(), 0, size - 1)));
    long[] arr = new long[size + 10];
    int r;
    Arrays.fill(arr, 1);
    r = packedInts.get(0, arr, 0, size - 1);
    assertEquals(size - 1, r);
    for (r--; r >= 0; r--) {
      assertEquals(0, arr[r]);
    }
    Arrays.fill(arr, 1);
    r = packedInts.get(10, arr, 0, size + 10);
    assertEquals(size - 10, r);
    for (int i = 0; i < size - 10; i++) {
      assertEquals(0, arr[i]);
    }
  }

  public void testBulkGet() {
    final int valueCount = 1111;
    final int index = random().nextInt(valueCount);
    final int len = TestUtil.nextInt(random(), 1, valueCount * 2);
    final int off = random().nextInt(77);

    for (int bpv = 1; bpv <= 64; ++bpv) {
      long mask = PackedInts.maxValue(bpv);
      List<PackedInts.Mutable> packedInts = createPackedInts(valueCount, bpv);

      for (PackedInts.Mutable ints : packedInts) {
        for (int i = 0; i < ints.size(); ++i) {
          ints.set(i, (31L * i - 1099) & mask);
        }
        long[] arr = new long[off + len];

        String msg =
            ints.getClass().getSimpleName()
                + " valueCount="
                + valueCount
                + ", index="
                + index
                + ", len="
                + len
                + ", off="
                + off;
        final int gets = ints.get(index, arr, off, len);
        assertTrue(msg, gets > 0);
        assertTrue(msg, gets <= len);
        assertTrue(msg, gets <= ints.size() - index);

        for (int i = 0; i < arr.length; ++i) {
          String m = msg + ", i=" + i;
          if (i >= off && i < off + gets) {
            assertEquals(m, ints.get(i - off + index), arr[i]);
          } else {
            assertEquals(m, 0, arr[i]);
          }
        }
      }
    }
  }

  public void testBulkSet() {
    final int valueCount = 1111;
    final int index = random().nextInt(valueCount);
    final int len = TestUtil.nextInt(random(), 1, valueCount * 2);
    final int off = random().nextInt(77);
    long[] arr = new long[off + len];

    for (int bpv = 1; bpv <= 64; ++bpv) {
      long mask = PackedInts.maxValue(bpv);
      List<PackedInts.Mutable> packedInts = createPackedInts(valueCount, bpv);
      for (int i = 0; i < arr.length; ++i) {
        arr[i] = (31L * i + 19) & mask;
      }

      for (PackedInts.Mutable ints : packedInts) {
        String msg =
            ints.getClass().getSimpleName()
                + " valueCount="
                + valueCount
                + ", index="
                + index
                + ", len="
                + len
                + ", off="
                + off;
        final int sets = ints.set(index, arr, off, len);
        assertTrue(msg, sets > 0);
        assertTrue(msg, sets <= len);

        for (int i = 0; i < ints.size(); ++i) {
          String m = msg + ", i=" + i;
          if (i >= index && i < index + sets) {
            assertEquals(m, arr[off - index + i], ints.get(i));
          } else {
            assertEquals(m, 0, ints.get(i));
          }
        }
      }
    }
  }

  public void testCopy() {
    final int valueCount = TestUtil.nextInt(random(), 5, 600);
    final int off1 = random().nextInt(valueCount);
    final int off2 = random().nextInt(valueCount);
    final int len = random().nextInt(Math.min(valueCount - off1, valueCount - off2));
    final int mem = random().nextInt(1024);

    for (int bpv = 1; bpv <= 64; ++bpv) {
      long mask = PackedInts.maxValue(bpv);
      for (PackedInts.Mutable r1 : createPackedInts(valueCount, bpv)) {
        for (int i = 0; i < r1.size(); ++i) {
          r1.set(i, (31L * i - 1023) & mask);
        }
        for (PackedInts.Mutable r2 : createPackedInts(valueCount, bpv)) {
          String msg =
              "src="
                  + r1
                  + ", dest="
                  + r2
                  + ", srcPos="
                  + off1
                  + ", destPos="
                  + off2
                  + ", len="
                  + len
                  + ", mem="
                  + mem;
          PackedInts.copy(r1, off1, r2, off2, len, mem);
          for (int i = 0; i < r2.size(); ++i) {
            String m = msg + ", i=" + i;
            if (i >= off2 && i < off2 + len) {
              assertEquals(m, r1.get(i - off2 + off1), r2.get(i));
            } else {
              assertEquals(m, 0, r2.get(i));
            }
          }
        }
      }
    }
  }

  public void testGrowableWriter() {
    final int valueCount = 113 + random().nextInt(1111);
    GrowableWriter wrt = new GrowableWriter(1, valueCount, PackedInts.DEFAULT);
    wrt.set(4, 2);
    wrt.set(7, 10);
    wrt.set(valueCount - 10, 99);
    wrt.set(99, 999);
    wrt.set(valueCount - 1, 1 << 10);
    assertEquals(1 << 10, wrt.get(valueCount - 1));
    wrt.set(99, (1 << 23) - 1);
    assertEquals(1 << 10, wrt.get(valueCount - 1));
    wrt.set(1, Long.MAX_VALUE);
    wrt.set(2, -3);
    assertEquals(64, wrt.getBitsPerValue());
    assertEquals(1 << 10, wrt.get(valueCount - 1));
    assertEquals(Long.MAX_VALUE, wrt.get(1));
    assertEquals(-3L, wrt.get(2));
    assertEquals(2, wrt.get(4));
    assertEquals((1 << 23) - 1, wrt.get(99));
    assertEquals(10, wrt.get(7));
    assertEquals(99, wrt.get(valueCount - 10));
    assertEquals(1 << 10, wrt.get(valueCount - 1));
    assertEquals(RamUsageTester.ramUsed(wrt), wrt.ramBytesUsed());
  }

  public void testPagedGrowableWriter() {
    Random random = random();
    int pageSize = 1 << (TestUtil.nextInt(random, 6, 30));
    // supports 0 values?
    PagedGrowableWriter writer =
        new PagedGrowableWriter(0, pageSize, TestUtil.nextInt(random, 1, 64), random.nextFloat());
    assertEquals(0, writer.size());

    // compare against AppendingDeltaPackedLongBuffer
    PackedLongValues.Builder buf = PackedLongValues.deltaPackedBuilder(random.nextFloat());
    int size = TEST_NIGHTLY ? random.nextInt(1000000) : random.nextInt(100000);
    long max = 5;
    for (int i = 0; i < size; ++i) {
      buf.add(TestUtil.nextLong(random, 0, max));
      if (rarely(random)) {
        max =
            PackedInts.maxValue(
                rarely(random) ? TestUtil.nextInt(random, 0, 63) : TestUtil.nextInt(random, 0, 31));
      }
    }
    writer =
        new PagedGrowableWriter(
            size, pageSize, TestUtil.nextInt(random, 1, 64), random.nextFloat());
    assertEquals(size, writer.size());
    final LongValues values = buf.build();
    for (int i = size - 1; i >= 0; --i) {
      writer.set(i, values.get(i));
    }
    for (int i = 0; i < size; ++i) {
      assertEquals(values.get(i), writer.get(i));
    }

    // test ramBytesUsed
    assertEquals((double) RamUsageTester.ramUsed(writer), (double) writer.ramBytesUsed(), 8.d);

    // test copy
    PagedGrowableWriter copy =
        writer.resize(TestUtil.nextLong(random, writer.size() / 2, writer.size() * 3 / 2));
    for (long i = 0; i < copy.size(); ++i) {
      if (i < writer.size()) {
        assertEquals(writer.get(i), copy.get(i));
      } else {
        assertEquals(0, copy.get(i));
      }
    }

    // test grow
    PagedGrowableWriter grow =
        writer.grow(TestUtil.nextLong(random, writer.size() / 2, writer.size() * 3 / 2));
    for (long i = 0; i < grow.size(); ++i) {
      if (i < writer.size()) {
        assertEquals(writer.get(i), grow.get(i));
      } else {
        assertEquals(0, grow.get(i));
      }
    }
  }

  public void testPagedMutable() {
    Random random = random();
    final int bitsPerValue = TestUtil.nextInt(random, 1, 64);
    final long max = PackedInts.maxValue(bitsPerValue);
    int pageSize = 1 << (TestUtil.nextInt(random, 6, 30));
    // supports 0 values?
    PagedMutable writer = new PagedMutable(0, pageSize, bitsPerValue, random.nextFloat() / 2);
    assertEquals(0, writer.size());

    // compare against AppendingDeltaPackedLongBuffer
    PackedLongValues.Builder buf = PackedLongValues.deltaPackedBuilder(random.nextFloat());
    int size = TEST_NIGHTLY ? random.nextInt(1000000) : random.nextInt(100000);

    for (int i = 0; i < size; ++i) {
      buf.add(bitsPerValue == 64 ? random.nextLong() : TestUtil.nextLong(random, 0, max));
    }
    writer = new PagedMutable(size, pageSize, bitsPerValue, random.nextFloat());
    assertEquals(size, writer.size());
    final LongValues values = buf.build();
    for (int i = size - 1; i >= 0; --i) {
      writer.set(i, values.get(i));
    }
    for (int i = 0; i < size; ++i) {
      assertEquals(values.get(i), writer.get(i));
    }

    // test ramBytesUsed
    assertEquals(
        RamUsageTester.ramUsed(writer) - RamUsageTester.ramUsed(writer.format),
        writer.ramBytesUsed());

    // test copy
    PagedMutable copy =
        writer.resize(TestUtil.nextLong(random, writer.size() / 2, writer.size() * 3 / 2));
    for (long i = 0; i < copy.size(); ++i) {
      if (i < writer.size()) {
        assertEquals(writer.get(i), copy.get(i));
      } else {
        assertEquals(0, copy.get(i));
      }
    }

    // test grow
    PagedMutable grow =
        writer.grow(TestUtil.nextLong(random, writer.size() / 2, writer.size() * 3 / 2));
    for (long i = 0; i < grow.size(); ++i) {
      if (i < writer.size()) {
        assertEquals(writer.get(i), grow.get(i));
      } else {
        assertEquals(0, grow.get(i));
      }
    }
  }

  // memory hole
  @Ignore
  public void testPagedGrowableWriterOverflow() {
    final long size =
        TestUtil.nextLong(random(), 2 * (long) Integer.MAX_VALUE, 3 * (long) Integer.MAX_VALUE);
    final int pageSize = 1 << (TestUtil.nextInt(random(), 16, 30));
    final PagedGrowableWriter writer =
        new PagedGrowableWriter(size, pageSize, 1, random().nextFloat());
    final long index = TestUtil.nextLong(random(), (long) Integer.MAX_VALUE, size - 1);
    writer.set(index, 2);
    assertEquals(2, writer.get(index));
    for (int i = 0; i < 1000000; ++i) {
      final long idx = TestUtil.nextLong(random(), 0, size);
      if (idx == index) {
        assertEquals(2, writer.get(idx));
      } else {
        assertEquals(0, writer.get(idx));
      }
    }
  }

  public void testEncodeDecode() {
    for (PackedInts.Format format : PackedInts.Format.values()) {
      for (int bpv = 1; bpv <= 64; ++bpv) {
        if (!format.isSupported(bpv)) {
          continue;
        }
        String msg = format + " " + bpv;

        final PackedInts.Encoder encoder =
            PackedInts.getEncoder(format, PackedInts.VERSION_CURRENT, bpv);
        final PackedInts.Decoder decoder =
            PackedInts.getDecoder(format, PackedInts.VERSION_CURRENT, bpv);
        final int longBlockCount = encoder.longBlockCount();
        final int longValueCount = encoder.longValueCount();
        final int byteBlockCount = encoder.byteBlockCount();
        final int byteValueCount = encoder.byteValueCount();
        assertEquals(longBlockCount, decoder.longBlockCount());
        assertEquals(longValueCount, decoder.longValueCount());
        assertEquals(byteBlockCount, decoder.byteBlockCount());
        assertEquals(byteValueCount, decoder.byteValueCount());

        final int longIterations = random().nextInt(100);
        final int byteIterations = longIterations * longValueCount / byteValueCount;
        assertEquals(longIterations * longValueCount, byteIterations * byteValueCount);
        final int blocksOffset = random().nextInt(100);
        final int valuesOffset = random().nextInt(100);
        final int blocksOffset2 = random().nextInt(100);
        final int blocksLen = longIterations * longBlockCount;

        // 1. generate random inputs
        final long[] blocks = new long[blocksOffset + blocksLen];
        for (int i = 0; i < blocks.length; ++i) {
          blocks[i] = random().nextLong();
          @SuppressWarnings("deprecation")
          PackedInts.Format PACKED_SINGLE_BLOCK = PackedInts.Format.PACKED_SINGLE_BLOCK;
          if (format == PACKED_SINGLE_BLOCK && 64 % bpv != 0) {
            // clear highest bits for packed
            final int toClear = 64 % bpv;
            blocks[i] = (blocks[i] << toClear) >>> toClear;
          }
        }

        // 2. decode
        final long[] values = new long[valuesOffset + longIterations * longValueCount];
        decoder.decode(blocks, blocksOffset, values, valuesOffset, longIterations);
        for (long value : values) {
          assertTrue(value <= PackedInts.maxValue(bpv));
        }
        // test decoding to int[]
        final int[] intValues;
        if (bpv <= 32) {
          intValues = new int[values.length];
          decoder.decode(blocks, blocksOffset, intValues, valuesOffset, longIterations);
          assertTrue(equals(intValues, values));
        } else {
          intValues = null;
        }

        // 3. re-encode
        final long[] blocks2 = new long[blocksOffset2 + blocksLen];
        encoder.encode(values, valuesOffset, blocks2, blocksOffset2, longIterations);
        assertArrayEquals(
            msg,
            ArrayUtil.copyOfSubArray(blocks, blocksOffset, blocks.length),
            ArrayUtil.copyOfSubArray(blocks2, blocksOffset2, blocks2.length));
        // test encoding from int[]
        if (bpv <= 32) {
          final long[] blocks3 = new long[blocks2.length];
          encoder.encode(intValues, valuesOffset, blocks3, blocksOffset2, longIterations);
          assertArrayEquals(msg, blocks2, blocks3);
        }

        // 4. byte[] decoding
        final byte[] byteBlocks = new byte[8 * blocks.length];
        ByteBuffer.wrap(byteBlocks).asLongBuffer().put(blocks);
        final long[] values2 = new long[valuesOffset + longIterations * longValueCount];
        decoder.decode(byteBlocks, blocksOffset * 8, values2, valuesOffset, byteIterations);
        for (long value : values2) {
          assertTrue(msg, value <= PackedInts.maxValue(bpv));
        }
        assertArrayEquals(msg, values, values2);
        // test decoding to int[]
        if (bpv <= 32) {
          final int[] intValues2 = new int[values2.length];
          decoder.decode(byteBlocks, blocksOffset * 8, intValues2, valuesOffset, byteIterations);
          assertTrue(msg, equals(intValues2, values2));
        }

        // 5. byte[] encoding
        final byte[] blocks3 = new byte[8 * (blocksOffset2 + blocksLen)];
        encoder.encode(values, valuesOffset, blocks3, 8 * blocksOffset2, byteIterations);
        assertEquals(msg, LongBuffer.wrap(blocks2), ByteBuffer.wrap(blocks3).asLongBuffer());
        // test encoding from int[]
        if (bpv <= 32) {
          final byte[] blocks4 = new byte[blocks3.length];
          encoder.encode(intValues, valuesOffset, blocks4, 8 * blocksOffset2, byteIterations);
          assertArrayEquals(msg, blocks3, blocks4);
        }
      }
    }
  }

  private static boolean equals(int[] ints, long[] longs) {
    if (ints.length != longs.length) {
      return false;
    }
    for (int i = 0; i < ints.length; ++i) {
      if ((ints[i] & 0xFFFFFFFFL) != longs[i]) {
        return false;
      }
    }
    return true;
  }

  enum DataType {
    PACKED,
    DELTA_PACKED,
    MONOTONIC
  }

  public void testPackedLongValuesOnZeros() {
    // Make sure that when all values are the same, they use 0 bits per value
    final int pageSize = 1 << TestUtil.nextInt(random(), 6, 20);
    final float acceptableOverheadRatio = random().nextFloat();

    assertEquals(
        PackedLongValues.packedBuilder(pageSize, acceptableOverheadRatio)
            .add(0)
            .build()
            .ramBytesUsed(),
        PackedLongValues.packedBuilder(pageSize, acceptableOverheadRatio)
            .add(0)
            .add(0)
            .build()
            .ramBytesUsed());

    final long l = random().nextLong();
    assertEquals(
        PackedLongValues.deltaPackedBuilder(pageSize, acceptableOverheadRatio)
            .add(l)
            .build()
            .ramBytesUsed(),
        PackedLongValues.deltaPackedBuilder(pageSize, acceptableOverheadRatio)
            .add(l)
            .add(l)
            .build()
            .ramBytesUsed());

    final long avg = random().nextInt(100);
    assertEquals(
        PackedLongValues.monotonicBuilder(pageSize, acceptableOverheadRatio)
            .add(l)
            .add(l + avg)
            .build()
            .ramBytesUsed(),
        PackedLongValues.monotonicBuilder(pageSize, acceptableOverheadRatio)
            .add(l)
            .add(l + avg)
            .add(l + 2 * avg)
            .build()
            .ramBytesUsed());
  }

  private static final class IgnoreNullReaderSingletonAccumulator
      extends RamUsageTester.Accumulator {
    @Override
    public long accumulateObject(
        Object o, long shallowSize, Map<Field, Object> fieldValues, Collection<Object> queue) {
      if (o == PackedInts.NullReader.forCount(PackedLongValues.DEFAULT_PAGE_SIZE)) {
        return 0;
      }
      return super.accumulateObject(o, shallowSize, fieldValues, queue);
    }
  }

  public void testPackedLongValues() {
    final long[] arr =
        new long[RandomNumbers.randomIntBetween(random(), 1, TEST_NIGHTLY ? 1000000 : 10000)];
    float[] ratioOptions = new float[] {PackedInts.DEFAULT, PackedInts.COMPACT, PackedInts.FAST};
    for (int bpv : new int[] {0, 1, 63, 64, RandomNumbers.randomIntBetween(random(), 2, 62)}) {
      for (DataType dataType : DataType.values()) {
        final int pageSize = 1 << TestUtil.nextInt(random(), 6, 20);
        float acceptableOverheadRatio =
            ratioOptions[TestUtil.nextInt(random(), 0, ratioOptions.length - 1)];
        PackedLongValues.Builder buf;
        final int inc;
        switch (dataType) {
          case PACKED:
            buf = PackedLongValues.packedBuilder(pageSize, acceptableOverheadRatio);
            inc = 0;
            break;
          case DELTA_PACKED:
            buf = PackedLongValues.deltaPackedBuilder(pageSize, acceptableOverheadRatio);
            inc = 0;
            break;
          case MONOTONIC:
            buf = PackedLongValues.monotonicBuilder(pageSize, acceptableOverheadRatio);
            inc = TestUtil.nextInt(random(), -1000, 1000);
            break;
          default:
            throw new RuntimeException("added a type and forgot to add it here?");
        }

        if (bpv == 0) {
          arr[0] = random().nextLong();
          for (int i = 1; i < arr.length; ++i) {
            arr[i] = arr[i - 1] + inc;
          }
        } else if (bpv == 64) {
          for (int i = 0; i < arr.length; ++i) {
            arr[i] = random().nextLong();
          }
        } else {
          final long minValue =
              TestUtil.nextLong(
                  random(), Long.MIN_VALUE, Long.MAX_VALUE - PackedInts.maxValue(bpv));
          for (int i = 0; i < arr.length; ++i) {
            arr[i] =
                minValue + inc * i + random().nextLong()
                    & PackedInts.maxValue(bpv); // _TestUtil.nextLong is too slow
          }
        }

        for (int i = 0; i < arr.length; ++i) {
          buf.add(arr[i]);
          if (rarely() && !TEST_NIGHTLY) {
            final long expectedBytesUsed =
                RamUsageTester.ramUsed(buf, new IgnoreNullReaderSingletonAccumulator());
            final long computedBytesUsed = buf.ramBytesUsed();
            assertEquals(expectedBytesUsed, computedBytesUsed);
          }
        }
        assertEquals(arr.length, buf.size());
        final PackedLongValues values = buf.build();
        expectThrows(
            IllegalStateException.class,
            () -> {
              buf.add(random().nextLong());
            });
        assertEquals(arr.length, values.size());

        for (int i = 0; i < arr.length; ++i) {
          assertEquals(arr[i], values.get(i));
        }

        final PackedLongValues.Iterator it = values.iterator();
        for (int i = 0; i < arr.length; ++i) {
          if (random().nextBoolean()) {
            assertTrue(it.hasNext());
          }
          assertEquals(arr[i], it.next());
        }
        assertFalse(it.hasNext());

        final long expectedBytesUsed =
            RamUsageTester.ramUsed(values, new IgnoreNullReaderSingletonAccumulator());
        final long computedBytesUsed = values.ramBytesUsed();
        assertEquals(expectedBytesUsed, computedBytesUsed);
      }
    }
  }

  public void testPackedInputOutput() throws IOException {
    final long[] longs = new long[random().nextInt(8192)];
    final int[] bitsPerValues = new int[longs.length];
    final boolean[] skip = new boolean[longs.length];
    for (int i = 0; i < longs.length; ++i) {
      final int bpv = RandomNumbers.randomIntBetween(random(), 1, 64);
      bitsPerValues[i] = random().nextBoolean() ? bpv : TestUtil.nextInt(random(), bpv, 64);
      if (bpv == 64) {
        longs[i] = random().nextLong();
      } else {
        longs[i] = TestUtil.nextLong(random(), 0, PackedInts.maxValue(bpv));
      }
      skip[i] = rarely();
    }

    final Directory dir = newDirectory();
    final IndexOutput out = dir.createOutput("out.bin", IOContext.DEFAULT);
    PackedDataOutput pout = new PackedDataOutput(out);
    long totalBits = 0;
    for (int i = 0; i < longs.length; ++i) {
      pout.writeLong(longs[i], bitsPerValues[i]);
      totalBits += bitsPerValues[i];
      if (skip[i]) {
        pout.flush();
        totalBits = 8 * (long) Math.ceil((double) totalBits / 8);
      }
    }
    pout.flush();
    assertEquals((long) Math.ceil((double) totalBits / 8), out.getFilePointer());
    out.close();
    final IndexInput in = dir.openInput("out.bin", IOContext.READONCE);
    final PackedDataInput pin = new PackedDataInput(in);
    for (int i = 0; i < longs.length; ++i) {
      assertEquals("" + i, longs[i], pin.readLong(bitsPerValues[i]));
      if (skip[i]) {
        pin.skipToNextByte();
      }
    }
    assertEquals((long) Math.ceil((double) totalBits / 8), in.getFilePointer());
    in.close();
    dir.close();
  }

  public void testBlockPackedReaderWriter() throws IOException {
    Random random = random();
    final int iters = atLeast(2);
    for (int iter = 0; iter < iters; ++iter) {
      final int blockSize = 1 << TestUtil.nextInt(random, 6, 18);
      final int valueCount;
      if (TEST_NIGHTLY) {
        valueCount = random.nextInt(1 << 18);
      } else {
        valueCount = random.nextInt(1 << 15);
      }
      final long[] values = new long[valueCount];
      long minValue = 0;
      int bpv = 0;
      for (int i = 0; i < valueCount; ++i) {
        if (i % blockSize == 0) {
          minValue = rarely(random) ? random.nextInt(256) : rarely(random) ? -5 : random.nextLong();
          bpv = random.nextInt(65);
        }
        if (bpv == 0) {
          values[i] = minValue;
        } else if (bpv == 64) {
          values[i] = random.nextLong();
        } else {
          values[i] = minValue + TestUtil.nextLong(random, 0, (1L << bpv) - 1);
        }
      }

      final Directory dir = newDirectory();
      final IndexOutput out = dir.createOutput("out.bin", IOContext.DEFAULT);
      final BlockPackedWriter writer = new BlockPackedWriter(out, blockSize);
      for (int i = 0; i < valueCount; ++i) {
        assertEquals(i, writer.ord());
        writer.add(values[i]);
      }
      assertEquals(valueCount, writer.ord());
      writer.finish();
      assertEquals(valueCount, writer.ord());
      final long fp = out.getFilePointer();
      out.close();

      IndexInput in1 = dir.openInput("out.bin", IOContext.DEFAULT);
      byte[] buf = new byte[(int) fp];
      in1.readBytes(buf, 0, (int) fp);
      in1.seek(0L);
      ByteArrayDataInput in2 = new ByteArrayDataInput(buf);
      final DataInput in = random.nextBoolean() ? in1 : in2;
      final BlockPackedReaderIterator it =
          new BlockPackedReaderIterator(in, PackedInts.VERSION_CURRENT, blockSize, valueCount);
      for (int i = 0; i < valueCount; ) {
        if (random.nextBoolean()) {
          assertEquals("" + i, values[i], it.next());
          ++i;
        } else {
          final LongsRef nextValues = it.next(TestUtil.nextInt(random, 1, 1024));
          for (int j = 0; j < nextValues.length; ++j) {
            assertEquals("" + (i + j), values[i + j], nextValues.longs[nextValues.offset + j]);
          }
          i += nextValues.length;
        }
        assertEquals(i, it.ord());
      }
      assertEquals(
          fp,
          in instanceof ByteArrayDataInput
              ? ((ByteArrayDataInput) in).getPosition()
              : ((IndexInput) in).getFilePointer());
      expectThrows(
          IOException.class,
          () -> {
            it.next();
          });

      if (in instanceof ByteArrayDataInput) {
        ((ByteArrayDataInput) in).setPosition(0);
      } else {
        ((IndexInput) in).seek(0L);
      }
      final BlockPackedReaderIterator it2 =
          new BlockPackedReaderIterator(in, PackedInts.VERSION_CURRENT, blockSize, valueCount);
      int i = 0;
      while (true) {
        final int skip = TestUtil.nextInt(random, 0, valueCount - i);
        it2.skip(skip);
        i += skip;
        assertEquals(i, it2.ord());
        if (i == valueCount) {
          break;
        } else {
          assertEquals(values[i], it2.next());
          ++i;
        }
      }
      assertEquals(
          fp,
          in instanceof ByteArrayDataInput
              ? ((ByteArrayDataInput) in).getPosition()
              : ((IndexInput) in).getFilePointer());
      expectThrows(
          IOException.class,
          () -> {
            it2.skip(1);
          });
      in1.close();
      dir.close();
    }
  }

  public void testMonotonicBlockPackedReaderWriter() throws IOException {
    final int iters = atLeast(2);
    for (int iter = 0; iter < iters; ++iter) {
      final int blockSize = 1 << TestUtil.nextInt(random(), 6, 18);
      final int valueCount = random().nextInt(1 << 18);
      final long[] values = new long[valueCount];
      if (valueCount > 0) {
        values[0] =
            random().nextBoolean() ? random().nextInt(10) : random().nextInt(Integer.MAX_VALUE);
        int maxDelta = random().nextInt(64);
        for (int i = 1; i < valueCount; ++i) {
          if (random().nextDouble() < 0.1d) {
            maxDelta = random().nextInt(64);
          }
          values[i] = Math.max(0, values[i - 1] + TestUtil.nextInt(random(), -16, maxDelta));
        }
      }

      final Directory dir = newDirectory();
      final IndexOutput out = dir.createOutput("out.bin", IOContext.DEFAULT);
      final MonotonicBlockPackedWriter writer = new MonotonicBlockPackedWriter(out, blockSize);
      for (int i = 0; i < valueCount; ++i) {
        assertEquals(i, writer.ord());
        writer.add(values[i]);
      }
      assertEquals(valueCount, writer.ord());
      writer.finish();
      assertEquals(valueCount, writer.ord());
      final long fp = out.getFilePointer();
      out.close();

      final IndexInput in = dir.openInput("out.bin", IOContext.DEFAULT);
      final MonotonicBlockPackedReader reader =
          MonotonicBlockPackedReader.of(in, PackedInts.VERSION_CURRENT, blockSize, valueCount);
      assertEquals(fp, in.getFilePointer());
      for (int i = 0; i < valueCount; ++i) {
        assertEquals("i=" + i, values[i], reader.get(i));
      }
      in.close();
      dir.close();
    }
  }

  @Nightly
  public void testBlockReaderOverflow() throws IOException {
    final long valueCount =
        TestUtil.nextLong(random(), 1L + Integer.MAX_VALUE, (long) Integer.MAX_VALUE * 2);
    final int blockSize = 1 << TestUtil.nextInt(random(), 20, 22);
    final Directory dir = newDirectory();
    final IndexOutput out = dir.createOutput("out.bin", IOContext.DEFAULT);
    final BlockPackedWriter writer = new BlockPackedWriter(out, blockSize);
    long value = random().nextInt() & 0xFFFFFFFFL;
    long valueOffset = TestUtil.nextLong(random(), 0, valueCount - 1);
    for (long i = 0; i < valueCount; ) {
      assertEquals(i, writer.ord());
      if ((i & (blockSize - 1)) == 0
          && (i + blockSize < valueOffset || i > valueOffset && i + blockSize < valueCount)) {
        writer.addBlockOfZeros();
        i += blockSize;
      } else if (i == valueOffset) {
        writer.add(value);
        ++i;
      } else {
        writer.add(0);
        ++i;
      }
    }
    writer.finish();
    out.close();
    final IndexInput in = dir.openInput("out.bin", IOContext.DEFAULT);
    final BlockPackedReaderIterator it =
        new BlockPackedReaderIterator(in, PackedInts.VERSION_CURRENT, blockSize, valueCount);
    it.skip(valueOffset);
    assertEquals(value, it.next());
    in.close();
    dir.close();
  }
}
