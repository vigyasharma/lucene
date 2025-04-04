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
package org.apache.lucene.sandbox.facet;

import static org.apache.lucene.facet.FacetsConfig.DEFAULT_INDEX_FIELD_NAME;

import com.carrotsearch.randomizedtesting.generators.RandomNumbers;
import java.io.IOException;
import java.util.List;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.DoubleDocValuesField;
import org.apache.lucene.document.DoublePoint;
import org.apache.lucene.document.LongPoint;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.SortedNumericDocValuesField;
import org.apache.lucene.facet.DrillDownQuery;
import org.apache.lucene.facet.DrillSideways;
import org.apache.lucene.facet.FacetField;
import org.apache.lucene.facet.FacetResult;
import org.apache.lucene.facet.FacetsConfig;
import org.apache.lucene.facet.LabelAndValue;
import org.apache.lucene.facet.MultiDoubleValuesSource;
import org.apache.lucene.facet.MultiLongValuesSource;
import org.apache.lucene.facet.range.DoubleRange;
import org.apache.lucene.facet.range.LongRange;
import org.apache.lucene.facet.range.Range;
import org.apache.lucene.facet.taxonomy.TaxonomyReader;
import org.apache.lucene.facet.taxonomy.directory.DirectoryTaxonomyReader;
import org.apache.lucene.facet.taxonomy.directory.DirectoryTaxonomyWriter;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.sandbox.facet.cutters.TaxonomyFacetsCutter;
import org.apache.lucene.sandbox.facet.cutters.ranges.DoubleRangeFacetCutter;
import org.apache.lucene.sandbox.facet.cutters.ranges.LongRangeFacetCutter;
import org.apache.lucene.sandbox.facet.labels.OrdToLabel;
import org.apache.lucene.sandbox.facet.labels.RangeOrdToLabel;
import org.apache.lucene.sandbox.facet.recorders.CountFacetRecorder;
import org.apache.lucene.search.DoubleValues;
import org.apache.lucene.search.DoubleValuesSource;
import org.apache.lucene.search.Explanation;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.LongValuesSource;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.MultiCollectorManager;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.index.RandomIndexWriter;
import org.apache.lucene.tests.search.DummyTotalHitCountCollector;
import org.apache.lucene.tests.util.TestUtil;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.NumericUtils;

/**
 * Test sandbox facet ranges. Mostly test cases from LongRangeFacetCounts adopted for sandbox
 * faceting.
 */
public class TestRangeFacet extends SandboxFacetTestCase {

  public void testBasicLong() throws Exception {
    Directory d = newDirectory();
    RandomIndexWriter w = new RandomIndexWriter(random(), d);
    Document doc = new Document();
    NumericDocValuesField field = new NumericDocValuesField("field", 0L);
    doc.add(field);
    for (long l = 0; l < 100; l++) {
      field.setLongValue(l);
      w.addDocument(doc);
    }

    // Also add Long.MAX_VALUE
    field.setLongValue(Long.MAX_VALUE);
    w.addDocument(doc);

    IndexReader r = w.getReader();
    w.close();

    IndexSearcher s = newSearcher(r);
    LongRange[] inputRanges =
        new LongRange[] {
          new LongRange("less than 10", 0L, true, 10L, false),
          new LongRange("less than or equal to 10", 0L, true, 10L, true),
          new LongRange("over 90", 90L, false, 100L, false),
          new LongRange("90 or above", 90L, true, 100L, false),
          new LongRange("over 1000", 1000L, false, Long.MAX_VALUE, true),
        };

    MultiLongValuesSource valuesSource = MultiLongValuesSource.fromLongField("field");
    LongRangeFacetCutter longRangeFacetCutter =
        LongRangeFacetCutter.create(valuesSource, inputRanges);
    CountFacetRecorder countRecorder = new CountFacetRecorder();

    FacetFieldCollectorManager<CountFacetRecorder> collectorManager =
        new FacetFieldCollectorManager<>(longRangeFacetCutter, countRecorder);
    s.search(new MatchAllDocsQuery(), collectorManager);
    OrdToLabel ordToLabel = new RangeOrdToLabel(inputRanges);

    assertEquals(
        "dim=field path=[] value=-2147483648 childCount=5\n  less than 10 (10)\n  less than or equal to 10 (11)\n  over 90 (9)\n  90 or above (10)\n  over 1000 (1)\n",
        getAllSortByOrd(getRangeOrdinals(inputRanges), countRecorder, "field", ordToLabel)
            .toString());

    r.close();
    d.close();
  }

  private int[] getRangeOrdinals(Range[] inputRanges) {
    // Naive method to get a list of facet ordinals for range facets,
    // it is used to get all range ordinals, including the ones that didn't match any docs.
    int[] result = new int[inputRanges.length];
    for (int i = 0; i < inputRanges.length; i++) {
      result[i] = i;
    }
    return result;
  }

  public void testBasicLongMultiValued() throws Exception {
    Directory d = newDirectory();
    RandomIndexWriter w = new RandomIndexWriter(random(), d);
    Document doc = new Document();
    // just index the same value twice each time and make sure we don't double count
    SortedNumericDocValuesField field1 = new SortedNumericDocValuesField("field", 0L);
    SortedNumericDocValuesField field2 = new SortedNumericDocValuesField("field", 0L);
    doc.add(field1);
    doc.add(field2);
    for (long l = 100; l < 200; l++) {
      field1.setLongValue(l);
      // Make second value sometimes smaller, sometimes bigger, and sometimes equal
      if (l % 3 == 0) {
        field2.setLongValue(l - 100);
      } else if (l % 3 == 1) {
        field2.setLongValue(l + 100);
      } else {
        field2.setLongValue(l);
      }
      w.addDocument(doc);
    }

    // Also add Long.MAX_VALUE
    field1.setLongValue(Long.MAX_VALUE);
    field2.setLongValue(Long.MAX_VALUE);
    w.addDocument(doc);

    IndexReader r = w.getReader();
    w.close();

    IndexSearcher s = newSearcher(r);

    ////////// Not overlapping ranges
    LongRange[] inputRanges =
        new LongRange[] {
          new LongRange("110-120", 110L, true, 120L, true),
          new LongRange("121-130", 121L, true, 130L, true),
        };

    MultiLongValuesSource valuesSource = MultiLongValuesSource.fromLongField("field");
    LongRangeFacetCutter longRangeFacetCutter =
        LongRangeFacetCutter.create(valuesSource, inputRanges);
    CountFacetRecorder countRecorder = new CountFacetRecorder();

    FacetFieldCollectorManager<CountFacetRecorder> collectorManager =
        new FacetFieldCollectorManager<>(longRangeFacetCutter, countRecorder);
    s.search(new MatchAllDocsQuery(), collectorManager);
    OrdToLabel ordToLabel = new RangeOrdToLabel(inputRanges);

    assertEquals(
        "dim=field path=[] value=-2147483648 childCount=2\n"
            + "  110-120 (11)\n"
            + "  121-130 (10)\n",
        getAllSortByOrd(getRangeOrdinals(inputRanges), countRecorder, "field", ordToLabel)
            .toString());

    ///////// Overlapping ranges
    inputRanges =
        new LongRange[] {
          new LongRange("110-120", 110L, true, 120L, true),
          new LongRange("115-125", 115L, true, 125L, true),
        };

    valuesSource = MultiLongValuesSource.fromLongField("field");
    longRangeFacetCutter = LongRangeFacetCutter.create(valuesSource, inputRanges);
    countRecorder = new CountFacetRecorder();

    collectorManager = new FacetFieldCollectorManager<>(longRangeFacetCutter, countRecorder);
    s.search(new MatchAllDocsQuery(), collectorManager);
    ordToLabel = new RangeOrdToLabel(inputRanges);

    assertEquals(
        "dim=field path=[] value=-2147483648 childCount=2\n"
            + "  110-120 (11)\n"
            + "  115-125 (11)\n",
        getAllSortByOrd(getRangeOrdinals(inputRanges), countRecorder, "field", ordToLabel)
            .toString());

    ////////// Multiple ranges (similar to original test)
    inputRanges =
        new LongRange[] {
          new LongRange("[100-110)", 100L, true, 110L, false),
          new LongRange("[100-110]", 100L, true, 110L, true),
          new LongRange("(190-200)", 190L, false, 200L, false),
          new LongRange("[190-200]", 190L, true, 200L, false),
          new LongRange("over 1000", 1000L, false, Long.MAX_VALUE, true)
        };

    valuesSource = MultiLongValuesSource.fromLongField("field");
    longRangeFacetCutter = LongRangeFacetCutter.create(valuesSource, inputRanges);
    countRecorder = new CountFacetRecorder();

    collectorManager = new FacetFieldCollectorManager<>(longRangeFacetCutter, countRecorder);
    s.search(new MatchAllDocsQuery(), collectorManager);
    ordToLabel = new RangeOrdToLabel(inputRanges);

    assertEquals(
        "dim=field path=[] value=-2147483648 childCount=5\n"
            + "  [100-110) (10)\n"
            + "  [100-110] (11)\n"
            + "  (190-200) (9)\n"
            + "  [190-200] (10)\n"
            + "  over 1000 (1)\n",
        getAllSortByOrd(getRangeOrdinals(inputRanges), countRecorder, "field", ordToLabel)
            .toString());

    r.close();
    d.close();
  }

  public void testBasicLongMultiValuedMixedSegmentTypes() throws Exception {
    Directory d = newDirectory();
    RandomIndexWriter w = new RandomIndexWriter(random(), d);
    SortedNumericDocValuesField field1 = new SortedNumericDocValuesField("field", 0L);
    SortedNumericDocValuesField field2 = new SortedNumericDocValuesField("field", 0L);
    // write docs as two segments (50 in each). the first segment will contain a mix of single- and
    // multi-value cases, while the second segment will be all single values.
    for (int l = 0; l < 100; l++) {
      field1.setLongValue(l);
      field2.setLongValue(l);
      Document doc = new Document();
      doc.add(field1);
      if (l == 0) {
        doc.add(field2);
      } else if (l < 50) {
        if (random().nextBoolean()) {
          doc.add(field2);
        }
      }
      w.addDocument(doc);
      if (l == 50) {
        w.commit();
      }
    }

    IndexReader r = w.getReader();
    w.close();

    IndexSearcher s = newSearcher(r);

    LongRange[] inputRanges =
        new LongRange[] {
          new LongRange("less than 10", 0L, true, 10L, false),
          new LongRange("less than or equal to 10", 0L, true, 10L, true),
          new LongRange("over 90", 90L, false, 100L, false),
          new LongRange("90 or above", 90L, true, 100L, false),
          new LongRange("over 1000", 1000L, false, Long.MAX_VALUE, true)
        };

    MultiLongValuesSource valuesSource = MultiLongValuesSource.fromLongField("field");
    LongRangeFacetCutter longRangeFacetCutter =
        LongRangeFacetCutter.create(valuesSource, inputRanges);
    CountFacetRecorder countRecorder = new CountFacetRecorder();

    FacetFieldCollectorManager<CountFacetRecorder> collectorManager =
        new FacetFieldCollectorManager<>(longRangeFacetCutter, countRecorder);
    s.search(new MatchAllDocsQuery(), collectorManager);
    OrdToLabel ordToLabel = new RangeOrdToLabel(inputRanges);

    assertEquals(
        "dim=field path=[] value=-2147483648 childCount=5\n  less than 10 (10)\n  less than or equal to 10 (11)\n  over 90 (9)\n  90 or above (10)\n  over 1000 (0)\n",
        getAllSortByOrd(getRangeOrdinals(inputRanges), countRecorder, "field", ordToLabel)
            .toString());

    r.close();
    d.close();
  }

  public void testLongMinMax() throws Exception {
    Directory d = newDirectory();
    RandomIndexWriter w = new RandomIndexWriter(random(), d);
    Document doc = new Document();
    NumericDocValuesField field = new NumericDocValuesField("field", 0L);
    doc.add(field);
    field.setLongValue(Long.MIN_VALUE);
    w.addDocument(doc);
    field.setLongValue(0);
    w.addDocument(doc);
    field.setLongValue(Long.MAX_VALUE);
    w.addDocument(doc);

    IndexReader r = w.getReader();
    w.close();

    IndexSearcher s = newSearcher(r);

    LongRange[] inputRanges =
        new LongRange[] {
          new LongRange("min", Long.MIN_VALUE, true, Long.MIN_VALUE, true),
          new LongRange("max", Long.MAX_VALUE, true, Long.MAX_VALUE, true),
          new LongRange("all0", Long.MIN_VALUE, true, Long.MAX_VALUE, true),
          new LongRange("all1", Long.MIN_VALUE, false, Long.MAX_VALUE, true),
          new LongRange("all2", Long.MIN_VALUE, true, Long.MAX_VALUE, false),
          new LongRange("all3", Long.MIN_VALUE, false, Long.MAX_VALUE, false)
        };

    MultiLongValuesSource valuesSource = MultiLongValuesSource.fromLongField("field");
    LongRangeFacetCutter longRangeFacetCutter =
        LongRangeFacetCutter.create(valuesSource, inputRanges);
    CountFacetRecorder countRecorder = new CountFacetRecorder();

    FacetFieldCollectorManager<CountFacetRecorder> collectorManager =
        new FacetFieldCollectorManager<>(longRangeFacetCutter, countRecorder);
    s.search(new MatchAllDocsQuery(), collectorManager);
    OrdToLabel ordToLabel = new RangeOrdToLabel(inputRanges);

    assertEquals(
        "dim=field path=[] value=-2147483648 childCount=6\n  min (1)\n  max (1)\n  all0 (3)\n  all1 (2)\n  all2 (2)\n  all3 (1)\n",
        getAllSortByOrd(getRangeOrdinals(inputRanges), countRecorder, "field", ordToLabel)
            .toString());

    r.close();
    d.close();
  }

  public void testOverlappedEndStart() throws Exception {
    Directory d = newDirectory();
    RandomIndexWriter w = new RandomIndexWriter(random(), d);
    Document doc = new Document();
    NumericDocValuesField field = new NumericDocValuesField("field", 0L);
    doc.add(field);
    for (long l = 0; l < 100; l++) {
      field.setLongValue(l);
      w.addDocument(doc);
    }
    field.setLongValue(Long.MAX_VALUE);
    w.addDocument(doc);

    IndexReader r = w.getReader();
    w.close();

    IndexSearcher s = newSearcher(r);

    LongRange[] inputRanges =
        new LongRange[] {
          new LongRange("0-10", 0L, true, 10L, true),
          new LongRange("10-20", 10L, true, 20L, true),
          new LongRange("20-30", 20L, true, 30L, true),
          new LongRange("30-40", 30L, true, 40L, true)
        };

    MultiLongValuesSource valuesSource = MultiLongValuesSource.fromLongField("field");
    LongRangeFacetCutter longRangeFacetCutter =
        LongRangeFacetCutter.create(valuesSource, inputRanges);
    CountFacetRecorder countRecorder = new CountFacetRecorder();

    FacetFieldCollectorManager<CountFacetRecorder> collectorManager =
        new FacetFieldCollectorManager<>(longRangeFacetCutter, countRecorder);
    s.search(new MatchAllDocsQuery(), collectorManager);
    OrdToLabel ordToLabel = new RangeOrdToLabel(inputRanges);

    assertEquals(
        "dim=field path=[] value=-2147483648 childCount=4\n  0-10 (11)\n  10-20 (11)\n  20-30 (11)\n  30-40 (11)\n",
        getAllSortByOrd(getRangeOrdinals(inputRanges), countRecorder, "field", ordToLabel)
            .toString());

    r.close();
    d.close();
  }

  public void testEmptyRangesSingleValued() throws Exception {
    Directory d = newDirectory();
    RandomIndexWriter w = new RandomIndexWriter(random(), d);
    Document doc = new Document();
    NumericDocValuesField field = new NumericDocValuesField("field", 0L);
    doc.add(field);
    for (long l = 0; l < 100; l++) {
      field.setLongValue(l);
      w.addDocument(doc);
    }

    IndexReader r = w.getReader();
    w.close();

    IndexSearcher s = newSearcher(r);

    LongRange[] inputRanges = new LongRange[0];

    MultiLongValuesSource valuesSource = MultiLongValuesSource.fromLongField("field");
    LongRangeFacetCutter longRangeFacetCutter =
        LongRangeFacetCutter.create(valuesSource, inputRanges);
    CountFacetRecorder countRecorder = new CountFacetRecorder();

    FacetFieldCollectorManager<CountFacetRecorder> collectorManager =
        new FacetFieldCollectorManager<>(longRangeFacetCutter, countRecorder);
    s.search(new MatchAllDocsQuery(), collectorManager);
    OrdToLabel ordToLabel = new RangeOrdToLabel(inputRanges);

    assertEquals(
        "dim=field path=[] value=-2147483648 childCount=0\n",
        getAllSortByOrd(getRangeOrdinals(inputRanges), countRecorder, "field", ordToLabel)
            .toString());

    r.close();
    d.close();
  }

  public void testEmptyRangesMultiValued() throws Exception {
    Directory d = newDirectory();
    RandomIndexWriter w = new RandomIndexWriter(random(), d);
    Document doc = new Document();
    SortedNumericDocValuesField field1 = new SortedNumericDocValuesField("field", 0L);
    SortedNumericDocValuesField field2 = new SortedNumericDocValuesField("field", 0L);
    doc.add(field1);
    doc.add(field2);
    for (long l = 0; l < 100; l++) {
      field1.setLongValue(l);
      field2.setLongValue(l);
      w.addDocument(doc);
    }

    IndexReader r = w.getReader();
    w.close();

    IndexSearcher s = newSearcher(r);

    LongRange[] inputRanges = new LongRange[0];

    MultiLongValuesSource valuesSource = MultiLongValuesSource.fromLongField("field");
    LongRangeFacetCutter longRangeFacetCutter =
        LongRangeFacetCutter.create(valuesSource, inputRanges);
    CountFacetRecorder countRecorder = new CountFacetRecorder();

    FacetFieldCollectorManager<CountFacetRecorder> collectorManager =
        new FacetFieldCollectorManager<>(longRangeFacetCutter, countRecorder);
    s.search(new MatchAllDocsQuery(), collectorManager);
    OrdToLabel ordToLabel = new RangeOrdToLabel(inputRanges);

    assertEquals(
        "dim=field path=[] value=-2147483648 childCount=0\n",
        getAllSortByOrd(getRangeOrdinals(inputRanges), countRecorder, "field", ordToLabel)
            .toString());

    r.close();
    d.close();
  }

  /**
   * Tests single request that mixes Range and non-Range faceting, with DrillSideways and taxonomy.
   */
  public void testMixedRangeAndNonRangeTaxonomy() throws Exception {
    Directory d = newDirectory();
    RandomIndexWriter w = new RandomIndexWriter(random(), d);
    Directory td = newDirectory();
    DirectoryTaxonomyWriter tw = new DirectoryTaxonomyWriter(td, IndexWriterConfig.OpenMode.CREATE);

    FacetsConfig config = new FacetsConfig();

    for (long l = 0; l < 100; l++) {
      Document doc = new Document();
      // For computing range facet counts:
      doc.add(new NumericDocValuesField("field", l));
      // For drill down by numeric range:
      doc.add(new LongPoint("field", l));

      if ((l & 3) == 0) {
        doc.add(new FacetField("dim", "a"));
      } else {
        doc.add(new FacetField("dim", "b"));
      }
      w.addDocument(config.build(tw, doc));
    }

    final IndexReader r = w.getReader();
    final TaxonomyReader tr = new DirectoryTaxonomyReader(tw);

    IndexSearcher s = getNewSearcherForDrillSideways(r);
    // DrillSideways requires the entire range of docs to be scored at once, so it doesn't support
    // timeouts whose implementation scores one window of doc IDs at a time.
    s.setTimeout(null);

    if (VERBOSE) {
      System.out.println("TEST: searcher=" + s);
    }

    DrillSideways ds =
        new DrillSideways(s, config, tr) {
          @Override
          protected boolean scoreSubDocsAtOnce() {
            return random().nextBoolean();
          }
        };

    // Data for range facets
    LongRange[] inputRanges =
        new LongRange[] {
          new LongRange("less than 10", 0L, true, 10L, false),
          new LongRange("less than or equal to 10", 0L, true, 10L, true),
          new LongRange("over 90", 90L, false, 100L, false),
          new LongRange("90 or above", 90L, true, 100L, false),
          new LongRange("over 1000", 1000L, false, Long.MAX_VALUE, false)
        };
    MultiLongValuesSource valuesSource = MultiLongValuesSource.fromLongField("field");
    LongRangeFacetCutter fieldCutter = LongRangeFacetCutter.create(valuesSource, inputRanges);
    CountFacetRecorder fieldCountRecorder = new CountFacetRecorder();
    FacetFieldCollectorManager<CountFacetRecorder> fieldCollectorManager =
        new FacetFieldCollectorManager<>(fieldCutter, fieldCountRecorder);
    OrdToLabel fieldOrdToLabel = new RangeOrdToLabel(inputRanges);

    // Data for taxonomy facets
    TaxonomyFacetsCutter dimCutter = new TaxonomyFacetsCutter(DEFAULT_INDEX_FIELD_NAME, config, tr);
    CountFacetRecorder dimCountRecorder = new CountFacetRecorder();
    FacetFieldCollectorManager<CountFacetRecorder> dimCollectorManager =
        new FacetFieldCollectorManager<>(dimCutter, dimCountRecorder);

    MultiCollectorManager collectorManager =
        new MultiCollectorManager(fieldCollectorManager, dimCollectorManager);

    ////// First search, no drill-downs:
    DrillDownQuery ddq = new DrillDownQuery(config);
    ds.search(ddq, collectorManager, List.of());

    // assertEquals(100, dsr.hits.totalHits.value());
    assertEquals(
        "dim=dim path=[] value=-2147483648 childCount=2\n  b (75)\n  a (25)\n",
        getTopChildrenByCount(dimCountRecorder, tr, 10, "dim").toString());
    assertEquals(
        "dim=field path=[] value=-2147483648 childCount=5\n  less than 10 (10)\n  less than or equal to 10 (11)\n  over 90 (9)\n  90 or above (10)\n  over 1000 (0)\n",
        getAllSortByOrd(getRangeOrdinals(inputRanges), fieldCountRecorder, "field", fieldOrdToLabel)
            .toString());

    ////// Second search, drill down on dim=b:
    fieldCountRecorder = new CountFacetRecorder();
    fieldCollectorManager = new FacetFieldCollectorManager<>(fieldCutter, fieldCountRecorder);
    dimCountRecorder = new CountFacetRecorder();
    dimCollectorManager = new FacetFieldCollectorManager<>(dimCutter, dimCountRecorder);
    ddq = new DrillDownQuery(config);
    ddq.add("dim", "b");
    ds.search(ddq, fieldCollectorManager, List.of(dimCollectorManager));

    // assertEquals(75, dsr.hits.totalHits.value());
    assertEquals(
        "dim=dim path=[] value=-2147483648 childCount=2\n  b (75)\n  a (25)\n",
        getTopChildrenByCount(dimCountRecorder, tr, 10, "dim").toString());
    assertEquals(
        "dim=field path=[] value=-2147483648 childCount=5\n  less than 10 (7)\n  less than or equal to 10 (8)\n  over 90 (7)\n  90 or above (8)\n  over 1000 (0)\n",
        getAllSortByOrd(getRangeOrdinals(inputRanges), fieldCountRecorder, "field", fieldOrdToLabel)
            .toString());

    ////// Third search, drill down on "less than or equal to 10":
    fieldCountRecorder = new CountFacetRecorder();
    fieldCollectorManager = new FacetFieldCollectorManager<>(fieldCutter, fieldCountRecorder);
    dimCountRecorder = new CountFacetRecorder();
    dimCollectorManager = new FacetFieldCollectorManager<>(dimCutter, dimCountRecorder);
    ddq = new DrillDownQuery(config);
    ddq.add("field", LongPoint.newRangeQuery("field", 0L, 10L));
    ds.search(ddq, dimCollectorManager, List.of(fieldCollectorManager));

    // assertEquals(11, dsr.hits.totalHits.value());
    assertEquals(
        "dim=dim path=[] value=-2147483648 childCount=2\n  b (8)\n  a (3)\n",
        getTopChildrenByCount(dimCountRecorder, tr, 10, "dim").toString());
    assertEquals(
        "dim=field path=[] value=-2147483648 childCount=5\n  less than 10 (10)\n  less than or equal to 10 (11)\n  over 90 (9)\n  90 or above (10)\n  over 1000 (0)\n",
        getAllSortByOrd(getRangeOrdinals(inputRanges), fieldCountRecorder, "field", fieldOrdToLabel)
            .toString());

    w.close();
    IOUtils.close(tw, tr, td, r, d);
  }

  public void testBasicDouble() throws Exception {
    Directory d = newDirectory();
    RandomIndexWriter w = new RandomIndexWriter(random(), d);
    Document doc = new Document();
    DoubleDocValuesField field = new DoubleDocValuesField("field", 0.0);
    doc.add(field);
    for (int i = 0; i < 100; i++) {
      field.setDoubleValue(i);
      w.addDocument(doc);
    }

    IndexReader r = w.getReader();

    IndexSearcher s = newSearcher(r);
    DoubleRange[] inputRanges =
        new DoubleRange[] {
          new DoubleRange("less than 10", 0.0, true, 10.0, false),
          new DoubleRange("less than or equal to 10", 0.0, true, 10.0, true),
          new DoubleRange("over 90", 90.0, false, 100.0, false),
          new DoubleRange("90 or above", 90.0, true, 100.0, false),
          new DoubleRange("over 1000", 1000.0, false, Double.POSITIVE_INFINITY, false)
        };

    MultiDoubleValuesSource valuesSource = MultiDoubleValuesSource.fromDoubleField("field");
    DoubleRangeFacetCutter doubleRangeFacetCutter =
        new DoubleRangeFacetCutter(valuesSource, inputRanges);
    CountFacetRecorder countRecorder = new CountFacetRecorder();

    FacetFieldCollectorManager<CountFacetRecorder> collectorManager =
        new FacetFieldCollectorManager<>(doubleRangeFacetCutter, countRecorder);
    s.search(new MatchAllDocsQuery(), collectorManager);
    OrdToLabel ordToLabel = new RangeOrdToLabel(inputRanges);

    assertEquals(
        "dim=field path=[] value=-2147483648 childCount=5\n  less than 10 (10)\n  less than or equal to 10 (11)\n  over 90 (9)\n  90 or above (10)\n  over 1000 (0)\n",
        getAllSortByOrd(getRangeOrdinals(inputRanges), countRecorder, "field", ordToLabel)
            .toString());

    w.close();
    IOUtils.close(r, d);
  }

  public void testBasicDoubleMultiValued() throws Exception {
    Directory d = newDirectory();
    RandomIndexWriter w = new RandomIndexWriter(random(), d);
    Document doc = new Document();
    // index the same value twice and make sure we don't double count
    SortedNumericDocValuesField field1 = new SortedNumericDocValuesField("field", 0);
    SortedNumericDocValuesField field2 = new SortedNumericDocValuesField("field", 0);
    doc.add(field1);
    doc.add(field2);
    for (int i = 0; i < 100; i++) {
      field1.setLongValue(NumericUtils.doubleToSortableLong(i));
      field2.setLongValue(NumericUtils.doubleToSortableLong(i));
      w.addDocument(doc);
    }

    IndexReader r = w.getReader();

    IndexSearcher s = newSearcher(r);
    DoubleRange[] inputRanges =
        new DoubleRange[] {
          new DoubleRange("less than 10", 0.0, true, 10.0, false),
          new DoubleRange("less than or equal to 10", 0.0, true, 10.0, true),
          new DoubleRange("over 90", 90.0, false, 100.0, false),
          new DoubleRange("90 or above", 90.0, true, 100.0, false),
          new DoubleRange("over 1000", 1000.0, false, Double.POSITIVE_INFINITY, false)
        };

    MultiDoubleValuesSource valuesSource = MultiDoubleValuesSource.fromDoubleField("field");
    DoubleRangeFacetCutter doubleRangeFacetCutter =
        new DoubleRangeFacetCutter(valuesSource, inputRanges);
    CountFacetRecorder countRecorder = new CountFacetRecorder();

    FacetFieldCollectorManager<CountFacetRecorder> collectorManager =
        new FacetFieldCollectorManager<>(doubleRangeFacetCutter, countRecorder);
    s.search(new MatchAllDocsQuery(), collectorManager);
    OrdToLabel ordToLabel = new RangeOrdToLabel(inputRanges);

    assertEquals(
        "dim=field path=[] value=-2147483648 childCount=5\n  less than 10 (10)\n  less than or equal to 10 (11)\n  over 90 (9)\n  90 or above (10)\n  over 1000 (0)\n",
        getAllSortByOrd(getRangeOrdinals(inputRanges), countRecorder, "field", ordToLabel)
            .toString());

    w.close();
    IOUtils.close(r, d);
  }

  public void testBasicDoubleMultiValuedMixedSegmentTypes() throws Exception {
    Directory d = newDirectory();
    RandomIndexWriter w = new RandomIndexWriter(random(), d);
    SortedNumericDocValuesField field1 = new SortedNumericDocValuesField("field", 0L);
    SortedNumericDocValuesField field2 = new SortedNumericDocValuesField("field", 0L);
    // write docs as two segments (50 in each). the first segment will contain a mix of single- and
    // multi-value cases, while the second segment will be all single values.
    for (int l = 0; l < 100; l++) {
      field1.setLongValue(NumericUtils.doubleToSortableLong(l));
      field2.setLongValue(NumericUtils.doubleToSortableLong(l));
      Document doc = new Document();
      doc.add(field1);
      if (l == 0) {
        doc.add(field2);
      } else if (l < 50) {
        if (random().nextBoolean()) {
          doc.add(field2);
        }
      }
      w.addDocument(doc);
      if (l == 50) {
        w.commit();
      }
    }

    IndexReader r = w.getReader();
    w.close();

    IndexSearcher s = newSearcher(r);

    DoubleRange[] inputRanges =
        new DoubleRange[] {
          new DoubleRange("less than 10", 0.0, true, 10.0, false),
          new DoubleRange("less than or equal to 10", 0.0, true, 10.0, true),
          new DoubleRange("over 90", 90.0, false, 100.0, false),
          new DoubleRange("90 or above", 90.0, true, 100.0, false),
          new DoubleRange("over 1000", 1000.0, false, Double.POSITIVE_INFINITY, false)
        };

    MultiDoubleValuesSource valuesSource = MultiDoubleValuesSource.fromDoubleField("field");
    DoubleRangeFacetCutter doubleRangeFacetCutter =
        new DoubleRangeFacetCutter(valuesSource, inputRanges);
    CountFacetRecorder countRecorder = new CountFacetRecorder();

    FacetFieldCollectorManager<CountFacetRecorder> collectorManager =
        new FacetFieldCollectorManager<>(doubleRangeFacetCutter, countRecorder);
    s.search(new MatchAllDocsQuery(), collectorManager);
    OrdToLabel ordToLabel = new RangeOrdToLabel(inputRanges);

    assertEquals(
        "dim=field path=[] value=-2147483648 childCount=5\n  less than 10 (10)\n  less than or equal to 10 (11)\n  over 90 (9)\n  90 or above (10)\n  over 1000 (0)\n",
        getAllSortByOrd(getRangeOrdinals(inputRanges), countRecorder, "field", ordToLabel)
            .toString());
    r.close();
    d.close();
  }

  public void testRandomLongsSingleValued() throws Exception {
    Directory dir = newDirectory();
    RandomIndexWriter w = new RandomIndexWriter(random(), dir);

    int numDocs = atLeast(1000);
    if (VERBOSE) {
      System.out.println("TEST: numDocs=" + numDocs);
    }
    long[] values = new long[numDocs];
    long minValue = Long.MAX_VALUE;
    long maxValue = Long.MIN_VALUE;
    for (int i = 0; i < numDocs; i++) {
      Document doc = new Document();
      long v = random().nextLong();
      values[i] = v;
      doc.add(new NumericDocValuesField("field", v));
      doc.add(new LongPoint("field", v));
      w.addDocument(doc);
      minValue = Math.min(minValue, v);
      maxValue = Math.max(maxValue, v);
    }
    IndexReader r = w.getReader();

    IndexSearcher s = newSearcher(r, false);
    FacetsConfig config = new FacetsConfig();

    int numIters = atLeast(10);
    for (int iter = 0; iter < numIters; iter++) {
      if (VERBOSE) {
        System.out.println("TEST: iter=" + iter);
      }
      int numRange = TestUtil.nextInt(random(), 1, 100);
      LongRange[] ranges = new LongRange[numRange];
      int[] expectedCounts = new int[numRange];
      long minAcceptedValue = Long.MAX_VALUE;
      long maxAcceptedValue = Long.MIN_VALUE;
      for (int rangeID = 0; rangeID < numRange; rangeID++) {
        long min;
        if (rangeID > 0 && random().nextInt(10) == 7) {
          // Use an existing boundary:
          LongRange prevRange = ranges[random().nextInt(rangeID)];
          if (random().nextBoolean()) {
            min = prevRange.min;
          } else {
            min = prevRange.max;
          }
        } else {
          min = random().nextLong();
        }
        long max;
        if (rangeID > 0 && random().nextInt(10) == 7) {
          // Use an existing boundary:
          LongRange prevRange = ranges[random().nextInt(rangeID)];
          if (random().nextBoolean()) {
            max = prevRange.min;
          } else {
            max = prevRange.max;
          }
        } else {
          max = random().nextLong();
        }

        if (min > max) {
          long x = min;
          min = max;
          max = x;
        }
        boolean minIncl;
        boolean maxIncl;

        // NOTE: max - min >= 0 is here to handle the common overflow case!
        if (max - min >= 0 && max - min < 2) {
          // If max == min or max == min+1, we always do inclusive, else we might pass an empty
          // range and hit exc from LongRange's ctor:
          minIncl = true;
          maxIncl = true;
        } else {
          minIncl = random().nextBoolean();
          maxIncl = random().nextBoolean();
        }
        ranges[rangeID] = new LongRange("r" + rangeID, min, minIncl, max, maxIncl);
        if (VERBOSE) {
          System.out.println("  range " + rangeID + ": " + ranges[rangeID]);
        }

        // Do "slow but hopefully correct" computation of
        // expected count:
        for (int i = 0; i < numDocs; i++) {
          boolean accept = true;
          if (minIncl) {
            accept &= values[i] >= min;
          } else {
            accept &= values[i] > min;
          }
          if (maxIncl) {
            accept &= values[i] <= max;
          } else {
            accept &= values[i] < max;
          }
          if (accept) {
            expectedCounts[rangeID]++;
            minAcceptedValue = Math.min(minAcceptedValue, values[i]);
            maxAcceptedValue = Math.max(maxAcceptedValue, values[i]);
          }
        }
      }

      // TODO: fastMatchQuery functionality is not implemented for sandbox faceting yet, do we need
      // it?
      /*Query fastMatchQuery;
      if (random().nextBoolean()) {
        if (random().nextBoolean()) {
          fastMatchQuery = LongPoint.newRangeQuery("field", minValue, maxValue);
        } else {
          fastMatchQuery = LongPoint.newRangeQuery("field", minAcceptedValue, maxAcceptedValue);
        }
      } else {
        fastMatchQuery = null;
      }*/

      final MultiLongValuesSource mvs;
      if (random().nextBoolean()) {
        LongValuesSource vs = LongValuesSource.fromLongField("field");
        mvs = MultiLongValuesSource.fromSingleValued(vs);
      } else {
        mvs = MultiLongValuesSource.fromLongField("field");
      }

      LongRangeFacetCutter longRangeFacetCutter = LongRangeFacetCutter.create(mvs, ranges);
      CountFacetRecorder countRecorder = new CountFacetRecorder();

      FacetFieldCollectorManager<CountFacetRecorder> collectorManager =
          new FacetFieldCollectorManager<>(longRangeFacetCutter, countRecorder);
      s.search(new MatchAllDocsQuery(), collectorManager);

      OrdToLabel ordToLabel = new RangeOrdToLabel(ranges);
      FacetResult result =
          getAllSortByOrd(getRangeOrdinals(ranges), countRecorder, "field", ordToLabel);
      assertEquals(numRange, result.labelValues.length);
      for (int rangeID = 0; rangeID < numRange; rangeID++) {
        if (VERBOSE) {
          System.out.println("  range " + rangeID + " expectedCount=" + expectedCounts[rangeID]);
        }
        LabelAndValue subNode = result.labelValues[rangeID];
        assertEquals("r" + rangeID, subNode.label);
        assertEquals(expectedCounts[rangeID], subNode.value.intValue());

        LongRange range = ranges[rangeID];

        // Test drill-down:
        DrillDownQuery ddq = new DrillDownQuery(config);
        ddq.add("field", LongPoint.newRangeQuery("field", range.min, range.max));
        assertEquals(expectedCounts[rangeID], s.count(ddq));
      }
    }

    w.close();
    IOUtils.close(r, dir);
  }

  public void testRandomLongsMultiValued() throws Exception {
    Directory dir = newDirectory();
    RandomIndexWriter w = new RandomIndexWriter(random(), dir);

    int numDocs = atLeast(1000);
    if (VERBOSE) {
      System.out.println("TEST: numDocs=" + numDocs);
    }
    long[][] values = new long[numDocs][];
    long minValue = Long.MAX_VALUE;
    long maxValue = Long.MIN_VALUE;
    for (int i = 0; i < numDocs; i++) {
      Document doc = new Document();
      int numVals = RandomNumbers.randomIntBetween(random(), 1, 50);
      if (random().nextInt(10) == 0) {
        numVals = 1; // make sure we have ample testing of single-value cases
      }
      values[i] = new long[numVals];
      for (int j = 0; j < numVals; j++) {
        long v = random().nextLong();
        values[i][j] = v;
        doc.add(new SortedNumericDocValuesField("field", v));
        doc.add(new LongPoint("field", v));
        minValue = Math.min(minValue, v);
        maxValue = Math.max(maxValue, v);
      }
      w.addDocument(doc);
    }
    IndexReader r = w.getReader();

    IndexSearcher s = newSearcher(r, false);
    FacetsConfig config = new FacetsConfig();

    int numIters = atLeast(10);
    for (int iter = 0; iter < numIters; iter++) {
      if (VERBOSE) {
        System.out.println("TEST: iter=" + iter);
      }
      int numRange = TestUtil.nextInt(random(), 1, 100);
      LongRange[] ranges = new LongRange[numRange];
      int[] expectedCounts = new int[numRange];
      long minAcceptedValue = Long.MAX_VALUE;
      long maxAcceptedValue = Long.MIN_VALUE;
      for (int rangeID = 0; rangeID < numRange; rangeID++) {
        long min;
        if (rangeID > 0 && random().nextInt(10) == 7) {
          // Use an existing boundary:
          LongRange prevRange = ranges[random().nextInt(rangeID)];
          if (random().nextBoolean()) {
            min = prevRange.min;
          } else {
            min = prevRange.max;
          }
        } else {
          min = random().nextLong();
        }
        long max;
        if (rangeID > 0 && random().nextInt(10) == 7) {
          // Use an existing boundary:
          LongRange prevRange = ranges[random().nextInt(rangeID)];
          if (random().nextBoolean()) {
            max = prevRange.min;
          } else {
            max = prevRange.max;
          }
        } else {
          max = random().nextLong();
        }

        if (min > max) {
          long x = min;
          min = max;
          max = x;
        }
        boolean minIncl;
        boolean maxIncl;

        // NOTE: max - min >= 0 is here to handle the common overflow case!
        if (max - min >= 0 && max - min < 2) {
          // If max == min or max == min+1, we always do inclusive, else we might pass an empty
          // range and hit exc from LongRange's ctor:
          minIncl = true;
          maxIncl = true;
        } else {
          minIncl = random().nextBoolean();
          maxIncl = random().nextBoolean();
        }
        ranges[rangeID] = new LongRange("r" + rangeID, min, minIncl, max, maxIncl);
        if (VERBOSE) {
          System.out.println("  range " + rangeID + ": " + ranges[rangeID]);
        }

        // Do "slow but hopefully correct" computation of
        // expected count:
        for (int i = 0; i < numDocs; i++) {
          for (int j = 0; j < values[i].length; j++) {
            boolean accept = true;
            if (minIncl) {
              accept &= values[i][j] >= min;
            } else {
              accept &= values[i][j] > min;
            }
            if (maxIncl) {
              accept &= values[i][j] <= max;
            } else {
              accept &= values[i][j] < max;
            }
            if (accept) {
              expectedCounts[rangeID]++;
              minAcceptedValue = Math.min(minAcceptedValue, values[i][j]);
              maxAcceptedValue = Math.max(maxAcceptedValue, values[i][j]);
              break; // ensure each doc can contribute at most 1 count to each range
            }
          }
        }
      }

      // TODO: fastMatchQuery functionality is not implemented for sandbox faceting yet, do we need
      // it?
      /*Query fastMatchQuery;
      if (random().nextBoolean()) {
        if (random().nextBoolean()) {
          fastMatchQuery = LongPoint.newRangeQuery("field", minValue, maxValue);
        } else {
          fastMatchQuery = LongPoint.newRangeQuery("field", minAcceptedValue, maxAcceptedValue);
        }
      } else {
        fastMatchQuery = null;
      }*/
      final MultiLongValuesSource mvs = MultiLongValuesSource.fromLongField("field");

      LongRangeFacetCutter longRangeFacetCutter = LongRangeFacetCutter.create(mvs, ranges);
      CountFacetRecorder countRecorder = new CountFacetRecorder();

      FacetFieldCollectorManager<CountFacetRecorder> collectorManager =
          new FacetFieldCollectorManager<>(longRangeFacetCutter, countRecorder);
      s.search(new MatchAllDocsQuery(), collectorManager);

      OrdToLabel ordToLabel = new RangeOrdToLabel(ranges);
      FacetResult result =
          getAllSortByOrd(getRangeOrdinals(ranges), countRecorder, "field", ordToLabel);
      assertEquals(numRange, result.labelValues.length);
      for (int rangeID = 0; rangeID < numRange; rangeID++) {
        if (VERBOSE) {
          System.out.println("  range " + rangeID + " expectedCount=" + expectedCounts[rangeID]);
        }
        LabelAndValue subNode = result.labelValues[rangeID];
        assertEquals("r" + rangeID, subNode.label);
        assertEquals(expectedCounts[rangeID], subNode.value.intValue());

        LongRange range = ranges[rangeID];

        // Test drill-down:
        DrillDownQuery ddq = new DrillDownQuery(config);
        if (random().nextBoolean()) {
          ddq.add("field", LongPoint.newRangeQuery("field", range.min, range.max));
        } else {
          ddq.add(
              "field",
              SortedNumericDocValuesField.newSlowRangeQuery("field", range.min, range.max));
        }
        assertEquals(expectedCounts[rangeID], s.count(ddq));
      }
    }

    w.close();
    IOUtils.close(r, dir);
  }

  public void testRandomDoublesSingleValued() throws Exception {
    Directory dir = newDirectory();
    RandomIndexWriter w = new RandomIndexWriter(random(), dir);

    int numDocs = atLeast(1000);
    double[] values = new double[numDocs];
    double minValue = Double.POSITIVE_INFINITY;
    double maxValue = Double.NEGATIVE_INFINITY;
    for (int i = 0; i < numDocs; i++) {
      Document doc = new Document();
      double v = random().nextDouble();
      values[i] = v;
      doc.add(new DoubleDocValuesField("field", v));
      doc.add(new DoublePoint("field", v));
      w.addDocument(doc);
      minValue = Math.min(minValue, v);
      maxValue = Math.max(maxValue, v);
    }
    IndexReader r = w.getReader();

    IndexSearcher s = newSearcher(r, false);
    FacetsConfig config = new FacetsConfig();

    int numIters = atLeast(10);
    for (int iter = 0; iter < numIters; iter++) {
      if (VERBOSE) {
        System.out.println("TEST: iter=" + iter);
      }
      int numRange = TestUtil.nextInt(random(), 1, 5);
      DoubleRange[] ranges = new DoubleRange[numRange];
      int[] expectedCounts = new int[numRange];
      double minAcceptedValue = Double.POSITIVE_INFINITY;
      double maxAcceptedValue = Double.NEGATIVE_INFINITY;
      for (int rangeID = 0; rangeID < numRange; rangeID++) {
        double min;
        if (rangeID > 0 && random().nextInt(10) == 7) {
          // Use an existing boundary:
          DoubleRange prevRange = ranges[random().nextInt(rangeID)];
          if (random().nextBoolean()) {
            min = prevRange.min;
          } else {
            min = prevRange.max;
          }
        } else {
          min = random().nextDouble();
        }
        double max;
        if (rangeID > 0 && random().nextInt(10) == 7) {
          // Use an existing boundary:
          DoubleRange prevRange = ranges[random().nextInt(rangeID)];
          if (random().nextBoolean()) {
            max = prevRange.min;
          } else {
            max = prevRange.max;
          }
        } else {
          max = random().nextDouble();
        }

        if (min > max) {
          double x = min;
          min = max;
          max = x;
        }

        boolean minIncl;
        boolean maxIncl;

        long minAsLong = NumericUtils.doubleToSortableLong(min);
        long maxAsLong = NumericUtils.doubleToSortableLong(max);
        // NOTE: maxAsLong - minAsLong >= 0 is here to handle the common overflow case!
        if (maxAsLong - minAsLong >= 0 && maxAsLong - minAsLong < 2) {
          minIncl = true;
          maxIncl = true;
        } else {
          minIncl = random().nextBoolean();
          maxIncl = random().nextBoolean();
        }
        ranges[rangeID] = new DoubleRange("r" + rangeID, min, minIncl, max, maxIncl);

        // Do "slow but hopefully correct" computation of
        // expected count:
        for (int i = 0; i < numDocs; i++) {
          boolean accept = true;
          if (minIncl) {
            accept &= values[i] >= min;
          } else {
            accept &= values[i] > min;
          }
          if (maxIncl) {
            accept &= values[i] <= max;
          } else {
            accept &= values[i] < max;
          }
          if (accept) {
            expectedCounts[rangeID]++;
            minAcceptedValue = Math.min(minAcceptedValue, values[i]);
            maxAcceptedValue = Math.max(maxAcceptedValue, values[i]);
          }
        }
      }

      // TODO: fastMatchQuery functionality is not implemented for sandbox faceting yet, do we need
      // it?
      /*Query fastMatchFilter;
      if (random().nextBoolean()) {
        if (random().nextBoolean()) {
          fastMatchFilter = DoublePoint.newRangeQuery("field", minValue, maxValue);
        } else {
          fastMatchFilter = DoublePoint.newRangeQuery("field", minAcceptedValue, maxAcceptedValue);
        }
      } else {
        fastMatchFilter = null;
      }*/

      final MultiDoubleValuesSource mvs;
      if (random().nextBoolean()) {
        DoubleValuesSource vs = DoubleValuesSource.fromDoubleField("field");
        mvs = MultiDoubleValuesSource.fromSingleValued(vs);
      } else {
        mvs = MultiDoubleValuesSource.fromDoubleField("field");
      }

      DoubleRangeFacetCutter doubleRangeFacetCutter = new DoubleRangeFacetCutter(mvs, ranges);
      CountFacetRecorder countRecorder = new CountFacetRecorder();

      FacetFieldCollectorManager<CountFacetRecorder> collectorManager =
          new FacetFieldCollectorManager<>(doubleRangeFacetCutter, countRecorder);
      s.search(new MatchAllDocsQuery(), collectorManager);

      OrdToLabel ordToLabel = new RangeOrdToLabel(ranges);
      FacetResult result =
          getAllSortByOrd(getRangeOrdinals(ranges), countRecorder, "field", ordToLabel);
      assertEquals(numRange, result.labelValues.length);
      for (int rangeID = 0; rangeID < numRange; rangeID++) {
        if (VERBOSE) {
          System.out.println("  range " + rangeID + " expectedCount=" + expectedCounts[rangeID]);
        }
        LabelAndValue subNode = result.labelValues[rangeID];
        assertEquals("r" + rangeID, subNode.label);
        assertEquals(expectedCounts[rangeID], subNode.value.intValue());

        DoubleRange range = ranges[rangeID];

        // Test drill-down:
        DrillDownQuery ddq = new DrillDownQuery(config);
        ddq.add("field", DoublePoint.newRangeQuery("field", range.min, range.max));

        assertEquals(expectedCounts[rangeID], s.count(ddq));
      }
    }

    w.close();
    IOUtils.close(r, dir);
  }

  public void testRandomDoublesMultiValued() throws Exception {
    Directory dir = newDirectory();
    RandomIndexWriter w = new RandomIndexWriter(random(), dir);

    int numDocs = atLeast(1000);
    double[][] values = new double[numDocs][];
    double minValue = Double.POSITIVE_INFINITY;
    double maxValue = Double.NEGATIVE_INFINITY;
    for (int i = 0; i < numDocs; i++) {
      Document doc = new Document();
      int numVals = RandomNumbers.randomIntBetween(random(), 1, 50);
      if (random().nextInt(10) == 0) {
        numVals = 1; // make sure we have ample testing of single-value cases
      }
      values[i] = new double[numVals];
      for (int j = 0; j < numVals; j++) {
        double v = random().nextDouble();
        values[i][j] = v;
        doc.add(new SortedNumericDocValuesField("field", Double.doubleToLongBits(v)));
        doc.add(new DoublePoint("field", v));
        minValue = Math.min(minValue, v);
        maxValue = Math.max(maxValue, v);
      }
      w.addDocument(doc);
    }
    IndexReader r = w.getReader();

    IndexSearcher s = newSearcher(r, false);
    FacetsConfig config = new FacetsConfig();

    int numIters = atLeast(10);
    for (int iter = 0; iter < numIters; iter++) {
      if (VERBOSE) {
        System.out.println("TEST: iter=" + iter);
      }
      int numRange = TestUtil.nextInt(random(), 1, 5);
      DoubleRange[] ranges = new DoubleRange[numRange];
      int[] expectedCounts = new int[numRange];
      double minAcceptedValue = Double.POSITIVE_INFINITY;
      double maxAcceptedValue = Double.NEGATIVE_INFINITY;
      for (int rangeID = 0; rangeID < numRange; rangeID++) {
        double min;
        if (rangeID > 0 && random().nextInt(10) == 7) {
          // Use an existing boundary:
          DoubleRange prevRange = ranges[random().nextInt(rangeID)];
          if (random().nextBoolean()) {
            min = prevRange.min;
          } else {
            min = prevRange.max;
          }
        } else {
          min = random().nextDouble();
        }
        double max;
        if (rangeID > 0 && random().nextInt(10) == 7) {
          // Use an existing boundary:
          DoubleRange prevRange = ranges[random().nextInt(rangeID)];
          if (random().nextBoolean()) {
            max = prevRange.min;
          } else {
            max = prevRange.max;
          }
        } else {
          max = random().nextDouble();
        }

        if (min > max) {
          double x = min;
          min = max;
          max = x;
        }

        boolean minIncl;
        boolean maxIncl;

        long minAsLong = NumericUtils.doubleToSortableLong(min);
        long maxAsLong = NumericUtils.doubleToSortableLong(max);
        // NOTE: maxAsLong - minAsLong >= 0 is here to handle the common overflow case!
        if (maxAsLong - minAsLong >= 0 && maxAsLong - minAsLong < 2) {
          minIncl = true;
          maxIncl = true;
        } else {
          minIncl = random().nextBoolean();
          maxIncl = random().nextBoolean();
        }
        ranges[rangeID] = new DoubleRange("r" + rangeID, min, minIncl, max, maxIncl);

        // Do "slow but hopefully correct" computation of
        // expected count:
        for (int i = 0; i < numDocs; i++) {
          for (int j = 0; j < values[i].length; j++) {
            boolean accept = true;
            if (minIncl) {
              accept &= values[i][j] >= min;
            } else {
              accept &= values[i][j] > min;
            }
            if (maxIncl) {
              accept &= values[i][j] <= max;
            } else {
              accept &= values[i][j] < max;
            }
            if (accept) {
              expectedCounts[rangeID]++;
              minAcceptedValue = Math.min(minAcceptedValue, values[i][j]);
              maxAcceptedValue = Math.max(maxAcceptedValue, values[i][j]);
              break; // ensure each doc can contribute at most 1 count to each range
            }
          }
        }
      }
      // TODO: fastMatchQuery functionality is not implemented for sandbox faceting yet, do we need
      // it?
      /*Query fastMatchFilter;
      if (random().nextBoolean()) {
        if (random().nextBoolean()) {
          fastMatchFilter = DoublePoint.newRangeQuery("field", minValue, maxValue);
        } else {
          fastMatchFilter = DoublePoint.newRangeQuery("field", minAcceptedValue, maxAcceptedValue);
        }
      } else {
        fastMatchFilter = null;
      }*/
      final MultiDoubleValuesSource mvs = MultiDoubleValuesSource.fromDoubleField("field");
      DoubleRangeFacetCutter doubleRangeFacetCutter = new DoubleRangeFacetCutter(mvs, ranges);
      CountFacetRecorder countRecorder = new CountFacetRecorder();
      FacetFieldCollectorManager<CountFacetRecorder> collectorManager =
          new FacetFieldCollectorManager<>(doubleRangeFacetCutter, countRecorder);
      s.search(new MatchAllDocsQuery(), collectorManager);

      OrdToLabel ordToLabel = new RangeOrdToLabel(ranges);
      FacetResult result =
          getAllSortByOrd(getRangeOrdinals(ranges), countRecorder, "field", ordToLabel);
      assertEquals(numRange, result.labelValues.length);
      for (int rangeID = 0; rangeID < numRange; rangeID++) {
        if (VERBOSE) {
          System.out.println("  range " + rangeID + " expectedCount=" + expectedCounts[rangeID]);
        }
        LabelAndValue subNode = result.labelValues[rangeID];
        assertEquals("r" + rangeID, subNode.label);
        assertEquals(expectedCounts[rangeID], subNode.value.intValue());

        DoubleRange range = ranges[rangeID];

        // Test drill-down:
        DrillDownQuery ddq = new DrillDownQuery(config);
        if (random().nextBoolean()) {
          ddq.add("field", DoublePoint.newRangeQuery("field", range.min, range.max));
        } else {
          ddq.add(
              "field",
              SortedNumericDocValuesField.newSlowRangeQuery(
                  "field", Double.doubleToLongBits(range.min), Double.doubleToLongBits(range.max)));
        }

        assertEquals(expectedCounts[rangeID], s.count(ddq));
      }
    }

    w.close();
    IOUtils.close(r, dir);
  }

  // LUCENE-5178
  public void testMissingValues() throws Exception {
    Directory d = newDirectory();
    RandomIndexWriter w = new RandomIndexWriter(random(), d);
    Document doc = new Document();
    NumericDocValuesField field = new NumericDocValuesField("field", 0L);
    doc.add(field);
    for (long l = 0; l < 100; l++) {
      if (l % 5 == 0) {
        // Every 5th doc is missing the value:
        w.addDocument(new Document());
        continue;
      }
      field.setLongValue(l);
      w.addDocument(doc);
    }

    IndexReader r = w.getReader();

    IndexSearcher s = newSearcher(r);
    LongRange[] inputRanges =
        new LongRange[] {
          new LongRange("less than 10", 0L, true, 10L, false),
          new LongRange("less than or equal to 10", 0L, true, 10L, true),
          new LongRange("over 90", 90L, false, 100L, false),
          new LongRange("90 or above", 90L, true, 100L, false),
          new LongRange("over 1000", 1000L, false, Long.MAX_VALUE, false)
        };

    MultiLongValuesSource valuesSource = MultiLongValuesSource.fromLongField("field");
    LongRangeFacetCutter longRangeFacetCutter =
        LongRangeFacetCutter.create(valuesSource, inputRanges);
    CountFacetRecorder countRecorder = new CountFacetRecorder();

    FacetFieldCollectorManager<CountFacetRecorder> collectorManager =
        new FacetFieldCollectorManager<>(longRangeFacetCutter, countRecorder);
    s.search(new MatchAllDocsQuery(), collectorManager);
    OrdToLabel ordToLabel = new RangeOrdToLabel(inputRanges);

    assertEquals(
        "dim=field path=[] value=-2147483648 childCount=5\n  less than 10 (8)\n  less than or equal to 10 (8)\n  over 90 (8)\n  90 or above (8)\n  over 1000 (0)\n",
        getAllSortByOrd(getRangeOrdinals(inputRanges), countRecorder, "field", ordToLabel)
            .toString());

    w.close();
    IOUtils.close(r, d);
  }

  public void testMissingValuesMultiValued() throws Exception {
    Directory d = newDirectory();
    RandomIndexWriter w = new RandomIndexWriter(random(), d);
    Document doc = new Document();
    // index the same field twice to test multi-valued logic
    SortedNumericDocValuesField field1 = new SortedNumericDocValuesField("field", 0L);
    SortedNumericDocValuesField field2 = new SortedNumericDocValuesField("field", 0L);
    doc.add(field1);
    doc.add(field2);
    for (long l = 0; l < 100; l++) {
      if (l % 5 == 0) {
        // Every 5th doc is missing the value:
        w.addDocument(new Document());
        continue;
      }
      field1.setLongValue(l);
      field2.setLongValue(l);
      w.addDocument(doc);
    }

    IndexReader r = w.getReader();

    IndexSearcher s = newSearcher(r);
    LongRange[] inputRanges =
        new LongRange[] {
          new LongRange("less than 10", 0L, true, 10L, false),
          new LongRange("less than or equal to 10", 0L, true, 10L, true),
          new LongRange("over 90", 90L, false, 100L, false),
          new LongRange("90 or above", 90L, true, 100L, false),
          new LongRange("over 1000", 1000L, false, Long.MAX_VALUE, false)
        };

    MultiLongValuesSource valuesSource = MultiLongValuesSource.fromLongField("field");
    LongRangeFacetCutter longRangeFacetCutter =
        LongRangeFacetCutter.create(valuesSource, inputRanges);
    CountFacetRecorder countRecorder = new CountFacetRecorder();

    FacetFieldCollectorManager<CountFacetRecorder> collectorManager =
        new FacetFieldCollectorManager<>(longRangeFacetCutter, countRecorder);
    s.search(new MatchAllDocsQuery(), collectorManager);
    OrdToLabel ordToLabel = new RangeOrdToLabel(inputRanges);

    assertEquals(
        "dim=field path=[] value=-2147483648 childCount=5\n  less than 10 (8)\n  less than or equal to 10 (8)\n  over 90 (8)\n  90 or above (8)\n  over 1000 (0)\n",
        getAllSortByOrd(getRangeOrdinals(inputRanges), countRecorder, "field", ordToLabel)
            .toString());

    w.close();
    IOUtils.close(r, d);
  }

  private static class PlusOneValuesSource extends DoubleValuesSource {

    @Override
    public DoubleValues getValues(LeafReaderContext ctx, DoubleValues scores) throws IOException {
      return new DoubleValues() {
        int doc = -1;

        @Override
        public double doubleValue() throws IOException {
          return doc + 1;
        }

        @Override
        public boolean advanceExact(int doc) throws IOException {
          this.doc = doc;
          return true;
        }
      };
    }

    @Override
    public boolean needsScores() {
      return false;
    }

    @Override
    public boolean isCacheable(LeafReaderContext ctx) {
      return false;
    }

    @Override
    public Explanation explain(LeafReaderContext ctx, int docId, Explanation scoreExplanation)
        throws IOException {
      return Explanation.match(docId + 1, "");
    }

    @Override
    public DoubleValuesSource rewrite(IndexSearcher searcher) throws IOException {
      return this;
    }

    @Override
    public int hashCode() {
      return 0;
    }

    @Override
    public boolean equals(Object obj) {
      return obj.getClass() == PlusOneValuesSource.class;
    }

    @Override
    public String toString() {
      return null;
    }
  }

  public void testCustomDoubleValuesSource() throws Exception {
    Directory dir = newDirectory();
    RandomIndexWriter writer = new RandomIndexWriter(random(), dir);

    Document doc = new Document();
    writer.addDocument(doc);
    writer.addDocument(doc);
    writer.addDocument(doc);

    // Test wants 3 docs in one segment:
    writer.forceMerge(1);

    final DoubleValuesSource vs = new PlusOneValuesSource();

    FacetsConfig config = new FacetsConfig();

    IndexReader r = writer.getReader();

    IndexSearcher s = getNewSearcherForDrillSideways(r);
    // DrillSideways requires the entire range of docs to be scored at once, so it doesn't support
    // timeouts whose implementation scores one window of doc IDs at a time.
    s.setTimeout(null);

    final DoubleRange[] ranges =
        new DoubleRange[] {
          new DoubleRange("< 1", 0.0, true, 1.0, false),
          new DoubleRange("< 2", 0.0, true, 2.0, false),
          new DoubleRange("< 5", 0.0, true, 5.0, false),
          new DoubleRange("< 10", 0.0, true, 10.0, false),
          new DoubleRange("< 20", 0.0, true, 20.0, false),
          new DoubleRange("< 50", 0.0, true, 50.0, false)
        };

    // TODO: fastMatchQuery functionality is not implemented for sandbox faceting yet, do we need
    // it?
    /*final Query fastMatchFilter;
    final AtomicBoolean filterWasUsed = new AtomicBoolean();
    if (random().nextBoolean()) {
      // Sort of silly:
      final Query in = new MatchAllDocsQuery();
      fastMatchFilter = new UsedQuery(in, filterWasUsed);
    } else {
      fastMatchFilter = null;
    }

    if (VERBOSE) {
      System.out.println("TEST: fastMatchFilter=" + fastMatchFilter);
    }*/

    MultiDoubleValuesSource valuesSource = MultiDoubleValuesSource.fromSingleValued(vs);
    DoubleRangeFacetCutter doubleRangeFacetCutter =
        new DoubleRangeFacetCutter(valuesSource, ranges);
    CountFacetRecorder countRecorder = new CountFacetRecorder();

    FacetFieldCollectorManager<CountFacetRecorder> collectorManager =
        new FacetFieldCollectorManager<>(doubleRangeFacetCutter, countRecorder);
    s.search(new MatchAllDocsQuery(), collectorManager);
    OrdToLabel ordToLabel = new RangeOrdToLabel(ranges);

    assertEquals(
        "dim=field path=[] value=-2147483648 childCount=6\n  < 1 (0)\n  < 2 (1)\n  < 5 (3)\n  < 10 (3)\n  < 20 (3)\n  < 50 (3)\n",
        getAllSortByOrd(getRangeOrdinals(ranges), countRecorder, "field", ordToLabel).toString());
    // assertTrue(fastMatchFilter == null || filterWasUsed.get());

    DrillDownQuery ddq = new DrillDownQuery(config);
    if (random().nextBoolean()) {
      ddq.add("field", ranges[1].getQuery(null, vs));
    } else {
      ddq.add("field", ranges[1].getQuery(null, MultiDoubleValuesSource.fromSingleValued(vs)));
    }

    // Test simple drill-down:
    assertEquals(1, s.search(ddq, 10).totalHits.value());

    // Test drill-sideways after drill-down
    DrillSideways ds =
        new DrillSideways(s, config, (TaxonomyReader) null) {
          @Override
          protected boolean scoreSubDocsAtOnce() {
            return random().nextBoolean();
          }
        };

    countRecorder = new CountFacetRecorder();

    DrillSideways.Result<Integer, CountFacetRecorder> result =
        ds.search(
            ddq,
            DummyTotalHitCountCollector.createManager(),
            List.of(new FacetFieldCollectorManager<>(doubleRangeFacetCutter, countRecorder)));
    assertEquals(1, result.drillDownResult().intValue());
    assertEquals(
        "dim=field path=[] value=-2147483648 childCount=6\n  < 1 (0)\n  < 2 (1)\n  < 5 (3)\n  < 10 (3)\n  < 20 (3)\n  < 50 (3)\n",
        getAllSortByOrd(getRangeOrdinals(ranges), countRecorder, "field", ordToLabel).toString());

    writer.close();
    IOUtils.close(r, dir);
  }
}
