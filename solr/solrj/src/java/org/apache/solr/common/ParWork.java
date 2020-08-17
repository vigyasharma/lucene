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
package org.apache.solr.common;

import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.solr.client.solrj.impl.HttpClientUtil;
import org.apache.solr.common.util.ExecutorUtil;
import org.apache.solr.common.util.ObjectReleaseTracker;
import org.apache.solr.common.util.OrderedExecutor;
import org.apache.solr.common.util.SysStats;
import org.apache.zookeeper.KeeperException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Closeable;
import java.lang.invoke.MethodHandles;
import java.lang.management.ManagementFactory;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Set;
import java.util.Timer;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.FutureTask;
import java.util.concurrent.SynchronousQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicReference;

/**
 * ParWork. A workhorse utility class that tries to use good patterns,
 * parallelism
 * 
 */
public class ParWork implements Closeable {
  public static final int PROC_COUNT = ManagementFactory.getOperatingSystemMXBean().getAvailableProcessors();
  private static final String WORK_WAS_INTERRUPTED = "Work was interrupted!";

  private static final String RAN_INTO_AN_ERROR_WHILE_DOING_WORK =
      "Ran into an error while doing work!";

  private static final Logger log = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass());

  protected final static ThreadLocal<ExecutorService> THREAD_LOCAL_EXECUTOR = new ThreadLocal<>();
  private final boolean requireAnotherThread;
  private final String rootLabel;

  private volatile Set<ParObject> collectSet = ConcurrentHashMap.newKeySet(32);

  private static volatile ThreadPoolExecutor EXEC;

  // pretty much don't use it
  public static ThreadPoolExecutor getEXEC() {
    if (EXEC == null) {
      synchronized (ParWork.class) {
        if (EXEC == null) {
          EXEC = (ThreadPoolExecutor) getParExecutorService(2, Integer.MAX_VALUE, 15000, new SynchronousQueue<>());
        }
      }
    }
    return EXEC;
  }


  public static void shutdownExec() {
    synchronized (ParWork.class) {
      if (EXEC != null) {
        EXEC.setKeepAliveTime(1, TimeUnit.NANOSECONDS);
        EXEC.allowCoreThreadTimeOut(true);
        EXEC.shutdownNow();
        ExecutorUtil.shutdownAndAwaitTermination(EXEC);
        EXEC = null;
      }
    }
  }


  private static SysStats sysStats = SysStats.getSysStats();

  public static SysStats getSysStats() {
    return sysStats;
  }

  public static void closeExecutor() {
    closeExecutor(true);
  }

  public static void closeExecutor(boolean unlockClose) {
    ParWorkExecService exec = (ParWorkExecService) THREAD_LOCAL_EXECUTOR.get();
    if (exec != null) {
      if (unlockClose) {
        exec.closeLock(false);
      }
      ParWork.close(exec);
      THREAD_LOCAL_EXECUTOR.set(null);
    }
  }

    private static class WorkUnit {
    private final List<ParObject> objects;
    private final TimeTracker tracker;

    public WorkUnit(List<ParObject> objects, TimeTracker tracker) {
      objects.remove(null);
      boolean ok;
      for (ParObject parobject : objects) {
        Object object = parobject.object;
        assert !(object instanceof ParObject);
        ok  = false;
        for (Class okobject : OK_CLASSES) {
          if (object == null || okobject.isAssignableFrom(object.getClass())) {
            ok = true;
            break;
          }
        }
        if (!ok) {
          log.error(" -> I do not know how to close: " + object.getClass().getName());
          throw new IllegalArgumentException(" -> I do not know how to close: " + object.getClass().getName());
        }
      }

      this.objects = objects;
      this.tracker = tracker;

      assert checkTypesForTests(objects);
    }

    private boolean checkTypesForTests(List<ParObject> objects) {
      for (ParObject object : objects) {
        assert !(object.object instanceof Collection);
        assert !(object.object instanceof Map);
        assert !(object.object.getClass().isArray());
      }

      return true;
    }
  }

  private static final Set<Class> OK_CLASSES = new HashSet<>();

  static {
    OK_CLASSES.add(ExecutorService.class);

    OK_CLASSES.add(OrderedExecutor.class);

    OK_CLASSES.add(Closeable.class);

    OK_CLASSES.add(AutoCloseable.class);

    OK_CLASSES.add(Callable.class);

    OK_CLASSES.add(Runnable.class);

    OK_CLASSES.add(Timer.class);

    OK_CLASSES.add(CloseableHttpClient.class);

  }

  private List<WorkUnit> workUnits = Collections.synchronizedList(new ArrayList<>());

  private volatile TimeTracker tracker;

  private final boolean ignoreExceptions;

  private Set<Throwable> warns = ParWork.concSetSmallO();

  // TODO should take logger as well
  public static class Exp extends Exception {

    private static final String ERROR_MSG = "Solr ran into an unexpected Exception";

    /**
     * Handles exceptions correctly for you, including logging.
     * 
     * @param msg message to include to clarify the problem
     */
    public Exp(String msg) {
      this(null, msg, null);
    }

    /**
     * Handles exceptions correctly for you, including logging.
     * 
     * @param th the exception to handle
     */
    public Exp(Throwable th) {
      this(null, th.getMessage(), th);
    }

    /**
     * Handles exceptions correctly for you, including logging.
     * 
     * @param msg message to include to clarify the problem
     * @param th  the exception to handle
     */
    public Exp(String msg, Throwable th) {
      this(null, msg, th);
    }

    public Exp(Logger classLog, String msg, Throwable th) {
      super(msg == null ? ERROR_MSG : msg, th);

      Logger logger;
      if (classLog != null) {
        logger = classLog;
      } else {
        logger = log;
      }

      logger.error(ERROR_MSG, th);
      if (th != null && th instanceof InterruptedException) {
        Thread.currentThread().interrupt();
      }
      if (th != null && th instanceof KeeperException) { // TODO maybe start using ZooKeeperException
        if (((KeeperException) th).code() == KeeperException.Code.SESSIONEXPIRED) {
          log.warn("The session has expired, give up any leadership roles!");
        }
      }
    }
  }

  public ParWork(Object object) {
    this(object, false);
  }


  public ParWork(Object object, boolean ignoreExceptions) {
    this(object, ignoreExceptions, false);
  }

  public ParWork(Object object, boolean ignoreExceptions, boolean requireAnotherThread) {
    this.ignoreExceptions = ignoreExceptions;
    this.requireAnotherThread = requireAnotherThread;
    this.rootLabel = object instanceof String ?
        (String) object : object.getClass().getSimpleName();
    assert (tracker = new TimeTracker(object, object == null ? "NullObject" : object.getClass().getName())) != null;
    // constructor must stay very light weight
  }

  public void collect(String label, Object object) {
    if (object == null) {
      return;
    }
    ParObject ob = new ParObject();
    ob.object = object;
    ob.label = label;
    collectSet.add(ob);
  }

  public void collect(Object object) {
    if (object == null) {
      return;
    }
    ParObject ob = new ParObject();
    ob.object = object;
    ob.label = object.getClass().getSimpleName();
    collectSet.add(ob);
  }

  public void collect(Object... objects) {
    for (Object object : objects) {
      collect(object);
    }
  }

  /**
   * @param callable A Callable to run. If an object is return, it's toString is
   *                 used to identify it.
   */
  public void collect(String label, Callable<?> callable) {
    ParObject ob = new ParObject();
    ob.object = callable;
    ob.label = label;
    collectSet.add(ob);
  }

  /**
   * @param runnable A Runnable to run. If an object is return, it's toString is
   *                 used to identify it.
   */
  public void collect(String label, Runnable runnable) {
    if (runnable == null) {
      return;
    }
    ParObject ob = new ParObject();
    ob.object = runnable;
    ob.label = label;
    collectSet.add(ob);
  }

  public void addCollect() {
    if (collectSet.isEmpty()) {
      if (log.isDebugEnabled()) log.debug("No work collected to submit");
      return;
    }
    try {
      for (ParObject ob : collectSet) {
        assert (!(ob.object instanceof ParObject));
        add(ob);
      }
    } finally {
      collectSet.clear();
    }
  }

  @SuppressWarnings({ "unchecked", "rawtypes" })
  private void gatherObjects(Object object, List<ParObject> objects) {
    if (log.isDebugEnabled()) {
      log.debug("gatherObjects(Object object={}, List<Object> objects={}) - start", object, objects);
    }

    if (object != null) {
      if (object.getClass().isArray()) {
        if (log.isDebugEnabled()) {
          log.debug("Found an array to gather against");
        }

        for (Object obj : (Object[]) object) {
          gatherObjects(obj, objects);
        }

      } else if (object instanceof Collection) {
        if (log.isDebugEnabled()) {
          log.debug("Found a Collectiom to gather against");
        }
        for (Object obj : (Collection) object) {
          gatherObjects(obj, objects);
        }
      } else if (object instanceof Map<?, ?>) {
        if (log.isDebugEnabled()) {
          log.debug("Found a Map to gather against");
        }
        ((Map) object).forEach((k, v) -> gatherObjects(v, objects));
      } else {
        if (log.isDebugEnabled()) {
          log.debug("Found a non collection object to add {}", object.getClass().getName());
        }
        if (object instanceof ParObject) {
          objects.add((ParObject) object);
        } else {
          ParObject ob = new ParObject();
          ob.object = object;
          ob.label = object.getClass().getSimpleName();
          objects.add(ob);
        }
      }
    }
  }

  private void add(ParObject object) {
    if (log.isDebugEnabled()) {
      log.debug("add(String label={}, Object object={}, Callable Callables={}) - start", object.label, object);
    }

    List<ParObject> objects = new ArrayList<>();

    gatherObjects(object.object, objects);

    WorkUnit workUnit = new WorkUnit(objects, tracker);
    workUnits.add(workUnit);
  }

  @Override
  public void close() {
    if (log.isDebugEnabled()) {
      log.debug("close() - start");
    }

    addCollect();

    boolean needExec = false;
    for (WorkUnit workUnit : workUnits) {
      if (workUnit.objects.size() > 1) {
        needExec = true;
      }
    }

    ParWorkExecService executor = null;
    if (needExec) {
      executor = (ParWorkExecService) getExecutor();
    }
    //initExecutor();
    AtomicReference<Throwable> exception = new AtomicReference<>();
    try {
      for (WorkUnit workUnit : workUnits) {
        if (log.isDebugEnabled()) log.debug("Process workunit {} {}", rootLabel, workUnit.objects);
        TimeTracker workUnitTracker = null;
        assert (workUnitTracker = workUnit.tracker.startSubClose(rootLabel)) != null;
        try {
          List<ParObject> objects = workUnit.objects;

          if (objects.size() == 1) {
            handleObject(exception, workUnitTracker, objects.get(0));
          } else {

            List<Callable<Object>> closeCalls = new ArrayList<>(objects.size());

            for (ParObject object : objects) {

              if (object == null)
                continue;

              TimeTracker finalWorkUnitTracker = workUnitTracker;
              if (requireAnotherThread) {
                closeCalls.add(new NoLimitsCallable<Object>() {
                  @Override
                  public Object call() throws Exception {
                    try {
                      handleObject(exception, finalWorkUnitTracker,
                          object);
                    } catch (Throwable t) {
                      log.error(RAN_INTO_AN_ERROR_WHILE_DOING_WORK, t);
                      if (exception.get() == null) {
                        exception.set(t);
                      }
                    }
                    return object;
                  }
                });
              } else {
                closeCalls.add(() -> {
                  try {
                    handleObject(exception, finalWorkUnitTracker,
                        object);
                  } catch (Throwable t) {
                    log.error(RAN_INTO_AN_ERROR_WHILE_DOING_WORK, t);
                    if (exception.get() == null) {
                      exception.set(t);
                    }
                  }
                  return object;
                });
              }

            }
            if (closeCalls.size() > 0) {

                List<Future<Object>> results = new ArrayList<>(closeCalls.size());

                for (Callable call : closeCalls) {
                    Future future = executor.submit(call);
                    results.add(future);
                }

//                List<Future<Object>> results = executor.invokeAll(closeCalls, 8, TimeUnit.SECONDS);
              int i = 0;
                for (Future<Object> future : results) {
                  try {
                    future.get(
                        Integer.getInteger("solr.parwork.task_timeout", 10000),
                        TimeUnit.MILLISECONDS); // nocommit
                    if (!future.isDone() || future.isCancelled()) {
                      log.warn("A task did not finish isDone={} isCanceled={}",
                          future.isDone(), future.isCancelled());
                      //  throw new SolrException(SolrException.ErrorCode.SERVER_ERROR, "A task did nor finish" +future.isDone()  + " " + future.isCancelled());
                    }
                  } catch (TimeoutException e) {
                    throw new SolrException(SolrException.ErrorCode.SERVER_ERROR, objects.get(i).label, e);
                  } catch (InterruptedException e1) {
                    log.warn(WORK_WAS_INTERRUPTED);
                    // TODO: save interrupted status and reset it at end?
                  }

                }



            }
          }
        } finally {
          if (workUnitTracker != null)
            workUnitTracker.doneClose();
        }

      }
    } catch (Throwable t) {
      log.error(RAN_INTO_AN_ERROR_WHILE_DOING_WORK, t);

      if (exception.get() == null) {
        exception.set(t);
      }
    } finally {

      assert tracker.doneClose();
      
      //System.out.println("DONE:" + tracker.getElapsedMS());

      warns.forEach((it) -> log.warn(RAN_INTO_AN_ERROR_WHILE_DOING_WORK, new RuntimeException(it)));

      if (exception.get() != null) {
        Throwable exp = exception.get();
        if (exp instanceof Error) {
          throw (Error) exp;
        }
        if (exp instanceof  RuntimeException) {
          throw (RuntimeException) exp;
        }
        throw new RuntimeException(exp);
      }
    }

    if (log.isDebugEnabled()) {
      log.debug("close() - end");
    }
  }

  public static ExecutorService getExecutor() {
     // if (executor != null) return executor;
    ExecutorService exec = THREAD_LOCAL_EXECUTOR.get();
    if (exec == null) {
      if (log.isDebugEnabled()) {
        log.debug("Starting a new executor");
      }

      Integer minThreads;
      Integer maxThreads;
      minThreads = 3;
      maxThreads = PROC_COUNT;
      exec = getExecutorService(Math.max(minThreads, maxThreads)); // keep alive directly affects how long a worker might
      ((ParWorkExecService)exec).closeLock(true);
      // be stuck in poll without an enqueue on shutdown
      THREAD_LOCAL_EXECUTOR.set(exec);
    }

    return exec;
  }

  public static ExecutorService getParExecutorService(int corePoolSize, int maxPoolSize, int keepAliveTime, BlockingQueue queue) {
    ThreadPoolExecutor exec;
    exec = new ParWorkExecutor("ParWork-" + Thread.currentThread().getName(),
            corePoolSize, maxPoolSize, keepAliveTime, queue);
    return exec;
  }

  public static ExecutorService getExecutorService(int maximumPoolSize) {
    return new ParWorkExecService(getEXEC(), maximumPoolSize);
  }

  private void handleObject(AtomicReference<Throwable> exception, final TimeTracker workUnitTracker, ParObject ob) {
    if (log.isDebugEnabled()) {
      log.debug(
          "handleObject(AtomicReference<Throwable> exception={}, CloseTimeTracker workUnitTracker={}, Object object={}) - start",
          exception, workUnitTracker, ob.object);
    }
    Object object = ob.object;
    if (object != null) {
      assert !(object instanceof Collection);
      assert !(object instanceof Map);
      assert !(object.getClass().isArray());
    }

    Object returnObject = null;
    TimeTracker subTracker = null;
    assert (subTracker = workUnitTracker.startSubClose(object)) != null;
    try {
      boolean handled = false;
      if (object instanceof OrderedExecutor) {
        ((OrderedExecutor) object).shutdownAndAwaitTermination();
        handled = true;
      } else if (object instanceof ExecutorService) {
        shutdownAndAwaitTermination((ExecutorService) object);
        handled = true;
      } else if (object instanceof CloseableHttpClient) {
        HttpClientUtil.close((CloseableHttpClient) object);
        handled = true;
      } else if (object instanceof Closeable) {
        ((Closeable) object).close();
        handled = true;
      } else if (object instanceof AutoCloseable) {
        ((AutoCloseable) object).close();
        handled = true;
      } else if (object instanceof Callable) {
        returnObject = ((Callable<?>) object).call();
        handled = true;
      } else if (object instanceof Runnable) {
        ((Runnable) object).run();
        handled = true;
      } else if (object instanceof Timer) {
        ((Timer) object).cancel();
        handled = true;
      }

      if (!handled) {
        String msg = ob.label + " -> I do not know how to close " + ob.label  + ": " + object.getClass().getName();
        log.error(msg);
        IllegalArgumentException illegal = new IllegalArgumentException(msg);
        exception.set(illegal);
      }
    } catch (Throwable t) {

      if (t instanceof NullPointerException) {
        log.info("NPE closing " + object == null ? "Null Object" : object.getClass().getName());
      } else {
        if (ignoreExceptions) {
          warns.add(t);
          log.error("Error handling close for an object: " + ob.label + ": " + object.getClass().getSimpleName() , new ObjectReleaseTracker.ObjectTrackerException(t));
          if (t instanceof Error && !(t instanceof AssertionError)) {
            throw (Error) t;
          }
        } else {
          log.error("handleObject(AtomicReference<Throwable>=" + exception + ", CloseTimeTracker=" + workUnitTracker   + ", Label=" + ob.label + ")"
              + ", Object=" + object + ")", t);
          propegateInterrupt(t);
          if (t instanceof Error) {
            throw (Error) t;
          }
          if (t instanceof  RuntimeException) {
            throw (RuntimeException) t;
          } else {
            throw new WorkException(RAN_INTO_AN_ERROR_WHILE_DOING_WORK, t); // TODO, hmm how do I keep zk session timeout and interrupt in play?
          }
        }
      }
    } finally {
      assert subTracker.doneClose(returnObject instanceof String ? (String) returnObject : (returnObject == null ? "" : returnObject.getClass().getName()));
    }

    if (log.isDebugEnabled()) {
      log.debug("handleObject(AtomicReference<Throwable>, CloseTimeTracker, List<Callable<Object>>, Object) - end");
    }
  }

  /**
   * Sugar method to close objects.
   * 
   * @param object to close
   */
  public static void close(Object object, boolean ignoreExceptions) {
    if (object == null) return;
    assert !(object instanceof ParObject);
    try (ParWork dw = new ParWork(object, ignoreExceptions)) {
      dw.collect(object);
    }
  }

  public static void close(Object object) {
    close(object, false);
  }

  public static <K> Set<K> concSetSmallO() {
    return ConcurrentHashMap.newKeySet(50);
  }

  public static <K, V> ConcurrentHashMap<K, V> concMapSmallO() {
    return new ConcurrentHashMap<K, V>(132, 0.75f, 50);
  }

  public static <K, V> ConcurrentHashMap<K, V> concMapReqsO() {
    return new ConcurrentHashMap<>(128, 0.75f, 2048);
  }

  public static <K, V> ConcurrentHashMap<K, V> concMapClassesO() {
    return new ConcurrentHashMap<>(132, 0.75f, 8192);
  }

  public static void propegateInterrupt(Throwable t) {
    propegateInterrupt(t, false);
  }

  public static void propegateInterrupt(Throwable t, boolean infoLogMsg) {
    if (t instanceof InterruptedException) {
      log.info("Interrupted", t.getMessage());
      Thread.currentThread().interrupt();
    } else {
      if (infoLogMsg) {
        log.info(t.getMessage());
      } else {
        log.warn("Solr ran into an unexpected exception", t);
      }
    }

    if (t instanceof Error) {
      throw (Error) t;
    }
  }

  public static void propegateInterrupt(String msg, Throwable t) {
    propegateInterrupt(msg, t, false);
  }

  public static void propegateInterrupt(String msg, Throwable t, boolean infoLogMsg) {
    if (t instanceof InterruptedException) {
      log.info("Interrupted", t);
      Thread.currentThread().interrupt();
    } else {
      if (infoLogMsg) {
        log.info(msg);
      } else {
        log.warn(msg, t);
      }
    }
    if (t instanceof Error) {
      throw (Error) t;
    }
  }

  public static void shutdownAndAwaitTermination(ExecutorService pool) {
    if (pool == null)
      return;
    pool.shutdown(); // Disable new tasks from being submitted
    awaitTermination(pool);
    if (!(pool.isShutdown())) {
      throw new RuntimeException("Timeout waiting for executor to shutdown");
    }

  }

  public static void awaitTermination(ExecutorService pool) {
    boolean shutdown = false;
    while (!shutdown) {
      try {
        // Wait a while for existing tasks to terminate
        shutdown = pool.awaitTermination(30, TimeUnit.SECONDS);
      } catch (InterruptedException ie) {
        // Preserve interrupt status
        Thread.currentThread().interrupt();
      }
    }
  }

  public static abstract class NoLimitsCallable<V> implements Callable {
    @Override
    public abstract Object call() throws Exception;
  }

  public static class SolrFutureTask extends FutureTask {
    public SolrFutureTask(Callable callable) {
      super(callable);
    }
  }

  private static class ParObject {
    String label;
    Object object;
  }
}