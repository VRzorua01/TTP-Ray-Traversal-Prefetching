# TTP Ray Traversal Prefetching on Vortex

A software-simulator implementation and evaluation of **stack-directed ray traversal prefetching** in the [Vortex GPGPU](https://github.com/vortexgpgpu/vortex) ray tracing unit (RTU). This project is motivated by **“TTP: A Hardware-Efficient Design for Precise Prefetching in Ray Tracing” (ISCA 2026)** and adapts its central idea—using BVH traversal-stack behavior to anticipate future node accesses—to Vortex's **SimX** RTU and memory hierarchy.

> **Implementation scope:** This is an experimental **SimX** implementation, not an RTL hardware implementation or a cycle-exact port of the paper's Vulkan-Sim model. It preserves Vortex's existing demand-driven traversal/replay engine and adds a replay-aware, stack-based prefetch predictor and a speculative memory-request path. The mechanisms and results below describe the implementation tested in this repository, not claims about the original paper's measured performance.

## Motivation

Ray tracing accelerates ray/geometry intersection queries with a **bounding volume hierarchy (BVH)**. A ray traverses the BVH from the root, tests child bounding boxes, descends into relevant subtrees, and eventually intersects triangles or other primitives. Nodes are often stored in device memory, so a demand access to a node that is absent from the RTCache can stall traversal.

A depth-first BVH walker already maintains a stack of nodes it expects to visit next. **TTP (traversal-tree prefetching)** exploits this information to issue speculative reads of likely future nodes before the walker demands them. If those reads complete early enough, later demand accesses can hit in cache or join an outstanding request rather than incurring the entire miss latency.

The expected benefit depends on scene geometry, cache behavior, memory latency, request concurrency, and *when* prefetches issue. Prefetches can also cause redundant accesses, cache pollution, and contention, so fewer **demand** misses do not necessarily imply fewer **total** memory transactions.

## Architecture overview

```text
                         BVH scene in device memory
                                    |
                        +-----------v-----------+
                        |  Vortex RTU / SimX    |
                        |  rtu_walker.cpp       |
                        |  depth-first BVH walk |
                        |  bounded ShortStack   |
                        +-----------+-----------+
                                    |
                     actual stack pop / lookahead
                                    |
                        +-----------v-----------+
                        |   TTP predictor       |
                        | pop trend: 1 / 2 / 16 |
                        | prefetch cursor       |
                        | replay event watermark|
                        +-----------+-----------+
                                    |
                           cache-line candidates
                                    |
                        +-----------v-----------+
                        |  rtu_core.cpp         |
                        |  context / replay     |
                        |  dispatch to memory   |
                        +-----------+-----------+
                                    |
                        +-----------v-----------+
                        |  rtu_memory.cpp       |
                        | prefetch FIFO, filter |
                        | demand-first issue    |
                        | MSHR / response merge |
                        +-----------+-----------+
                                    |
                           MemReq.is_prefetch
                                    |
                        +-----------v-----------+
                        |  RTCache / cache.cpp  |
                        |  hit, miss, fill      |
                        |  prefetch statistics  |
                        +-----------------------+
```

### Existing Vortex traversal model

The SimX RTU fetches scene data **on demand**, a cache line at a time. `read_scene_bytes()` looks up the required line in the context's local line map. When the line is unavailable, the walker reports a miss; `rtu_core.cpp` schedules the request and replays the deterministic walk after the response arrives. Previously completed traversal work is not charged again: the core computes the incremental difference from the last probe's cumulative counters.

This is important for TTP: a straightforward predictor embedded in the walker would otherwise rediscover the same stack events and repeatedly generate speculative requests during replay.

## Implemented TTP behavior

### 1. Stack-directed candidate selection

The BVH4/BVH6 walk uses a bounded `ShortStack` of deferred child nodes. The predictor responds to **actual stack-pop events** and considers future entries on the stack. A small pop-trend state determines how many entries to look ahead:

| Consecutive pop trend | Prefetch lookahead |
| --- | ---: |
| First pop | 1 entry |
| Second pop | 2 entries |
| Third and subsequent pops | 16 entries |

A physical stack push resets the pop trend and prefetch cursor. The walker advances a **prefetch cursor** instead of scanning the same top entries anew at every pop. Candidate node offsets are translated to cache-line-aligned addresses before being queued. The predictor no longer counts descent into the immediately selected child as an artificial push/pop pair.

This is a stack-based approximation of the design described by TTP. The stack organization, BVH representation, simulator timing, and precise prefetch scheduling are not identical to Vulkan-Sim.

### 2. Replay-aware predictor events

`rtu_core.cpp` recreates the functional traversal stack on every memory-stall replay. To avoid replaying speculative side effects, the implementation adds a per-context **processed stack-event watermark**:

- Every functional stack push/pop contributes to a replay event sequence.
- The walker reconstructs its pop trend and cursor during a replay, including events from the already-visited prefix.
- Only *newly reached* events may emit prefetch candidates to the memory engine.
- The context retains the processed-event count across demand misses and resets it on new context binding.

`SceneView` carries the input watermark and the walker's resulting progress, while `Context` retains the high-water mark and diagnostic predictor snapshots. This is **replay-aware reconstruction**, not storage of a persistent hardware traversal stack. An assumption is that the traversal-event prefix remains deterministic across probes; the current core checks for a shrinking event count.

### 3. Prefetch queue and scheduling

The original `rtu_memory.cpp` serviced **demand-only** fetches. The modified memory engine also accepts speculative cache-line addresses:

- `prefetch_queue_`: FIFO of candidate lines awaiting issue.
- `queued_prefetches_`: suppresses duplicate queued addresses.
- `prefetched_lines_`: records completed speculative addresses for temporal-use accounting and additional filtering.
- Checks against outstanding requests and context-local fetched lines suppress more redundant candidates.
- Demand requests have priority; prefetches issue only when no demand is selected for the available request port.
- Both request classes share a finite table sized by `VX_CFG_RTU_NUM_CTX`; prefetch admission leaves at least one free entry at issue time.
- A demand requesting the same address as an outstanding demand/prefetch can join its existing entry and receive the same response.
- A speculative request is marked with `MemReq.is_prefetch = true` for downstream accounting.

Leaving one entry free does **not** guarantee that all subsequent demands can issue without contention. The observed `mshr_full` counter can increase when speculative requests occupy entries. The RTU request table is separate from the cache's internal miss-handling structures.

### 4. Cache-level instrumentation

The RTCache implementation distinguishes demand versus speculative requests, maintaining counters for:

```text
[ttp-cache] cache=... demand_hits=... demand_misses=... demand_merges=...
            prefetch_hits=... prefetch_misses=... prefetch_merges=...
```

The prefetch marker is propagated through the cache request/response processing paths so that speculative cache hits and misses can be counted separately from demand hits and misses. For relevant test configurations, the cache prints statistics when the simulated cache is destroyed.

### 5. RTU counters and feature switch

TTP is selected at runtime:

- `VX_RTU_TTP=0`: TTP disabled (baseline).
- `VX_RTU_TTP=1`: TTP enabled.
- `VX_RTU_STATS=1`: print RTU statistics, including prefetch requests, responses, demand accesses, and traversal counters.

The original demand-only execution path remains available for A/B comparisons.

## Source-code map

| Component | File | Role / project changes |
| --- | --- | --- |
| Traversal and prediction | `sim/simx/rtu/rtu_walker.cpp` | `ShortStack`, actual push/pop hooks, 1/2/16 lookahead, prefetch cursor, line-address candidates, replay-event numbering |
| Ray scheduling | `sim/simx/rtu/rtu_core.cpp` | Enables TTP from the environment, passes `SceneView` progress, queues predictions, prints RTU performance counters |
| Shared RTU state | `sim/simx/rtu/rtu_types.h` | TTP fields in `SceneView`/`Context`, per-context replay watermark, statistics |
| Memory engine | `sim/simx/rtu/rtu_memory.cpp` | Prefetch FIFO, deduplication, demand-first arbitration, speculative request entries, joining outstanding reads, response handling |
| Memory-engine declarations | `sim/simx/rtu/rtu_memory.h` | Corresponding speculative state and method declarations |
| Memory-request type | SimX `MemReq` definition (e.g., `sim/simx/types.h`) | `is_prefetch` metadata forwarded to the cache |
| RTCache | `sim/simx/mem/cache.cpp` | Propagates the prefetch flag and tracks demand/prefetch hits, misses, merges |
| Ray-tracing benchmark | `tests/raytracing/rt_raycast/main.cpp` | OBJ loading, BVH construction, scene-size guard, rendering, CPU reference, image comparison |
| Test build | `tests/raytracing/rt_raycast/Makefile` | Builds the RTU ray-tracing test; default small scene |

Some modified logic is spread across multiple files: `rtu_walker.cpp` predicts *which* lines to fetch, `rtu_memory.cpp` decides *when and how* to issue them, and `cache.cpp` observes their cache outcomes.

## Build and run

### Prerequisites

A working Vortex checkout with its toolchain, SimX runtime, and ray-tracing test configured. If setting up from a fresh checkout, initialize the repository's submodules:

```bash
git submodule update --init --recursive
```

The commands below assume an **out-of-tree build directory** at `~/vortex/ttp/build`; adjust the path for your installation. The RTU extension must remain enabled when compiling SimX.

### Build the SimX runtime after RTU changes

```bash
cd ~/vortex/ttp/build
make -C sw/runtime/simx -j4 CONFIGS="-DVX_CFG_EXT_RTU_ENABLE"
```

This targets the SimX runtime without intentionally rebuilding the RTL/Verilator simulators. Avoid `make -B` in ordinary incremental development: it forces recompilation and can trigger a large rebuild. If this command says everything is up to date after a source edit, check the source/object/library timestamps and build dependencies before assuming the new code was linked.

### Build the benchmark after changing `main.cpp`

```bash
cd ~/vortex/ttp/build/tests/raytracing/rt_raycast
make -j4
```

### Example: A/B benchmark

```bash
cd ~/vortex/ttp/build/tests/raytracing/rt_raycast
export LD_LIBRARY_PATH="$HOME/vortex/ttp/build/sw/runtime:${LD_LIBRARY_PATH:-}"
export MODEL="$HOME/vortex/ttp/tests/regression/raycast/assets/dragon.obj"

for TTP in 0 1; do
  VX_RTU_TTP="$TTP" VX_RTU_STATS=1 VORTEX_DRIVER=simx \
    ./rt_raycast -m "$MODEL" -w 32 -h 24 \
    2>&1 | tee "/tmp/dragon-32x24-$TTP.log"
  status=${PIPESTATUS[0]}
  echo "TTP=$TTP exit code: $status"
  if [ "$status" -ne 0 ]; then
    echo "Run failed; stopping comparison" >&2
    break
  fi
done

grep -E 'PERF:|ttp-cache|rtu-stats.*(mem_reads|ttp=|stack_pushes=)|PASSED|FAILED' \
  /tmp/dragon-32x24-*.log
```

The benchmark writes `output.ppm` and `reference.ppm` in the working directory; successive runs overwrite them. To compare images from both settings, rename or copy the first output before running the second. `tee` otherwise reports its own exit status, so the script explicitly checks `${PIPESTATUS[0]}`.

### OBJ models and scene-size limit

Example assets are stored at:

```text
tests/regression/raycast/assets/
```

Add a triangulated `.obj` model and supply its path with `-m`. The benchmark's loader may not support every OBJ feature; triangulate polygonal faces and verify that geometry is loaded correctly. Increasing render resolution increases ray count; increasing mesh complexity generally increases BVH size, but neither guarantees a higher miss rate.

The raycast benchmark originally rejected BVHs over **16 KiB**. For larger models, this project's test guard in `tests/raytracing/rt_raycast/main.cpp` was raised to allow the dragon's **4,397,456-byte** serialized BVH (an **8 MiB** test limit was used in development). This change is to the *test-side validation*: it does not assert support for arbitrary scene sizes in RTL or every memory configuration. `vx_buffer_create()` allocates the test scene using its serialized byte count.

The CPU reference image is computed by brute-force ray/triangle intersections, so very large meshes and high resolutions can make the reference step expensive even when simulated RTU traversal itself is manageable.

## Experimental evaluation

The measurements below were collected with **Vortex SimX**, using the replay-aware stack-pointer implementation. They are simulator-cycle measurements, not wall-clock times or measurements on physical GPU hardware.

### Dragon: 53,560 triangles; 75,405 BVH nodes; 4,397,456 bytes

| Resolution | TTP off cycles | TTP on cycles | Cycles saved | Cycle reduction |
| --- | ---: | ---: | ---: | ---: |
| 16 × 16 | 160,465 | 156,847 | 3,618 | 2.25% |
| 32 × 24 | 346,751 | 339,250 | 7,501 | 2.16% |
| 64 × 64 | 1,954,807 | 1,922,764 | 32,043 | 1.64% |
| 128 × 128 | 7,366,561 | 7,270,955 | 95,606 | 1.30% |

| Resolution | Demand misses, off | Demand misses, on | Prefetch misses, on | Prefetch requests, on | RTU `mshr_full`, on |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16 × 16 | 2,615 | 2,397 | 229 | 277 | 27 |
| 32 × 24 | 5,434 | 4,981 | 480 | 583 | 42 |
| 64 × 64 | 26,687 | 24,446 | 2,369 | 3,033 | 276 |
| 128 × 128 | 82,002 | 75,476 | 7,121 | 9,976 | 489 |

At 128 × 128, TTP reduced demand RTCache misses by **6,526 (7.96%)**, while total RTCache misses **increased** from 82,002 to 82,597 when prefetch misses are included. This illustrates the difference between **hiding demand latency** and **reducing total memory traffic**. Relative cycle savings declined with increasing image resolution; changing ray distribution, cache reuse, and contention are possible contributors, but the current counters do not isolate their individual effects.

The dragon runs passed the benchmark's configured image-comparison criterion. At 16 × 16, the output had **1 pixel differing beyond the threshold** (`1/256`, `maxdiff=117`) but still passed the test's tolerance; the 32 × 24 run reported zero mismatches. A `PASSED!` result does not always mean bit-identical images.

### Teapot: 1,024 triangles; 1,365 nodes; 79,184 bytes (32 × 24)

| Configuration | Cycles | Demand misses | Prefetch requests | Prefetch misses |
| --- | ---: | ---: | ---: | ---: |
| TTP disabled | 159,033 | 654 | 0 | 0 |
| Earlier aggressive predictor | 156,738 | 514 | 244 | 140 |
| Stack-pointer predictor | 158,380 | 602 | 99 | 52 |
| Replay-aware stack-pointer predictor | 158,387 | 602 | 95 | 52 |

The aggressive version generated prefetches during some descent operations and is **not** the same algorithm as the later, more selective stack-pointer predictor. These numbers are included for development history, not as an apples-to-apples comparison of paper fidelity.

Smaller BVHs (e.g., sphere/cone at roughly 4.7–4.8 KB) showed very high baseline cache hit rates and negligible or negative prefetch benefit. This motivated larger-scene testing.

## Interpreting statistics correctly

- **Demand hits/misses** report RTCache outcomes for requests marked as non-prefetch.
- **Prefetch hits/misses** report outcomes of speculative requests. A *prefetch miss* is not automatically a useful prefetch.
- **Total RTCache misses** = demand misses + prefetch misses (plus any separately counted merges if a particular analysis requires them). TTP may increase this total while still improving execution time.
- **`prefetch_useful` is a temporal-association counter**, not proof that the corresponding request avoided a miss or reduced execution cycles. A later demand can match an earlier prefetch even if the prefetch was itself an RTCache hit. In the current benchmark logs it often equals `prefetch_requests`; do **not** interpret that as 100% effective prediction accuracy.
- **`mshr_full`** counts RTU memory-engine allocation stalls, not necessarily occupancy of the RTCache's own MSHRs.
- **Cycle reduction** is computed as `(cycles_off - cycles_on) / cycles_off`; actual speedup is `cycles_off / cycles_on`.
- All reported experiments are from the specified tested configurations; alternate RTU configurations, scenes, or simulator revisions need fresh measurements.

## Current limitations and future work

1. **Prefetch timing is approximate.** The walker computes functional traversal progress and generates candidate addresses during a probe. The core queues those candidates after the walk returns; the request is not necessarily issued at the precise simulated cycle of the triggering stack pop. A cycle-stepped traversal/prefetch pipeline would improve timing fidelity.
2. **Traversal is replayed rather than resumed from a stored hardware stack.** The high-water mark suppresses replay-duplicate predictions and reconstructs predictor state, but this is not a persistent RTL traversal stack or complete architectural state checkpoint.
3. **`prefetched_lines_` is not eviction-aware.** Completed prefetch addresses can remain recorded after the corresponding RTCache line is evicted, potentially suppressing a beneficial later prefetch.
4. **Speculative/demand contention.** Both use the RTU's bounded outstanding-request table. Demand-first arbitration and reserving one free entry mitigate—but do not eliminate—demand stalls under load.
5. **Prefetch usefulness needs better instrumentation.** Future counters should distinguish redundant cache hits, late prefetches joined by demands, timely cache fills subsequently used, unused fills, and any cache pollution. Cycle timestamps for prediction, issue, completion, and demand would quantify lead time.
6. **Algorithmic fidelity remains to be verified against Vulkan-Sim.** The conceptual stack-pop predictor is implemented, but stack semantics, pointer lifecycle, cache granularity, prefetch scheduling, and the latency model differ across simulators. Side-by-side tracing with matched BVHs/rays is a stronger comparison than raw source-code line counts.
7. **This is SimX-only.** The work has not been validated as synthesizable RTL and should not be described as an FPGA or physical GPU performance result.
8. **Benchmark coverage.** Additional scenes, cache sizes, memory latencies, RTU context counts, and controlled repeated runs are needed to characterize the design beyond the tested configurations.

## Suggested reproduction protocol

1. Record the Git commit, SimX configuration, RTCache parameters, scene file, resolution, and relevant environment variables.
2. Run with `VX_RTU_TTP=0`, capture the complete log and image, and confirm the exit code.
3. Repeat with `VX_RTU_TTP=1`, changing no other variables.
4. Check image correctness and compare `PERF`, `[ttp-cache]`, and `[rtu-stats]` values.
5. Report *both* demand and speculative misses; don't equate speculative traffic with useful prefetches.
6. Use multiple scene sizes and resolutions, and rerun after any change to simulator timing or cache behavior.

## Reference

- **TTP: A Hardware-Efficient Design for Precise Prefetching in Ray Tracing**, ISCA 2026. This work is an independent adaptation of the paper's proposed technique to Vortex SimX. Consult the paper and its Vulkan-Sim implementation for the original design, methodology, and results.
- **Vortex GPGPU**: https://github.com/vortexgpgpu/vortex

## Project status

**Functional SimX prototype with reproducible A/B benchmarks.** The current implementation passes the configured raycast comparison on tested scenes and demonstrates modest simulated cycle reductions on larger BVHs. The next focus is measuring prefetch timeliness and improving request scheduling/timing fidelity.
