// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "rtu_memory.h"

#include <algorithm>
#include <cassert>
#include <cstring>

#include "rtu_types.h"  // Context, CtxState, PerfStats

namespace vortex { namespace rtu {

namespace {
// Match the RTU context capacity. Keep one free entry for demand requests
// when issuing speculative requests, even with a single configured context.
constexpr uint32_t kMshrEntries = VX_CFG_RTU_NUM_CTX;
}  // namespace

void MemoryEngine::reset() {
  mshr_.assign(kMshrEntries, Mshr{});
  pending_.clear();
  prefetch_queue_.clear();
  queued_prefetches_.clear();
  prefetched_lines_.clear();
  next_tag_ = 0;
  rr_ctx_ = 0;
  // The existing owner is responsible for resetting perf_ between runs.
}

void MemoryEngine::queue_prefetches(const std::vector<uint64_t>& addresses) {
  for (uint64_t addr : addresses) {
    bool in_flight = false;
    for (const auto& entry : mshr_) {
      if (entry.valid && entry.addr == addr) {
        in_flight = true;
        break;
      }
    }
    if (in_flight || prefetched_lines_.count(addr) != 0 ||
        queued_prefetches_.count(addr) != 0) {
      continue;
    }
    prefetch_queue_.push_back(addr);
    queued_prefetches_[addr] = true;
  }
}

void MemoryEngine::issue_memory() {
  if (mshr_.empty()) mshr_.assign(kMshrEntries, Mshr{});
  const uint32_t nctx = uint32_t(contexts_.size());
  bool issued_this_tick = false;

  auto free_mshr = [&]() -> uint32_t {
    for (uint32_t i = 0; i < mshr_.size(); ++i) {
      if (!mshr_[i].valid) return i;
    }
    return uint32_t(mshr_.size());
  };

  auto free_mshr_count = [&]() -> uint32_t {
    uint32_t count = 0;
    for (const auto& entry : mshr_) {
      if (!entry.valid) ++count;
    }
    return count;
  };

  // Erase a queued prefetch from BOTH its lookup map and FIFO. Merely erasing
  // the map leaves a stale FIFO entry that could be issued later.
  auto cancel_queued_prefetch = [&](uint64_t addr) {
    queued_prefetches_.erase(addr);
    prefetch_queue_.erase(
        std::remove(prefetch_queue_.begin(), prefetch_queue_.end(), addr),
        prefetch_queue_.end());
  };

  auto issue_prefetch = [&](SimChannel<MemReq>& port) -> bool {
    while (!prefetch_queue_.empty()) {
      // An RTU with one MSHR cannot safely prefetch without blocking demand.
      if (free_mshr_count() <= 1) return false;

      const uint64_t addr = prefetch_queue_.front();
      prefetch_queue_.pop_front();
      queued_prefetches_.erase(addr);

      bool demand_present = false;
      for (const auto& cx : contexts_) {
        if (!cx.valid) continue;
        if (((cx.state == CtxState::REQ || cx.state == CtxState::WAIT) &&
             cx.req_addr == addr) ||
            cx.lines.count(addr) != 0) {
          demand_present = true;
          break;
        }
      }

      bool in_flight = false;
      for (const auto& entry : mshr_) {
        if (entry.valid && entry.addr == addr) {
          in_flight = true;
          break;
        }
      }

      if (demand_present || in_flight || prefetched_lines_.count(addr) != 0)
        continue;

      const uint32_t eidx = free_mshr();
      if (eidx == mshr_.size()) {
        prefetch_queue_.push_front(addr);
        queued_prefetches_[addr] = true;
        return false;
      }

      Mshr& e = mshr_[eidx];
      e = Mshr{};
      e.valid = true;
      e.prefetch = true;
      e.addr = addr;

      const uint32_t tag = next_tag_++;
      pending_[tag] = eidx;

      MemReq m;
      m.addr = addr;
      m.op = MemOp::LD;
      m.tag = tag;
      m.hart_id = 0;
      m.uuid = 0;
      m.is_prefetch = true;
      port.send(m);

      ++perf_.prefetch_requests;
      return true;
    }
    return false;
  };

  for (auto& port : dcache_req_) {
    if (port.full()) continue;

    uint32_t leader = nctx;
    for (uint32_t i = 0; i < nctx; ++i) {
      const uint32_t idx = (rr_ctx_ + i) % nctx;
      if (contexts_[idx].valid && contexts_[idx].state == CtxState::REQ) {
        leader = idx;
        break;
      }
    }

    if (leader == nctx) {
      if (issue_prefetch(port)) issued_this_tick = true;
      continue;
    }

    const uint64_t addr = contexts_[leader].req_addr;
    rr_ctx_ = (leader + 1) % nctx;
    cancel_queued_prefetch(addr);

    // Coalesce a demand with ANY outstanding read of the same address,
    // including an earlier speculative prefetch. The existing response tag
    // will wake all attached contexts; do not issue a duplicate load.
    bool joined = false;
    for (auto& e : mshr_) {
      if (!e.valid || e.addr != addr) continue;
      if (std::find(e.waiters.begin(), e.waiters.end(), leader) ==
          e.waiters.end()) {
        e.waiters.push_back(leader);
      }
      if (e.prefetch) e.prefetch_used = true;
      contexts_[leader].state = CtxState::WAIT;
      joined = true;
      break;
    }
    if (joined) continue;

    const uint32_t eidx = free_mshr();
    if (eidx == mshr_.size()) {
      // Keep this context in REQ so it can retry on a later tick.
      ++perf_.mshr_full_stalls;
      break;
    }

    Mshr& e = mshr_[eidx];
    e = Mshr{};
    e.valid = true;
    e.prefetch = false;
    e.addr = addr;
    e.waiters.push_back(leader);

    // This is a *temporal association*, not proof that the prefetch avoided
    // an RTCache miss: the prefetch might itself have hit an existing line.
    auto prefetched = prefetched_lines_.find(addr);
    if (prefetched != prefetched_lines_.end()) {
      ++perf_.prefetch_useful;
      prefetched_lines_.erase(prefetched);
    }

    const uint32_t tag = next_tag_++;
    pending_[tag] = eidx;

    MemReq m;
    m.addr = addr;
    m.op = MemOp::LD;
    m.tag = tag;
    m.hart_id = 0;
    m.uuid = 0;
    port.send(m);

    contexts_[leader].state = CtxState::WAIT;
    ++perf_.mem_reads;
    issued_this_tick = true;
  }

  if (issued_this_tick) ++perf_.mem_issue_ticks;
}

void MemoryEngine::drain_mem_rsp() {
  // Drain at most one response per channel per cycle.
  for (auto& ch : dcache_rsp_) {
    if (ch.empty()) continue;

    auto& rsp = ch.peek();
    auto it = pending_.find(uint32_t(rsp.tag));
    if (it == pending_.end()) {
      ch.pop();
      continue;
    }

    Mshr& e = mshr_.at(it->second);
    pending_.erase(it);

    if (e.prefetch) {
      ++perf_.prefetch_responses;

      if (e.prefetch_used) {
        // A waiting demand will be served by the prefetch response below.
        // This represents an RTU-level useful prefetch, not necessarily a
        // prevented RTCache miss.
        ++perf_.prefetch_useful;
      } else {
        // A bookkeeping hint only: without RTCache eviction callbacks this
        // address is NOT guaranteed to be resident indefinitely.
        prefetched_lines_[e.addr] = true;
      }
    }

    // Both demand responses and prefetched responses with merged demand
    // waiters deliver the returned data and wake the waiting ray contexts.
    for (uint32_t c : e.waiters) {
      Context& cx = contexts_.at(c);
      assert(cx.state != CtxState::DONE &&
             "RTU: retired context in a memory fill group");
      if (!cx.valid || cx.state != CtxState::WAIT) continue;
      assert(rsp.data && "RTU: read response has no data");
      std::memcpy(cx.lines[e.addr].data(), rsp.data->data(),
                  VX_CFG_MEM_BLOCK_SIZE);
      cx.state = CtxState::PROBE;
    }

    e = Mshr{};
    ch.pop();
  }
}

}}  // namespace vortex::rtu
