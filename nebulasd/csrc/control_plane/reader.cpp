// Autonomous owner observation: bounded hints, stable copies and recovery in
// one call. Only version/retry metadata persists; payloads belong to Engine.
#include <vector>
#include <deque>
#include <array>
#include <algorithm>
#include <stdexcept>
#include <string>

namespace {
struct ReaderPartition { std::uintptr_t base; unsigned stride, size, count, kind; };
struct ReaderRing { std::uintptr_t base; U capacity; };
struct ReaderOutput { unsigned kind, row; U seq; unsigned size, reserved; };
struct ReaderRow {
  void* address; unsigned size, kind, slot; U seen=invalid, hint=invalid;
  bool pending=false; U turn=0; unsigned output=0;
};
struct Reader {
  std::vector<ReaderRow> rows;
  std::vector<ReaderRing> rings;
  std::array<int, 18> starts;
  std::array<unsigned, 18> counts{};
  std::vector<unsigned> priority;
  std::deque<unsigned> pending;
  unsigned cursor=0, scan=0;
  bool scanning=false, rescan=false;
  U turn=0;
  std::vector<unsigned char> scratch;
  Reader(const ReaderPartition* ps, unsigned np, const ReaderRing* rs, unsigned nr,
         const unsigned* keys, unsigned nk) : rings(rs,rs+nr) {
    starts.fill(-1);
    unsigned largest=0;
    for (unsigned p=0;p<np;++p) {
      auto x=ps[p];
      if (!x.kind || x.kind>=starts.size() || starts[x.kind]>=0 || x.size<8 || x.stride<x.size)
        throw std::runtime_error("invalid reader partition registration");
      starts[x.kind]=rows.size(); counts[x.kind]=x.count;
      for (unsigned r=0;r<x.count;++r)
        rows.push_back({reinterpret_cast<void*>(x.base+U(r)*x.stride),x.size,x.kind,r});
      largest=std::max(largest,x.size-8);
    }
    scratch.resize(largest);
    for (unsigned i=0;i<nk;++i) {
      int id=index(keys[2*i],keys[2*i+1]);
      if (id<0) throw std::runtime_error("invalid priority registration");
      priority.push_back(id);
    }
  }
  int index(unsigned kind,unsigned row) const {
    if (!kind || kind>=starts.size() || starts[kind]<0 || row>=counts[kind]) return -1;
    return starts[kind]+row;
  }
  static bool newer(U a,U b) {
    // OP_SEQ uses modulo UINT64_MAX (the all-ones sentinel is excluded).
    U distance=a>=b ? a-b : invalid-b+a;
    return distance && distance<=invalid/2;
  }
  void hint(unsigned id,U seq) {
    auto& r=rows[id];
    if (!r.pending) {r.pending=true;r.hint=seq;pending.push_back(id);}
    else if (newer(seq,r.hint)) r.hint=seq;
  }
  bool capture(unsigned id, ReaderOutput* output,unsigned char* payload,unsigned stride,unsigned& found) {
    auto& r=rows[id]; U seq;
    bool ok=false;
    for (unsigned retry=0;retry<16;++retry)
      if (sd_table_read(r.address,r.size,scratch.data(),&seq)) {ok=true;break;}
    if (!ok) return false;
    if (r.seen==seq) return true;
    if (r.turn!=turn) {r.turn=turn;r.output=found++;}
    output[r.output]={r.kind,r.slot,seq,r.size-8,0};
    std::memcpy(payload+U(r.output)*stride,scratch.data(),r.size-8);
    r.seen=seq;
    return true;
  }
  unsigned poll(unsigned budget,ReaderOutput* output,unsigned char* payload,unsigned stride,
                unsigned& scanned,bool& overflow) {
    if (!budget || stride<scratch.size()) throw std::runtime_error("invalid reader output budget");
    ++turn; unsigned remaining=budget;
    overflow=false;scanned=0;
    struct Hint {unsigned kind,row;U seq;};
    for (unsigned offset=0;offset<rings.size();++offset) {
      auto ring=rings[(cursor+offset)%rings.size()];void* base=reinterpret_cast<void*>(ring.base);
      unsigned share=remaining/(rings.size()-offset);
      store(bytes(base)+136,0);
      overflow|=sd_exchange(bytes(base)+128,0)!=0;
      U head=load(base),tail=load(bytes(base)+64);
      unsigned count=std::min<U>(tail-head,share);
      for (unsigned i=0;i<count;++i) {
        Hint h;std::memcpy(&h,bytes(base)+192+((head+i)%ring.capacity)*sizeof(Hint),sizeof(Hint));
        // Engine-local rows and per-request H2D completion (owned by Worker
        // execution) are deliberately unregistered. Match ENGINE_IGNORED_KINDS.
        if (h.kind==1 || h.kind==2 || h.kind==6 || h.kind==15) continue;
        int id=index(h.kind,h.row);
        if (id<0 || h.seq==invalid) throw std::runtime_error("invalid observation hint");
        hint(id,h.seq);
      }
      if (count) store(base,head+count);
      remaining-=count;
    }
    if (!rings.empty()) cursor=(cursor+1)%rings.size();
    if (overflow) rescan=true;
    if (!scanning && rescan) {scan=0;scanning=!rows.empty();rescan=false;}
    unsigned found=0;
    for (auto id:priority) {
      U seq=load(rows[id].address);
      if (seq!=invalid && seq!=rows[id].seen) capture(id,output,payload,stride,found);
    }
    unsigned count=std::min<size_t>(pending.size(),scanning ? budget/2 : budget);
    for (unsigned i=0;i<count;++i) {
      unsigned id=pending.front();pending.pop_front();auto& r=rows[id];r.pending=false;
      if (r.seen!=invalid && !newer(r.hint,r.seen)) continue;
      if (!capture(id,output,payload,stride,found)) hint(id,r.hint);
    }
    while (scanning && scanned<budget-count) {
      auto& r=rows[scan];U seq=load(r.address);
      if (seq!=invalid && seq!=r.seen && !capture(scan,output,payload,stride,found)) hint(scan,seq);
      ++scanned;
      if (++scan==rows.size()) scanning=false;
    }
    return found;
  }
  bool has_pending() const {
    if (!pending.empty() || scanning || rescan) return true;
    for (auto ring:rings) {
      void* p=reinterpret_cast<void*>(ring.base);
      if (load(p)!=load(bytes(p)+64) || load(bytes(p)+128)) return true;
    }
    return false;
  }
  void reset(const unsigned* kinds,unsigned count) {
    // Called only at the all-owner retirement barrier.
    for (unsigned i=0;i<count;++i) {
      unsigned k=kinds[i];
      if (k<starts.size() && starts[k]>=0)
        for (unsigned r=0;r<counts[k];++r) rows[starts[k]+r].seen=invalid;
    }
    pending.clear();scanning=rescan=false;
    for (auto& r:rows) r.pending=false;
    for (auto ring:rings) {
      void* p=reinterpret_cast<void*>(ring.base);
      store(bytes(p)+136,0);sd_exchange(bytes(p)+128,0);store(p,load(bytes(p)+64));
    }
  }
};
thread_local std::string reader_error;
}
extern "C" {
const char* sd_reader_error() {return reader_error.c_str();}
void* sd_reader_create(const ReaderPartition* p,unsigned np,const ReaderRing* r,unsigned nr,
                       const unsigned* priority,unsigned n) {
  try {return new Reader(p,np,r,nr,priority,n);} catch(const std::exception& e) {reader_error=e.what();return nullptr;}
}
void sd_reader_destroy(void* p) {delete static_cast<Reader*>(p);}
int sd_reader_poll(void* p,unsigned budget,ReaderOutput* out,unsigned char* payload,unsigned stride,
                   unsigned* scanned,unsigned* overflow) {
  try {bool o;auto n=static_cast<Reader*>(p)->poll(budget,out,payload,stride,*scanned,o);*overflow=o;return n;}
  catch(const std::exception& e) {reader_error=e.what();return -1;}
}
unsigned sd_reader_state(void* p) {
  auto& r=*static_cast<Reader*>(p);
  return (!r.pending.empty()) | (r.scanning ? 2 : 0) | (r.rescan ? 4 : 0);
}
void sd_reader_rescan(void* p) {static_cast<Reader*>(p)->rescan=true;}
int sd_reader_pending(void* p) {return static_cast<Reader*>(p)->has_pending();}
void sd_reader_reset(void* p,const unsigned* kinds,unsigned n) {static_cast<Reader*>(p)->reset(kinds,n);}
}
