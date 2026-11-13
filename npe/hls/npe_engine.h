#ifndef NPE_ENGINE_H_
#define NPE_ENGINE_H_

#include "npe_conv.h"
#include "npe_dwconv.h"
#include "npe_fc.h"
#include "npe_gap.h"
#include "npe_pool.h"

// Runs one instruction. Returns false if no unit of this build can run it.
// Each unit is one `if` here.
inline bool npe_step(const npe_instr_t &instr, NPE_BUF_PARAMS, NPE_MEM_PARAMS) {
  npe_dec_t d = npe_decode(instr);
  if (d.run_conv) {
    npe_conv(d, NPE_BUF_ARGS, NPE_MEM_ARGS);
    return true;
  }
  if (d.run_dw) {
    npe_dwconv(d, NPE_BUF_ARGS, NPE_MEM_ARGS);
    return true;
  }
  if (d.run_pool) {
    npe_pool_max(d, NPE_BUF_ARGS);
    return true;
  }
  if (d.run_gap) {
    npe_gap(d, NPE_BUF_ARGS);
    return true;
  }
  if (d.run_fc) {
    npe_fc(d, NPE_BUF_ARGS, NPE_MEM_ARGS);
    return true;
  }
  return false;
}

#endif // NPE_ENGINE_H_
