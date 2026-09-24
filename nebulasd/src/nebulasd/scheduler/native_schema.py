"""Single source for the private numeric scheduler ABI (not a shared-table ABI)."""
REQUEST = 'slot epoch arrival prompt output maximum depth blocks admitted ready'.split()
ROW_FIELDS = {
 'e': ('REQUEST_ENGINE', 'request_epoch lifecycle current_round_id'),
 'x': ('REQUEST_DISPATCH', 'draft_issue_seq draft_round_id draft_worker_id draft_worker_generation draft_owner_epoch draft_prepare_seq target_run_seq target_round_id planned_target_id'),
 'd': ('REQUEST_DRAFT', 'request_epoch status result_code round_id observed_issue_seq worker_id worker_generation owner_epoch snapshot_version logical_kv_len valid_blocks dirty_block_count'),
 't': ('REQUEST_TARGET_COMPUTE', 'request_epoch status result_code round_id observed_run_seq target_id target_generation target_kv_version logical_kv_len dirty_block_count bank_id bank_epoch'),
 'h': ('REQUEST_D2H', 'request_epoch status ready_version round_id d2h_op_seq source_bank_id source_bank_epoch'),
 'dh': ('REQUEST_DRAFT_D2H', 'request_epoch status ready_version snapshot_version source_op_seq owner_epoch source_worker_generation'),
 'a': ('REQUEST_DRAFT_HOSTKV', 'request_epoch'),
}
REQUEST += [prefix+'_'+field for prefix,(_,fields) in ROW_FIELDS.items() for field in fields.split()]
REQUEST += 'draft_compute_round draft_compute_end_ns target_compute_round target_compute_end_ns'.split()
WORKER = 'id draft generation online max_batch block_size bank_blocks initial_rows prepare_rows initial_blocks prepare_blocks initial_tokens prepare_tokens runtime_seq runtime_start runtime_status copy_status'.split()
RECORD = 'worker operation seq compute_done physical_done h2d_blocks h2d_rows d2h_blocks d2h_rows count'.split()
STAGES = ('target_prefill','target_verify','draft_first','draft_cached','H2D','D2H')

def header():
    lines=['#pragma once', '#include <cstdint>']
    for name,fields in [('Request',REQUEST),('Worker',WORKER),('Record',RECORD)]:
        lines.append('struct '+name+' { '+ '; '.join('int64_t '+f for f in fields)+'; };')
    return '\n'.join(lines)+'\n'
