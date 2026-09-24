"""Golden tests for ABI struct layout declarations."""

from __future__ import annotations

from nebulasd.table.layout import (
    ABI_VERSION,
    ALL_STRUCT_LAYOUTS,
    CACHE_LINE_BYTES,
    ENDIANNESS,
    REQUEST_BLOCKS,
    active_request_table_bytes,
    physical_request_row_size_bytes,
    request_row_size_bytes,
)


EXPECTED_SIZES = {'EngineBlock': 88,
 'DispatchBlock': 216,
 'DraftWorkerBlock': 168,
 'TargetComputeBlock': 152,
 'D2HBlock': 112,
 'H2DBlock': 104,
 'HostKVAllocationBlock': 48,
 'DraftHostKVAllocationBlock': 72,
 'DraftD2HBlock': 200,
 'DraftH2DBlock': 216,
 'WorkerCommonBlock': 48,
 'DraftRuntimeBlock': 40,
 'TargetComputeRuntimeBlock': 40,
 'TargetCopyRuntimeBlock': 40,
 'BankBlock': 56,
 'DraftCopyRuntimeBlock': 40,
 'DraftBankBlock': 56}

EXPECTED_ROW_STRIDES = {'EngineBlock': 88,
 'DispatchBlock': 216,
 'DraftWorkerBlock': 192,
 'TargetComputeBlock': 192,
 'D2HBlock': 128,
 'H2DBlock': 128,
 'HostKVAllocationBlock': 48,
 'DraftHostKVAllocationBlock': 72,
 'DraftD2HBlock': 256,
 'DraftH2DBlock': 256,
 'WorkerCommonBlock': 48,
 'DraftRuntimeBlock': 40,
 'TargetComputeRuntimeBlock': 40,
 'TargetCopyRuntimeBlock': 40,
 'BankBlock': 56,
 'DraftCopyRuntimeBlock': 40,
 'DraftBankBlock': 56}

EXPECTED_OFFSETS = {'EngineBlock': {'publish_seq': 0,
                 'request_epoch': 8,
                 'current_round_id': 16,
                 'arrival_seq': 24,
                 'lifecycle': 32,
                 'prompt_token_count': 36,
                 'max_new_tokens': 40,
                 'spec_token_limit': 44,
                 'input_tokens_handle': 48,
                 'generation_config_handle': 64,
                 'classified_result_ticket': 80},
 'DispatchBlock': {'publish_seq': 0,
                   'draft_issue_seq': 8,
                   'draft_worker_generation': 16,
                   'draft_round_id': 24,
                   'target_prepare_seq': 32,
                   'planned_target_generation': 40,
                   'planned_bank_epoch': 48,
                   'target_run_seq': 56,
                   'target_round_id': 64,
                   'draft_worker_id': 72,
                   'planned_target_id': 76,
                   'planned_bank_id': 80,
                   'request_epoch': 88,
                   'draft_owner_epoch': 96,
                   'draft_prepare_seq': 104,
                   'planned_draft_id': 112,
                   'planned_draft_generation': 120,
                   'planned_draft_bank_id': 128,
                   'planned_draft_bank_epoch': 136,
                   'draft_prepared_batch_seq': 144,
                   'draft_source_snapshot_version': 152,
                   'draft_snapshot_round_id': 160,
                   'draft_snapshot_handle': 168,
                   'draft_next_owner_epoch': 184,
                   'draft_run_bank_id': 192,
                   'draft_run_bank_epoch': 200,
                   'draft_run_batch_seq': 208},
 'DraftWorkerBlock': {'publish_seq': 0,
                      'request_epoch': 8,
                      'round_id': 16,
                      'observed_issue_seq': 24,
                      'worker_generation': 32,
                      'worker_id': 40,
                      'status': 44,
                      'result_code': 48,
                      'compute_start_ns': 56,
                      'compute_end_ns': 64,
                      'proposal_token_count': 72,
                      'proposal_handle': 80,
                      'draft_state_handle': 96,
                      'owner_epoch': 112,
                      'snapshot_version': 120,
                      'logical_kv_len': 128,
                      'valid_blocks': 132,
                      'bank_id': 136,
                      'bank_epoch': 144,
                      'batch_seq': 152,
                      'dirty_begin_block': 160,
                      'dirty_block_count': 164},
 'TargetComputeBlock': {'publish_seq': 0,
                        'request_epoch': 8,
                        'round_id': 16,
                        'observed_run_seq': 24,
                        'target_generation': 32,
                        'bank_epoch': 40,
                        'target_kv_version': 48,
                        'target_id': 56,
                        'status': 60,
                        'result_code': 64,
                        'bank_id': 68,
                        'compute_start_ns': 72,
                        'compute_end_ns': 80,
                        'output_count': 88,
                        'output_finished': 92,
                        'output_handle': 96,
                        'accepted_draft_count': 112,
                        'committed_delta_count': 116,
                        'last_committed_token': 120,
                        'logical_kv_len': 124,
                        'dirty_begin_block': 128,
                        'dirty_block_count': 132,
                        'committed_delta_handle': 136},
 'D2HBlock': {'publish_seq': 0,
              'request_epoch': 8,
              'round_id': 16,
              'd2h_op_seq': 24,
              'target_generation': 32,
              'source_bank_epoch': 40,
              'host_slot_generation': 48,
              'writer_version': 56,
              'ready_version': 64,
              'target_id': 72,
              'status': 76,
              'result_code': 80,
              'source_bank_id': 84,
              'committed_blocks': 88,
              'logical_kv_len': 92,
              'copy_start_time_ns': 96,
              'copy_bytes': 104},
 'H2DBlock': {'publish_seq': 0,
              'request_epoch': 8,
              'round_id': 16,
              'observed_prepare_seq': 24,
              'target_generation': 32,
              'source_host_version': 40,
              'destination_bank_epoch': 48,
              'gpu_ready_version': 56,
              'target_id': 64,
              'status': 68,
              'result_code': 72,
              'destination_bank_id': 76,
              'copied_blocks': 80,
              'copy_start_time_ns': 88,
              'copy_bytes': 96},
 'HostKVAllocationBlock': {'publish_seq': 0,
                           'request_epoch': 8,
                           'host_slot_generation': 16,
                           'writer_lease_generation': 24,
                           'host_slot': 32,
                           'capacity_blocks': 36,
                           'offset_blocks': 40},
 'DraftHostKVAllocationBlock': {'publish_seq': 0,
                                'request_epoch': 8,
                                'arena_id': 16,
                                'arena_generation': 20,
                                'layout_id': 24,
                                'host_slot_generation': 32,
                                'writer_lease_generation': 40,
                                'offset_blocks': 48,
                                'host_slot': 56,
                                'capacity_blocks': 60,
                                'block_size': 64},
 'DraftD2HBlock': {'publish_seq': 0,
                   'request_epoch': 8,
                   'snapshot_round_id': 16,
                   'snapshot_version': 24,
                   'snapshot_handle': 32,
                   'logical_kv_len': 48,
                   'status': 52,
                   'result_code': 56,
                   'source_worker_id': 60,
                   'source_worker_generation': 64,
                   'source_op_seq': 72,
                   'owner_epoch': 80,
                   'source_bank_id': 88,
                   'source_bank_epoch': 96,
                   'source_batch_seq': 104,
                   'ready_version': 112,
                   'valid_blocks': 120,
                   'arena_id': 124,
                   'arena_generation': 128,
                   'layout_id': 136,
                   'host_slot_generation': 144,
                   'writer_lease_generation': 152,
                   'offset_blocks': 160,
                   'host_slot': 168,
                   'capacity_blocks': 172,
                   'block_size': 176,
                   'copy_start_time_ns': 184,
                   'copy_bytes': 192},
 'DraftH2DBlock': {'publish_seq': 0,
                   'request_epoch': 8,
                   'snapshot_round_id': 16,
                   'snapshot_version': 24,
                   'snapshot_handle': 32,
                   'logical_kv_len': 48,
                   'status': 52,
                   'result_code': 56,
                   'next_round_id': 64,
                   'observed_prepare_seq': 72,
                   'destination_worker_id': 80,
                   'destination_worker_generation': 88,
                   'next_owner_epoch': 96,
                   'destination_bank_id': 104,
                   'destination_bank_epoch': 112,
                   'prepared_batch_seq': 120,
                   'gpu_ready_version': 128,
                   'copied_blocks': 136,
                   'local_row': 140,
                   'arena_id': 144,
                   'arena_generation': 148,
                   'layout_id': 152,
                   'host_slot_generation': 160,
                   'writer_lease_generation': 168,
                   'offset_blocks': 176,
                   'host_slot': 184,
                   'capacity_blocks': 188,
                   'block_size': 192,
                   'copy_start_time_ns': 200,
                   'copy_bytes': 208},
 'WorkerCommonBlock': {'publish_seq': 0,
                       'worker_id': 8,
                       'role': 12,
                       'worker_generation': 16,
                       'status': 24,
                       'command_consumer_seq': 32,
                       'max_batch_size': 40,
                       'max_batch_tokens': 44},
 'DraftRuntimeBlock': {'publish_seq': 0,
                       'current_batch_seq': 8,
                       'compute_status': 16,
                       'compute_start_time_ns': 24,
                       'batch_request_count': 32,
                       'batch_token_count': 36},
 'TargetComputeRuntimeBlock': {'publish_seq': 0,
                               'compute_batch_seq': 8,
                               'compute_status': 16,
                               'compute_start_time_ns': 24,
                               'compute_request_count': 32,
                               'compute_token_count': 36},
 'TargetCopyRuntimeBlock': {'publish_seq': 0,
                            'copy_op_seq': 8,
                            'copy_status': 16,
                            'copy_start_time_ns': 24,
                            'copy_bytes': 32},
 'BankBlock': {'publish_seq': 0,
               'bank_id': 8,
               'bank_epoch': 16,
               'role': 24,
               'state': 28,
               'batch_seq': 32,
               'capacity_blocks': 40,
               'alloc_ptr_blocks': 44,
               'capacity_rows': 48,
               'alloc_rows': 52},
 'DraftCopyRuntimeBlock': {'publish_seq': 0,
                           'copy_op_seq': 8,
                           'copy_status': 16,
                           'copy_start_time_ns': 24,
                           'copy_bytes': 32},
 'DraftBankBlock': {'publish_seq': 0,
                    'bank_id': 8,
                    'bank_epoch': 16,
                    'role': 24,
                    'state': 28,
                    'batch_seq': 32,
                    'capacity_blocks': 40,
                    'alloc_ptr_blocks': 44,
                    'capacity_rows': 48,
                    'alloc_rows': 52}}


def test_layout_metadata_is_native_abi_not_python_object_layout() -> None:
    assert ABI_VERSION == 6
    assert ENDIANNESS == "little"
    assert CACHE_LINE_BYTES == 64
    for layout in ALL_STRUCT_LAYOUTS:
        assert layout.version == ABI_VERSION
        assert layout.alignment == 8
        assert layout.partition_alignment == CACHE_LINE_BYTES


def test_struct_sizes_are_stable() -> None:
    assert {layout.name: layout.size for layout in ALL_STRUCT_LAYOUTS} == EXPECTED_SIZES


def test_row_strides_are_stable_and_include_false_sharing_padding() -> None:
    assert {layout.name: layout.row_stride for layout in ALL_STRUCT_LAYOUTS} == EXPECTED_ROW_STRIDES
    for layout in ALL_STRUCT_LAYOUTS:
        if layout.multi_writer:
            assert layout.row_stride % CACHE_LINE_BYTES == 0
            assert layout.row_alignment == CACHE_LINE_BYTES
        else:
            assert layout.row_stride == layout.size


def test_field_offsets_are_stable() -> None:
    for layout in ALL_STRUCT_LAYOUTS:
        assert {field.field.name: field.offset for field in layout.field_layouts} == EXPECTED_OFFSETS[layout.name]


def test_hot_fields_are_a_contiguous_prefix() -> None:
    for layout in ALL_STRUCT_LAYOUTS:
        seen_cold = False
        for field in layout.fields:
            if field.hot:
                assert not seen_cold, f"{layout.name}.{field.name} places a hot field after cold fields"
            else:
                seen_cold = True


def test_each_owner_block_has_single_publication_marker_at_offset_zero() -> None:
    for layout in ALL_STRUCT_LAYOUTS:
        publish_fields = [field for field in layout.fields if field.name == "publish_seq"]
        assert len(publish_fields) == 1
        assert layout.field_offset("publish_seq") == 0


def test_request_layout_budget_matches_design_reference_range() -> None:
    assert request_row_size_bytes() == 1376
    assert physical_request_row_size_bytes() == 1576
    assert active_request_table_bytes(10_000) == 15760000
    assert active_request_table_bytes(100_000) == 157600000


def test_request_blocks_keep_writer_owners_separate() -> None:
    owners = [block.owner for block in REQUEST_BLOCKS]
    assert owners == [
        "engine",
        "dispatcher",
        "draft_worker",
        "target_compute_lane",
        "target_copy_lane",
        "target_copy_lane",
        "engine_hostkv_allocator",
        "engine_draft_hostkv_allocator",
        "draft_source_copy_lane",
        "draft_destination_copy_lane",
    ]
