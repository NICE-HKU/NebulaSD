import dataclasses
import argparse

@dataclasses.dataclass
class EngineConfig:
    """
    Configuration for the SwiftLLM engine.
    """

    # Model loading parameters
    model_path: str
    use_dummy: bool

    # PagedAttention-related parameters
    block_size: int
    gpu_mem_utilization: float
    num_cpu_blocks: int
    max_seqs_in_block_table: int
    max_blocks_per_seq: int

    # Scheduling-related parameters
    max_batch_size: int
    max_tokens_in_batch: int

    # Speculative decoding parameters. Defaults preserve normal SwiftLLM serving.
    speculative_method: str = "none"
    speculative_max_draft_tokens: int = 0
    speculative_draft_model_path: str | None = None
    speculative_target_layer_ids: str | None = None

    # StarSD experimental KV path. Default false preserves legacy bitmap allocation.
    enable_double_bank: bool = False

    def __post_init__(self):
        valid_methods = {"none", "dflash", "eagle"}
        if self.speculative_method not in valid_methods:
            raise ValueError(f"speculative_method must be one of {sorted(valid_methods)}")
        if self.speculative_method == "eagle":
            raise NotImplementedError(
                "EAGLE interface is reserved but tree verification is not implemented yet."
            )
        if self.speculative_max_draft_tokens < 0:
            raise ValueError("speculative_max_draft_tokens must be non-negative")

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        """
        Add CLI arguments for the engine configuration
        """
        parser.add_argument(
            "--model-path",
            type=str,
            required=True,
            help="Path to the model directory (currently SwiftLLM does not support downloading from HuggingFace, so please download in advance)",
        )
        parser.add_argument(
            "--use-dummy",
            action="store_true",
            help="Use dummy weights (mainly for profiling)",
        )

        parser.add_argument(
            "--block-size",
            type=int,
            default=16,
            help="Block size for PagedAttention",
        )
        parser.add_argument(
            "--gpu-mem-utilization",
            type=float,
            default=0.97,
            help="Fraction of GPU memory to be used",
        )
        parser.add_argument(
            "--num-cpu-blocks",
            type=int,
            default=2048,
            help="Number of CPU blocks",
        )
        parser.add_argument(
            "--max-seqs-in-block-table",
            type=int,
            default=4096,
            help="Maximum number of sequences in the block table",
        )
        parser.add_argument(
            "--max-blocks-per-seq",
            type=int,
            default=32768,
            help="Maximum number of blocks per sequence",
        )

        parser.add_argument(
            "--max-batch-size",
            type=int,
            default=512,
            help="Maximum batch size",
        )
        parser.add_argument(
            "--max-tokens-in-batch",
            type=int,
            default=32768,
            help="Maximum number of tokens in a batch",
        )
        parser.add_argument(
            "--speculative-method",
            choices=["none", "dflash", "eagle"],
            default="none",
            help="Speculative decoding method. EAGLE is reserved but not implemented.",
        )
        parser.add_argument(
            "--speculative-max-draft-tokens",
            type=int,
            default=0,
            help="Maximum draft tokens accepted per speculative verification task",
        )
        parser.add_argument(
            "--speculative-draft-model-path",
            type=str,
            default=None,
            help="Optional embedded draft model path; StarSD target-worker mode supplies draft tokens externally",
        )
        parser.add_argument(
            "--speculative-target-layer-ids",
            type=str,
            default=None,
            help="Comma-separated target hidden layer ids reserved for StarSD-DFlash integration",
        )
        parser.add_argument(
            "--enable-double-bank",
            action="store_true",
            help="Enable experimental StarSD active/standby KV bank allocator",
        )

