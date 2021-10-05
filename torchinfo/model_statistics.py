""" model_statistics.py """
from typing import Any, List, Tuple

from .formatting import FormattingOptions
from .layer_info import LayerInfo, prod


class ModelStatistics:
    """Class for storing results of the summary."""

    def __init__(
        self,
        summary_list: List[LayerInfo],
        input_size: Any,
        total_input_size: int,
        formatting: FormattingOptions,
    ) -> None:
        self.summary_list = summary_list
        self.input_size = input_size
        self.formatting = formatting
        self.total_input = total_input_size
        self.total_fp_params, self.trainable_fp_params = 0, 0
        self.total_fp_output, self.total_fp_mult_adds = 0, 0
        self.total_quantized_params, self.trainable_quantized_params = 0, 0
        self.total_quantized_output, self.total_quantized_mult_adds = 0, 0

        for layer_info in summary_list:
            if layer_info.is_leaf_layer or layer_info.quantized:
                if layer_info.quantized:
                    self.total_quantized_mult_adds += layer_info.macs
                else:
                    self.total_fp_mult_adds += layer_info.macs
                if layer_info.is_recursive:
                    continue
                if layer_info.quantized:
                    self.total_quantized_params += layer_info.num_params
                else:
                    self.total_fp_params += layer_info.num_params
                if layer_info.trainable:
                    if layer_info.quantized:
                        self.trainable_quantized_params += layer_info.num_params
                    else:
                        self.trainable_fp_params += layer_info.num_params

                if layer_info.num_params > 0:
                    # x2 for gradients
                    if layer_info.quantized:
                        self.total_quantized_output += 2 * prod(layer_info.output_size)
                    else:
                        self.total_fp_output += 2 * prod(layer_info.output_size)
                

        self.formatting.set_layer_name_width(summary_list)

    def __repr__(self) -> str:
        """Print results of the summary."""
        divider = "=" * self.formatting.get_total_width()
        summary_str = (
            f"{divider}\n"
            f"{self.formatting.header_row()}{divider}\n"
            f"{self.formatting.layers_to_str(self.summary_list)}{divider}\n"
            f"Total full precision params: {self.total_fp_params:,}\n"
            f"Trainable full precision params: {self.trainable_fp_params:,}\n"
            f"Non-trainable full precision params: {self.total_fp_params - self.trainable_fp_params:,}\n{divider}\n"
            f"Total quantized params: {self.total_quantized_params:,}\n"
            f"Trainable quantized params: {self.trainable_quantized_params:,}\n"
            f"Non-trainable full precision params: {self.total_quantized_params - self.trainable_quantized_params:,}\n{divider}\n"
        )
        if self.input_size:
            total_size = (
                        self.to_megabytes(self.total_input)
                        + self.float_to_megabytes(self.total_fp_output + self.total_fp_params)
                        + self.bit_to_megabytes(self.total_quantized_output + self.total_quantized_params)
                    )
            summary_str += (
                f"Total full precision mult-adds: {self.to_readable(self.total_fp_mult_adds)}\n"
                f"Total quantized mult-adds: {self.to_readable(self.total_quantized_mult_adds)}\n{divider}\n"
                f"Input size (MB): {self.to_megabytes(self.total_input):0.2f}\n"
                f"Forward/backward pass size full precision (MB): {self.float_to_megabytes(self.total_fp_output):0.2f}\n"
                f"Estimated Forward/backward pass size quantized (assuming 1 bit resolution) (MB): {self.bit_to_megabytes(self.total_quantized_output):0.2f}\n"
                f"Forward/backward pass size total (MB): {self.float_to_megabytes(self.total_fp_output) + self.bit_to_megabytes(self.total_quantized_output):0.2f}\n"
                f"Params size full precision (MB): {self.float_to_megabytes(self.total_fp_params):0.2f}\n"
                f"Params size quantized (assuming 1 bit resolution) (MB): {self.bit_to_megabytes(self.total_quantized_params):0.2f}\n"
                f"Params size total (MB): {self.float_to_megabytes(self.total_fp_params) + self.bit_to_megabytes(self.total_quantized_params):0.2f}\n"
                f"Estimated Total Size (MB): {total_size:0.2f}\n"
            )
        summary_str += divider
        return summary_str

    @staticmethod
    def float_to_megabytes(num: int) -> float:
        """Converts a number (assume floats, 4 bytes each) to megabytes."""
        return num * 4 / 1e6

    @staticmethod
    def bit_to_megabytes(num: int) -> float:
        return num / 8 / 1e6

    @staticmethod
    def to_megabytes(num: int) -> float:
        """Converts a number (assume floats, 4 bytes each) to megabytes."""
        return num / 1e6

    @staticmethod
    def to_readable(num: int) -> Tuple[str, float]:
        """Converts a number to millions, billions, or trillions."""
        if num >= 1e12:
            return f"{num / 1e12:0.2f} trillion"
        if num >= 1e9:
            return f"{num / 1e9:0.2f} billion"
        return f"{num / 1e6:0.2f} million"
