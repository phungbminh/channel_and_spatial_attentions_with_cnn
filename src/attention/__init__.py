"""
Attention mechanism implementations.

This package contains:
- CBAM (Convolutional Block Attention Module)
- BAM (Bottleneck Attention Module) 
- scSE (spatial and channel Squeeze & Excitation)
"""

from .attentions_module import cbam_block, bam_block, scse_block

__all__ = [
    "cbam_block",
    "bam_block", 
    "scse_block"
]