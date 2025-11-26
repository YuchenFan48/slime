
import torch
import sys
import os

# 添加路径
sys.path.append("/mnt/shared-storage-user/p1-shared/yuchenzhang/infllmv2_cuda_impl")

from infllm_v2 import infllmv2_attn_stage1

def test_stage1_shapes():
    print("Testing infllmv2_attn_stage1 shapes...")
    
    # 模拟参数
    total_q = 7552
    nheads = 16
    nheads_k = 1
    head_dim = 128
    max_seqlen_k = 2134
    
    # 创建输入
    q = torch.randn(total_q, nheads, head_dim, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(total_q, nheads_k, head_dim, device='cuda', dtype=torch.bfloat16)
    v = torch.randn(total_q, nheads_k, head_dim, device='cuda', dtype=torch.bfloat16)
    
    # 模拟 cu_seqlens
    # 假设只有一个 sequence
    cu_seqlens_q = torch.tensor([0, total_q], device='cuda', dtype=torch.int32)
    cu_seqlens_k = torch.tensor([0, total_q], device='cuda', dtype=torch.int32)
    
    print(f"Input Q shape: {q.shape}")
    print(f"Input K shape: {k.shape}")
    
    # 调用 Stage 1
    try:
        scores = infllmv2_attn_stage1(
            q, k, v,
            cu_seqlens_q, cu_seqlens_k, cu_seqlens_k,
            total_q, total_q,
            dropout_p=0.0,
            causal=True,
            return_attn_probs=True
        )
        
        print(f"Output Scores shape: {scores.shape}")
        
        expected_expanded_q = total_q * (nheads // nheads_k)
        print(f"Expected expanded Q dim: {expected_expanded_q}")
        
        if scores.shape[1] == expected_expanded_q:
            print("Result: Output IS expanded (needs aggregation)")
        elif scores.shape[1] == total_q:
            print("Result: Output IS NOT expanded (already aggregated or no expansion)")
        else:
            print(f"Result: Unexpected shape {scores.shape}")
            
    except Exception as e:
        print(f"Error during execution: {e}")

if __name__ == "__main__":
    test_stage1_shapes()


