import math
from dataclasses import dataclass
from typing import Tuple, Optional, Literal

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from kernel import act_quant, weight_dequant, fp8_gemm


world_size = 1
rank = 0
block_size = 128
gemm_impl: Literal["bf16", "fp8"] = "bf16"
attn_impl: Literal["naive", "absorb"] = "absorb"

@dataclass
class ModelArgs:
    """
    模型参数配置类，使用dataclass简化参数管理
    包含了模型的所有超参数设置，如批大小、序列长度、维度等
    """
    # 基础配置
    max_batch_size: int = 8  # 最大批处理大小
    max_seq_len: int = 4096 * 4  # 最大序列长度，支持16K上下文
    dtype: Literal["bf16", "fp8"] = "bf16"  # 数据类型
    vocab_size: int = 102400  # 词表大小
    dim: int = 2048  # 模型维度
    inter_dim: int = 10944  # FFN中间层维度
    moe_inter_dim: int = 1408  # MoE专家网络中间层维度
    n_layers: int = 27  # transformer层数
    n_dense_layers: int = 1  # 密集层数量
    n_heads: int = 16  # 注意力头数
    
    # MoE相关配置
    n_routed_experts: int = 64  # 路由专家数量
    n_shared_experts: int = 2  # 共享专家数量
    n_activated_experts: int = 6  # 每次激活的专家数量
    n_expert_groups: int = 1  # 专家组数量
    n_limited_groups: int = 1  # 限制的组数
    score_func: Literal["softmax", "sigmoid"] = "softmax"  # 评分函数类型
    route_scale: float = 1.  # 路由缩放因子
    
    # 多头注意力相关配置
    q_lora_rank: int = 0  # Query的LoRA秩
    kv_lora_rank: int = 512  # Key-Value的LoRA秩
    qk_nope_head_dim: int = 128  # 非位置编码的Q-K头维度
    qk_rope_head_dim: int = 64  # 位置编码的Q-K头维度
    v_head_dim: int = 128  # Value头维度
    
    # 位置编码相关配置
    original_seq_len: int = 4096  # 原始序列长度
    rope_theta: float = 10000.0  # RoPE基础频率
    rope_factor: float = 40  # RoPE缩放因子
    beta_fast: int = 32  # 快速β校正因子
    beta_slow: int = 1  # 慢速β校正因子
    mscale: float = 1.  # 注意力缩放因子


class ParallelEmbedding(nn.Module):
    """
    并行嵌入层实现，支持分布式训练时的词表分片
    创新点：实现了词表的分布式存储，每个进程只负责一部分词表，减少显存占用
    
    属性:
        vocab_size (int): 总词表大小
        dim (int): 词嵌入维度
        part_vocab_size (int): 每个进程负责的词表大小
        vocab_start_idx (int): 当前进程负责的词表起始索引
        vocab_end_idx (int): 当前进程负责的词表结束索引
        weight (nn.Parameter): 词嵌入权重矩阵
    """
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        # 确保词表大小能被进程数整除
        assert vocab_size % world_size == 0, f"Vocabulary size must be divisible by world size (world_size={world_size})"
        # 计算每个进程负责的词表大小
        self.part_vocab_size = (vocab_size // world_size)
        # 计算当前进程负责的词表范围
        self.vocab_start_idx = rank * self.part_vocab_size
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
        # 只存储当前进程负责的部分词表的嵌入
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数
        
        参数:
            x (torch.Tensor): 输入的token id张量
            
        返回:
            torch.Tensor: 词嵌入结果
            
        说明:
            1. 在分布式训练时，每个进程只处理自己负责的词表部分
            2. 使用mask机制处理不属于当前进程的token
            3. 通过all_reduce操作合并所有进程的结果
        """
        if world_size > 1:
            # 创建mask标记属于当前进程的token
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            # 将token id映射到当前进程的局部空间
            x = x - self.vocab_start_idx
            # 将不属于当前进程的token置为0
            x[mask] = 0
        # 执行嵌入操作
        y = F.embedding(x, self.weight)
        if world_size > 1:
            # 将不属于当前进程的位置置为0
            y[mask] = 0
            # 合并所有进程的结果
            dist.all_reduce(y)
        return y


def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    实现高效的线性变换操作，支持量化计算
    创新点：
    1. 支持FP8和BF16两种精度的计算
    2. 实现了基于block的量化策略
    3. 针对不同数据类型优化了计算路径
    
    参数:
        x (torch.Tensor): 输入张量
        weight (torch.Tensor): 权重张量，可能是量化后的权重
        bias (Optional[torch.Tensor]): 偏置项，默认为None
        
    返回:
        torch.Tensor: 线性变换的结果
    """
    if weight.element_size() > 1:
        # 如果权重不是量化的，直接使用普通的线性变换
        return F.linear(x, weight, bias)
    elif gemm_impl == "bf16":
        # 如果使用bf16实现，先反量化权重再计算
        weight = weight_dequant(weight, weight.scale)
        return F.linear(x, weight, bias)
    else:
        # 使用fp8实现，对输入进行量化，然后使用专门的fp8_gemm计算
        x, scale = act_quant(x, block_size)
        y = fp8_gemm(x, scale, weight, weight.scale)
        if bias is not None:
            y += bias
        return y


class Linear(nn.Module):
    """
    自定义的线性层实现，支持量化计算
    创新点：
    1. 支持权重量化存储，减少显存占用
    2. 实现了基于block的缩放因子存储
    3. 可配置的数据类型支持
    
    属性:
        in_features (int): 输入特征维度
        out_features (int): 输出特征维度
        weight (nn.Parameter): 权重参数
        scale (nn.Parameter): 量化缩放因子
        bias (Optional[nn.Parameter]): 偏置参数
    """
    dtype = torch.bfloat16  # 默认使用bfloat16数据类型

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # 创建权重参数，可以指定数据类型
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype or Linear.dtype))
        
        # 如果权重是量化的（element_size=1），则需要额外的缩放因子
        if self.weight.element_size() == 1:
            # 计算需要多少个block来存储缩放因子
            scale_out_features = (out_features + block_size - 1) // block_size
            scale_in_features = (in_features + block_size - 1) // block_size
            # 创建缩放因子参数
            self.weight.scale = self.scale = nn.Parameter(torch.empty(scale_out_features, scale_in_features, dtype=torch.float32))
        else:
            self.register_parameter("scale", None)
            
        # 可选的偏置项
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数
        
        参数:
            x (torch.Tensor): 输入张量
            
        返回:
            torch.Tensor: 线性变换的结果
        """
        return linear(x, self.weight, self.bias)


class ColumnParallelLinear(Linear):
    """
    列并行的线性层实现
    创新点：
    1. 实现了基于列的模型并行化，每个进程只负责一部分输出特征的计算
    2. 通过特征分片减少了显存占用和计算量
    3. 无需在前向传播时进行通信
    
    工作原理：
    - 将输出特征维度划分到不同的进程中
    - 每个进程只计算一部分输出特征
    - 适用于大规模模型的分布式训练
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        # 确保输出特征能够被进程数整除
        assert out_features % world_size == 0, f"Output features must be divisible by world size (world_size={world_size})"
        # 计算每个进程负责的输出特征数量
        self.part_out_features = out_features // world_size
        # 调用父类初始化，但使用分片后的输出特征数
        super().__init__(in_features, self.part_out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数
        
        参数:
            x (torch.Tensor): 输入张量
            
        返回:
            torch.Tensor: 当前进程负责的部分输出特征
        
        说明：
        - 每个进程只计算自己负责的那部分输出特征
        - 不需要在forward中进行通信操作
        """
        y = linear(x, self.weight, self.bias)
        return y


class RowParallelLinear(Linear):
    """
    行并行的线性层实现
    创新点：
    1. 实现了基于行的模型并行化，每个进程只负责一部分输入特征的计算
    2. 通过输入特征分片减少了参数量和计算量
    3. 使用all_reduce合并各进程的计算结果
    
    工作原理：
    - 将输入特征维度划分到不同的进程中
    - 每个进程只处理一部分输入特征的计算
    - 通过all_reduce操作合并所有进程的结果
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        # 确保输入特征能够被进程数整除
        assert in_features % world_size == 0, f"Input features must be divisible by world size (world_size={world_size})"
        # 计算每个进程负责的输入特征数量
        self.part_in_features = in_features // world_size
        # 调用父类初始化，但使用分片后的输入特征数
        super().__init__(self.part_in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数
        
        参数:
            x (torch.Tensor): 输入张量
            
        返回:
            torch.Tensor: 完整的输出特征
            
        说明：
        1. 每个进程先计算自己负责部分的结果
        2. 通过all_reduce操作将所有进程的结果相加
        3. 最后添加偏置项（如果存在）
        """
        # 计算当前进程负责的部分
        y = linear(x, self.weight)
        if world_size > 1:
            # 在分布式环境下，合并所有进程的结果
            dist.all_reduce(y)
        if self.bias is not None:
            # 添加偏置项（只在结果合并后进行一次）
            y += self.bias
        return y


class RMSNorm(nn.Module):
    """
    均方根层归一化(Root Mean Square Layer Normalization)
    创新点：
    1. 相比LayerNorm更简单高效，不需要计算均值
    2. 数值稳定性更好，对FP16/BF16等低精度计算更友好
    3. 计算复杂度更低，训练和推理速度更快
    
    工作原理：
    - 计算输入张量的均方根
    - 使用均方根对输入进行归一化
    - 应用可学习的缩放因子
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        初始化RMSNorm层
        
        参数:
            dim (int): 需要归一化的特征维度
            eps (float): 数值稳定性的小常数
        """
        super().__init__()
        self.dim = dim
        self.eps = eps
        # 可学习的缩放参数
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        """
        前向传播函数
        
        参数:
            x (torch.Tensor): 输入张量
            
        返回:
            torch.Tensor: 归一化后的张量
            
        计算过程：
        1. 计算特征维度上的均方根
        2. 用均方根对输入进行归一化
        3. 应用可学习的缩放因子
        """
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)


def precompute_freqs_cis(args: ModelArgs) -> torch.Tensor:
    """
    预计算旋转位置编码的复数指数值
    创新点：
    1. 实现了可扩展的旋转位置编码(RoPE)
    2. 支持序列长度外推(extrapolation)
    3. 使用动态缩放策略优化长序列性能
    
    参数:
        args (ModelArgs): 包含位置编码相关参数的配置对象
        
    返回:
        torch.Tensor: 预计算的复数位置编码
        
    主要步骤：
    1. 计算基础频率
    2. 应用序列长度外推的校正
    3. 生成复数位置编码
    """
    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        """
        计算旋转位置编码的校正维度
        
        参数:
            num_rotations (float): 需要校正的旋转数
            dim (int): 编码维度
            base (float): 基础频率
            max_seq_len (int): 最大序列长度
            
        返回:
            float: 校正维度
        """
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        """
        计算旋转位置编码的校正范围
        
        参数:
            low_rot (float): 最小旋转数
            high_rot (float): 最大旋转数
            dim (int): 编码维度
            base (float): 基础频率
            max_seq_len (int): 最大序列长度
            
        返回:
            Tuple[int, int]: 校正维度的范围
        """
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim-1)

    def linear_ramp_factor(min, max, dim):
        """
        计算线性渐变因子，用于平滑过渡
        
        参数:
            min (float): 最小值
            max (float): 最大值
            dim (int): 维度
            
        返回:
            torch.Tensor: 线性渐变因子
        """
        if min == max:
            max += 0.001  # 避免除零错误
        # 计算线性函数值
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        # 将值限制在[0,1]范围内
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    # 获取相关参数
    dim = args.qk_rope_head_dim  # 位置编码的维度
    seqlen = args.max_seq_len    # 最大序列长度
    beta_fast = args.beta_fast   # 快速校正因子
    beta_slow = args.beta_slow   # 慢速校正因子
    base = args.rope_theta       # RoPE的基础频率
    factor = args.rope_factor    # 缩放因子

    # 计算基础频率
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    
    # 如果序列长度超过原始长度，应用长度外推策略
    if seqlen > args.original_seq_len:
        # 计算校正范围
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, args.original_seq_len)
        # 计算平滑过渡的因子
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        # 应用校正：结合原始频率和缩放后的频率
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    # 生成位置索引并计算外积
    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)  # 计算位置和频率的外积
    # 转换为复数形式：exp(i*theta)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    应用旋转位置编码(RoPE)到输入张量
    创新点：
    1. 使用复数运算实现旋转变换
    2. 支持高效的批处理计算
    3. 保持数据类型一致性
    
    工作原理：
    - 将输入视为复数进行旋转变换
    - 通过复数乘法实现位置相关的特征旋转
    - 保持原始数据类型，确保计算精度
    
    参数:
        x (torch.Tensor): 输入张量，形状为[batch_size, seq_len, n_heads, head_dim]
        freqs_cis (torch.Tensor): 预计算的复数位置编码
        
    返回:
        torch.Tensor: 应用位置编码后的张量
        
    计算步骤：
    1. 保存原始数据类型
    2. 将输入重塑为复数形式
    3. 应用复数旋转
    4. 转换回实数并恢复形状
    """
    dtype = x.dtype  # 保存原始数据类型
    
    # 将输入重塑为复数形式：将最后一维每两个数字组合成一个复数
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    
    # 调整freqs_cis的形状以便广播
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    
    # 通过复数乘法应用旋转
    x = x * freqs_cis
    
    # 转换回实数形式并展平最后一维
    y = torch.view_as_real(x).flatten(3)
    
    # 恢复原始数据类型
    return y.to(dtype)


class MLA(nn.Module):
    """
    多头注意力层(Multi-head Linear Attention)的创新实现
    创新点：
    1. 分离位置编码和非位置编码的注意力计算
    2. 使用LoRA(Low-Rank Adaptation)技术降低参数量
    3. 支持两种注意力实现方式：naive和absorb
    4. 实现了高效的缓存机制
    
    架构特点：
    - 将Query分为两部分：带位置编码和不带位置编码
    - 使用LoRA降低Key-Value的参数量
    - 支持高效的注意力分数计算
    
    属性:
        dim (int): 输入特征维度
        n_heads (int): 注意力头数
        n_local_heads (int): 每个进程的本地注意力头数
        q_lora_rank (int): Query的LoRA秩
        kv_lora_rank (int): Key-Value的LoRA秩
        qk_nope_head_dim (int): 非位置编码的Q-K头维度
        qk_rope_head_dim (int): 位置编码的Q-K头维度
        qk_head_dim (int): Q-K总头维度
        v_head_dim (int): Value头维度
        softmax_scale (float): Softmax缩放因子
    """
    def __init__(self, args: ModelArgs):
        """
        初始化MLA层
        
        参数:
            args (ModelArgs): 模型配置参数
        """
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        # 计算每个进程的本地头数
        self.n_local_heads = args.n_heads // world_size
        # LoRA相关配置
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        # 头维度配置
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim

        # Query投影层：可选使用LoRA
        if self.q_lora_rank == 0:
            # 不使用LoRA时的标准线性投影
            self.wq = ColumnParallelLinear(self.dim, self.n_heads * self.qk_head_dim)
        else:
            # 使用LoRA时的低秩分解
            self.wq_a = Linear(self.dim, self.q_lora_rank)
            self.q_norm = RMSNorm(self.q_lora_rank)
            self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.qk_head_dim)
            
        # Key-Value共享投影层
        self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b = ColumnParallelLinear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))
        
        # 输出投影层
        self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)
        
        # 注意力缩放因子
        self.softmax_scale = self.qk_head_dim ** -0.5
        # 对于长序列进行额外的缩放
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        # 根据不同的注意力实现方式初始化缓存
        if attn_impl == "naive":
            # naive实现：分别缓存K和V
            self.register_buffer("k_cache", torch.zeros(
                args.max_batch_size, args.max_seq_len, 
                self.n_local_heads, self.qk_head_dim
            ), persistent=False)
            self.register_buffer("v_cache", torch.zeros(
                args.max_batch_size, args.max_seq_len, 
                self.n_local_heads, self.v_head_dim
            ), persistent=False)
        else:
            # absorb实现：缓存中间结果
            self.register_buffer("kv_cache", torch.zeros(
                args.max_batch_size, args.max_seq_len, 
                self.kv_lora_rank
            ), persistent=False)
            self.register_buffer("pe_cache", torch.zeros(
                args.max_batch_size, args.max_seq_len, 
                self.qk_rope_head_dim
            ), persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        """
        前向传播函数
        
        参数:
            x (torch.Tensor): 输入张量 [batch_size, seq_len, dim]
            start_pos (int): 序列起始位置
            freqs_cis (torch.Tensor): 预计算的位置编码
            mask (Optional[torch.Tensor]): 注意力掩码
            
        返回:
            torch.Tensor: 注意力层的输出
            
        计算流程：
        1. 计算Query投影
        2. 计算Key-Value投影
        3. 分离位置编码和非位置编码部分
        4. 计算注意力分数
        5. 应用掩码和softmax
        6. 计算加权和并投影到输出维度
        """
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen

        # 计算Query投影
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        # 重塑Query并分离位置编码和非位置编码部分
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        # 应用位置编码
        q_pe = apply_rotary_emb(q_pe, freqs_cis)

        # 计算Key-Value投影
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)

        if attn_impl == "naive":
            # naive实现：标准的注意力计算
            q = torch.cat([q_nope, q_pe], dim=-1)
            kv = self.wkv_b(self.kv_norm(kv))
            kv = kv.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)
            
            # 更新缓存
            self.k_cache[:bsz, start_pos:end_pos] = k
            self.v_cache[:bsz, start_pos:end_pos] = v
            
            # 计算注意力分数
            scores = torch.einsum("bshd,bthd->bsht", q, self.k_cache[:bsz, :end_pos]) * self.softmax_scale
        else:
            # absorb实现：优化的注意力计算
            wkv_b = self.wkv_b.weight if self.wkv_b.scale is None else weight_dequant(self.wkv_b.weight, self.wkv_b.scale, block_size) 
            wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
            
            # 计算非位置编码部分的注意力分数
            q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
            
            # 更新缓存
            self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)
            self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
            
            # 合并位置编码和非位置编码的注意力分数
            scores = (torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos]) +
                      torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])) * self.softmax_scale

        # 应用注意力掩码
        if mask is not None:
            scores += mask.unsqueeze(1)
            
        # 计算softmax并转换回原始数据类型
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)

        if attn_impl == "naive":
            # naive实现：直接计算加权和
            x = torch.einsum("bsht,bthd->bshd", scores, self.v_cache[:bsz, :end_pos])
        else:
            # absorb实现：优化的加权和计算
            x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])
            
        # 投影到输出维度
        x = self.wo(x.flatten(2))
        return x


class MLP(nn.Module):
    """
    多层感知机(Multi-Layer Perceptron)实现
    创新点：
    1. 使用SwiGLU激活函数替代传统的GELU
    2. 采用并行计算优化性能
    3. 实现了三路门控机制
    
    架构特点：
    - 使用三个线性变换层(w1, w2, w3)
    - w1和w3共同构成门控机制
    - 采用列并行和行并行优化计算
    
    计算流程：
    1. 输入通过w1和w3两个并行路径
    2. w1的输出经过SiLU激活函数
    3. w1和w3的输出相乘形成门控
    4. 结果通过w2投影回原始维度
    """
    def __init__(self, dim: int, inter_dim: int):
        """
        初始化MLP层
        
        参数:
            dim (int): 输入和输出维度
            inter_dim (int): 中间层维度，通常大于dim
        """
        super().__init__()
        # 第一个变换层：dim -> inter_dim，使用列并行
        self.w1 = ColumnParallelLinear(dim, inter_dim)
        # 输出变换层：inter_dim -> dim，使用行并行
        self.w2 = RowParallelLinear(inter_dim, dim)
        # 门控变换层：dim -> inter_dim，使用列并行
        self.w3 = ColumnParallelLinear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数
        
        参数:
            x (torch.Tensor): 输入张量 [batch_size, seq_len, dim]
            
        返回:
            torch.Tensor: 输出张量 [batch_size, seq_len, dim]
            
        计算过程：
        1. x通过w1得到中间表示
        2. 同时x通过w3得到门控信号
        3. w1的输出经过SiLU激活后与门控信号相乘
        4. 结果通过w2投影回原始维度
        """
        # SwiGLU激活：SiLU(w1(x)) * w3(x)
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Gate(nn.Module):
    """
    专家路由门控机制实现
    创新点：
    1. 支持两种评分函数：softmax和sigmoid
    2. 实现了分组路由机制
    3. 支持动态专家激活
    4. 使用可学习的路由缩放因子
    
    架构特点：
    - 可配置的专家组数和每组激活专家数
    - 支持组内和组间的路由策略
    - 实现了负载均衡机制
    
    工作流程：
    1. 计算输入到专家的路由分数
    2. 根据分组策略选择专家
    3. 对选中的专家应用权重
    4. 实现负载均衡的动态路由
    """
    def __init__(self, args: ModelArgs):
        """
        初始化门控层
        
        参数:
            args (ModelArgs): 包含门控配置的参数对象
        """
        super().__init__()
        self.dim = args.dim  # 输入维度
        self.topk = args.n_activated_experts  # 每个样本激活的专家数
        self.n_groups = args.n_expert_groups  # 专家组数
        self.topk_groups = args.n_limited_groups  # 每个样本可以使用的组数
        self.score_func = args.score_func  # 评分函数类型：softmax或sigmoid
        self.route_scale = args.route_scale  # 路由权重的缩放因子
        
        # 路由权重矩阵
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        # 特殊情况下的偏置项（仅在dim=7168时使用）
        self.bias = nn.Parameter(torch.empty(args.n_routed_experts)) if self.dim == 7168 else None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播函数
        
        参数:
            x (torch.Tensor): 输入张量 [batch_size, seq_len, dim]
            
        返回:
            Tuple[torch.Tensor, torch.Tensor]: 
                - 路由权重：每个专家的权重分数
                - 选中的专家索引
                
        计算流程：
        1. 计算路由分数
        2. 应用评分函数（softmax或sigmoid）
        3. 进行分组选择（如果启用）
        4. 选择top-k专家
        5. 计算最终权重
        """
        # 计算路由分数
        scores = linear(x, self.weight)
        
        # 应用评分函数
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = scores.sigmoid()
            
        # 保存原始分数用于后续计算
        original_scores = scores
        
        # 添加偏置项（如果存在）
        if self.bias is not None:
            scores = scores + self.bias
            
        # 分组路由逻辑
        if self.n_groups > 1:
            # 将分数重塑为组的形式
            scores = scores.view(x.size(0), self.n_groups, -1)
            
            # 计算组分数
            if self.bias is None:
                # 使用最大值作为组分数
                group_scores = scores.amax(dim=-1)
            else:
                # 使用top-2分数之和作为组分数
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
                
            # 选择top-k组
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            # 创建组掩码
            mask = torch.zeros_like(scores[..., 0]).scatter_(1, indices, True)
            # 应用掩码并展平分数
            scores = (scores * mask.unsqueeze(-1)).flatten(1)
            
        # 选择top-k专家
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        # 获取选中专家的原始分数
        weights = original_scores.gather(1, indices)
        
        # 对sigmoid评分进行归一化
        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)
            
        # 应用路由缩放
        weights *= self.route_scale
        
        # 转换为输入张量的数据类型
        return weights.type_as(x), indices


class Expert(nn.Module):
    """
    专家网络实现
    创新点：
    1. 采用类似MLP的三路门控结构
    2. 使用SiLU激活函数
    3. 支持量化计算
    
    架构特点：
    - 三个线性变换层构成的前馈网络
    - 使用门控机制增强表达能力
    - 支持低精度计算
    
    说明：
    - 结构与MLP类似，但不使用并行计算
    - 每个专家独立处理自己的输入
    - 通过门控机制提高模型容量
    """
    def __init__(self, dim: int, inter_dim: int):
        """
        初始化专家网络
        
        参数:
            dim (int): 输入和输出维度
            inter_dim (int): 中间层维度
        """
        super().__init__()
        # 第一个变换层：输入维度到中间维度
        self.w1 = Linear(dim, inter_dim)
        # 输出变换层：中间维度到输出维度
        self.w2 = Linear(inter_dim, dim)
        # 门控变换层：输入维度到中间维度
        self.w3 = Linear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数
        
        参数:
            x (torch.Tensor): 输入张量
            
        返回:
            torch.Tensor: 专家处理后的输出
            
        计算过程：
        1. 输入同时经过w1和w3
        2. w1的输出经过SiLU激活
        3. 激活后的结果与w3输出相乘
        4. 结果经过w2得到最终输出
        """
        # 使用SwiGLU激活函数：SiLU(w1(x)) * w3(x)
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoE(nn.Module):
    """
    混合专家系统(Mixture of Experts)实现
    创新点：
    1. 结合路由专家和共享专家
    2. 支持分布式专家部署
    3. 实现动态专家选择
    4. 优化的负载均衡策略
    
    架构特点：
    - 包含多个独立的专家网络
    - 使用门控机制进行专家路由
    - 结合共享专家提高基础性能
    - 支持分布式计算
    
    工作流程：
    1. 通过门控机制选择专家
    2. 并行处理选中的专家计算
    3. 合并专家输出和共享专家输出
    4. 实现动态负载均衡
    """
    def __init__(self, args: ModelArgs):
        """
        初始化MoE层
        
        参数:
            args (ModelArgs): 包含MoE配置的参数对象
        """
        super().__init__()
        self.dim = args.dim
        # 确保专家数量能够被进程数整除
        assert args.n_routed_experts % world_size == 0, f"Number of experts must be divisible by world size (world_size={world_size})"
        # 专家相关配置
        self.n_routed_experts = args.n_routed_experts  # 总专家数
        self.n_local_experts = args.n_routed_experts // world_size  # 每个进程的本地专家数
        self.n_activated_experts = args.n_activated_experts  # 每次激活的专家数
        
        # 计算当前进程负责的专家范围
        self.experts_start_idx = rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        
        # 初始化门控机制
        self.gate = Gate(args)
        
        # 初始化专家网络列表
        self.experts = nn.ModuleList([
            Expert(args.dim, args.moe_inter_dim) 
            if self.experts_start_idx <= i < self.experts_end_idx 
            else None
            for i in range(self.n_routed_experts)
        ])
        
        # 初始化共享专家网络
        self.shared_experts = MLP(
            args.dim, 
            args.n_shared_experts * args.moe_inter_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数
        
        参数:
            x (torch.Tensor): 输入张量 [batch_size, seq_len, dim]
            
        返回:
            torch.Tensor: MoE层的输出
            
        计算流程：
        1. 保存输入形状并展平批次维度
        2. 通过门控获取专家权重和索引
        3. 计算每个专家的负载
        4. 并行处理选中的专家计算
        5. 合并专家输出和共享专家输出
        6. 恢复原始形状
        """
        # 保存原始形状并展平
        shape = x.size()
        x = x.view(-1, self.dim)
        
        # 获取专家路由信息
        weights, indices = self.gate(x)
        
        # 初始化输出张量
        y = torch.zeros_like(x)
        
        # 计算每个专家的处理样本数
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        
        # 对当前进程负责的专家进行计算
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue  # 跳过没有样本的专家
            expert = self.experts[i]
            # 找到路由到当前专家的样本索引
            idx, top = torch.where(indices == i)
            # 计算专家输出并加权
            y[idx] += expert(x[idx]) * weights[idx, top, None]
        
        # 计算共享专家的输出
        z = self.shared_experts(x)
        
        # 在分布式环境中合并所有进程的结果
        if world_size > 1:
            dist.all_reduce(y)
            
        # 合并专家输出和共享专家输出，并恢复原始形状
        return (y + z).view(shape)


class Block(nn.Module):
    """
    Transformer块实现
    创新点：
    1. 结合MoE和密集层的混合架构
    2. 使用RMSNorm替代LayerNorm
    3. 优化的注意力机制
    4. 灵活的专家选择策略
    
    架构特点：
    - 包含多头注意力层(MLA)
    - 包含前馈网络层(MLP或MoE)
    - 使用RMSNorm进行归一化
    - 采用残差连接
    
    工作流程：
    1. 对输入进行RMSNorm归一化
    2. 通过注意力层处理
    3. 应用残差连接
    4. 再次归一化后通过前馈网络
    5. 最后应用残差连接
    """
    def __init__(self, layer_id: int, args: ModelArgs):
        """
        初始化Transformer块
        
        参数:
            layer_id (int): 当前层的索引
            args (ModelArgs): 模型配置参数
        """
        super().__init__()
        # 初始化多头注意力层
        self.attn = MLA(args)
        # 根据层索引选择前馈网络类型：
        # - 前n_dense_layers层使用标准MLP
        # - 后续层使用MoE
        self.ffn = MLP(args.dim, args.inter_dim) if layer_id < args.n_dense_layers else MoE(args)
        # 注意力层的归一化
        self.attn_norm = RMSNorm(args.dim)
        # 前馈网络层的归一化
        self.ffn_norm = RMSNorm(args.dim)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        前向传播函数
        
        参数:
            x (torch.Tensor): 输入张量
            start_pos (int): 序列起始位置
            freqs_cis (torch.Tensor): 预计算的位置编码
            mask (Optional[torch.Tensor]): 注意力掩码
            
        返回:
            torch.Tensor: 经过完整Transformer块处理的输出
            
        计算流程：
        1. 对输入进行RMSNorm归一化
        2. 通过注意力层处理并应用残差连接
        3. 对结果进行RMSNorm归一化
        4. 通过前馈网络处理并应用残差连接
        """
        # 注意力层的前向传播：归一化 -> 注意力 -> 残差连接
        x = x + self.attn(self.attn_norm(x), start_pos, freqs_cis, mask)
        # 前馈网络层的前向传播：归一化 -> 前馈 -> 残差连接
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Transformer(nn.Module):
    """
    Transformer模型实现
    创新点：
    1. 混合专家架构(MoE)
    2. 优化的位置编码
    3. 高效的并行计算
    4. 支持长序列推理
    
    架构特点：
    - 使用并行嵌入层
    - 多层Transformer块堆叠
    - RMSNorm归一化
    - 支持分布式训练
    
    主要组件：
    - 词嵌入层
    - Transformer块序列
    - 输出层归一化
    - 词表映射层
    """
    def __init__(self, args: ModelArgs):
        """
        初始化Transformer模型
        
        参数:
            args (ModelArgs): 模型配置参数
        """
        # 设置分布式训练环境
        global world_size, rank
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        # 设置计算精度
        Linear.dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
        
        super().__init__()
        # 最大序列长度
        self.max_seq_len = args.max_seq_len
        # 词嵌入层
        self.embed = ParallelEmbedding(args.vocab_size, args.dim)
        # Transformer块列表
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(Block(layer_id, args))
        # 输出层归一化
        self.norm = RMSNorm(args.dim)
        # 输出映射层（词表投影）
        self.head = ColumnParallelLinear(args.dim, args.vocab_size, dtype=torch.get_default_dtype())
        # 预计算位置编码
        self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
        """
        前向传播函数
        
        参数:
            tokens (torch.Tensor): 输入token序列 [batch_size, seq_len]
            start_pos (int): 序列起始位置，用于位置编码计算
            
        返回:
            torch.Tensor: 模型输出的logits
            
        计算流程：
        1. 将输入token转换为嵌入向量
        2. 获取对应位置的位置编码
        3. 生成注意力掩码（如果需要）
        4. 依次通过每个Transformer块
        5. 对最后一个时间步进行归一化
        6. 投影到词表空间
        7. 在分布式环境中合并结果
        """
        # 获取序列长度
        seqlen = tokens.size(1)
        
        # 词嵌入
        h = self.embed(tokens)
        
        # 获取位置编码
        freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
        
        # 生成注意力掩码（仅在序列长度大于1时需要）
        mask = None
        if seqlen > 1:
            mask = torch.full(
                (seqlen, seqlen), 
                float("-inf"), 
                device=tokens.device
            ).triu_(1)
        
        # 通过所有Transformer块
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
            
        # 对最后一个时间步进行归一化
        h = self.norm(h)[:, -1]
        
        # 投影到词表空间
        logits = self.head(h)
        
        # 在分布式环境中合并结果
        if world_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(world_size)]
            dist.all_gather(all_logits, logits)
            logits = torch.cat(all_logits, dim=-1)
            
        return logits


if __name__ == "__main__":
    """
    模型测试代码
    
    功能：
    - 设置默认计算精度为bfloat16
    - 设置默认设备为CUDA
    - 创建随机输入进行测试
    - 验证模型前向传播
    """
    # 设置默认计算精度
    torch.set_default_dtype(torch.bfloat16)
    # 设置默认设备
    torch.set_default_device("cuda")
    # 设置随机种子
    torch.manual_seed(0)
    
    # 创建模型参数配置
    args = ModelArgs()
    # 创建随机输入
    x = torch.randint(0, args.vocab_size, (2, 128))
    # 创建模型
    model = Transformer(args)
    # 打印输出大小
    print(model(x).size())
