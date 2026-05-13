请根据任务列表和单个任务描述，执行任务。跳过已实现的任务，

对于任务列表中，相同 call function 调用多个 kernel 的。可以合并处理。
查找 tests/ 下有无对应用例，有则参考

## 任务并行执行
- 1，2，3，4，5，6，7 可以在任务间并行执行;
<!-- - 8，9需要在其余任务执行完，逐个执行 -->

## 任务列表
| 文件目录 | kernel name | call function |
|--------|--------|--------|
| `fla/modules/fused_norm_gate.py` | layer_norm_gated_bwd_kernel | `layer_norm_gated_bwd` |
| `fla/modules/fused_norm_gate.py` | layer_norm_gated_bwd_kernel1 | `layer_norm_gated_bwd` |
| `fla/modules/fused_linear_cross_entropy.py` | logsumexp_fwd_kernel | `logsumexp_fwd` |
| `fla/modules/fused_linear_cross_entropy.py` | cross_entropy_kernel | `fused_linear_cross_entropy_forward` |
| `fla/modules/fused_linear_cross_entropy.py` | elementwise_mul_kernel | `fused_linear_cross_entropy_backward` |
| `fla/modules/layernorm.py` | layer_norm_fwd_kernel | `layer_norm_fwd` |
| `fla/modules/layernorm.py` | layer_norm_fwd_kernel1 | `layer_norm_fwd` |
| `fla/modules/layernorm.py` | layer_norm_bwd_kernel | `layer_norm_bwd` |
| `fla/modules/layernorm.py` | layer_norm_bwd_kernel1 | `layer_norm_bwd` |
| `fla/modules/activations.py` | swish_fwd_kernel | `swish_fwd` |
| `fla/modules/activations.py` | swish_bwd_kernel | `swish_bwd` |
| `fla/modules/activations.py` | swiglu_fwd_kernel | `swiglu_fwd` |
| `fla/modules/activations.py` | swiglu_fwdbwd_kernel | `swiglu_fwdbwd` |
| `fla/modules/fused_bitlinear.py` | layer_norm_fwd_kernel_quant | `layer_norm_fwd_kernel_quant` |
| `fla/modules/fused_cross_entropy.py` | cross_entropy_fwd_kernel | `fused_cross_entropy_forward` |
| `fla/modules/fused_cross_entropy.py` | cross_entropy_bwd_kernel | `CrossEntropyLossFunction.backward()` |
| `fla/modules/conv/triton/kernels.py` | causal_conv1d_fwd_kernel | `layer_norm_bwd` |
| `fla/modules/conv/triton/kernels.py` | causal_conv1d_bwd_kernel | `causal_conv1d_fwd`(in file `fla/modules/conv/triton/ops.py`) |
| `fla/ops/simple_gla/parallel.py` | parallel_simple_gla_fwd_kernel | `parallel_simple_gla_fwd` |
| `fla/ops/simple_gla/parallel.py` | parallel_simple_gla_bwd_kernel | `parallel_simple_gla_bwd` |
| `fla/ops/abc/chunk.py` | chunk_abc_fwd_kernel_h | `ChunkABCFunction.forward` |
| `fla/ops/abc/chunk.py` | chunk_abc_fwd_kernel_intra_K | `ChunkABCFunction.forward` |
| `fla/ops/abc/chunk.py` | chunk_abc_fwd_kernel_K | `ChunkABCFunction.forward` |
| `fla/ops/abc/chunk.py` | chunk_abc_fwd_kernel_intra_V | `ChunkABCFunction.forward` |
| `fla/ops/abc/chunk.py` | chunk_abc_fwd_kernel_V | `ChunkABCFunction.forward` |
| `fla/ops/abc/chunk.py` | chunk_abc_bwd_kernel_dh | `ChunkABCFunction.backward` |
| `fla/ops/abc/chunk.py` | chunk_abc_bwd_kernel_V | `ChunkABCFunction.backward` |
| `fla/ops/abc/chunk.py` | chunk_abc_bwd_kernel_intra_V | `ChunkABCFunction.backward` |
| `fla/ops/abc/chunk.py` | chunk_abc_bwd_kernel_intra_K | `ChunkABCFunction.backward` |
| `fla/ops/abc/chunk.py` | chunk_abc_bwd_kernel_K | `ChunkABCFunction.backward` |
| `fla/ops/abc/chunk.py` | chunk_abc_bwd_kernel_intra_KV | `ChunkABCFunction.backward` |
| `fla/ops/abc/chunk.py` | chunk_abc_bwd_kernel_rcum_inter | `ChunkABCFunction.backward` |
| `fla/ops/abc/chunk.py` | chunk_abc_bwd_kernel_rcum_intra | `ChunkABCFunction.backward` |
| `fla/ops/mesa_net/chunk_cg_solver_fwd.py` | chunk_fwd_mesa_cg_dim64_kernel | `chunk_mesa_cg_fwd` |
| `fla/ops/mesa_net/chunk_h_kv_intra_bwd.py` | chunk_mesa_net_h_kv_bwd_intra_kernel | `chunk_mesa_net_h_kv_bwd_intra_fn` |
| `fla/ops/mesa_net/chunk_h_kk_intra_bwd.py` | chunk_mesa_net_h_kk_bwd_intra_kernel | `chunk_mesa_net_h_kk_bwd_intra_fn` |
| `fla/ops/mesa_net/chunk_h_kv_intra_bwd_separate.py` | chunk_mesa_net_h_kv_bwd_intra_kernel_dkv | `chunk_mesa_net_h_kv_bwd_intra_separate_fn` |
| `fla/ops/mesa_net/chunk_h_kv_intra_bwd_separate.py` | chunk_mesa_net_h_kv_bwd_intra_kernel_dq | `chunk_mesa_net_h_kv_bwd_intra_separate_fn` |
| `fla/ops/mesa_net/chunk_h_fwd.py` | chunk_mesa_net_fwd_kernel_h | `chunk_mesa_fwd_h` |
| `fla/ops/mesa_net/decoding_one_step.py` | mesa_net_decoding_one_step_kernel | `mesa_net_decoding_one_step` |
| `fla/ops/rwkv6/fused_recurrent.py` | fused_recurrent_rwkv6_fwd_kernel | `fused_recurrent_rwkv6_fwd` |
| `fla/ops/rwkv6/fused_recurrent.py` | fused_recurrent_rwkv6_bwd_kernel_dq | `fused_recurrent_rwkv6_bwd` |
| `fla/ops/rwkv6/fused_recurrent.py` | fused_recurrent_rwkv6_bwd_kernel_dkv | `fused_recurrent_rwkv6_bwd` |
| `fla/ops/rwkv6/fused_recurrent.py` | fused_recurrent_rwkv6_bwd_kernel_dw | `fused_recurrent_rwkv6_bwd` |
| `fla/ops/rwkv6/chunk.py` | chunk_rwkv6_fwd_cumsum_kernel | `chunk_rwkv6_fwd_cumsum` |
| `fla/ops/rwkv6/chunk.py` | chunk_rwkv6_fwd_A_kernel_intra_sub_inter | `chunk_rwkv6_fwd_intra` |
| `fla/ops/rwkv6/chunk.py` | chunk_rwkv6_fwd_A_kernel_intra_sub_intra | `chunk_rwkv6_fwd_intra` |
| `fla/ops/rwkv6/chunk.py` | chunk_rwkv6_fwd_A_kernel_intra_sub_intra_split | `chunk_rwkv6_fwd_intra` |
| `fla/ops/rwkv6/chunk.py` | chunk_rwkv6_fwd_A_kernel_intra_sub_intra_merge | `chunk_rwkv6_fwd_intra` |
| `fla/ops/rwkv6/chunk.py` | chunk_rwkv6_bwd_kernel_dh | `chunk_rwkv6_bwd_dh` |
| `fla/ops/rwkv6/chunk.py` | chunk_rwkv6_bwd_kernel_intra | `chunk_rwkv6_bwd_dqk_intra` |
| `fla/ops/rwkv6/chunk.py` | chunk_rwkv6_bwd_kernel_inter | `chunk_rwkv6_bwd_dqkgu` |
| `fla/ops/gsa/fused_recurrent.py` | fused_recurrent_gsa_inference_kernel | `fused_recurrent_gsa_inference` |
| `fla/ops/gsa/chunk.py` | chunk_gsa_fwd_k_kernel_inter | `chunk_gsa_fwd_k` |
| `fla/ops/gsa/chunk.py` | chunk_gsa_fwd_k_kernel_intra | `chunk_gsa_fwd_k` |
| `fla/ops/gsa/chunk.py` | chunk_gsa_bwd_k_kernel_dA | `chunk_gsa_bwd_k` |
| `fla/ops/gsa/chunk.py` | chunk_gsa_bwd_k_kernel_intra_dvg | `chunk_gsa_bwd_k` |
| `fla/ops/based/parallel.py` | parallel_based_fwd_kernel | `ParallelBasedFunction.forward` |
| `fla/ops/based/parallel.py` | parallel_based_bwd_kernel | `ParallelBasedFunction.backward` |
| `fla/ops/based/fused_chunk.py` | fused_chunk_based_fwd_kernel | `FusedChunkBasedFunction.forward` |
| `fla/ops/based/fused_chunk.py` | fused_chunk_based_bwd_kernel | `FusedChunkBasedFunction.backward` |

## 单个任务描述
1. 将 {文件目录} 中 {kernel name} 和调用它的 {call function} 提取出来，生成对应的测试用例。要求仅满足最小化跑通 kernel，及必要的依赖。
2. 查找 tests/ 目录下，直接或间接调用 {kernel name} 或者 {call function} 的用例，输出用例文件和用例名
3. 判断算子类型，如果包含 `tl.dot()` ，算子类型为 CV；否则为 VV
4. 将测试用例输出到 /home/scy/flash-linear-attention/fla_op_tests 目录下，新建一个以 {kernel name} 命名的文件夹，来存储对应用例。
5. 请把原算子定义拷贝一份到测试用例中，**不要import原文件中的算子**，**严格保证抽取的kernel和原始kernel是一致的**。
6. 测试用例仅构造**一组config信息**，尽量保证输入**shape较大**，方便后续测试算子性能。
7. 同时需在算子测试用例中添加profiling的测试。profiling测试可复用 `/home/scy/flash-linear-attention/fla_op_tests/profiler_utils.py`
<!-- 8. 生成完测试用例后，请直接执行一次测试用例，**逐个执行，不允许并行**，
    - 如果测试用例不可运行，可以尝试调整算子输入，但是遇到kernel报错，请返回错误信息（包括但不限于编译失败、运行失败），**禁止**修改任何原始算子实现。
    - 如果测试用例正确执行，则进行profiling，输出出来对应的 proiling 文件到 `/home/scy/flash-linear-attention/fla_op_test/{kernel name}/profiling` 目录下，
9. 如果profiling正常，在proiling文件中找到 op_summary*.csv或者op_statistic.csv文件，统计对应kernel的执行时间 -->

执行前导入环境变量
```
source /usr/local/Ascend/cann/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export TRITON_ALWAYS_COMPILE=1
export TRITON_DISABLE_FFTS=1
export TRITON_DEBUG=1
export ASCEND_RT_VISIBLE_DEVICES=1
```
