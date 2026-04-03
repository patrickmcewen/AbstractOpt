
# MoE
##  Different Tiling Schedules
| Model | Tiling Schedule | Code location |
| -------- | -------- | -------- |
| Mixtral | Dynamic Tile | (test setup) dyn_tiling/test_mixtral_sweep_revision.py::test_mixtral_b64 |
|         |             | (additional setup) dyn_tiling/test_weight_stationary_gemm_dyn_tile.py::run_ws_tile_mn_mk_dyn_tile |
|         |             | (simulator invokation) dyn_tiling/test_weight_stationary_gemm_dyn_tile.py::call_ws_tile_mn_mk_gemm_reshape_dyn_tile |
|         |             | (build STeP graph) dyn_tiling/test_weight_stationary_gemm_dyn_tile.py::ws_tile_mn_mk_gemm_reshape_dyn_tile |
| Mixtral | Static Tile | (test setup) dyn_tiling/test_mixtral_sweep_revision.py::test_mixtral_b64 |
|         |             | (additional setup) dyn_tiling/test_weight_stationary_gemm.py::run_ws_tile_mn_mk |
|         |             | (simulator invokation) dyn_tiling/test_weight_stationary_gemm.py::call_ws_tile_mn_mk_gemm_reshape |
|         |             | (build STeP graph) dyn_tiling/test_weight_stationary_gemm.py::ws_tile_mn_mk_gemm_reshape |
| Qwen | Dynamic Tile | (test setup) dyn_tiling/test_qwen_sweep_revision.py::test_qwen_b64_ablation |
|         |             | (additional setup) dyn_tiling/test_weight_stationary_gemm_dyn_tile.py::run_ws_tile_mn_mk_dyn_tile |
|         |             | (simulator invokation) dyn_tiling/test_weight_stationary_gemm_dyn_tile.py::call_ws_tile_mn_mk_gemm_reshape_dyn_tile |
|         |             | (build STeP graph) dyn_tiling/test_weight_stationary_gemm_dyn_tile.py::ws_tile_mn_mk_gemm_reshape_dyn_tile |
| Qwen | Static Tile | (test setup) dyn_tiling/test_qwen_sweep_revision.py::test_qwen_b64_ablation |
|         |             | (additional setup) dyn_tiling/test_weight_stationary_gemm_revet.py::run_ws_tile_mn_mk_revet |
|         |             | (simulator invokation) dyn_tiling/test_weight_stationary_gemm_revet.py::call_ws_tile_mn_mk_gemm_reshape_revet |
|         |             | (build STeP graph) dyn_tiling/test_weight_stationary_gemm_revet.py::ws_tile_mn_mk_gemm_reshape_revet |


**Useful Notes**
* The (test setup) phase invokes simulation for dynamic tile and sweep multiple static tile sizes.
* Here, the tiling schedule refers to the tile size used for the batch size for each experts.
* The main difference between the Qwen and Mixtral is the size for the weight dimensions, number of total experts, number of activated experts per token.
* The static tile size implementation for Qwen and Mixtral are basically the same STeP graph but uses a different logic to count the off-chip traffic, on-chip memory requirement, allocated FLOPs to count the overhead in Revet as it statically allocated resource for unselected experts too.



# Attention

| Parallelization Schedule | Code location |
| -------- | -------- |
| Static Interleaved | (test setup) dynamic_par/sweep_ae.py::test_b64_sweep |
|                    | (simulator invokation) dynamic_par/static_coarse_parallel.py::run_static_coarse_par |
|                    | (build graph) dynamic_par/static_coarse_parallel.py::build_static_coarse_par |
| Static Coarse-grained | (test setup) dynamic_par/sweep_ae.py::test_b64_sweep |
|                    | (simulator invokation) dynamic_par/static_coarse_parallel.py::run_static_par |
|                    | (build graph) dynamic_par/static_coarse_parallel.py::build_static_par |
| Dynamic | (test setup) dynamic_par/sweep_ae.py::test_b64_sweep |
|                    | (simulator invokation) dynamic_par/static_coarse_parallel.py::run_dynmic_par |
|                    | (build graph) dynamic_par/static_coarse_parallel.py::build_dynmic_par |


**Useful Notes**
* More info on the parallelization methods can be found in the "dynamic parallelization" section in the STeP paper.



# Full end-to-end
Full end-to-end is implemented in `stanford-ppl/step_tl` (https://github.com/stanford-ppl/step_tl/tree/end-to-end-layer).
There has been some updates in the STeP frontend and the simulator, so the examples would need to be modified to run in step_artifact. Recommend running it in `step_tl`'s `end-to-end-layer` branch.


| Implementation | File location |
| -------- | -------- |
| Static mem-matched  | (test setup) end_to_end/static_baseline.py::test_static_qwen_tile8 |
|                     | (graph building) end_to_end/static_baseline.py::run_static_baseline |
| Static perf-matched | (test setup) end_to_end/static_baseline.py::test_static_qwen_tile64 |
|                     | (graph building) end_to_end/static_baseline.py::run_static_baseline |
| Dynamic             | (test setup) end_to_end/dynamic_combined.py::test_dynamic_models_qwen |
|                     | (graph building) end_to_end/dynamic_combined.py::run_dynamic_layer |


**Useful Notes**
* The `run_static_baseline` calls functions that builds computations in the decoder layer (e.g. RMS Norm, QKV generation, attention, projection, residual add, MoE etc.)