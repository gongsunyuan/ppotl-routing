#!/usr/bin/env bash
# Phase-0 §2.6 测量驱动:启动开销 → cuda 钉卡扫描 → 4-GPU 轮转扩散 → cpu 扫描 → 线程配比
# 顺序执行(测量互不污染);日志 → profiling/battery.log
set -e
cd /mnt/data/gsy/ppotl-routing
PY=.venv/bin/python

echo "### startup probe (cpu dry-run subprocess)"
$PY profiling/pool_bench.py --device cpu --startup-probe --threads 1

echo "### cuda concurrency sweep, all workers pinned GPU1, threads=1"
$PY profiling/pool_bench.py --device cuda --levels 1,2,4,8,16 --threads 1 --tag conc_cuda1 --cuda-gpus 1

echo "### cuda spread across GPUs 1,2,3 (round-robin), threads=1"
$PY profiling/pool_bench.py --device cuda --levels 4,8,16 --threads 1 --tag conc_spread --cuda-gpus 1,2,3

echo "### cpu concurrency sweep, threads=1"
$PY profiling/pool_bench.py --device cpu --levels 1,2,4,8,16 --threads 1 --tag conc_cpu

echo "### thread allocation at N=8 cuda: 1 vs 2 vs default"
$PY profiling/pool_bench.py --device cuda --levels 8 --threads 2 --tag thr_cuda2 --cuda-gpus 1
$PY profiling/pool_bench.py --device cuda --levels 8 --threads default --tag thr_cudadef --cuda-gpus 1

echo "### thread allocation at N=8 cpu: 1 vs 2 vs default"
$PY profiling/pool_bench.py --device cpu --levels 8 --threads 2 --tag thr_cpu2
$PY profiling/pool_bench.py --device cpu --levels 8 --threads default --tag thr_cpudef

echo BATTERY_DONE
