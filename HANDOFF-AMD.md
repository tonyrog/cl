# Handoff: AMD ICD + low-latency events for erlang/cl

Target machine: terror.local (AMD Ryzen AI Max+ 395, Strix Halo, gfx1151, unified memory).
Goal: cut Erlang -> GPU -> Erlang round-trip latency for small compute jobs, ideally ~1000x.

## Working rules
- Commit locally only, never push. Tony pulls from terror and pushes himself.
- Measure before building: break down latency enqueue -> doorbell -> GPU done -> thread woken -> Erlang process woken.

## Bugs found in c_src/cl_nif.c (not yet fixed)
1. ecl_pre_load (~line 6955): `enif_alloc(num_platforms*sizeof(ecl_platform_t*))`
   allocates pointer size but ecl_platform_t is a 24-byte struct -> heap overflow.
   Should be sizeof(ecl_platform_t).
2. ecl_context_main, ECL_MESSAGE_FINISH (~line 3373): calls clFlush instead of clFinish.
3. ecl_load_dynfunctions (~line 7613): dlopen("libOpenCL.so") needs the dev symlink;
   try "libOpenCL.so.1" first.
4. ecl_make_context (~line 3017): context thread created with suggested_stack_size 8
   (kilowords = 64 KB); it calls clWaitForEvents deep into ROCclr. Raise to >= 256 KB.
5. One context thread per context, and clWaitForEvents blocks it -> head-of-line
   blocking for all waits/flushes on that context.
6. SVM functions are in the ECL_FUNC table but have no NIFs (svm_alloc/free,
   set_kernel_arg_svm_pointer, fine-grain flags) - needed for fast paths.

## Current wait path (per kernel)
enqueue_nd_range_kernel (NIF) -> async_wait_for_event -> mutex/cond queue ->
context thread -> clWaitForEvents (ROCclr, may sleep on interrupt) -> clGetEventInfo
-> enif_send -> scheduler wakes process -> receive.
Tony suspects ms-level latency. futex wake itself is ~5-50 us; ms more likely from
ROCclr sleep/usleep loops, timer slack, CPU C-states, GPU clock/power ramp from idle.

## Plan
- Step 0: check rocminfo/clinfo, build, fix bugs above, add round-trip benchmark
  (empty kernel, percentiles; see test/cl_noop.erl). Try ROCclr env knobs
  (direct dispatch / active wait - verify exact names against installed ROCm).
- Step 1: within OpenCL: bounded spin-poll on clGetEventInfo in the NIF, then fall
  back to async; replace blocking context thread with clSetEventCallback -> enif_send;
  add SVM NIFs.
- Step 2: HSA direct dispatch ("doorbells"): get code object via
  clGetProgramInfo(CL_PROGRAM_BINARIES), hsa_executable_load_agent_code_object,
  own AQL queue (hsa_queue_create), write dispatch packet, store write index,
  hsa_signal_store_relaxed(queue->doorbell_signal, idx), spin on completion signal
  in userspace. Hard part: kernarg layout incl. hidden args -> parse amdhsa.kernels
  metadata in the code object.
- Step 3: persistent kernel polling a command ring in fine-grain SVM; NIF spins on a
  done flag. Only realistic path toward ~1000x. Costs: a CU busy, power, preemption.

## Step 0 results (2026-10-25)
- Stack: Ubuntu rocm-opencl-icd 7.1 (/usr/lib/x86_64-linux-gnu/libamdocl64.so,
  driver 3581.0, OpenCL 2.1 platform / 2.0 device, fine-grain buffer SVM: yes).
  A second ROCm 10.1 is installed under /opt/rocm (core-10.1/lib/opencl) but
  the ICD loader picks the system one.
- Build: `make -C c_src aimax` (fixed: target used to override CFLAGS and
  lose the erts include dir).
- Bugs 1-4 fixed (platform alloc size, FINISH->clFinish, dlopen
  libOpenCL.so.1 first, context thread stack 512 KB). 5-6 remain (step 1).
- Benchmark: test/cl_rtt.erl, `cl_rtt:run(N, [wait,finish,poll], GapMs)`.
  ROCclr stamps queued/submit with the host clock and start/end with the GPU
  clock (offset ~291 s) -> only intra-domain differences are meaningful.
- Hot loop (gap 0), p50 total: wait ~15-19 us, finish ~15-18 us,
  poll (flush + spin on get_event_info) ~9-11 us. GPU exec of empty
  kernel 0.6 us, enqueue NIF ~1 us. p99 ~20-60 us.
- ROCclr knobs (AMD_DIRECT_DISPATCH, ROC_ACTIVE_WAIT_TIMEOUT,
  ROC_CPU_WAIT_FOR_SIGNAL, ROC_SYSTEM_SCOPE_SIGNAL): all within run-to-run
  noise, +-few us. No ms to win there.
- Idle gap between jobs is where ms appear (p50 total):
  gap 1 ms ~200-240 us, 10 ms ~300 us, 100 ms ~2.5-2.9 ms.
  Everything slows: s2e 0.6 -> 17 us (GPU sclk 600 MHz idle of 2900,
  power_dpm_force_performance_level=auto), enq 1 -> 85 us (CPU governor
  powersave / C-states), rest ~2.4 ms (GPU wake-up).
- Conclusion: the ~1000x Tony saw is idle-vs-hot, not the wait path.
  Next things to try: performance_level=high / cpu governor performance
  (needs root), or a keep-warm strategy; then step 1 spin-poll in the NIF
  (poll already beats the context thread by ~5-8 us when hot).
