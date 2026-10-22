now i find error after running setup for qwen3, in rtx 5080, help me to fix it : 2026-10-22 00:34:33,068 INFO rag_benchmark.techniques.qwen3.runner: qwen3_embedding generation 332/340: its-qa-304673b145fd34f2a4694c74 (ok)
2026-10-22 00:34:41,475 INFO rag_benchmark.techniques.qwen3.runner: qwen3_embedding generation 333/340: its-qa-44097ea6bdf7deb65b8dceaf (ok)

Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/home//Documents/DiLLeMa-RAG-Benchmark/src/rag_benchmark/__main__.py", line 5, in <module>
    main()
  File "/home//Documents/DiLLeMa-RAG-Benchmark/src/rag_benchmark/cli.py", line 164, in main
    result = run_benchmark(args.config, args.stage, args.output)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home//Documents/DiLLeMa-RAG-Benchmark/src/rag_benchmark/benchmark.py", line 294, in run_benchmark
    reports.update(evaluate_qwen3(spec, bundle, output))
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home//Documents/DiLLeMa-RAG-Benchmark/src/rag_benchmark/techniques/qwen3/runner.py", line 235, in run_evaluate
    record = evaluate_questions(
             ^^^^^^^^^^^^^^^^^^^
  File "/home//Documents/DiLLeMa-RAG-Benchmark/src/rag_benchmark/shared/evaluation.py", line 55, in evaluate_questions
    result = runner.answer(collection_name(dataset, question.language), question.question)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home//Documents/DiLLeMa-RAG-Benchmark/src/rag_benchmark/techniques/qwen3/pipeline.py", line 45, in answer
    prediction = self.runtime.generate_prompt(base_prompt(question, context)) if context else ""
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home//Documents/DiLLeMa-RAG-Benchmark/src/rag_benchmark/shared/local_llm.py", line 107, in generate_prompt
    outputs = self.model.generate(
              ^^^^^^^^^^^^^^^^^^^^
  File "/home//Documents/DiLLeMa-RAG-Benchmark/.venv/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 124, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home//Documents/DiLLeMa-RAG-Benchmark/.venv/lib/python3.12/site-packages/transformers/generation/utils.py", line 2566, in generate
    result = decoding_method(
             ^^^^^^^^^^^^^^^^
  File "/home//Documents/DiLLeMa-RAG-Benchmark/.venv/lib/python3.12/site-packages/transformers/generation/utils.py", line 2781, in _sample
    while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home//Documents/DiLLeMa-RAG-Benchmark/.venv/lib/python3.12/site-packages/transformers/generation/utils.py", line 2599, in _has_unfinished_sequences
    elif this_peer_finished:
         ^^^^^^^^^^^^^^^^^^
torch.AcceleratorError: CUDA error: unspecified launch failure
Search for `cudaErrorLaunchFailure' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1