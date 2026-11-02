# Qwen Agent Functional Checks

The local Qwen LoRA baseline was exercised with training-set examples for both FHR and DUS inputs. The checks covered:

- English requests routing to reconstruction, segmentation, risk assessment, or complete reporting.
- Task1, Task2, and Task3 tool execution and reuse of completed results.
- Task3 checkpoint selection for Task3 dataset FHR and Task1-reconstructed FHR.
- Structured report fields, event intervals, and trace output.

The integration script is `tests/qwen_integration.py`. It uses the released training manifests and writes generated artifacts under the ignored `output/` directory.
