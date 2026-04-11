# Test Organization

Tests are organized by where they run in CI:

 | Marker                      | Where it runs   | Use case                            |
 | --------------------------- | --------------- | ----------------------------------- |
 | (none)                      | CPU CI only     | Unit tests                          |
 | `@pytest.mark.gpu_optional` | CPU CI + GPU CI | Training integration tests          |
 | `@pytest.mark.gpu_required` | GPU CI only     | Evaluation integration tests (vLLM) |
 | `@pytest.mark.expensive`    | Manual only     | Full model runs (e.g., Llama-8B)    |

Code that uses vLLM is not run on CPU because vLLM does not support CPU (except
for if vLLM is compiled specifically for CPU).

For example,
- To run all CPU CI tests: `pytest -m "not gpu_required and not expensive"`
- To run all GPU CI tests: `pytest -m "(gpu_optional or gpu_required) and not expensive"`
