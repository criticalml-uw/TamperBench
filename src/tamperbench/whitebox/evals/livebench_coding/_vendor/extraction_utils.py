# Vendored from LiveBench commit 18b524d
# Source: livebench/lcb_runner/utils/extraction_utils.py
# Modifications: Removed LMStyle import/param (we always pass None)


def extract_code(model_output: str, lmstyle=None):
    outputlines = model_output.rstrip().split("\n")
    if lmstyle is None:
        indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
    else:
        indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
    if len(indexlines) < 2:
        if len(model_output) > 1 and model_output[0] == '`' and model_output[-1] == '`':
            return model_output[1:-1]
        elif len(indexlines) == 1 and indexlines[0] == len(outputlines) - 1:
            return '\n'.join(outputlines[:-1])
        return model_output.rstrip()
    return "\n".join(outputlines[indexlines[-2] + 1 : indexlines[-1]])


def extract_test_output_code(model_output: str, lmstyle=None):
    outputlines = model_output.split("\n")
    # find the last line startwith assert...
    indexlines = [i for i, line in enumerate(outputlines) if line.startswith("assert")]
    if indexlines:
        return outputlines[indexlines[-1]]
    # first try to extract ```python if not then try ```
    indexlines = [
        i
        for i, line in enumerate(outputlines)
        if "```python" in line or "```Python" in line
    ]
    if indexlines:
        start_index = indexlines[0]
    else:
        start_index = None
    indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
    if start_index is not None:
        indexlines = [i for i in indexlines if i > start_index]
        indexlines = [start_index] + indexlines

    if len(indexlines) < 2:
        return ""
    return "\n".join(outputlines[indexlines[0] + 1 : indexlines[1]])
