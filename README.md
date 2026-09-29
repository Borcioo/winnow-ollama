# Winnow-E4B decisions through Ollama

Run EldanRing's **Winnow-E4B Q8** as a typed decision model through Ollama, including on AMD GPUs. This Python adapter reads candidate-token log probabilities instead of asking a model to write confidence numbers in JSON.

**Tested:** Windows 11, Ollama 0.34.4, AMD Radeon RX 7900 XTX through Vulkan. All 43 model layers ran on GPU. Three short questions took approximately **0.30 seconds after loading** in a small local test. This is an independent integration, not an official Winnow, Ollama or TypeSafe project.

## Requirements

- Python 3.10+. Standard library only; no pip packages.
- [Ollama](https://ollama.com/download), running at `http://127.0.0.1:11434`. Tested with 0.34.4; requires `/api/generate` with `raw`, `logprobs` and `top_logprobs` support.
- Original Winnow-E4B Q8 weights: 8.01 GB. Allow approximately 17 GB free disk space for download plus Ollama import.
- A GPU supported by Ollama and enough VRAM for the selected context. The tested GPU has 24 GB. The author's 10 GB+ suggestion concerns their Q8 text workload, not every backend.

CPU inference works but is slower. These scripts do not install drivers, change persistent environment variables, or automatically start a server.

## Quick start

Clone this repository and open a terminal in its directory. Start Ollama, then run:

```sh
python scripts/install_model.py
python scripts/smoke_test.py --require-gpu
python winnow_ollama.py examples/refund.json
```

Use `python3` if that is your Python command. The installer downloads a pinned revision from the author, verifies SHA-256, and imports **`winnow-e4b`**. It does not retrain or requantize the model. Weights are not included in Git.

To reuse an existing original Q8 file (an Ollaya blob containing the same bytes also works):

```sh
python scripts/install_model.py --gguf "/path/to/Winnow-E4B-Q8_0.gguf"
```

Paths with spaces are supported. Import uses the default server on port 11434; do not point `OLLAMA_HOST` at another instance during installation. Download interruptions restart the download when rerun. An existing model of the selected name is replaced. To preserve it, choose another name:

```sh
python scripts/install_model.py --model winnow-e4b-test --gguf "/path/to/model.gguf"
python scripts/smoke_test.py --model winnow-e4b-test --require-gpu
```

### AMD on Windows

The verified backend is **Vulkan**. See [Ollama GPU documentation](https://docs.ollama.com/gpu) for current compatibility.

If your normal instance does not detect the GPU, fully quit the Ollama tray application, then try a session-scoped server in PowerShell:

```powershell
$env:OLLAMA_VULKAN = "1"
ollama serve
```

Leave that terminal open and use another terminal for the smoke test. This does not change persistent settings. Do not copy device-visibility overrides from someone else's computer. Use Ollama's documentation to select a device on multi-GPU systems.

```sh
ollama ps
```

The verified setup reports `100% GPU`. The smoke test checks `/api/ps` and fails with `--require-gpu` unless the model is fully GPU-resident. Residency does not mean every minor operation avoids the CPU. A full quit/relaunch after an Ollama update was necessary on the test machine to restore GPU discovery.

## Usage

Edit `state` and `questions` in `examples/refund.json`:

| Type | Result |
|---|---|
| `choice` | Selected option, option probabilities and confidence |
| `noul` | Probability of `true` |
| `score` | Ordered-level probabilities, expected level index and confidence |

With `winnow_ollama.py` on your Python import path:

```python
from winnow_ollama import decide

result = decide({
    "state": "Please return the duplicate payment.",
    "questions": {"refund": {
        "type": "noul",
        "instructions": "Does the customer request a refund?"
    }}
})
print(result["answers"]["refund"]["noul"])
```

The function accepts `base_url` and `model`; the CLI offers `--url` and `--model`. This adapter is specific to Winnow-E4B Q8. Substituting another model requires validation.

This is a **library and CLI**, not an HTTP `/v1/systemone` server. `ollama run winnow-e4b` opens ordinary chat and does not apply the adapter. Images are not supported.

## How it works

1. Builds the author's E4B decision prompt, including Gemma boundaries without the empty reasoning marker.
2. Requests one output-token step per question with candidate log probabilities and sampling filters disabled.
3. Renormalizes over supplied option labels using the author's Q8 temperature **1.2574172017327816**.
4. Returns the selected option, score or boolean probability. Confidence is `1 - normalized entropy`, as in the author's server. Ollaya's confidence formula differs.

For label log probabilities `l_i`, it calculates `softmax(l_i / T)` over requested labels; the vocabulary normalizing constant cancels. It selects from that distribution, not from the sampled letter or a model-written confidence number.

Ollama computes the full vocabulary output and questions run sequentially. This does not reproduce the author's selected-row output-head optimization, parallel branches, exact cache layout or bitwise numerical output. Backend and prompt batching can affect probabilities and near ties.

## Limits

- **2–10 options** per question, **1–64 questions** per request.
- The tested Ollama API returns at most **20 top token alternatives**. If a required label is absent, the adapter **fails explicitly**, even when there are few options. It never assigns missing options zero probability or silently normalizes an incomplete distribution.
- Conservative **6,000 UTF-8 byte prompt limit**, with an 8,192-token context. Longer inputs are rejected.
- Requires single-token A–J labels from this model; every requested label must appear as an exact token in the API response.
- Calibration is inherited from the author's text profile, not validated for your domain or this backend. Test thresholds on held-out data.
- No claim of prompt-injection resistance, universal language accuracy or Jev equivalence.

## Tests and measured results

Offline tests:

```sh
python -m unittest discover -s tests -v
```

Live GPU test:

```sh
python scripts/smoke_test.py --require-gpu
```

The smoke test runs two requests, validates the department and refund answers, and reports timings and GPU residency. If already loaded, its first request is not a cold-load measurement.

The initial comparison used eight synthetic tickets (seven Polish, one English), three questions each. Department and refund decisions passed **16/16 checks**. Urgency lacked independent numeric labels and was excluded from that count. Warm latency was approximately **0.30 s on AMD/Vulkan**, versus **5.75 s in Ollaya on CPU**. The maximum option-probability difference from Ollaya CPU was **0.0362**. These are small, workload-specific observations, not general benchmarks. See `examples/refund-pl.json` for a Polish example.

Reproduce the eight-ticket evaluation with `python scripts/evaluate.py`. It reads `examples/evaluation.json` and writes a local, gitignored `test-results.json`. This test requires the default `winnow-e4b` model name. The fixtures intentionally contain Polish text to exercise multilingual behavior; documentation and code are English.

## Visual consistency experiments

Run 35 synthetic scenarios, 20 repeats each (700 decisions):

```sh
python scripts/visual_tests.py --require-gpu --repeats 20
```

Open `reports/consistency.html` in a browser. This self-contained English report works offline and includes a trolley diagram, risk curves, per-variant comparisons, individual trials and JSON download. It is a saved snapshot; opening it does not run inference. Both generated HTML and raw JSON stay in the gitignored `reports/` directory.

- **Trolley:** paraphrase, option reversal, Polish state and changed casualty counts. No moral answer is labeled correct.
- **Risk:** nine probabilities, each asked with no rule and with an explicit expected-loss rule. Matched policy pairs show the effect of adding that rule; changed facts are not treated as inconsistency. Equal expected losses are unscored.
- **Intent:** refund requests, negation, paraphrases, reversed options, Polish states and one embedded instruction.

Use `--suite trolley`, `--suite risk` or `--suite intent` for a subset. `--model`, `--url` and `--output` are configurable. Re-render saved results without a running model:

```sh
python scripts/visual_tests.py --render reports/consistency.json
```

Each round shuffles scenario order using a fixed scheduling seed. Calls are stateless, with the adapter's fixed token seed and deterministic argmax decision. Repeating the same input measures stability under this setup, **not 20 independent accuracy observations**. Summaries separate failed requests, successful-trial accuracy, repeat agreement and adjacent flip rate. Probability ranges are descriptive, not confidence intervals; option probabilities are not established real-world correctness probabilities. Language variants translate the state only, leaving the question and labels in English.

The recorded run is included as `examples/consistency-results.json`. To view it without installing Ollama, run `python scripts/visual_tests.py --render examples/consistency-results.json`, then open the generated HTML.

### Observed run, 2026-09-30

On the same RX 7900 XTX / Vulkan / Ollama 0.34.4 setup: **700 successes, zero request errors**, all 35 variants had 100% repeat agreement. Warm single-decision median was **91.2 ms**, p95 **120.4 ms**, excluding initial warmup and including HTTP. GPU residency was confirmed before and after the run.

Despite perfect repeat agreement, **5 of 18 labeled variants failed**: the Polish refund request and all four expected-loss cases below 20% risk. Each failure repeated 20/20 times. The English refund baseline selected refund; its Polish counterpart selected no refund with P(refund) about 0.453. The model always preferred the certain loss of 20 units, even when the explicit rule required choosing expected losses of 5, 10, 15 or 19 units. This is an instruction-following failure on these fixtures, not evidence that risk aversion itself is incorrect.

The trolley paraphrase changed P(switch) by about **35.7 percentage points** without changing the selected action. These small synthetic experiments expose sensitivity; they do not establish general model quality, moral correctness, language accuracy or production safety. A single embedded-instruction example also cannot establish injection resistance.

New runs capture source hashes, fixture hash, Git HEAD and dirty status. The original run predated runner/source-hash capture; its JSON explicitly marks that provenance limitation. Its model responses are preserved, and added policy comparisons are calculated offline. Automated metric and escaping tests pass; browser UI verification was blocked by the execution environment's local-file URL policy.

## Model identity and licensing

- [EldanRing/Winnow-E4B](https://huggingface.co/EldanRing/Winnow-E4B)
- Revision: `734302fe5fbfeb3f21a7ece62653c9539be4aaf3`
- File: `gguf/Winnow-E4B-Q8_0.gguf`
- SHA-256: `840e3f50e5a9c218727f44e121d1b37cc9e2c3b318c8eb422ba6ef2e27b618a2`

Adapter and scripts: MIT. The model and Ollama retain their own licenses. The author's model card lists Apache-2.0; consult its license and notices for use and redistribution. No model weights or copied inference-server sources are bundled.

References: [author's server](https://github.com/EldanRing/winnow-inference), [Ollaya integration notes](https://github.com/ollaya-dev/ollaya/blob/main/docs/families/winnow.md), [Ollama API](https://docs.ollama.com/api/generate).
