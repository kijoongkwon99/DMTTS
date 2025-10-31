# Evaluation

---

## Generating Samples for Evaluation
All processes run via **CLI**.  
An example prompt (“police”) is provided.  
The evaluation samples are composed of **six languages**.

---

## Batch Inference for Test Set
A `.lst` file is required.  
Use `eval_infer_batch.py` to generate inference results.  
The synthesized outputs are saved under the `synthesized_speech/` directory.

---

## Batch Evaluation on Generated Results
Metrics: **CER**, **WER**, and **UTMOS**.  
Use `python eval_metric_batch.py` to evaluate generated results.
The evaluation results are stored under the `result/` directory.

In addition to loading the metalist required for evaluation,
details of each metric can be found in `dmtts/utils/eval_utils.py.`