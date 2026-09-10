<div align="center" style="font-family: charter;">

<h1>MolReps: Taking SELFIES, Making SMILES, Teaching MLLM</h1>

<a href="https://huggingface.co/datasets/kdeng03/mol-rep-ocr-v1.1" target="_blank"><img alt="HF Dataset" src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-MolReps (OCR)-ffc107?color=ffc107&logoColor=white" height="20"></a>
<a href="https://huggingface.co/datasets/kdeng03/mol-rep-conversion-v1.1" target="_blank"><img alt="HF Dataset" src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-MolReps (Conversion)-ffc107?color=ffc107&logoColor=white" height="20"></a>

<a href="https://huggingface.co/kdeng03/MolQwen3-4B-Instruct-SFT" target="_blank"><img alt="HF Model" src="https://img.shields.io/badge/%F0%9F%A4%97%20LLM-MolQwen3--4B--Instruct-ffc107?color=ffc107&logoColor=white" height="20"></a>
<a href="https://huggingface.co/kdeng03/MolQwen3-VL-4B-Instruct-SFT" target="_blank"><img alt="HF Model" src="https://img.shields.io/badge/%F0%9F%A4%97%20MLLM-MolQwen3--VL--4B--Instruct-ffc107?color=ffc107&logoColor=white" height="20"></a>

<div>
    <a href="https://github.com/kdeng03" target="_blank">Ken Deng</a><sup>1</sup>&emsp;
</div>

<div>
    <sup>1</sup>University of Oxford&emsp;
</div>

---

The same molecule can be expressed in multiple, syntactically distinct representations, yet it remains unclear whether large language models (LLMs) can recognize and relate these equivalent encodings. We introduce **MolReps**, a pipeline that generates 9,600 question--answer pairs across ten representations for conversion, normalization, and 2D structure recognition. Post-training on MolReps improves text-only and multimodal models, **MolQwen3-4B-Instruct** and **MolQwen3-VL-4B-Instruct**, from $4.01\%$ to $75.13\%$ and from $7.11\%$ to $73.14\%$, respectively. The resulting models further achieve a relative $27.84\%$ improvement over their base model on MolRepBench, a comprehensive benchmark for evaluating molecular property prediction. These results show the effectness of targeted post-training for enhancing molecular representation understanding and its potential to improve downstream molecular tasks.

</div>