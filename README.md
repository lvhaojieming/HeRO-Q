# HeRO-Q
ICML-HeRO-Q

To ensure reproducibility and rigorous performance evaluation, all experiments presented in the paper were conducted in a high-performance computing environment.

### Hardware Environment
* **GPU:** The main experiments, including Perplexity (PPL) evaluation and Downstream Task benchmarks, were performed on a server equipped with **8 $\times$ NVIDIA A800 GPUs**.
* **Latency Benchmarking:** As detailed in the paper (Appendix A.5), inference latency and throughput measurements were conducted on a single NVIDIA A800 GPU to assess the real-world efficiency of the CALM framework compared to uniform quantization baselines like GPTQ.

### Software Environment
* **CUDA Version:** [Insert your CUDA version, 12.8]
* **PyTorch Version:** [Insert your PyTorch version, 2.1.2]
* **Dependencies:** Please refer to `environment.yml` for the complete list of libraries.
