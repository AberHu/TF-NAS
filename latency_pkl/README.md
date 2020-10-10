# latency_pkl

Python pickle files for latency lookup tables.

`latency_gpu.pkl`: measured with a batch size of 32 on Titan RTX 24G GPU. We set the number of threads for OpenMP to 1 and use Pytorch1.1+cuDNN7.6.0 to measure the latency.

`latency_cpu.pkl`: measured with a batch size of 1 on Intel Xeon Gold 6130 @ 2.10GHz. We set the number of threads for MKL to 1 and use Pytorch1.1 to measure the latency.

`make_lat_lut_example.py`: An example script to make latency lookup tables. Due to the fine-grained width search, we enumerate all possible width choices, which is time consuming especially for CPUs. Note, before running the script, please set the number of threads for OpenMP and MKL to 1 and 1, respectively.