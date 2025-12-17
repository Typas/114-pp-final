## Set up
```bash
$ pip setup.py install
```

## Usage
# replace x by version number
```bash
$ python inference_vit_pytorch_nvtx_vx.py
```

## Nsight Systems Profiling
# replace x by version number
```bash
$ nsys profile -f true -o vit_profile  -t cuda,nvtx,osrt  --capture-range=nvtx  --nvtx-capture=NSYS_CAPTURE   --env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=0  --capture-range-end=stop-shutdown  python inference_vit_pytorch_nvtx_vx.py

$ nsys stats --force-export=true --report cuda_gpu_kern_sum vit_profile.nsys-rep

$ nsys stats --force-export=true --report nvtx_sum vit_profile.nsys-rep
```