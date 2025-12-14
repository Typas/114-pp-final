## Nsight Systems Profiling

```bash
$ nsys profile --stats=true --output=vit_pytorch_profile python inference_vit_pytorch_nvtx.py

$ nsys stats --force-export=true --report cuda_gpu_kern_sum vit_pytorch_profile.nsys-rep

$ nsys stats --force-export=true --report nvtx_sum vit_pytorch_profile.nsys-rep
```