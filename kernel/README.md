Using our defined kernel function (just for reference)

## build our kernel 
```bash
$ pip install .
```

## usage (run with our defined kernel)

```bash
python ./vit_with_custom_cuda.py
```

## Profiling
```bash
$ nsys profile --stats=true --output=vit_pytorch_profile python ./vit_with_custom_cuda.py
```