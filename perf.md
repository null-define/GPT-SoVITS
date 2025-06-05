
# perf record

input:你好呀，我们是一群追逐梦想的人！

output wav time: 3s

hw: AMD Ryzen 7 7700 8-Core Processor

os: "Ubuntu 22.04.5 LTS"

env: wsl2

## original fp32

```
idx: 115
T2S S Decoder Cost is 26.489337507s
SoVITS Cost is 467.091231ms
```

### + onnxsim

```
idx: 111
T2S S Decoder Cost is 12.919324119s
SoVITS Cost is 446.198604ms
```

### + optimize

```
idx: 92
T2S S Decoder Cost is 9.601306706s
SoVITS Cost is 322.141814ms
```

## using optimize_aio.py

```
idx: 81
T2S S Decoder Cost is 4.902654217s
SoVITS Cost is 281.727026ms
```

## current issue

1. long input may cause inference time increase a lot, need to fix
2. output may have bad quality in int4 compared to original
