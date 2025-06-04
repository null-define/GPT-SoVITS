
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

## update version to 21 with onnxsim

```
idx: 90
T2S S Decoder Cost is 8.115899307s
SoVITS Cost is 307.266305ms
```

### + optimize

```
idx: 67
T2S S Decoder Cost is 6.307872049s
SoVITS Cost is 254.294834ms
```

## quant decoder

```
idx: 87
T2S S Decoder Cost is 7.188921812s
SoVITS Cost is 265.99686ms
```

### + onnxsim(decreased)

```
idx: 94
T2S S Decoder Cost is 11.960103281s
SoVITS Cost is 305.039737ms
```

## onnxsim -> onnxslim(vits not support)  ->quant + update op version (best)

```
idx: 80
T2S S Decoder Cost is 4.978810878s
SoVITS Cost is 278.040516ms
```

## current issue

1. long input may cause inference time increase a lot, need to fix
2. output may have bad quality in int4 compared to original
