# PEL202 - FUNDAMENTOS DA INTELIGÊNCIA ARTIFICIAL

Análise do Comportamento de Modelos Generativos Profundos Utilizando Conjunto de Dados com Pouca Amostragem

## Dependências
+ opencv
+ matplotlib
+ pytorch
+ kornia
+ torchsummary
+ torch-fidelity
+ numpy
+ pillow
+ easydict
+ pyyaml
+ cython

Comandos para compilar a Retina Face
```
$ cd repository/util/RFace
$ make
```


## Como usar
**Treinamento**
```shell
$ python train.py --cfg cfg/unifesp.yaml # modelo 128x128
$ python train.py --cfg cfg/unifesp_v2.yaml --resolution 512 # modelo 512x512
```
Para utilizar a operação bicubic e a Retina Face habilitar os comandos `--bicubic` e `--retinaface`, respectivamente.

**Teste**
```shell
$ python eval.py --cfg cfg/unifesp.yaml --z_dim 100 --ckpt /Path/to/ckpt/128x128 # modelo 128x128
$ python train.py --cfg cfg/unifesp_v2.yaml --resolution 512 --z_dim 100 --ckpt /Path/to/ckpt/512x512 # modelo 512x512
```
Os resultados estarão no diretório `./results`.