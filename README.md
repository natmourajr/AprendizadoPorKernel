# AprendizadoPorKernel
Repositório criado para a Disciplina de Aprendizado por Kernel da COPPE

## Antes de Rodar
### Clone este repositório
Depois de clonar, crie um diretório chamado **data** na pasta raiz do repositório, e na pasta data crie os diretórios **indexes**, **models** e **plots**.
### Baixe a imagem docker que foi criada para este repositório
```
$ docker pull natmourajr/kernel:lastest
```
### Execute a imagem
Fique atento pois os volumes devem estar montados corretamente. Ai você vai abrir um browser e pode rodar os notebooks
```
$ docker run -it --rm -v <Volume na Máquina Pessoal>:/tf -p 8888:8888 natmourajr/kernel:lastest
```
## Execução
Cada notebook deve ser rodado na ordem especificada