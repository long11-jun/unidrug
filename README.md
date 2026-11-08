# UniDrug

四个任务共用一套模型：Qwen2.5-3B + LoRA，分子图用 GraphMVP，分子 3D token 用 TokenMol，知识图谱用 BMKG。CMSA 补缺失的图谱向量，CMON 做模态正交分解。

下面这些权重和图谱不在本仓库里，需要自己下载，再放到 `UNIDRUG_ASSET_ROOT`。配置里的 `{asset_root}` 会换成这个路径。

```bash
export UNIDRUG_ASSET_ROOT=/path/to/assets
```

目录需要是这样：

```text
$UNIDRUG_ASSET_ROOT/
  qwen2.5-3b-instruct/
  ckpts/pretrained/                         # TokenMol
  ckpts/gnn_ckpts/graphmvp/pretraining_model.pth
  assets/kg/BMKG/
  assets/kg/BMKG-DP/
```

## 环境

Python 3.10，PyTorch 2.4.1，CUDA 12.1。

```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
```

PubMedBERT（`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract`）第一次跑文本特征时会从 Hugging Face 拉取。

## 需要单独下载的模型

### Qwen2.5-3B-Instruct

从 [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) 下到 `qwen2.5-3b-instruct/`。

```bash
huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir $UNIDRUG_ASSET_ROOT/qwen2.5-3b-instruct
```

### TokenMol

预训练权重在 [Zenodo](https://zenodo.org/records/13428828)，代码和词表说明在 [Token-Mol](https://github.com/jkwang93/Token-Mol)。把 `config.json` 和 `pytorch_model.bin` 放到 `ckpts/pretrained/`。本仓库自带词表 `torsion_version/torsion_voc_property.csv`。

3D token 可以训练前先算好：

```bash
python scripts/precompute_tokenmol_3d.py --dataset SIDER
python precompute_tokenmol_dti.py --dataset yamanishi08
```

### GraphMVP

权重来自 [GraphMVP](https://github.com/chao1224/GraphMVP) 的预训练 GIN，Hugging Face 上的文件是 [model.pth](https://huggingface.co/chao1224/MoleculeSTM/blob/main/pretrained_GraphMVP/GraphMVP_G/model.pth)。下载后改名为 `pretraining_model.pth`，放到 `ckpts/gnn_ckpts/graphmvp/`。

### BMKG

图谱用 [KEDD](https://github.com/icycookies/KEDD_temporal) 发布的 [BMKG](https://drive.google.com/drive/folders/1U2M3383-3dDAyLTAcXGcUagAEjlB6QgN?usp=sharing)。解压后放到 `assets/kg/BMKG/`，里面要有 `drug.json`、`protein.json`、`embedding.pkl`、`links.csv`。

分子性质用的 `assets/kg/BMKG-DP/` 是同一套文件。把 BMKG 拷一份到这个目录即可。`prone_embedding.pkl` 没有也行，训练时缺 `embedding.pkl` 才会用 ProNE 现算。

## DTI

Yamanishi08 和 BMKG-DTI 在 `unidrug/datasets/dti/`。原始数据见 [KEDD 的 DTI 数据](https://drive.google.com/drive/folders/1AaUWLlOOua5BH7Q-bBVUBgOugDfWF3ip?usp=sharing)，两个数据集都放在 `unidrug/datasets/dti/` 下。

5 折，指标是 AUROC 和 AUPR。划分：`warm`、`cold_drug`、`cold_protein`、`cold_cluster`。第四个参数 `false` 是 interaction-cold，`true` 是 knowledge-cold（去掉留出实体的图谱三元组）。

```bash
bash run_dti.sh yamanishi08 warm 0
bash run_dti.sh yamanishi08 cold_drug 0 false
bash run_dti.sh yamanishi08 cold_drug 0 true
bash run_dti.sh bmkg-dti cold_cluster all false
```

## DP

SIDER、ClinTox、HIV、BACE 在 `unidrug/datasets/dp/moleculenet/`。MoleculeNet 原始包：

```bash
wget http://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip
unzip chem_dataset.zip
mkdir -p unidrug/datasets/dp
mv dataset unidrug/datasets/dp/moleculenet
```

解压后删掉各数据集里的 `processed/`。scaffold 划分，指标 AUROC。

```bash
bash run_dp.sh SIDER
bash run_dp.sh ClinTox
bash run_dp.sh HIV
bash run_dp.sh Bace
```

## DDI

论文里的实验是 Luo 的二分类，数据在 `unidrug/datasets/ddi/MSSL2drug/`。原始边表见 [DDInet.txt](https://github.com/pengsl-lab/MSSL/blob/main/data/DownStreamdata/DDInet.txt)。指标 AUROC 和 AUPR。

```bash
bash run_ddi.sh LUO
```

## PPI

SHS27k 和 SHS148k 在 `unidrug/datasets/ppi/`。原始数据见 [GNN_PPI](https://github.com/lvguofeng/GNN_PPI/tree/main/data)。7 类，划分用 `BFS` 或 `DFS`，指标 Micro-F1。

```bash
bash run_ppi.sh SHS27k DFS single
bash run_ppi.sh SHS148k BFS single
```

## 消融

对应论文里 DDI、PPI、DP 的组件消融。A 到 G 依次是：去掉 CMSA 和 CMON、只留 CMSA、只留 CMON、去掉 2-hop、去掉 digest、去掉 LODC、完整模型。

```bash
bash run_ablation.sh dp SIDER
bash run_ablation.sh ddi
bash run_ablation.sh ppi SHS27k DFS
```

日志和 checkpoint 写到 `ckpts/`。没设 `CUDA_VISIBLE_DEVICES` 时用 0 号卡。
