#  Transformer DSM Attention - : A Pytorch Implementation

This is a PyTorch implementation of the Transformer DSM model


A novel sequence to sequence framework utilizes the **self-attention mechanism**, instead of Convolution operation or Recurrent structure, and achieve the state-of-the-art performance on **WMT 2014 English-to-German translation task**. (2017/06/12)

> To learn more about self-attention mechanism, you could read "[A Structured Self-attentive Sentence Embedding](https://arxiv.org/abs/1703.03130)".

> We used two models in the same source code (the basic model with DSM) with a parameter BSM = False which indicates the basic model, BSM = True indicates  the DSM model.

The project support training and translation with trained model now.
<p align="center">
<img src="https://github.com/belainine/TransformerDSM/blob/main/TransformerFR.jpg" width="600">

</p>

# Requirement
- python 3.4+
- pytorch 1.3.1
- torchtext 0.4.0
- spacy 2.2.2+
- tqdm
- dill
- numpy
- nltk
- jiwer
- tensorboardX
- matplotlib
- gensim

# Usage

### 0) Download the spacy language model.
```bash
# conda install -c conda-forge spacy 
python -m spacy download en
python -m spacy download de
python -m spacy download fr
```
# DataSet 
>  We used the machine translation tasks WMT 2014 English-German and English-French.
The 2014 English-German (http://www.statmt sets.org/wmt14) dataset includes approximately 4.5 million sentence pairs for source vocabularies and target share nearly 37,000 words.

> WMT 2014 English-French (http://www.statmt sets.org/wmt14) dataset consisting of approximately 40 million sentence pairs for source and target vocabularies share nearly 32,000 words. We used the Moses Parser to split the different sentences (https://www.nltk.org/_modules/nltk/tokenize/moses.html). Input and output sentences used a maximum sequence length of 120.

> Finally, each of the development (http://www.statmt.org/wmt14/dev.tgz) and test (http://www.statmtsets.org/wmt14/test-filtered.tgz) of the two translation tasks included 3,003 sentences.
### 1) Preprocess the data with torchtext and spacy.
```bash
python preprocess.py -lang_src de -lang_trg en -share_vocab -save_data m30k_deen_shr.pkl
```

### 2) Train the model
```bash
python train.py -data_pkl m30k_deen_shr.pkl -log m30k_deen_shr -embs_share_weight -proj_share_weight -label_smoothing -save_model trained -b 256 -warmup 128000 -epoch 400
```

### 3) Test the model
```bash
python translate.py -data_pkl m30k_deen_shr.pkl -model trained.chkpt -output prediction.txt
```

### 1) Download and preprocess the data with bpe:

> Since the interfaces is not unified, you need to switch the main function call from `main_wo_bpe` to `main`.

```bash
python preprocess.py -raw_dir /tmp/raw_deen -data_dir ./bpe_deen -save_data bpe_vocab.pkl -codes codes.txt -prefix deen
```

### 2) Train the model
```bash
python train.py -data_pkl ./bpe_deen/bpe_vocab.pkl -train_path ./bpe_deen/deen-train -val_path ./bpe_deen/deen-val -log deen_bpe -embs_share_weight -proj_share_weight -label_smoothing -save_model trained -b 256 -warmup 128000 -epoch 400
```

### 3) Test the model
```
python translate.py -data_pkl m30k_deen_shr.pkl -model trained.chkpt -output prediction.txt
```
# Performance
## Training


- Parameter settings:
  - default parameter and optimizer settings
  - label smoothing 
  - target embedding / pre-softmax linear layer weight sharing. 

- Elapse per epoch (on NVIDIA GTX):
  - Training set: 7 days
  - Validation set: 2.011 minutes
  # Eval
  - Evaluation on the generated text.
  ```bash
  python TransformerMDA/translate.py corpus_name
   ```
  - Attention weight plot.
  ```bash
   python showAttention.py corpus_name
   ```
---
# Acknowledgement
- The byte pair encoding parts are borrowed from [subword-nmt](https://github.com/rsennrich/subword-nmt/).
- The project structure, some scripts and the dataset preprocessing steps are heavily borrowed from (https://github.com/hyunwoongko/transformer).
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
