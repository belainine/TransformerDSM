''' Handling the data io '''
# coding: utf8
import os
import argparse
import logging
import dill as pickle
import urllib
from tqdm import tqdm
import sys
import codecs
import spacy
import torch
import tarfile
import torchtext.data
import torchtext.datasets
from transformerDSM.translation import TranslationDataset
import transformerDSM.Constants as Constants
from learn_bpe import learn_bpe
from apply_bpe import BPE
spacy.load('fr')

__author__ = "author: belainine"


_TRAIN_DATA_SOURCES = [
    #{"url": "http://data.statmt.org/wmt17/translation-task/training-parallel-nc-v12.tgz",
    # "trg": "news-commentary-v12.de-en.de",
    # "src": "news-commentary-v12.de-en.en"}
    #{"url": "http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz",
    # "trg": "commoncrawl.de-en.en",
    # "src": "commoncrawl.de-en.de"},
    {"url": "http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz",
     "src": "europarl-v7.de-en.en","trg": "europarl-v7.de-en.de",
     #"trg": "europarl-v7.fr-en.fr","src": "europarl-v7.fr-en.en"
     }
    ]

_VAL_DATA_SOURCES = [{"url": "http://data.statmt.org/wmt17/translation-task/dev.tgz",
     "trg": "newstest2013.de","src": "newstest2013.en"
     #"trg": "newstest2014-fren.fr","src": "newstest2014-fren.en"
     }]

_TEST_DATA_SOURCES = [{"url": "http://data.statmt.org/wmt17/translation-task/dev.tgz",
     "trg": "newstest2010.de","src": "newstest2010.en"
     #"trg": "newstest2010.fr","src": "newstest2010.en"
     }]

#_TEST_DATA_SOURCES  = [
#    { "url" : "https://storage.googleapis.com/tf-perf-public/" \
#                "official_transformer / test_data / newstest2014.tgz" ,
#     "trg" : "newstest2014.en" ,
#     "src" : "newstest2014.de" }]

class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def file_exist(dir_name, file_name):
    for sub_dir, _, files in os.walk(dir_name):
        if file_name in files:
            return os.path.join(sub_dir, file_name)
    return None


def download_and_extract(download_dir, url, src_filename, trg_filename):
    src_path = file_exist(download_dir, src_filename)
    trg_path = file_exist(download_dir, trg_filename)

    if src_path and trg_path:
        sys.stderr.write(f"Already downloaded and extracted {url}.\n")
        return src_path, trg_path

    compressed_file = _download_file(download_dir, url)

    sys.stderr.write(f"Extracting {compressed_file}.\n")
    with tarfile.open(compressed_file, "r:gz") as corpus_tar:
        pass
        #corpus_tar.extractall(download_dir)

    src_path = file_exist(download_dir, src_filename)
    trg_path = file_exist(download_dir, trg_filename)

    if src_path and trg_path:
        return src_path, trg_path

    raise OSError(f"Download/extraction failed for url {url} to path {download_dir}")


def _download_file(download_dir, url):
    filename = url.split("/")[-1]
    if file_exist(download_dir, filename):
        sys.stderr.write(f"Already downloaded: {url} (at {filename}).\n")
    else:
        sys.stderr.write(f"Downloading from {url} to {filename}.\n")
        with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
            pass
            #urllib.request.urlretrieve(url, filename=filename, reporthook=t.update_to)
    return filename


def get_raw_files(raw_dir, sources):
    raw_files = { "src": [], "trg": [], }
    for d in sources:
        src_file, trg_file = download_and_extract(raw_dir, d["url"], d["src"], d["trg"])
        raw_files["src"].append(src_file)
        raw_files["trg"].append(trg_file)
    return raw_files


def mkdir_if_needed(dir_name):
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)


def compile_files(raw_dir, raw_files, prefix):
    src_fpath = os.path.join(raw_dir, f"raw-{prefix}.src")
    trg_fpath = os.path.join(raw_dir, f"raw-{prefix}.trg")

    if os.path.isfile(src_fpath) and os.path.isfile(trg_fpath):
        sys.stderr.write(f"Merged files found, skip the merging process.\n")
        return src_fpath, trg_fpath

    sys.stderr.write(f"Merge files into two files: {src_fpath} and {trg_fpath}.\n")

    with open(src_fpath, 'w', encoding='utf8') as src_outf, open(trg_fpath, 'w', encoding='utf8') as trg_outf:
        for src_inf, trg_inf in zip(raw_files['src'], raw_files['trg']):
            sys.stderr.write(f'  Input files: \n'\
                    f'    - SRC: {src_inf}, and\n' \
                    f'    - TRG: {trg_inf}.\n')
            with open(src_inf,  encoding='utf8', newline='\n') as src_inf, open(trg_inf, encoding='utf8', newline='\n') as trg_inf:
                cntr = 0
                for i, line in enumerate(src_inf):
                    cntr += 1
                    line=replace_caracters(line)
                    src_outf.write(line.replace('\r', ' ').strip() + '\n')
                for j, line in enumerate(trg_inf):
                    cntr -= 1
                    line=replace_caracters(line)
                    trg_outf.write(line.replace('\r', ' ').strip() + '\n')
                assert cntr == 0, 'Number of lines in two files are inconsistent.'
    return src_fpath, trg_fpath

def replace_caracters(line):
        dic={'Ž':'é','@@ ':'','�':' ','ã¨':'è','Ã©':'é','Ãª':'ê','Ã¨':'è','Ã ':'à ','č':'è','‰':'%','œ':'oe',
             '’':'\'','amp#160;':' ','&nbsp;':' ','&160;':' ','&#160;':' ',
             'amp#45;':'-','&#45;':'-','“':'"',', ':' , ','; ':' ; ',': ':' : ','"':' " ','.':' .','  ':' ',
             '\u0027':'\'','&quot;':'"','&lt;':' > ','&gt;':' < ','ampquot;':'"','&amp;':'&','\xa0 ':'','&#x02bc;':'\'','&#x010d;':'è', u"\u202F":''}
        for k in dic :
            line=line.replace(k,dic[k])
            line=line.replace(k.upper(),dic[k])
        return line
def encode_file(bpe, in_file, out_file,lang='fr'):
    sys.stderr.write(f"Read raw content from {in_file} and \n"\
            f"Write encoded content to {out_file}\n")
    lang_model = spacy.load(lang)
    with codecs.open(in_file, encoding='utf-8') as in_f:
        with codecs.open(out_file, 'w', encoding='utf-8') as out_f:
            print(in_file)
            for line in in_f:
                #print(bpe.process_line(line).encode('utf-8').decode('utf-8'))
                line=replace_caracters(bpe.process_line(line))
                
                #line=' '.join([tok.text for tok in lang_model.tokenizer(line)])
                out_f.write(line.lower() )
                

def encode_files(bpe, src_in_file, trg_in_file, data_dir, prefix,lang_src='en',lang_trg='fr'):
    src_out_file = os.path.join(data_dir, f"{prefix}.src")
    trg_out_file = os.path.join(data_dir, f"{prefix}.trg")

    if os.path.isfile(src_out_file) and os.path.isfile(trg_out_file):
        sys.stderr.write(f"Encoded files found, skip the encoding process ...\n")

    encode_file(bpe, src_in_file, src_out_file,lang=lang_src)
    encode_file(bpe, trg_in_file, trg_out_file,lang=lang_trg)
    return src_out_file, trg_out_file


def main():
    parser = argparse.ArgumentParser()
    spacy_support_langs = ['de', 'el', 'en', 'es', 'fr', 'it', 'lt', 'nb', 'nl', 'pt']
    parser.add_argument('-lang_src', required=False, choices=spacy_support_langs,default='en')
    parser.add_argument('-lang_trg', required=False, choices=spacy_support_langs,default='de')
    parser.add_argument('-raw_dir', required=False ,default='data')
    parser.add_argument('-data_dir', required=False ,default='data')
    parser.add_argument('-codes', required=False,default='data')
    parser.add_argument('-save_data', required=False,default='m30k_deen_shr.pkl')
    parser.add_argument('-prefix', required=False,default='EN_DE')
    parser.add_argument('-max_len', type=int, default=50)
    parser.add_argument('--symbols', '-s', type=int, default=32000, help="Vocabulary size")
    parser.add_argument(
        '--min-frequency', type=int, default=6, metavar='FREQ',
        help='Stop if no symbol pair has frequency >= FREQ (default: %(default)s))')
    parser.add_argument('--dict-input', action="store_true",
        help="If set, input file is interpreted as a dictionary where each line contains a word-count pair")
    parser.add_argument(
        '--separator', type=str, default='@@', metavar='STR',
        help="Separator between non-final subword units (default: '%(default)s'))")
    parser.add_argument('--total-symbols', '-t', action="store_true")
    
    
    
    opt = parser.parse_args()
    opt.prefix=opt.lang_src.upper()+'_'+opt.lang_trg.upper()
    MAX_LEN = opt.max_len
    opt.min_freq=3
    
    src_lang_model = spacy.load(opt.lang_src)
    trg_lang_model = spacy.load(opt.lang_trg)
    
    
    
    def tokenize_src(text):
        return [tok.text.lower() for tok in src_lang_model.tokenizer(text )]

    def tokenize_trg(text):
        return [tok.text.lower()  for tok in trg_lang_model.tokenizer(text )]
    # Create folder if needed.
    mkdir_if_needed(opt.raw_dir)
    mkdir_if_needed(opt.data_dir)

    # Download and extract raw data.
    raw_train = get_raw_files(opt.raw_dir, _TRAIN_DATA_SOURCES)
    raw_val = get_raw_files(opt.raw_dir, _VAL_DATA_SOURCES)
    raw_test = get_raw_files(opt.raw_dir, _TEST_DATA_SOURCES)

    # Merge files into one.
    train_src, train_trg = compile_files(opt.raw_dir, raw_train, opt.prefix + '-train')
    val_src, val_trg = compile_files(opt.raw_dir, raw_val, opt.prefix + '-val')
    test_src, test_trg = compile_files(opt.raw_dir, raw_test, opt.prefix + '-test')

    # Build up the code from training files if not exist
    opt.codes = os.path.join(opt.data_dir, opt.codes)
    if not os.path.isfile(opt.codes):
        sys.stderr.write(f"Collect codes from training data and save to {opt.codes}.\n")
        learn_bpe(raw_train['src'] + raw_train['trg'], opt.codes, opt.symbols, opt.min_frequency, True)
    sys.stderr.write(f"BPE codes prepared.\n")

    sys.stderr.write(f"Build up the tokenizer.\n")
    with codecs.open(opt.codes, encoding='utf-8') as codes: 
        bpe = BPE(codes, separator=opt.separator)

    sys.stderr.write(f"Encoding ...\n")
    encode_files(bpe, train_src, train_trg, opt.data_dir, opt.prefix + '-train',lang_src=opt.lang_src,lang_trg=opt.lang_trg)
    encode_files(bpe, val_src, val_trg, opt.data_dir, opt.prefix + '-val',lang_src=opt.lang_src,lang_trg=opt.lang_trg)
    encode_files(bpe, test_src, test_trg, opt.data_dir, opt.prefix + '-test',lang_src=opt.lang_src,lang_trg=opt.lang_trg)
    sys.stderr.write(f"Done.\n")

    def replace_caracters(line):
        dic={'Ž':'é','@@ ':'','�':' ','ã¨':'è','Ã©':'é','Ãª':'ê','Ã¨':'è','Ã ':'à ','č':'è','‰':'%','œ':'oe',
             '’':'\'','amp#160;':' ','&nbsp;':' ','&160;':' ','&#160;':' ',
             'amp#45;':'-','&#45;':'-','“':'"',
             '\u0027':'\'','&quot;':'"','&lt;':'>','&gt;':'<','ampquot;':'"','&amp;':'&','\xa0 ':'','&#x02bc;':'\'','&#x010d;':'è'}
        for k in dic :
            line=line.replace(k,dic[k])
        return line

    SRC = torchtext.data.Field(
        tokenize=tokenize_src, lower=True,
        pad_token=Constants.PAD_WORD, init_token=Constants.BOS_WORD, eos_token=Constants.EOS_WORD)

    TRG = torchtext.data.Field(
        tokenize=tokenize_trg, lower=True,
        pad_token=Constants.PAD_WORD, init_token=Constants.BOS_WORD, eos_token=Constants.EOS_WORD)


    fields = (SRC, TRG)


    def filter_examples_with_length(x):
        #print([ a.encode('utf8').decode() for a in vars(x)['trg']])
        return len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN

    enc_train_files_prefix = opt.prefix + '-train'
    train = TranslationDataset(
        fields=fields,
        path=os.path.join(opt.data_dir, enc_train_files_prefix),
        exts=('.src', '.trg'),
        maxsize=opt.max_len,
        filter_pred=filter_examples_with_length)
    field = torchtext.data.Field(
        tokenize=lambda x: x.split(), lower=True,
        pad_token=Constants.PAD_WORD, init_token=Constants.BOS_WORD, eos_token=Constants.EOS_WORD)    
    from itertools import chain
    field.build_vocab(chain(train.src, train.trg), min_freq=opt.min_freq)    
    #fields = (field, field)
    SRC.build_vocab(train.src, min_freq=opt.min_freq)
    print('[Info] Get source language vocabulary size:', len(SRC.vocab))
    TRG.build_vocab(train.trg, min_freq=opt.min_freq)
    print('[Info] Get target language vocabulary size:', len(TRG.vocab))
    print('[Info] Get source sentenses size:', len(train.examples))
    print('[Info] Get target sentenses size:', len(train.examples))
    #print(TRG.vocab.itos)
    #from itertools import chain
    #field.build_vocab(chain(train.src, train.trg), min_freq=opt.min_freq)
    #print('field=====================',list(chain(train.src, train.trg)))
    data = { 'settings': opt, 'vocab': fields, }
    data = {
        'settings': opt,
        'vocab': fields,
        #'train': train.examples,
        #'valid': val.examples,
        #'test': test.examples
        }
    opt.save_data = os.path.join(opt.data_dir, opt.save_data)

    print('[Info] Dumping the processed data to pickle file', opt.save_data)
    pickle.dump(data, open(opt.save_data, 'wb'))
    return data

data=main()
if __name__ == '__main__':
    pass
    #main_wo_bpe()
    #data=main()
