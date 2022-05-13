'''
This script handles the training process.
'''

import argparse
import math
import time
import dill as pickle
from tqdm import tqdm
import spacy
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchtext.data import Field, Dataset, BucketIterator
from transformerDSM.translation import TranslationDataset,MyIterator,batch_size_fn
from transformerDSM.Translator import Translator
import transformerDSM.Constants as Constants
from transformerDSM.Models import Transformer
from transformerDSM.Optim import ScheduledOptim
from translate import load_model
from tensorboardX import SummaryWriter
from showAttention import showAttention,showAttention1

from jiwer import wer
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu,corpus_bleu
__author__ = "Belainine Billal"

def cal_performance(pred, gold, trg_pad_idx, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, trg_pad_idx, smoothing=smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(trg_pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return loss, n_correct, n_word


def cal_loss(pred, gold, trg_pad_idx, smoothing=False):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(trg_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')
    return loss


def patch_src(src, pad_idx):
    src = src.transpose(0, 1)
    return src


def patch_trg(trg, pad_idx):
    trg = trg.transpose(0, 1)
    trg, gold = trg[:, :-1], trg[:, 1:].contiguous().view(-1)
    
    return trg, gold


def train_epoch(model, training_data, optimizer, opt, device,writer_file, smoothing):
    ''' Epoch operation in training phase'''

    model.train()
    total_loss, n_word_total, n_word_correct = 0, 0, 0 
    i=0
    desc = '  - (Training)   '
    for batch in tqdm(training_data, mininterval=2, desc=desc, leave=False):
        
        # prepare data
        src_seq = patch_src(batch.src, opt.src_pad_idx).to(device)
        trg_seq, gold = map(lambda x: x.to(device), patch_trg(batch.trg, opt.trg_pad_idx))

        # forward
        optimizer.zero_grad()
        pred = model(src_seq, trg_seq)
        smoothing=True
        #return 0,0
        # backward and update parameters
        loss, n_correct, n_word = cal_performance(
            pred, gold, opt.trg_pad_idx, smoothing=smoothing) 
        loss.backward()
        optimizer.step_and_update_lr()

        # note keeping
        n_word_total += n_word
        n_word_correct += n_correct
        total_loss += loss.item()
        i+=1
        if i% 100==0:
            writer_file.add_scalar('train loss', min(total_loss/n_word_total,30), i)
            writer_file.add_scalar('train accu', 100*(n_correct/n_word), i)
            writer_file.add_scalar('train perplexity', math.exp(min(total_loss/n_word_total, 100)), i)
    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy


def eval_epoch(model, validation_data, device, opt):
    ''' Epoch operation in evaluation phase '''

    model.eval()
    total_loss, n_word_total, n_word_correct = 0, 0, 0

    desc = '  - (Validation) '
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2, desc=desc, leave=False):

            # prepare data
            src_seq = patch_src(batch.src, opt.src_pad_idx).to(device)
            trg_seq, gold = map(lambda x: x.to(device), patch_trg(batch.trg, opt.trg_pad_idx))

            # forward
            pred = model(src_seq, trg_seq)
            loss, n_correct, n_word = cal_performance(
                pred, gold, opt.trg_pad_idx, smoothing=False)

            # note keeping
            n_word_total += n_word
            n_word_correct += n_correct
            total_loss += loss.item()

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy


def train(model, training_data, validation_data,val, optimizer, device, opt,src_lang_model,trg_lang_model):
    ''' Start training '''
    writer = SummaryWriter('runs/logs')
    
    log_train_file, log_valid_file = None, None

    if opt.log:
        log_train_file = opt.log + '.train.log'
        log_valid_file = opt.log + '.valid.log'

        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))

        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch,loss,ppl,accuracy\n')
            log_vf.write('epoch,loss,ppl,accuracy\n')

    def print_performances(header, loss, accu, start_time):
        print('  - {header:12} ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
              'elapse: {elapse:3.3f} min'.format(
                  header=f"({header})", ppl=math.exp(min(loss, 100)),
                  accu=100*accu, elapse=(time.time()-start_time)/60))

    #valid_accus = []
    valid_losses = []
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu = train_epoch(
            model, training_data, optimizer, opt, device,writer, smoothing=opt.label_smoothing)
        print_performances('Training', train_loss, train_accu, start)

        start = time.time()
        valid_loss, valid_accu = eval_epoch(model, validation_data, device, opt)
        print_performances('Validation', valid_loss, valid_accu, start)

        valid_losses += [valid_loss]

        checkpoint = {'epoch': epoch_i, 'settings': opt, 'model': model.state_dict()}

        if opt.save_model:
            if opt.save_mode == 'all':
                model_name = opt.save_model + '_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_accu)
                torch.save(checkpoint, model_name)
            elif opt.save_mode == 'best':
                model_name = opt.save_model + '.chkpt'
                if valid_loss <= min(valid_losses):
                    torch.save(checkpoint, model_name)
                    writer.add_scalar('best loss', valid_loss, epoch_i)
                    writer.add_scalar('best accu', 100*valid_accu, epoch_i)
                    writer.add_scalar('best perplexity', math.exp(min(valid_loss, 100)), epoch_i)
                    print('    - [Info] The checkpoint file has been updated.')
                    if epoch_i >= 0  :
                        score ,error=calculate_bleu_score(opt,model,val,src_lang_model,trg_lang_model)
                        writer.add_scalar('WER', error, epoch_i)
                        writer.add_scalar('bleu score 1', score[0], epoch_i)
                        writer.add_scalar('bleu score 2', score[1], epoch_i)
                        writer.add_scalar('bleu score 3', score[2], epoch_i)
                        writer.add_scalar('bleu score 4', score[3], epoch_i)
        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                writer.add_scalar('valid loss', valid_loss, epoch_i)
                writer.add_scalar('valid perplexity', math.exp(min(valid_loss, 100)), epoch_i)
                log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=train_loss,
                    ppl=math.exp(min(train_loss, 100)), accu=100*train_accu))
                log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=valid_loss,
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu))
src_lang_model = spacy.load('en')
trg_lang_model = spacy.load('fr')              
                
def tokenize_src(text,src_lang_model):
        return text.split()#[tok.text.lower() for tok in src_lang_model.tokenizer(text)]

def tokenize_trg(text,trg_lang_model):
        return text.split()#[tok.text.lower() for tok in trg_lang_model.tokenizer(text)]                

def calculate_bleu_score(opt,model,validation_data,SRC,TRG):
    trg_bos_idx= TRG.vocab.stoi[Constants.BOS_WORD]
    trg_eos_idx= TRG.vocab.stoi[Constants.EOS_WORD]
    translator = Translator(
        model=model,
        beam_size=opt.beam_size,
        max_seq_len=opt.max_seq_len,
        src_pad_idx=opt.src_pad_idx,
        trg_pad_idx=opt.trg_pad_idx,
        trg_bos_idx=trg_bos_idx,
        trg_eos_idx=trg_eos_idx).cuda()
    score = [0.,0.,0.,0.]
    error = 0.
    output='candidate_pred.txt'
    cc = SmoothingFunction()
    unk_idx = SRC.vocab.stoi[SRC.unk_token]
    size=0
    candidates=[]
    referances=[]
    with open(output, 'w', encoding='utf-8') as f:
        for it,example in tqdm(enumerate(validation_data[:]), mininterval=2, desc='  - (Test)', leave=False):
            
            if(it < 1000):
                src_seq = [SRC.vocab.stoi.get(word, unk_idx) for word in ' '.join(example.src).strip().split(' ')]
                pred_seq, dec_enc_attn_list = translator.translate_sentence(torch.LongTensor([src_seq]).cuda())
       
                pred_line = ' '.join(TRG.vocab.itos[idx] for idx in pred_seq).strip()
                pred=pred_line.replace(Constants.BOS_WORD, '').replace(Constants.EOS_WORD, '')
                pred_line = pred_line.replace(Constants.BOS_WORD, '').replace(Constants.EOS_WORD, '')
                f.write('ref  orig: '+' '.join(example.src) + '\n')
                f.write('candidate: '+pred_line.strip() + '\n')
                f.write('referance: '+' '.join(example.trg) + '\n')
                f.flush()
                size+=1
                candidate=tokenize_trg(pred_line.strip(),trg_lang_model)
                candidates+=[candidate]
                referance=tokenize_trg(' '.join(example.trg).strip(),trg_lang_model)
                referances+=[referance]
                if(len(' '.join(candidate).strip())==0 ):
                    candidate+=[SRC.unk_token]
                if it< 100 :
                    #showAttention(' '.join(example.src),  dec_enc_attn_list,pred,rank=it,path='images/dec_enc')
                    showAttention1(' '.join(example.src), ' '+' '.join(example.trg)#pred
                    , dec_enc_attn_list,rank=it,path='images/fusion')
        try:
            score[0] = corpus_bleu([[referance]for referance in referances], candidates, smoothing_function=cc.method1)
            score[1] = corpus_bleu([[referance]for referance in referances], candidates, smoothing_function=cc.method2)
            score[2] = corpus_bleu([[referance]for referance in referances], candidates, smoothing_function=cc.method3)
            score[3] = corpus_bleu([[referance]for referance in referances], candidates, smoothing_function=cc.method4)
            error = wer([' '.join(referance) for referance in referances], [' '.join(candidate) for candidate in candidates])
        except:
            pass
    score = [250*s  for s in score]
    #error=error/size
    print("The bleu score is: ",str(score))
    print('Word error rate', error)
    return score ,error
def main():
    ''' 
    Usage:
    python train.py -data_pkl m30k_deen_shr.pkl -log m30k_deen_shr -embs_share_weight -proj_share_weight -label_smoothing -save_model trained -b 256 -warmup 128000
    '''
    #tensorboard --logdir=D:\attention-is-all-you-need-pytorch-master\runs
    parser = argparse.ArgumentParser()

    parser.add_argument('-data_pkl', default='data/m30k_deen_shr.pkl')     # all-in-1 data pickle or bpe field
    #parser.add_argument('-train_path', default='data/EN_FR-val')   # bpe encoded data
    parser.add_argument('-train_path', default='data/EN_DE-train')   # bpe encoded data
    parser.add_argument('-val_path', default='data/EN_DE-val')     # bpe encoded data

    parser.add_argument('-epoch', type=int, default=400)
    parser.add_argument('-b', '--batch_size', type=int, default=45)

    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    parser.add_argument('-d_k', type=int, default=32)
    parser.add_argument('-d_v', type=int, default=32)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-warmup','--n_warmup_steps', type=int, default=4000)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')
    parser.add_argument('-max_seq_len', type=int, default=50)
    parser.add_argument('-log', default='m30k_deen_shr')
    parser.add_argument('-save_model', default='model')
    parser.add_argument('-load_model', default=False)#'model.chkpt')#
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')
    parser.add_argument('-beam_size', type=int, default=1)
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-label_smoothing', action='store_true')

    opt = parser.parse_args()
    opt.cuda = True
    opt.lang_src='en'
    opt.lang_trg='fr'
    opt.d_word_vec = opt.d_model
    opt.proj_share_weight=True
    opt.embs_share_weight=True
    opt.label_smoothing=True
    if not opt.log and not opt.save_model:
        print('No experiment result will be saved.')
        raise

    if opt.batch_size < 2048 and opt.n_warmup_steps <= 1000:
        print('[Warning] The warmup steps may be not enough.\n'\
              '(sz_b, warmup) = (2048, 4000) is the official setting.\n'\
              'Using smaller batch w/o longer warmup may cause '\
              'the warmup stage ends with only little data trained.')

    device = torch.device('cuda' if opt.cuda else 'cpu')

    #========= Loading Dataset =========#

    if all((opt.train_path, opt.val_path)):
        training_data, validation_data, src_lang_model, trg_lang_model,val = prepare_dataloaders_from_bpe_files(opt, device)
    elif opt.data_pkl:
        training_data, validation_data = prepare_dataloaders(opt, device)
    else:
        raise

    print(opt)
    
    transformer = Transformer(
        opt.src_vocab_size,
        opt.trg_vocab_size,
        src_pad_idx=opt.src_pad_idx,
        trg_pad_idx=opt.trg_pad_idx,
        trg_emb_prj_weight_sharing=opt.proj_share_weight,
        emb_src_trg_weight_sharing=opt.embs_share_weight,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout, BSM=True).to(device)
    if opt.load_model :
        opt.model=opt.load_model
        transformer=load_model(opt, device).to(device)
    optimizer = ScheduledOptim(
        optim.Adam(transformer.parameters(), betas=(0.9, 0.98), eps=1e-9),
        2.0, opt.d_model, opt.n_warmup_steps)
    
    train(transformer, training_data, validation_data,val, optimizer, device, opt,src_lang_model,trg_lang_model)


def prepare_dataloaders_from_bpe_files(opt, device):
    batch_size = opt.batch_size
    MIN_FREQ = 3
    MAX_LEN=40
    if not opt.embs_share_weight:
        raise

    data = pickle.load(open(opt.data_pkl, 'rb'))
    MAX_LEN = data['settings'].max_len
    field = data['vocab']
    fields = field#(field, field)
    SRC,TRG=fields
    def replace_caracters(line):
        dic={'Ž':'é','@@ ':'','�':' ','ã¨':'è','Ã©':'é','Ãª':'ê','Ã¨':'è','Ã ':'à ','č':'è','‰':'%','œ':'oe'}
        for k in dic :
            line=line.replace(k,dic[k])
        return line
    src_lang_model = spacy.load('en')
    trg_lang_model = spacy.load('fr')


    def filter_examples_with_length(x):
        #print([ a.encode('utf8').decode() for a in vars(x)['trg']])
        return len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN
    train = TranslationDataset(
        fields=fields,
        path=opt.train_path, 
        exts=('.src', '.trg'),
        maxsize=MAX_LEN,
        filter_pred=filter_examples_with_length)
    val = TranslationDataset(
        fields=fields,
        path=opt.val_path, 
        exts=('.src', '.trg'),
        maxsize=MAX_LEN,
        filter_pred=filter_examples_with_length)
    
    opt.max_seq_len = MAX_LEN + 2
    opt.max_token_seq_len = MAX_LEN + 2
    opt.src_pad_idx = SRC.vocab.stoi[Constants.PAD_WORD]
    opt.trg_pad_idx = TRG.vocab.stoi[Constants.PAD_WORD]
    
    opt.src_vocab_size = len(SRC.vocab)
    opt.trg_vocab_size = len(TRG.vocab)
    #opt.fields=SRC, TRG
    train_iterator = MyIterator(train, batch_size=batch_size, device=device, train=True)
    val_iterator = MyIterator(val, batch_size=batch_size, device=device)
    return train_iterator, val_iterator,SRC,TRG,val


def prepare_dataloaders(opt, device):
    batch_size = opt.batch_size
    data = pickle.load(open(opt.data_pkl, 'rb'))

    opt.max_token_seq_len = data['settings'].max_len
    opt.src_pad_idx = data['vocab']['src'].vocab.stoi[Constants.PAD_WORD]
    opt.trg_pad_idx = data['vocab']['trg'].vocab.stoi[Constants.PAD_WORD]

    opt.src_vocab_size = len(data['vocab']['src'].vocab)
    opt.trg_vocab_size = len(data['vocab']['trg'].vocab)

    #========= Preparing Model =========#
    if opt.embs_share_weight:
        assert data['vocab']['src'].vocab.stoi == data['vocab']['trg'].vocab.stoi, \
            'To sharing word embedding the src/trg word2idx table shall be the same.'

    fields = {'src': data['vocab']['src'], 'trg':data['vocab']['trg']}

    train = Dataset(examples=data['train'], fields=fields)
    val = Dataset(examples=data['valid'], fields=fields)

    train_iterator = BucketIterator(train, batch_size=batch_size, device=device, train=True)
    val_iterator = BucketIterator(val, batch_size=batch_size, device=device)

    return train_iterator, val_iterator


if __name__ == '__main__':
    main()
