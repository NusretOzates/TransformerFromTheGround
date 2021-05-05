from tokenizers import BertWordPieceTokenizer, Tokenizer, SentencePieceBPETokenizer
from transformers import T5TokenizerFast, BertTokenizerFast
from typing import Union
import pandas as pd
from glosaNLP.corpora.tokenizers import Tokenizer # This is from a custom library which is not publicly released yet!
import tensorflow_datasets as tfds
import tensorflow as tf


# print('Read json')
# df = pd.read_json('data/sikayetvar.jsonl', lines=True, encoding='utf8')
# print('Tokenizing')
# sentences = Tokenizer.bstokenize(df['text'].values, flatten=True, lower=True)
# print('Writing to a file')
# with open('sentences.txt', 'w', encoding='utf8') as file:
#     for sentence in sentences:
#         print(sentence, file=file)


# print('Read csv')
# df = pd.read_csv('data/summary_data_dunyahalleri.csv', encoding='utf8')
# df.dropna(inplace=True, axis=0)
# print('Tokenizing')
# sentences = Tokenizer.bstokenize(df['Body_Text'].values, flatten=True, lower=True)
# print('Writing to a file')
# with open('sentences_dunya_halleri.txt', 'w', encoding='utf8') as file:
#     for sentence in sentences:
#         print(sentence, file=file)


#
# Creating a sentence dataset to train a tokenizer
# examples, metadata = tfds.load('ted_hrlr_translate/tr_to_en', with_info=True, as_supervised=True)
# train, val, test = examples['train'], examples['validation'], examples['test']
#
# turkish_sents = open('turkish_sentences.txt', 'w', encoding='utf8')
# english_sents = open('english_sentences.txt', 'w', encoding='utf8')
#
# for i, (tr, en) in enumerate(train):
#     tr_text = tf.compat.as_text(tr.numpy()).lower()
#     en_text = tf.compat.as_text(en.numpy()).lower()
#     print(tr_text, file=turkish_sents)
#     print(en_text, file=english_sents)
#
#
# turkish_sents.close()
# english_sents.close()

# tokenizer_tr = BertWordPieceTokenizer(clean_text=False, lowercase=False)
# print("training")
# tokenizer_tr.train('turkish_sentences.txt',vocab_size=700000)
# print("Saving")
# tokenizer_tr.save_model('.', 'turkish-tokenizer')
#
#
# tokenizer_en = BertWordPieceTokenizer(clean_text=False, lowercase=False)
# print("training")
# tokenizer_en.train('english_sentences.txt',vocab_size=700000)
# print("Saving")
# tokenizer_en.save_model('.', 'english-tokenizer')


# import sentencepiece as spm
# spm.SentencePieceTrainer.Train(input='english_sentences.txt', model_prefix='english', vocab_size=28650)


# bert_tokenizer = BertTokenizerFast(vocab_file='turkish-tokenizer-vocab.txt')
#
#
# tensor_2 = bert_tokenizer(text='deneme',return_tensors='tf')
#
# print(tensor_2)

# print(tokenizer.encode("Sizin yapacağınız ürünü severim. Sizden de nefret ederim").ids)
# print(tokenizer.encode("Sizin yapacağınız ürünü severim. Sizden de nefret ederim").tokens)
# print(tokenizer.encode("Sizin yapacağınız ürünü severim. Sizden de nefret ederim").attention_mask)
# print(tokenizer.encode("Sizin yapacağınız ürünü severim. Sizden de nefret ederim").word_ids)
# print(tokenizer.encode("Sizin yapacağınız ürünü severim. Sizden de nefret ederim"))

# token = BartTokenizerFast('vocab-vocab.txt',merges_file=)
# print(token.encode("Sizin yapacağınız ürünü severim. Sizden de nefret ederim"))
