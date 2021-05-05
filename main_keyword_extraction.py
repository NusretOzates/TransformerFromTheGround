### This is a deprecated file, to see nice example look at the main_tr_to_en.py file!

import tensorflow as tf
import time
from tokenizers import BertWordPieceTokenizer
from transformers import T5TokenizerFast

from learning_scheduler import CustomScheduler
from models.transformer import Transformer
from utilities import filter_max_length, loss_function, accuracy_function

BUFFER_SIZE = 5000
BATCH_SIZE = 32
EPOCH = 5
# Hyperparameters

num_layers = 8
d_model = 512
dff = 512
num_heads = 8
dropout_rate = 0.2

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.InteractiveSession(config=config)

print("Loading data")
import pandas as pd

df = pd.read_json('data/sikayetvar.jsonl', lines=True, encoding='utf8')
df['keywords'] = df['keywords'].apply(lambda x: ','.join(x))
df = df.sort_values(by='text', key=lambda x: x.str.len())
ds = tf.data.Dataset.from_tensor_slices((df['text'].to_numpy(), df['keywords'].to_numpy()))
cardinality = tf.data.experimental.cardinality(ds).numpy()
print('Create the Tokenizer')

tokenizer = BertWordPieceTokenizer('turkish-tokenizer-vocab.txt', clean_text=False, lowercase=False)

print(tokenizer.encode('merhaba ben nusret').tokens)


def encode(tr, en):
    tr_text = tf.compat.as_text(tr.numpy()).lower()
    en_text = tf.compat.as_text(en.numpy()).lower()
    tr = tokenizer.encode(tr_text).ids
    en = tokenizer.encode(en_text).ids
    tr = tf.constant(tr, dtype=tf.int32)
    en = tf.constant(en, dtype=tf.int32)

    return tr, en


def tf_encode(tr, en):
    result_tr, result_en = tf.py_function(encode, [tr, en], [tf.int32, tf.int32])
    result_tr.set_shape([None])
    result_en.set_shape([None])

    return result_tr, result_en


print('Encode Dataset')
test_dataset = ds.take(2000)
train_dataset = ds.skip(2000)

train_dataset = train_dataset.map(tf_encode, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.filter(filter_max_length)

train_dataset = train_dataset.cache()

train_dataset = train_dataset.padded_batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

print('encode val dataset')
val_dataset = test_dataset.map(tf_encode, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.filter(filter_max_length).padded_batch(BATCH_SIZE)

# i = 0
# for (batch, (inp, tar)) in enumerate(train_dataset):
#     i+=1
#
# print(i)


print('Create Scheduler and Optimizer etc.')
learning_rate = CustomScheduler(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

val_loss = tf.keras.metrics.Mean(name='val_loss')
val_accuracy = tf.keras.metrics.Mean(name='val_accuracy')

vocab_size = tokenizer.get_vocab_size()
print('Create Transformer model')
transformer = Transformer(num_layers, d_model, num_heads, dff,
                          vocab_size, vocab_size,
                          pe_input=vocab_size,
                          pe_target=vocab_size,
                          rate=dropout_rate)

inputs = tf.keras.Input(shape=(1, None), batch_size=BATCH_SIZE, name='inputs')
targets = tf.keras.Input(shape=(1, None), batch_size=BATCH_SIZE, name='targets')
preds, tar_real = Transformer(num_layers, d_model, num_heads, dff,
                              vocab_size, vocab_size,
                              pe_input=vocab_size,
                              pe_target=vocab_size,
                              rate=dropout_rate)([inputs, targets])

model = tf.keras.Model(inputs=[inputs, targets], outputs=[preds, tar_real])
model.compile(optimizer=optimizer, loss=loss_object, metrics=['accuracy'], run_eagerly=True)
model.fit(x=train_dataset, validation_data=val_dataset)

# inp = tf.convert_to_tensor([[2, 44, 22, 66, 77, 3]], dtype=tf.int32)
# tar_inp = tf.convert_to_tensor([[2]], dtype=tf.int32)
#
# print(transformer((inp, tar_inp), training=False))
# print(transformer.summary())
# transformer.save('saved_tf', save_format='tf')
# print('==='*35)
# model = tf.keras.models.load_model('saved_tf')
# print(model((inp, tar_inp), training=False))

checkpoint_path = './checkpoints/train'

ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
# ckpt = tf.train.Checkpoint(transformer=model, optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if ckpt_manager.latest_checkpoint:
#     ckpt.restore(ckpt_manager.latest_checkpoint)
#     print('Latest Checkpoint restored!!')

"""
The @tf.function trace-compiles train_step into a TF graph for faster execution.
The function specialized to the precise shape of the argument tensors. To avoid
re-tracing due to the variable sequence lengths or variable batch sizes (the last
batch is smaller), use input_signature to specify more generic shape
"""

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    tf.TensorSpec(shape=(None, None), dtype=tf.int32)
]


@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
    with tf.GradientTape() as tape:
        preds, tar_real = transformer([inp, tar], training=True, mask=None)
        loss = loss_function(tar_real, preds, loss_object)
    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    #     preds, tar_real, _ = model([inp, tar])
    #     loss = loss_function(tar_real, preds, loss_object)
    # gradients = tape.gradient(loss, model.trainable_variables)
    # optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(accuracy_function(tar_real, preds))


@tf.function(input_signature=train_step_signature)
def evaluate(inp, tar):
    preds, tar_real = transformer([inp, tar], True)
    loss = loss_function(tar_real, preds, loss_object)

    val_loss(loss)
    val_accuracy(accuracy_function(tar_real, preds))


print('Begin Training')
for epoch in range(EPOCH):
    start = time.time()
    train_loss.reset_states()
    train_accuracy.reset_states()

    val_loss.reset_states()
    val_accuracy.reset_states()

    for (batch, (inp, tar)) in enumerate(train_dataset):
        train_step(inp, tar)

        if batch % 200 == 0:
            print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                epoch + 1, batch, train_loss.result(), train_accuracy.result()))

    if (epoch + 1) % 1 == 0:
        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                            ckpt_save_path))
    print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                        train_loss.result(),
                                                        train_accuracy.result()))

    # print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

    # Add validation code here!
    for (batch, (inp, tar)) in enumerate(val_dataset):
        evaluate(inp, tar)
    print('-' * 35)
    print('Validation Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                                   val_loss.result(),
                                                                   val_accuracy.result()))
    print('-' * 35)
transformer.summary()
transformer.save('saved_tf', save_format='tf')

# for sentence, prediction in val_dataset.take(5):
#     sentence_text = tokenizer.decode(sentence.numpy().tolist()[0], skip_special_tokens=False)
#     end = tf.convert_to_tensor([tokenizer.token_to_id('[SEP]')], dtype=tf.int64)
#     output = tf.convert_to_tensor([tokenizer.token_to_id('[CLS]')], dtype=tf.int64)
#     output = tf.expand_dims(output, 0)
#     encoder_input = sentence
#     for i in range(20):
#         enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
#             encoder_input, output)
#
#         # predictions.shape == (batch_size, seq_len, vocab_size)
#         predictions, attention_weights = transformer(encoder_input,
#                                                      output,
#                                                      False,
#                                                      enc_padding_mask,
#                                                      combined_mask,
#                                                      dec_padding_mask)
#
#         # select the last word from the seq_len dimension
#         predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
#
#         predicted_id = tf.argmax(predictions, axis=-1)
#
#         # concatentate the predicted_id to the output which is given to the decoder
#         # as its input.
#         output = tf.concat([output, predicted_id], axis=-1)
#
#         # return the result if the predicted_id is equal to the end token
#         if predicted_id == end:
#             break
#
#     tokens = output.numpy().tolist()[0]
#     text = tokenizer.decode(tokens)  # shape: ()
#     # print(output)
#     print('text: ')
#     print(sentence_text)
#     print('prediction:')
#     print(text)
#     print('real value')
#     real = prediction.numpy().tolist()[0]
#     real = tokenizer.decode(real)
#     print(real)
#
#     # tokens = tokenizer_keyword.lookup(output.numpy().tolist())[0]
#     #
#     # print(tokens)
# as the target is english, the first word to the transformer should be the
# english start token.
# start, end = tf.convert_to_tensor([tokenizer_keyword.vocab_size], dtype=tf.int64), tf.convert_to_tensor(
#     [tokenizer_keyword.vocab_size + 1], dtype=tf.int64)
# output = start
# output = tf.expand_dims(output, 0)
#
# for i in range(20):
#     enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
#         encoder_input, output)
#
#     # predictions.shape == (batch_size, seq_len, vocab_size)
#     predictions, attention_weights = transformer(encoder_input,
#                                                  output,
#                                                  False,
#                                                  enc_padding_mask,
#                                                  combined_mask,
#                                                  dec_padding_mask)
#
#     # select the last word from the seq_len dimension
#     predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
#
#     predicted_id = tf.argmax(predictions, axis=-1)
#
#     # concatentate the predicted_id to the output which is given to the decoder
#     # as its input.
#     output = tf.concat([output, predicted_id], axis=-1)
#
#     # return the result if the predicted_id is equal to the end token
#     if predicted_id == end:
#         break
#
#     # output.shape (1, tokens)
# tokens = output.numpy().tolist()[0][1:-1]
# text = tokenizer_keyword.decode(tokens)  # shape: ()
# print(output)
# print(text)
# tokens = tokenizer_keyword.lookup(output.numpy().tolist())[0]
#
# print(tokens)
