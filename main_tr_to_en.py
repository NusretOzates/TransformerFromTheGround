import tensorflow_datasets as tfds
import tensorflow as tf

from tensorflow.python.ops.gen_dataset_ops import PrefetchDataset
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizerFast
from models.transformer import Transformer
from utilities import filter_max_length

BUFFER_SIZE = 20000
BATCH_SIZE = 32
EPOCH = 1
# Hyperparameters

num_layers = 2
d_model = 128
dff = 64
num_heads = 1
dropout_rate = 0.1

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.InteractiveSession(config=config)

print("Loading data")
# Use tffs English-Turkish Dataset
examples, metadata = tfds.load('ted_hrlr_translate/tr_to_en', with_info=True, as_supervised=True,
                               split=['train[:1000]', 'validation[:1000]', 'test[:1000]'])
train, val, test = examples[0], examples[1], examples[2]


# tokenizer_tr = BertWordPieceTokenizer('turkish-tokenizer-vocab.txt', clean_text=False, lowercase=False)
# tokenizer_en = BertWordPieceTokenizer('english-tokenizer-vocab.txt', clean_text=False, lowercase=False)
#
# vocab_size_input = tokenizer_tr.get_vocab_size()
# vocab_size_output = tokenizer_en.get_vocab_size()

tokenizer_tr = BertTokenizerFast(vocab_file='turkish-tokenizer-vocab.txt')
tokenizer_en = BertTokenizerFast(vocab_file='english-tokenizer-vocab.txt')

vocab_size_input = tokenizer_tr.vocab_size
vocab_size_output = tokenizer_en.vocab_size


def encode(tr, en):
    tr_text = tf.compat.as_text(tr.numpy()).lower()
    en_text = tf.compat.as_text(en.numpy()).lower()
    tr = tokenizer_tr.encode(tr_text)
    en = tokenizer_en.encode(en_text)
    tr = tf.constant(tr, dtype=tf.int32)
    en = tf.constant(en, dtype=tf.int32)

    return tr, en


def tf_encode(tr, en):
    result_tr, result_en = tf.py_function(encode, [tr, en], [tf.int32, tf.int32])
    result_tr.set_shape([None])
    result_en.set_shape([None])

    return result_tr, result_en


print('Encode Dataset')
train_dataset: PrefetchDataset = train.map(tf_encode,
                                           deterministic=tf.data.Options.experimental_deterministic)
train_dataset = train_dataset.filter(filter_max_length)

train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

val_dataset = val.map(tf_encode,
                      deterministic=tf.data.Options.experimental_deterministic)
val_dataset = val_dataset.filter(filter_max_length).cache().padded_batch(BATCH_SIZE)

print('Create Optimizer etc.')
optimizer = tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.98, epsilon=1e-9)
sgd = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='sum')


end_of_sentence = tokenizer_en.convert_tokens_to_ids(['[SEP]'])
end_of_sentence = tf.convert_to_tensor([end_of_sentence], dtype=tf.int32)

print('Create Transformer model')
transformer = Transformer(num_layers, d_model, num_heads, dff,
                          vocab_size_input, vocab_size_output,
                          EOS_TOKEN=end_of_sentence,
                          pe_input=vocab_size_input,
                          pe_target=vocab_size_output,
                          rate=dropout_rate)

# checkpoint_path = './checkpoints/train'
# ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
# ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
#
# if ckpt_manager.latest_checkpoint:
#     ckpt.restore(ckpt_manager.latest_checkpoint)
#     print('Latest Checkpoint restored!!')

transformer.compile(optimizer=sgd, loss=loss_object, metrics='accuracy', run_eagerly=False)
transformer.fit(train_dataset, validation_data=val_dataset, epochs=EPOCH)
transformer.save('saved_tf', save_format='tf')
print('saved')
"""
The @tf.function trace-compiles train_step into a TF graph for faster execution.
The function specialized to the precise shape of the argument tensors. To avoid
re-tracing due to the variable sequence lengths or variable batch sizes (the last
batch is smaller), use input_signature to specify more generic shape
"""

# train_step_signature = [
#     tf.TensorSpec(shape=(None, None), dtype=tf.int32),
#     tf.TensorSpec(shape=(None, None), dtype=tf.int32)
# ]
#
#
# @tf.function(input_signature=train_step_signature)
# def train_step(inp, tar):
#     with tf.GradientTape() as tape:
#         preds, tar_real = transformer([inp, tar], True)
#         loss = loss_function(tar_real, preds)
#     gradients = tape.gradient(loss, transformer.trainable_variables)
#     optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
#
#     #     preds, tar_real, _ = model([inp, tar])
#     #     loss = loss_function(tar_real, preds, loss_object)
#     # gradients = tape.gradient(loss, model.trainable_variables)
#     # optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#
#     train_loss(loss)
#     train_accuracy(accuracy_function(tar_real, preds))
#
#
# @tf.function(input_signature=train_step_signature)
# def evaluate(inp, tar):
#     preds, tar_real = transformer([inp, tar], True)
#     loss = loss_function(tar_real, preds, )
#
#     val_loss(loss)
#     val_accuracy(accuracy_function(tar_real, preds))
#
#
# print('Begin Training')
# for epoch in range(EPOCH):
#     start = time.time()
#     train_loss.reset_states()
#     train_accuracy.reset_states()
#
#     val_loss.reset_states()
#     val_accuracy.reset_states()
#
#     for (batch, (inp, tar)) in enumerate(train_dataset):
#         train_step(inp, tar)
#
#         if batch % 500 == 0:
#             print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
#                 epoch + 1, batch, train_loss.result(), train_accuracy.result()))
#
#     if (epoch + 1) % 1 == 0:
#         ckpt_save_path = ckpt_manager.save()
#         print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
#                                                             ckpt_save_path))
#     print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
#                                                         train_loss.result(),
#                                                         train_accuracy.result()))
#
#     # print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
#
#     # Add validation code here!
#     for (batch, (inp, tar)) in enumerate(val_dataset):
#         evaluate(inp, tar)
#     print('-' * 35)
#     print('Validation Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
#                                                                    val_loss.result(),
#                                                                    val_accuracy.result()))
#     print('-' * 35)
#
# transformer.save('saved_tf', save_format='tf')
#
#
# def predict(text: str, language: str = 'tr'):
#     if language == 'tr':
#         tokenized_text = tokenizer_tr.encode(text).ids
#         tokenized_text = tf.convert_to_tensor([tokenized_text], dtype=tf.int64)
#
#         tokenized_en = tokenizer_en.token_to_id('[CLS]')
#         tokenized_en = tf.convert_to_tensor([[tokenized_en]], dtype=tf.int64)
#
#         end_of_sentence = tokenizer_en.token_to_id('[SEP]')
#         end_of_sentence = tf.convert_to_tensor([[end_of_sentence]], dtype=tf.int64)
#         while True:
#             pred, _, _ = transformer([tokenized_text, tokenized_en], training=False)
#             prediction = tf.argmax(pred[:, -1:, :], axis=-1)
#             tokenized_en = tf.concat([tokenized_en, prediction], axis=-1)
#
#             if prediction == end_of_sentence:
#                 break
#
#         print(tokenizer_en.decode(tokenized_en.numpy().tolist()[0], skip_special_tokens=False))


