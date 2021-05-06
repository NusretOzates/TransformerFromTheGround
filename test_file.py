import tensorflow as tf
from tokenizers import BertWordPieceTokenizer

model = tf.keras.models.load_model('saved_tf')
tokenizer_tr = BertWordPieceTokenizer('turkish-tokenizer-vocab.txt', clean_text=False, lowercase=False)
tokenizer_en = BertWordPieceTokenizer('english-tokenizer-vocab.txt', clean_text=False, lowercase=False)


def predict(text: str, language: str = 'tr'):
    if language == 'tr':
        tokenized_text = tokenizer_tr.encode(text).ids
        tokenized_text = tf.convert_to_tensor([tokenized_text], dtype=tf.int32)

        tokenized_en = tokenizer_en.token_to_id('[CLS]')
        tokenized_en = tf.convert_to_tensor([[tokenized_en]], dtype=tf.int32)

        end_of_sentence = tokenizer_en.token_to_id('[SEP]')
        end_of_sentence = tf.convert_to_tensor([[end_of_sentence]], dtype=tf.int32)
        tokenized_en = tf_predict(end_of_sentence, tokenized_en, tokenized_text)

        tf.print(tokenizer_en.decode(tokenized_en.numpy().tolist()[0], skip_special_tokens=False))


train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
]


def tf_predict(end_of_sentence, tokenized_en, tokenized_text):

    for i in range(50):
        prediction, tokenized_en = predict_step(tokenized_en, tokenized_text)

        if prediction == end_of_sentence:
            tf.print('UPpsss')
            break


    return tokenized_en

@tf.function(input_signature=train_step_signature)
def predict_step(tokenized_en, tokenized_text):
    pred = model([tokenized_text, tokenized_en], training=False)
    prediction = tf.argmax(pred[:, -1:, :], axis=-1)
    prediction = tf.cast(prediction, dtype=tf.int32)
    tokenized_en = tf.concat([tokenized_en, prediction], axis=-1)
    return prediction, tokenized_en


predict('Birlikte olacağımız bir gelecek hayal et')
