import gradio as gr
from gradio.chat_interface import ChatInterface
import tensorflow as tf
assert tf.__version__.startswith('2')
tf.random.set_seed(1234)
import tensorflow_datasets as tfds
import os
import re
import numpy as np
import pandas as pd
import pydot 
import matplotlib.pyplot as plt

# Load tokenizer
tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file('tokenizer_vocab')
START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
VOCAB_SIZE = tokenizer.vocab_size + 2
MAX_LENGTH = 128

# Self-Attention Function
def scaled_dot_product_attention(query, key, value, mask):
  """Calculate the attention weights."""
  query = tf.cast(query, tf.float32)
  key = tf.cast(key, tf.float32)
  value = tf.cast(value, tf.float32)

  matmul_qk = tf.matmul(query, key, transpose_b=True)

  # scale matmul_qk
  depth = tf.cast(tf.shape(key)[-1], tf.float32)
  logits = matmul_qk / tf.math.sqrt(depth)

  # add the mask to zero out padding tokens
  if mask is not None:
    logits += (mask * -1e9)

  # softmax is normalized on the last axis (seq_len_k)
  attention_weights = tf.nn.softmax(logits, axis=-1)

  output = tf.matmul(attention_weights, value)

  return output

class MultiHeadAttention(tf.keras.layers.Layer):

  def __init__(self, d_model, num_heads, name="multi_head_attention"):
    super(MultiHeadAttention, self).__init__(name=name)
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.query_dense = tf.keras.layers.Dense(units=d_model)
    self.key_dense = tf.keras.layers.Dense(units=d_model)
    self.value_dense = tf.keras.layers.Dense(units=d_model)

    self.dense = tf.keras.layers.Dense(units=d_model)

  def split_heads(self, inputs, batch_size):
    inputs = tf.reshape(
        inputs, shape=(batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(inputs, perm=[0, 2, 1, 3])

  def call(self, inputs):
    query, key, value, mask = inputs['query'], inputs['key'], inputs['value'], inputs['mask']
    batch_size = tf.shape(query)[0]

    # linear layers
    query = self.query_dense(query)
    key = self.key_dense(key)
    value = self.value_dense(value)

    # split heads
    query = self.split_heads(query, batch_size)
    key = self.split_heads(key, batch_size)
    value = self.split_heads(value, batch_size)

    # scaled dot-product attention
    scaled_attention = scaled_dot_product_attention(query, key, value, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

    # concatenation of heads
    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model))

    # final linear layer
    outputs = self.dense(concat_attention)

    return outputs

def create_padding_mask(x):
  mask = tf.cast(tf.math.equal(x, 0), tf.float32)
  # (batch_size, 1, 1, sequence length)
  return mask[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(x):
  seq_len = tf.shape(x)[1]
  look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
  padding_mask = create_padding_mask(x)
  return tf.maximum(look_ahead_mask, padding_mask)

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model
        )

        # Apply sin to even indices in the array
        sines = tf.math.sin(angle_rads[:, 0::2])

        # Apply cos to odd indices in the array
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]

        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        # Make sure inputs are dense tensors
        if isinstance(inputs, tf.SparseTensor):
            inputs = tf.sparse.to_dense(inputs)

        # Add positional encoding
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

    def compute_output_shape(self, input_shape):
        return input_shape
    
def encoder_layer(units, d_model, num_heads, dropout, name="encoder_layer"):
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    # Multi-head attention
    attention = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout
    )(
        query=inputs, value=inputs, key=inputs, attention_mask=padding_mask
    )

    # Add & normalize
    attention = tf.keras.layers.Dropout(rate=dropout)(attention)
    attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attention)

    # Feed-forward network
    outputs = tf.keras.layers.Dense(units, activation="relu")(attention)
    outputs = tf.keras.layers.Dense(d_model)(outputs)

    # Add & normalize
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + outputs)

    return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)

def encoder(vocab_size, num_layers, units, d_model, num_heads, dropout, name="encoder"):
    inputs = tf.keras.Input(shape=(None,), name="inputs")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    # Make sure we're using dense embeddings
    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))

    # Add positional encoding
    positional_encoding = PositionalEncoding(vocab_size, d_model)
    outputs = positional_encoding(embeddings)

    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)

    # Stack encoder layers
    for i in range(num_layers):
        enc_layer = encoder_layer(
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name=f"encoder_layer_{i}"
        )
        outputs = enc_layer([outputs, padding_mask])

    return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)

def decoder_layer(units, d_model, num_heads, dropout, name="decoder_layer"):
  inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
  enc_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_outputs")
  look_ahead_mask = tf.keras.Input(
      shape=(1, None, None), name="look_ahead_mask")
  padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

  attention1 = MultiHeadAttention(
      d_model, num_heads, name="attention_1")(inputs={
          'query': inputs,
          'key': inputs,
          'value': inputs,
          'mask': look_ahead_mask
      })
  attention1 = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(attention1 + inputs)

  attention2 = MultiHeadAttention(
      d_model, num_heads, name="attention_2")(inputs={
          'query': attention1,
          'key': enc_outputs,
          'value': enc_outputs,
          'mask': padding_mask
      })
  attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2)
  attention2 = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(attention2 + attention1)

  outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention2)
  outputs = tf.keras.layers.Dense(units=d_model)(outputs)
  outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
  outputs = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(outputs + attention2)

  return tf.keras.Model(
      inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
      outputs=outputs,
      name=name)

def decoder(vocab_size,
            num_layers,
            units,
            d_model,
            num_heads,
            dropout,
            name='decoder'):
  inputs = tf.keras.Input(shape=(None,), name='inputs')
  enc_outputs = tf.keras.Input(shape=(None, d_model), name='encoder_outputs')
  look_ahead_mask = tf.keras.Input(
      shape=(1, None, None), name='look_ahead_mask')
  padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

  embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
  embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
  embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

  outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

  for i in range(num_layers):
    outputs = decoder_layer(
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
        name='decoder_layer_{}'.format(i),
    )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

  return tf.keras.Model(
      inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
      outputs=outputs,
      name=name)

def transformer(vocab_size,
                num_layers,
                units,
                d_model,
                num_heads,
                dropout,
                name="transformer"):
  inputs = tf.keras.Input(shape=(None,), name="inputs")
  dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")

  enc_padding_mask = tf.keras.layers.Lambda(
      create_padding_mask, output_shape=(1, 1, None),
      name='enc_padding_mask')(inputs)
  # mask the future tokens for decoder inputs at the 1st attention block
  look_ahead_mask = tf.keras.layers.Lambda(
      create_look_ahead_mask,
      output_shape=(1, None, None),
      name='look_ahead_mask')(dec_inputs)
  # mask the encoder outputs for the 2nd attention block
  dec_padding_mask = tf.keras.layers.Lambda(
      create_padding_mask, output_shape=(1, 1, None),
      name='dec_padding_mask')(inputs)

  enc_outputs = encoder(
      vocab_size=vocab_size,
      num_layers=num_layers,
      units=units,
      d_model=d_model,
      num_heads=num_heads,
      dropout=dropout,
  )(inputs=[inputs, enc_padding_mask])

  dec_outputs = decoder(
      vocab_size=vocab_size,
      num_layers=num_layers,
      units=units,
      d_model=d_model,
      num_heads=num_heads,
      dropout=dropout,
  )(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

  outputs = tf.keras.layers.Dense(units=vocab_size, name="outputs")(dec_outputs)

  return tf.keras.Model(inputs={'inputs': inputs, 'dec_inputs': dec_inputs}, outputs=outputs, name=name)

NUM_LAYERS = 2
D_MODEL = 256
NUM_HEADS = 8
UNITS = 384
DROPOUT = 0.1

model_U384 = transformer(
    vocab_size=VOCAB_SIZE,
    num_layers=NUM_LAYERS,
    units=UNITS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dropout=DROPOUT)

assert os.path.exists('bot_v4_Final_U384_E50.weights.h5'), "Weights not found!"
model_U384.load_weights('bot_v4_Final_U384_E50.weights.h5')
# Inference function

def evaluate(sentence):
    sentence = tf.expand_dims(START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)
    output = tf.expand_dims(START_TOKEN, 0)

    for i in range(MAX_LENGTH):
        predictions = model_U384({'inputs': sentence, 'dec_inputs': output}, training=False)
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        if tf.equal(predicted_id, END_TOKEN[0]):
            break

        output = tf.concat([output, predicted_id], axis=-1)

    return tokenizer.decode([i for i in tf.squeeze(output, axis=0) if i < tokenizer.vocab_size])

# Chatbot with history (even if ignored, required by Gradio)
def chatbot(message, history):
    response = evaluate(message)
    return response

# Create info content for the development page
info_content = """
## 📋 Informasi Pengembangan

### 🎯 Tentang Aplikasi
Chatbot Kesehatan Mental Perceraian adalah aplikasi yang dirancang khusus untuk memberikan dukungan emosional kepada anak-anak yang mengalami dampak dari perceraian orang tua.

### ✨ Fitur Utama
- **Mendengarkan Aktif**: Bot akan mendengarkan cerita dan perasaan Anda
- **Dukungan Emosional**: Memberikan respon yang empati dan mendukung
- **Privasi Terjamin**: Percakapan Anda aman dan terlindungi
- **Tersedia 24/7**: Siap membantu kapan saja Anda membutuhkan

### 🔧 Status Pengembangan
- **Versi**: Beta 1.0
- **Teknologi**: Python, Gradio, Machine Learning
- **Terakhir Diperbarui**: Juni 2025

### 👥 Tim Pengembang
Dikembangkan oleh tim yang peduli dengan kesehatan mental anak dan remaja.

### 📞 Kontak & Dukungan
Jika Anda memerlukan bantuan profesional, jangan ragu untuk menghubungi:
- Hotline Kesehatan Mental: 119 ext. 8
- Konselor Sekolah
- Psikolog atau Psikiater terdekat

### ⚠️ Catatan Penting
Bot ini adalah alat bantu dan tidak menggantikan konsultasi dengan profesional kesehatan mental.
"""

# Create Gradio ChatInterface with custom UI
with gr.Blocks(theme="soft", title="🧠 Chatbot Kesehatan Mental Perceraian") as iface:
    with gr.Tab("💬 Chat"):
        chat_interface = gr.ChatInterface(
            fn=chatbot,
            chatbot=gr.Chatbot(
                avatar_images=("https://i.ibb.co/f8f5c2T/user-icon.png", "https://i.ibb.co/m9hF6TH/bot-icon.png"),
                bubble_full_width=False,
                show_copy_button=True,
                height=500
            ),
            title="🧠 Chatbot Kesehatan Mental Perceraian",
            description="Selamat datang di Chatbot Kesehatan Mental. Curhatin aja apa yang kamu rasakan tentang kondisi orang tua kamu. Bot ini akan berusaha jadi teman yang mendengarkan. ❤️"
        )
    
    with gr.Tab("ℹ️ Informasi Pengembangan"):
        gr.Markdown(info_content)

# Launch app
if __name__ == "__main__":
    try:
        iface.launch(share=False)
    except Exception as e:
        print("Failed to launch:", e)
