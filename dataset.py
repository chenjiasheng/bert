import random
import numpy as np
import tensorflow as tf
tf.enable_eager_execution()

class SouhuDataset():
  def __init__(self, file, vocab_file, batch_size=64, max_seq_len=64, mask_prob=0.15, max_mask_cnt=20):
    self.id2token = [line.strip() for line in open(vocab_file, encoding='utf-8')]
    self.token2id = {self.id2token[i]: i for i in range(len(self.id2token))}
    self.texts = [line.strip() for line in open(file, encoding='utf-8')]
    self.texts = self.texts[:len(self.texts) // batch_size * batch_size]
    self.max_seq_len = max_seq_len
    self.batch_size = batch_size
    self.mask_prob = mask_prob
    self.max_mask_cnt = max_mask_cnt
    self.cur_epoch = 0
    self.cur_batch = 0
    self.CLS = self.token2id['[CLS]']
    self.SEP = self.token2id['[SEP]']
    self.MASK = self.token2id['[MASK]']

  def apply_mask(self, text):
    ids = [self.token2id[c] for c in text]
    mask_cnt = max(1, min(int(len(text) * self.mask_prob), self.max_mask_cnt))
    masked_positions = sorted(random.sample(range(len(text)), mask_cnt))
    masked_ids = [ids[pos] for pos in masked_positions]
    for i in range(len(masked_positions)):
      pos = masked_positions[i]
      rand_num = random.random()
      if rand_num < 0.8:
        ids[pos] = self.MASK
      elif rand_num < 0.9:
        ids[pos] = random.randint(self.MASK + 1, len(self.id2token) - 1)
      else:
        pass
    return ids, masked_positions, masked_ids

  def generate(self):
    batch_size = self.batch_size
    batches_per_epoch = len(self.texts) // batch_size
    max_mask_cnt = self.max_mask_cnt
    max_seq_len = self.max_seq_len
    for batch_id in range(batches_per_epoch):
      batch_texts = self.texts[batch_id * batch_size: (batch_id + 1) * batch_size]
      assert all(0 < len(text) <= max_seq_len - 2 for text in batch_texts)

      batch_input_ids = np.zeros((batch_size, max_seq_len), dtype=np.int32)
      batch_input_mask = np.zeros((batch_size, max_seq_len), dtype=np.int32)
      batch_segment_ids = np.zeros((batch_size, max_seq_len), dtype=np.int32)
      batch_masked_lm_positions = np.zeros((batch_size, max_mask_cnt), dtype=np.int32)
      batch_masked_lm_ids = np.zeros((batch_size, max_mask_cnt), dtype=np.int32)
      batch_masked_lm_weights = np.zeros((batch_size, max_mask_cnt), dtype=np.float32)
      batch_next_sentence_labels = np.zeros((batch_size, 1), dtype=np.int32)

      for i in range(batch_size):
        text = batch_texts[i]
        (input_ids, masked_positions, masked_ids) = self.apply_mask(text)
        batch_input_ids[i][0] = self.CLS
        batch_input_ids[i][1: 1 + len(input_ids)] = input_ids
        batch_input_ids[i][1 + len(input_ids)] = self.SEP
        batch_input_mask[i][:2 + len(input_ids)] = 1
        batch_masked_lm_positions[i][:len(masked_positions)] = masked_positions
        batch_masked_lm_positions[i][:len(masked_positions)] += 1
        batch_masked_lm_ids[i][:len(masked_positions)] = masked_ids
        batch_masked_lm_weights[i][:len(masked_positions)] = 1.0

      result = {"input_ids": batch_input_ids,
                "input_mask": batch_input_mask,
                "segment_ids": batch_segment_ids,
                "masked_lm_positions": batch_masked_lm_positions,
                "masked_lm_ids": batch_masked_lm_ids,
                "masked_lm_weights": batch_masked_lm_weights,
                "next_sentence_labels": batch_next_sentence_labels}

      self.cur_batch += 1
      if self.cur_batch % batches_per_epoch == 0:
        self.cur_epoch += 1
        self.cur_batch = 0
        random.shuffle(self.texts)

      yield result

  def __call__(self, params=None):
    output_types = {"input_ids": tf.int32,
                    "input_mask": tf.int32,
                    "segment_ids": tf.int32,
                    "masked_lm_positions": tf.int32,
                    "masked_lm_ids": tf.int32,
                    "masked_lm_weights": tf.float32,
                    "next_sentence_labels": tf.int32}

    output_shapes = {"input_ids": [self.batch_size, self.max_seq_len],
                     "input_mask": [self.batch_size, self.max_seq_len],
                     "segment_ids": [self.batch_size, self.max_seq_len],
                     "masked_lm_positions": [self.batch_size, self.max_mask_cnt],
                     "masked_lm_ids": [self.batch_size, self.max_mask_cnt],
                     "masked_lm_weights": [self.batch_size, self.max_mask_cnt],
                     "next_sentence_labels": [self.batch_size, 1]}

    d = tf.data.Dataset.from_generator(generator=self.generate,
                                       output_types=output_types,
                                       output_shapes=output_shapes)
    d = d.repeat()
    return d


if __name__ == "__main__":
  ds = SouhuDataset("data/sohu/dev.txt", "data/chinese_L-12_H-768_A-12/cn_vocab.txt")
  input_fn = ds()
  it = input_fn.make_one_shot_iterator()

  getter = it.next()
  print(getter)