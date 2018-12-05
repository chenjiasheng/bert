# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#"""Run masked LM/next sentence masked_lm pre-training for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import modeling
import optimization
import tensorflow as tf
import numpy as np
from tensorflow.python import debug as tf_debug

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "input_file", None,
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")

flags.DEFINE_integer(
    "max_predictions_per_seq", 20,
    "Maximum number of masked LM predictions per sequence. "
    "Must match data generation.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_predict", False, "Whether to run predict on the input_file/stdin.")

flags.DEFINE_string("full_vocab_file", "",
                    "Original vocab file. If provided and do_predict is True, the word embedding and output bias of "
                    "the mask lm will be picked from the init_checkpoint according the two vocab files, and the new "
                    "weights will be saved to {init_checkpoint}.downsampled checkpoint. This useful to down-sampling "
                    "a mask lm model with a subset vocab from the pre-trained model with an original full vocab. "
                    
                    "This will do down-sampling: \
                     python run_pretraining.py --do_predict True --input_file ''  --output_dir results/ \
                     --bert_config_file data/chinese_L-12_H-768_A-12/cn_bert_config.json \
                     --vocab_file data/chinese_L-12_H-768_A-12/cn_vocab.txt \
                     --full_vocab_file data/chinese_L-12_H-768_A-12/vocab.txt \
                     --init_checkpoint data/chinese_L-12_H-768_A-12/bert_model.ckpt "
                    
                    "And then use the new down-sampled checkpoint next: \
                     python run_pretraining.py --do_predict True --input_file ''  --output_dir results/ \
                     --bert_config_file data/chinese_L-12_H-768_A-12/cn_bert_config.json \
                     --vocab_file data/chinese_L-12_H-768_A-12/cn_vocab.txt \
                     --init_checkpoint data/chinese_L-12_H-768_A-12/bert_model.ckpt.downsampled"
                    )


flags.DEFINE_string("vocab_file", "data/chinese_L-12_H-768_A-12/cn_vocab.txt", "Vocab file.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_integer("num_train_steps", 100000, "Number of training steps.")

flags.DEFINE_integer("num_warmup_steps", 10000, "Number of warmup steps.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("max_eval_steps", 100, "Maximum number of eval steps.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings,
                     full_vocab_file=None, subset_vocab_file=None):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    masked_lm_positions = features["masked_lm_positions"]
    masked_lm_ids = features["masked_lm_ids"]
    masked_lm_weights = features["masked_lm_weights"]
    next_sentence_labels = features["next_sentence_labels"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    (masked_lm_loss,
     masked_lm_example_loss, masked_lm_log_probs, output_bias) = \
      get_masked_lm_output(
         bert_config, model.get_sequence_output(), model.get_embedding_table(),
         masked_lm_positions, masked_lm_ids, masked_lm_weights)

    (next_sentence_loss, next_sentence_example_loss,
     next_sentence_log_probs) = get_next_sentence_output(
         bert_config, model.get_pooled_output(), next_sentence_labels)

    total_loss = masked_lm_loss + next_sentence_loss

    tvars = tf.trainable_variables()

    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      # replace the embeddngs weights
      if full_vocab_file:
        def pick_sub_set(embeddings, full_vocab_file, new_vocab_file):
          old_vocab = [line.strip() for line in open(full_vocab_file, encoding='utf-8')]
          old_vocab_inv = {old_vocab[i]: i for i in range(len(old_vocab))}
          _new_vocab = [line.strip() for line in open(new_vocab_file, encoding='utf-8')]
          if len(_new_vocab) != len(set(_new_vocab)):
            tf.logging.ERROR("Dupllicated entries in %s." % new_vocab_file)
          new_vocab = [x for x in _new_vocab if x in old_vocab_inv]
          if len(new_vocab) != len(_new_vocab):
            tf.logging.ERROR("Threre are OOVs in %s that not in %s. " % (new_vocab_file, full_vocab_file))
          ids = [old_vocab_inv[c] for c in new_vocab]
          return np.asarray([embeddings[x] for x in ids])
        from tensorflow.python.training import checkpoint_utils
        ckpt_variables = dict(checkpoint_utils.list_variables(init_checkpoint))
        v1 = model.embedding_table
        v2 = output_bias

        tvars = [x for x in tvars if x.name not in [v1.name, v2.name]]
        for v in [v1, v2]:
          with tf.device(v.device), tf.device("/cpu:0"):
            constant_op = tf.constant(pick_sub_set(checkpoint_utils.load_variable(init_checkpoint, v.op.name),
                                                   full_vocab_file, subset_vocab_file))
            v._initializer_op = v.assign(constant_op)
            v._initial_value = constant_op

      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
                    masked_lm_weights, next_sentence_example_loss,
                    next_sentence_log_probs, next_sentence_labels):
        """Computes the loss and accuracy of the model."""
        with tf.name_scope("metric"):
          masked_lm_log_probs = tf.reshape(masked_lm_log_probs,
                                           [-1, masked_lm_log_probs.shape[-1]])
          masked_lm_predictions = tf.argmax(
              masked_lm_log_probs, axis=-1, output_type=tf.int32)
          masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])
          masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
          masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
          masked_lm_accuracy = tf.metrics.accuracy(
              labels=masked_lm_ids,
              predictions=masked_lm_predictions,
              weights=masked_lm_weights)
          masked_lm_mean_loss = tf.metrics.mean(
              values=masked_lm_example_loss, weights=masked_lm_weights)

          next_sentence_log_probs = tf.reshape(
              next_sentence_log_probs, [-1, next_sentence_log_probs.shape[-1]])
          next_sentence_predictions = tf.argmax(
              next_sentence_log_probs, axis=-1, output_type=tf.int32)
          next_sentence_labels = tf.reshape(next_sentence_labels, [-1])
          next_sentence_accuracy = tf.metrics.accuracy(
              labels=next_sentence_labels, predictions=next_sentence_predictions)
          next_sentence_mean_loss = tf.metrics.mean(
              values=next_sentence_example_loss)

        return {
            "masked_lm_accuracy": masked_lm_accuracy,
            "masked_lm_loss": masked_lm_mean_loss,
            "next_sentence_accuracy": next_sentence_accuracy,
            "next_sentence_loss": next_sentence_mean_loss,
        }

      eval_metrics = (metric_fn, [
          masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
          masked_lm_weights, next_sentence_example_loss,
          next_sentence_log_probs, next_sentence_labels
      ])
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)

    elif mode == tf.estimator.ModeKeys.PREDICT:
      """Computes the loss and accuracy of the model."""
      with tf.name_scope("predict"):
        masked_lm_log_probs = tf.reshape(
          masked_lm_log_probs, (-1, FLAGS.max_predictions_per_seq, masked_lm_log_probs.shape[-1]))
        masked_lm_predictions = tf.argmax(
          masked_lm_log_probs, axis=-1, output_type=tf.int32)

        next_sentence_predictions = tf.argmax(
          next_sentence_log_probs, axis=-1, output_type=tf.int32)

      predictions =  {
        "input_ids": input_ids,
        "input_mask": input_mask,
        "masked_lm_positions": masked_lm_positions,
        "masked_lm_ids": masked_lm_ids,
        "masked_lm_weights": masked_lm_weights,
        "masked_lm_log_probs": masked_lm_log_probs,
        "masked_lm_predictions": masked_lm_predictions,
        "next_sentence_predictions": next_sentence_predictions,
      }

      output_spec = tf.estimator.EstimatorSpec(
        mode=mode,
        loss=total_loss,
        predictions=predictions)
    else:
      raise ValueError("Only TRAIN and EVAL modes are supported: %s" % (mode))

    return output_spec

  return model_fn


def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         label_ids, label_weights):
  """Get loss and log probs for the masked LM."""
  input_tensor = gather_indexes(input_tensor, positions)

  with tf.variable_scope("cls/predictions"):
    # We apply one more non-linear transformation before the output layer.
    # This matrix is not used after pre-training.
    with tf.variable_scope("transform"):
      input_tensor = tf.layers.dense(
          input_tensor,
          units=bert_config.hidden_size,
          activation=modeling.get_activation(bert_config.hidden_act),
          kernel_initializer=modeling.create_initializer(
              bert_config.initializer_range))
      input_tensor = modeling.layer_norm(input_tensor)

    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each token.
    output_bias = tf.get_variable(
        "output_bias",
        shape=[bert_config.vocab_size],
        initializer=tf.zeros_initializer())
    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    label_ids = tf.reshape(label_ids, [-1])
    label_weights = tf.reshape(label_weights, [-1])

    one_hot_labels = tf.one_hot(
        label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

    # The `positions` tensor might be zero-padded (if the sequence is too
    # short to have the maximum number of predictions). The `label_weights`
    # tensor has a value of 1.0 for every real prediction and 0.0 for the
    # padding predictions.
    per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
    numerator = tf.reduce_sum(label_weights * per_example_loss)
    denominator = tf.reduce_sum(label_weights) + 1e-5
    loss = numerator / denominator

  return (loss, per_example_loss, log_probs, output_bias)


def get_next_sentence_output(bert_config, input_tensor, labels):
  """Get loss and log probs for the next sentence prediction."""

  # Simple binary classification. Note that 0 is "next sentence" and 1 is
  # "random sentence". This weight matrix is not used after pre-training.
  with tf.variable_scope("cls/seq_relationship"):
    output_weights = tf.get_variable(
        "output_weights",
        shape=[2, bert_config.hidden_size],
        initializer=modeling.create_initializer(bert_config.initializer_range))
    output_bias = tf.get_variable(
        "output_bias", shape=[2], initializer=tf.zeros_initializer())

    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    labels = tf.reshape(labels, [-1])
    one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)
    return (loss, per_example_loss, log_probs)


def gather_indexes(sequence_tensor, positions):
  with tf.name_scope("gather_indexes"):
    """Gathers the vectors at the specific positions over a minibatch."""
    sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    flat_offsets = tf.reshape(
        tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor,
                                      [batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    return output_tensor


def predict_input_fn(input_files,
                     max_seq_length,
                     max_predictions_per_seq,
                     mode,
                     num_cpu_threads=4):
  def stdin_gen():
    max_seq_length = FLAGS.max_seq_length
    max_predictions_per_seq = FLAGS.max_predictions_per_seq
    from tokenization import FullTokenizer
    if not hasattr(stdin_gen, "tokenizer"):
      stdin_gen.tokenizer = FullTokenizer(
        vocab_file=FLAGS.vocab_file)

    if FLAGS.input_file:
      file = open(FLAGS.input_file, encoding="utf-8")
    else:
      import readline
      readline.parse_and_bind('tab: complete')
      readline.parse_and_bind('set editing-mode vi')
      print("Input your sentence, mask out characters by a following \'*\' or \'×\':")

    # for user_input in file:
    while True:
      try:
        if not FLAGS.input_file:
          user_input = input("input:")
        else:
          user_input = file.readline()
        user_input = user_input.strip().replace("×", "*")

        _input_ids = np.zeros(max_seq_length, dtype=np.int32)
        _input_mask = np.zeros(max_seq_length, dtype=np.int32)
        _segment_ids = np.zeros(max_seq_length, dtype=np.int32)
        _masked_lm_positions = np.zeros(max_predictions_per_seq, dtype=np.int32)
        _masked_lm_ids = np.zeros(max_predictions_per_seq, dtype=np.int32)
        _masked_lm_weights = np.zeros(max_predictions_per_seq, dtype=np.float32)
        _next_sentence_labels = np.zeros(1, dtype=np.int32)

        if len(user_input) == 0:
          continue

        masked_positions = []
        pos = 0
        while True:
          pos = user_input.find("*", pos + 1)
          if pos == -1:
            break
          masked_positions.append(pos)
        masked_positions = [masked_positions[i] - i - 1 for i in range(len(masked_positions))]
        if len(masked_positions) == 0 or any(x < 0 for x in masked_positions):
          print("Error: Invalid masked positions.")
          continue

        sentence = user_input.replace("*", "")
        tokens = stdin_gen.tokenizer.tokenize(sentence)
        if len(sentence) != len(tokens):
          print("Error: Invalid input '%s', supports only chinese." % user_input)
          continue
        if len(sentence) > max_seq_length - 2:
          print("Error: Invalid input '%s', sequence_length(=%d) > max_seq_length-2(=%d)."
                % (user_input, len(sentence), max_seq_length - 2))
          continue

        ids = stdin_gen.tokenizer.convert_tokens_to_ids(tokens)

        masked_ids = []
        for pos in masked_positions:
          masked_ids.append(ids[pos])
          ids[pos] = 103
        masked_positions = np.asarray(masked_positions) + 1
        ids = [101] + ids

        _input_ids[:len(ids)] = ids
        _input_mask[:len(ids)] = 1
        _masked_lm_positions[:len(masked_positions)] = masked_positions
        _masked_lm_ids[:len(masked_positions)] = masked_ids
        _masked_lm_weights[:len(masked_positions)] = 1.0

        result = {"input_ids": _input_ids,
                  "input_mask": _input_mask,
                  "segment_ids": _segment_ids,
                  "masked_lm_positions": _masked_lm_positions,
                  "masked_lm_ids": _masked_lm_ids,
                  "masked_lm_weights": _masked_lm_weights,
                  "next_sentence_labels": _next_sentence_labels}
        yield result
      except EOFError:
        print()
        break
      except:
        print("Error: Invalid inputs.")
        continue

  output_types = {"input_ids": tf.int32,
                  "input_mask": tf.int32,
                  "segment_ids": tf.int32,
                  "masked_lm_positions": tf.int32,
                  "masked_lm_ids": tf.int32,
                  "masked_lm_weights": tf.float32,
                  "next_sentence_labels": tf.int32}

  output_shapes = {"input_ids": [max_seq_length],
                   "input_mask": [max_seq_length],
                   "segment_ids": [max_seq_length],
                   "masked_lm_positions": [max_predictions_per_seq],
                   "masked_lm_ids": [max_predictions_per_seq],
                   "masked_lm_weights": [max_predictions_per_seq],
                   "next_sentence_labels": [1]}

  d = tf.data.Dataset.from_generator(generator=stdin_gen,
                                     output_types=output_types,
                                     output_shapes=output_shapes)
  d = d.batch(batch_size=1)
  return d


def input_fn_builder(input_files,
                     max_seq_length,
                     max_predictions_per_seq,
                     mode,
                     num_cpu_threads=4):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  assert mode in ["train", "eval", "predict"]

  def input_fn(params=None):
    """The actual input function."""
    if mode == "predict":
      return predict_input_fn(input_files,
                     max_seq_length,
                     max_predictions_per_seq,
                     mode,
                     num_cpu_threads=4)

    batch_size = params["batch_size"] if params else 1

    name_to_features = {
        "input_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "masked_lm_positions":
            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_ids":
            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_weights":
            tf.FixedLenFeature([max_predictions_per_seq], tf.float32),
        "next_sentence_labels":
            tf.FixedLenFeature([1], tf.int64),
    }

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    if mode == "train":
      d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
      d = d.repeat()
      d = d.shuffle(buffer_size=len(input_files))

      # `cycle_length` is the number of parallel files that get read.
      cycle_length = min(num_cpu_threads, len(input_files))

      # `sloppy` mode means that the interleaving is not exact. This adds
      # even more randomness to the training pipeline.
      d = d.apply(
          tf.contrib.data.parallel_interleave(
              tf.data.TFRecordDataset,
              sloppy=True,
              cycle_length=cycle_length))
      d = d.shuffle(buffer_size=100)
    else:
      d = tf.data.TFRecordDataset(input_files)
      # Since we evaluate for a fixed number of steps we don't want to encounter
      # out-of-range exceptions.
      d = d.repeat()

    # We must `drop_remainder` on training because the TPU requires fixed
    # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
    # and we *don't* want to drop the remainder, otherwise we wont cover
    # every sample.
    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            num_parallel_batches=num_cpu_threads,
            drop_remainder=True))
    return d

  return input_fn


def _decode_record(record, name_to_features):
  """Decodes a record to a TensorFlow example."""
  example = tf.parse_single_example(record, name_to_features)

  # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
  # So cast all int64 to int32.
  for name in list(example.keys()):
    t = example[name]
    if t.dtype == tf.int64:
      t = tf.to_int32(t)
    example[name] = t

  return example


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  if not any([FLAGS.do_train, FLAGS.do_eval, FLAGS.do_predict]):
    raise ValueError("At least one of `do_train`, `do_eval` or `do_predict` must be True.")

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  tf.gfile.MakeDirs(FLAGS.output_dir)

  input_files = []
  for input_pattern in FLAGS.input_file.split(","):
    input_files.extend(tf.gfile.Glob(input_pattern))

  tf.logging.info("*** Input Files ***")
  for input_file in input_files:
    tf.logging.info("  %s" % input_file)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=FLAGS.num_train_steps,
      num_warmup_steps=FLAGS.num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu,
      full_vocab_file=FLAGS.full_vocab_file,
      subset_vocab_file=FLAGS.vocab_file
  )

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size)

  if FLAGS.do_train:
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    train_input_fn = input_fn_builder(
        input_files=input_files,
        max_seq_length=FLAGS.max_seq_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        mode="train")
    estimator.train(input_fn=train_input_fn, max_steps=FLAGS.num_train_steps)

  if FLAGS.do_eval:
    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    eval_input_fn = input_fn_builder(
        input_files=input_files,
        max_seq_length=FLAGS.max_seq_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        mode="eval")

    result = estimator.evaluate(input_fn=eval_input_fn, steps=FLAGS.max_eval_steps)

    output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
    with tf.gfile.GFile(output_eval_file, "w") as writer:
      tf.logging.info("***** Eval results *****")
      for key in sorted(result.keys()):
        tf.logging.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))

  if FLAGS.do_predict:
    tf.logging.info("***** Running predict *****")
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    predict_input_fn = input_fn_builder(
      input_files=input_files,
      max_seq_length=FLAGS.max_seq_length,
      max_predictions_per_seq=FLAGS.max_predictions_per_seq,
      mode="predict")

    predict_estimator = tf.estimator.Estimator(
      model_fn=model_fn,
      config=run_config)

    hooks = []
    # hooks.append(tf_debug.LocalCLIDebugHook(ui_type="readline"))
    if FLAGS.full_vocab_file:
      class SaveTransferedWeightsHook(tf.train.SessionRunHook):
        def begin(self):
          self.saver = tf.train.Saver()
        def after_create_session(self, session, coord):
          save_path = FLAGS.init_checkpoint + ".downsampled"
          self.saver.save(session, FLAGS.init_checkpoint + ".downsampled")
          print("Done saving down-sampled weights to %s." % save_path)
        def end(self, session):
          del self.saver
      hooks.append(SaveTransferedWeightsHook())

    result_generator = predict_estimator.predict(
      input_fn=predict_input_fn,
      hooks=hooks
    )
    for result in result_generator:
      print_predict_result(result)
      print("====================")


def print_predict_result(result):
  def convert_ids_to_tokens(ids):
    import tokenization
    if not hasattr(convert_ids_to_tokens, "tokenizer"):
      convert_ids_to_tokens.tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file)
    def id_to_token(id):
      if id == 101: return ""
      elif id == 102: return "//"
      elif id == 103: return "*"
      else: return convert_ids_to_tokens.tokenizer.inv_vocab[id]
    return [id_to_token(x) for x in ids]

  input_ids = result["input_ids"]
  input_mask = result["input_mask"]
  masked_lm_positions = result["masked_lm_positions"]
  masked_lm_ids = result["masked_lm_ids"]
  masked_lm_weights = result["masked_lm_weights"]
  masked_lm_log_probs = result["masked_lm_log_probs"]
  masked_lm_predictions = result["masked_lm_predictions"]
  next_sentence_predictions = result["next_sentence_predictions"]


  ids_len = np.count_nonzero(input_mask)
  masked_ids_len = np.count_nonzero(masked_lm_weights)
  input_ids = input_ids[:ids_len]
  masked_lm_positions = masked_lm_positions[:masked_ids_len]
  masked_lm_ids = masked_lm_ids[:masked_ids_len]
  masked_lm_log_probs = masked_lm_log_probs[:masked_ids_len]
  masked_lm_predictions = masked_lm_predictions[:masked_ids_len]

  tokens = convert_ids_to_tokens(input_ids[:ids_len])
  masked_ids_dict = {masked_lm_positions[i]: masked_lm_ids[i] for i in range(masked_ids_len)}
  predict_ids_dict = {masked_lm_positions[i]: masked_lm_predictions[i] for i in range(masked_ids_len)}
  masked_tokens = convert_ids_to_tokens(masked_lm_ids[:masked_ids_len])
  predict_tokens = convert_ids_to_tokens(masked_lm_predictions[:masked_ids_len])

  GUESS_CNT = 5
  guess = np.argpartition(masked_lm_log_probs, -np.arange(GUESS_CNT))[:, -GUESS_CNT:][:, ::-1]

  j = 0
  for i in range(ids_len):
    if i in masked_lm_positions:
      tokens[i] = convert_ids_to_tokens([masked_ids_dict[i]])[0]
      tokens[i] += "("
      tokens[i] += "%.3f" % -masked_lm_log_probs[j][masked_ids_dict[i]]
      tokens[i] += "|"
      for id in guess[j]:
        tokens[i] += convert_ids_to_tokens([id])[0]
      tokens[i] += ")"

      j += 1

  print("".join(tokens))

if __name__ == "__main__":
  flags.mark_flag_as_required("input_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()
