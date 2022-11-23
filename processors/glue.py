""" GLUE processors and helpers """

import logging
import os
import torch
from .utils import DataProcessor, InputExample, InputFeatures
from model import tokenization_albert

logger = logging.getLogger(__name__)


def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """

    all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_tag_ids, all_def_ids, all_tag_label, all_labels = map(torch.stack, zip(*batch))
    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    all_token_type_ids = all_token_type_ids[:, :max_len]
    all_tag_ids = all_tag_ids[:, :max_len]
    all_def_ids = all_def_ids[:, :max_len]
    all_tag_label = all_tag_label[:, :max_len]

    return all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_tag_ids, all_def_ids, all_tag_label, all_labels


def glue_convert_examples_to_features(examples, tokenizer, num_labels,
                                      max_seq_length=512,
                                      task=None,
                                      label_list=None,
                                      output_mode=None):
    """
    Loads a data file into a list of ``InputFeatures``
    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    if task is not None:
        processor = glue_processors[task]()
        if label_list is None:
            label_list = processor.get_labels(num_labels=num_labels)
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = glue_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    max_len = 20
    writer = open('token_market.txt', 'w')
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % (ex_index))
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_a_tag, tokens_a_def, tag_a_label = tokenizer.extend_tag_mask_label(example.text_a, example.text_a_tag, \
                                                        example.text_a_def, \
                                                        example.text_a_tag_label)

        # For text b
        tokens_b = tokenizer.tokenize(example.text_b)
        # add tag word and tag definition
        tokens_b_tag, tokens_b_def, tag_b_label = tokenizer.extend_tag_mask_label(example.text_b, example.text_b_tag, \
                                                                  example.text_b_def, \
                                                                  example.text_b_tag_label)

        # _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        # _truncate_seq_pair(tokens_a_tag, tokens_b_tag, max_seq_length - 3)
        # _truncate_seq_pair(tokens_a_def, tokens_b_def, max_seq_length - 3)
        # _truncate_seq_pair(tag_a_label, tag_b_label, max_seq_length - 3)

        input_a_tag, input_a_def = tokenizer.convert_tag_def_by_vocab(tokens_a_tag, tokens_a_def, max_len)
        input_b_tag, input_b_def = tokenizer.convert_tag_def_by_vocab(tokens_b_tag, tokens_b_def, max_len)
        # END

        # 原本放入bert的纯文本信息处理
        # for text a
        tokens = []
        token_type_ids = []
        # tag_mask = []
        tag_label = []
        tag_ids = []
        def_ids = []

        tokens.append("[CLS]")
        token_type_ids.append(0)

        tag_ids.append([0, 0, 0, 0])
        def_ids.append([[0]*max_len, [0]*max_len, [0]*max_len, [0]*max_len])

        # tag_mask.append([0, 0, 0, 0])
        tag_label.append(5)
        tokens_a = tokens_a[:120]
        for idx, token in enumerate(tokens_a):
            tokens.append(token)
            token_type_ids.append(0)

            # tag_mask.append(tag_a_mask[idx])
            tag_label.append(tag_a_label[idx])

            tag_ids.append(input_a_tag[idx])
            def_ids.append(input_a_def[idx])

        tokens.append("[SEP]")
        token_type_ids.append(0)

        tag_ids.append([0, 0, 0, 0])
        def_ids.append([[0]*max_len, [0]*max_len, [0]*max_len, [0]*max_len])

        # tag_mask.append([0, 0, 0, 0])
        tag_label.append(5)

        # for text b
        tokens_b = tokens_b[:10]
        for idx, token in enumerate(tokens_b):
            tokens.append(token)
            token_type_ids.append(1)

            # tag_mask.append(tag_b_mask[idx])
            tag_label.append(tag_b_label[idx])

            tag_ids.append(input_b_tag[idx])
            def_ids.append(input_b_def[idx])

        tokens.append("[SEP]")
        token_type_ids.append(1)

        print(tokens)
        try:
            a = tokens.index('▁market')
            print(a)
        except:
            a = tokens.index('▁markets')
            print(a)
        writer.write(' '.join(tokens) + '\t' + str(a) + '\n')

        tag_ids.append([0, 0, 0, 0])
        def_ids.append([[0]*max_len, [0]*max_len, [0]*max_len, [0]*max_len])

        # tag_mask.append([0, 0, 0, 0])
        tag_label.append(5)
        # for tag

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1] * len(input_ids)
        input_len = len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            attention_mask.append(0)
            token_type_ids.append(0)

            tag_ids.append([0, 0, 0, 0])
            def_ids.append([[0]*max_len, [0]*max_len, [0]*max_len, [0]*max_len])

            # tag_mask.append([0, 0, 0, 0])
            tag_label.append(5)

        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(token_type_ids) == max_seq_length
        # assert len(tag_mask) == max_seq_length
        assert len(tag_label) == max_seq_length
        assert len(tag_ids) == max_seq_length
        assert len(def_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))
            logger.info("input length: %d" % (input_len))

        features.append(
            InputFeatures(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        label=label_id,
                        input_len=input_len,
                        input_def_ids=def_ids,
                        input_tag_ids=tag_ids,
                        input_tag_label=tag_label))
    writer.close()
    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()

class WnliProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_def_dict(self, data_dir):
        return self._read_dict(os.path.join(data_dir, "definition.txt"))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), self._read_dict(os.path.join(data_dir, "definition.txt")), self._read_tsv(os.path.join(data_dir, "train_label.tsv")),  "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), self._read_dict(os.path.join(data_dir, "definition.txt")), self._read_tsv(os.path.join(data_dir, "dev_label.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), self._read_dict(os.path.join(data_dir, "definition.txt")), self._read_tsv(os.path.join(data_dir, "test_label.tsv")), "test")

    def get_labels(self, num_labels):
        """See base class."""
        if num_labels == 2:
            return ["0", "1"]
        elif num_labels == 3:
            return ["0", "1", "2"]

    def  split_word_tag(self, line, def_dict):

        temp_tag_mask, temp_tag, temp_def = [], [], []

        # temp_tag_label = []
        pure_word_line = ' '.join([item.split('|')[0] for item in line.strip().split()])

        for idx, item in enumerate(line.strip().split()):
            if '%' in item and "|" in item and ':' in item:
                temp = item.split('|')
                sense_len = len(temp) - 1
                # temp_tag_label.append(int(line_label.split()[idx]))
                if sense_len == 1:

                    # temp_tag_mask.append([1, 0, 0, 0])
                    temp_tag.append([temp[1], temp[0], temp[0], temp[0]])
                    temp_def.append([def_dict[temp[1]], pure_word_line, pure_word_line, pure_word_line])

                elif sense_len == 2:

                    # temp_tag_mask.append([1, 1, 0, 0])
                    temp_tag.append([temp[1], temp[2], temp[0], temp[0]])
                    temp_def.append([def_dict[temp[1]], def_dict[temp[2]], pure_word_line, pure_word_line])

                elif sense_len == 3:

                    # temp_tag_mask.append([1, 1, 1, 0])
                    temp_tag.append([temp[1], temp[2], temp[3], temp[0]])
                    temp_def.append([def_dict[temp[1]], def_dict[temp[2]], def_dict[temp[3]], pure_word_line])

                elif sense_len == 4:

                    # temp_tag_mask.append([1, 1, 1, 1])
                    temp_tag.append([temp[1], temp[2], temp[3], temp[4]])
                    temp_def.append([def_dict[temp[1]], def_dict[temp[2]], def_dict[temp[3]], def_dict[temp[4]]])

            else:

                # temp_tag_mask.append([0, 0, 0, 0])
                temp_tag.append([item, item, item, item])
                temp_def.append([pure_word_line, pure_word_line, pure_word_line, pure_word_line])
                # temp_tag_label.append(5)

        return pure_word_line, temp_tag, temp_def

    def _create_examples(self, lines, def_dict, label_lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            text_a, text_a_tag, \
            text_a_def = self.split_word_tag(line[1], def_dict)

            text_b, text_b_tag, \
            text_b_def = self.split_word_tag(line[2], def_dict)


            label = line[-1]
            examples.append(
                InputExample(guid=guid, \
                            text_a=text_a, text_b=text_b, \
                            text_a_tag=text_a_tag, text_b_tag=text_b_tag, \
                            text_a_def=text_a_def, text_b_def=text_b_def, \
                            # text_a_tag_mask=text_a_tag_mask, text_b_tag_mask=text_b_tag_mask, \
                            text_a_tag_label=list(map(int, label_lines[i][0].split())), text_b_tag_label=list(map(int, label_lines[i][1].split())), \
                            label=label))
        return examples


glue_tasks_num_labels = {
    "mnli": 3,
    "mrpc": 2,
    "sst-2": 2,
    "sts-b": 1,
    "qqp": 2,
    "qnli": 2,
    "rte": 2,
    'lcqmc': 2,
    "xnli": 3,
}

glue_processors = {
    # "cola": ColaProcessor,
    # "mnli": MnliProcessor,
    # "mnli-mm": MnliMismatchedProcessor,
    # "mrpc": MrpcProcessor,
    # "sst-2": Sst2Processor,
    # "sts-b": StsbProcessor,
    # "qqp": QqpProcessor,
    # "qnli": QnliProcessor,
    # "rte": RteProcessor,
    # 'lcqmc': LcqmcProcessor,
    "wnli": WnliProcessor,
}

glue_output_modes = {
    "cola": "classification",
    "mnli": "classification",
    "mnli-mm": "classification",
    "mrpc": "classification",
    "sst-2": "classification",
    "sts-b": "regression",
    "qqp": "classification",
    "qnli": "classification",
    "rte": "classification",
    "wnli": "classification",
    'lcqmc': "classification",
}
