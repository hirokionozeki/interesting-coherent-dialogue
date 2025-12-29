import json
from collections import defaultdict

TOKEN_NOCHOSEN = 'no_passages_used'
TOKEN_KNOWLEDGE = '__knowledge__'
TOKEN_END_KNOWLEDGE = '__endknowledge__'
TOKEN_LABEL = '__label__'
TOKEN_END_LABEL = '__endlabel__'


def _first_val(dictionary):
    vals = list(dictionary.values())
    if len(vals) > 0:
        return vals[0]
    return ''


def _first_key(dictionary):
    keys = list(dictionary.keys())
    if len(keys) > 0:
        return keys[0]
    return ''


def _get_chosen_title_and_sent(wizard_entry, k_dict):
    """
    Return a nicely extracted title and chosen sentence.
    :return: pair (title, sentence)
    """
    title_dict = wizard_entry.get('checked_passage', 'none')
    sentence_dict = wizard_entry.get('checked_sentence', {})
    title = None
    sentence = None
    if sentence_dict == {}:
        title = sentence = TOKEN_NOCHOSEN
    else:
        sentence = _first_val(sentence_dict)
        if sentence == TOKEN_NOCHOSEN:
            title = TOKEN_NOCHOSEN
        else:
            title = ''
            # cand_title1 is the title from the `checked_passage`
            cand_title1 = _first_val(title_dict) if title_dict else ''
            # cand_title2 is the extracted title of the passage from the
            #   sentence dict, which is e.g. `self_Vermont_Syrup_0`
            cand_title2 = ' '.join(_first_key(sentence_dict).split('_')[1:-1])
            if (
                    cand_title1
                    and cand_title1 in k_dict
                    and sentence in k_dict[cand_title1]
            ):
                title = cand_title1
            elif cand_title2 in k_dict and sentence in k_dict[cand_title2]:
                title = cand_title2
            else:  # neither candidate title is the right one
                for t, passage in k_dict.items():
                    if sentence in passage:
                        title = t
                        break

    return title, sentence


class WizardDialogKnowledgeTeacher:
    def __init__(self, raw_data, datatype='test'):
        self.raw_data = raw_data
        self._init_attributes({})
        self.datatype = datatype

    def _init_attributes(self, opt):
        """
        Initialize teacher attributes.
        """
        self.add_missing_turns = opt.get('add_missing_turns', 'train')
        self.label_type = opt.get('label_type', 'response')
        self.include_knowledge = opt.get('include_knowledge', True)
        self.include_checked_sentence = opt.get('include_checked_sentence', True)
        self.knowledge_separator = opt.get('include_knowledge_separator', False)
        self.chosen_topic_delimiter = opt.get('chosen_topic_delimiter', '\n')
        self.title_cache = {}
        self.sent_cache = {}

    def len_episode(self, ep):
        d = self.raw_data[ep]
        wizard_first = 'Wizard' in d['dialog'][0]['speaker']
        if wizard_first:
            if self.add_missing_turns == 'none':
                len_ep = (len(d['dialog']) - 1) // 2
            elif self.add_missing_turns == 'train' and self.datatype != 'train':
                len_ep = (len(d['dialog']) - 1) // 2
            else:
                len_ep = (len(d['dialog']) - 1) // 2 + 1
            return len_ep
        return len(d['dialog']) // 2

    def _format_example(self, episode_idx, entry_idx=0):
        d = self.raw_data[episode_idx]
        episode_done = entry_idx == (self.len_episode(episode_idx) - 1)

        wizard_first = 'Wizard' in d['dialog'][0]['speaker']
        idx = entry_idx * 2 if wizard_first else (entry_idx * 2) + 1

        if idx >= len(d['dialog']):
            return None

        # first, get knowledge
        apprentice_ret_passages = wizard_ret_passages = {}

        if not wizard_first or idx != 0:
            apprentice_entry = d['dialog'][idx - 1]
            apprentice_ret_passages = apprentice_entry['retrieved_passages']
        if idx - 2 >= 0:
            wizard_prev_entry = d['dialog'][idx - 2]
            wizard_ret_passages = wizard_prev_entry['retrieved_passages']

        chosen_topic = d.get('chosen_topic', '')
        chosen_topic_passages = d['chosen_topic_passage']
        chosen_topic = d.get('chosen_topic', '')

        knowledge_dict = {chosen_topic: chosen_topic_passages}
        for ret_passes in [apprentice_ret_passages, wizard_ret_passages]:
            for passage in ret_passes:
                for k, v in passage.items():
                    if k not in knowledge_dict.keys():
                        knowledge_dict[k] = v

        # then, get text
        if idx == 0:
            # first message - only have the chosen topic
            text = chosen_topic
        elif idx == 1:
            # first response - only have the first message
            text = (
                f"{chosen_topic}{self.chosen_topic_delimiter}{apprentice_entry['text']}"
            )
        else:
            text = ''
            if self.label_type == 'chosen_sent':
                # if chosen_sent, add wizard response to dialog history
                text += '{}\n'.format(wizard_prev_entry['text'])
            text += apprentice_entry['text']

        # next, get label
        wizard_entry = d['dialog'][idx]
        if self.label_type == 'response':
            labels = [wizard_entry['text']]
        else:
            title, sentence = _get_chosen_title_and_sent(wizard_entry, knowledge_dict)
            if self.knowledge_separator and title != TOKEN_NOCHOSEN:
                labels = ['{} {} {}'.format(title, TOKEN_KNOWLEDGE, sentence)]
            else:
                labels = ['{} {}'.format(title, sentence)]

        # finally, get label_candidates
        label_cands = ['{} {}'.format(TOKEN_NOCHOSEN, TOKEN_NOCHOSEN)]
        knowledge_str = defaultdict(list)
        for title, passage in knowledge_dict.items():
            for p in passage:
                if self.knowledge_separator:
                    cand = '{} {} {}'.format(title, TOKEN_KNOWLEDGE, p)
                else:
                    cand = '{} {}'.format(title, p)
                knowledge_str[title].append(p)
                label_cands.append(cand)
        if self.label_type == 'response':
            if 'train' in self.datatype:
                label_cands = []
            else:
                label_cands = wizard_entry.get('candidate_responses', [])

        dialog_context = []
        for bef in range(idx):
            if f'{episode_idx}_{bef}' in self.title_cache:
                dialog_context.append({'speaker': d['dialog'][bef]['speaker'],
                                       'text': d['dialog'][bef]['text'],
                                       'title': self.title_cache[f'{episode_idx}_{bef}'],
                                       'checked_sentence': self.sent_cache[f'{episode_idx}_{bef}']})
            else:
                dialog_context.append({'speaker': d['dialog'][bef]['speaker'], 'text': d['dialog'][bef]['text']})

        action = dict(
            {
                'id': 'WizardDialogKnowledgeTeacher',
                'text': text,
                'labels': labels,
                'chosen_topic': chosen_topic,
                'episode_done': episode_done,
                'label_candidates': label_cands,
                'context': dialog_context
            }
        )

        action['knowledge'] = knowledge_str
        title, sentence = _get_chosen_title_and_sent(wizard_entry, knowledge_dict)
        action['title'] = title
        action['checked_sentence'] = sentence

        self.title_cache[f'{episode_idx}_{idx}'] = title
        self.sent_cache[f'{episode_idx}_{idx}'] = sentence

        return action

    def example(self, episode_idx, entry_idx=0):
        return self._format_example(episode_idx, entry_idx)


def process_data(input_file, output_file):
    """Process Wizard of Wikipedia raw data."""
    print(f"Processing {input_file} -> {output_file}")
    raw_data = json.load(open(input_file))
    agent = WizardDialogKnowledgeTeacher(raw_data, datatype='test')

    print(f"Number of dialogs: {len(raw_data)}")
    saved = []
    for i in range(len(raw_data)):
        parent = None
        for j in range(100):
            example = agent.example(i, j)
            if example is None:
                break
            if example['title'] != 'no_passages_used' and example['checked_sentence'] not in example['knowledge'][example['title']]:
                print(f"Warning: sentence not in knowledge for dialog {i}, turn {j}")
            example['parent'] = parent
            example['dialog_id'] = i
            example['turn_id'] = j
            saved.append(example)
            parent = len(saved) - 1
            if example['episode_done']:
                break

    print(f"Total examples: {len(saved)}")
    json.dump(saved, open(output_file, 'w'))
    print(f"Saved to {output_file}")


def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Process Wizard of Wikipedia data')
    parser.add_argument('--split', type=str, choices=['seen', 'unseen', 'both'], default='both',
                        help='Which split to process')
    parser.add_argument('--raw-dir', type=str, default='../../../data/raw',
                        help='Directory containing raw data')
    parser.add_argument('--output-dir', type=str, default='../../../data/genks/wizard',
                        help='Directory to save processed data')
    args = parser.parse_args()

    # Get absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.abspath(os.path.join(script_dir, args.raw_dir))
    output_dir = os.path.abspath(os.path.join(script_dir, args.output_dir))

    os.makedirs(output_dir, exist_ok=True)

    if args.split in ['seen', 'both']:
        input_file = os.path.join(raw_dir, 'test_random_split.json')
        output_file = os.path.join(output_dir, 'seen_full.json')
        process_data(input_file, output_file)

    if args.split in ['unseen', 'both']:
        input_file = os.path.join(raw_dir, 'test_topic_split.json')
        output_file = os.path.join(output_dir, 'unseen_full.json')
        process_data(input_file, output_file)


if __name__ == '__main__':
    main()
