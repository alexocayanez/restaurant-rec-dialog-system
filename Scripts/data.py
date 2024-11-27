import collections
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from utils import create_word_vocabulary

class Data:
    DIALOG_ACTS = ['ack', 'affirm', 'bye', 'confirm', 'deny', 'hello', 'inform', 'negate',
               'null', 'repeat', 'reqalts', 'reqmore', 'request', 'restart', 'thankyou']
        
    DIALOG_ACTS_SET = set(DIALOG_ACTS)
    
    # Load data from relative path, preprocess and creation of lists of tags and utterances.
    def __init__(self, path: str, deduplicated: bool=False) -> None:
        with open(Path(path), 'r') as file:
            self.data = file.read()  
            self.lines = self.data.splitlines() 
        #self.lowercase_lines = [line.lower() for line in self.lines]

        self.splitted_lines = [line.split() for line in self.lines]
        all_tags = [splitted_line[0] for splitted_line in self.splitted_lines]
        all_utterances = [self.lines[k][len(splitted_line[0])+1:] 
                            for k, splitted_line in enumerate(self.splitted_lines)]
        
        tags_and_utterances = zip(all_tags, all_utterances)
        self.deduplicated = deduplicated
        if deduplicated:
            seen = set()
            deduplicated_tags_and_utterances = [(tag, utterance) for tag, utterance in tags_and_utterances if not (utterance in seen or seen.add(utterance))]
            self.tags = [tag_and_utterance[0] for tag_and_utterance in deduplicated_tags_and_utterances]
            self.utterances = [tag_and_utterance[1] for tag_and_utterance in deduplicated_tags_and_utterances]

            # Remove classes with only 1 non duplicated sample.
            number_samples_tags = {tag: 0 for tag in self.DIALOG_ACTS}
            for tag in self.tags:
                number_samples_tags[tag] += 1
            for key, value in number_samples_tags.items():
                if value == 1:
                    print(f"Class {key} only has one non duplicated sample, so it will be deleted")
                    index = self.tags.index(key)
                    self.tags.remove(key)
                    self.utterances.remove(self.utterances[index])
                 
        else:
            self.tags = all_tags
            self.utterances = all_utterances
            
        self.vocabulary = create_word_vocabulary(self.lines)
        pass

    def split(self, stratify: bool=True):
        if stratify and not self.deduplicated:
            splitted = train_test_split(self.utterances, self.tags, test_size=0.15, train_size=0.85, shuffle = True, stratify=self.tags)
        else:
            splitted = train_test_split(self.utterances, self.tags, test_size=0.15, train_size=0.85, shuffle = True)
        self.x_train, self.x_test, self.y_train, self.y_test = splitted
        return splitted


    # DATA ANALYSIS

    def bag_of_words_by_tag(self) -> dict:
        """
        Creates a dictionary that for each dialog tag assigns a list of all words appearing in the tag class.

        :return bag_of_words: Dictionary containing all words by tag.
        """
        bag_of_words = {}
        for act in self.DIALOG_ACTS_SET:
            bag_of_words[act] = []
        for k, tag in enumerate(self.tags):
            bag_of_words[tag].extend(self.utterances[k].split())
        return bag_of_words
    
    def counter_by_tag(self) -> dict:
        """
        Creates a dictionary with a Counter object managing word appearances for each dialog act tag.
        """
        counter_by_tag = {}
        for tag, list_of_words in self.bag_of_words_by_tag().items():
            counter_by_tag[tag] = collections.Counter(list_of_words)
        return counter_by_tag
    

    def ratios_N_most_common_words_by_tag(self, N: int) -> dict:
        ratio_N_most_common_by_tag = {}
        tag_counter = collections.Counter(self.tags)
        word_counter_by_tag = self.counter_by_tag()
        for tag in self.DIALOG_ACTS_SET:
            most_common_words = word_counter_by_tag[tag].most_common(N)
            ratios_by_word = {}
            for k in range(len(most_common_words)):
                ratios_by_word[most_common_words[k][0]] = most_common_words[k][1]/tag_counter[tag]
            ratio_N_most_common_by_tag[tag] = ratios_by_word
        return ratio_N_most_common_by_tag
    
    def plot_ratios_N_most_common_words_by_tag(self, N: int) -> None:
        data = self.ratios_N_most_common_words_by_tag(N)
        sorted_values = sorted(data.values(), reverse=True, key = lambda x: max(x.values()))
        sorted_keys = [list(data.keys())[(list(data.values()).index(value))] for value in sorted_values]

        x = np.arange(0, (N+1)*len(sorted_keys), N+1)
        width = 0.7

        fig, ax = plt.subplots(layout='constrained')
        for k, tag in enumerate(sorted_keys):
            add=0
            for word, ratio in data[tag].items():
                if add == 0:
                    ax.bar(x[k] + add , ratio, width, color='red')
                else:
                    ax.bar(x[k] + add , ratio, width, color='orange')
                ax.text(x[k] + add , ratio-add*0.007, word, size=8)
                add+=1
                
        ax.set_xlabel('Dialog act')
        ax.set_ylabel('Frecuence of appearance')
        ax.set_title(f'Top {N} most common word for each Dialog Act')
        ax.set_xticks(x + width, list(sorted_keys))
        ax.set_ylim(0, 1.05)
        plt.show()
        pass

    # Label distribution
    def label_distribution(self) -> dict:
        label_counts = collections.Counter(self.tags)
        return label_counts
    
    def plot_label_distribution(self):
        label_distribution = self.label_distribution()
        sorted_label_distribution = dict(sorted(label_distribution.items(), key=lambda item: item[1], reverse=True))
        labels, counts = zip(*sorted_label_distribution.items())

        plt.figure(figsize=(10, 5))
        plt.bar(labels, counts)
        plt.xlabel("Dialog Act Labels")
        plt.ylabel("Frequency")
        plt.title("Label Distribution (Sorted)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # Utterance length statistics
    def utterance_length_statistics(self) -> dict:
        utterance_lengths = [len(utterance.split()) for utterance in self.utterances]
        statistics = {
            'min_length': min(utterance_lengths),
            'max_length': max(utterance_lengths),
            'mean_length': sum(utterance_lengths) / len(utterance_lengths),
            'median_length': sorted(utterance_lengths)[len(utterance_lengths) // 2]
        }
        return statistics
    
    def print_utterance_length_statistics(self) -> None:
        utterance_length_stats = self.utterance_length_statistics()
        print("\nUtterance Length Statistics:")
        print(f"Min Length: {utterance_length_stats['min_length']}")
        print(f"Max Length: {utterance_length_stats['max_length']}")
        print(f"Mean Length: {utterance_length_stats['mean_length']}")
        print(f"Median Length: {utterance_length_stats['median_length']}")
        print()

    def plot_utterance_length_statistics(self):
        utterance_length_stats = self.utterance_length_statistics()
        labels = ["Min Length", "Max Length", "Mean Length", "Median Length"]
        stats = [utterance_length_stats["min_length"], utterance_length_stats["max_length"],
                 utterance_length_stats["mean_length"], utterance_length_stats["median_length"]]

        plt.figure(figsize=(10, 5))
        plt.bar(labels, stats)
        plt.ylabel("Length")
        plt.title("Utterance Length Statistics")
        plt.tight_layout()
        plt.show()
        
    def count_out_of_vocab_words(self, vocabulary: set) -> int:
        out_of_vocab_count = 0
        for utterance in self.utterances:
            words = utterance.split()
            for word in words:
                if word not in vocabulary:
                    out_of_vocab_count += 1
        return out_of_vocab_count
        
    def dialog_act_frequency_by_length(self) -> dict:
        dialog_act_lengths = {}
        for tag, utterance in zip(self.tags, self.utterances):
            utterance_length = len(utterance.split())
            if tag not in dialog_act_lengths:
                dialog_act_lengths[tag] = collections.Counter()
            dialog_act_lengths[tag][utterance_length] += 1
        return dialog_act_lengths
    
    def plot_dialog_act_frequency_by_length(self):
        dialog_act_frequency_by_length = self.dialog_act_frequency_by_length()

        lengths = list(dialog_act_frequency_by_length.values())[0].keys()
        dialog_acts = list(dialog_act_frequency_by_length.keys())
        frequencies_by_act = [[] for _ in range(len(dialog_acts))]

        for i, tag in enumerate(dialog_acts):
            length_counter = dialog_act_frequency_by_length[tag]
            frequencies = [length_counter[length] for length in lengths]
            frequencies_by_act[i] = frequencies

        bottom = np.zeros(len(lengths))

        plt.figure(figsize=(12, 6))

        for i, tag in enumerate(dialog_acts):
            plt.bar(lengths, frequencies_by_act[i], label=tag, bottom=bottom)
            bottom += frequencies_by_act[i]

        plt.xlabel("Utterance Length")
        plt.ylabel("Frequency")
        plt.title("Dialog Act Frequency by Length")
        plt.legend(title="Dialog Act Labels", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

    def print_dialog_act_frequency_by_length(self):
        dialog_act_frequency_by_length = self.dialog_act_frequency_by_length()

        for tag, length_counter in dialog_act_frequency_by_length.items():
            sorted_lengths = sorted(length_counter.keys())  # Sort the lengths
            print(f"Dialog Act: {tag}")
            for length in sorted_lengths:
                count = length_counter[length]
                print(f"Utterance Length: {length}, Frequency: {count}")
            print()
        
    def dialog_act_length_statistics(self) -> dict:
        dialog_act_lengths = {}
        for tag, utterance in zip(self.tags, self.utterances):
            utterance_length = len(utterance.split())
            if tag not in dialog_act_lengths:
                dialog_act_lengths[tag] = []
            dialog_act_lengths[tag].append(utterance_length)

        statistics = {}
        for tag, lengths in dialog_act_lengths.items():
            statistics[tag] = {
                'min_length': min(lengths),
                'max_length': max(lengths),
                'mean_length': sum(lengths) / len(lengths),
                'median_length': sorted(lengths)[len(lengths) // 2]
            }
        return statistics
    

    def plot_dialog_act_length_statistics(self):
        """
        Plot statistics for the length (number of words) of user utterances for each dialog act.
        """
        dialog_act_length_statistics = self.dialog_act_length_statistics()

        dialog_acts = list(dialog_act_length_statistics.keys())
        labels = ["Min Length", "Max Length", "Mean Length", "Median Length"]
        num_dialog_acts = len(dialog_acts)

        # Create a list of values for each statistic, grouped by dialog act
        values = [[] for _ in range(num_dialog_acts)]
        for tag in dialog_acts:
            stats = dialog_act_length_statistics[tag]
            values[dialog_acts.index(tag)] = [stats["min_length"], stats["max_length"], stats["mean_length"], stats["median_length"]]

        x = range(num_dialog_acts)
        width = 0.2  # Width of each bar

        plt.figure(figsize=(12, 6))

        for i, label in enumerate(labels):
            y = [value[i] for value in values]
            plt.bar([x_val + i * width for x_val in x], y, width=width, label=label)

        plt.xlabel("Dialog Act Labels")
        plt.ylabel("Length")
        plt.title("Dialog Act Length Statistics")
        plt.xticks([x_val + (width * (len(labels) - 1) / 2) for x_val in x], dialog_acts)
        plt.legend(title="Statistics", loc="upper right")
        plt.tight_layout()
        plt.show()
    
def main():
    PATH_DIALOG_ACTS = """../Data/dialog_acts.dat"""
    data = Data(PATH_DIALOG_ACTS)
    data.plot_ratios_N_most_common_words_by_tag(10)

    # Label distribution
    data.plot_label_distribution()
    label_distribution = data.label_distribution()
    print("Label Distribution:")
    for label, count in label_distribution.items():
        print(f"{label}: {count}")
    # Utterance length statistics
    data.print_utterance_length_statistics()
    data.plot_utterance_length_statistics()

    # Dialog act frequency by length
    data.print_dialog_act_frequency_by_length()
    data.plot_dialog_act_frequency_by_length()

    # Dialog act length statistics
    data.plot_dialog_act_length_statistics()

    # Dialog act length statistics
    dialog_act_length_statistics = data.dialog_act_length_statistics()
    print("Dialog Act Length Statistics:")
    for tag, stats in dialog_act_length_statistics.items():
        print(f"Dialog Act: {tag}")
        print(f"Min Length: {stats['min_length']}")
        print(f"Max Length: {stats['max_length']}")
        print(f"Mean Length: {stats['mean_length']}")
        print(f"Median Length: {stats['median_length']}")
        print()
    return

if __name__=="__main__":
    main()