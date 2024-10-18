from abc import ABC, abstractmethod
import random

class LMQLearnInterface(ABC):
    @abstractmethod
    def train(self, data):
        raise NotImplementedError("Subclasses should implement this!")
    
    @abstractmethod
    def _trim_prompt(self, prompt:str):
        raise NotImplementedError("Subclasses should implement this!")

    @abstractmethod
    def generate_character(self):
        raise NotImplementedError("Subclasses should implement this!")

class CharNGramLanguageModel(LMQLearnInterface):
    def __init__(self, n: int, data: list) -> None:
        self.n = n
        self.train(data)
        if n > 1:
            self.gram1 = CharNGramLanguageModel(1, data)

    def train(self, data: list):
        self.freq = {}
        for row in data:
            if len(row) < self.n + 1:
                continue
            for index in range(len(row) - self.n):
                self.freq[row[index:index+self.n]] = self.freq.get(row[index:index+self.n], {})
                self.freq[row[index:index+self.n]][row[index+self.n]] = self.freq[row[index:index+self.n]].get(row[index+self.n], 0) + 1
        self._update_lm()
        return self
    
    def _fix_negative(self, nums: list):
        # from scipy.special import softmax
        # return softmax(nums)
        minimum = min(nums)
        return [(num - minimum + 1) for num in nums]

    def _update_lm(self):
        self.lm = {}
        for key, occurences in self.freq.items():
            total_count = sum([*occurences.values()])
            for character, count in occurences.items():
                self.lm[key] = self.lm.get(key, {})
                self.lm[key][character] = count / total_count
        return self

    def _trim_prompt(self, prompt: str):
        return prompt[-self.n:]

    def generate_character(self, prompt: str) -> str:
        try:
            probability = self.lm[self._trim_prompt(prompt)]
            characters = [*probability.keys()]
            weights = [*probability.values()]
            if min(weights) < 0:
                weights = self._fix_negative(weights)
            self._used = self  # `_used` points to the current used model: self
            return random.choices(characters, weights)[0]
        except KeyError:
            # fall back to using 1 gram model when can't find matches in n gram model
            self._used = self.gram1  # `_used` points to the current used model: gram1
            return self.gram1.generate_character(prompt)
    
    def generate(self, prompt: str) -> str:
        next = self.generate_character(prompt)
        while next != "\n" and next != '—':
            prompt += next
            next = self.generate_character(prompt)
        prompt += next
        return prompt

class ReinforcementLearning:

  def __init__(self, model: CharNGramLanguageModel, alpha = 0.7, gamma = 0.99):
    self.model = model
    self.alpha = alpha
    self.gamma = gamma

  # Ask the user for num_prompts number of prompts. Perform Q learning
  # to update weights in provided char ngram model. The criteria argument must be
  # a function which takes a string and returns a numerical score, which will be
  # used as the reward for Q learning.
  def Q_learn(self, criteria:callable, num_prompts:int = 1, iterations_per_prompt:int = 10000):
    for _ in range(num_prompts):
        learner_prompt = input("Input learner prompt: ")
        for _ in range(iterations_per_prompt):
            new_char = self.model.generate_character(learner_prompt)
            old_trimmed_prompt = self.model._used._trim_prompt(learner_prompt)
            learner_prompt += new_char
            new_trimmed_prompt = self.model._used._trim_prompt(learner_prompt)
            reward = criteria(learner_prompt)
            self.model._used.lm[old_trimmed_prompt][new_char] = (1 - self.alpha) * self.model._used.lm[old_trimmed_prompt][new_char] \
                                                                  + self.alpha * (reward \
                                                                                  + self.gamma * max(self.model._used.lm.get(new_trimmed_prompt, {"":0}).values()))
            if new_char == '\n' or new_char == '—':
                break

def decrease_vowels(sentence: str):
    return -1000 * (count_vowels(sentence) / len(sentence))**4

def count_vowels(sentence: str):
    return sum([sentence.lower().count(vowel) for vowel in ["a", "e", "i", "o", "u"]])

def main():

    DATASET_PATH = "./data/Olivia_Rodrigo_Songs"

    training_file_paths = []
    from os import walk, path
    for subdir, _, file_names in walk(DATASET_PATH):
        for file_name in file_names:
            if path.splitext(file_name)[1] == ".txt":
                training_file_paths.append("{}/{}".format(subdir, file_name))

    training_data = []
    for training_file_path in training_file_paths:
        with open(training_file_path, errors='ignore') as training_file:
            training_data += training_file.readlines()

    test_char_ns = [3, 5, 20, 50]
    prompt = input("Input prompt: ")

    from textwrap import dedent
    from tqdm import tqdm

    for test_char_n in test_char_ns:
        model = CharNGramLanguageModel(test_char_n, training_data)
        test_lengths = [len(model.generate(prompt)) for _ in tqdm(range(1000))]
        avg_lengths = sum(test_lengths) / len(test_lengths)
        print(dedent("""
                     Char{}Gram(Before):
                     - Average Length: {}
                     - Sample Output: {}
                     """.format(test_char_n, avg_lengths, repr(model.generate(prompt)))
        ))
        ReinforcementLearning(model).Q_learn(lambda x: -len(x))
        test_lengths = [len(model.generate(prompt)) for _ in tqdm(range(1000))]
        avg_lengths = sum(test_lengths) / len(test_lengths)
        print(dedent("""
                     Char{}Gram(After):
                     - Average Length: {}
                     - Sample Output: {}
                     """.format(test_char_n, avg_lengths, repr(model.generate(prompt)))
        ))


    for test_char_n in test_char_ns:
        model = CharNGramLanguageModel(test_char_n, training_data)
        test_numbers = []
        test_lengths = []
        for _ in tqdm(range(1000)):
            sentence = model.generate(prompt)
            test_numbers.append(count_vowels(sentence))
            test_lengths.append(len(sentence))
        avg_vowels = sum(test_numbers) / len(test_numbers)
        avg_lengths = sum(test_lengths) / len(test_lengths)
        print(dedent("""
                     Char{}Gram(Before):
                     - Average #Vowels: {}
                     - Average Length: {}
                     - Sample Output: {}
                     """.format(test_char_n, avg_vowels, avg_lengths, repr(model.generate(prompt)))
        ))
        ReinforcementLearning(model).Q_learn(lambda x: decrease_vowels(x))
        test_numbers = []
        test_lengths = []
        for _ in tqdm(range(1000)):
            sentence = model.generate(prompt)
            test_numbers.append(count_vowels(sentence))
            test_lengths.append(len(sentence))
        avg_vowels = sum(test_numbers) / len(test_numbers)
        avg_lengths = sum(test_lengths) / len(test_lengths)
        print(dedent("""
                     Char{}Gram(After):
                     - Average #Vowels: {}
                     - Average Length: {}
                     - Sample Output: {}
                     """.format(test_char_n, avg_vowels, avg_lengths, repr(model.generate(prompt)))
        ))


if __name__ == "__main__":
    main()