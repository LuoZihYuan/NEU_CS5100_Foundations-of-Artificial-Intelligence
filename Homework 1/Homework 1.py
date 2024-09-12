import random

class CharNGramLanguageModel:
    def __init__(self, n: int) -> None:
        self.n = n

    def train(self, data: list):
        self.lm = {}
        for row in data:
            if len(row) < self.n + 1:
                continue
            for index in range(len(row) - self.n):
                self.lm[row[index:index+self.n]] = self.lm.get(row[index:index+self.n], {})
                self.lm[row[index:index+self.n]][row[index+self.n]] = self.lm[row[index:index+self.n]].get(row[index+self.n], 0) + 1
        for key, occurences in self.lm.items():
            total_count = sum([*occurences.values()])
            for character, count in occurences.items():
                self.lm[key][character] = count / total_count

    def generate_character(self, prompt: str) -> str:
        probability = self.lm[prompt[-self.n:]]
        return random.choices([*probability.keys()], [*probability.values()])[0]
    
    def generate(self, prompt: str) -> str:
        next = self.generate_character(prompt)
        while next != "\n":
            prompt += next
            next = self.generate_character(prompt)
        prompt += next
        return prompt

class WordNGramLanguageModel:
    def __init__(self, n: int, bidirection: bool = False) -> None:
        self.n = n
        self.bidirection = bidirection

    def train(self, data: list):
        self.lm = {}
        for row in data:
            words = row.split() + ["\n"]
            reversed_words = row.split()[::-1] + ["\n"]
            if len(words) < self.n + 1:
                continue
            for index in range(len(words) - self.n):
                self.lm[" ".join(words[index:index+self.n])] = self.lm.get(" ".join(words[index:index+self.n]), {})
                self.lm[" ".join(words[index:index+self.n])][words[index+self.n]] = self.lm[" ".join(words[index:index+self.n])].get(words[index+self.n], 0) + 1
                if not self.bidirection:
                    continue
                print("here")
                self.lm[" ".join(reversed_words[index:index+self.n])] = self.lm.get(" ".join(reversed_words[index:index+self.n]), {})
                self.lm[" ".join(reversed_words[index:index+self.n])][reversed_words[index+self.n]] = self.lm[" ".join(reversed_words[index:index+self.n])].get(reversed_words[index+self.n], 0) + 1
            
        # print(self.lm)
        for key, occurences in self.lm.items():
            total_count = sum([*occurences.values()])
            for word, count in occurences.items():
                self.lm[key][word] = count / total_count
        # print(self.lm)

    def generate_word(self, prompt: str) -> str:
        words = prompt.split()
        probability = self.lm[" ".join(words[-self.n:])]
        return random.choices([*probability.keys()], [*probability.values()])[0]
    
    def generate(self, prompt: str) -> str:
        next = self.generate_word(prompt)
        while next != "\n":
            prompt += (" " + next)
            next = self.generate_word(prompt)
        prompt += (" " + next)
        return prompt

def main():
    training_file_paths = []
    from os import walk, path
    for subdir, _, file_names in walk("./data"):
        for file_name in file_names:
            if path.splitext(file_name)[1] == ".txt":
                training_file_paths.append("{}/{}".format(subdir, file_name))
    
    training_data = []
    for training_file_path in training_file_paths:        
        with open(training_file_path) as training_file:
            training_data += training_file.readlines()
    
    test_char_ns = [3, 5, 20, 50]
    prompt = input("Input prompt: ")
    for test_char_n in test_char_ns:
        model = CharNGramLanguageModel(test_char_n)
        model.train(training_data)
        try:
            result = repr(model.generate(prompt))
        except KeyError:
            result = ""
        print("CharNGram({}): {}".format(test_char_n, result))
    
    test_word_ns = [1, 2, 3, 5]
    test_bidirections = [False, True]
    for test_bidirection in test_bidirections:
        for test_word_n in test_word_ns:
            model = WordNGramLanguageModel(test_word_n)
            model.train(training_data)
            try:
                result = repr(model.generate(prompt))
            except KeyError:
                result = ""
            print("WordNGram({}, {}): {}".format(test_word_n, test_bidirection, result))

if __name__ == "__main__":
    main()