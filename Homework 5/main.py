from __future__ import annotations # for type hints in recursive node structure

class Lattice:
    def __init__(self, letter:str, probability:float, previous:Lattice = None) -> None:
        self.letter = letter
        self.probability = probability
        self.previous = previous

    def get_word(self) -> str:
        current = self
        word = ""
        while current is not None:
            word = current.letter + word
            current = current.previous
        return word


class SpellingCorrector:
    def __init__(self, data:object) -> None:
        self.train(data)
    
    def train(self, data:object) -> SpellingCorrector:
        self.emission = {}
        for correct, typos in data.items():
            max_length = max([len(word) for word in typos+[correct]])
            for typo in typos:
                for letter_c, letter_t in zip(correct.ljust(max_length,'$'), typo.ljust(max_length, '$')):
                    self.emission[letter_c] = self.emission.get(letter_c, {})
                    self.emission[letter_c][letter_c] = self.emission[letter_c].get(letter_c, 0) + 1
                    self.emission[letter_c][letter_t] = self.emission[letter_c].get(letter_t, 0) + 1
        for key, occurences in self.emission.items():
            total_count = sum([*occurences.values()])
            for character, count in occurences.items():
                self.emission[key][character] = count / total_count
        
        self.transition = {}
        for correct in data.keys():
            word = "^"+correct+"$"
            for index in range(len(word) - 1):
                self.transition[word[index]] = self.transition.get(word[index], {})
                self.transition[word[index]][word[index+1]] = self.transition[word[index]].get(word[index+1], 0) + 1
        for key, occurences in self.transition.items():
            total_count = sum([*occurences.values()])
            for character, count in occurences.items():
                self.transition[key][character] = count / total_count
        return self
    
    def correct(self, word:str) -> tuple:
        pend_word = word+"$"
        previous_Lattices = [Lattice("^",1.0)]
        for letter in pend_word:
            new_Lattices = []
            state_max = {}
            for lattice in previous_Lattices:
                if lattice.letter == "$":
                    state_max["$"] = lattice
                    continue
                for state, prob in self.transition[lattice.letter].items():
                    new_lattice_candidate = Lattice(state, lattice.probability*prob, lattice)
                    if state in state_max:
                        if state_max[state].probability < new_lattice_candidate.probability:
                            state_max[state] = new_lattice_candidate
                        else:
                            del new_lattice_candidate
                    else:
                        state_max[state] = new_lattice_candidate
            for state, lattice in state_max.items():
                if self.emission.get(state, {}).get(letter, 0) > 0:
                    lattice.probability *= self.emission[state][letter]
                    new_Lattices.append(lattice)
            previous_Lattices = new_Lattices
        max_lattice = None
        for lattice in previous_Lattices:
            if max_lattice is None:
                max_lattice = lattice
                continue
            if lattice.probability > max_lattice.probability:
                max_lattice = lattice
        return max_lattice.get_word()[1:-1]

def main():

    DATASET_PATH = "./data/"

    from re import compile
    training_file_paths = []
    from os import walk, path
    for subdir, _, file_names in walk(DATASET_PATH):
        for file_name in file_names:
            if path.splitext(file_name)[1] == ".txt":
                training_file_paths.append("{}/{}".format(subdir, file_name))
    regex = compile(r"^.+:(?: \S+)+$")
    training_data = {}
    for training_file_path in training_file_paths:
        with open(training_file_path, errors='ignore') as training_file:
            for line in training_file:
                if not regex.match(line):
                    continue
                correct, typos = line.strip().split(": ")
                training_data[correct] = typos.split(" ")
                    
    model = SpellingCorrector(training_data)
    words = input("").strip().split(" ")
    corrections = []
    for word in words:
        corrections.append(model.correct(word))
    print(" ".join(corrections))

if __name__ == "__main__":
    main()