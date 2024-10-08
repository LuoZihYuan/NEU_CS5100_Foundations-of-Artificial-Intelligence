from __future__ import annotations # for type hints in recursive node structure
from Levenshtein import distance

single_char_apart: dict[str, set[str]] = {}

class WordPathNode:
  def __init__(self, word: str, parent: WordPathNode = None):
    self.word = word
    self.parent = parent
    if parent is None:
      self.depth = 0
    else:  
      self.depth = parent.depth + 1

  # Two WordPathNodes are equal if they both have the same word
  def __eq__(self, other: WordPathNode) -> bool:
    return self.word == other.word

  # The hashcode is the hashcode of the word
  def __hash__(self) -> int:
    return self.word.__hash__()

  # Returns the path from the current node to the root.
  # as a list of str
  def get_path_to_root(self) -> list[str]:
    current = self
    list_of_words = []
    while current is not None:
      list_of_words.append(current.word)
      current = current.parent
    return list_of_words

def find_edit_path_bfs(start_word: str, end_word: str) -> list[str]:
  history: set[WordPathNode] = set()
  frontier = [WordPathNode(start_word, None)]
  while len(frontier) > 0:
    current = frontier.pop(0)
    history.add(current)
    if current.word == end_word:
      return current.get_path_to_root()
    for next_word in single_char_apart[current.word]:
      nxt = WordPathNode(next_word, current)
      if nxt not in history:
        frontier.append(nxt)
  return None

def find_edit_path_dfs(start_word: str, end_word: str, limit: int=0) -> list[str]:
  history: set[WordPathNode] = set()
  frontier = [WordPathNode(start_word, None)]
  while len(frontier) > 0:
    current = frontier.pop()
    history.add(current)
    if current.word == end_word:
      return current.get_path_to_root()
    if limit and current.depth == limit:
      continue
    for next_word in single_char_apart[current.word]:
      nxt = WordPathNode(next_word, current)
      if nxt not in history:
        frontier.append(nxt)
  return None

def find_edit_path_iterative_deepening(start_word: str, end_word: str) -> list[str]:
  i = 1
  while not (result:=find_edit_path_dfs(start_word, end_word, i)):
    i+=1
  return result

def find_edit_path_A_star_search(start_word: str, end_word: str):
  history: set[WordPathNode] = set()
  frontier = [WordPathNode(start_word, None)]
  while len(frontier) > 0:
    min_val:int = None
    min_index: int = -1
    for i, f in enumerate(frontier):
      node_val = f.depth + distance(f.word, end_word)
      if not min_val:
        min_val = node_val
      if min_val > node_val:
        min_val = node_val
        min_index = i
    current = frontier.pop(min_index)
    print(current.word)
    history.add(current)
    if current.word == end_word:
      return current.get_path_to_root()
    for next_word in single_char_apart[current.word]:
      nxt = WordPathNode(next_word, current)
      if nxt not in history:
        frontier.append(nxt)
  return None

def hamming_distance(word1: str, word2: str) -> int:
  return sum(c1 != c2 for c1, c2 in zip(word1, word2))

def main():
  # Get start and end words from user
  start_word = input('Please enter the start word: ').strip()
  SIZE = len(start_word)
  end_word = input('Please enter the end word: ').strip()
  while len(end_word) != SIZE:
    end_word = input(f'Please enter an end word of length {SIZE}: ').strip()
  
  # Get a dictionary, maybe https://github.com/dwyl/english-words/blob/master/words_alpha.txt
  dictionary = [word.strip() for word in open('./data/words_alpha.txt', 'r').readlines()]

  # Limit dictionary to words of certain size
  dictionary_small = [word for word in dictionary if len(word) == SIZE]
  # print(len(dictionary_small))

  # Graph adjacency list
  for idx, word1 in enumerate(dictionary_small):
    single_char_apart[word1] = set()
    for word2 in dictionary_small[:idx]:
      if distance(word1, word2, score_cutoff=1) <= 1:
        single_char_apart[word1].add(word2)
        single_char_apart[word2].add(word1)
  
  print(find_edit_path_iterative_deepening(start_word, end_word))
    
if __name__ == "__main__":
  main()