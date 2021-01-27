# Search Suggestions System
class TrieNode:
    def __init__(self):
        self.children = dict()
        self.words = list()
        self.n = 0
class Trie:
    def __init__(self):
        self.root = TrieNode()
    def add(self, word):
        node = self.root
        for c in word:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
            if node.n < 3:
                node.words.append(word)
                node.n += 1
    def findWord(self, prefix):
        node = self.root
        for c in prefix:
            if c not in node.children:
                return ''
            node = node.children[c]
        return node.words
def suggestedProducts(self, A, searchWord):
    A.sort()
    trie = Trie()
    for word in A:
        trie.add(word)
    ans, cur = [], ''
    for c in searchWord:
        cur += c
        ans.append(trie.findWord(cur))
    return ans
