# Subtree of Another Tree
def isSubtree(self, s, t):
    def checkTree(root1, root2):
        if not root1 or not root2:
            return root1 == root2
        if root1.val != root2.val:
            return False
        return checkTree(root1.left, root2.left) and checkTree(root1.right, root2.right)
    def dfs(s, t):
        if not s:
            return False
        if s.val == t.val and checkTree(s, t):
            return True
        else:
            return dfs(s.left, t) or dfs(s.right, t)
    return dfs(s, t)
