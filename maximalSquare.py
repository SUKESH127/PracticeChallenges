# Maximal Square
def maximalSquare(self, matrix: List[List[str]]) -> int:
    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0
    maxSquare = 0
    dp = [[0] * (cols + 1) for i in range(rows + 1)]
    
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            if (matrix[i - 1][j - 1] == '1'):
                dp[i][j] = min(dp[i][j - 1], dp[i - 1][j], dp[i - 1][j - 1]) + 1
                maxSquare = max(maxSquare, dp[i][j])
    return maxSquare ** 2
