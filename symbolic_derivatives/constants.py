# Constants defining operators and their precedence

# Act on value before and value after them, in order of decreasing precendence
BINARY_OPERATOR_LIST = [['^'], ['*', '/'], ['+', '-']]
BINARY_OPERATORS = []
for li in BINARY_OPERATOR_LIST:
    BINARY_OPERATORS += li

COMMUTING_OPERATORS = ['+', '*']

# Act on value after them, all equal precedence
# (rightmost symbol in expr has highest precendence)
UNARY_OPERATORS = ['sin', 'cos', 'tan', 'log', 'exp', 'NEG']