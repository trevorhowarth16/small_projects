from parse import parse_string

expressions = [
    '1',
    'x',
    'x + y',
    'x + 2x + 3',
    '5 * x',
    'x / 2',
    'x^2',
    'y^x',
    '1 / x',
    'x ^ 2 / x',
    'sin(1 / x)',
    'log(x ^ 2)',
    'sin(tan(x)) ^ (3 / (cos(x)))',
    '1 / cos(x)'
]
for expr in expressions:
    print('Original expression:')
    print(expr)
    tree = parse_string(expr, True)
    print('Parsed and simplified expression:')
    print(tree.print_string())
    print('Expression Tree:')
    tree.print_tree()
    derivative_tree = tree.take_derivative('x').simplify()

    print("d/dx Expression:")
    print(derivative_tree.print_string())
    print("d/dx Expression Tree:")
    derivative_tree.print_tree()
    print('')