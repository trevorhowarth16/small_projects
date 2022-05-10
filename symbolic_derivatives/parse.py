
from backend import (
    CosNode, ExpNode, LogNode, NumberNode,
    ProductNode, SinNode, SumNode, TanNode,
    VariableNode
)
from constants import BINARY_OPERATORS, BINARY_OPERATOR_LIST, UNARY_OPERATORS
from exceptions import ParseError


def MakeAddNode(first, second):
    return SumNode([first, second], [1, 1])


def MakeSubtractNode(first, second):
    return SumNode([first, second], [1, -1])


def MakeMultiplyNode(first, second):
    return ProductNode([first, second], [1, 1])


def MakeDivideNode(first, second):
    return ProductNode([first, second], [1, -1])


def MakePowerNode(first, second):
    if isinstance(second, NumberNode):
        return ProductNode([first], [second.value])
    else:
        return ExpNode(
            ProductNode(
                [LogNode(first),
                 second],
                [1,
                 1]
            )
        )


def MakeNegNode(child):
    return SumNode(
        [child],
        [-1]
    )


unary_nodes = {
    'sin': SinNode,
    'cos': CosNode,
    'tan': TanNode,
    'log': LogNode,
    'exp': ExpNode,
    'NEG': MakeNegNode
}

binary_nodes = {
    '+': MakeAddNode,
    '-': MakeSubtractNode,
    '*': MakeMultiplyNode,
    '/': MakeDivideNode,
    '^': MakePowerNode
}


def clean(in_str):
    return in_str.replace(' ', '')


def test_buffer(buffer):
    if buffer in UNARY_OPERATORS:
        return True, [buffer], ['U']
    if buffer in BINARY_OPERATORS:
        return True, [buffer], ['B']
    for i in range(1, len(buffer)):
        pt1 = buffer[:i]
        pt2 = buffer[i:]
        if pt2 in UNARY_OPERATORS:
            return True, [pt1, pt2], ['V', 'U']
        if pt2 in BINARY_OPERATORS:
            return True, [pt1, pt2], ['V', 'B']
    return False, None, None


def preparse(in_str):
    # Converts plaintext expression into list of
    # symbols with attached semantics.
    # Raises errors for invalid expressions.
    in_str = clean(in_str)
    symbols = []
    semantics = []
    buffer = ''
    count = 0
    for i, char in enumerate(in_str):
        if count < 0:
            raise ParseError
        elif count > 0:
            if char == ')':
                if count == 1:
                    symbols.append(buffer)
                    semantics.append('C')
                    buffer = ''
                else:
                    buffer += char
                count -= 1
            elif char == '(':
                count += 1
                buffer += char
            else:
                buffer += char

        else:
            has_operator, parts, types = test_buffer(buffer)
            if has_operator:
                symbols += parts
                semantics += types
                buffer = ''
            if char == ')':
                count -= 1
            elif char == '(':
                count += 1
            else:
                buffer += char
    if count == 0:
        if buffer:
            has_operator, parts, types = test_buffer(buffer)
            if has_operator:
                symbols += parts
                semantics += types
            else:
                symbols.append(buffer)
                semantics.append('V')
    else:
        raise ParseError

    return symbols, semantics


def parse_symbols(symbols, semantics, simplify=False):
    # Converts symbol list into equivalent node tree,
    # respecting order or operations.
    # Convert numbers, variables, and parentheses to nodes
    for i, (sym, sem) in enumerate(zip(symbols, semantics)):
        if sem == 'V':
            try:
                float(sym)
                symbols[i] = NumberNode(sym)
            except:
                symbols[i] = VariableNode(sym)
            semantics[i] = 'N'
        if sem == 'C':
            sub_symbols, sub_semantics = preparse(sym)
            symbols[i] = parse_symbols(sub_symbols, sub_semantics)
            semantics[i] = 'N'

    # Convert Unary Operators to Nodes
    total_len = len(symbols)
    for j in range(total_len):
        i = total_len - j - 1
        sym = symbols[i]
        sem = semantics[i]
        if sem == 'U':
            if i == len(symbols) - 1:
                print(i, symbols, semantics)
                raise ParseError
            if semantics[i + 1] != 'N':
                print(i, symbols, semantics)
                raise ParseError
            symbols[i] = unary_nodes[sym](symbols[i + 1])
            semantics[i] = 'N'
            del symbols[i + 1]
            del semantics[i + 1]
        elif sym == '-' and ((i == 0) or (semantics[i - 1] != 'N')):
            if i == len(symbols) - 1:
                print(i, symbols, semantics)
                raise ParseError
            if semantics[i + 1] != 'N':
                print(i, symbols, semantics)
                raise ParseError
            symbols[i] = unary_nodes['NEG'](symbols[i + 1]) # disallow this
            semantics[i] = 'N'
            del symbols[i + 1]
            del semantics[i + 1]

    # Convert binary operators to nodes
    for ops in BINARY_OPERATOR_LIST:
        i = 0
        while i < len(symbols):
            sym = symbols[i]
            sem = semantics[i]
            if sem == 'B' and sym in ops:
                if i == 0 or i == len(symbols) - 1:
                    print(i, symbols, semantics)
                    raise ParseError
                if semantics[i - 1] != 'N' or semantics[i + 1] != 'N':
                    print(i, symbols, semantics)
                    raise ParseError
                symbols[i] = binary_nodes[sym](symbols[i - 1], symbols[i + 1])
                semantics[i] = 'N'
                del symbols[i + 1]
                del semantics[i + 1]
                del symbols[i - 1]
                del semantics[i - 1]

            else:
                i += 1
    if simplify:
        return symbols[0].simplify()
    return symbols[0]


def parse_string(in_str, simplify=False):
    symbols, semantics = preparse(in_str)
    tree = parse_symbols(symbols, semantics, simplify)
    return tree