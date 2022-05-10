import numpy as np
from copy import deepcopy

from exceptions import EvaluationError


class Node():
    def __init__(self, children=None, symbol=None, value=None):
        if children is None:
            self.children = []
        else:
            self.children = children

        # The symbol used to represent this node in printed expression.
        self.symbol = symbol

        # The value of this node used in evaluation.
        self.value = value
        self.sorted = False

    def print_tree(self, indent=0):
        pass

    def print_string(self):
        pass

    def evaluate(self, mapping):
        pass

    def take_derivative(self, variable):
        pass

    def function_of(self, variable):
        for child in self.children:
            if child.function_of(variable):
                return True
        return False

    def simplify(self):
        """ Reorders the tree with this node as a root. The simplifying is only
        partially implemented, but ensures the following:
        -  Sum nodes will never be the children of product nodes. This has the
           side effect that all multiplications are fully distributed.
        -  Like terms will be gathered under any product node (x * x -> x ^ 2).
        -  Like terms will be gathered under any sum node (2 + 2 -> 4).
        -  Multiplications by 1 will be ignored, multiplications by 0 will result in
           removal of the term.
        - Raising to the power of 0 will be replaced by 1.

        """
        return self

    def is_zero(self):
        return False

    def is_one(self):
        return False

    def sort_nodes(self):
        if not self.sorted:
            for child in self.children:
                child.sort_nodes()
            self.children = sorted(self.children, key=lambda x: x.id_string())
            self.sorted = True

    def id_string(self):
        return ''

# Nodes that represent a single value (either variable or fixed number)


class ValueNode(Node):
    def __init__(self, symbol):
        super(ValueNode, self).__init__(symbol=str(symbol))
        self.sorted = True

    def print_tree(self, indent=0):
        printed_val = '     ' * indent + self.symbol
        output = [printed_val]
        if indent == 0:
            for val in output:
                print(val)
            return
        return output

    def print_string(self):
        return self.symbol

    def sort_nodes(self):
        pass


# Class representing a float value.
class NumberNode(ValueNode):
    def __init__(self, value):
        super(NumberNode, self).__init__(value)
        self.value = float(value)

    def evaluate(self, mapping=None):
        return self.value

    def take_derivative(self, variable):
        return NumberNode(0)

    def function_of(self, variable):
        return False

    def is_zero(self):
        return self.value == 0

    def is_one(self):
        return self.value == 1

    def id_string(self):
        return '!' + str(self.value)


# Class representing a variable. Derivatives can be taken w.r.t this variable
# and numeric values can be substituted for it in evaluation.
class VariableNode(ValueNode):
    def __init__(self, value):
        super(VariableNode, self).__init__(value)

    def evaluate(self, mapping=None):
        if mapping is None or self.symbol not in mapping:
            raise EvaluationError("No value passed for variable %s" % self.symbol)
        return mapping[self.symbol]

    def take_derivative(self, variable):
        if self.symbol == variable:
            return NumberNode(1)
        return NumberNode(0)

    def function_of(self, variable):
        return self.symbol == variable

    def id_string(self):
        return '#' + str(self.symbol)


# Unary Operators


class UnaryNode(Node):
    def __init__(self, symbol, function, child):
        super(UnaryNode, self).__init__(children=[child], symbol=str(symbol))
        self.function = function
        self.sorted = True

    def print_tree(self, indent=0):
        printed_val = '     ' * indent + self.symbol
        output = [printed_val] + self.children[0].print_tree(indent + 1)
        if indent == 0:
            for val in output:
                print(val)
            return
        return output

    def evaluate(self, mapping=None):
        return self.function(self.children[0].evaluate(mapping))

    def print_string(self):
        return self.symbol + '(' + self.children[0].print_string() + ')'

    def simplify(self):
        try:
            value = self.evaluate({})
            return NumberNode(value)
        except EvaluationError:
            self.children[0] = self.children[0].simplify()
            return self

    def sort_nodes(self):
        self.children[0].sort_nodes()

    def id_string(self):
        return '$' + self.symbol + '(' + self.children[0].id_string() + ')'


class SinNode(UnaryNode):
    def __init__(self, child):
        super(SinNode, self).__init__('sin', np.sin, child)

    def take_derivative(self, variable):
        # d f(g(x)) = g'(x) * f'(g(x))
        g_prime = self.children[0].take_derivative(variable)
        f_prime = CosNode(
            deepcopy(self.children[0])
        )
        return ProductNode(
            [g_prime,
             f_prime],
            [1,
             1]
        )


class CosNode(UnaryNode):
    def __init__(self, child):
        super(CosNode, self).__init__('cos', np.cos, child)

    def take_derivative(self, variable):
        # d f(g(x)) = g'(x) * f'(g(x))
        g_prime = self.children[0].take_derivative(variable)
        f_prime = SumNode(
            [SinNode(
                deepcopy(self.children[0])
             )],
            [-1]
        )
        return ProductNode(
            [g_prime,
             f_prime],
            [1,
             1]
        )


class TanNode(UnaryNode):
    def __init__(self, child):
        super(TanNode, self).__init__('tan', np.tan, child)

    def take_derivative(self, variable):
        # d f(g(x)) = g'(x) * f'(g(x))
        g_prime = self.children[0].take_derivative(variable)
        f_prime = ProductNode(
            [CosNode(
                deepcopy(self.children[0]))],
            [-2]
            )
        return ProductNode(
            [g_prime,
             f_prime],
            [1,
             1]
        )


class LogNode(UnaryNode):
    def __init__(self, child):
        super(LogNode, self).__init__('log', np.log, child)

    def take_derivative(self, variable):
        # d f(g(x)) = g'(x) * f'(g(x))
        g_prime = self.children[0].take_derivative(variable)
        f_prime = ProductNode(
            [deepcopy(self.children[0])],
            [-1]
        )
        return ProductNode(
            [g_prime,
             f_prime],
            [1,
             1]
        )


class ExpNode(UnaryNode):
    def __init__(self, child):
        super(ExpNode, self).__init__('exp', np.exp, child)

    def take_derivative(self, variable):
        # d f(g(x)) = g'(x) * f'(g(x))
        g_prime = self.children[0].take_derivative(variable)
        f_prime = ExpNode(
            deepcopy(self.children[0])
        )
        return ProductNode(
            [g_prime,
             f_prime],
            [1,
             1]
        )

# Deprecated
# class NegNode(UnaryNode):
#     def __init__(self, child):
#         def neg_fn(val):
#               return -1 * val
#         super(NegNode, self).__init__('-', neg_fn, child)

#     def take_derivative(self, variable):
#         return NegNode(
#             self.children[0].take_derivative(variable)
#         )

#     def print_string(self):
#         if isinstance(self.children[0], BinaryNode):
#             return self.symbol + '(' + self.children[0].print_string() + ')'
#         return self.symbol + self.children[0].print_string()

#     def simplify(self):
#         child = self.children[0]
#         # -number(num) = number(-num)
#         if isinstance(child, NumberNode):
#             return NumberNode(-1 * child.value)

#         #-(-f(x)) = f(x)
#         if isinstance(child, NegNode):
#             subchild = child.children[0]
#             return subchild.simplify()

#         # -(a-b) = b-a
#         if isinstance(child, SubtractNode):
#             first, second = child.children
#             return SubtractNode(
#                 second,
#                 first
#             ).simplify

#         # - (a + b) = (-a) + (-b)
#         if isinstance(child, AddNode):
#             first, second = child.children
#             return AddNode(
#                 NegNode(second),
#                 NegNode(first)
#             ).simplify

#         # - (a * b) = -1 * a * b


# Binary Operators (deprecated)
# class BinaryNode(Node):
#     def __init__(self, symbol, prechild, postchild, lower_precedence_ops=None):
#         super(BinaryNode, self).__init__(
#             children=[prechild, postchild], symbol=str(symbol))
#         if lower_precedence_ops is None:
#             self.lower_precedence_ops = []
#         else:
#             self.lower_precedence_ops = lower_precedence_ops

#     def print_tree(self, indent=0):
#         printed_val = '     ' * indent + self.symbol
#         output = self.children[0].print_tree(indent + 1) + [printed_val] + self.children[1].print_tree(indent + 1)
#         if indent == 0:
#             for val in output:
#                 print(val)
#             return
#         return output

#     def print_string(self):
#         if self.children[0].symbol in self.lower_precedence_ops:
#             child_str_0 = '(' + self.children[0].print_string() + ') '
#         else:
#             child_str_0 = self.children[0].print_string() + ' '
#         if self.children[1].symbol in self.lower_precedence_ops:
#             child_str_1 = ' (' + self.children[1].print_string() + ')'
#         else:
#             child_str_1 = ' ' + self.children[1].print_string()
#         return child_str_0 + self.symbol + child_str_1


# class AddNode(BinaryNode):
#     def __init__(self, prechild, postchild):
#         super(AddNode, self).__init__('+', prechild, postchild)

#     def evaluate(self, mapping=None):
#         return self.children[0].evaluate(mapping) + self.children[1].evaluate(mapping)

#     def take_derivative(self, variable):
#         first, second = self.children
#         return AddNode(
#             first.take_derivative(variable),
#             second.take_derivative(variable)
#         )

# class SubtractNode(BinaryNode):
#     def __init__(self, prechild, postchild):
#         super(SubtractNode, self).__init__('-', prechild, postchild)

#     def evaluate(self, mapping=None):
#         return self.children[0].evaluate(mapping) - self.children[1].evaluate(mapping)

#     def take_derivative(self, variable):
#         first, second = sefl.children
#         return SubtractNode(
#             first.take_derivative(variable),
#             second.take_derivative(variable)
#         )

# class MultiplyNode(BinaryNode):
#     def __init__(self, prechild, postchild):
#         super(MultiplyNode, self).__init__('*', prechild, postchild, ['+', '-'])

#     def evaluate(self, mapping=None):
#         return self.children[0].evaluate(mapping) * self.children[1].evaluate(mapping)

#     def take_derivative(self, variable):
#         first, second = self.children
#         return AddNode(
#             MultiplyNode(
#                 first.take_derivative(variable),
#                 deepcopy(second)
#             ),
#             MultiplyNode(
#                 deepcopy(first),
#                 second.take_derivative(variable)
#             )
#         )

# class DivideNode(BinaryNode):
#     def __init__(self, prechild, postchild):
#         super(DivideNode, self).__init__('/', prechild, postchild, ['+', '-'])

#     def evaluate(self, mapping=None):
#         return self.children[0].evaluate(mapping) / self.children[1].evaluate(mapping)

#     def take_derivative(self, variable):
#         numerator, denominator = self.children
#         return DivideNode(
#             SubtractNode(
#                 MultiplyNode(
#                     numerator.take_derivative(variable),
#                     deepcopy(denominator)
#                 ),
#                 MultiplyNode(
#                     self.children[1].take_derivative(variable),
#                     deepcopy(numerator)
#                 )
#             ),
#             PowerNode(
#                 deepcopy(denominator),
#                 NumberNode(2)
#             )
#         )


# class PowerNode(BinaryNode):
#     def __init__(self, prechild, postchild):
#         super(PowerNode, self).__init__('^', prechild, postchild, ['+', '-', '*', '/'])

#     def evaluate(self, mapping=None):
#         return self.children[0].evaluate(mapping) ** self.children[1].evaluate(mapping)

#     def take_derivative(self, variable):
#         base, exponent = self.children

#         if exponent.function_of(variable):
#             # Case 1: y = f(x?) ^ g(x)
#             # y = exp(log(f(x?)) * g(x))
#             node_refactored = ExpNode(
#                 MultiplyNode(
#                     LogNode(deepcopy(base)),
#                     deepcopy(exponent)
#                 )
#             )
#             return node_refactored.take_derivative(variable)
#         else:
#             if base.function_of(variable):
#                 # Case 2: y = f(x) ^ const
#                 # y' = const * f(x) ^ (const - 1) * f'(x)
#                 return MultiplyNode(
#                     deepcopy(exponent),
#                     MultiplyNode(
#                         PowerNode(
#                             deepcopy(base),
#                             SubtractNode(
#                                 deepcopy(exponent),
#                                 NumberNode(1)
#                             )
#                         ),
#                         base.take_derivative(variable)
#                     )
#                 )
#             else:
#                 # Case 4: y = const ^ const
#                 # y' = 0
#                 return NumberNode(0)

# We do not have explicit classes for the binary
# operators as representing them this way
# makes simplification difficult. Instead we
# will collapse +,-,*,/,^ into
# a product node and a sum node which capture the
# commutative and distributive nature of these operations


class SumNode(Node):
    # Represents a sum of terms multiplied by constants:
    #
    # SIGMA a_i * f_i
    #
    # Where a_i are floats stored in multipliers, and
    # f_i are other nodes stored in addends.
    def __init__(self, addends, multipliers):
        super(SumNode, self).__init__(
            children=addends, symbol='SUM')
        # Factors must be numbers
        self.multipliers = multipliers

    def print_tree(self, indent=0):
        printed_val = '     ' * indent + self.symbol
        output = [printed_val]
        for multiplier, child in zip(self.multipliers, self.children):
            if multiplier == 1:
                output += child.print_tree(indent + 1)
            else:
                output += ['     ' * (indent + 2) + str(multiplier),
                           '     ' * (indent + 1) + '*']
                output += child.print_tree(indent + 2)
        if indent == 0:
            for val in output:
                print(val)
            return
        return output

    def print_string(self):
        out_str = ''
        for i, multiplier, child in zip(range(len(self.children)), self.multipliers, self.children):
            if multiplier > 0:
                symbol = '+'
            else:
                symbol = '-'
            if i:
                out_str += ' ' + symbol + ' '
            elif symbol == '-':
                out_str += symbol
            if abs(multiplier) != 1:
                out_str += str(abs(multiplier)) + ' * '
            out_str += child.print_string()
        return out_str.strip()

    def evaluate(self, mapping):
        out_val = 0
        for multiplier, child in zip(self.multipliers, self.children):
            out_val += multiplier * child.evaluate(mapping)

        return out_val

    def take_derivative(self, variable):
        child_derivatives = [
            child.take_derivative(variable) for child in self.children
        ]
        return SumNode(child_derivatives, self.multipliers)

    def simplify(self):
        if len(self.children) != len(self.multipliers):
            print(self.children)
            print(self.multipliers)
            self.print_tree()
            assert 0
        # Simplify subnodes and remove multiplier zeros
        for i in range(len(self.children) - 1, -1,  -1):
            if self.multipliers[i]:
                self.children[i] = self.children[i].simplify()
            else:
                self.children.pop(i)
                self.multipliers.pop(i)

        # Combine sub-add nodes with this one, remove child zeros
        for i in range(len(self.children) - 1, -1,  -1):
            if self.children[i].is_zero():
                self.children.pop(i)
                self.multipliers.pop(i)
            elif isinstance(self.children[i], SumNode):
                child = self.children.pop(i)
                multiplier = self.multipliers.pop(i)

                new_multipliers = [multiplier * x for x in child.multipliers]
                new_children = child.children

                self.children += new_children
                self.multipliers += new_multipliers
                self.sorted = False

        # Combine like terms
        self.sort_nodes()

        # Number nodes
        total_number = 0
        for i in range(len(self.children) - 1, -1,  -1):
            child = self.children[i]
            multiplier = self.multipliers[i]
            if isinstance(child, NumberNode):
                total_number += child.value * multiplier
                self.children.pop(i)
                self.multipliers.pop(i)
        if total_number != 0:
            self.children = [NumberNode(total_number)] + self.children
            self.multipliers = [1] + self.multipliers

        # Variable/other nodes
        for i in range(len(self.children) - 2, -1,  -1):
            child1 = self.children[i]
            child2 = self.children[i + 1]
            if child1.id_string() == child2.id_string():
                self.multipliers[i] += self.multipliers[i + 1]
                self.children.pop(i + 1)
                self.multipliers.pop(i + 1)

        if not len(self.children):
            return NumberNode(0)

        if len(self.children) == 1 and self.multipliers[0] == 1:
            return self.children[0]

        return self

    def id_string(self):
        self.sort_nodes()
        id_str = '&' + self.symbol + '['
        for multiplier, child in zip(self.multipliers, self.children):
            id_str += str(multiplier) + ',' + child.id_string() + ','
        id_str += ']'

        return id_str

    def sort_nodes(self):
        if not self.sorted:
            for child in self.children:
                child.sort_nodes()
            sorted_values = sorted(zip(self.children, self.multipliers), key=lambda x: x[0].id_string())
            self.children = [x[0] for x in sorted_values]
            self.multipliers = [x[1] for x in sorted_values]
            self.sorted = True


class ProductNode(Node):
    # Represents the product of terms to constant powers:
    #
    # PI f_i ^ a_ i
    #
    # Where a_i are floats stored in powers and
    # f_i are other nodes stored in children.
    def __init__(self, multiplicands, powers):
        self.isroot = False
        super(ProductNode, self).__init__(
            children=multiplicands, symbol='PRODUCT')
        # Powers must be numbers
        self.powers = powers
        if len(self.children) != len(self.powers):
            print(self.children)
            print(self.powers)
            self.print_tree()
            assert 0

    def print_tree(self, indent=0):
        printed_val = '     ' * indent + self.symbol
        output = [printed_val]
        for power, child in zip(self.powers, self.children):
            if power == 1:
                output += child.print_tree(indent + 1)
            else:
                output += child.print_tree(indent + 2) + [
                    '     ' * (indent + 1) + '^',
                    '     ' * (indent + 2) + str(power)]
        if indent == 0:
            for val in output:
                print(val)
            return
        return output

    def print_string(self):
        out_str = ''
        for i, power, child in zip(range(len(self.children)), self.powers, self.children):
            if i:
                out_str += ' * '
            if isinstance(child, SumNode):
                out_str += '(' + child.print_string() + ')'
            else:
                out_str += child.print_string()
            if power != 1:
                out_str += ' ^ ' + str(power)

        return out_str.strip()

    def evaluate(self, mapping):
        out_val = 1
        for power, child in zip(self.powers, self.children):
            out_val *= child.evaluate(mapping) ** power

        return out_val

    def take_derivative(self, variable):
        # d Product_i(f(x)_i ^ p_i) =
        #     Sum_i((p_i) * f(x)_i' * f(x)_i ^ (p_i - 1) * Product_(j!=i)(f(x)_j^p_j))
        child_derivatives = [
            child.take_derivative(variable) for child in self.children
        ]
        sum_nodes = []
        sum_multipliers = []
        for i, d_node in enumerate(child_derivatives):
            product_nodes = []
            product_powers = []
            for j, (node, power) in enumerate(zip(self.children, self.powers)):
                if j == i:
                    product_nodes += [deepcopy(d_node), deepcopy(node)]
                    product_powers += [1, power - 1]
                    sum_multipliers.append(power)
                else:
                    product_nodes.append(deepcopy(node))
                    product_powers.append(power)
            sum_nodes.append(ProductNode(product_nodes, product_powers))

        return SumNode(sum_nodes, sum_multipliers)

    def simplify(self):
        if len(self.children) != len(self.powers):
            print(self.children)
            print(self.powers)
            self.print_tree()
            assert 0

        if not len(self.children):
            return NumberNode(1)

        # Combine sub-product nodes with this one, remove child ones,
        # check for child zeros
        i = len(self.children) - 1
        while i >= 0:
            power = self.powers[i]
            child = self.children[i]
            if not power:
                # Remove nodes with a power of zero
                self.children.pop(i)
                self.powers.pop(i)
                i -= 1
                continue
            child = child.simplify()
            self.children[i] = child

            if child.is_zero():
                # If any term is 0, the whole product is
                return child

            if child.is_one():
                # We can ignore any term with a base of 1
                self.children.pop(i)
                self.powers.pop(i)
                i -= 1
                continue

            if isinstance(child, ProductNode):
                # Any child product node can be added into this one.
                # Simplified product nodes will never have children which
                # are also product nodes.
                new_powers = [power * x for x in child.powers]
                new_children = child.children
                self.children.pop(i)
                self.powers.pop(i)
                self.children = self.children + new_children
                self.powers = self.powers + new_powers

                self.sorted = False

                i -= 1
                continue
            i -= 1

        # Combine terms with the same base
        self.sort_nodes()
        total_number = 1
        for i in range(len(self.children) - 1, -1,  -1):
            child = self.children[i]
            power = self.powers[i]
            # Merge nodes with the same base
            if i and child.id_string() == self.children[i - 1].id_string():
                self.powers[i - 1] += power
                self.children.pop(i)
                self.powers.pop(i)
                continue

            # Deal with number nodes
            if isinstance(child, NumberNode):
                total_number *= child.value ** power
                self.children.pop(i)
                self.powers.pop(i)

        if not len(self.children):
            return NumberNode(total_number)

        # Distribute across sums.
        # We do this in a recursive manner, so it can be really slow for
        # high powers.
        for i in range(len(self.children)):
            child = self.children[i]
            power = self.powers[i]
            if isinstance(child, SumNode):
                self.powers[i] -= 1
                new_product_nodes = []
                for sum_term in child.children:
                    new_children = [deepcopy(sum_term)] + deepcopy(self.children)
                    new_powers = [1] + deepcopy(self.powers)
                    new_product_node = ProductNode(new_children, new_powers)
                    new_product_nodes.append(new_product_node)
                new_multipliers = [total_number * multiplier for multiplier in child.multipliers]
                distributed_sum_node = SumNode(new_product_nodes, new_multipliers)
                distributed_sum_node.simplify()

                return distributed_sum_node

#         There will be no purely numeric terms in ProductNodes in a simplified expression.
#         These factors will be wrapped in SumNodes.
        if total_number != 1:
            return SumNode([self.simplify()], [total_number])
        if len(self.children) == 1 and self.powers[0] == 1:
            return self.children[0]
        return self

    def id_string(self):
        self.sort_nodes()
        id_str = '&' + self.symbol + '['
        for power, child in zip(self.powers, self.children):
            id_str += str(power) + ',' + child.id_string() + ','
        id_str += ']'

        return id_str

    def sort_nodes(self):
        if not self.sorted:
            for child in self.children:
                child.sort_nodes()
            sorted_values = sorted(zip(self.children, self.powers), key=lambda x: x[0].id_string())
            self.children = [x[0] for x in sorted_values]
            self.powers = [x[1] for x in sorted_values]
            self.sorted = True