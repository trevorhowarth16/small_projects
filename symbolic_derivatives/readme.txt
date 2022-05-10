This project is like a very basic version of Mathematica-- it allows users to parse symbolic math expressions into representative trees. mathematical operations,
Constant numbers, common  and unknown variables can all be part of the expressions being parsed. The tree representing a given
expression can be used to simplify the expression, evaluate it (plugging in for unknown variables), or take its derivative. Simplifying expressions
is by far the hardest part of this from a design perspective, requiring the creation of compound nodes representing +/* or */^ operations.
Future extensions/work to be done:
	- Implement simplification for Unary nodes.
	- Extend to complex numbers (this would ease the simplification of trig nodes and exponentials).
	- Allow partial evaluation of expressions (some but not all variables plugged in for).
	- Use simplification for equation solving.

See examples.py for basic useage.

