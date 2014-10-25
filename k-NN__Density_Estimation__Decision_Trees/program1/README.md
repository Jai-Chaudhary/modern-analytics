Paradoxically good programming practices are essential to Machine Learning, since unlike other cases, 
distinguishing programming errors from real errors is very difficult.

Rules & Recommendations:
=======================
1.  Throw away un-needed data as early as possible.
2.  Wherever possible use python Generators, or equivalent in other languages.
3.  Use python logging module.
4.  Don't repeat your code, this is especially true for evaluation functions, data loaders etc.
5.  Explore Numpy. It can do most matrix operations, these include multiplication, elementwise multiplication, elementwise subtraction etc. 
6.  Resist the temptation to write a for loop, numpy typically contains code written in c for most typical operations
7.  Be careful with Array appends in Numpy, Unlike python lists appending to an same array results in a reallocation.
8.  Pandas frames are a superior alternative to numpy arrays particularly for datasets involving mixed fields.
9.  Convert strings to floats or ints: "1213.123213213" takes more memory than the equivalent float representation
10. Derive normalization and scale parameters on training data and re-use them with test data.
11. Use enumerate instead of a range len loop
