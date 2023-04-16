# How to interpret the csv file


Example
```csv
isUnsafe,x1,x2,x3,x4,x5,x6,x7,x8,x9,A11,A21,A31,A41,A51,A61,A71,A81,A91,A12,A22,A32,A42,A52,A62,A72,A82,A92,A13,A23,A33,A43,A53,A63,A73,A83,A93,A14,A24,A34,A44,A54,A64,A74,A84,A94,A15,A25,A35,A45,A55,A65,A75,A85,A95,A16,A26,A36,A46,A56,A66,A76,A86,A96,A17,A27,A37,A47,A57,A67,A77,A87,A97,A18,A28,A38,A48,A58,A68,A78,A88,A98,A19,A29,A39,A49,A59,A69,A79,A89,A99,b1,b2,b3,b4,b5,b6,b7,b8,b9
```

- `isUnsafe` represents if the counter-example was unsafe or not. If 1.0, it is unsafe. The conroller at the unsafe point is randomly generated and so ignored in the verification step.
- `x1 - x9` represent the elements of the vector x. The number `i` denotes the index of the vector.
- `A11 - A99` represent the elements of the matrix A. The number `(i,j)` denotes the row and column indices of the matrix A.
- `b1 - x9` represent the elements of the vector b. The number `i` denotes the index of the vector.
