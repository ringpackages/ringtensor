load "ringtensor.ring"



see "=== RingTensor Extension Test ===" + nl


# ---------------------------------------------------------
# 1. Test Addition (Element-Wise Add)
# ---------------------------------------------------------
see "1. Testing tensor_add:" + nl
A = [[1.0, 2.0], 
     [3.0, 4.0]]
B = [[10.0, 10.0], 
     [10.0, 10.0]]

# Expected: [[11, 12], [13, 14]]
C = tensor_add(A, B) 
printMat(C)

# ---------------------------------------------------------
# 2. Test Matrix Multiplication (MatMul)
# ---------------------------------------------------------
see "2. Testing tensor_matmul (Dot Product):" + nl
# (1x2) * (2x1) = (1x1)
# [1, 2] * [3, 4]^T = (1*3 + 2*4) = 11
MatA = [[1.0, 2.0]]
MatB = [[3.0], 
        [4.0]]

Res = tensor_matmul(MatA, MatB)
printMat(Res)

# ---------------------------------------------------------
# 3. Test Transpose - To ensure bug fix
# ---------------------------------------------------------
see "3. Testing tensor_transpose (Non-Square):" + nl
# Input: 1x3
T_In = [[1.0, 2.0, 3.0]]
# Expected: 3x1
T_Out = tensor_transpose(T_In)
printMat(T_Out)

# ---------------------------------------------------------
# 4. Test Scalar Operations - To ensure Float Precision
# ---------------------------------------------------------
see "4. Testing tensor_scalar_mul (Float Precision):" + nl
S_In = [[10.0, 20.0]]
# Expected: [5.5, 11.0] (If cast to int it would be 0)
tensor_scalar_mul(S_In, 0.55) 
printMat(S_In)

# ---------------------------------------------------------
# 5. Test Softmax (Stable)
# ---------------------------------------------------------
see "5. Testing tensor_softmax:" + nl
# Large values to test stability (Should not yield NaN)
Soft_In = [[1000.0, 1000.0, 1000.0]]
# Expected: [0.333, 0.333, 0.333]
Res_Soft = tensor_softmax(Soft_In)
printMat(Res_Soft)

# ---------------------------------------------------------
# 6. Test ArgMax
# ---------------------------------------------------------
see "6. Testing tensor_argmax:" + nl
Arg_In = [
    [0.1, 0.9, 0.0],  # Max at index 2
    [0.8, 0.1, 0.1]   # Max at index 1
]
# Expected: [[2], [1]] (Depending on C indexing it could be 1-based or 0-based)
# In our code we made it return Ring compatible index (1-based)
Res_Arg = tensor_argmax(Arg_In)
printMat(Res_Arg)

# ---------------------------------------------------------
# 7. Test Adam Optimizer (Fused Kernel)
# ---------------------------------------------------------
see "7. Testing tensor_update_adam:" + nl
# Simulate single weight
W = [[0.5]]
G = [[0.1]]  # Gradient
M = [[0.0]]
V = [[0.0]]
LR = 0.01
Beta1 = 0.9
Beta2 = 0.999
Eps = 0.00000001
Time = 1

# Call function (Updates in place)
tensor_update_adam(W, G, M, V, LR, Beta1, Beta2, Eps, Time)

see "New Weight (Should change from 0.5): " 
printMat(W)
see "New Momentum (Should not be 0): " 
printMat(M)

see "=== Test Finished ===" + nl

# --- Helper function to print matrices ---
func printMat aList
    see "[" + nl
    for row in aList
        see "  "
        see row
        see nl
    next
    see "]" + nl + nl
