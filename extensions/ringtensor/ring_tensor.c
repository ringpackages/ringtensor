/*
** RingTensor Extension
** Description: Complete implementation (No FastPro dependency)
** Author: Azzeddine Remmal
*/

#include "ring_tensor.h"

/* --- Helper: Generic 2-List Operation --- */
void tensor_op_generic(void *pPointer, int nOp) {
    List *pListA, *pListB, *pRowA, *pRowB;
    int nRows, nCols, r, c;
    double vA, vB, res;

    if (RING_API_PARACOUNT != 2) { RING_API_ERROR(RING_API_MISS2PARA); return; }
    pListA = RING_API_GETLIST(1);
    pListB = RING_API_GETLIST(2);
    nRows  = ring_list_getsize(pListA);

    for (r = 1; r <= nRows; r++) {
        pRowA = ring_list_getlist(pListA, r);
        pRowB = ring_list_getlist(pListB, r);
        nCols = ring_list_getsize(pRowA);
        for (c = 1; c <= nCols; c++) {
            vA = ring_list_getdouble(pRowA, c);
            vB = ring_list_getdouble(pRowB, c);
            switch(nOp) {
                case 1: res = vA + vB; break; // Add
                case 2: res = vA - vB; break; // Sub
                case 3: res = vA * vB; break; // Mul
                case 4: res = (vB != 0) ? vA / vB : 0.0; break; // Div
            }
            ring_list_setdouble_gc(RING_API_STATE, pRowA, c, res);
        }
    }
    RING_API_RETLIST(pListA);
}

RING_FUNC(ring_tensor_add)      { tensor_op_generic(pPointer, 1); }
RING_FUNC(ring_tensor_sub)      { tensor_op_generic(pPointer, 2); }
RING_FUNC(ring_tensor_mul_elem) { tensor_op_generic(pPointer, 3); }
RING_FUNC(ring_tensor_div)      { tensor_op_generic(pPointer, 4); }

/* --- Helper: Math Transformations (Square, Sqrt, Exp) --- */
void tensor_math_single(void *pPointer, int nOp) {
    List *pList, *pRow;
    int r, c, rows, cols;
    double v, res;
    pList = RING_API_GETLIST(1);
    rows = ring_list_getsize(pList);

    for (r = 1; r <= rows; r++) {
        pRow = ring_list_getlist(pList, r);
        cols = ring_list_getsize(pRow);
        for (c = 1; c <= cols; c++) {
            v = ring_list_getdouble(pRow, c);
            switch(nOp) {
                case 1: res = v * v; break; // Square
                case 2: res = sqrt(v); break; // Sqrt
                case 3: res = exp(v); break; // Exp
            }
            ring_list_setdouble_gc(RING_API_STATE, pRow, c, res);
        }
    }
    RING_API_RETLIST(pList);
}

RING_FUNC(ring_tensor_square) { tensor_math_single(pPointer, 1); }
RING_FUNC(ring_tensor_sqrt)   { tensor_math_single(pPointer, 2); }
RING_FUNC(ring_tensor_exp)    { tensor_math_single(pPointer, 3); }

/* --- Scalars & Fill --- */
RING_FUNC(ring_tensor_scalar_mul) {
    List *pList, *pRow;
    int r, c, rows, cols;
    double val;
    pList = RING_API_GETLIST(1);
    val   = RING_API_GETNUMBER(2);
    rows  = ring_list_getsize(pList);
    for (r = 1; r <= rows; r++) {
        pRow = ring_list_getlist(pList, r);
        cols = ring_list_getsize(pRow);
        for (c = 1; c <= cols; c++) {
            double v = ring_list_getdouble(pRow, c);
            ring_list_setdouble_gc(RING_API_STATE, pRow, c, v * val);
        }
    }
    RING_API_RETLIST(pList);
}

RING_FUNC(ring_tensor_add_scalar) {
    List *pList, *pRow;
    int r, c, rows, cols;
    double val;
    pList = RING_API_GETLIST(1);
    val   = RING_API_GETNUMBER(2);
    rows  = ring_list_getsize(pList);
    for (r = 1; r <= rows; r++) {
        pRow = ring_list_getlist(pList, r);
        cols = ring_list_getsize(pRow);
        for (c = 1; c <= cols; c++) {
            double v = ring_list_getdouble(pRow, c);
            ring_list_setdouble_gc(RING_API_STATE, pRow, c, v + val);
        }
    }
    RING_API_RETLIST(pList);
}

RING_FUNC(ring_tensor_fill) {
    List *pList, *pRow;
    int r, c, rows, cols;
    double val;
    pList = RING_API_GETLIST(1);
    val   = RING_API_GETNUMBER(2);
    rows  = ring_list_getsize(pList);
    for (r = 1; r <= rows; r++) {
        pRow = ring_list_getlist(pList, r);
        cols = ring_list_getsize(pRow);
        for (c = 1; c <= cols; c++) {
            ring_list_setdouble_gc(RING_API_STATE, pRow, c, val);
        }
    }
    RING_API_RETLIST(pList);
}

RING_FUNC(ring_tensor_random) {
    List *pList, *pRow;
    int r, c, rows, cols;
    double rnd;
    pList = RING_API_GETLIST(1);
    rows  = ring_list_getsize(pList);
    
    // Seed randomization (Ideally done once, but ensuring randomness here)
    // srand(time(NULL)); 

    for (r = 1; r <= rows; r++) {
        pRow = ring_list_getlist(pList, r);
        cols = ring_list_getsize(pRow);
        for (c = 1; c <= cols; c++) {
            // Random 0.0 to 1.0
            rnd = (double)rand() / (double)RAND_MAX;
            ring_list_setdouble_gc(RING_API_STATE, pRow, c, rnd);
        }
    }
    RING_API_RETLIST(pList);
}

/* --- Matrix Ops --- */
RING_FUNC(ring_tensor_matmul) {
    List *pA, *pB, *pC, *pRowA, *pRowB, *pRowC;
    int rA, cA, rB, cB, i, j, k;
    double sum;

    pA = RING_API_GETLIST(1);
    pB = RING_API_GETLIST(2);
    rA = ring_list_getsize(pA);
    if(rA==0) return;
    cA = ring_list_getsize(ring_list_getlist(pA, 1));
    rB = ring_list_getsize(pB);
    if(rB==0) return;
    cB = ring_list_getsize(ring_list_getlist(pB, 1));

    if (cA != rB) { RING_API_ERROR("MatMul Dims Mismatch"); return; }

    pC = RING_API_NEWLISTUSINGBLOCKS2D(rA, cB);

    for (i = 1; i <= rA; i++) {
        pRowA = ring_list_getlist(pA, i);
        pRowC = ring_list_getlist(pC, i);
        for (j = 1; j <= cB; j++) {
            sum = 0.0;
            for (k = 1; k <= cA; k++) {
                pRowB = ring_list_getlist(pB, k);
                sum += (ring_list_getdouble(pRowA, k) * ring_list_getdouble(pRowB, j));
            }
            ring_list_setdouble_gc(RING_API_STATE, pRowC, j, sum);
        }
    }
    RING_API_RETLIST(pC);
}

RING_FUNC(ring_tensor_transpose) {
    List *pA, *pC, *pRowA, *pRowC;
    int nRows, nCols, i, j;
    pA = RING_API_GETLIST(1);
    nRows = ring_list_getsize(pA);
    if(nRows==0) return;
    nCols = ring_list_getsize(ring_list_getlist(pA, 1));

    pC = RING_API_NEWLISTUSINGBLOCKS2D(nCols, nRows);

    for (i = 1; i <= nRows; i++) {
        pRowA = ring_list_getlist(pA, i);
        for (j = 1; j <= nCols; j++) {
            double val = ring_list_getdouble(pRowA, j);
            pRowC = ring_list_getlist(pC, j);
            ring_list_setdouble_gc(RING_API_STATE, pRowC, i, val);
        }
    }
    RING_API_RETLIST(pC);
}

RING_FUNC(ring_tensor_sum) {
    List *pList, *pRow, *pRes, *pResRow;
    int r, c, rows, cols, axis;
    double sum;

    pList = RING_API_GETLIST(1);
    axis  = (int)RING_API_GETNUMBER(2);
    rows = ring_list_getsize(pList);
    if(rows==0) return;
    pRow = ring_list_getlist(pList, 1);
    cols = ring_list_getsize(pRow);

    if (axis == 1) { // Sum Rows -> (Rows x 1)
        pRes = RING_API_NEWLISTUSINGBLOCKS2D(rows, 1);
        for (r = 1; r <= rows; r++) {
            pRow = ring_list_getlist(pList, r);
            sum = 0.0;
            for (c = 1; c <= cols; c++) sum += ring_list_getdouble(pRow, c);
            pResRow = ring_list_getlist(pRes, r);
            ring_list_setdouble_gc(RING_API_STATE, pResRow, 1, sum);
        }
    } else { // Sum Cols -> (1 x Cols)
        pRes = RING_API_NEWLISTUSINGBLOCKS2D(1, cols);
        pResRow = ring_list_getlist(pRes, 1);
        for (c = 1; c <= cols; c++) {
            sum = 0.0;
            for (r = 1; r <= rows; r++) {
                pRow = ring_list_getlist(pList, r);
                sum += ring_list_getdouble(pRow, c);
            }
            ring_list_setdouble_gc(RING_API_STATE, pResRow, c, sum);
        }
    }
    RING_API_RETLIST(pRes);
}

RING_FUNC(ring_tensor_mean) {
    List *pList, *pRow;
    int r, c, rows, cols;
    double sum = 0.0;
    pList = RING_API_GETLIST(1);
    rows = ring_list_getsize(pList);
    if(rows==0) { RING_API_RETNUMBER(0); return; }
    cols = ring_list_getsize(ring_list_getlist(pList, 1));
    
    for(r=1; r<=rows; r++) {
        pRow = ring_list_getlist(pList, r);
        for(c=1; c<=cols; c++) sum += ring_list_getdouble(pRow, c);
    }
    RING_API_RETNUMBER(sum / (rows * cols));
}

/* --- Activations --- */
// ... (Helper function same as previous response) ...
void apply_activation(void *pPointer, int type) {
    List *pList, *pRow;
    int r, c, rows, cols;
    double v, res;
    pList = RING_API_GETLIST(1);
    rows = ring_list_getsize(pList);
    for (r = 1; r <= rows; r++) {
        pRow = ring_list_getlist(pList, r);
        cols = ring_list_getsize(pRow);
        for (c = 1; c <= cols; c++) {
            v = ring_list_getdouble(pRow, c);
            switch(type) {
                case 1: res = 1.0 / (1.0 + exp(-v)); break; // Sigmoid
                case 2: res = v * (1.0 - v); break;         // SigmoidPrime
                case 3: res = tanh(v); break;               // Tanh
                case 4: res = 1.0 - (v * v); break;         // TanhPrime
                case 5: res = (v > 0) ? v : 0; break;       // ReLU
                case 6: res = (v > 0) ? 1.0 : 0.0; break;   // ReLUPrime
            }
            ring_list_setdouble_gc(RING_API_STATE, pRow, c, res);
        }
    }
    RING_API_RETLIST(pList);
}

RING_FUNC(ring_tensor_sigmoid)       { apply_activation(pPointer, 1); }
RING_FUNC(ring_tensor_sigmoid_prime) { apply_activation(pPointer, 2); }
RING_FUNC(ring_tensor_tanh)          { apply_activation(pPointer, 3); }
RING_FUNC(ring_tensor_tanh_prime)    { apply_activation(pPointer, 4); }
RING_FUNC(ring_tensor_relu)          { apply_activation(pPointer, 5); }
RING_FUNC(ring_tensor_relu_prime)    { apply_activation(pPointer, 6); }

RING_FUNC(ring_tensor_softmax) {
    List *pList, *pRow;
    int r, c, rows, cols;
    double maxVal, rowSum, v;
    pList = RING_API_GETLIST(1);
    rows = ring_list_getsize(pList);
    for (r = 1; r <= rows; r++) {
        pRow = ring_list_getlist(pList, r);
        cols = ring_list_getsize(pRow);
        maxVal = -DBL_MAX;
        for (c = 1; c <= cols; c++) {
            v = ring_list_getdouble(pRow, c);
            if (v > maxVal) maxVal = v;
        }
        rowSum = 0.0;
        for (c = 1; c <= cols; c++) {
            v = exp(ring_list_getdouble(pRow, c) - maxVal);
            ring_list_setdouble_gc(RING_API_STATE, pRow, c, v);
            rowSum += v;
        }
        for (c = 1; c <= cols; c++) {
            v = ring_list_getdouble(pRow, c);
            if(rowSum != 0) v /= rowSum;
            ring_list_setdouble_gc(RING_API_STATE, pRow, c, v);
        }
    }
    RING_API_RETLIST(pList);
}

RING_FUNC(ring_tensor_dropout) {
    List *pList, *pRow;
    int r, c, rows, cols;
    double rate, scale, rnd;
    pList = RING_API_GETLIST(1);
    rate = RING_API_GETNUMBER(2);
    rows = ring_list_getsize(pList);
    scale = 1.0 / (1.0 - rate);
    
    for (r = 1; r <= rows; r++) {
        pRow = ring_list_getlist(pList, r);
        cols = ring_list_getsize(pRow);
        for (c = 1; c <= cols; c++) {
            rnd = (double)rand() / (double)RAND_MAX;
            if (rnd < rate) ring_list_setdouble_gc(RING_API_STATE, pRow, c, 0.0);
            else {
                double v = ring_list_getdouble(pRow, c);
                ring_list_setdouble_gc(RING_API_STATE, pRow, c, v * scale);
            }
        }
    }
    RING_API_RETLIST(pList);
}

/* --- Optimizers --- */
RING_FUNC(ring_tensor_update_sgd) {
    List *pW, *pG, *rW, *rG;
    double lr;
    int rows, cols, r, c;
    pW = RING_API_GETLIST(1);
    pG = RING_API_GETLIST(2);
    lr = RING_API_GETNUMBER(3);
    rows = ring_list_getsize(pW);
    for(r=1; r<=rows; r++) {
        rW = ring_list_getlist(pW, r);
        rG = ring_list_getlist(pG, r);
        cols = ring_list_getsize(rW);
        for(c=1; c<=cols; c++) {
            double val = ring_list_getdouble(rW, c) - (lr * ring_list_getdouble(rG, c));
            ring_list_setdouble_gc(RING_API_STATE, rW, c, val);
        }
    }
}

RING_FUNC(ring_tensor_update_adam) {
    List *pW, *pG, *pM, *pV, *rW, *rG, *rM, *rV;
    double lr, b1, b2, eps;
    int t, rows, cols, r, c;
    double g, m, v, m_hat, v_hat;

    pW = RING_API_GETLIST(1);
    pG = RING_API_GETLIST(2);
    pM = RING_API_GETLIST(3);
    pV = RING_API_GETLIST(4);
    lr = RING_API_GETNUMBER(5);
    b1 = RING_API_GETNUMBER(6);
    b2 = RING_API_GETNUMBER(7);
    eps= RING_API_GETNUMBER(8);
    t  = (int)RING_API_GETNUMBER(9);

    rows = ring_list_getsize(pW);
    double corr1 = 1.0 - pow(b1, t);
    double corr2 = 1.0 - pow(b2, t);
    if(corr1==0) corr1=1e-9;
    if(corr2==0) corr2=1e-9;

    for(r=1; r<=rows; r++) {
        rW = ring_list_getlist(pW, r);
        rG = ring_list_getlist(pG, r);
        rM = ring_list_getlist(pM, r);
        rV = ring_list_getlist(pV, r);
        cols = ring_list_getsize(rW);
        for(c=1; c<=cols; c++) {
            g = ring_list_getdouble(rG, c);
            m = ring_list_getdouble(rM, c);
            v = ring_list_getdouble(rV, c);
            
            m = (b1 * m) + ((1.0 - b1) * g);
            v = (b2 * v) + ((1.0 - b2) * (g * g));
            
            ring_list_setdouble_gc(RING_API_STATE, rM, c, m);
            ring_list_setdouble_gc(RING_API_STATE, rV, c, v);
            
            m_hat = m / corr1;
            v_hat = v / corr2;
            if(v_hat < 0) v_hat = 0;
            
            double w = ring_list_getdouble(rW, c);
            w -= (lr * m_hat) / (sqrt(v_hat) + eps);
            ring_list_setdouble_gc(RING_API_STATE, rW, c, w);
        }
    }
}

RING_FUNC(ring_tensor_argmax) {
    List *pList, *pRow, *pRes, *pResRow;
    int r, c, rows, cols, maxIdx;
    double maxVal, v;
    pList = RING_API_GETLIST(1);
    rows = ring_list_getsize(pList);
    pRes = RING_API_NEWLISTUSINGBLOCKS2D(rows, 1);
    for(r=1; r<=rows; r++) {
        pRow = ring_list_getlist(pList, r);
        cols = ring_list_getsize(pRow);
        maxVal = -DBL_MAX; maxIdx = 1;
        for(c=1; c<=cols; c++) {
            v = ring_list_getdouble(pRow, c);
            if(v > maxVal) { maxVal = v; maxIdx = c; }
        }
        pResRow = ring_list_getlist(pRes, r);
        ring_list_setdouble_gc(RING_API_STATE, pResRow, 1, (double)maxIdx);
    }
    RING_API_RETLIST(pRes);
}

RING_LIBINIT {
    RING_API_REGISTER("tensor_add", ring_tensor_add);
    RING_API_REGISTER("tensor_sub", ring_tensor_sub);
    RING_API_REGISTER("tensor_mul_elem", ring_tensor_mul_elem);
    RING_API_REGISTER("tensor_div", ring_tensor_div);
    RING_API_REGISTER("tensor_scalar_mul", ring_tensor_scalar_mul);
    RING_API_REGISTER("tensor_add_scalar", ring_tensor_add_scalar);
    RING_API_REGISTER("tensor_fill", ring_tensor_fill);
    RING_API_REGISTER("tensor_random", ring_tensor_random);
    RING_API_REGISTER("tensor_square", ring_tensor_square);
    RING_API_REGISTER("tensor_sqrt", ring_tensor_sqrt);
    RING_API_REGISTER("tensor_exp", ring_tensor_exp);
    RING_API_REGISTER("tensor_sum", ring_tensor_sum);
    RING_API_REGISTER("tensor_mean", ring_tensor_mean);
    RING_API_REGISTER("tensor_matmul", ring_tensor_matmul);
    RING_API_REGISTER("tensor_transpose", ring_tensor_transpose);
    RING_API_REGISTER("tensor_sigmoid", ring_tensor_sigmoid);
    RING_API_REGISTER("tensor_sigmoid_prime", ring_tensor_sigmoid_prime);
    RING_API_REGISTER("tensor_tanh", ring_tensor_tanh);
    RING_API_REGISTER("tensor_tanh_prime", ring_tensor_tanh_prime);
    RING_API_REGISTER("tensor_relu", ring_tensor_relu);
    RING_API_REGISTER("tensor_relu_prime", ring_tensor_relu_prime);
    RING_API_REGISTER("tensor_softmax", ring_tensor_softmax);
    RING_API_REGISTER("tensor_dropout", ring_tensor_dropout);
    RING_API_REGISTER("tensor_update_sgd", ring_tensor_update_sgd);
    RING_API_REGISTER("tensor_update_adam", ring_tensor_update_adam);
    RING_API_REGISTER("tensor_argmax", ring_tensor_argmax);
}