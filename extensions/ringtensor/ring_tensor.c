/*
** RingTensor Extension Implementation
** Optimized for Dual-Core / Hyper-threaded CPUs
** Fixed for MSVC C3015 Error & High Overhead
*/

#include "ring_tensor.h"

/* 
** TUNING:
** Threshold raised to 10,000 to prevent overhead on small matrices.
** On i3-5005U, small tasks are faster in serial mode.
*/
#define PARALLEL_THRESHOLD 50000

/* --- Memory Management --- */
void ring_tensor_free(void *pState, void *pPointer) {
    tensor_t *t = (tensor_t *)pPointer;
    if (t != NULL) {
        if (t->data != NULL) free(t->data);
        free(t);
    }
}

/* ==================================================================== */
/* --- 1. LIFECYCLE --------------------------------------------------- */
/* ==================================================================== */

RING_FUNC(ring_tensor_init) {
    int rows, cols;
    tensor_t *t;

    if (RING_API_PARACOUNT != 2) {
        RING_API_ERROR(RING_API_MISS2PARA);
        return;
    }

    rows = (int)RING_API_GETNUMBER(1);
    cols = (int)RING_API_GETNUMBER(2);
    
    if (rows <= 0 || cols <= 0) { RING_API_ERROR("Invalid Dims"); return; }

    t = (tensor_t *)malloc(sizeof(tensor_t));
    if(!t) { RING_API_ERROR("Malloc Fail"); return; }
    
    // Default to 2D
    t->ndim = 2;
    t->shape[0] = 1; // Batch
    t->shape[1] = 1; // Heads
    t->shape[2] = rows;
    t->shape[3] = cols;
    
    // Legacy Aliases
    t->rows = rows;
    t->cols = cols;
    
    t->size = rows * cols;
    
    t->data = (double *)calloc(t->size, sizeof(double));
    if (!t->data) { free(t); RING_API_ERROR("Malloc Data Fail"); return; }
    
    RING_API_RETMANAGEDCPOINTER(t, RING_VM_POINTER_TENSOR, ring_tensor_free);
}

/*
** Reshape Tensor
** Usage: tensor_reshape(pTensor, Batch, Heads, Rows, Cols)
** Pass 1 for unused dimensions. Product must equal total size.
*/
RING_FUNC(ring_tensor_reshape) {
    tensor_t *t;
    int b, h, r, c;
    int new_size;

    if (RING_API_PARACOUNT != 5) {
        RING_API_ERROR("Reshape requires 4 dims (Batch, Heads, Rows, Cols). Use 1 for unused.");
        return;
    }

    t = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    b = (int)RING_API_GETNUMBER(2);
    h = (int)RING_API_GETNUMBER(3);
    r = (int)RING_API_GETNUMBER(4);
    c = (int)RING_API_GETNUMBER(5);
    
    new_size = b * h * r * c;
    
    if (new_size != t->size) {
        RING_API_ERROR("Reshape Error: Total size cannot change.");
        return;
    }
    
    // Update Logical Shape
    t->shape[0] = b;
    t->shape[1] = h;
    t->shape[2] = r;
    t->shape[3] = c;
    
    // Determine ndim for internal logic
    if (b > 1) t->ndim = (h > 1) ? 4 : 3;
    else t->ndim = 2;
    
    // Update legacy aliases (Always points to last two dims)
    t->rows = r;
    t->cols = c;
}

/*
** Batch Matrix Multiplication (3D)
** A: (Batch, RowsA, ColsA)
** B: (Batch, RowsB, ColsB) -> ColsA must equal RowsB
** C: (Batch, RowsA, ColsB)
*/
RING_FUNC(ring_tensor_matmul_batch) {
    tensor_t *A = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    tensor_t *B = (tensor_t *)RING_API_GETCPOINTER(2, RING_VM_POINTER_TENSOR);
    tensor_t *C = (tensor_t *)RING_API_GETCPOINTER(3, RING_VM_POINTER_TENSOR);
    
    // Get Batch Size (Assuming stored in shape[0] or implicitly calculated)
    // For safety, we use the explicitly reshaped 'batch' dim
    int batch = A->shape[0];
    
    if (B->shape[0] != batch || C->shape[0] != batch) {
        RING_API_ERROR("BMM: Batch size mismatch");
        return;
    }
    
    int rA = A->rows; // Last dim - 1
    int cA = A->cols; // Last dim
    int cB = B->cols; // Last dim
    
    if (B->rows != cA) { RING_API_ERROR("BMM: Inner dim mismatch"); return; }
    
    int strideA = rA * cA;
    int strideB = cA * cB;
    int strideC = rA * cB;
    
    int b_idx, i, j, k;
    
    double *rowC, *rowA;

    // Parallelize over the BATCH dimension (Perfect scaling!)
    #pragma omp parallel for if(batch > 1) private(b_idx, i, j, k, rowC, rowA)
    for (b_idx = 0; b_idx < batch; b_idx++) {
        
        // Pointers to the start of current batch matrix
        double *ptrA = &A->data[b_idx * strideA];
        double *ptrB = &B->data[b_idx * strideB];
        double *ptrC = &C->data[b_idx * strideC];
        
        // Standard MatMul logic for this batch
        // Zero out C for this batch
        memset(ptrC, 0, strideC * sizeof(double)); // memset is fast per thread
        
        for(i = 0; i < rA; i++) {
            rowC = &ptrC[i * cB];
            rowA = &ptrA[i * cA];
            
            for(k = 0; k < cA; k++) {
                double valA = rowA[k];
                if(valA == 0.0) continue;
                
                double *rowB = &ptrB[k * cB];
                for(j = 0; j < cB; j++) {
                    rowC[j] += valA * rowB[j];
                }
            }
        }
    }
}

RING_FUNC(ring_tensor_set) {
    tensor_t *t;
    int r, c;
    double val;

    if (RING_API_PARACOUNT != 4) return;
    t = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    r = (int)RING_API_GETNUMBER(2);
    c = (int)RING_API_GETNUMBER(3);
    val = RING_API_GETNUMBER(4);
    
    // Bounds check
    if (r < 1 || r > t->rows || c < 1 || c > t->cols) return;
    t->data[(r-1) * t->cols + (c-1)] = val;
}

RING_FUNC(ring_tensor_get) {
    tensor_t *t;
    int r, c;

    if (RING_API_PARACOUNT != 3) return;
    t = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    r = (int)RING_API_GETNUMBER(2);
    c = (int)RING_API_GETNUMBER(3);
    
    if (r < 1 || r > t->rows || c < 1 || c > t->cols) {
        RING_API_RETNUMBER(0.0);
        return;
    }
    RING_API_RETNUMBER(t->data[(r-1) * t->cols + (c-1)]);
}

/* ==================================================================== */
/* --- 2. ELEMENT-WISE MATH (OPTIMIZED) ------------------------------- */
/* ==================================================================== */

void tensor_op_elem(void *pPointer, int op) {
    tensor_t *A = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    tensor_t *B = (tensor_t *)RING_API_GETCPOINTER(2, RING_VM_POINTER_TENSOR);
    int i;
    int size = A->size;

    if (A->size != B->size) { RING_API_ERROR("Tensor Size Mismatch"); return; }

    #pragma omp parallel for if(size > PARALLEL_THRESHOLD)
    for(i=0; i<size; i++) {
        switch(op) {
            case 1: A->data[i] += B->data[i]; break;
            case 2: A->data[i] -= B->data[i]; break;
            case 3: A->data[i] *= B->data[i]; break;
            case 4: A->data[i] = (B->data[i] != 0) ? A->data[i] / B->data[i] : 0.0; break;
        }
    }
}

RING_FUNC(ring_tensor_add)      { tensor_op_elem(pPointer, 1); }
RING_FUNC(ring_tensor_sub)      { tensor_op_elem(pPointer, 2); }
RING_FUNC(ring_tensor_mul_elem) { tensor_op_elem(pPointer, 3); }
RING_FUNC(ring_tensor_div)      { tensor_op_elem(pPointer, 4); }

RING_FUNC(ring_tensor_scalar_mul) {
    tensor_t *A = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    double v = RING_API_GETNUMBER(2);
    int i; 
    int size = A->size;
    
    #pragma omp parallel for if(size > PARALLEL_THRESHOLD)
    for(i=0; i<size; i++) A->data[i] *= v;
}

RING_FUNC(ring_tensor_add_scalar) {
    tensor_t *A = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    double v = RING_API_GETNUMBER(2);
    int i; 
    int size = A->size;
    
    #pragma omp parallel for if(size > PARALLEL_THRESHOLD)
    for(i=0; i<size; i++) A->data[i] += v;
}

RING_FUNC(ring_tensor_sub_scalar) {
    tensor_t *A = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    double v = RING_API_GETNUMBER(2);
    int i;
    int size = A->size;
    
    #pragma omp parallel for if(size > PARALLEL_THRESHOLD)
    for(i=0; i<size; i++) A->data[i] -= v;
}

/* ==================================================================== */
/* --- 3. TRANSFORMS & ACTIVATIONS ------------------------------------ */
/* ==================================================================== */

RING_FUNC(ring_tensor_fill) {
    tensor_t *A = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    double v = RING_API_GETNUMBER(2);
    int i;
    int size = A->size;
    
    #pragma omp parallel for if(size > PARALLEL_THRESHOLD)
    for(i=0; i<size; i++) A->data[i] = v;
}

RING_FUNC(ring_tensor_random) {
    tensor_t *A = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    int i;
    // Keep Random Serial for Reproducibility
    for(i=0; i<A->size; i++) A->data[i] = (double)rand() / (double)RAND_MAX;
}

void tensor_act(void *pPointer, int op) {
    tensor_t *A = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    int i;
    int size = A->size;
    double v;
    
    #pragma omp parallel for if(size > PARALLEL_THRESHOLD) private(v)
    for(i=0; i<size; i++) {
        v = A->data[i];
        switch(op) {
            case 1: A->data[i] = v * v; break; // Square
            case 2: A->data[i] = sqrt(v); break; // Sqrt
            case 3: A->data[i] = exp(v); break; // Exp
            case 4: A->data[i] = 1.0 / (1.0 + exp(-v)); break; // Sigmoid
            case 5: A->data[i] = v * (1.0 - v); break; // SigmoidPrime
            case 6: A->data[i] = tanh(v); break; // Tanh
            case 7: A->data[i] = 1.0 - (v * v); break; // TanhPrime
            case 8: A->data[i] = (v > 0) ? v : 0; break; // ReLU
            case 9: A->data[i] = (v > 0) ? 1.0 : 0.0; break; // ReLUPrime
        }
    }
}

RING_FUNC(ring_tensor_square)        { tensor_act(pPointer, 1); }
RING_FUNC(ring_tensor_sqrt)          { tensor_act(pPointer, 2); }
RING_FUNC(ring_tensor_exp)           { tensor_act(pPointer, 3); }
RING_FUNC(ring_tensor_sigmoid)       { tensor_act(pPointer, 4); }
RING_FUNC(ring_tensor_sigmoid_prime) { tensor_act(pPointer, 5); }
RING_FUNC(ring_tensor_tanh)          { tensor_act(pPointer, 6); }
RING_FUNC(ring_tensor_tanh_prime)    { tensor_act(pPointer, 7); }
RING_FUNC(ring_tensor_relu)          { tensor_act(pPointer, 8); }
RING_FUNC(ring_tensor_relu_prime)    { tensor_act(pPointer, 9); }

RING_FUNC(ring_tensor_softmax) {
    tensor_t *T = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    int r, c;
    
    // Row-wise Softmax
    #pragma omp parallel for if(T->rows > 64) private(c)
    for(r=0; r<T->rows; r++) {
        double maxVal = -DBL_MAX;
        double sum = 0.0;
        int offset = r * T->cols;
        double invSum;
        
        for(c=0; c<T->cols; c++) if(T->data[offset+c] > maxVal) maxVal = T->data[offset+c];
        
        for(c=0; c<T->cols; c++) {
            T->data[offset+c] = exp(T->data[offset+c] - maxVal);
            sum += T->data[offset+c];
        }
        
        invSum = (sum != 0) ? 1.0/sum : 0.0;
        for(c=0; c<T->cols; c++) T->data[offset+c] *= invSum;
    }
}

/* ==================================================================== */
/* --- 4. MATRIX OPS (OPTIMIZED MATMUL) ------------------------------- */
/* ==================================================================== */

RING_FUNC(ring_tensor_matmul) {
    tensor_t *A = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    tensor_t *B = (tensor_t *)RING_API_GETCPOINTER(2, RING_VM_POINTER_TENSOR);
    tensor_t *C = (tensor_t *)RING_API_GETCPOINTER(3, RING_VM_POINTER_TENSOR);
    
    int rA = A->rows; 
    int cA = A->cols; 
    int cB = B->cols;
    int i, j, k;
    int sizeC = rA * cB;
    
    // Pre-declare pointers for MSVC OpenMP compatibility
    double *rowC, *rowA;

    if (cA != B->rows) { RING_API_ERROR("MatMul Dims Mismatch"); return; }
    
    // Clear C (Parallel if large)
    #pragma omp parallel for if(sizeC > PARALLEL_THRESHOLD)
    for(i=0; i<sizeC; i++) C->data[i] = 0.0;

    // Core MatMul: i-k-j loop order for Cache Locality
    // This is the most critical loop for performance
    #pragma omp parallel for if(rA > 4) schedule(static) private(rowC, rowA, k, j)
    for(i = 0; i < rA; i++) {
        rowC = &C->data[i * cB]; 
        rowA = &A->data[i * cA];
        
        for(k = 0; k < cA; k++) {
            double valA = rowA[k]; 
            // Sparse Optimization: Skip if A is 0
            if (valA == 0.0) continue;

            double *rowB = &B->data[k * cB];
            // Inner loop is vectorized by compiler
            for(j = 0; j < cB; j++) {
                rowC[j] += valA * rowB[j];
            }
        }
    }
}

RING_FUNC(ring_tensor_transpose) {
    tensor_t *A = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    tensor_t *C = (tensor_t *)RING_API_GETCPOINTER(2, RING_VM_POINTER_TENSOR);
    int rA = A->rows; int cA = A->cols;
    int i, j;
    
    // Sequential write optimization
    #pragma omp parallel for if(A->size > PARALLEL_THRESHOLD) private(i)
    for(j = 0; j < cA; j++) {
        for(i = 0; i < rA; i++) {
            C->data[j * rA + i] = A->data[i * cA + j];
        }
    }
}

RING_FUNC(ring_tensor_sum) {
    tensor_t *T = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    int axis = (int)RING_API_GETNUMBER(2);
    tensor_t *R = (tensor_t *)RING_API_GETCPOINTER(3, RING_VM_POINTER_TENSOR);
    
    int r, c;
    memset(R->data, 0, R->size * sizeof(double));

    if (axis == 1) { // Sum Rows (Collapse Cols)
        #pragma omp parallel for if(T->rows > 64) private(c)
        for(r=0; r<T->rows; r++) {
            double s = 0;
            double *ptr = &T->data[r * T->cols];
            for(c=0; c<T->cols; c++) s += ptr[c];
            R->data[r] = s;
        }
    } else { // Sum Cols (Collapse Rows)
        // Serial accumulation is safer/faster for small column counts (e.g. bias)
        double *src = T->data;
        double *dst = R->data;
        for(r=0; r<T->rows; r++) {
            double *rowPtr = &src[r * T->cols];
            for(c=0; c<T->cols; c++) {
                dst[c] += rowPtr[c];
            }
        }
    }
}

RING_FUNC(ring_tensor_add_row_vec) {
    tensor_t *A = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    tensor_t *B = (tensor_t *)RING_API_GETCPOINTER(2, RING_VM_POINTER_TENSOR);
    int i, j;
    
    if (A->cols != B->cols) { RING_API_ERROR("Dim Mismatch"); return; }

    #pragma omp parallel for if(A->rows > 32) private(j)
    for(i=0; i<A->rows; i++) {
        double *rowA = &A->data[i * A->cols];
        double *rowB = B->data;
        for(j=0; j<A->cols; j++) rowA[j] += rowB[j];
    }
}

RING_FUNC(ring_tensor_mean) {
    tensor_t *T = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    double sum = 0;
    int i;
    #pragma omp parallel for reduction(+:sum) if(T->size > PARALLEL_THRESHOLD)
    for(i=0; i<T->size; i++) sum += T->data[i];
    RING_API_RETNUMBER(sum / T->size);
}

RING_FUNC(ring_tensor_argmax) {
    tensor_t *T = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    tensor_t *R = (tensor_t *)RING_API_GETCPOINTER(2, RING_VM_POINTER_TENSOR);
    int r, c;
    
    #pragma omp parallel for if(T->rows > 64) private(c)
    for(r=0; r<T->rows; r++) {
        double maxVal = -DBL_MAX;
        int maxIdx = 1;
        int offset = r * T->cols;
        for(c=0; c<T->cols; c++) {
            if(T->data[offset+c] > maxVal) {
                maxVal = T->data[offset+c];
                maxIdx = c + 1;
            }
        }
        R->data[r] = (double)maxIdx;
    }
}

/*
** Slice Rows (Extraction)
** Copies 'count' rows starting from 'start_row' in Src to Dest.
** Optimization: Uses single memcpy because rows are contiguous in memory.
*/
RING_FUNC(ring_tensor_slice_rows) {
    tensor_t *Src, *Dest;
    int start_row, count;
    
    if (RING_API_PARACOUNT != 4) {
        RING_API_ERROR(RING_API_MISS4PARA);
        return;
    }

    Src = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    Dest = (tensor_t *)RING_API_GETCPOINTER(2, RING_VM_POINTER_TENSOR);
    start_row = (int)RING_API_GETNUMBER(3); // 1-based index
    count = (int)RING_API_GETNUMBER(4);     // Number of rows

    // 1. Validation
    if (start_row < 1 || start_row + count - 1 > Src->rows) {
        RING_API_ERROR("Slice Rows: Index out of bounds");
        return;
    }
    if (Dest->cols != Src->cols) {
        RING_API_ERROR("Slice Rows: Column count mismatch");
        return;
    }
    if (Dest->rows != count) {
        RING_API_ERROR("Slice Rows: Destination rows must match count");
        return;
    }

    // 2. Calculation (The Fast Part)
    // Calculate where to start reading in Source (0-based index)
    // Offset = (StartRow - 1) * NumberOfColumns
    int src_offset_idx = (start_row - 1) * Src->cols;
    
    // Calculate total bytes to copy
    // Bytes = NumberOfRowsToCopy * NumberOfColumns * SizeOfDouble
    size_t total_elements = (size_t)count * Src->cols;
    size_t bytes = total_elements * sizeof(double);
    
    // 3. Execution (Single Memcpy)
    // Copy directly from Src memory address to Dest memory address
    memcpy(Dest->data, &Src->data[src_offset_idx], bytes);
}

/*
** Insert Rows (Injection)
** Copies all rows from Src into Dest starting at 'start_row'.
*/
RING_FUNC(ring_tensor_insert_rows) {
    tensor_t *Dest, *Src;
    int start_row;

    if (RING_API_PARACOUNT != 3) {
        RING_API_ERROR(RING_API_MISS3PARA);
        return;
    }

    Dest = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    Src = (tensor_t *)RING_API_GETCPOINTER(2, RING_VM_POINTER_TENSOR);
    start_row = (int)RING_API_GETNUMBER(3); // 1-based index

    // Validation
    if (start_row < 1 || start_row + Src->rows - 1 > Dest->rows) {
        RING_API_ERROR("Insert Rows: Index out of bounds or Src too big");
        return;
    }
    if (Dest->cols != Src->cols) {
        RING_API_ERROR("Insert Rows: Column count mismatch");
        return;
    }

    // Math
    size_t offset_elems = (size_t)(start_row - 1) * Dest->cols;
    size_t copy_elems   = (size_t)Src->rows * Src->cols;
    size_t copy_bytes   = copy_elems * sizeof(double);

    // Blazing fast copy
    memcpy(&Dest->data[offset_elems], Src->data, copy_bytes);
}

/* ==================================================================== */
/* --- 5. NLP & TRANSFORMER KERNELS (OpenMP Optimized) ---------------- */
/* ==================================================================== */

RING_FUNC(ring_tensor_embedding_forward) {
    tensor_t *Ind = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    tensor_t *W   = (tensor_t *)RING_API_GETCPOINTER(2, RING_VM_POINTER_TENSOR);
    tensor_t *Out = (tensor_t *)RING_API_GETCPOINTER(3, RING_VM_POINTER_TENSOR);
    
    int total = Ind->size;
    int dim = W->cols;
    int vocab = W->rows;
    int i;
    
    #pragma omp parallel for if(total > 32)
    for(i=0; i<total; i++) {
        int idx = (int)Ind->data[i];
        if (idx < 0) idx = 0; 
        if (idx >= vocab) idx = vocab - 1;
        
        memcpy(&Out->data[i * dim], &W->data[idx * dim], dim * sizeof(double));
    }
}

RING_FUNC(ring_tensor_embedding_backward) {
    tensor_t *Ind = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    tensor_t *GOut = (tensor_t *)RING_API_GETCPOINTER(2, RING_VM_POINTER_TENSOR);
    tensor_t *GW = (tensor_t *)RING_API_GETCPOINTER(3, RING_VM_POINTER_TENSOR);
    
    int total = Ind->size;
    int dim = GW->cols;
    int i, j;
    
    // Serial execution is safer here due to Scatter Add collisions
    for(i=0; i<total; i++) {
        int idx = (int)Ind->data[i];
        if (idx < 0 || idx >= GW->rows) continue;
        
        double *g_src = &GOut->data[i * dim];
        double *g_dst = &GW->data[idx * dim];
        
        for(j=0; j<dim; j++) g_dst[j] += g_src[j];
    }
}

RING_FUNC(ring_tensor_layernorm) {
    tensor_t *X = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    tensor_t *G = (tensor_t *)RING_API_GETCPOINTER(2, RING_VM_POINTER_TENSOR);
    tensor_t *B = (tensor_t *)RING_API_GETCPOINTER(3, RING_VM_POINTER_TENSOR);
    tensor_t *Y = (tensor_t *)RING_API_GETCPOINTER(4, RING_VM_POINTER_TENSOR);
    double eps = RING_API_GETNUMBER(5);
    int r, c, rows = X->rows, cols = X->cols;
    
    #pragma omp parallel for if(rows > 32) private(c)
    for(r=0; r<rows; r++) {
        double mean = 0, var = 0;
        double *px = &X->data[r*cols];
        double *py = &Y->data[r*cols];
        double invStd;
        
        for(c=0; c<cols; c++) mean += px[c];
        mean /= cols;
        
        for(c=0; c<cols; c++) var += (px[c]-mean)*(px[c]-mean);
        var /= cols;
        
        invStd = 1.0 / sqrt(var + eps);
        for(c=0; c<cols; c++) {
            py[c] = ((px[c] - mean) * invStd * G->data[c]) + B->data[c];
        }
    }
}

RING_FUNC(ring_tensor_attention_fast) {
    tensor_t *Q = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    tensor_t *K = (tensor_t *)RING_API_GETCPOINTER(2, RING_VM_POINTER_TENSOR);
    tensor_t *V = (tensor_t *)RING_API_GETCPOINTER(3, RING_VM_POINTER_TENSOR);
    tensor_t *Out = (tensor_t *)RING_API_GETCPOINTER(4, RING_VM_POINTER_TENSOR);
    double scale = RING_API_GETNUMBER(5);
    
    int seq = Q->rows;
    int dim = Q->cols;
    int i, j, k;
    
    // Allocating temp memory inside parallel region is expensive.
    // Optimization: Do parallel loop but handle malloc carefully or use small stack buffer if dim is small.
    // For large sequence, use malloc.
    
    #pragma omp parallel private(i, j, k)
    {
        double *scores = (double *)malloc(seq * sizeof(double));
        if(scores) {
            #pragma omp for
            for(i=0; i<seq; i++) {
                double *q_row = &Q->data[i*dim];
                double *out_row = &Out->data[i*dim];
                double maxVal = -DBL_MAX;
                double sum = 0;
                double invSum;
                
                // QK^T
                for(j=0; j<seq; j++) {
                    double *k_row = &K->data[j*dim];
                    double dot = 0;
                    for(k=0; k<dim; k++) dot += q_row[k] * k_row[k];
                    scores[j] = dot * scale;
                }
                
                // Softmax
                for(j=0; j<seq; j++) if(scores[j] > maxVal) maxVal = scores[j];
                for(j=0; j<seq; j++) {
                    scores[j] = exp(scores[j] - maxVal);
                    sum += scores[j];
                }
                invSum = 1.0/sum;
                for(j=0; j<seq; j++) scores[j] *= invSum;
                
                // Score * V
                memset(out_row, 0, dim * sizeof(double));
                for(j=0; j<seq; j++) {
                    double s = scores[j];
                    double *v_row = &V->data[j*dim];
                    for(k=0; k<dim; k++) out_row[k] += s * v_row[k];
                }
            }
            free(scores);
        }
    }
}

RING_FUNC(ring_tensor_attention_causal) {
    tensor_t *Q = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    tensor_t *K = (tensor_t *)RING_API_GETCPOINTER(2, RING_VM_POINTER_TENSOR);
    tensor_t *V = (tensor_t *)RING_API_GETCPOINTER(3, RING_VM_POINTER_TENSOR);
    tensor_t *Out = (tensor_t *)RING_API_GETCPOINTER(4, RING_VM_POINTER_TENSOR);
    double scale = RING_API_GETNUMBER(5);
    
    int seq = Q->rows;
    int dim = Q->cols;
    int i, j, k;
    
    #pragma omp parallel private(i, j, k)
    {
        double *scores = (double *)malloc(seq * sizeof(double));
        if(scores) {
            #pragma omp for
            for(i=0; i<seq; i++) {
                double *q_row = &Q->data[i*dim];
                double *out_row = &Out->data[i*dim];
                double maxVal = -1e9;
                double sum = 0;
                double invSum;
                
                // Masked QK^T
                for(j=0; j<seq; j++) {
                    if (j > i) { scores[j] = -1e9; continue; }
                    
                    double *k_row = &K->data[j*dim];
                    double dot = 0;
                    for(k=0; k<dim; k++) dot += q_row[k] * k_row[k];
                    scores[j] = dot * scale;
                }
                
                // Softmax
                for(j=0; j<seq; j++) if(scores[j] > maxVal) maxVal = scores[j];
                for(j=0; j<seq; j++) {
                    scores[j] = exp(scores[j] - maxVal);
                    sum += scores[j];
                }
                invSum = 1.0/sum;
                for(j=0; j<seq; j++) scores[j] *= invSum;
                
                // Score * V
                memset(out_row, 0, dim * sizeof(double));
                for(j=0; j<seq; j++) {
                    double s = scores[j];
                    if(s < 1e-9) continue;
                    double *v_row = &V->data[j*dim];
                    for(k=0; k<dim; k++) out_row[k] += s * v_row[k];
                }
            }
            free(scores);
        }
    }
}

RING_FUNC(ring_tensor_select_columns) {
    tensor_t *Src = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    tensor_t *Dest = (tensor_t *)RING_API_GETCPOINTER(2, RING_VM_POINTER_TENSOR);
    int start = (int)RING_API_GETNUMBER(3);
    int count = (int)RING_API_GETNUMBER(4);
    int r;
    int src_off;
    size_t bytes;
    
    if (start < 1 || start + count - 1 > Src->cols) { RING_API_ERROR("Bounds"); return; }
    
    src_off = start - 1;
    bytes = count * sizeof(double);
    
    #pragma omp parallel for if(Src->rows > 64)
    for(r=0; r<Src->rows; r++) {
        memcpy(&Dest->data[r*Dest->cols], &Src->data[r*Src->cols + src_off], bytes);
    }
}

RING_FUNC(ring_tensor_insert_columns) {
    tensor_t *Dest = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    tensor_t *Src = (tensor_t *)RING_API_GETCPOINTER(2, RING_VM_POINTER_TENSOR);
    int start = (int)RING_API_GETNUMBER(3);
    int r;
    int dest_off;
    size_t bytes;
    
    if (start < 1 || start + Src->cols - 1 > Dest->cols) { RING_API_ERROR("Bounds"); return; }
    
    dest_off = start - 1;
    bytes = Src->cols * sizeof(double);
    
    #pragma omp parallel for if(Dest->rows > 64)
    for(r=0; r<Dest->rows; r++) {
        memcpy(&Dest->data[r*Dest->cols + dest_off], &Src->data[r*Src->cols], bytes);
    }
}

/*
** Fused Batch Attention (Fast & Causal)
** Handles [Batch, Seq, Dim] in one go using OpenMP.
** 
** Params: 
** 1. Q, 2. K, 3. V, 4. Out (All Flattened: Batch*Seq*Dim)
** 5. Scale, 6. BatchSize, 7. SeqLen, 8. Dim, 9. IsCausal (1=Yes, 0=No)
*/
RING_FUNC(ring_tensor_attention_batch) {
    tensor_t *Q = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    tensor_t *K = (tensor_t *)RING_API_GETCPOINTER(2, RING_VM_POINTER_TENSOR);
    tensor_t *V = (tensor_t *)RING_API_GETCPOINTER(3, RING_VM_POINTER_TENSOR);
    tensor_t *Out = (tensor_t *)RING_API_GETCPOINTER(4, RING_VM_POINTER_TENSOR);
    
    double scale  = RING_API_GETNUMBER(5);
    int batch     = (int)RING_API_GETNUMBER(6);
    int seq       = (int)RING_API_GETNUMBER(7);
    int dim       = (int)RING_API_GETNUMBER(8);
    int is_causal = (int)RING_API_GETNUMBER(9);
    
    int b, i, j, k;
    int stride = seq * dim; // Size of one sentence block
    
    // Parallelize over Batches (The most efficient way)
    #pragma omp parallel for private(b, i, j, k)
    for (b = 0; b < batch; b++) {
        
        // Pointers to the start of the current batch
        double *q_base = &Q->data[b * stride];
        double *k_base = &K->data[b * stride];
        double *v_base = &V->data[b * stride];
        double *o_base = &Out->data[b * stride];
        
        // Temp scores buffer (per thread/batch)
        double *scores = (double *)malloc(seq * sizeof(double));
        
        if (scores) {
            // Loop over Sequence (Rows)
            for(i = 0; i < seq; i++) {
                double *q_row = &q_base[i * dim];
                double *o_row = &o_base[i * dim];
                
                // 1. Q . K^T
                for(j = 0; j < seq; j++) {
                    // Causal Masking Logic
                    if (is_causal && j > i) {
                        scores[j] = -1e9;
                        continue;
                    }
                    
                    double *k_row = &k_base[j * dim];
                    double dot = 0.0;
                    for(k = 0; k < dim; k++) {
                        dot += q_row[k] * k_row[k];
                    }
                    scores[j] = dot * scale;
                }
                
                // 2. Softmax
                double maxVal = -1e9;
                for(j=0; j<seq; j++) if(scores[j] > maxVal) maxVal = scores[j];
                
                double sum = 0.0;
                for(j=0; j<seq; j++) {
                    scores[j] = exp(scores[j] - maxVal);
                    sum += scores[j];
                }
                
                double invSum = 1.0 / sum;
                for(j=0; j<seq; j++) scores[j] *= invSum;
                
                // 3. Scores . V
                memset(o_row, 0, dim * sizeof(double));
                for(j=0; j<seq; j++) {
                    double s = scores[j];
                    if (s < 1e-9) continue; // Optimization
                    
                    double *v_row = &v_base[j * dim];
                    for(k=0; k<dim; k++) {
                        o_row[k] += s * v_row[k];
                    }
                }
            }
            free(scores);
        }
    }
}

/* ==================================================================== */
/* --- 6. OPTIMIZERS -------------------------------------------------- */
/* ==================================================================== */

RING_FUNC(ring_tensor_update_adam) {
    tensor_t *W = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    tensor_t *G = (tensor_t *)RING_API_GETCPOINTER(2, RING_VM_POINTER_TENSOR);
    tensor_t *M = (tensor_t *)RING_API_GETCPOINTER(3, RING_VM_POINTER_TENSOR);
    tensor_t *V = (tensor_t *)RING_API_GETCPOINTER(4, RING_VM_POINTER_TENSOR);
    
    double lr = RING_API_GETNUMBER(5);
    double b1 = RING_API_GETNUMBER(6);
    double b2 = RING_API_GETNUMBER(7);
    double eps= RING_API_GETNUMBER(8);
    int t = (int)RING_API_GETNUMBER(9);
    
    double corr1 = 1.0 - pow(b1, t);
    double corr2 = 1.0 - pow(b2, t);
    if(corr1 < 1e-9) corr1 = 1e-9;
    if(corr2 < 1e-9) corr2 = 1e-9;
    
    int i;
    int size = W->size;
    
    // Ensure loop variable is private or declared inside
    #pragma omp parallel for if(size > PARALLEL_THRESHOLD)
    for(i=0; i<size; i++) {
        double g = G->data[i];
        
        M->data[i] = (b1 * M->data[i]) + ((1.0 - b1) * g);
        V->data[i] = (b2 * V->data[i]) + ((1.0 - b2) * g * g);
        
        double m_hat = M->data[i] / corr1;
        double v_hat = V->data[i] / corr2;
        if(v_hat < 0) v_hat = 0;
        
        W->data[i] -= (lr * m_hat) / (sqrt(v_hat) + eps);
    }
}

RING_FUNC(ring_tensor_update_sgd) {
    tensor_t *W = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    tensor_t *G = (tensor_t *)RING_API_GETCPOINTER(2, RING_VM_POINTER_TENSOR);
    double lr = RING_API_GETNUMBER(3);
    int i;
    int size = W->size;
    
    #pragma omp parallel for if(size > PARALLEL_THRESHOLD)
    for(i=0; i<size; i++) W->data[i] -= (lr * G->data[i]);
}

RING_FUNC(ring_tensor_dropout) {
    tensor_t *T = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    double rate = RING_API_GETNUMBER(2);
    double scale = 1.0 / (1.0 - rate);
    int i;
    // Serial
    for(i=0; i<T->size; i++) {
        if ((double)rand() / RAND_MAX < rate) T->data[i] = 0.0;
        else T->data[i] *= scale;
    }
}

/* ==================================================================== */
/* --- 7. UTILS ------------------------------------------------------- */
/* ==================================================================== */

RING_FUNC(ring_tensor_get_cores) {
    int cores = 1;
    #ifdef _OPENMP
    cores = omp_get_num_procs();
    #endif
    RING_API_RETNUMBER(cores);
}

RING_FUNC(ring_tensor_set_threads) {
    int n = (int)RING_API_GETNUMBER(1);
    #ifdef _OPENMP
    omp_set_num_threads(n);
    #endif
}

/*
** ====================================================================
** --- CROSS ENTROPY KERNELS (FUSED) ----------------------------------
** ====================================================================
*/

/*
** Calculate CrossEntropy Loss
** Formula: -Sum(Target * Log(Probs)) / ActiveSamples
** Handles Masking: If Target row is all zeros, it is ignored.
*/
RING_FUNC(ring_tensor_crossentropy_loss) {
    tensor_t *Probs, *Targets;
    int r, c;
    double total_loss = 0.0;
    int active_samples = 0;
    double eps = 1e-9;

    if (RING_API_PARACOUNT != 2) {
        RING_API_ERROR(RING_API_MISS2PARA);
        return;
    }

    Probs   = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    Targets = (tensor_t *)RING_API_GETCPOINTER(2, RING_VM_POINTER_TENSOR);

    if (Probs->rows != Targets->rows || Probs->cols != Targets->cols) {
        RING_API_ERROR("CrossEntropy: Size mismatch");
        return;
    }

    int rows = Probs->rows;
    int cols = Probs->cols;

    // Use OpenMP reduction for fast summation
    #pragma omp parallel for reduction(+:total_loss, active_samples)
    for (r = 0; r < rows; r++) {
        
        // 1. Check if this row is active (Has a target)
        int target_idx = -1;
        double *row_tgt = &Targets->data[r * cols];
        
        for (c = 0; c < cols; c++) {
            if (row_tgt[c] > 0.5) {
                target_idx = c;
                break;
            }
        }

        // 2. If Active, Calculate Loss
        if (target_idx != -1) {
            double p = Probs->data[r * cols + target_idx];
            // Clip to prevent NaN
            if (p < eps) p = eps;
            if (p > 1.0) p = 1.0;
            
            total_loss += -log(p);
            active_samples++;
        }
    }

    if (active_samples == 0) {
        RING_API_RETNUMBER(0.0);
    } else {
        RING_API_RETNUMBER(total_loss / active_samples);
    }
}

/*
** Calculate CrossEntropy Gradient
** Formula: (Probs - Targets) / ActiveSamples
** Handles Masking: Zero out gradients for padding rows.
*/
RING_FUNC(ring_tensor_crossentropy_backward) {
    tensor_t *Probs, *Targets, *Grad;
    int active_samples = 0;
    int r, c;

    if (RING_API_PARACOUNT != 3) {
        RING_API_ERROR(RING_API_MISS3PARA);
        return;
    }

    Probs   = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    Targets = (tensor_t *)RING_API_GETCPOINTER(2, RING_VM_POINTER_TENSOR);
    Grad    = (tensor_t *)RING_API_GETCPOINTER(3, RING_VM_POINTER_TENSOR); // Output

    int rows = Probs->rows;
    int cols = Probs->cols;

    // 1. Count Active Samples First (Need this for normalization)
    // Serial loop is fast enough for just counting, or use OMP reduction
    for (r = 0; r < rows; r++) {
        double *row_tgt = &Targets->data[r * cols];
        for (c = 0; c < cols; c++) {
            if (row_tgt[c] > 0.5) {
                active_samples++;
                break;
            }
        }
    }

    double scale = (active_samples > 0) ? (1.0 / active_samples) : 0.0;

    // 2. Calculate Gradients with Masking
    #pragma omp parallel for private(c)
    for (r = 0; r < rows; r++) {
        double *p_row = &Probs->data[r * cols];
        double *t_row = &Targets->data[r * cols];
        double *g_row = &Grad->data[r * cols];
        
        // Check if active row
        int is_active = 0;
        for(c=0; c<cols; c++) if(t_row[c] > 0.5) { is_active=1; break; }

        if (is_active) {
            // Gradient = (P - T) * Scale
            for (c = 0; c < cols; c++) {
                g_row[c] = (p_row[c] - t_row[c]) * scale;
            }
        } else {
            // Masked (Padding): Zero out gradient
            memset(g_row, 0, cols * sizeof(double));
        }
    }
}

/*
** Bulk Set from List (Turbo Loading)
** Copies a Ring List directly into Tensor Memory.
** Eliminates the overhead of calling setVal() thousands of times.
*/
RING_FUNC(ring_tensor_set_from_list) {
    tensor_t *T;
    List *pList;
    int i, nSize, nLimit;

    if (RING_API_PARACOUNT != 2) {
        RING_API_ERROR(RING_API_MISS2PARA);
        return;
    }

    // 1. Get Tensor
    T = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    
    // 2. Get List
    if (!RING_API_ISLIST(2)) {
        RING_API_ERROR(RING_API_BADPARATYPE);
        return;
    }
    pList = RING_API_GETLIST(2);
    
    // 3. Determine safe copy limit
    nSize = ring_list_getsize(pList);
    nLimit = T->size;
    if (nSize < nLimit) nLimit = nSize;

    // 4. Turbo Copy Loop (Inside C)
    for(i = 1; i <= nLimit; i++) {
        if (ring_list_isnumber(pList, i)) {
            T->data[i-1] = ring_list_getdouble(pList, i);
        } else {
            T->data[i-1] = 0.0; // Default for non-numbers
        }
    }
}

/*
** Set One-Hot (Scatter)
** Takes a List of Indices and sets T[row, index] = value.
** Assumes Tensor is already zeroed (or overwrites).
** 1-based Indexing for both List and Values.
*/
RING_FUNC(ring_tensor_set_one_hot) {
    tensor_t *T;
    List *pList;
    double val;
    int i, nListSize, nMaxRows;

    if (RING_API_PARACOUNT != 3) {
        RING_API_ERROR(RING_API_MISS3PARA);
        return;
    }

    T = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    
    if (!RING_API_ISLIST(2)) {
        RING_API_ERROR("Param 2 must be a List of Indices");
        return;
    }
    pList = RING_API_GETLIST(2);
    val = RING_API_GETNUMBER(3);

    nListSize = ring_list_getsize(pList);
    nMaxRows = T->rows; // Total flattened rows
    
    // Safety: Process min(ListSize, TensorRows)
    if (nListSize > nMaxRows) nListSize = nMaxRows;

    // Parallel Scatter
    #pragma omp parallel for if(nListSize > 5000)
    for(i = 1; i <= nListSize; i++) {
        if (ring_list_isnumber(pList, i)) {
            // Get Column Index (Class ID)
            int col_idx = (int)ring_list_getdouble(pList, i);
            
            // Validate Bounds (1-based)
            if (col_idx >= 1 && col_idx <= T->cols) {
                // Map to 0-based C array
                // Row (i-1), Col (col_idx-1)
                T->data[(i-1) * T->cols + (col_idx-1)] = val;
            }
        }
    }
}

/* --- INIT --- */
RING_LIBINIT {
    RING_API_REGISTER("tensor_init", ring_tensor_init);
    RING_API_REGISTER("tensor_reshape", ring_tensor_reshape);
    RING_API_REGISTER("tensor_matmul_batch", ring_tensor_matmul_batch); 
    RING_API_REGISTER("tensor_set", ring_tensor_set);
    RING_API_REGISTER("tensor_get", ring_tensor_get);
    
    RING_API_REGISTER("tensor_add", ring_tensor_add);
    RING_API_REGISTER("tensor_sub", ring_tensor_sub);
    RING_API_REGISTER("tensor_mul_elem", ring_tensor_mul_elem);
    RING_API_REGISTER("tensor_div", ring_tensor_div);
    RING_API_REGISTER("tensor_scalar_mul", ring_tensor_scalar_mul);
    RING_API_REGISTER("tensor_add_scalar", ring_tensor_add_scalar);
    RING_API_REGISTER("tensor_sub_scalar", ring_tensor_sub_scalar);
    
    RING_API_REGISTER("tensor_fill", ring_tensor_fill);
    RING_API_REGISTER("tensor_random", ring_tensor_random);
    RING_API_REGISTER("tensor_square", ring_tensor_square);
    RING_API_REGISTER("tensor_sqrt", ring_tensor_sqrt);
    RING_API_REGISTER("tensor_exp", ring_tensor_exp);
    
    RING_API_REGISTER("tensor_matmul", ring_tensor_matmul);
    RING_API_REGISTER("tensor_transpose", ring_tensor_transpose);
    RING_API_REGISTER("tensor_sum", ring_tensor_sum);
    RING_API_REGISTER("tensor_mean", ring_tensor_mean);
    RING_API_REGISTER("tensor_argmax", ring_tensor_argmax);
    RING_API_REGISTER("tensor_add_row_vec", ring_tensor_add_row_vec);

    RING_API_REGISTER("tensor_sigmoid", ring_tensor_sigmoid);
    RING_API_REGISTER("tensor_sigmoid_prime", ring_tensor_sigmoid_prime);
    RING_API_REGISTER("tensor_tanh", ring_tensor_tanh);
    RING_API_REGISTER("tensor_tanh_prime", ring_tensor_tanh_prime);
    RING_API_REGISTER("tensor_relu", ring_tensor_relu);
    RING_API_REGISTER("tensor_relu_prime", ring_tensor_relu_prime);
    RING_API_REGISTER("tensor_softmax", ring_tensor_softmax);
    
    RING_API_REGISTER("tensor_embedding_forward", ring_tensor_embedding_forward);
    RING_API_REGISTER("tensor_embedding_backward", ring_tensor_embedding_backward);
    RING_API_REGISTER("tensor_layernorm", ring_tensor_layernorm);
    RING_API_REGISTER("tensor_attention_fast", ring_tensor_attention_fast);
    RING_API_REGISTER("tensor_attention_causal", ring_tensor_attention_causal);
    RING_API_REGISTER("tensor_select_columns", ring_tensor_select_columns);
    RING_API_REGISTER("tensor_insert_columns", ring_tensor_insert_columns);
    RING_API_REGISTER("tensor_attention_batch", ring_tensor_attention_batch);
    
    // --- NEW ---
    RING_API_REGISTER("tensor_slice_rows", ring_tensor_slice_rows);
    RING_API_REGISTER("tensor_insert_rows", ring_tensor_insert_rows);
    
    RING_API_REGISTER("tensor_update_sgd", ring_tensor_update_sgd);
    RING_API_REGISTER("tensor_update_adam", ring_tensor_update_adam);
    RING_API_REGISTER("tensor_dropout", ring_tensor_dropout);

    RING_API_REGISTER("tensor_crossentropy_loss", ring_tensor_crossentropy_loss);
    RING_API_REGISTER("tensor_crossentropy_backward", ring_tensor_crossentropy_backward);

    RING_API_REGISTER("tensor_get_cores", ring_tensor_get_cores);
    RING_API_REGISTER("tensor_set_threads", ring_tensor_set_threads);

    RING_API_REGISTER("tensor_set_from_list", ring_tensor_set_from_list);
    RING_API_REGISTER("tensor_set_one_hot", ring_tensor_set_one_hot);
    
    #ifdef _OPENMP
    omp_set_num_threads(omp_get_num_procs());
    #endif
}