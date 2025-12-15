/*
** RingTensor Extension
** Description: High-Performance Tensor Operations for RingML
** Header File
*/

#ifndef RING_TENSOR_H
#define RING_TENSOR_H

#include "ring.h"
#include <math.h>
#include <stdlib.h>
#include <float.h>
#include <time.h> 

#ifdef _WIN32
#define RING_EXPORT __declspec(dllexport)
#else
#define RING_EXPORT extern
#endif

/* --- Prototypes --- */

/* 1. Basic Math & Element-Wise */
RING_FUNC(ring_tensor_add);
RING_FUNC(ring_tensor_sub);
RING_FUNC(ring_tensor_mul_elem);
RING_FUNC(ring_tensor_div);
RING_FUNC(ring_tensor_scalar_mul);
RING_FUNC(ring_tensor_add_scalar);

/* 2. Math Transformations */
RING_FUNC(ring_tensor_square);
RING_FUNC(ring_tensor_sqrt);
RING_FUNC(ring_tensor_exp);
RING_FUNC(ring_tensor_fill);
RING_FUNC(ring_tensor_random); /* Uniform 0-1 */

/* 3. Matrix Operations */
RING_FUNC(ring_tensor_matmul);
RING_FUNC(ring_tensor_transpose);
RING_FUNC(ring_tensor_sum); 
RING_FUNC(ring_tensor_mean);

/* 4. Activations */
RING_FUNC(ring_tensor_sigmoid);
RING_FUNC(ring_tensor_sigmoid_prime);
RING_FUNC(ring_tensor_tanh);
RING_FUNC(ring_tensor_tanh_prime);
RING_FUNC(ring_tensor_relu);
RING_FUNC(ring_tensor_relu_prime);
RING_FUNC(ring_tensor_softmax);

/* 5. Optimizers & Regularization */
RING_FUNC(ring_tensor_update_sgd);
RING_FUNC(ring_tensor_update_adam);
RING_FUNC(ring_tensor_dropout);

/* 6. Utilities */
RING_FUNC(ring_tensor_argmax);

RING_EXPORT void ringlib_init(RingState *pRingState);

#endif