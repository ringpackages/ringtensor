# The Main File

load "extensions/ringtensor/ringtensor.ring"

func main
	? "
  _____  _             _______
 |  __ \(_)           |__   __|
 | |__) |_ _ __   __ _   | | ___ _ __  ___  ___  _ __
 |  _  /| | '_ \ / _` |  | |/ _ \ '_ \/ __|/ _ \| '__|
 | | \ \| | | | | (_| |  | |  __/ | | \__ \ (_) | |
 |_|  \_\_|_| |_|\__, |  |_|\___|_| |_|___/\___/|_|
                  __/ |
                 |___/
    "
	? "Welcome to RingTensor v1.0.0"
	? "High-Performance C Extension for Deep Learning in Ring"
	? copy("=", 50)
	? "Available Functions Summary:"
	? "1. Matrix Operations: tensor_add, tensor_sub, tensor_mul_elem, tensor_div"
	? "2. Linear Algebra:    tensor_matmul, tensor_transpose, tensor_sum"
	? "3. Transformations:   tensor_square, tensor_sqrt, tensor_exp"
	? "4. Activations:       tensor_sigmoid, tensor_tanh, tensor_relu, tensor_softmax"
	? "5. Optimizers:        tensor_update_sgd, tensor_update_adam"
	? "6. Regularizers:      tensor_dropout"
	? "7. for more info:     read README.md"	
	? copy("=", 50)