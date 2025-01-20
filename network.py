def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output

def train(network, loss, loss_prime, x_train, y_train, epochs=1000, learning_rate=0.01, verbose=True):
    best_error = float('inf')
    for e in range(epochs):
        error = 0
        for x, y in zip(x_train, y_train):
            output = predict(network, x)
            error += loss(y, output)
            
            grad = loss_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)
                
        error /= len(x_train)
        
        # track best model and provide more detailed progress
        if error < best_error:
            best_error = error
            
        if verbose and (e + 1) % max(1, epochs // 20) == 0:
            print(f"Epoch {e + 1}/{epochs}, error={error:.6f}, best={best_error:.6f}")
