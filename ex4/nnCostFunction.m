function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



X = [ones(m, 1) X];
z2 = X * Theta1';
a2 = sigmoid(z2);
a2 = [ones(m, 1) a2];
z3 = a2 * Theta2';
h_theta = sigmoid(z3);

%format y
y_new = zeros(m, num_labels);
for i=1:m,
  y_new(i, y(i)) = 1;
end

%loop
for i=1: m,
  for j=1:num_labels,
    J += -y_new(i, j)*log(h_theta(i,j)) - (1-y_new(i, j))*log(1-h_theta(i, j));
  endfor
end

    %or 
    %J = (1/m) * sum ( sum ( (-y_new) .* log(h_theta) - (1-y_new) .* log(1-h_theta) ));


%Regularization
Theta1_new = Theta1;
Theta2_new = Theta2;
Theta1_new(:, 1) = 0;
Theta2_new(:, 1) = 0;

Reg = sum(sum(Theta1_new.^2)) + sum(sum(Theta2_new.^2));

% Regularized cost function 
J = (1/m) * J + lambda * Reg /(2*m);


delta1 = zeros(size(Theta1));
delta2 = zeros(size(Theta2));
for i = 1: m,
  a1t = X(i, :);
  a2t = a2(i, :);
  z2t = [1 z2(i,:)]';
  % Vi phai theo chieu trong phep tinh cua d2 nen them cot 1
  a3t = h_theta(i, :);
  yt = y_new(i, :);
  d3 = a3t - yt;
  d2 = Theta2' * d3' .* sigmoidGradient(z2t);
  
  delta1 = delta1 + d2(2:end) * a1t;
  delta2 = delta2 + d3' * a2t;
end

Theta1_grad = (1/m) * delta1 + (lambda/m) * Theta1_new;
Theta2_grad = (1/m) * delta2 + (lambda/m) * Theta2_new;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
