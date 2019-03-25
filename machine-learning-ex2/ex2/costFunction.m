function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
h = zeros(m, 1);
tmp = 0;
tmp_grad = 0;
% Calculate cost
for i = 1:m
	h(i) = sigmoid(X(i, :)*theta);
	tmp = tmp + (-1)*(y(i)*log(h(i)) + (1-y(i))*log(1-h(i)));
end
J = tmp/m;

%Calculate gradient
for j = 1:size(grad, 1)
	for i = 1:m
		tmp_grad = tmp_grad + (h(i) - y(i))*X(i, j);
	end
	grad(j) = tmp_grad/m;
	tmp_grad = 0;
end
% =============================================================
end
