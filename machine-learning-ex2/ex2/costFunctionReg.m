function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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
h = zeros(m, 1);
tmp = 0;
tmp_grad = 0;
% Calculate cost
for i = 1:m
	h(i) = sigmoid(X(i, :)*theta);
	tmp = tmp + (-1)*(y(i)*log(h(i)) + (1-y(i))*log(1-h(i)));
end
J = tmp*2;

%Calculate gradient
for j = 1:size(grad, 1)
	if (j == 1)		
		for i = 1:m
			tmp_grad = tmp_grad + (h(i) - y(i))*X(i, j);
		end
		grad(j) = tmp_grad/m;		
	else
		for i = 1:m
			tmp_grad = tmp_grad + (h(i) - y(i))*X(i, j);
		end
		grad(j) = (tmp_grad+lambda*theta(j))/m;	
		J = J + lambda*theta(j)*theta(j);
	endif
	tmp_grad = 0;
end
J = J/(2*m);
% =============================================================

end
