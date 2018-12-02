function y = tv_norm(u, sphere_flag, incNP, weights_dx, weights_dy)
if sphere_flag
    if nargin>3 
        [dx, dy] = gradient_op_sphere(u, incNP, weights_dx, weights_dy);
    else
        [dx, dy] = gradient_op_sphere(u, incNP);
    end
else
    if nargin>3 
        [dx, dy] = gradient_op(u, weights_dx, weights_dy);
    else
        [dx, dy] = gradient_op(u);
    end
end
temp = sqrt(abs(dx).^2 + abs(dy).^2);
y = sum(temp(:));

end