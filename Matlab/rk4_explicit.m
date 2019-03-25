% Runge-Kutta order 4 integration

function x_next = rk4_explicit(fun, x, u, t_step)
    %import casadi.*      
    k1 = fun(x, u);
	k2 = fun(x + t_step*k1/2, u);
    k3 = fun(x + t_step*k2/2, u);
    k4 = fun(x + t_step*k3, u);
    
    x_next = x + (t_step/6)*(k1 + 2*k2 + 2*k3 + k4); 
end