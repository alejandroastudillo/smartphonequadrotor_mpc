%QUADROTOR_NMPC - Development of a Nonlinear MPC for a Quadrotor
% Developed by Alejandro ASTUDILLO VIGOYA, 2019
% MECO Research Team - KU Leuven

        clear; clc; close all;

    %--- Add the utility, CasADi and SpatialV2 directories to the Path ---%
        addpath(genpath('../utils'));
        
        CasADi_path = '/home/alejandro_av/Alejandro/PhD_Stuff/Casadi_source/build/install_matlab/matlab'; %~/Documents/casadi
        matlab2tikz_path = '/home/alejandro_av/Documents/matlab2tikz/src';
        
            addpath(CasADi_path);
            addpath(genpath(matlab2tikz_path));

        clear CasADi_path matlab2tikz_path
        
    %--- Import CasADi ---%
        import casadi.*
        
%% Quadrotor Parameters

        g = 9.81;
        m = 1.5680;
        Ixx = 0.0135;
        Iyy = 0.0124;
        Izz = 0.0336;
        
        %--- Number of states ---%
            nx = 12;
            
        %--- Number of inputs ---%
            nu = 4;
            
        %--- Initial states ---%
            initial_x = zeros(nx,1);
            
        %--- Desired final position ---%
            final_pos = [0.01; 0.01; 0.01];
        
%% Define Options

    ocp_options = struct;

    %--- OCP options ---%
        model = 'non-linear';                    % 'non-linear' or 'linearized'
        dt = 0.010;                             % sample time [s]
        N_horizon = 20;                         % discretization points (Horizon)
        
        U_max = [20; 15; 15; 15];               % input limits
        U_min = -U_max; 
        
        Angle_max = [1; 20; 20].*pi/180;        % [psi, theta, phi] limits
        Angle_min = -Angle_max;
        
        kkt_tol = 1e-1; % 1e-3
        
    %--- Simulation options ---%
        N_sim = 250;                         % samples to simulate 170, 300, 400
        
    %--- Just-in-time compilation options ---%
        jit_opts = struct;
        jit_opts.jit = true;
        jit_opts.jit_options.compiler_flags = {'-O3'};
        jit_opts.jit_options.verbose = true;
        jit_opts.jit_options.compiler = 'ccache gcc';
        %jit_opts.jit_options.temp_suffix = false;
        jit_opts.jit_temp_suffix = false;
        jit_opts.compiler = 'shell';

        
%% OCP Definition

    %--- Define Opti environment ---%
        opti = Opti();
        
    %--- Decision variables ---%        
        X = opti.variable(nx,N_horizon+1);
        U = opti.variable(nu,N_horizon);
        
    %--- Parameters ---%
        X_init = opti.parameter(nx);        % This parameter represents the state-feedback signal (which changes every dt [s])
        pos_end = opti.parameter(3);        % Parameter that defines the desired final position

    %--- Quadrotor dynamics ---%
        X_k = MX.sym('X_k',nx,1);
        U_k = MX.sym('U_k',nu,1);
        
        if strcmp(model,'non-linear')
        
            dx = X_k(2);
            ddx = ((U_k(1)/m)*sin(X_k(9))) + g*sin(X_k(9));
            dy = X_k(4);
            ddy = ((U_k(1)/m)*cos(X_k(9))*sin(X_k(11))) + g*cos(X_k(9))*sin(X_k(11));
            dz = X_k(6);
            ddz = ((U_k(1)/m)*cos(X_k(9))*cos(X_k(11))) + (g*cos(X_k(9))*cos(X_k(11))) - g;
            dpsi = X_k(8);
            ddpsi = (((Ixx-Iyy)/Izz)*X_k(12)*X_k(10)) + (U_k(2)/Izz);
            dtheta = X_k(10);
            ddtheta = (((Izz-Ixx)/Iyy)*X_k(12)*X_k(8)) + (U_k(3)/Iyy);
            dphi = X_k(12);
            ddphi = (((Iyy-Izz)/Ixx)*X_k(10)*X_k(8)) + (U_k(4)/Ixx);

            ode = [dx; ddx; dy; ddy; dz; ddz; dpsi; ddpsi; dtheta; ddtheta; dphi; ddphi];

            quad_dyn = Function('quad_dyn',{X_k,U_k},{ode});
        
        elseif strcmp(model,'linearized')
            A = [  0  1  0  0  0  0  0  0  0  0  0  0   ;
                   0  0  0  0  0  0  0  0  g  0  0  0   ;
                   0  0  0  1  0  0  0  0  0  0  0  0   ;
                   0  0  0  0  0  0  0  0  0  0  g  0   ;
                   0  0  0  0  0  1  0  0  0  0  0  0   ;
                   0  0  0  0  0  0  0  0  0  0  0  0   ;
                   0  0  0  0  0  0  0  1  0  0  0  0   ;
                   0  0  0  0  0  0  0  0  0  0  0  0   ;
                   0  0  0  0  0  0  0  0  0  1  0  0   ;
                   0  0  0  0  0  0  0  0  0  0  0  0   ;
                   0  0  0  0  0  0  0  0  0  0  0  1   ;
                   0  0  0  0  0  0  0  0  0  0  0  0 ] ;

            B = [  0  0  0  0  0 1/m  0  0     0  0      0    0;
                   0  0  0  0  0  0   0 1/Izz  0  0      0    0;
                   0  0  0  0  0  0   0  0     0  1/Iyy  0    0;
                   0  0  0  0  0  0   0  0     0  0      0    1/Ixx ]';

            quad_dyn = Function('quad_dyn',{X_k,U_k},{A*X_k + B*U_k});
        end
        
    %--- Constraint definitions ---%
        % Initial states
        opti.subject_to(X(:,1) == X_init);
        
        % Dynamics constraint
        for k = 1:N_horizon
            xf = rk4_explicit(quad_dyn,X(:,k),U(:,k),dt);
            opti.subject_to(xf == X(:,k+1));
        end
        
        % State constraints
            % Angle constraints
            opti.subject_to(Angle_min.*ones(size(X([7,9,11],:))) <= X([7,9,11],:) <= Angle_max.*ones(size(X([7,9,11],:))));
        
        % Final position constraint
            opti.subject_to(X([1,3,5],end) == pos_end);
        
        % Final velocity constraint
            opti.subject_to(X([2,4,6,8,10,12],end) == zeros(nx/2,1));
        
        % Input constraints
            opti.subject_to(U_min.*ones(size(U)) <= U <= U_max.*ones(size(U)));
        
    %--- Objective setting ---%
        %objective = 0;
        V_mats = {};
        W_eT = diag([0.1; 0.1; 0.1; 0.1*ones(nu,1)]);
        
        for k = 1:N_horizon
            V_mat = [X([1,3,5],k)-pos_end;
                     U(:,k)                         ];
            V_mats{end+1} = sqrtm(W_eT)*V_mat;
            V_mats{end} = [V_mats{end};1e-2*[X(:,k);U(:,k)]];
        end
        V_mats{end+1} = 1e-2*X(:,N_horizon+1);

        V_mats = vertcat(V_mats{:});

        objective = dot(V_mats,V_mats);

        opti.minimize(objective);
        
    %--- Set solver back-end for offline solution---%
        tol = 1e-5; % 1e-9
        options = struct;                   
        options.ipopt.tol = tol;
        options.ipopt.dual_inf_tol = tol;
        options.ipopt.compl_inf_tol = tol;
        options.ipopt.constr_viol_tol = tol;
        options.ipopt.acceptable_tol = tol;

        opti.solver('ipopt',options);
 
    %--- Set parameters value ---%
        opti.set_value(X_init,initial_x);
        opti.set_value(pos_end,final_pos);
        
    %--- Solve the optimization problem ---%
        offline_sol = opti.solve();
        
    %--- Save the offline optimization solution ---%
        prim_sol = offline_sol.value(opti.x);
        dual_sol = offline_sol.value(opti.lam_g);
        
        primal_solution   = DM(prim_sol);
        dual_solution = DM(dual_sol);

            
        
%% Use the QRQP solver  

        
        %--- Warm-start the solution ---%
            opti.set_initial(opti.x, primal_solution);
        
        %--- Set the solver options ---%               
            options = struct;
            options.qpsol = 'qrqp';
            options.qpsol_options.constr_viol_tol = kkt_tol;
            options.qpsol_options.dual_inf_tol = kkt_tol;   
            options.qpsol_options.verbose = true;
            options.tol_pr = kkt_tol;
            options.tol_du = kkt_tol;
            options.min_step_size = 1e-16;
            options.max_iter = 1; % 150
            options.max_iter_ls = 0; % 0, 1, 5, 10, 20
            options.qpsol_options.print_iter = true;
            options.qpsol_options.print_header = true;
            options.print_iteration = true;
            options.print_header = false;
            options.print_status = false;
            options.print_time = false; 
            
        %--- Set the solver back-end ---%
            opti.solver('sqpmethod',options)

        %--- Set parameters value ---%
            opti.set_value(X_init,initial_x);
            opti.set_value(pos_end,final_pos);
        
        %--- Solve the OCP ---%
            tic
            sol = opti.solve();
            fprintf('Initial solution found in %f secs\n',toc)
        
        
%% Code-generate solver

    %--- Create an nlp solver from the opti solver ---%
        nlp = struct;
        nlp.f = opti.f;
        nlp.g = opti.g;
        nlp.x = opti.x;
        nlp.p = opti.p;
        solver = nlpsol('solver','sqpmethod',nlp,options);
    
        opti_x = MX.sym('opti_x',size(opti.x));
        opti_lam = MX.sym('opti_lam',size(opti.lam_g));

        res = solver('x0',opti_x,'p',opti.p,'lam_g0',opti_lam,'lbg',opti.lbg,'ubg',opti.ubg);
        
    %--- Create a function that gets the control input to apply in the current iteration ---%
        select_U = Function('select_U',{opti.x},{U(:,1)});
    
    %--- Create the casadi function for the mpc solution ---%
        % It takes as inputs: (parameters, decision_variables_from_last_solution, lagrange_multipliers_from_last_solution)
        % Its outputs are: [U(:,1), decision_variables_sol, lagrange_multipliers_sol, cost_value]
        mpc_step = Function('mpc_step',{opti.p,opti_x,opti_lam},{select_U(res.x),res.x,res.lam_g,res.f});
        
    %--- Code-generate the mpc_step function (generates .c and .h) ---%
        fprintf('\nCode-generating MPC function\n');
        mpc_step.generate('mpc_c.c',struct('with_header',true));
        
    %--- Compile the code-generated function ---%
        fprintf('\nCompiling the MPC function externally\n...\n');
        % This following IF is checking if the code is running on mac, linux, or windows
        if ismac
            % Code to run on Mac platform
            fprintf('\n\nPlatform not supported. Not using code-generated file\n\n');
        elseif isunix
            % Command to compile the C file in linux using GCC and CCACHE
            % (CCACHE is not mandatory and can be removed from the command)
                compilation_command_linux = 'ccache gcc -O3 -ffast-math mpc_c.c -shared -fPIC -lm -o libmpc_c.so';        
            % Start measuring the compilation time
                mpc_comp_tic = tic;
            % Execute the compilation command
                system(compilation_command_linux); 
            % Print the compilation time that was measured
                fprintf('\nMPC function compilation took %f secs\n',toc(mpc_comp_tic)); 
            % Import the compiled function to test it (it is really faster than
            % the standard function.
                fprintf('\nImporting code-generated MPC function\n');
                mpc_step = external('mpc_step','./libmpc_c.so');    
        elseif ispc
            % Code to run on Windows platform
            fprintf('\n\nPlatform not supported. Not using code-generated file\n\n');
        else
            fprintf('\n\nPlatform not supported. Not using code-generated file\n\n');
        end

 
%% Simulate
        % Create matrices to log some simulation data
        runtime_history = zeros(1,N_sim);
        x_history = zeros(nx,N_sim);
        u_history = zeros(nu,N_sim);

    %--- Dynamics ---%
        % Create a CasADi function from the Runge-Kutta integrator
            F = Function('F', {X, U}, {xf});
        % Change the variables in the function from casadi.MX to casadi.SX to speed-up the function
            F = F.expand();
        % Generate a Just-in-time (JIT) compilation of the function to speed-up its evaluation
            F = Function('F', {X, U}, {F(X,U)}, jit_opts);
        
    %--- Offline solution ---%
        % Get the results from the offline solution
            % First-step control signal
                u_to_apply = sol.value(U(:,1));
            % Decision variables
                opti_x = sol.value(opti.x);
            % Lagrange multipliers
                opti_lam = sol.value(opti.lam_g);
        
        for i=1:N_sim
            
            u_history(:,i) = full(u_to_apply);
            
            % simulate the system over dt using this control
            initial_x = full(F(initial_x,u_to_apply));
            final_pos = final_pos + [0.001; 0.002; 0.005];

            t_mpc = tic;
            [u_to_apply,opti_x,opti_lam,opti_f] = mpc_step([initial_x;final_pos], opti_x, opti_lam);
            run_time = toc(t_mpc);
            fprintf('Iteration # %i, Solution found in %f secs\n',i,run_time);
        
            runtime_history(i) = run_time;
            x_history(:,i) = initial_x;         % Save the states for post-processing of results
        end

%% Post-processing results

        pos_fig = figure(1);
        plot3(x_history(1,:),x_history(3,:),x_history(5,:));
        xlabel('$x\ [m]$','Interpreter','latex');
        ylabel('$y\ [m]$','Interpreter','latex');
        zlabel('$z\ [m]$','Interpreter','latex');
        
        ang_fig = figure(2);
        hold on;
        plot(x_history(7,:)); % 'DisplayName','sin(x/2)'
        plot(x_history(9,:));
        plot(x_history(11,:));
        legend({'$\psi$','$\theta$', '$\phi$'},'Interpreter','latex');
        
        
