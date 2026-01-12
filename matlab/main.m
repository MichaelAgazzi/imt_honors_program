clear; clc; close all;
this_dir = fileparts(mfilename('fullpath'));
if ~isempty(this_dir)
    addpath(this_dir);
end

% ==========================================================
%  SYSTEM SETUP
% ==========================================================
% Define uncertainty vertices (two cases)
A1 = [1.1 0.1; 0 0.99];
A2 = [0.9 0.1; 0 0];
B1 = [0.1; 0.09*0.787];
B2 = [0.1; 0.1*0.787];
A = {A1, A2};
B = {B1, B2};

n = 2; m = 1;
Qc = eye(n);
R = eye(m);

Qc(1,1) = 10.0;
Qc(2,2) = 0.0;
R(1,1) = 0.00002;

% Simulation parameters
dt = 0.2;            % sampling time (Euler step)
Nsim = 30;          % number of simulation steps

% Input/output bounds
u_bound = 2.0;
y_bound = 1.5;
epsilon = 1e-5;

% Initial state
x = [0.05; 0.0];
x_hist = x;
u_hist = [];
solving_time_clqr = [];
time = 0;

% Choose which vertex to simulate (you can switch each iteration)
vertex_id = 2;

% ==========================================================
%  MAIN SIMULATION LOOP
% ==========================================================
for k = 1:Nsim
    fprintf('=== Step %d ===\n', k);

    % Compute robust feedback gain via LMIs
    tic
    [K, gamma, Q, Y] = lmi_clqr(A, B, Qc, R, x, u_bound, y_bound, epsilon);
    solving_time_clqr(k) = toc;

    if isempty(K)
        warning('Infeasible LMI at step %d', k);
        break;
    end

    % Compute control input
    u = K * x;

    % Enforce bounds
    % u = max(min(u, u_max), u_min);

    % Simulate system (Euler step or discrete propagation)
    % Here we choose one of the vertices randomly or fixed
    Ai = A{vertex_id};
    Bi = B{vertex_id};
    x = Ai * x + Bi * u ;   % difference from discrete to Euler

    % Store
    x_hist = [x_hist, x];
    u_hist = [u_hist, u];
    time = [time, k*dt];

    % (Optional) alternate the vertex each step to test robustness
    % vertex_id = 3 - vertex_id; % switch between 1 and 2
end

% ==========================================================
%  PLOTTING
% ==========================================================
figure;
subplot(3,1,1);
plot(time, x_hist(1,:), 'LineWidth', 1.5); hold on;
plot(time, x_hist(2,:), 'LineWidth', 1.5);
xlabel('Time [s]');
ylabel('States');
legend('x_1','x_2');
grid on;
title('System states');

subplot(3,1,2);
stairs(time(1:end-1), u_hist, 'LineWidth', 1.5);
xlabel('Time [s]');
ylabel('u(k)');
grid on;
title('Control input');

subplot(3,1,3);
plot(solving_time_clqr*1000, 'LineWidth', 1.5); hold on;
xlabel('iteration k');
ylabel('[ms]');
grid on;
title('solving time');

%% standard lqr

% Initial state
x = [0.05; 0.0];
x_hist_lqr = x;
u_hist_lqr = [];
time = 0;
for k = 1:Nsim
    fprintf('=== STANDARD LQR Step %d ===\n', k);
    
    % lqr
    % x_{k+1} = A*x_k + B*u_k
    [K, S, e] = dlqr(A{1}, B{1}, Qc, R);

    if isempty(K)
        warning('Infeasible LMI at step %d', k);
        break;
    end

    % Compute control input
    u = -K * x;

    % Enforce bounds
    u = max(min(u, u_bound), -u_bound);

    % Simulate system (Euler step or discrete propagation)
    % Here we choose one of the vertices randomly or fixed
    Ai = A{vertex_id};
    Bi = B{vertex_id};
    x = Ai * x + Bi * u;   % difference from discrete to Euler

    % Store
    x_hist_lqr = [x_hist_lqr, x];
    u_hist_lqr = [u_hist_lqr, u];
    time = [time, k*dt];

    % (Optional) alternate the vertex each step to test robustness
    % vertex_id = 3 - vertex_id; % switch between 1 and 2
end

%%
% ==========================================================
%  PLOTTING comparision
% ==========================================================
figure;
subplot(3,1,1);
plot(time, x_hist(1,:), 'LineWidth', 1.5); hold on;
plot(time, x_hist_lqr(1,:), 'LineWidth', 1.5);
xlabel('Time [s]');
ylabel('States');
legend('x_1 clqr','x_1 lqr');
grid on;
title('System states');

subplot(3,1,2);
plot(time, x_hist(2,:), 'LineWidth', 1.5); hold on;
plot(time, x_hist_lqr(2,:), 'LineWidth', 1.5);
xlabel('Time [s]');
ylabel('States');
legend('x_2 clqr','x_2 lqr');
grid on;
title('System states');

subplot(3,1,3);
stairs(time(1:end-1), u_hist, 'LineWidth', 1.5); hold on
stairs(time(1:end-1), u_hist_lqr, 'LineWidth', 1.5);
xlabel('Time [s]');
ylabel('u(k)');
legend('u clqr','u lqr');
yline(u_bound);
yline(-u_bound);
grid on;
title('Control input');
