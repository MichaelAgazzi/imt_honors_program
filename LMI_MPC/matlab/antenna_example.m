clear; clc; close all;
%%
% ==========================================================
%  ANTENNA MOTOR CONTROL CLQR vs LQR, LTV sytem
% ==========================================================
% Define uncertainty vertices
A1 = [1 0.1; 0 0.9];
A2 = [1 0.1; 0 0.1];
B1 = [0; 0.1*0.787];
B2 = [0; 0.1*0.787];
A = {A1, A2};
B = {B1, B2};

n = 2; m = 1;
Qc = eye(n);
R = eye(m);

Qc(1,1) = 1;
Qc(2,2)=0;
R = 2e-5;            

% Simulation parameters
dt = 0.1;           % sampling time (Euler step)
Nsim = 30;          % number of simulation steps

% Input/output bounds
u_bound = 2.0;
y_bound = 1000.5;
epsilon = 1e-5;

% Initial state
x = [0.05; 0.0];
x_hist = x;
u_hist = [];
solving_time_clqr = [];
poles_clqr = [];
time = 0;

% Choose which vertex to simulate
vertex_id = 2;

% ==========================================================
%  MAIN SIMULATION LOOP
% ==========================================================
for k = 1:Nsim
    fprintf('=== Step %d ===\n', k);

    % Compute robust feedback gain via LMIs
    tic
    [Kclqr, gamma, Q, Y] = lmi_clqr(A, B, Qc, R, x, u_bound, y_bound, epsilon);
    solving_time_clqr(k) = toc;

    if isempty(Kclqr)
        warning('Infeasible LMI at step %d', k);
        break;
    end

    % Compute control input
    u = Kclqr * x;
    Kclqr

    % Enforce bounds
    % u = max(min(u, u_max), u_min);

    % Simulate system (Euler step)
    L = length(A);
    alpha = rand(1,L);
    alpha = alpha / sum(alpha);
    A_random = zeros(size(A1));
    B_random = zeros(size(B1));
    for i = 1:L
        A_random = A_random + alpha(i) * A{i};
        B_random = B_random + alpha(i) * B{i};
    end
    % fixed A,B
    Ai = A{vertex_id};
    Bi = B{vertex_id};
    % random A B
    % Ai = A_random;
    % Bi = B_random;
    x = Ai * x + Bi * u;   % difference from discrete to Euler
    
    % Store
    poles_clqr = [poles_clqr, eig(Ai + Bi*Kclqr)];
    x_hist = [x_hist, x];
    u_hist = [u_hist, u];
    time = [time, k*dt];

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

%% STANDARD  UNCONSTRAINED LQR

% Initial state
x = [0.05; 0.0];
x_hist_lqr = x;
u_hist_lqr = [];
poles_lqr = [];
time = 0;
for k = 1:Nsim
    fprintf('=== STANDARD LQR Step %d ===\n', k);
    
    % lqr
    % x_{k+1} = A*x_k + B*u_k
    [Klqr, S, e] = dlqr([1.0 0.1; 0 0.9], B{1}, Qc, R);

    if isempty(Klqr)
        warning('Infeasible LMI at step %d', k);
        break;
    end

    % Compute control input
    u = -Klqr * x;

    % Enforce bounds
    u = max(min(u, u_bound), -u_bound);

    % Simulate system (Euler step)
    L = length(A);
    alpha = rand(1,L);
    alpha = alpha / sum(alpha);
    A_random = zeros(size(A1));
    B_random = zeros(size(B1));
    for i = 1:L
        A_random = A_random + alpha(i) * A{i};
        B_random = B_random + alpha(i) * B{i};
    end
    % fixed A,B
    Ai = A{vertex_id};
    Bi = B{vertex_id};
    % random A B
    % Ai = A_random;
    % Bi = B_random;
    x = Ai * x + Bi * u;   % difference from discrete to Euler

    % Store
    poles_lqr =[poles_lqr, eig(Ai+Bi*Klqr)]
    x_hist_lqr = [x_hist_lqr, x];
    u_hist_lqr = [u_hist_lqr, u];
    time = [time, k*dt];

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
yline(u_bound);
yline(-u_bound);
legend('u clqr','u lqr', 'u_{bound up}', 'u_{bound down}');
grid on;
title('Control input');

%%
% plot poles
figure;
plot(real(poles_clqr(1,:)), imag(poles_clqr(1,:)), 'o-'); hold on;
plot(real(poles_clqr(2,:)), imag(poles_clqr(2,:)), 'o-');
% Plot unit circle
theta = linspace(0, 2*pi, 400);
plot(cos(theta), sin(theta), 'k--', 'LineWidth', 1.5);
grid on;
xlabel('Real Axis');
ylabel('Imaginary Axis');
title('Pole Trajectories');
legend('Pole 1','Pole 2');
axis equal;

%%
% plot poles
figure;
plot(real(poles_lqr(1,:)), imag(poles_lqr(1,:)), 'o-'); hold on;
plot(real(poles_lqr(2,:)), imag(poles_lqr(2,:)), 'o-');
% Plot unit circle
theta = linspace(0, 2*pi, 400);
plot(cos(theta), sin(theta), 'k--', 'LineWidth', 1.5);
grid on;
xlabel('Real Axis');
ylabel('Imaginary Axis');
title('Pole Trajectories');
legend('Pole 1','Pole 2');
axis equal;
