clear; clc; close all;
this_dir = fileparts(mfilename('fullpath'));
if ~isempty(this_dir)
    addpath(this_dir);
end

% ---- parameters ----
dt = 0.1;             % sampling time as in paper
Nsim = 200;
Kgain = 0.787;        % motor gain in paper
Qc = eye(2);          % state cost (paper uses eye)
Qc(1,1) = 1; Qc(2,2)=0;
R = 2e-5;             % input cost near paper
u_bound = 2.0;
y_bound = 10.5;
epsilon = 1e-6;

% nominal and extreme friction a(k)
a_nom = 1.0;          % model used for design (unstable if real a is large)
a_high = 9.0;         % "true" plant in the destabilizing example (paper uses 9)
a_low = 0.1;          % lower vertex

% continuous-time matrices (paper)
Ac_nom = [0 1; 0 -a_nom];
Bc = [0; Kgain];

% discrete-time (Euler) used in paper:
Ad_nom = eye(2) + dt * Ac_nom;
Bd = dt * Bc;

% Build polytopic vertices for robust design (extremes)
Ac1 = [0 1; 0 -a_low];
Ac2 = [0 1; 0 -a_high];
Ad1 = eye(2) + dt * Ac1;
Ad2 = eye(2) + dt * Ac2;
Bd_vert = dt * Bc;        % same B for both vertices in paper

Acell = {Ad1, Ad2};
Bcell = {Bd_vert, Bd_vert};

% initial state
x0 = [0.05; 0];    % like the example in the paper
x = x0;
x_hist_nom = x;
x_hist_clqr = x;
u_hist_nom = [];
u_hist_clqr = [];
time = 0:dt:dt*(Nsim-1);

% ---- compute nominal LQR (designed on a_nom) ----
[Kdlqr, S, ~] = dlqr(Ad_nom, Bd, Qc, R);
% dlqr returns Kdlqr such that u = -Kdlqr * x

% ---- compute one-time CLQR robust gain (design over polytope vertices) ----
% Call lmi_clqr once (offline) to get Y,Q,gamma. Make sure lmi_clqr returns Y,Q.
% Here we assume lmi_clqr(Acell, Bcell, Qc, R, xk, u_bound, y_bound, epsilon)
% returns K_clqr_signed = Y*inv(Q) (in earlier code we had K = Y/Q)
[Ktmp, gamma, Qval, Yval] = lmi_clqr(Acell, Bcell, Qc, R, x0, u_bound, y_bound, epsilon);

if isempty(Ktmp)
    error('lmi_clqr failed - check solver and LMIs');
end

% Force the sign convention: we want u = -K_clqr * x like dlqr
Kclqr = - (Yval / Qval);    % ensure u = -Kclqr * x

% ---- simulate nominal LQR but with plant a = a_high (destabilizing) ----
x = x0;
for k = 1:Nsim
    % plant uses a_high
    Aplant = eye(2) + dt * [0 1; 0 -a_high];
    Bplant = dt * Bc;
    
    % nominal LQR control (designed on a_nom)
    u_nom = -Kdlqr * x;
    
    % apply saturation if desired
    u_nom = max(min(u_nom, u_bound), -u_bound);
    
    x = Aplant * x + Bplant * u_nom;
    
    x_hist_nom = [x_hist_nom, x];
    u_hist_nom = [u_hist_nom, u_nom];
end

% ---- simulate CLQR (robust design) on the same plant a = a_high ----
x = x0;
for k = 1:Nsim
    Aplant = eye(2) + dt * [0 1; 0 -a_high];
    Bplant = dt * Bc;
    [Ktmp, gamma, Qval, Yval] = lmi_clqr(Acell, Bcell, Qc, R, x, u_bound, y_bound, epsilon);

    if isempty(Ktmp)
    error('lmi_clqr failed - check solver and LMIs');
    end

    % Force the sign convention: we want u = -K_clqr * x like dlqr
    Kclqr = - (Yval / Qval);    % ensure u = -Kclqr * x
    u_clqr = -Kclqr * x;     % note negative sign, same law form
    u_clqr = max(min(u_clqr, u_bound), -u_bound);
    
    x = Aplant * x + Bplant * u_clqr;
    
    x_hist_clqr = [x_hist_clqr, x];
    u_hist_clqr = [u_hist_clqr, u_clqr];
end

% ---- plots ----
tvec = 0:dt:dt*Nsim;
figure;
subplot(3,1,1);
plot(tvec, x_hist_nom(1,:), 'r', 'LineWidth',1.5); hold on;
plot(tvec, x_hist_clqr(1,:), 'b', 'LineWidth',1.5);
xlabel('Time [s]'); ylabel('x_1'); legend('LQR-nom','CLQR-rob'); grid on;
title('State x_1 comparison');

subplot(3,1,2);
plot(tvec, x_hist_nom(2,:), 'r', 'LineWidth',1.5); hold on;
plot(tvec, x_hist_clqr(2,:), 'b', 'LineWidth',1.5);
xlabel('Time [s]'); ylabel('x_2'); legend('LQR-nom','CLQR-rob'); grid on;

subplot(3,1,3);
stairs(time, u_hist_nom, 'r', 'LineWidth',1.2); hold on;
stairs(time, u_hist_clqr, 'b', 'LineWidth',1.2);
xlabel('Time [s]'); ylabel('u'); legend('LQR-nom','CLQR-rob'); grid on;

disp('Finished simulation.');
