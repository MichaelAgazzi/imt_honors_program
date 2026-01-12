function [K, gamma_val, Q_val, Y_val] = lmi_mpc(A, B, xk, u_bound, y_bound)
    % LMI-based robust MPC design using YALMIP
    % A, B : cell arrays of vertices {A1, A2, ...}, {B1, B2, ...}
    % xk   : current state (n×1)
    % u_min, u_max, y_min, y_max : scalar bounds
    
    % Dimensions
    n = size(A{1},1);
    m = size(B{1},2);

    % Decision variables
    Q = sdpvar(n,n,'symmetric');
    Y = sdpvar(m,n,'full');
    R = sdpvar(m,m,'symmetric');
    gamma = sdpvar(1);

    % Basic positivity constraints
    Constraints = [Q >= 1e-6*eye(n), R >= 1e-6*eye(m), gamma >= 0];

    % Eq. (10): state-dependent LMI
    LMI_state = [1, xk'; xk, Q];
    Constraints = [Constraints, LMI_state >= 0];

    % Eq. (11): main LMI for each vertex (robust stability)
    for i = 1:length(A)
        Ai = A{i}; Bi = B{i};
        
        LMI = [ Q,              (Ai*Q + Bi*Y)',     Q*Ai' + Y'*Bi',    Y';
                (Ai*Q + Bi*Y),  Q,                  zeros(n),           zeros(n,m);
                (Q*Ai' + Y'*Bi')', zeros(n),        gamma*eye(n),       zeros(n,m);
                Y,              zeros(m,n),         zeros(m,n),         gamma*eye(m)];
           
        Constraints = [Constraints, LMI >= 0];
    end

    % Eq. (12)-(13): input and output magnitude constraints
    % (interpreted as L2 norm bounds)
    for j = 1:length(A)
        Ai = A{j}; Bi = B{j};
        % Input constraint
        Constraints = [Constraints, [u_bound^2*eye(m), Y; Y', Q] >= 0];
        % State/output constraint
        Constraints = [Constraints, [Q, (Ai*Q + Bi*Y)'; (Ai*Q + Bi*Y), y_bound^2*eye(n)] >= 0];
    end

    % Objective: minimize gamma
    Objective = gamma;

    % Solve
    options = sdpsettings('solver','mosek','verbose',1);
    sol = optimize(Constraints, Objective, options);

    if sol.problem == 0
        Q_val = value(Q);
        Y_val = value(Y);
        gamma_val = value(gamma);
        K = Y_val / Q_val;

        disp('--- Soluzione trovata ---');
        disp('K (guadagno di feedback) = ');
        disp(K);
        disp(['Gamma = ', num2str(gamma_val)]);
    else
        disp('Errore nella soluzione:');
        disp(sol.info);
        K = []; gamma_val = []; Q_val = []; Y_val = [];
    end
end
