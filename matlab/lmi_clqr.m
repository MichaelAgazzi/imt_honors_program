%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CLQR based on LMI with robustness guarantees
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [K, gamma_val, Q_val, Y_val] = lmi_clqr(A, B, Qc, R, xk, u_bound, y_bound, epsilon)

    % define optimization variables
    n = size(A{1},1);
    m = size(B{1},2);
    Q = sdpvar(n,n,'symmetric');
    Y = sdpvar(m,n,'full');
    gamma = sdpvar(1);

    % build constraints
    Constraints = [Q >= epsilon*eye(n), gamma >= epsilon];
    LMI_state = [1, xk'; xk, Q]; % LMI to minimize energy of the system 
    Constraints = [Constraints, LMI_state >= 0];

    % LMI to impose stability
    for i = 1:length(A)
        Ai = A{i};
        Bi = B{i};

        E = Qc^0.5 * Q;
        F = R^0.5 * Y;

        LMI = [ Q,          Q*Ai' + Y'*Bi',   E',  F';
                Ai*Q + Bi*Y, Q,               zeros(n), zeros(n,m);
                E,           zeros(n),       gamma*eye(n), zeros(n,m);
                F,           zeros(m,n),     zeros(m,n),   gamma*eye(m)];

        Constraints = [Constraints, LMI >= 0];
    end

    % LMI for input and output saturation
    for j = 1:length(A)
        Ai = A{j};
        Bi = B{j};

        % Input constraint
        Constraints = [Constraints, [u_bound^2*eye(m), Y; Y', Q] >= 0];

        % State/output constraint
        Constraints = [Constraints, [Q, (Ai*Q + Bi*Y)'; (Ai*Q + Bi*Y), y_bound^2*eye(n)] >= 0];
    end

    % Objective: minimize gamma -> energy of the system
    Objective = gamma;

    options = sdpsettings('solver','mosek','verbose',1);
    sol = optimize(Constraints, Objective, options);

    if sol.problem == 0
        Q_val = value(Q);
        Y_val = value(Y);
        gamma_val = value(gamma);
        K = Y_val / Q_val;

        disp('--- Solution find ---');
        disp('K (feedback gain) = ');
        disp(K);
        disp(['Gamma = ', num2str(gamma_val)]);
    else
        disp('Infeasibility:');
        disp(sol.info);
        K = [];
        gamma_val = [];
        Q_val = [];
        Y_val = [];
    end
end
