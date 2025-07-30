
%% ================================================================================================
% Paper: Optimality and Suboptimality of MPPI Control in Stochastic and Deterministic Settings
% Authors: Hannes Homburger, Florian Messerer, Moritz Diehl, and Johannes Reuter
% Corresponding Author: Hannes Homburger (hhomburg@htwg-konstanz.de)
% Affiliation: HTWG Konstanz, Institute of System Dynamics
% Journal: IEEE Control Systems Letters (LCSS), 2025
%
% Description:
% This MATLAB script implements the Example B described in:
% "Optimality and Suboptimality of MPPI Control in Stochastic and Deterministic Settings"
% Accepted for publication in IEEE LCSS, 2025.
%
% Usage:
% This code is intended to reproduce the results of Example B presented in the paper.
% Note that the code can be optimized for fast execution.
%
% License:
% for academic use only
%
% Disclaimer:
% This code is provided for academic and research purposes only, and 
% comes without any warranty or guarantee of performance.
%
% Last updated: 2025-05-21
% ==================================================================================================

close all; clear; clc; 
rng(700); %2025-05-21 added random seed for reproducability

%% Import CasADi (matlabR2016a-v3.5.5)
addpath('C:\Program Files\casadi-windows-matlabR2016a-v3.5.5')
import casadi.*

%% Parameters 
N = 2; % number of steps
x0 = -1; % initial state
n_x = 1; % state dimension
n_u = 1; % input dimension
Std = 0.75; % standard deviation
lambda = 1; % initial emperature
nu = sqrt(2)/2; % shrinking factor
K = 10; % number of shrinking iterations
M = 10^7; % number of MPPI samples

Sigma = Std*Std'; % variance
invSig = 1/Sigma; % inverse of Sigma

f_NL = @(x,u) x + atan2(u,1); % nonlinear dynamics
f_AF = @(x,u) x - 0.5*sin(3*x) + u; % affine dynamics

%% Select affine or nonlinear dynamics
f = f_NL;%f_AF;
% y_limit for visualization
Y_lim = 1.4; % [0.7 affine] or [1.4 nonlinear] 

c = @(x) 0; % state dependent cost
L = @(x,u) c(x) + lambda/2*u'*invSig*u; % stage cost
E = @(x) (x-1)^(6) + 6*x; % terminal cost
S = @(V,x0) c(x0) + c(f(x0,V(1))) + E(f(f(x0,V(1)),V(2))); % path cost

number_scenarios = 5*10^4; % scenario-based approach for comparison

%% Dynamics as CasADi function 
% Declare model variables
x_cas = SX.sym('x_cas');
u_cas = SX.sym('u_cas');

% Declare system dynamics as CasADi function
x_next = f(x_cas,u_cas);
F_cas = Function('F_cas', {x_cas, u_cas}, {x_next});


%% Numerical Solution to D-OCP with CasADi
w_init = [0;0]; % initial guess

% init
w_D = {}; % init decision variable
cost_D = 0; % init cost

% create CasADi decision variable
for k = 0 : N-1 
    Uk = MX.sym(['U_' num2str(k)]);
    w_D = [w_D(:)', {Uk}];
end

% Formulate the NLP
Xk = x0; 
for k = 0:N-1
    cost_D = cost_D + L(Xk,w_D{k+1}); % cost
    Xk = F_cas(Xk, w_D{k+1});  % dynamics
end
cost_D = cost_D + E(Xk); % add terminal cost

% create an NLP solver declare objective fct. and decition variables
prob_D = struct('f', cost_D, 'x', vertcat(w_D{:})); 
% define NLP and solver
solver_D = nlpsol('solver', 'ipopt', prob_D);

% solve the NLP
sol_D = solver_D('x0', w_init);
% convert solution to MATLAB variable
u_star_D_OCP = full(sol_D.x);

%% Numerical Solution to OLS-OCP with CasADi
w_OLS = {}; % init decision variable
cost_OLS = 0; % init cost

% create CasADi decision variable
for k=0:N-1
    Uk = MX.sym(['U_' num2str(k)]);
    w_OLS = [w_OLS(:)', {Uk}];
end

% Formulate the NLP 
% approximate expectation by mean over several scenarios
for scenario = 1:number_scenarios
Xk = x0; 
    for k=0:N-1
        cost_OLS = cost_OLS + L(Xk,w_OLS{k+1}); % cost
        Xk = F_cas(Xk, w_OLS{k+1} + randn()*Std); % realization rand. dyn. 
    end
    cost_OLS = cost_OLS + E(Xk); % add terminal cost
end
cost_OLS = cost_OLS / number_scenarios; % normalize costs
% Create an NLP solver
% declare objective fct. and decition variables
prob_OLS = struct('f', cost_OLS, 'x', vertcat(w_OLS{:})); 
% define NLP and solver
solver_OLS = nlpsol('solver', 'ipopt', prob_OLS);

% solve the NLP
sol_OLS = solver_OLS('x0', w_init);
% convert solution to MATLAB variable
u_star_ols = full(sol_OLS.x);

%% Numerical Solution to CLS-OCP with CasADi
w_CLS = {}; % init decision variable
cost_CLS = 0; % init cost

% create CasADi decision variable for u0
Ubase = MX.sym('Ubase');
w_CLS = [w_CLS(:)', {Ubase}];

% Formulate the NLP
% approximate expectation by optimizing over several scenarios
for scenario = 1:number_scenarios
    if mod(scenario,5000)==1
        disp(num2str(scenario))
    end
    Xk = x0; % init state
    cost_CLS = cost_CLS + L(Xk,Ubase); % add cost
    Xk = F_cas(Xk, Ubase+randn()*Std); % dynamics
    for k=1:N-1
        % add decision variables for each scenario to get feedback
        Uk_s = MX.sym(['U_' num2str(k) '_' num2str(scenario)]);
        w_CLS = [w_CLS(:)', {Uk_s}];
        cost_CLS = cost_CLS + L(Xk,Uk_s);
        Xk = F_cas(Xk, Uk_s + randn()*Std);  
    end
    cost_CLS = cost_CLS + E(Xk); % terminal cost
end
cost_CLS = cost_CLS/number_scenarios; % normalize

% create an NLP solver declare objective fct. and decition variables
prob_CLS = struct('f', cost_CLS, 'x', vertcat(w_CLS{:})); 
% define NLP and Solver
solver_CLS = nlpsol('solver', 'ipopt', prob_CLS);

w_init_CLS = zeros(number_scenarios+1,1); % set initial guess
% solve the NLP
sol_CLS = solver_CLS('x0', w_init_CLS);
% convert solution to MATLAB variable
u_star_cls = full(sol_CLS.x);

%% MPPI Algo [Algorithm 1 of paper] (specified for example)
% focus on readability, performance can significantly increased

% initial guess
u1 = 0; u2 = 0;

% memory allocation
w1 = zeros(M,1);
w2 = zeros(M,1);
S_MPPI = zeros(M,1);
u_star_MPPI = zeros(N*n_u,K);

for j=1:K % shrinking iterations
    % shrinking
    beta = nu^(j-1);
    o_lambda = lambda * beta^2;
    o_Sigma = Sigma * beta^2;
    inv_o_Sig = 1/o_Sigma;
    
    % sample noisy inputs and evaluate path costs 
    for m = 1:M % parallel execution possible
        w1(m) = sqrt(o_Sigma)*randn() ;
        w2(m) = sqrt(o_Sigma)*randn() ;
        S_MPPI(m) = S([u1+w1(m);u2+w2(m)],x0) ...
                    + o_lambda*u1*inv_o_Sig*(w1(m))...
                    + o_lambda*u2*inv_o_Sig*(w2(m)); 
    end
    % for normalization
    psi = min(S_MPPI);
    
    % compute weights
    weight = exp(-1/o_lambda*(S_MPPI-psi));
    
    % compute normalization
    eta_MPPI = sum(weight);

    % normalize
    weight = weight/eta_MPPI;

    % compute and store u^*
    u_star_MPPI(:,j) = [w1'*weight+u1
                        w2'*weight+u2];
           
    % warmstart next iteration
    u1 = u_star_MPPI(1,j);
    u2 = u_star_MPPI(2,j);  
end

%% Visualization of Fig. 3 A/B of the Paper
% Visualization parameters
color{1} = [212,175,55]./255;
color{2} = [54,170,241]./255;
color{3} = [107, 142, 35]./255;
color{4} = [0, 0, 128]./255;
set_width = [-2,2];
u0_grid = 0:0.005:2;
u1_grid = 0:0.005:Y_lim; % dependent on scenario
[u0, u1] = meshgrid(u0_grid, u1_grid);  

% evaluate function log J_x0 (deterministic)
log_J_x0 = @(u0,u1) log(L(x0,u0) + L(f(x0,u0),u1) + E(f(f(x0,u0),u1)));
log_J_values = arrayfun(log_J_x0, u0, u1);

figure('position',[100,100,850,260])

% plot contour
contour(u0, u1, log_J_values, 40);
hold on

% plot CasADi solution to different OCP [D-OCP,OLS-OCP,CLS-OCP]
plot([u_star_cls(1) u_star_cls(1)],set_width,'color','black','linewidth',2)% 
plot(u_star_ols(1),u_star_ols(2),'diamond','color','black','MarkerSize', 9,'linewidth',2) 
plot(u_star_D_OCP(1),u_star_D_OCP(2),'o','color','black','MarkerSize', 12,'linewidth',2)

% plot MPPI solution iterates
for j=1:K
    if j==1
        plot(u_star_MPPI(1,j),u_star_MPPI(2,j),'square','color','red','MarkerSize', 13,'linewidth',2)
    end
    plot(u_star_MPPI(1,j),u_star_MPPI(2,j),'x','color','red','MarkerSize', 9,'linewidth',2)
end

% additional information / plot options
xlabel('$u_0$', 'interpreter','latex')
ylabel('$u_1$', 'interpreter','latex')
set(gcf,'color','white')
set(gca,'fontsize',15)
set(gca,'ticklabelinterpreter','latex')
grid
xlim([0,2])
ylim([0,Y_lim])

% coordinates of the arrow and text placement
arrow_start = [0.38, 0.78]-0.025; 
arrow_end = [0.32, 0.68]-0.03;

% add the arrow pointing southwest
annotation('arrow', ...
    'X', [arrow_start(1)+0.05, arrow_end(1)+0.05], ...
    'Y', [arrow_start(2), arrow_end(2)], ...
    'LineWidth', 1.5, ...
    'Color', 'black');

% add the text next to the arrow
text(arrow_end(1)+0.4, arrow_end(2)+0.55, ...
    'shrinking $\beta$', ...
    'Interpreter', 'latex', ...
    'FontSize', 15, ...
    'HorizontalAlignment', 'right', ...
    'VerticalAlignment', 'top', ...
    'Color', 'black');

% add legend
legend('Contour of $ \log J_{\overline x_0}(U,0)$','CLS-OCP solution',...
    'OLS-OCP solution','DET-OCP solution',...
    'MPPI approx. of OLS-OCP','Deterministic MPPI iterates',...
    'interpreter','latex','fontsize',15,'location','southeast')


% ==================================================================================================

