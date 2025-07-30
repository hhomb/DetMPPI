%% ================================================================================================
% Code for: Optimality and Suboptimality of MPPI Control in Stochastic and Deterministic Settings
% Authors: Hannes Homburger, Florian Messerer, Moritz Diehl, and Johannes Reuter
% Corresponding Author: Hannes Homburger (hhomburg@htwg-konstanz.de)
% Affiliation: HTWG Konstanz, Institute of System Dynamics
% Journal: IEEE Control Systems Letters (LCSS), 2025
%
% Description:
% This MATLAB script implements the Example A described in:
% "Optimality and Suboptimality of MPPI Control in Stochastic and Deterministic Settings"
% Accepted for publication in IEEE LCSS, 2025.
%
% Usage:
% This code is intended to reproduce the results of Example A presented in the paper.
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

%% Settings
clear; close all; clc
  
J = @(V) 480.*V.^4+484.*V.^3+123.*V.^2; % Overall cost function specified in paper

dV = 0.0001; % stepsize for discretization in V
V_grid = -1:dV:1; % Grid used for discretization of noisy input V = U + W;

% Beta_grid  
beta_grid = [1,0.5,0.2,0.1]; % for Figure 1
%beta_grid = logspace(0,-3,400); % for Figure 2

length_beta = length(beta_grid);
length_V = length(V_grid);

% memory allocation
E = zeros(length_beta,1);
eta = zeros(length_beta,1);
q = zeros(length_V,length_beta);
%% Numerical computation of the q* pdf and its mean
beta_idx = 0;
for beta = beta_grid
    beta_idx = beta_idx + 1;   
    
    V_idx = 0;
    for V = V_grid
        V_idx = V_idx + 1;
        % Compute unnormalized density function q*(V) = exp(-J(V)/beta^2)
        q(V_idx,beta_idx) = exp(-1 / (beta^2) * J(V));
    end

    % Compute normalization factor eta (approximate integral of q)
    eta(beta_idx) = sum(q(:,beta_idx)) * dV;

    % Normalize q to obtain a proper PDF (i.e., sum = 1)
    q(:,beta_idx) = q(:,beta_idx) ./ eta(beta_idx); 

    % Compute expected value (mean) of V under q for this beta
    % Equivalent to integral V*q(V) dV (approximated via sum)
    E(beta_idx) = (V_grid * q(:,beta_idx)) *dV; 
end

%% Visualization Parameters 
lw = 3;
color{1} = [212,175,55]./255;
color{2} = [54,170,241]./255;
color{3} = [107, 142, 35]./255;
color{4} = [0, 0, 128]./255;

%% Figure 1 - Optimal PDF for different \beta
figure(1)
plot(V_grid,(q(:,1)),'linewidth',lw,'color',color{1})
hold on
plot(V_grid,(q(:,2)),'linewidth',lw,'color',color{2})
plot(V_grid,(q(:,3)),'linewidth',lw,'color',color{3})
plot(V_grid,(q(:,4)),'linewidth',lw,'color',color{4})

legend(['$\beta =$ ' mat2str((beta_grid(1)))],['$\beta =$ ' mat2str((beta_grid(2)))],...
['$\beta =$ ' mat2str((beta_grid(3)))],['$\beta =$ ' mat2str((beta_grid(4)))],...
'interpreter','latex','location','northwest')

xlabel('$W$','interpreter','latex')
ylabel('PDF $q^*_\beta(W)$','interpreter','latex')

set(gca,'fontsize',15)
set(gcf,'color','white')
set(gca,'ticklabelinterpreter','latex')

grid on
x = V_grid;

% Define ranges for zoomed-in areas
xRange1 = [-0.1, 0.1];
xRange2 = [-0.6, -0.4];

% Highlight zoomed areas
rect1 = rectangle('Position', [xRange1(1), min(q(:,4)), diff(xRange1), range(q(:,4))+2],...
        'EdgeColor', 'k', 'LineStyle', '-.','linewidth',0.7);
rect2 = rectangle('Position', [xRange2(1), min(q(:,1)), diff(xRange2), range(q(:,1))],...
        'EdgeColor', 'k', 'LineStyle', '-.','linewidth',0.7);

% Zoomed-in plot 1
zoomAxes1 = axes('Position', [0.6, 0.3, 0.25, 0.4]); % Adjust position and size
box on;
idx1 = x >= xRange1(1) & x <= xRange1(2); % Data in range
plot(V_grid(idx1),(q((idx1),1)),'linewidth',lw,'color',color{1})
hold on
plot(V_grid(idx1),(q((idx1),2)),'linewidth',lw,'color',color{2})
plot(V_grid(idx1),(q((idx1),3)),'linewidth',lw,'color',color{3})
plot(V_grid(idx1),(q((idx1),4)),'linewidth',lw,'color',color{4})
ylim([0,range(q(:,4))+2])
grid
set(gca,'ticklabelinterpreter','latex')

% Zoomed-in plot 2
zoomAxes2 = axes('Position', [0.2, 0.3, 0.25, 0.25]); % Adjust position and size
box on;
idx2 = x >= xRange2(1) & x <= xRange2(2); % Data in range
plot(V_grid(idx2),(q((idx2),1)),'linewidth',lw,'color',color{1})
hold on
plot(V_grid(idx2),(q((idx2),2)),'linewidth',lw,'color',color{2})
plot(V_grid(idx2),(q((idx2),3)),'linewidth',lw,'color',color{3})
plot(V_grid(idx2),(q((idx2),4)),'linewidth',lw,'color',color{4})
grid
set(gca,'ticklabelinterpreter','latex')


%% Figure 2 - Error evolution over \beta
U_star_det = 0;   % Optimizer of Cost Function (specific for given cost function)
slope_U = 0.0265; % Dominating Taylor-coefficient controls (specific for given cost function)
slope_V = 0.07;   % Dominating Taylor-coefficient value function (specific for given cost function)

figure(2)
loglog(beta_grid,abs(E-U_star_det),'-','linewidth',lw,'color',color{1})
grid on
ylabel('$||\tilde U_\mathrm{MPPI}^*(\beta)-U^\star_\mathrm{d}||$','interpreter','latex')
set(gca,'fontsize',15)
set(gcf,'color','white')
set(gca,'ticklabelinterpreter','latex')
hold on
loglog(beta_grid,abs(J(E)-J(U_star_det)),'-','linewidth',lw,'color',color{2})
xlabel('$\beta$','interpreter','latex')
ylabel('Suboptimality of MPPI','interpreter','latex')
loglog(beta_grid,slope_U*beta_grid.^2,'--','linewidth',lw/2,'color','black')
loglog(beta_grid,slope_V*beta_grid.^4,'--','linewidth',lw/2,'color','black')
set(gca,'fontsize',15)
set(gcf,'color','white')
set(gca,'ticklabelinterpreter','latex')
grid on
legend('$||\tilde U^\star_\mathrm{MPPI}(\beta)-U^\star_\mathrm{det}||=\mathcal{O}\left(\beta^2 \right)$',...
'$||J_{\overline x_0}(\tilde U^\star_\mathrm{MPPI}(\beta),0)-V_\mathrm{det}(\overline x_0)||=\mathcal{O} \left( \beta^4 \right)$',...
'interpreter','latex')
ylim([10^(-15) 10])

% ==================================================================================================