% Simple foraging task: one (observable) location; undiscounted returns;
% predator arrives with probability p_p per unit time and then stays for
% good; after arrival, the predator can catch the agent with probability
% p_c upon which agent incurs a large penalty and game terminates; actions are
% forage, assess, or escape; foraging is always successful, with value 1
% per unit time; escape is perfect; listening is good; agent either is or is not
% interruptible.

clear; clc; close all

%% interruptible agent or not?
interrupt = 1;  % 0=noninterrupt, 1=interrupt

%% PARAMETERS
% Simulations
dt = 1e-0;      % granularity of time
dtau = dt;      % granularity of choice of duration
dB = 5e-3;      % granularity of beliefs
tau_max = 15;   % max duration in which to engage in any activity (seconds)
Bvec = 0:dB:1;
nB = length(Bvec);
taus = dtau:dtau:tau_max;   % so these are the durations you can choose
ntaus = length(taus);
ts = dt:dt:tau_max;     % this is the corresponding discretization of this interval...
nts = length(ts);   % ... and the number of time steps
tol = 1e-3;         % stopping criterion for value iteration
% Environment
p_p = 0.01; % probability per unit time that the predator arrives (and afterwards, stays for good)
p_c = 0.1;  % probability per unit time that the predator catches the agent
ilampp = .7;    % rate of INDIRECT observations when predator present
ilampa = .3;    % rate of INDIRECT observations when predator absent
dlampp = .8;    % rate of DIRECT observations when predator present
dlampa = .2;    % rate of DIRECT observations when predator absent
z = -100;           % cost of getting caught by the predator: large and negative
z_decision = -.01;  % cost of making a decision
switch interrupt
    case 0
        z_interrupt = -1e3; % to get non-interruptible, just make interrupt really expensive
    case 1
        z_interrupt = 0;    % for interrupt case, make interrupt free!
end

%% ENVIRONMENT: MDP
% States, S
nS = 2; % number of underlying states (predator): 0=absent,1=present
% Actions, A
nA = 3;    % number of available actions: {feed, assess, escape}
feed = 1;
assess = 2;
escape = 3;
% Transition function. S x S x A ->; transitions of state are here
% independent of the animal's actions
T = [1-(p_p*dt) (p_p*dt); 0 1];
T = repmat(T,[1 1 nA]);
% Reward function. S x S x A -> ; gives the immediate expected reward given initial state and action
RS = zeros(nS,nS,nA);
RS(:,:,feed) = [1 1+(p_c*dt*z); 1 1+(p_c*dt*z)];
RS(:,:,assess) = [0 (p_c*dt*z); 0 (p_c*dt*z)];
RS(:,:,escape) = [0 0;0 0]; % assume that escape is perfect
R = zeros(nS,nA);
R(:,feed) = [T(1,:,feed)*RS(1,:,feed)'; T(2,:,feed)*RS(2,:,feed)'];
R(:,assess) = [T(1,:,assess)*RS(1,:,assess)'; T(2,:,assess)*RS(2,:,assess)'];
R(:,escape) = [T(1,:,escape)*RS(1,:,escape)'; T(2,:,escape)*RS(2,:,escape)'];

%% POMDP: states are not directly observed, but rather inferences occur thro' observations
% [Observations x States]; conditional probabilities of observations given true successor states; P(o|s')
% INDIRECT OBS ONLY
OI = [(1-ilampa*dt) (ilampa*dt);
    (1-ilampp*dt) (ilampp*dt)]';
% INDIRECT + DIRECT OBS
OID = [(1-ilampa*dt)*(1-dlampa*dt) (1-ilampa*dt)*(dlampa*dt) (ilampa*dt)*(1-dlampa*dt) (ilampa*dt)*(dlampa*dt);
    (1-ilampp*dt)*(1-dlampp*dt) (1-ilampp*dt)*(dlampp*dt) (ilampp*dt)*(1-dlampp*dt) (ilampp*dt)*(dlampp*dt)]';

%% BELIEF MDP
% derive the belief transition function for each activity
[ TBfeed, TPOfeed, OBPfeed ] = immediateBeliefFunction( Bvec, 1, T(:,:,feed), OI, OID );
[ TBassess, TPOassess, OBPassess ] = immediateBeliefFunction( Bvec, 2, T(:,:,assess), OI, OID );
[ TBescape, TPOescape, OBPescape ] = immediateBeliefFunction( Bvec, 1, T(:,:,escape), OI, OID );
% full belief transition function over each possible duration of
% activity
bAssess = beliefFunction( TBassess, dB, taus );
bFeed = beliefFunction( TBfeed, dB, taus );
bEscape = beliefFunction( TBescape, dB, taus );
% immediate reward function (i.e. the immediate reward in dt, starting in each belief
% state and initiating each action type)
RB(:,feed) = immediateRewardFunction( dB, R(:,feed) );
RB(:,assess) = immediateRewardFunction( dB, R(:,assess) );
RB(:,escape) = immediateRewardFunction( dB, R(:,escape) );
% full reward function (i.e., accumulated over extended duration, starting in each belief state)
rAssess = rewardFunction( bAssess, RB(:,assess), nB, taus );
rFeed = rewardFunction( bFeed, RB(:,feed), nB, taus );
rEscape = rewardFunction( bEscape, RB(:,escape), nB, taus );

%% VALUE ITERATION
V = zeros(nB,1);    % for storing value function
Vv = V;
Q = zeros(nB,ntaus,nA);     % for storing Q function (Q-value is for each belief state, for each activity, for each duration)
tempQ = zeros(nB,ntaus,2);  % continue or interrupt?
tempV = zeros(nB,ntaus+1);  % so this will be the value of being in belief state b at time s from when starting the activity
% run value iteration to convergence
variation = 1e2;
iter = 1;
while variation(iter) > tol
    
    iter = iter + 1;
    
    for i = 1:nA
        tempV(:,:) = 0;
        tempQ(:,:,:) = 0;
        switch i
            case feed
                for j = ntaus:-1:1  % for each choice of tau
                    tempV(:,j+1) = V; % if you've completed the full duration, you haven't interrupted yourself
                    for k = j:-1:1
                        tempQ(:,k,2) = z_interrupt + V;    % interrupt
                        tempQ(:,k,1) = RB(:,i) + TBfeed*tempV(:,k+1); % continue
                        tempV(:,k) = max( tempQ(:,k,:),[],3 );
                    end
                    Q(:,j,feed) = z_decision + tempV(:,1);
                end
            case assess
                for j = 1:1:ntaus
                    tempV(:,j+1) = V;
                    for k = j:-1:1
                        tempQ(:,k,2) = z_interrupt + V;    % interrupt
                        tempQ(:,k,1) = RB(:,i) + TBassess*tempV(:,k+1); % continue
                        tempV(:,k) = max( tempQ(:,k,:),[],3 );
                    end
                    Q(:,j,assess) = z_decision + tempV(:,1);
                end
            case escape
                Q(:,:,escape) = z_decision;
        end
    end
    
    QbestA = max(Q,[],2); % best Q values and corresponding index for duration for each activity
    QbestA = squeeze(QbestA);
    Vv = max(QbestA,[],2);
    variation(iter) = max( abs(Vv - V) );
    V = Vv;
    fprintf('Iteration %d: span = %6.2f\n', iter, variation(iter))
    
end

%% PLOTS
% visualize the resulting optimal policy
[ QbestA, IdurationA ] = max(Q,[],2); % best Q values and corresponding index for duration for each activity
QbestA = squeeze(QbestA);
IdurationA = squeeze(IdurationA);
[ Qbest, Abest ] = max( QbestA, [], 2 );     % Qbest gives us the max Q values for each initial belief state; Abest gives the index (1,2,or 3) of the best action
Ibest = IdurationA( [1:1:nB]' + [(Abest-1).*nB] ); % this gives indices of best latencies, regardless of what the best activity is
Tbest = taus(Ibest)';
figure
h = subplot(1,2,1);
map = [ 0.1453    0.7098    0.6646; 0.5709    0.7485    0.4494; 0.9763    0.9831    0.0538 ];
imagesc(linspace(0,1,101),1,Abest'); set(gca,'YDir','normal','ytick',[]), xlabel('B(predator=present)'), hcb = colorbar('Ticks',[1:1:3],'TickLabels',{'Feed','Assess','Escape'}); caxis([1 3]); title(hcb,'a')
axis square
colormap(h,map)
h = subplot(1,2,2);
imagesc(linspace(0,1,nB),linspace(1,0,nB),Tbest'); set(gca,'YDir','normal','ytick',[]), xlabel('B(predator=present)'), hcb = colorbar; caxis([dt tau_max]); title(hcb,'\tau')
axis square
colormap(h,pink)