function r = rewardFunction( bsucc, RB, nBtot, taus )

% for each possible initial belief state, work out for each possible
% duration of the activity in question the expected discounted reward
% for that activity duration
%
%INPUTS
% bsucc: for each initial belief state, this gives the distribution on next belief states for each duration
% RB: immediate reward function(i.e. undiscounted expected reward resulting from taking action a in state b, r(b,a))
% nBtot: number of belief states
% taus: possible durations
%
% OUTPUTS
% r: reward function; expected reward for choosing activity a for duration tau, starting in belief state b

ntaus = length(taus);

r = zeros(nBtot,ntaus);     % expected reward for choosing activity a for duration tau, starting in belief state b
for t = 1:ntaus
    for i = 1:nBtot     % for each initial belief state
        if t==1
            r(i,t) = RB(i);
        else
            [~,idxbsucc,~] = find(bsucc{t-1}(i,:));   % these are all the possible beliefs b(t); for each of these, look at their conditional distributions on b(t+1) and weigh by their own probability
            r(i,t) = r(i,t-1) + (bsucc{t-1}(i,idxbsucc)*RB(idxbsucc));
        end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%