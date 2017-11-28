function RB = immediateRewardFunction( dB, R )

% for an initial belief b, and for a chosen action
% (feed, assess, escape), what reward we can expect to obtain in the
% immediate interval dt
%
% INPUTS
% dB: discretization of beliefs
% R: immediate reward for this activity given the true state of the environment, {0,1}
%
% OUTPUTS
% RB: a vector of nBtot x 1 which tells us the expected reward given the
% belief state

Bvec = 0:dB:1;
nB = length(Bvec);     % number of belief states (considering belief spaces separately for habitat H and predator P variables)

RB = zeros(nB,1); % for any given belief state, we know what the immediate (undiscounted) reward for a given action is
for i = 1:nB
    bP = Bvec(i);
    bcurr = [(1-bP) bP];
    RB(i) = bcurr*R;
end

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%