function [ TBP, bPsucc, OBP ] = immediateBeliefFunction( Bvec, Pobs, TP, OI, OID )

% this returns the belief transition functions and belief reward
% functions for the given activity over dt, i.e., it is P(b'|b,a) in the
% immediate future.

% INPUTS
% Bvec: belief vector
% Pobs: variable specifying which information relevant to predator is observed: 1=o_i, 2=o_i+o_d
% TP: transition function, predator
% OI: probability of observations given underlying state, Predator (indirect observations only)
% OID: probability of observations given underlying state, Predator (direct + indirect observations)
%
% OUTPUTS
% TBP: transition function P(b'|b) on just the belief space about predator
% bPsucc: conditional transition function P(b'|b,o) for probability that Predator=Present
% OBP: probability of observations related to PREDATOR conditioned on beliefs

nB = length(Bvec);     % number of belief states

% Pre-allocation
TBP = zeros(nB,nB);     % partial transition function relevant to beliefs about PREDATOR (absent or present)
TSBP = zeros(nB,2); % P(s'|b)
if Pobs==1
    OBP = zeros(nB,2);      % ¬o_i or o_i
    bPsucc = zeros(nB,nB,2);
elseif Pobs==2
    OBP = zeros(nB,4);      % { (¬o_i,¬o_d),(¬o_i,o_d),(o_i,¬o_d),(o_i,o_d) }
    bPsucc = zeros(nB,nB,4);
end

for i = 1:nB
    
    % current belief
    bPcurr = Bvec(i);
    
    % P(s'|b): columns:: 1:P=a, 2:P=p
    TSBP(i,:) = [(1-bPcurr) bPcurr]*TP;
    
    % P(o|b): observations depend on location and activity; columns::
    % 1=¬o_i, 2=o_i, or 1=¬o_d, 2=o_d
    switch Pobs
        case 1
            OBP(i,:) = TSBP(i,:)*OI';
        case 2
            OBP(i,:) = TSBP(i,:)*OID';
    end
    
    %% successor beliefs at next time step contingent on observations (if any)
    % PREDATOR
    bPs = TSBP(i,2);    % prior belief that predator 'present', P(s'|b)
    switch Pobs
        case 1
            bPs1 = bPs*OI(1,2);                       % 1-ilampp*dt is the probability of observing ¬o_i if predator present
            bPs1 = bPs1/(bPs1 + OI(1,1)*(1-bPs) );    % need to normalize
            bPs2 = bPs*OI(2,2);                         % ilampp*dt is the probability of observing o_i if predator present
            bPs2 = bPs2/(bPs2 + OI(2,1)*(1-bPs));
            [~,idx] = min(abs(Bvec-bPs1));
            bPsucc(i,idx,1) = 1;
            [~,idx] = min(abs(Bvec-bPs2));
            bPsucc(i,idx,2) = 1;
        case 2
            bPs1 = bPs*OID(1,2);         % (1-ilampp*dt)*(1-dlampp*dt) is the probability of observing {¬o_i,¬o_d} if predator present
            bPs1 = bPs1/(bPs1 + OID(1,1)*(1-bPs) ); % need to normalize
            bPs2 = bPs*OID(2,2);                         % (1-ilampp*dt)*(dlampp*dt) is the probability of observing {¬o_i,o_d} if predator present
            bPs2 = bPs2/(bPs2 + OID(2,1)*(1-bPs));
            bPs3 = bPs*OID(3,2);                         % (ilampp*dt)*(1-dlampp*dt) is the probability of observing {o_i,¬o_d} if predator present
            bPs3 = bPs3/(bPs3 + OID(3,1)*(1-bPs));
            bPs4 = bPs*OID(4,2);                         % (ilampp*dt)*(dlampp*dt) is the probability of observing {o_i,o_d} if predator present
            bPs4 = bPs4/(bPs4 + OID(4,1)*(1-bPs));
            [~,idx] = min(abs(Bvec-bPs1));
            bPsucc(i,idx,1) = 1;
            [~,idx] = min(abs(Bvec-bPs2));
            bPsucc(i,idx,2) = 1;
            [~,idx] = min(abs(Bvec-bPs3));
            bPsucc(i,idx,3) = 1;
            [~,idx] = min(abs(Bvec-bPs4));
            bPsucc(i,idx,4) = 1;
    end
    
    % So what's P(b'|b)? Need to marginalize over possible observations
    switch Pobs
        case 1
            for o = 1:2
                idx = find(squeeze(bPsucc(i,:,o)));
                TBP(i,idx) = TBP(i,idx) + OBP(i,o);
            end
        case 2
            for o = 1:4
                idx = find(squeeze(bPsucc(i,:,o)));
                TBP(i,idx) = TBP(i,idx) + OBP(i,o);
            end
    end
    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%