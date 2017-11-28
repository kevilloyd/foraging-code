function bsucc = beliefFunction( TBP, dB, taus )

% for each possible initial belief state, work out for each possible
% duration of the activity in question just the distribution on resulting belief
% states
%
%INPUTS
% TBP: this is the immediate belief transition matrix for how beliefs about predator evolves (marginalizing over possible observations)
% dB: discretization of belief space
% taus: max time interval discretized by dtau
%
% OUTPUTS
% bsucc: belief function; distribution on successor beliefs given activity a for duration tau, starting in belief state b

Bvec = 0:dB:1;  % possible starting beliefs about whether predator 'present'
nB = length(Bvec);
ntaus = length(taus);

%% Beliefs about a predator
idxBP = cell(nB,ntaus);
valBP = cell(nB,ntaus);
for t = 1:ntaus
    for i = 1:nB    % for each initial belief about predator
        if t == 1
            [~,idxBP{i,t},valBP{i,t}] = find(TBP(i,:)); % we already know the 1-step belief transition matrix
        else
            idxBinit = idxBP{i,t-1};   % these are all the possible beliefs b(t); for each of these, look at their conditional distributions on b(t+1) and weigh by their own probability
            pBinit = valBP{i,t-1};
            bnew = sum( pBinit(ones(nB,1),:)'.*TBP(idxBinit,:), 1);
            [~,idxBP{i,t},valBP{i,t}] = find(bnew);
        end
    end
end

%% Now, construct the final, full belief transition matrix
bsucc = cell(1,ntaus);      % given initial belief b, and choice of activity a for duration tau, what is the distribution on belief states at the end of the period?
imat = zeros(nB,nB);
jmat = zeros(nB,nB);  % the column i indicates the current belief; now list the aspects of the successor beliefs in this column
vmat = zeros(nB,nB);  % values (i.e. probabilities) go into this matrix
for t = 1:ntaus
    imat(:,:) = 0;
    jmat(:,:) = 0;
    vmat(:,:) = 0;
    for i = 1:nB    % belief
        idxinit = i;
        idMat = idxBP{i,t};
        valMat = valBP{i,t};
        idxend = length(idMat);
        jmat(1:idxend,idxinit) = idMat;
        imat(1:idxend,idxinit) = idxinit;
        vmat(1:idxend,idxinit) = valMat;
    end
    bsucc{t} = sparse(nonzeros(imat),nonzeros(jmat),nonzeros(vmat),nB,nB);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%