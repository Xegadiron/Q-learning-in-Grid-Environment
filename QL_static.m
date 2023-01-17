GW = createGridWorld(9,9,"Kings")
%GW = createGridWorld(9,9,"Standard")

GW.CurrentState = '[2,1]';
GW.TerminalStates = '[8,9]';
%GW.ObstacleStates = ["[3,3]";"[3,4]";"[3,6]";"[3,7]";"[4,3]";"[6,3]";"[7,3]";"[7,4]";"[7,6]";"[7,7]";"[5,7]"];
GW.ObstacleStates = ["[1,1]";"[1,9]";"[3,2]";"[3,4]";"[3,5]";"[2,5]";"[3,7]";"[5,4]";"[4,3]";"[6,3]";"[7,3]";"[7,4]";"[7,6]";"[7,7]";"[6,7]";"[6,9]";"[5,7]";"[8,2]";"[8,6]";"[8,8]";"[9,1]";"[9,4]";"[9,8]";"[9,9]"];

nS = numel(GW.States);
nA = numel(GW.Actions);
GW.R = -1*ones(nS,nS,nA);                           % any action
GW.R(:,state2idx(GW,GW.ObstacleStates),:) = -5;    % obstacle collision
GW.R(:,state2idx(GW,GW.TerminalStates),:) = 50;     % reaching goal point

env = rlMDPEnv(GW)

plot(env)
%%
env.ResetFcn = @() 2;
%%rng(0)

%%%% Create Q-Learning Agent
% To create a Q-learning agent, first create a Q table using the observation and 
% action specifications from the grid world environment. Set the learn rate of the representation to 1.

qTable = rlTable(getObservationInfo(env),getActionInfo(env));
tableRep = rlRepresentation(qTable);
tableRep.Options.LearnRate = 0.5;

% Next, create a Q-learning agent using this table representation, configuring the epsilon-greedy exploration. 
% For more information on creating Q-learning agents, see rlQAgent and rlQAgentOptions.

agentOpts = rlQAgentOptions;
agentOpts.EpsilonGreedyExploration.Epsilon = 0.04;
qAgent = rlQAgent(tableRep,agentOpts);

%%%% Train Q-learning Agent

trainOpts = rlTrainingOptions;
trainOpts.MaxStepsPerEpisode = 100;
trainOpts.MaxEpisodes= 200;
trainOpts.StopTrainingCriteria = "AverageReward";
trainOpts.StopTrainingValue = 40;
trainOpts.ScoreAveragingWindowLength = 30;

% Train the agent.
trainingStats = train(qAgent,env,trainOpts);

%%%% Validate Results

plot(env)
env.Model.Viewer.ShowTrace = true;
env.Model.Viewer.clearTrace;

sim(qAgent,env)

