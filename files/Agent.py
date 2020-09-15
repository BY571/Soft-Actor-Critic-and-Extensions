from .ReplayBuffers import ReplayBuffer, PrioritizedReplay
from .networks import Actor, Critic
import torch.optim as optim
import random
from torch.distributions import MultivariateNormal, Normal
import torch.nn.functional as F
import torch
import numpy as np


class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self,  state_size,
                        action_size,
                        per,
                        ere,
                        munchausen,
                        random_seed,
                        hidden_size,
                        BATCH_SIZE,
                        BUFFER_SIZE,
                        GAMMA,
                        FIXED_ALPHA,
                        lr_c,
                        lr_a,
                        tau,
                        worker,
                        device,
                        action_prior="uniform"):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.seed_t = torch.manual_seed(random_seed)
        self.device = device
        self.per = per
        self.ere = ere
        self.munchausen = munchausen
        self.m_alpha = 0.9
        self.m_tau = 0.03
        self.lo = -1
        self.BATCH_SIZE = BATCH_SIZE
        self.BUFFER_SIZE = BUFFER_SIZE
        self.GAMMA = GAMMA
        self.worker = worker
        self.tau = tau

        self.target_entropy = -action_size  # -dim(A)
        self.alpha = 1
        self.FIXED_ALPHA = FIXED_ALPHA
        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=lr_a) 
        self._action_prior = action_prior
        
        # Actor Network 
        self.actor_local = Actor(state_size, action_size, random_seed, device, hidden_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_a)     
        
        # Critic Network (w/ Target Network)
        self.critic1 = Critic(state_size, action_size, random_seed, device, hidden_size).to(device)
        self.critic2 = Critic(state_size, action_size, random_seed, device, hidden_size).to(device)
        
        self.critic1_target = Critic(state_size, action_size, random_seed, device, hidden_size).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2_target = Critic(state_size, action_size, random_seed, device, hidden_size).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr_c, weight_decay=0)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr_c, weight_decay=0) 
        print(self.actor_local)
        print(self.critic1)

        # Replay memory
        if self.per == 1:
            self.memory = PrioritizedReplay(BUFFER_SIZE, self.BATCH_SIZE, self.device,  seed=random_seed, gamma=self.GAMMA, ere=ere, parallel_env=worker)
            self.learn = self.learn_per
        else:
            self.per = 0
            self.memory = ReplayBuffer(BUFFER_SIZE, self.BATCH_SIZE, self.device, random_seed, self.GAMMA, ere=ere, parallel_env=worker)
            self.learn = self.learn_
        print("Using PER: {}".format(self.per))        
        print("Using Munchausen RL: {}".format(self.munchausen))



    def step(self, state, action, reward, next_state, done, step, ERE=False):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)
        if ERE == False:
            # Learn, if enough samples are available in memory
            if len(self.memory) > self.BATCH_SIZE:
                experiences = self.memory.sample()

                self.learn(step, experiences, self.GAMMA)

    def ere_step(self, c_k):
        # Learn, if enough samples are available in memory
        experiences = self.memory.sample(c_k)
        self.learn(1, experiences, self.GAMMA)
            
    
    def act(self, state):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(self.device)
        with torch.no_grad():
            action = self.actor_local.get_action(state)
        return action.numpy()

    def learn_(self, step, experiences, gamma, d=1):
        """Updates actor, critics and entropy_alpha parameters using given batch of experience tuples.
        Q_targets = r + γ * (min_critic_target(next_state, actor_target(next_state)) - α *log_pi(next_action|next_state))
        Critic_loss = MSE(Q, Q_target)
        Actor_loss = α * log_pi(a|s) - Q(s,a)
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        next_action, log_pis_next = self.actor_local.evaluate(next_states)

        Q_target1_next = self.critic1_target(next_states.to(self.device), next_action.squeeze(0).to(self.device))
        Q_target2_next = self.critic2_target(next_states.to(self.device), next_action.squeeze(0).to(self.device))

        # take the mean of both critics for updating
        Q_target_next = torch.min(Q_target1_next, Q_target2_next)
        if not self.munchausen:
            if self.FIXED_ALPHA == None:
                # Compute Q targets for current states (y_i)
                Q_targets = rewards.cpu() + (gamma * (1 - dones.cpu()) * (Q_target_next.cpu() - self.alpha * log_pis_next.mean(1).unsqueeze(1).cpu()))
            else:
                Q_targets = rewards.cpu() + (gamma * (1 - dones.cpu()) * (Q_target_next.cpu() - self.FIXED_ALPHA * log_pis_next.mean(1).unsqueeze(1).cpu()))
        else:
            mu_m, log_std_m = self.actor_local(states)
            std = log_std_m.exp()
            dist = Normal(mu_m, std)
            log_pi_a = self.m_tau*dist.log_prob(actions).mean(1).unsqueeze(1).cpu()
            #m_Q = self.critic1(states, actions).cpu()  
            #logsum = torch.logsumexp(\
            #    (m_Q /self.m_tau).detach(), 1).unsqueeze(-1) #logsum trick
            #log_pi_a = m_Q - self.m_tau*logsum 
            assert log_pi_a.shape == (self.BATCH_SIZE, 1)
            munchausen_reward = (rewards.cpu() + self.m_alpha*torch.clamp(log_pi_a, min=self.lo, max=0))
            assert munchausen_reward.shape == (self.BATCH_SIZE, 1)
            if self.FIXED_ALPHA == None:
                # Compute Q targets for current states (y_i)
                Q_targets = munchausen_reward + (gamma * (1 - dones.cpu()) * (Q_target_next.cpu() - self.alpha * log_pis_next.mean(1).unsqueeze(1).cpu()))
            else:
                Q_targets = munchausen_reward + (gamma * (1 - dones.cpu()) * (Q_target_next.cpu() - self.FIXED_ALPHA * log_pis_next.mean(1).unsqueeze(1).cpu()))



        # Compute critic loss
        Q_1 = self.critic1(states, actions).cpu()
        Q_2 = self.critic2(states, actions).cpu()
        assert Q_1.shape == Q_targets.shape, "Exp: {}  -- Target: {}".format(Q_1.shape, Q_targets.shape)
        critic1_loss = 0.5*F.mse_loss(Q_1, Q_targets.detach())
        critic2_loss = 0.5*F.mse_loss(Q_2, Q_targets.detach())
        # Update critics
        # critic 1
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        # critic 2
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        if step % d == 0:
        # ---------------------------- update actor ---------------------------- #
            if self.FIXED_ALPHA == None:
                alpha = torch.exp(self.log_alpha)
                # Compute alpha loss
                actions_pred, log_pis = self.actor_local.evaluate(states)
                alpha_loss = - (self.log_alpha.cpu() * (log_pis.cpu() + self.target_entropy).detach().cpu()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                
                self.alpha = alpha
                # Compute actor loss
                if self._action_prior == "normal":
                    policy_prior = MultivariateNormal(loc=torch.zeros(self.action_size), scale_tril=torch.ones(self.action_size).unsqueeze(0))
                    policy_prior_log_probs = policy_prior.log_prob(actions_pred)
                elif self._action_prior == "uniform":
                    policy_prior_log_probs = 0.0
    
                actor_loss = (alpha * log_pis.squeeze(0).cpu() - self.critic1(states, actions_pred.squeeze(0)).cpu() - policy_prior_log_probs ).mean()
            else:
                
                actions_pred, log_pis = self.actor_local.evaluate(states)
                if self._action_prior == "normal":
                    policy_prior = MultivariateNormal(loc=torch.zeros(self.action_size), scale_tril=torch.ones(self.action_size).unsqueeze(0))
                    policy_prior_log_probs = policy_prior.log_prob(actions_pred)
                elif self._action_prior == "uniform":
                    policy_prior_log_probs = 0.0
    
                actor_loss = (self.FIXED_ALPHA * log_pis.squeeze(0).cpu() - self.critic1(states, actions_pred.squeeze(0)).cpu()- policy_prior_log_probs ).mean()
            # Minimize the loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # ----------------------- update target networks ----------------------- #
            self.soft_update(self.critic1, self.critic1_target)
            self.soft_update(self.critic2, self.critic2_target)

    def learn_per(self, step, experiences, gamma, d=1):
            """Updates actor, critics and entropy_alpha parameters using given batch of experience tuples.
            Q_targets = r + γ * (min_critic_target(next_state, actor_target(next_state)) - α *log_pi(next_action|next_state))
            Critic_loss = MSE(Q, Q_target)
            Actor_loss = α * log_pi(a|s) - Q(s,a)
            where:
                actor_target(state) -> action
                critic_target(state, action) -> Q-value
            Params
            ======
                experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
                gamma (float): discount factor
            """
            states, actions, rewards, next_states, dones, idx, weights = experiences

            # ---------------------------- update critic ---------------------------- #
            # Get predicted next-state actions and Q values from target models
            next_action, log_pis_next = self.actor_local.evaluate(next_states)

            Q_target1_next = self.critic1_target(next_states.to(self.device), next_action.squeeze(0).to(self.device))
            Q_target2_next = self.critic2_target(next_states.to(self.device), next_action.squeeze(0).to(self.device))

            Q_target_next = torch.min(Q_target1_next, Q_target2_next)
            if not self.munchausen:
                if self.FIXED_ALPHA == None:
                    # Compute Q targets for current states (y_i)
                    Q_targets = rewards.cpu() + (gamma * (1 - dones.cpu()) * (Q_target_next.cpu() - self.alpha * log_pis_next.mean(1).unsqueeze(1).cpu()))
                else:
                    Q_targets = rewards.cpu() + (gamma * (1 - dones.cpu()) * (Q_target_next.cpu() - self.FIXED_ALPHA * log_pis_next.mean(1).unsqueeze(1).cpu()))
            else:
                mu_m, log_std_m = self.actor_local(states)
                std = log_std_m.exp()
                dist = Normal(mu_m, std)
                log_pi_a = dist.log_prob(actions).mean(1).unsqueeze(1).cpu()
                assert log_pi_a.shape == (self.BATCH_SIZE, 1)
                munchausen_reward = (rewards.cpu() + self.m_alpha*torch.clamp(self.m_tau*log_pi_a, min=self.lo, max=0))
                assert munchausen_reward.shape == (self.BATCH_SIZE, 1)
                if self.FIXED_ALPHA == None:
                    # Compute Q targets for current states (y_i)
                    Q_targets = munchausen_reward + (gamma * (1 - dones.cpu()) * (Q_target_next.cpu() - self.alpha * log_pis_next.squeeze(0).cpu()))
                else:
                    Q_targets = munchausen_reward + (gamma * (1 - dones.cpu()) * (Q_target_next.cpu() - self.FIXED_ALPHA * log_pis_next.squeeze(0).cpu()))
            
            # Compute critic loss
            Q_1 = self.critic1(states, actions).cpu()
            Q_2 = self.critic2(states, actions).cpu()
            td_error1 = Q_targets.detach()-Q_1
            td_error2 = Q_targets.detach()-Q_2
            critic1_loss = 0.5* (td_error1.pow(2)*weights).mean()
            critic2_loss = 0.5* (td_error2.pow(2)*weights).mean()
            prios = abs(((td_error1 + td_error2)/2.0 + 1e-5).squeeze())
            
            # Update critics
            # critic 1
            self.critic1_optimizer.zero_grad()
            critic1_loss.backward()
            self.critic1_optimizer.step()
            # critic 2
            self.critic2_optimizer.zero_grad()
            critic2_loss.backward()
            self.critic2_optimizer.step()

            self.memory.update_priorities(idx, prios.data.cpu().numpy())
            if step % d == 0:
            # ---------------------------- update actor ---------------------------- #
                
                alpha = torch.exp(self.log_alpha)
                # Compute alpha loss
                actions_pred, log_pis = self.actor_local.evaluate(states)
                alpha_loss = - (self.log_alpha.cpu() * (log_pis.cpu() + self.target_entropy).detach().cpu()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                
                self.alpha = alpha
                # Compute actor loss
                if self._action_prior == "normal":
                    policy_prior = MultivariateNormal(loc=torch.zeros(self.action_size), scale_tril=torch.ones(self.action_size).unsqueeze(0))
                    policy_prior_log_probs = policy_prior.log_prob(actions_pred)
                elif self._action_prior == "uniform":
                    policy_prior_log_probs = 0.0
        
                actor_loss = ((alpha * log_pis.squeeze(0).cpu() - self.critic1(states, actions_pred.squeeze(0)).cpu() - policy_prior_log_probs )*weights).mean()

                # Minimize the loss
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # ----------------------- update target networks ----------------------- #
                self.soft_update(self.critic1, self.critic1_target)
                self.soft_update(self.critic2, self.critic2_target)

    
    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)