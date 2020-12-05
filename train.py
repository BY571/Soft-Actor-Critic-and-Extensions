



@ray.remote  
def evaluate(frame, eval_runs=5, capture=False, render=False):
    """
    Makes an evaluation run with the current epsilon
    """

    reward_batch = []
    for i in range(eval_runs):
        state = eval_env.reset()
        if render: eval_env.render()
        rewards = 0
        while True:
            action = agent.act(np.expand_dims(state, axis=0))
            action_v = np.clip(action*action_high, action_low, action_high)
            state, reward, done, _ = eval_env.step(action_v[0])
            rewards += reward
            if done:
                break
        reward_batch.append(rewards)
    if capture == False:   
        writer.add_scalar("Reward", np.mean(reward_batch), frame)
    return np.mean(reward_batch)


@ray.remote
class SharedStorage(object):
    """
    Storage that keeps the current weights of the model and counts the interactions of the worker with the environment.
    Input: 
    model

    TODO: 
    ADD metrics that I want to track
    """
    def __init__(self, model):
        self.step_counter = 0
        self.interaction_counter = 0
        self.model = model
        self.evaluation_reward_history = []


    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        return self.model.set_weights(weights)

    def incr_counter(self):
        self.step_counter += 1

    def get_counter(self):
        return self.step_counter

    def set_eval_reward(self, step, rewards):
        self.evaluation_reward_history.append(step, rewards)
    
    def incr_interactions(self, steps):
        self.interaction_counter += steps
    
    def get_interactions(self):
        return self.interaction_counter


@ray.remote
class Worker(object):
    """
    Agent that collects data while interacting with the environment using MCTS. Collected episode trajectories are saved in the replay buffer.
    ============================
    Inputs:
    worker_id  (int)  ID of the worker
    config 
    shared_storage 
    replay buffer     
    """
    def __init__(self, worker_id, config, shared_storage, replay_buffer):
        self.worker_id = worker_id
        self.config = config
        self.shared_storage = shared_storage
        self.replay_buffer = replay_buffer

    def run(self):
        """
        
        """
        model = self.config.init_new_model()
        with torch.no_grad():
            while ray.get(self.shared_storage.get_counter.remote()) < self.config.training_steps:
                model.set_weights(ray.get(self.shared_storage.get_weights.remote()))
                model.eval()
                env = self.config.new_game(self.config.seed + self.worker_id)

                state = env.reset()
                done = False
                rewards = 0
                step = 0
                trained_steps = ray.get(self.shared_storage.get_counter.remote())
                T = self.config.visit_softmax_temperature_fn(trained_steps=trained_steps)
                
                while step <= self.config.max_moves:
                    root = Node(0)
                    state = torch.from_numpy(state).unsqueeze(0)
                    hidden_state = model.representation(state)
                    root.expand(env.to_play(), env.legal_actions(), hidden_state)
                    root.add_exploration_noise(dirichlet_alpha=self.config.root_dirichlet_alpha, exploration_fraction=self.config.root_exploration_fraction)
                    MCTS(self.config).run(root, env.action_history(), model)
                    action = get_action(root, T)
                    state, reward, done, _ = env.step(action)
                    env.set_root_values(root)

                    rewards += reward
                    step += 1
                    if done:
                        break
                self.shared_storage.incr_interactions.remote(step)
                env.close()
                self.replay_buffer.add.remote(env)


def train(args):
    
    """
    Trains the Agent. Distributed Worker collect data and store it in the replay buffer. On certain update steps the weights of the Networks get optimized. 

    Returns the trained model.
    =====
    Inputs

    config: Agent configuration
    summary writer: 

    """

    writer = SummaryWriter("runs/"+args.info)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))
    env = gym.make(args.env)
    envs.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    action_high = eval_env.action_space.high[0]
    action_low = eval_env.action_space.low[0]
    state_size = eval_env.observation_space.shape[0]
    action_size = eval_env.action_space.shape[0]
    agent = Agent(state_size=state_size, action_size=action_size, per=args.per, ere=args.ere, munchausen=args.munchausen, distributional=args.distributional,
                 D2RL=D2RL, random_seed=seed, hidden_size=HIDDEN_SIZE, BATCH_SIZE=BATCH_SIZE, BUFFER_SIZE=BUFFER_SIZE, GAMMA=GAMMA,
                 FIXED_ALPHA=FIXED_ALPHA, lr_a=LR_ACTOR, lr_c=LR_CRITIC, tau=TAU, worker=worker, device=device,  action_prior="uniform") #"normal"


    # initialize storage and replay buffer 
    storage = SharedStorage.remote(agent)
    replay_buffer = ReplayBuffer.remote(buffer_size=config.window_size, batch_size=config.batch_size, device=config.device, seed=config.seed)
        #batch_size=config.batch_size, capacity=config.window_size, prob_alpha=config.priority_prob_alpha, seed=config.seed, device=config.device)
    # create a number of distributed worker 
    workers = [Worker.remote(worker_id, config, storage, replay_buffer).run.remote() for worker_id in range(0, config.num_actors)]
    # add evaluation worker 
    workers += [_test.remote(config, storage)]
    learn(config, storage, replay_buffer, summary_writer)
    ray.wait(workers, len(workers))

    return config.get_uniform_network().set_weights(ray.get(storage.get_weights.remote()))