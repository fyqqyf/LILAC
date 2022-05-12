import torch.nn as nn
import torch.nn.functional as F
import torch as th
import torch.distributions as D
import numpy as np


class GMMroleAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(GMMroleAgent, self).__init__()
        self.args = args
        self.input_shape = input_shape
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.latent_dim = args.latent_dim
        self.hidden_dim = args.rnn_hidden_dim
        self.bs = 0  # batchsize
        self.role_num = args.role_num
        self.pi_floor=args.pi_floor
        
        self.state_dim = int(np.prod(args.state_shape))
        self.obs_dim = int(np.prod(args.obs_shape))

        self.embed_fc_input_size = (
            self.obs_dim + self.n_actions) * self.n_agents + self.hidden_dim
        NN_HIDDEN_SIZE = args.NN_HIDDEN_SIZE
        activation_func = nn.LeakyReLU()

        self.embed_net = nn.Sequential(
            nn.Linear(self.embed_fc_input_size, NN_HIDDEN_SIZE),
            nn.BatchNorm1d(NN_HIDDEN_SIZE), 
            nn.Linear(NN_HIDDEN_SIZE,
                      (args.latent_dim + 1) * args.role_num),activation_func)  # mu,sita,pi

        self.latent_net = nn.Sequential(
            nn.Linear(args.latent_dim, NN_HIDDEN_SIZE),
            nn.BatchNorm1d(NN_HIDDEN_SIZE), activation_func)

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)

        self.fc2_w_nn = nn.Linear(NN_HIDDEN_SIZE,
                                  args.rnn_hidden_dim * args.n_actions)
        self.fc2_b_nn = nn.Linear(NN_HIDDEN_SIZE, args.n_actions)

    def forward(
            self,
            inputs,  # o
            hidden_state,
            obs,  # all obs
            actions,  # all actions
            t=0,
            batch=None,
            test_mode=None,
            train_mode=False):
        inputs = inputs.reshape(-1, self.input_shape)  # o_i
        h_in = hidden_state.reshape(-1, self.hidden_dim)  # hidden_dim
        actions_cat = self.actions_cat(actions)
        obs_cat = self.obs_cat(obs)
        full_inputs = th.cat(
            [obs_cat, actions_cat],
            dim=-1)  # O+A [bs*n_agent,n_agent*(n_actions+obs_dim)]

        x = F.relu(self.fc1(inputs))
        h = self.rnn(x, h_in)
        h_ = h.reshape(-1, self.args.rnn_hidden_dim)
        h = h.reshape(-1, 1, self.args.rnn_hidden_dim)

        full_inputs = th.cat([full_inputs, h_], dim=-1)

        self.prior = self.embed_net(
            full_inputs
        )  # [bs*n_agent,(args.latent_dim + 1) * args.role_num) ]

        pi = self.prior[:, self.latent_dim * self.role_num:].abs() 
        pi = F.normalize(pi,p=1,dim=1)  
        mix = D.Categorical(pi)

        comp = D.Independent(
            D.Normal(
                self.prior[:, :self.latent_dim * self.role_num].view(
                    -1, self.role_num, self.latent_dim),
                th.ones(
                    self.prior[:, :self.latent_dim * self.role_num].view(
                        -1, self.role_num, self.latent_dim).shape[0],
                    self.role_num, self.latent_dim).to(self.args.device)), 1) 
        gmm = D.MixtureSameFamily(mix, comp)

        latent = gmm.sample()  # dim = latent_dim

        dis_loss = th.tensor(0.0).to(self.args.device)
        mi_loss = th.tensor(0.0).to(self.args.device)
        loss = th.tensor(0.0).to(self.args.device)

        if train_mode:
            dis_loss = -th.log(th.cdist(
                self.prior[:, :self.latent_dim * self.role_num].view(
                    -1, self.role_num, self.latent_dim).contiguous(),
                self.prior[:, :self.latent_dim * self.role_num].view(
                    -1, self.role_num, self.latent_dim).contiguous(),
                p=2).sum(dim=-1).mean())  # norm of each pair of mu
            mi_loss = mix.entropy().sum(dim=-1).mean()  
            loss = self.args.dis_weight * dis_loss+self.args.mi_weight * mi_loss

        # Role -> FC2 Params
        latent = self.latent_net(latent)

        # hypernet
        fc2_w = self.fc2_w_nn(latent)
        fc2_b = self.fc2_b_nn(latent)
        fc2_w = fc2_w.reshape(-1, self.args.rnn_hidden_dim,
                              self.args.n_actions)
        fc2_b = fc2_b.reshape((-1, 1, self.args.n_actions))

        # (bs*n,(obs+act+id)) at time t
        q = th.bmm(h, fc2_w) + fc2_b
        h = h.reshape(-1, self.args.rnn_hidden_dim)

        return q.view(-1, self.args.n_actions), h.view(
            -1, self.args.rnn_hidden_dim), loss, dis_loss, mi_loss

    def actions_cat(self, actions):
        # actions: [bs, n_agents, n_actions]
        assert actions.shape[1] == self.n_agents

        actions_cat = []
        for i in range(self.n_agents):
            actions_cat_ = []
            for j in range(self.n_agents):
                actions_cat_.append(actions[:, j])
            actions_cat_ = th.cat(actions_cat_, dim=-1)
            actions_cat.append(actions_cat_)

        actions_cat = th.stack(actions_cat, dim=1).contiguous().view(
            -1, (self.n_agents) * self.n_actions)
        return actions_cat

    def obs_cat(self, obs):
        # obs: [bs, n_agents, obs_dim]
        assert obs.shape[1] == self.n_agents

        obs_cat = []
        for i in range(self.n_agents):
            obs_cat_ = []
            for j in range(self.n_agents):
                obs_cat_.append(obs[:, j])
            obs_cat_ = th.cat(obs_cat_, dim=-1)
            obs_cat.append(obs_cat_)

        obs_cat = th.stack(obs_cat, dim=1).contiguous().view(
            -1, (self.n_agents) * self.obs_dim)
        return obs_cat