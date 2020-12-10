import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import time

class player():
    def __init__(self, start_pos, ball):
        self.pos=start_pos
        self.ball=ball
        self.reward=0


class soccer():
    def __init__(self):
        self.A = player([0, 2], 0)
        self.B = player([0, 1], 1)
        self.A_goal = [[0, 0], [1, 0]]
        self.B_goal = [[0, 3], [1, 3]]
        self.row=2
        self.col=4
        self.end=0
        self.actions=[[0,1],[0,-1],[1,0],[-1,0],[0,0]]
        self.states=[[0,0],[0,1],[0,2],[0,3],
                     [1,0],[1,1],[1,2],[1,3]]

    def move(self, P1, P2, action):
        pos=[P1.pos[0]+action[0],P1.pos[1]+action[1]]
        if 0<=pos[0] < self.row and 0<=pos[1] < self.col:
            if pos!=P2.pos:
                P1.pos=pos
            if P1.ball==1 and pos==P2.pos:
                P1.ball=0
                P2.ball=1

    def step (self, a_action, b_action):
        if self.end==0:
            # a_action = self.actions[a_action]
            # b_action = self.actions[b_action]
            if np.random.random()<0.5:
                self.move(self.A, self.B, a_action)
                self.move(self.B, self.A, b_action)
            else:
                self.move(self.B, self.A, b_action)
                self.move(self.A, self.B, a_action)

            if self.A.ball==1:
                if self.A.pos in self.A_goal:
                    self.end=1
                    self.A.reward += 100
                    self.B.reward += -100
                elif self.A.pos in self.B_goal:
                    self.end=1
                    self.A.reward += -100
                    self.B.reward += 100

            if self.B.ball==1:
                if self.B.pos in self.B_goal:
                    self.end=1
                    self.B.reward += 100
                    self.A.reward += -100
                elif self.B.pos in self.A_goal:
                    self.end=1
                    self.B.reward += -100
                    self.A.reward += 100

        # a_state = self.states.index(self.A.pos)
        # b_state = self.states.index(self.A.pos)
        # return

    def reset(self):
        self.A = player([0, 2], 0)
        self.B = player([0, 1], 1)
        self.end = 0

def Qlearning(n_iter, alpha, gamma, epsilon, alp_decay, eps_decay):

    actions = [[0, 1], [0, -1], [1, 0], [-1, 0], [0, 0]]
    grid = [[0, 0], [0, 1], [0, 2], [0, 3],
            [1, 0], [1, 1], [1, 2], [1, 3]]
    states = [(s1, s2) for s1 in grid for s2 in grid]
    init_state=states.index(([0,2],[0,1]))
    Q = np.zeros((2, len(states), len(actions)))
    game=soccer()
    Q_diff=[]

    tic=time.time()
    for ep in range(n_iter):
        if (ep+1)%5000==0:
            toc=time.time()
            print('Iter: {}    Time Elapsed: {} sec'.format(ep+1,toc-tic))
        Q_val1=Q[0,init_state,2]
        end=game.end
        if end==1:
            game.reset()
        A_pos = game.A.pos
        B_pos = game.B.pos
        A_ball = game.A.ball
        state = states.index((A_pos,B_pos))

        b_idx = np.random.randint(5)
        if np.random.random()<epsilon:
            a_idx = np.random.randint(5)
        else:
            a_idx = np.argmax(Q[A_ball][state])
        a_action = actions[a_idx]
        b_action = actions[b_idx]
        game.step(a_action,b_action)
        A_ball_ = game.A.ball
        state_ = states.index((game.A.pos, game.B.pos))
        Q[A_ball][state][a_idx] = (1 - alpha) * Q[A_ball][state][a_idx] + alpha * (
                game.A.reward + gamma * Q[A_ball_][state_].max())

        alpha=max(alpha*alp_decay, 0.001)
        epsilon=max(epsilon*eps_decay, 0.001)

        Q_val2 = Q[0, init_state, 2]
        diff=abs(Q_val1-Q_val2)
        Q_diff.append(diff)
    return Q_diff, Q

def FriendQ(n_iter, alpha, gamma, epsilon, alp_decay, eps_decay):

    actions = [[0, 1], [0, -1], [1, 0], [-1, 0], [0, 0]]
    grid = [[0, 0], [0, 1], [0, 2], [0, 3],
            [1, 0], [1, 1], [1, 2], [1, 3]]
    states = [(s1, s2) for s1 in grid for s2 in grid]
    init_state=states.index(([0,2],[0,1]))
    Q = np.zeros((2, len(states), len(actions),len(actions)))
    game=soccer()
    Q_diff=[]

    tic=time.time()
    for ep in range(n_iter):
        if (ep+1)%5000==0:
            toc=time.time()
            print('Iter: {}    Time Elapsed: {} sec'.format(ep+1,toc-tic))
        Q_val1=Q[0,init_state,2,4]
        end=game.end
        if end==1:
            game.reset()
        A_pos = game.A.pos
        B_pos = game.B.pos
        A_ball = game.A.ball
        state = states.index((A_pos,B_pos))

        b_idx = np.random.randint(5)
        if np.random.random()<epsilon:
            a_idx = np.random.randint(5)
        else:
            a_idx = np.argmax(Q[A_ball][state].max(axis=1))
        a_action = actions[a_idx]
        b_action = actions[b_idx]
        game.step(a_action,b_action)
        A_ball_ = game.A.ball
        state_ = states.index((game.A.pos, game.B.pos))
        Q[A_ball][state][a_idx][b_idx] = (1 - alpha) * Q[A_ball][state][a_idx][b_idx] + alpha * \
                                         (game.A.reward + gamma * Q[A_ball_][state_].max())

        alpha=max(alpha*alp_decay, 0.001)
        epsilon=max(epsilon*eps_decay, 0.001)

        Q_val2 = Q[0, init_state, 2, 4]
        diff=abs(Q_val1-Q_val2)
        Q_diff.append(diff)
    return Q_diff, Q

def FoeQ(n_iter, alpha, gamma, epsilon, alp_decay, eps_decay):

    def solver(A):
        A = np.vstack((A,[-1,-1,-1,-1,-1]))
        A = np.vstack((A.T,np.array([[1, 0, 0, 0, 0, 0],
                                     [0, 1, 0, 0, 0, 0],
                                     [0, 0, 1, 0, 0, 0],
                                     [0, 0, 0, 1, 0, 0],
                                     [0, 0, 0, 0, 1, 0]])))
        c = np.array([0, 0, 0, 0, 0, 1])
        b = [0]*A.shape[0]
        x = cp.Variable(6)

        prob = cp.Problem(cp.Maximize(c.T @ x),
                          [A @ x >= b, np.array([1, 1, 1, 1, 1, 0]).T @ x == 1])
        prob.solve()
        return x.value[:5]

    actions = [[0, 1], [0, -1], [1, 0], [-1, 0], [0, 0]]
    grid = [[0, 0], [0, 1], [0, 2], [0, 3],
            [1, 0], [1, 1], [1, 2], [1, 3]]
    states = [(s1, s2) for s1 in grid for s2 in grid]
    init_state=states.index(([0,2],[0,1]))
    Q = np.ones((2, len(states), len(actions), len(actions)))
    V = np.ones((2, len(states)))
    Pi = np.ones((2, len(states),len(actions)))*0.2
    game=soccer()
    Q_diff=[]

    tic=time.time()
    for ep in range(n_iter):
        if (ep+1)%5000==0:
            toc=time.time()
            print('Iter: {}    Time Elapsed: {} sec'.format(ep+1,toc-tic))
        Q_val1=Q[0,init_state,2,4]
        end=game.end
        if end==1:
            game.reset()
        A_pos = game.A.pos
        B_pos = game.B.pos
        A_ball = game.A.ball
        state = states.index((A_pos,B_pos))

        b_idx = np.random.randint(5)
        if np.random.random()<epsilon:
            a_idx = np.random.randint(5)
        else:
            prob=np.maximum(Pi[A_ball][state],0)
            prob=prob/sum(prob)
            a_idx = np.random.choice(5, p=prob)
        a_action = actions[a_idx]
        b_action = actions[b_idx]
        game.step(a_action,b_action)
        A_ball_ = game.A.ball
        state_ = states.index((game.A.pos, game.B.pos))
        Q[A_ball][state][a_idx][b_idx] = ((1-alpha)*Q[A_ball][state][a_idx][b_idx] +
                                         alpha * (game.A.reward + gamma * V[A_ball_][state_]))
        A=Q[A_ball][state]
        Pi[A_ball][state]=solver(A)
        V[A_ball][state]= np.matmul(Pi[A_ball][state], A).min()

        alpha=max(alpha*alp_decay, 0.001)
        epsilon=max(epsilon*eps_decay, 0.001)

        Q_val2 = Q[0, init_state, 2, 4]
        diff=abs(Q_val1-Q_val2)
        Q_diff.append(diff)
    return Q_diff, Q

def uCEQ(n_iter, alpha, gamma, epsilon, alp_decay, eps_decay):

    def solver(A,B):
        c=(A+B).reshape(25)
        A_=[]
        for x in range(5):
            for y in range(5):
                if x!=y:
                    row = np.zeros((5, 5))
                    row[x, :] = A[x] - A[y]
                    A_.append(row.reshape((25,)).tolist())
        for x in range(5):
            for y in range(5):
                if x!=y:
                    row = np.zeros((5, 5))
                    row[:, x] = B[:,x] - B[:,y]
                    A_.append(row.reshape((25,)).tolist())

        for x in range(25):
            row=[0]*25
            row[x]=1
            A_.append(row)

        b = [0]*len(A_)
        x = cp.Variable(25)
        A_=np.array(A_)

        prob = cp.Problem(cp.Maximize(c.T @ x),
                          [A_ @ x >= b, np.array([1]*25).T @ x == 1.0])
        prob.solve()
        return x.value.reshape((5,5))

    actions = [[0, 1], [0, -1], [1, 0], [-1, 0], [0, 0]]
    grid = [[0, 0], [0, 1], [0, 2], [0, 3],
            [1, 0], [1, 1], [1, 2], [1, 3]]
    states = [(s1, s2) for s1 in grid for s2 in grid]
    init_state=states.index(([0,2],[0,1]))
    QA = np.ones((2, len(states), len(actions), len(actions)))
    QB = np.ones((2, len(states), len(actions), len(actions)))
    VA = np.ones((2, len(states)))
    VB = np.ones((2, len(states)))
    Pi = np.ones((2, len(states),len(actions),len(actions)))*(1/25)
    game=soccer()
    Q_diff=[]

    tic=time.time()
    for ep in range(n_iter):
        if (ep+1)%5000==0:
            toc=time.time()
            print('Iter: {}    Time Elapsed: {} sec'.format(ep+1,toc-tic))
        Q_val1=QA[0,init_state,2,4]
        end=game.end
        if end==1:
            game.reset()
        A_pos = game.A.pos
        B_pos = game.B.pos
        A_ball = game.A.ball
        state = states.index((A_pos,B_pos))

        b_idx = np.random.randint(5)
        if np.random.random()<epsilon:
            a_idx = np.random.randint(5)
        else:
            prob=np.maximum(Pi[A_ball][state],0).reshape((25,))
            prob=prob/sum(prob)
            r = np.random.choice(25, p=prob)
            a_idx = r // 5

        a_action = actions[a_idx]
        b_action = actions[b_idx]
        game.step(a_action,b_action)
        A_ball_ = game.A.ball
        state_ = states.index((game.A.pos, game.B.pos))
        QA[A_ball][state][a_idx][b_idx] = ((1-alpha)*QA[A_ball][state][a_idx][b_idx] +
                                         alpha * (game.A.reward + gamma * VA[A_ball_][state_]))
        QB[A_ball][state][a_idx][b_idx] = ((1 - alpha) * QB[A_ball][state][a_idx][b_idx] +
                                           alpha * (game.B.reward + gamma * VB[A_ball_][state_]))
        A = QA[A_ball][state]
        B = QB[A_ball][state]
        try:
            Pi[A_ball][state]=solver(A,B)
        except:
            print('Solver failed at {}'.format(ep))
            # return Q_diff, A, B

        VA[A_ball][state] = np.sum(Pi[A_ball][state]*A)
        VB[A_ball][state] = np.sum(Pi[A_ball][state]*B)

        alpha=max(alpha*alp_decay, 0.001)
        epsilon=max(epsilon*eps_decay, 0.001)

        Q_val2 = QA[0, init_state, 2, 4]
        diff=abs(Q_val1-Q_val2)
        Q_diff.append(diff)
    return Q_diff, QA

def plot(Q_diff, algo):
    plt.clf()
    plt.plot(Q_diff, linewidth=0.5)
    plt.title(algo)
    plt.ylim(0, 0.5)
    plt.xlabel('Simulation Iteration')
    plt.ylabel('Q-value Difference')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(5,5), useMathText=True)
    plt.savefig(algo+".png")

Q_diff, Q=FoeQ(n_iter=1000000,alpha=1.0,gamma=0.9,epsilon=0.9,alp_decay=0.999994,eps_decay=0.999994)

plot(Q_diff, 'Foe-Q')

Q_diff, Q=FriendQ(n_iter=1000000,alpha=1.0,gamma=0.9,epsilon=0.9,alp_decay=0.999994,eps_decay=0.999994)

plot(Q_diff, 'Friend-Q')

Q_diff, Q=Qlearning(n_iter=1000000,alpha=1.0,gamma=0.9,epsilon=0.9,alp_decay=0.999994,eps_decay=0.999994)

plot(Q_diff, 'Q-learning')

Q_diff, Q=uCEQ(n_iter=1000000,alpha=1.0,gamma=0.9,epsilon=0.9,alp_decay=0.999994,eps_decay=0.999994)

plot(Q_diff, 'Correlated-Q')