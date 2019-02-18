#pragma once
#include "algorithm.h"
#include "linalg.h"
#include <ostream>

struct qlearning : public algorithm {
	matrix<double> Q;

	qlearning(Environment& env_, Parameters par_) :
			algorithm{env_,par_},
			Q{make_matrix(env.Ns,env.Na,0.0)}
	{}
	~qlearning(){}

	void episode() override {
		state_t s = env.s0;
		while(env.grid[s] != terminal){
			action_t a = static_cast<action_t>(random_choice(0,env.Na-1));
			state_t sp = env.next_state[s][a];
			double r   = env.reward[sp];
			Q[s][a]   += par.alpha*(r + par.gamma*env.grid[sp]*Q[sp][argmax(Q[sp])] - Q[s][a]);
			s = sp;
		}
	}

	test_t test(std::size_t maxi) override{
		test_t t;
		state_t s = env.s0;
		for(;t.n_iter < maxi && env.grid[s] != terminal; t.n_iter++){
			s = env.next_state[s][argmax(Q[s])];
			t.reward += env.reward[s];
			if(env.grid[s] == terminal){
				t.double_output = flatten(Q);
				return t;
			}
		}
		t.double_output = flatten(Q);
		return t;
	}
	std::ostream& print(std::ostream& o) override {
		auto N = Q.size();
		std::vector<double> v(N,0.0);
		for(std::size_t i = 0; i < N; i++)
			v[i] = Q[i][argmax(Q[i])];
		return o << "Q: " << std::endl << Q << std::endl << std::endl << "V (max Q):" << std::endl << v;
	}
};
