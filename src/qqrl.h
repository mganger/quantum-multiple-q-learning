#pragma once
#include "algorithm.h"
#include "linalg.h"
#include <ostream>

struct qqrl : public algorithm {
	matrix<double> Q, p;

	qqrl(Environment env, Parameters par_) :
		algorithm{env,par_},
		Q{make_matrix(env.Ns,env.Na,0.0)},
		p{make_matrix(env.Ns, env.Na, 1.0/sqrt(env.Na))}
	{}

	void episode() override {
		for(state_t s = 0; s < env.Ns; s++){
			if(env.grid[s] == invalid || env.grid[s] == terminal) continue;
			action_t a = static_cast<action_t>(square_weighted_choice(p[s]));
			state_t sp = env.next_state[s][a];
			double r   = env.reward[sp];

			double the = atan2(p[s][a], sqrt(square_sum(p[s]) - p[s][a]*p[s][a]));
			double Vsp = Q[sp][argmax(Q[sp])];
			std::size_t L = std::max(0,std::min(int(par.k*(r+Vsp)), int(M_PI/(4*the) - 0.5 + 1e-6)));

			auto pp = p[s];
			pp[a] = 0;
			norm2(pp);
			for(auto& ap : pp)
				ap *= cos((2*L+1)*the);
			pp[a] = sin((2*L+1)*the);
			p[s] = pp;
			norm2(p[s]);

			Q[s][a] += par.alpha*(r + par.gamma*env.grid[sp]*Q[sp][argmax(Q[sp])] - Q[s][a]);
		}
	}

	test_t test(std::size_t maxi) override {
		test_t t;
		state_t s = env.s0;
		for(;t.n_iter < maxi && env.grid[s] != terminal; t.n_iter++){
			s = env.next_state[s][square_weighted_choice(p[s])];
			t.reward += env.reward[s];
		}
		t.double_output = flatten(Q);
		return t;
	}

	std::ostream& print(std::ostream& o) override {
		return o << "Q: " << std::endl << Q << std::endl << "p: " << std::endl << p;
	}
};
