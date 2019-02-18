#include "algorithm.h"
#include "linalg.h"
#include <ostream>

template<std::size_t Nq>
struct mqqrl : public algorithm {
	matrix p;
	std::vector<matrix> Q;

	mqqrl(Environment env, Parameters par_) :
		algorithm{env,par_},
		Q(Nq,make_matrix(env.Ns,env.Na,0.0)),
		p{make_matrix(env.Ns, env.Na, 1.0/sqrt(env.Na))}
	{}

	void episode() override {
		for(state_t s = 0; s < env.Ns; s++){
			if(env.grid[s] == invalid || env.grid[s] == terminal) continue;
			action_t a = static_cast<action_t>(square_weighted_choice(p[s]));
			state_t sp = env.next_state[s][a];
			double r   = env.reward(sp);

			double the = atan2(p[s][a], sqrt(square_sum(p[s]) - p[s][a]*p[s][a]));
			auto Qsp = Q[0][sp];
			for(std::size_t i = 1; i < Nq; i++)
				Qsp = Qsp + Q[i][sp];
			double Vsp = Qsp[argmax(Qsp)];
			std::size_t L = std::max(0,std::min(int(par.k*(r+Vsp/Nq)), int(M_PI/(4*the) - 0.5 + 1e-6)));

			auto pp = p[s];
			pp[a] = 0;
			norm2(pp);
			for(auto& ap : pp)
				ap *= cos((2*L+1)*the);
			pp[a] = sin((2*L+1)*the);
			p[s] = pp;
			norm2(p[s]);

			auto choice = random_choice(0,Nq-1);
			auto ap = argmax(Q[choice][sp]);
			Vsp = -Q[choice][sp][ap];
			for(std::size_t i = 0; i < Nq; i++)
				Vsp += Q[i][sp][ap];
			Q[choice][s][a] += par.alpha*(r + par.gamma*env.grid[sp]*Vsp/(Nq-1) - Q[choice][s][a]);
		}
	}

	test_t test(std::size_t maxi) override {
		test_t t;
		state_t s = env.s0;
		for(;t.n_iter < maxi && env.grid[s] != terminal; t.n_iter++){
			auto a = static_cast<action_t>(square_weighted_choice(p[s]));
			t.path.push_back(a);
			s = env.next_state[s][a];
			t.reward += env.reward(s);
		}
		auto Qavg = Q[0];
		for(std::size_t i = 1; i < Nq; i++)
			Qavg = Qavg + Q[i];
		t.double_output = flatten(Qavg);
		return t;
	}

	std::ostream& print(std::ostream& o) override {
		auto Qavg = Q[0];
		for(std::size_t i = 1; i < Nq; i++)
			Qavg = Qavg + Q[i];
		return o << "Q: " << std::endl << Qavg << std::endl << "p: " << std::endl << p;
	}
};
