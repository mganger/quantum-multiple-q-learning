#include "algorithm.h"
#include "linalg.h"
#include <ostream>

template <std::size_t Nv>
struct mqrl : public algorithm {
	std::vector<std::vector<double>> V;
	matrix p;

	mqrl(Environment env, Parameters par_) :
		algorithm{env,par_},
		V(Nv, std::vector<double>(env.Ns,0.0)),
		p{make_matrix(env.Ns, env.Na, 1.0/sqrt(env.Na))}
	{}

	void episode() override {
		for(state_t s = 0; s < env.Ns; s++){
			if(env.grid[s] == invalid || env.grid[s] == terminal) continue;
			action_t a = static_cast<action_t>(square_weighted_choice(p[s]));
			state_t sp = env.next_state[s][a];
			double r   = env.reward(sp);

			double the = atan2(p[s][a], sqrt(square_sum(p[s]) - p[s][a]*p[s][a]));
			double Vsp = 0;
			for(std::size_t i = 0; i < Nv; i++)
				Vsp += V[i][sp];
			std::size_t L = std::max(0,std::min(int(par.k*(r+Vsp/Nv)), int(M_PI/(4*the) - 0.5 + 1e-6)));

			auto pp = p[s];
			pp[a] = 0;
			norm2(pp);
			for(auto& ap : pp)
				ap *= cos((2*L+1)*the);
			pp[a] = sin((2*L+1)*the);
			p[s] = pp;
			norm2(p[s]);

			auto choice = random_choice(0,Nv-1);
			V[choice][s] += par.alpha*(r + par.gamma*env.grid[sp]*(Vsp - V[choice][sp])/(Nv-1) - V[choice][s]);
		}
	}

	test_t test(std::size_t maxi) override {
		test_t t;
		state_t s = env.s0;
		auto Vavg = V[0];
		for(std::size_t i = 1; i < Nv; i++)
			Vavg = Vavg + V[i];
		Vavg = Vavg / Nv;
		for(;t.n_iter < maxi && env.grid[s] != terminal; t.n_iter++){
			auto a = static_cast<action_t>(square_weighted_choice(p[s]));
			t.path.push_back(a);
			s = env.next_state[s][a];
			t.reward += env.reward(s);
		}
		t.double_output = Vavg;
		return t;
	}

	std::ostream& print(std::ostream& o) override {
		auto Vavg = V[0];
		for(std::size_t i = 1; i < Nv; i++)
			Vavg = Vavg + V[i];
		Vavg = Vavg / Nv;
		return o << "V: " << std::endl << Vavg << std::endl << "p: " << std::endl << p;
	}
};
