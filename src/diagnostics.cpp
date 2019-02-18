#include <algorithm>
#include <numeric>
#include <map>
#include <vector>
#include <iostream>
#include <sstream>
#include <iterator>
#include <iomanip>

using path_t = std::string;

struct line_data {
	int64_t i;
	path_t path;
	int64_t n_branch;
	std::vector<double> doubles;
};

std::ostream& operator<<(std::ostream& o, line_data ld){
	o << ld.i << " " << ld.path << " " << ld.n_branch << " ";
	std::copy(ld.doubles.begin(), ld.doubles.end(), std::ostream_iterator<double> (o, " "));
	return o;
}

std::istream& operator>>(std::istream& i, line_data& ld){
	std::string s;
	std::getline(i,s);
	std::stringstream ss(s);
	ld.doubles.clear();
	ss >> ld.i >> ld.path >> ld.n_branch;
	std::copy(std::istream_iterator<double>(ss),std::istream_iterator<double>(), std::back_inserter(ld.doubles));
	return i;
}

std::vector<path_t> most_common_paths(std::vector<line_data>& data, uint32_t n){
	auto get_first = [](auto a){return a.first;};
	auto comp_sec = [](auto a,auto b){return a.second > b.second;};
	std::vector<std::pair<path_t, int64_t>> common_pair(n);
	std::vector<path_t> common(n);
	auto paths = std::accumulate(
		data.begin(),
		data.end(),
		std::map<path_t, int64_t>(),
		[](auto m, auto ld){ m[ld.path] += ld.n_branch; return m;});
	std::partial_sort_copy(
		paths.begin(),
		paths.end(),
		common_pair.begin(),
		common_pair.end(),
		comp_sec);
	std::transform(
		common_pair.begin(),
		common_pair.end(),
		common.begin(),
		get_first);
	return common;
}

std::vector<path_t> final_paths(std::vector<line_data>& data){
	auto last = data.end()-1;
	auto n = last->i;
	while(last->i == n)
		--last;
	std::vector<path_t> output;
	std::transform(last,data.end(), std::back_inserter(output), [](auto& ld){return ld.path;});
	return output;
}

template <class C, class T>
bool contains(C& c, T& t){
	return std::find(std::begin(c), std::end(c), t) != std::end(c);
}

int64_t number_of_runs(std::vector<line_data>& data){
	int64_t n_runs = 0;
	for(auto it = data.begin(); it != data.end() && it->i != 1; ++it)
		if(it->path.size() == 1)
			n_runs += it->n_branch;
	return n_runs;
}

std::vector<std::map<path_t, double>> branching_ratio(std::vector<line_data>& data, std::vector<path_t>& paths, int64_t n_runs){
	std::vector<std::map<path_t, double>> vm(data.back().i+1);
	for(auto& ld : data)
		if(contains(paths,ld.path))
			vm[ld.i][ld.path] = double(ld.n_branch)/n_runs;
	return vm;
}

int main(int argc, char** argv){
	//if(argc < 2){
	//	std::cout << "diagnostics <type>" << std::endl;
	//	return 0;
	//}

	using iter = std::istream_iterator<line_data>;
	std::vector<line_data> data;
	std::copy(iter(std::cin), iter(), std::back_inserter(data));

	//auto common = most_common_paths(data, 5);
	auto paths = final_paths(data);
	std::cout << std::setw(10) << "" << " ";
	for(auto& p : paths)
		std::cout << std::setw(10) << p << " ";
	std::cout << std::endl;

	auto n_runs = number_of_runs(data);
	auto ratios = branching_ratio(data, paths, n_runs);
	for(std::size_t i = 0; i < ratios.size(); i++){
		std::cout << std::setw(10) << i << " ";
		for(auto& p : paths)
			std::cout << std::setw(10) << ratios[i][p] << " ";
		std::cout << std::endl;
	}
}
