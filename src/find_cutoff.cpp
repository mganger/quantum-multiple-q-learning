#include <fstream>
#include <iostream>
#include <vector>
#include <tuple>
#include <cmath>

// Finds k which maximizes sum_i (2u(i-k)-1)*(x_i-xbar)
double find_cutoff(std::vector<std::array<double,4>>& data){
	double avg = 0;
	for(auto& d : data)
		avg += d[1];
	avg /= data.size();

	std::size_t maxi = 0;
	double maxsum = 0, sum = 0;
	for(std::size_t i = 0; i < data.size(); i++){
		sum -= data[i][1] - avg;
		if(sum > maxsum){
			maxsum = sum;
			maxi   = i;
		}
	}
	return data[maxi][0];
}

struct measurement {double x=0, dx=0; };
measurement find_cutoff_uncertainty(std::vector<std::array<double,4>>& data){
	double avg = 0;
	for(auto& d : data)
		avg += d[1];
	avg /= data.size();

	double sum = 0, x=0, xx=0, sumsum = 0;
	for(std::size_t i = 0; i < data.size(); i++){
		sum -= data[i][1] - avg;
		sumsum += sum;
		x  += data[i][0]*sum;
		xx += sum*data[i][0]*data[i][0];
	}
	x /= sumsum;
	xx /= sumsum;
	return {x,std::sqrt((xx-x)/data.size())};
}

measurement find_cutoff_kmeans(std::vector<std::array<double,4>>& data){
	double mi1 = 0, mi2 = data.size()-1;
	
	for(double delta = 1; delta > 0;){
		double ni1=0, ni2=0, m1=0, m2=0, half = (mi1+mi2)/2;
		for(auto& d : data){
			(d[0] < half ? m1 : m2)   += d[1];
			(d[0] < half ? ni1 : ni2) += d[1]*d[0];
		}
		ni1 /= m1;
		ni2 /= m2;
		delta = std::max(std::abs(ni1-mi1), std::abs(ni2-mi2));
		mi1 = ni1;
		mi2 = ni2;
		std::cout << delta << std::endl;
	}

	double half = (mi1+mi2)/2;
	double s1 = 0, s2 = 0, sx1 = 0, sx2 = 0, sxx1 = 0, sxx2 = 0;
	for(auto& d : data){
		(d[0] < half ? s1 : s2)     += d[1];
		(d[0] < half ? sx1 : sx2)   += d[1]*d[0];
		(d[0] < half ? sxx1 : sxx2) += d[1]*d[0]*d[0];
	}

	sx1  /= s1;
	sx2  /= s2;
	sxx1 /= s1;
	sxx2 /= s2;

	double u1 = ((sxx1-sx1*sx1)/s1);
	double u2 = ((sxx2-sx2*sx2)/s2);

	return {(sx1+sx2)/2, std::sqrt(u1+u2) };
}

measurement find_cutoff_standard_deviation(std::vector<std::array<double,4>>& data){
	double s1=0,ss1=0,s2=0,ss2=0;
	for(auto& d : data){
		s2  += d[1];
		ss2 += d[1]*d[1];
	}

	double x=0, xx=0, sum=0;
	for(std::size_t i = 0; i < data.size(); i++){
		s1  += data[i][1];
		ss1 += data[i][1]*data[i][1];
		s2  -= data[i][1];
		ss2 -= data[i][1]*data[i][1];
		double v1 = std::sqrt(ss1/(i+1) - (s1/(i+1))*(s1/(i+1)));
		double v2 = std::sqrt(ss2/(data.size()-i) - (s2/(data.size()-i))*(s2/(data.size()-i)));
		double dist = std::abs(v1-v2);
		dist = std::isnormal(dist) ? dist : 0;
		sum += dist;
		x += data[i][0]*dist;
		xx += data[i][0]*data[i][0]*dist;
	}
	x /= sum;
	xx /= sum;
	return {x, std::sqrt(xx-x*x)};
}

measurement find_cutoff_threshold(std::vector<std::array<double,4>>& data, double threshold, std::size_t n){
	std::size_t estimate = data.size()/2;
	for(auto i = data.size()-1; i >= 0; i--){
		if(data[i][1] < threshold){
			estimate = i;
			break;
		}
	}

	//perform linear regression
	double x=0, y=0, xx=0, xy=0, yy=0;
	for(std::size_t i = estimate-n+1; i <= estimate+n; i++){
		x += data[i][1], y += data[i][0], xx += data[i][1]*data[i][1], xy += data[i][0]*data[i][1], yy += data[i][0]*data[i][0];
	}

	double n2 = 2*n;
	double sxx = xx - x*x/n2;
	double sxy = xy - x*y/n2;
	double syy = yy - y*y/n2;

	double m = sxy/sxx;
	double b = (y - m*x)/n2;

	double sr = sqrt((syy - m*m*sxx)/(n2-2));

	double y_out = m*threshold + b;
	double syout = sr*sqrt(1 + 1/n2 + (threshold-x)*(threshold-x)/sxx);

	return {y_out,syout};
}

template <class T, std::size_t N>
std::istream& operator>>(std::ifstream& f, std::array<T,N>& a){
	for(std::size_t i = 0; i < N && f >> a[i]; i++);
	return f;
}

std::vector<std::array<double,4>> get_points(std::ifstream& file){
	std::vector<std::array<double,4>> data;
	for(std::array<double,4> arr; file >> arr; data.push_back(arr));
		//arr[1] = std::log(arr[1]);
	return data;
}

int main(int argc, char** argv){
	if(argc < 4){
		std::cout << "find_cutoff <thresh> <n_est> <files...>" << std::endl;
		return 0;
	}

	double thr = std::stod(argv[1]);
	std::size_t n = std::stoul(argv[2]);

	std::vector<std::tuple<const char*, std::ifstream>> files;
	for(int i = 3; i < argc; i++)
		files.emplace_back(argv[i], argv[i]);

	for(auto& f : files){
		auto p = get_points(std::get<1>(f));
		auto i = find_cutoff_threshold(p,thr,n);
		//auto m = find_cutoff_standard_deviation(p);
		std::cout << std::get<0>(f) << " " << i.x << " " << i.dx << std::endl;
	}
}
