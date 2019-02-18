CXXFLAGS = -std=c++14 -O3 -Wreturn-type -g3 -pthread

N = $(shell seq 2 10)
bins = headers qrl dqrl ql dql qqrl dqqrl $(foreach i, $(N), mql_$(i) mqrl_$(i) mqqrl_$(i)) diagnostics find_cutoff

all: $(patsubst %, bin/%, $(bins))

bin/%: %.cpp $(wildcard *.h)
	$(CXX) $(CXXFLAGS) $< -o $@
