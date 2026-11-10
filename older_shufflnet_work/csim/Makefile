# CPU C-simulation of the HLS kernels without Vitis.
# Uses AMD's open-source ap_int headers (Apache-2.0, fetched on first build)
# plus two small shims in shim/ for hls_stream.h and ap_axi_sdata.h.
# HLS pragmas are ignored by g++, so this checks function, not timing.
#
#   make -C csim            # build tb_accel
#   make -C csim run N=5    # classify N golden images, compare logits bit-exact
#   make -C csim run        # all images under data/golden

ROOT    := ..
APDIR   := build/hlsap
CXX     ?= g++
CXXFLAGS = -std=c++14 -O2 -w -Ishim -I$(APDIR)/include
N       ?= 0

all: build/tb_accel

$(APDIR)/include/ap_int.h:
	mkdir -p build
	git clone -q --depth 1 https://github.com/Xilinx/HLS_arbitrary_Precision_Types $(APDIR)

build/tb_accel: $(APDIR)/include/ap_int.h $(wildcard $(ROOT)/src/*.cpp $(ROOT)/src/*.h $(ROOT)/src/*/*.h) $(ROOT)/tb/tb_accel.cpp
	$(CXX) $(CXXFLAGS) -o $@ $(ROOT)/tb/tb_accel.cpp $(ROOT)/src/accel.cpp

run: build/tb_accel
	./build/tb_accel $(ROOT)/data/golden $(ROOT)/data/unit_weights.bin $(N)

clean:
	rm -rf build

.PHONY: all run clean
