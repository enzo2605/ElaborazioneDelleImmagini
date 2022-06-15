all: my ferone

my:
	g++ MySplitAndMerge.cpp -o MySplitAndMerge.out `pkg-config --cflags --libs opencv`

ferone:
	g++ SplitAndMerge.cpp -o SplitAndMerge.out `pkg-config --cflags --libs opencv`

clean:
	rm *.out