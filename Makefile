CFILES=$(wildcard *.cpp)
OBJS=$(CFILES:.c=.o)
CC=g++
CFLAGS=-s -O3

TARGET = gmm

${TARGET}:${OBJS}
	$(CC) $(CFLAGS) -o $@ $^
%.o:%.cpp
	$(CC) $(CFLAGS) -c -o $@ $<
clean:${OBJS} gmm
	rm $^
