CC = gcc
CFLAGS = -O2 -std=c11 -Wall -Wextra -fopenmp
LDFLAGS = -fopenmp

SRCS = demo-vector.c demo-pi.c
BINS = $(SRCS:.c=.exe)

.PHONY: all clean

all: $(BINS)

%.exe: %.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(BINS)
