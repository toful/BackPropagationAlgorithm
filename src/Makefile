######################################################################
#                            Makefile
#
# by: Toful
######################################################################

# General defines
CC = cc

SRC = backPropagation.c

BIN = backPropagation.obj

CFLAGS = -lm

######################################################################
all :
		$(CC) -Wall -O3 $(SRC) $(CFLAGS) -o $(BIN)

clean :
		rm -rf $(BIN)


run : clean all
		#./$(BIN) "data/ring_separable.txt" "data/ring_test.txt" 1500 100 0.1 0.5 3 15 10 1
		./$(BIN) "dataset/ring_separable.txt" "data/ring_test.txt" 1000 100 0.1 0.5 4 20 10 5 1

run2 : clean all
		./$(BIN) "dataset/ring_merged.txt" "data/ring_test.txt" 1500 100 0.1 0.5 4 20 10 5 1
